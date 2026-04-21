"""
v4/graph.py
===========
LangGraph-style agentic state machine for automated code correction.

Architecture
------------
  START
    → [read_inputs]      Read v3 report + current model code
    → [analyze]          LLM identifies code-level issues from failure data
    → [generate_model]   LLM generates improved model class (Python code)
    → [validate]         Syntax-check generated code
        ↓ valid              ↓ invalid (& iterations < MAX)
    → [build_script]     ← [generate_model] (with error feedback)
    → [run_training]     Execute generated script
        ↓ success            ↓ error (& iterations < MAX)
    → [evaluate]         ← [generate_model] (with runtime error)
    → [report]           Print final comparison to user
    → END

Each node is a pure function:  f(state) → state
"""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .config import (
    PIPE_DIR, PROJECT_ROOT,
    DEFAULT_LLM_MODEL, DEFAULT_PORT,
    LLM_TEMPERATURE, ANALYZE_MAX_TOKENS, GENERATE_MAX_TOKENS,
    MAX_FIX_ITERATIONS, DEFAULT_TRAIN_EPOCHS, TRAINING_TIMEOUT,
    DEFAULT_REPORT, DEFAULT_MODEL_CODE, GENERATED_SCRIPT, LOGS_DIR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeCorrectionState:
    """All state carried between nodes."""
    # config
    report_path:    Path = DEFAULT_REPORT
    model_code_path: Path = DEFAULT_MODEL_CODE
    output_script:  Path = GENERATED_SCRIPT
    train_epochs:   int  = DEFAULT_TRAIN_EPOCHS

    # populated by read_inputs
    report_summary:  str = ""
    original_code:   str = ""

    # populated by analyze
    analysis:        str = ""

    # populated by generate_model
    generated_model: str = ""

    # populated by validate
    code_valid:      bool = False
    validation_error: str = ""

    # populated by build_script
    full_script:     str = ""

    # populated by run_training
    training_log_path: str = ""
    training_output:   str = ""
    training_success:  bool = False
    training_error:    str = ""

    # populated by evaluate
    results_summary:  str = ""
    new_accuracy:     float = 0.0
    new_misclassified: int = 0

    # control
    iterations:      int = 0
    error:           Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PARSER — extract key findings from the 770KB v3 markdown report
# ═══════════════════════════════════════════════════════════════════════════════

def parse_report(report_path: Path) -> str:
    """
    Extract the key findings from ai_reasoning_summary_v3.md.
    Strips all base64 image data (which makes up ~99% of file size)
    and returns a concise text summary for the LLM.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 1) Remove <img> tags with base64 data (the bulk of the file)
    cleaned = re.sub(
        r'<img[^>]*src="data:image/[^"]*"[^>]*/?>',
        '', raw, flags=re.DOTALL
    )
    # 2) Remove any remaining raw base64 strings
    cleaned = re.sub(
        r'data:image/[a-z]+;base64,[A-Za-z0-9+/=\s]{100,}',
        '', cleaned, flags=re.DOTALL
    )

    # 3) Split into sections by ## headers
    sections: Dict[str, str] = {}
    current_header = ""
    current_lines: List[str] = []

    for line in cleaned.split('\n'):
        if line.startswith('## '):
            if current_header:
                sections[current_header] = '\n'.join(current_lines).strip()
            current_header = line.lstrip('#').strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header:
        sections[current_header] = '\n'.join(current_lines).strip()

    # 4) Build concise summary (only the sections the LLM needs)
    keep_keys = [
        "1. Training Summary",
        "2. Failure Distribution by Distortion Type",
        "3. Top Confusion Pairs",
    ]
    parts = []
    for key in keep_keys:
        for sec_key, sec_val in sections.items():
            if key.split('. ', 1)[-1].lower() in sec_key.lower():
                # Truncate each section to keep prompt tokens manageable
                truncated = sec_val[:800]
                parts.append(f"## {sec_key}\n{truncated}")
                break

    # 5) Also grab any recommendation / blind-spot section
    for sec_key, sec_val in sections.items():
        if any(kw in sec_key.lower() for kw in ['recommend', 'blind spot', 'actionable']):
            parts.append(f"## {sec_key}\n{sec_val[:600]}")

    summary = '\n\n'.join(parts)

    # 6) Final safety: cap at 3000 chars to fit in LLM context
    if len(summary) > 3000:
        summary = summary[:3000] + "\n[...truncated...]"

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CODE EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_model_class(test_py_path: Path) -> str:
    """Extract the SimpleCNN class + setup_data_loaders from test.py."""
    with open(test_py_path, "r") as f:
        content = f.read()

    # Extract from 'class SimpleCNN' to end of setup_data_loaders
    match = re.search(
        r'(class SimpleCNN.*?)(# ── Pipeline|def _run|def run_pipeline)',
        content, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return content  # fallback: return everything


def extract_python_from_llm(response: str) -> str:
    """Extract Python code from an LLM response that may contain markdown fences."""
    # Try to find ```python ... ``` blocks
    blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)

    # Try ``` ... ``` blocks
    blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)

    # No fences — assume the entire response is code
    # Strip common non-code preambles
    lines = response.strip().split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith(('import ', 'from ', 'class ', 'def ', '@', '#')) or in_code:
            in_code = True
            code_lines.append(line)
        elif in_code:
            code_lines.append(line)

    return '\n'.join(code_lines) if code_lines else response


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT TEMPLATE — wraps the LLM-generated model class
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Auto-generated improved CIFAR-10 training script.
Generated by v4 Code Correction Agent (LangGraph).
"""
import os
import sys
from pathlib import Path

_pipe_dir = str(Path(__file__).resolve().parent)
if _pipe_dir not in sys.path:
    sys.path.insert(0, _pipe_dir)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from debug_logger import DebugLogger


# ══════════════════════════════════════════════════════════════════════
# IMPROVED MODEL — generated by LLM analysis of v3 failure report
# ══════════════════════════════════════════════════════════════════════

{model_code}


# ══════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION — distortion-aware (addresses blur/jpeg failures)
# ══════════════════════════════════════════════════════════════════════

def setup_data_loaders(batch_size=64):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value='random'),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True,  transform=train_transform, download=True)
    val_ds   = datasets.CIFAR10(root='./data', train=False, transform=val_transform,   download=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    EPOCHS = {epochs}
    print("=" * 70)
    print("  Training ImprovedCNN on CIFAR-10 (auto-generated by v4 agent)")
    print("=" * 70)

    train_loader, val_loader = setup_data_loaders(batch_size=64)
    model = ImprovedCNN()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {{n_params:,}}")
    print(f"  Epochs:     {{EPOCHS}}")

    best_pt_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    debug_logger = DebugLogger(save_dir='./logs', distortion_model_path=best_pt_path)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        callbacks=[debug_logger],
        enable_progress_bar=True,
        logger=False,
    )

    print("\\n🎯 Starting training...\\n")
    trainer.fit(model, train_loader, val_loader)

    print("\\n" + "=" * 70)
    print("✅ Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK MODEL — if LLM code generation fails after MAX iterations
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_MODEL_CODE = textwrap.dedent("""\
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        return x * self.excitation(y).view(b, c, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1, drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.se = SEBlock(outc)
        self.drop = nn.Dropout2d(drop)
        self.shortcut = nn.Sequential(
            nn.Conv2d(inc, outc, 1, stride=stride, bias=False),
            nn.BatchNorm2d(outc),
        ) if stride != 1 or inc != outc else nn.Sequential()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se(self.drop(self.bn2(self.conv2(out))))
        return F.relu(out + self.shortcut(x))

class ImprovedCNN(pl.LightningModule):
    def __init__(self, lr=0.001, epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.s2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.s3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.s4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Dropout(0.3), nn.Linear(512, 10))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    def forward(self, x):
        return self.head(self.s4(self.s3(self.s2(self.s1(self.stem(x))))))
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
    def training_step(self, batch, _):
        x, y = batch; logits = self(x); loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        x, y = batch; logits = self(x); loss = self.loss_fn(logits, y)
        self.log('val_loss', loss); self.log('accuracy', (logits.argmax(1) == y).float().mean())
        return loss
""")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_llm(model: str, port: int, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key="local-3090",
        openai_api_base=f"http://localhost:{port}/v1",
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def read_inputs_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Read and parse the v3 failure report + current model source code."""
    print("\n📄 [read_inputs] Reading v3 report and model code...")

    if not state.report_path.exists():
        state.error = f"Report not found: {state.report_path}"
        return state
    if not state.model_code_path.exists():
        state.error = f"Model code not found: {state.model_code_path}"
        return state

    state.report_summary = parse_report(state.report_path)
    state.original_code = extract_model_class(state.model_code_path)

    print(f"   ✅ Report summary: {len(state.report_summary)} chars")
    print(f"   ✅ Model code:     {len(state.original_code)} chars")
    return state


def analyze_node(
    state: CodeCorrectionState, llm: ChatOpenAI
) -> CodeCorrectionState:
    """LLM analyzes the model code against failure report findings."""
    if state.error:
        return state

    print("\n🔍 [analyze] LLM analyzing model weaknesses...")

    prompt = f"""\
You are an expert ML engineer. A CIFAR-10 CNN classifier achieved only 68% accuracy.
Below is the failure analysis report and the model code. Identify the top 5 specific
code-level issues causing these failures. Be concrete — reference specific layers,
missing techniques, and which failure pattern each issue causes.

=== FAILURE REPORT ===
{state.report_summary}

=== CURRENT MODEL CODE ===
{state.original_code}

List exactly 5 issues, each as: ISSUE N: [title] — [explanation] — CAUSES: [which failures]"""

    try:
        msgs = [HumanMessage(content=prompt)]
        result = (llm | StrOutputParser()).invoke(msgs).strip()
        state.analysis = result
        print(f"   ✅ Analysis complete ({len(result)} chars)")
        # Print a preview
        for line in result.split('\n')[:8]:
            if line.strip():
                print(f"   │ {line.strip()[:90]}")
    except Exception as e:
        state.error = f"analyze_node failed: {e}"
        print(f"   ❌ Error: {e}")

    return state


def generate_model_node(
    state: CodeCorrectionState, llm: ChatOpenAI
) -> CodeCorrectionState:
    """LLM generates an improved model class based on the analysis."""
    if state.error:
        return state

    state.iterations += 1
    print(f"\n🛠️  [generate_model] Iteration {state.iterations}/{MAX_FIX_ITERATIONS} — generating improved model...")

    # Build prompt with error feedback if this is a retry
    error_ctx = ""
    if state.validation_error:
        error_ctx = f"""
YOUR PREVIOUS CODE HAD THIS ERROR — FIX IT:
{state.validation_error}
"""
    elif state.training_error:
        error_ctx = f"""
YOUR PREVIOUS CODE HAD THIS RUNTIME ERROR — FIX IT:
{state.training_error[:500]}
"""

    prompt = f"""\
Generate a PyTorch Lightning model class named `ImprovedCNN` for CIFAR-10 (32x32, 10 classes).
It must fix these issues found in the current model:

{state.analysis[:1000]}

{error_ctx}

REQUIREMENTS:
- Class name: ImprovedCNN (extends pl.LightningModule)
- Input: 3x32x32 images, output: 10 classes
- Must have __init__, forward, configure_optimizers, training_step, validation_step
- training_step must log 'train_loss' with self.log
- validation_step must log 'val_loss' and 'accuracy' with self.log
- Use BatchNorm, residual connections, and modern techniques
- Include all helper classes (attention blocks etc.) in the same code
- Use imports already available: torch, torch.nn, F, pl, CosineAnnealingLR

Output ONLY the Python code inside ```python``` fences. No explanations."""

    try:
        msgs = [HumanMessage(content=prompt)]
        result = (llm | StrOutputParser()).invoke(msgs).strip()
        state.generated_model = extract_python_from_llm(result)
        state.code_valid = False
        state.validation_error = ""
        print(f"   ✅ Model code generated ({len(state.generated_model)} chars)")
    except Exception as e:
        state.error = f"generate_model_node failed: {e}"
        print(f"   ❌ Error: {e}")

    return state


def validate_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Syntax-check the generated model code."""
    if state.error:
        return state

    print("\n✅ [validate] Checking generated code syntax...")

    code = state.generated_model
    if not code.strip():
        state.code_valid = False
        state.validation_error = "Empty code generated."
        print("   ❌ Empty code")
        return state

    # Check if ImprovedCNN class is present
    if 'class ImprovedCNN' not in code:
        state.code_valid = False
        state.validation_error = "Missing 'class ImprovedCNN' — model class not found."
        print("   ❌ Missing ImprovedCNN class")
        return state

    # Try to compile
    try:
        full_code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "import pytorch_lightning as pl\n"
            "from torch.optim.lr_scheduler import CosineAnnealingLR\n\n"
            + code
        )
        ast.parse(full_code)
        state.code_valid = True
        state.validation_error = ""
        print("   ✅ Syntax OK")
    except SyntaxError as e:
        state.code_valid = False
        state.validation_error = f"SyntaxError: {e}"
        print(f"   ❌ Syntax error: {e}")

    return state


def build_script_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Combine the validated model code into a full training script."""
    if state.error:
        return state

    print("\n📦 [build_script] Assembling full training script...")

    model_code = state.generated_model
    state.full_script = TRAINING_SCRIPT_TEMPLATE.format(
        model_code=model_code,
        epochs=state.train_epochs,
    )
    print(f"   ✅ Script assembled ({len(state.full_script)} chars)")
    return state


def write_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Write the full script to disk."""
    if state.error:
        return state

    print(f"\n💾 [write] Writing to {state.output_script}...")
    state.output_script.parent.mkdir(parents=True, exist_ok=True)
    with open(state.output_script, 'w') as f:
        f.write(state.full_script)
    print(f"   ✅ Written: {state.output_script}")
    return state


def run_training_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Execute the generated training script as a subprocess."""
    if state.error:
        return state

    print(f"\n🚀 [run_training] Executing {state.output_script.name}...")
    print(f"   Training for {state.train_epochs} epoch(s) — this may take a while.\n")

    try:
        result = subprocess.run(
            [sys.executable, str(state.output_script)],
            cwd=str(PIPE_DIR),
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT,
        )

        state.training_output = result.stdout
        if result.returncode == 0:
            state.training_success = True
            state.training_error = ""
            print("   ✅ Training completed successfully!")
            # Print last few lines of output
            for line in result.stdout.strip().split('\n')[-5:]:
                print(f"   │ {line}")
        else:
            state.training_success = False
            state.training_error = result.stderr[-1000:] if result.stderr else "Unknown error"
            print(f"   ❌ Training failed (exit code {result.returncode})")
            print(f"   │ {state.training_error[:200]}")

    except subprocess.TimeoutExpired:
        state.training_success = False
        state.training_error = f"Training timed out after {TRAINING_TIMEOUT}s"
        print(f"   ❌ {state.training_error}")
    except Exception as e:
        state.training_success = False
        state.training_error = str(e)
        print(f"   ❌ Error: {e}")

    return state


def evaluate_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Parse the training log to extract results."""
    if state.error or not state.training_success:
        return state

    print("\n📊 [evaluate] Parsing training results...")

    # Find the latest training log
    try:
        log_files = sorted(LOGS_DIR.glob("training_log_*.json"), key=lambda p: p.stat().st_mtime)
        if not log_files:
            state.results_summary = "No training log found."
            return state

        latest_log = log_files[-1]
        state.training_log_path = str(latest_log)

        with open(latest_log) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        epochs = data.get("epochs", [])

        best_acc = summary.get("best_accuracy", 0)
        best_epoch = summary.get("best_epoch", "?")
        total_mc = summary.get("total_misclassified", "?")
        final_tl = summary.get("final_train_loss", "?")
        final_vl = summary.get("final_val_loss", "?")

        state.new_accuracy = best_acc if isinstance(best_acc, (int, float)) else 0
        state.new_misclassified = total_mc if isinstance(total_mc, int) else 0

        epoch_lines = []
        for e in epochs:
            epoch_lines.append(
                f"  Epoch {e.get('epoch', '?')}: "
                f"acc={e.get('overall_accuracy', 0):.4f}, "
                f"f1={e.get('f1_score', 0):.4f}, "
                f"misclassified={e.get('num_misclassified', '?')}"
            )

        state.results_summary = (
            f"Best Accuracy:  {best_acc}\n"
            f"Best Epoch:     {best_epoch}\n"
            f"Misclassified:  {total_mc}\n"
            f"Final Train Loss: {final_tl}\n"
            f"Final Val Loss:   {final_vl}\n"
            f"Per-epoch:\n" + '\n'.join(epoch_lines)
        )
        print(f"   ✅ Best accuracy: {best_acc}")
        print(f"   ✅ Total misclassified: {total_mc}")

    except Exception as e:
        state.results_summary = f"Error parsing results: {e}"
        print(f"   ❌ Error: {e}")

    return state


def report_node(state: CodeCorrectionState) -> CodeCorrectionState:
    """Print the final comparison report."""
    print("\n" + "═" * 70)
    print("  📋 CODE CORRECTION AGENT — FINAL REPORT")
    print("═" * 70)

    if state.error:
        print(f"\n  ❌ Pipeline error: {state.error}")
        print("═" * 70)
        return state

    print(f"\n  🔍 Analysis:")
    for line in (state.analysis or "N/A").split('\n')[:6]:
        if line.strip():
            print(f"    {line.strip()[:100]}")

    print(f"\n  🛠️  Iterations: {state.iterations}")
    print(f"  📝 Generated script: {state.output_script}")

    if state.training_success:
        print(f"\n  📊 RESULTS:")
        for line in state.results_summary.split('\n'):
            print(f"    {line}")

        print(f"\n  ─── COMPARISON ───")
        print(f"    Baseline (SimpleCNN):   68.11% accuracy, 10083 misclassified")
        if state.new_accuracy:
            delta = state.new_accuracy - 0.6811
            sign = "+" if delta >= 0 else ""
            print(f"    Improved (ImprovedCNN): {state.new_accuracy:.2%} accuracy, "
                  f"{state.new_misclassified} misclassified "
                  f"({sign}{delta:.2%})")
    else:
        print(f"\n  ❌ Training did not complete successfully.")
        if state.training_error:
            print(f"    Error: {state.training_error[:200]}")

    print("\n" + "═" * 70)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED GRAPH — state machine execution engine (same pattern as v3)
# ═══════════════════════════════════════════════════════════════════════════════

class CompiledGraph:
    """Executable LangGraph-style compiled graph."""

    def __init__(
        self,
        nodes:      Dict[str, Callable],
        edges:      Dict[str, str],
        cond_edges: Dict[str, Callable],
        entry:      str,
        finish:     str,
    ):
        self._nodes      = nodes
        self._edges      = edges
        self._cond_edges = cond_edges
        self._entry      = entry
        self._finish     = finish

    def run(self, state: CodeCorrectionState) -> CodeCorrectionState:
        current = self._entry
        visited = []

        while current != self._finish:
            visited.append(current)

            # Short-circuit on unrecoverable error
            if state.error and current not in ("report",):
                current = "report"
                continue

            node_fn = self._nodes.get(current)
            if node_fn is None:
                state.error = f"Unknown node: {current}"
                break

            state = node_fn(state)

            # Route to next node
            if current in self._cond_edges:
                current = self._cond_edges[current](state)
            elif current in self._edges:
                current = self._edges[current]
            else:
                break

        # Run finish node
        finish_fn = self._nodes.get(self._finish)
        if finish_fn:
            state = finish_fn(state)

        return state


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL ROUTERS
# ═══════════════════════════════════════════════════════════════════════════════

def _route_after_validate(state: CodeCorrectionState) -> str:
    """After code validation: retry generate or proceed to build."""
    if state.code_valid:
        return "build_script"
    if state.iterations < MAX_FIX_ITERATIONS:
        return "generate_model"   # loop back with error feedback
    # Max iterations reached — use fallback
    print(f"\n⚠️  Max iterations ({MAX_FIX_ITERATIONS}) reached — using fallback model.")
    state.generated_model = FALLBACK_MODEL_CODE
    state.code_valid = True
    return "build_script"


def _route_after_training(state: CodeCorrectionState) -> str:
    """After training: evaluate on success, retry on error."""
    if state.training_success:
        return "evaluate"
    if state.iterations < MAX_FIX_ITERATIONS:
        return "generate_model"   # loop back with runtime error feedback
    return "evaluate"  # proceed to report even on failure


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(
    model_name: str = DEFAULT_LLM_MODEL,
    port:       int = DEFAULT_PORT,
) -> CompiledGraph:
    """
    Build the code correction agent graph.

    Graph topology:
      read_inputs → analyze → generate_model → validate ─┐
                                    ↑                      │
                                    └─── (invalid) ────────┘
                                                           │ (valid)
                                               build_script → write → run_training ─┐
                                    ↑                                                 │
                                    └─── (error)  ────────────────────────────────────┘
                                                           │ (success)
                                                       evaluate → report → END
    """
    analyze_llm  = build_llm(model_name, port, ANALYZE_MAX_TOKENS)
    generate_llm = build_llm(model_name, port, GENERATE_MAX_TOKENS)

    nodes = {
        "read_inputs":   lambda s: read_inputs_node(s),
        "analyze":       lambda s: analyze_node(s, analyze_llm),
        "generate_model": lambda s: generate_model_node(s, generate_llm),
        "validate":      lambda s: validate_node(s),
        "build_script":  lambda s: build_script_node(s),
        "write":         lambda s: write_node(s),
        "run_training":  lambda s: run_training_node(s),
        "evaluate":      lambda s: evaluate_node(s),
        "report":        lambda s: report_node(s),
    }

    edges = {
        "read_inputs":  "analyze",
        "analyze":      "generate_model",
        "generate_model": "validate",
        # validate uses conditional routing
        "build_script": "write",
        "write":        "run_training",
        # run_training uses conditional routing
        "evaluate":     "report",
    }

    cond_edges = {
        "validate":     _route_after_validate,
        "run_training": _route_after_training,
    }

    return CompiledGraph(
        nodes=nodes,
        edges=edges,
        cond_edges=cond_edges,
        entry="read_inputs",
        finish="report",
    )
