"""
v5/pipeline.py
==============
Combined v3 + v4 end-to-end pipeline engine.

Handles the full flow:
  user code → train → collect stats → LLM analyze → LLM fix → train improved → compare

All functions accept an optional `emit` callback for streaming progress
updates to the web frontend via SSE.
"""

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import (
    PIPE_DIR, LOGS_DIR, REPORTS_DIR,
    LLM_MODEL, LLM_PORT, LLM_API_BASE, LLM_API_KEY,
    LLM_TEMPERATURE,
    ANALYZE_TOKENS, GENERATE_TOKENS,
    DEFAULT_EPOCHS, TRAINING_TIMEOUT, MAX_FIX_RETRIES,
    CIFAR10_CLASSES,
)

# Type alias for the progress callback
Emitter = Optional[Callable[[str, str, Any], None]]


# ═══════════════════════════════════════════════════════════════════════════════
# LLM UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _build_llm(max_tokens: int, port: int = LLM_PORT, model: str = LLM_MODEL):
    """Build a ChatOpenAI pointed at the Ollama server (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_API_BASE,
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens,
    )


def _llm_call(prompt: str, max_tokens: int) -> str:
    """Single LLM call, returns text."""
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    llm = _build_llm(max_tokens)
    return (llm | StrOutputParser()).invoke([HumanMessage(content=prompt)]).strip()


def check_llm_server(port: int = LLM_PORT) -> bool:
    """Check if the Ollama server is reachable."""
    import urllib.request
    try:
        # Ollama serves on /api/tags, not /health
        with urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# CODE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_code(code: str) -> Tuple[bool, str]:
    """AST-parse code. Returns (is_valid, error_message)."""
    preamble = (
        "import torch\nimport torch.nn as nn\n"
        "import torch.nn.functional as F\nimport pytorch_lightning as pl\n"
        "from torch.optim.lr_scheduler import CosineAnnealingLR\n\n"
    )
    try:
        ast.parse(preamble + code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_python_blocks(text: str) -> str:
    """Extract Python code from markdown-fenced LLM response."""
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)
    blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)
    # No fences — try to extract code-looking lines
    lines = []
    started = False
    for line in text.split('\n'):
        if line.startswith(('import ', 'from ', 'class ', 'def ', '@', '#')) or started:
            started = True
            lines.append(line)
    return '\n'.join(lines) if lines else text


def detect_model_class_name(code: str) -> Optional[str]:
    """Find the class name that extends LightningModule or nn.Module."""
    match = re.search(
        r'class\s+(\w+)\s*\(\s*(?:pl\.LightningModule|nn\.Module|LightningModule)',
        code
    )
    return match.group(1) if match else None


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated training script — v5 pipeline."""
import os, sys
from pathlib import Path
_pipe = str(Path(__file__).resolve().parent)
if _pipe not in sys.path:
    sys.path.insert(0, _pipe)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from debug_logger import DebugLogger

# ── MODEL ──
{model_code}

# ── DATA ──
def setup_data(bs=64):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
        transforms.ToTensor(), transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value='random'),
    ])
    t_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10('./data', train=True,  transform=t_train, download=True)
    va = datasets.CIFAR10('./data', train=False, transform=t_val,   download=True)
    return DataLoader(tr, bs, shuffle=True, num_workers=2), DataLoader(va, bs, num_workers=2)

if __name__ == "__main__":
    train_ld, val_ld = setup_data()
    model = {class_name}()
    best_pt = os.path.join(os.path.dirname(__file__), 'best.pt')
    logger = DebugLogger(save_dir='./logs', distortion_model_path=best_pt)
    trainer = pl.Trainer(max_epochs={epochs}, callbacks=[logger],
                         enable_progress_bar=True, logger=False)
    trainer.fit(model, train_ld, val_ld)
    print("TRAINING_COMPLETE")
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_user_code(user_code: str, epochs: int = DEFAULT_EPOCHS,
                   tag: str = "original") -> Tuple[Path, str]:
    """
    Wrap user model code into a runnable training script.
    Returns (script_path, class_name).
    """
    class_name = detect_model_class_name(user_code)
    if not class_name:
        class_name = "SimpleCNN"  # default fallback

    script = TRAIN_TEMPLATE.format(
        model_code=user_code,
        class_name=class_name,
        epochs=epochs,
    )
    script_path = PIPE_DIR / f"_v5_train_{tag}.py"
    script_path.write_text(script)
    return script_path, class_name


def train_model(script_path: Path, emit: Emitter = None) -> Dict:
    """
    Run a training script as subprocess.
    Returns dict with success, output, log_path, error.
    """
    tag = script_path.stem
    if emit:
        emit("progress", f"Training {tag}...", {"phase": "training"})

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PIPE_DIR),
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT,
        )
        if proc.returncode == 0:
            return {"success": True, "output": proc.stdout, "error": ""}
        else:
            return {"success": False, "output": proc.stdout,
                    "error": proc.stderr[-1500:] if proc.stderr else "Training failed"}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": f"Timeout after {TRAINING_TIMEOUT}s"}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}


def collect_stats(emit: Emitter = None) -> Dict:
    """
    Read the latest training log and return failure statistics.
    Reuses logic from v3/data_loader.
    """
    if emit:
        emit("progress", "Collecting failure statistics...", {"phase": "stats"})

    logs = sorted(LOGS_DIR.glob("training_log_*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {"error": "No training log found"}

    latest = logs[-1]
    with open(latest) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    epochs = data.get("epochs", [])

    # Build failure stats
    stats = {
        "log_path": str(latest),
        "best_accuracy": summary.get("best_accuracy", 0),
        "best_epoch": summary.get("best_epoch", 0),
        "total_misclassified": summary.get("total_misclassified", 0),
        "final_train_loss": summary.get("final_train_loss", 0),
        "final_val_loss": summary.get("final_val_loss", 0),
        "epochs": [],
        "confusion_pairs": [],
        "by_distortion": {},
    }

    for ep in epochs:
        stats["epochs"].append({
            "epoch": ep.get("epoch", 0),
            "accuracy": ep.get("overall_accuracy", 0),
            "f1": ep.get("f1_score", 0),
            "misclassified": ep.get("num_misclassified", 0),
        })

    # Try loading misclassified JSON for deeper analysis
    ts = latest.stem.replace("training_log_", "")
    mis_path = LOGS_DIR / f"misclassified_{ts}.json"

    if mis_path.exists():
        try:
            with open(mis_path) as f:
                mc = json.load(f)
            stats["confusion_pairs"] = mc.get("top_confusion_pairs", [])[:15]
            stats["by_distortion"] = mc.get("by_distortion", {})
            stats["total_misclassified"] = mc.get("n_misclassified",
                                                   stats["total_misclassified"])
        except Exception:
            pass
    else:
        # Try export_misclassified
        try:
            subprocess.run(
                [sys.executable, str(PIPE_DIR / "export_misclassified.py"),
                 "--input", str(latest), "--output", str(mis_path)],
                cwd=str(PIPE_DIR), capture_output=True, timeout=120,
            )
            if mis_path.exists():
                with open(mis_path) as f:
                    mc = json.load(f)
                stats["confusion_pairs"] = mc.get("top_confusion_pairs", [])[:15]
                stats["by_distortion"] = mc.get("by_distortion", {})
        except Exception:
            pass

    return stats


def format_stats_for_llm(stats: Dict) -> str:
    """Format stats dict into concise text for LLM prompt."""
    lines = [
        f"Best Accuracy: {stats.get('best_accuracy', '?')}",
        f"Total Misclassified: {stats.get('total_misclassified', '?')}",
        f"Final Train Loss: {stats.get('final_train_loss', '?')}",
        f"Final Val Loss: {stats.get('final_val_loss', '?')}",
    ]
    # Epoch progression
    for ep in stats.get("epochs", []):
        lines.append(f"  Epoch {ep['epoch']}: acc={ep['accuracy']:.4f}")

    # Confusion pairs
    pairs = stats.get("confusion_pairs", [])
    if pairs:
        lines.append("\nTop Confusion Pairs:")
        for p in pairs[:10]:
            lines.append(f"  {p['true']} → {p['pred']}: {p['count']}")

    # Distortion distribution
    by_dist = stats.get("by_distortion", {})
    if by_dist:
        lines.append("\nFailure Distribution:")
        total = stats.get("total_misclassified", 1) or 1
        for dt, info in by_dist.items():
            cnt = info.get("count", 0)
            pct = round(cnt / total * 100, 1)
            lines.append(f"  {dt}: {cnt} ({pct}%)")

    return '\n'.join(lines)


def llm_analyze_code(user_code: str, stats_text: str = "",
                     emit: Emitter = None) -> str:
    """
    LLM analyses user code against failure statistics.
    Returns analysis text identifying code-level issues.
    """
    if emit:
        emit("progress", "LLM analyzing model code...", {"phase": "analyze"})

    stats_block = ""
    if stats_text:
        stats_block = f"""
=== TRAINING RESULTS & FAILURE ANALYSIS ===
{stats_text}
"""

    prompt = f"""\
You are an expert ML engineer reviewing a CIFAR-10 image classifier.
Analyze the model code below and identify the top 5 specific code-level
issues that are causing poor performance. Be concrete and actionable.
{stats_block}
=== MODEL CODE ===
{user_code}

For each issue, write:
ISSUE N: [title]
PROBLEM: [what's wrong in the code]
IMPACT: [which failures this causes]
FIX: [specific code change needed]
"""
    return _llm_call(prompt, ANALYZE_TOKENS)


def llm_generate_improved(user_code: str, analysis: str,
                          error_feedback: str = "",
                          emit: Emitter = None) -> str:
    """
    LLM generates an improved model class.
    Returns Python code string.
    """
    if emit:
        emit("progress", "LLM generating improved model...", {"phase": "generate"})

    error_ctx = ""
    if error_feedback:
        error_ctx = f"\nYOUR PREVIOUS CODE HAD THIS ERROR — FIX IT:\n{error_feedback}\n"

    prompt = f"""\
Generate an improved PyTorch Lightning model class for CIFAR-10 (32x32, 10 classes)
that fixes these issues:

{analysis[:1200]}
{error_ctx}
REQUIREMENTS:
- Class name: ImprovedCNN (extends pl.LightningModule)
- Input: 3x32x32 images → output: 10 classes
- Must have: __init__, forward, configure_optimizers, training_step, validation_step
- training_step logs 'train_loss' with self.log
- validation_step logs 'val_loss' and 'accuracy' with self.log
- Use BatchNorm, residual connections where appropriate
- Include any helper classes (e.g. attention blocks) in the same code
- Available imports: torch, nn, F, pl, CosineAnnealingLR

Output ONLY Python code inside ```python``` fences. No explanations."""

    raw = _llm_call(prompt, GENERATE_TOKENS)
    return extract_python_blocks(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK MODEL (guaranteed to work if LLM fails)
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_MODEL = textwrap.dedent("""\
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        m = max(ch // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, m, bias=False), nn.ReLU(True),
                                 nn.Linear(m, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1, drop=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride, 1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, 1, 1, bias=False), nn.BatchNorm2d(outc),
            SEBlock(outc), nn.Dropout2d(drop))
        self.skip = (nn.Sequential(nn.Conv2d(inc, outc, 1, stride, bias=False),
                     nn.BatchNorm2d(outc)) if stride != 1 or inc != outc else nn.Identity())
    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))

class ImprovedCNN(pl.LightningModule):
    def __init__(self, lr=0.001, wd=1e-4, epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(True))
        self.stages = nn.Sequential(
            ResBlock(64, 64), ResBlock(64, 64),
            ResBlock(64, 128, stride=2), ResBlock(128, 128),
            ResBlock(128, 256, stride=2), ResBlock(256, 256),
            ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Dropout(0.3), nn.Linear(512, 10))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    def forward(self, x):
        return self.head(self.stages(self.stem(x)))
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.wd)
        sch = CosineAnnealingLR(opt, self.hparams.epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
    def training_step(self, batch, _):
        x, y = batch; loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss, prog_bar=True); return loss
    def validation_step(self, batch, _):
        x, y = batch; logits = self(x); loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss); self.log('accuracy', acc); return loss
""")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs the full end-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_quick_analysis(user_code: str, emit: Emitter = None) -> Dict:
    """
    Quick mode: LLM analyzes code statically (no training).
    Returns {analysis, improved_code, issues_count}.
    """
    if emit:
        emit("phase", "Quick Analysis", {"step": 1, "total": 3})

    # 1) Analyze
    if emit:
        emit("step", "Analyzing model code...", {"step": 1})
    analysis = llm_analyze_code(user_code, emit=emit)

    # 2) Generate improved code
    if emit:
        emit("step", "Generating improved model...", {"step": 2})
    improved = ""
    error_fb = ""
    for attempt in range(MAX_FIX_RETRIES):
        improved = llm_generate_improved(user_code, analysis, error_fb, emit=emit)
        valid, err = validate_code(improved)
        if valid and "class ImprovedCNN" in improved:
            break
        error_fb = err or "Missing ImprovedCNN class"
        if emit:
            emit("warning", f"Retry {attempt+1}: {error_fb}", {})
    else:
        improved = FALLBACK_MODEL
        if emit:
            emit("warning", "Using fallback model", {})

    # 3) Done
    if emit:
        emit("step", "Analysis complete!", {"step": 3})

    return {
        "analysis": analysis,
        "improved_code": improved,
        "used_fallback": improved == FALLBACK_MODEL,
    }


def run_full_pipeline(user_code: str, epochs: int = DEFAULT_EPOCHS,
                      emit: Emitter = None) -> Dict:
    """
    Full mode: train original → analyze stats → LLM fix → train improved → compare.
    """
    result = {
        "original": {}, "improved_code": "", "improved": {},
        "analysis": "", "comparison": {},
    }

    total_steps = 7

    # ── Step 1: Wrap & train original ─────────────────────────────────────────
    if emit:
        emit("step", "Preparing original model...", {"step": 1, "total": total_steps})

    script_orig, orig_class = wrap_user_code(user_code, epochs, tag="original")

    if emit:
        emit("step", f"Training {orig_class} for {epochs} epochs...",
             {"step": 2, "total": total_steps})

    train_result = train_model(script_orig, emit)
    if not train_result["success"]:
        result["error"] = f"Original model training failed: {train_result['error']}"
        if emit:
            emit("error", result["error"], {})
        return result

    # ── Step 2: Collect stats ─────────────────────────────────────────────────
    if emit:
        emit("step", "Collecting failure statistics...", {"step": 3, "total": total_steps})

    stats = collect_stats(emit)
    result["original"] = stats

    stats_text = format_stats_for_llm(stats)

    # ── Step 3: LLM analyze ──────────────────────────────────────────────────
    if emit:
        emit("step", "LLM analyzing model weaknesses...", {"step": 4, "total": total_steps})

    analysis = llm_analyze_code(user_code, stats_text, emit)
    result["analysis"] = analysis

    # ── Step 4: LLM generate improved code ────────────────────────────────────
    if emit:
        emit("step", "LLM generating improved model...", {"step": 5, "total": total_steps})

    improved_code = ""
    error_feedback = ""
    for attempt in range(MAX_FIX_RETRIES):
        improved_code = llm_generate_improved(user_code, analysis, error_feedback, emit)
        valid, err = validate_code(improved_code)
        if valid and "class ImprovedCNN" in improved_code:
            break
        error_feedback = err or "Missing ImprovedCNN class"
        if emit:
            emit("warning", f"Code fix retry {attempt+1}: {error_feedback}", {})
    else:
        improved_code = FALLBACK_MODEL
        if emit:
            emit("warning", "Using pre-built fallback model", {})

    result["improved_code"] = improved_code

    # ── Step 5: Train improved model ──────────────────────────────────────────
    if emit:
        emit("step", f"Training ImprovedCNN for {epochs} epochs...",
             {"step": 6, "total": total_steps})

    script_imp, _ = wrap_user_code(improved_code, epochs, tag="improved")
    train_imp = train_model(script_imp, emit)

    if not train_imp["success"]:
        # Try fallback
        if improved_code != FALLBACK_MODEL:
            if emit:
                emit("warning", "Improved model failed, trying fallback...", {})
            improved_code = FALLBACK_MODEL
            result["improved_code"] = improved_code
            script_imp, _ = wrap_user_code(improved_code, epochs, tag="improved")
            train_imp = train_model(script_imp, emit)

        if not train_imp["success"]:
            result["error"] = f"Improved model training failed: {train_imp['error']}"
            if emit:
                emit("error", result["error"], {})
            return result

    # ── Step 6: Collect improved stats ────────────────────────────────────────
    if emit:
        emit("step", "Collecting improved model results...",
             {"step": 7, "total": total_steps})

    improved_stats = collect_stats(emit)
    result["improved"] = improved_stats

    # ── Step 7: Build comparison ──────────────────────────────────────────────
    orig_acc = stats.get("best_accuracy", 0)
    imp_acc = improved_stats.get("best_accuracy", 0)
    orig_mc = stats.get("total_misclassified", 0)
    imp_mc = improved_stats.get("total_misclassified", 0)

    result["comparison"] = {
        "original_accuracy": orig_acc,
        "improved_accuracy": imp_acc,
        "accuracy_delta": round((imp_acc - orig_acc) * 100, 2) if isinstance(orig_acc, (int, float)) and isinstance(imp_acc, (int, float)) else 0,
        "original_misclassified": orig_mc,
        "improved_misclassified": imp_mc,
        "misclassified_reduction": orig_mc - imp_mc if isinstance(orig_mc, int) and isinstance(imp_mc, int) else 0,
    }

    if emit:
        emit("complete", "Pipeline complete!", result["comparison"])

    return result
