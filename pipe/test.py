"""
Test script for training_analytics package
Trains a simple CNN model on CIFAR-10 and logs all metrics using DebugLogger.

Usage
-----
  python3 test.py                          # training only (same as before)
  python3 test.py --version v1             # train → export → cluster → v1 report
  python3 test.py --version v2             # train → export → cluster → v2 report
  python3 test.py --version v3             # train → export → cluster → v3 report
  python3 test.py --version v3 --no-vlm   # same but skip VLM calls (stats only)

Pipeline steps (when --version is given)
-----------------------------------------
  Step 1  train.py            → logs/training_log_*.json
  Step 2  export_misclassified.py  → logs/misclassified_*.json
  Step 3  distortion_diagnostic_report.py
                              → reports/distortion_report.json
                              → reports/distortion_clusters.png
  Step 4  vision_reasoning_report[_v2|_v3].py
                              → ai_reasoning_summary[_v2|_v3].md (project root)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Ensure pipe/ is on sys.path regardless of how this file is invoked
# (python3 test.py, python3 -m pipe, python3 -m pipe.test, or import)
_pipe_dir = str(Path(__file__).resolve().parent)
if _pipe_dir not in sys.path:
    sys.path.insert(0, _pipe_dir)

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import DebugLogger from training_analytics package
from debug_logger import DebugLogger

_PIPE_DIR = Path(__file__).resolve().parent

_REPORT_SCRIPTS = {
    "v1": _PIPE_DIR / "v1" / "vision_reasoning_report.py",
    "v2": _PIPE_DIR / "v2" / "vision_reasoning_report_v2.py",
    "v3": _PIPE_DIR / "v3" / "vision_reasoning_report_v3.py",
}


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, 10)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss)
        self.log('accuracy', (preds == y).float().mean())
        return loss


# ── Data ──────────────────────────────────────────────────────────────────────

def setup_data_loaders(batch_size=32):
    """Setup CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True,  transform=transform, download=True)
    val_dataset   = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run(cmd: list, step: str):
    """Run a subprocess command, printing the step label; raise on failure."""
    print(f"\n{'='*70}")
    print(f"  {step}")
    print('='*70)
    result = subprocess.run(cmd, cwd=str(_PIPE_DIR))
    if result.returncode != 0:
        print(f"\n❌  {step} failed (exit {result.returncode}) — aborting pipeline.")
        sys.exit(result.returncode)


def run_pipeline(version: str, no_vlm: bool):
    """Steps 2-4: export → cluster → AI report, using the latest training log."""

    logs_dir    = _PIPE_DIR / "logs"
    reports_dir = _PIPE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ── Step 2: export misclassified ──────────────────────────────────────────
    training_logs = sorted(logs_dir.glob("training_log_*.json"))
    if not training_logs:
        print("❌  No training_log_*.json found in logs/ — did Step 1 succeed?")
        sys.exit(1)
    latest_log = training_logs[-1]
    ts         = latest_log.stem.replace("training_log_", "")
    mis_json   = logs_dir / f"misclassified_{ts}.json"

    _run(
        [sys.executable, str(_PIPE_DIR / "export_misclassified.py"),
         "--input",  str(latest_log),
         "--output", str(mis_json)],
        "Step 2/4 — Export misclassified samples",
    )

    # ── Step 3: distortion cluster diagnostic ────────────────────────────────
    _run(
        [sys.executable, str(_PIPE_DIR / "distortion_diagnostic_report.py"),
         "--base-dir", str(mis_json),
         "--output",   str(reports_dir / "distortion_report.json"),
         "--plot",     str(reports_dir / "distortion_clusters.png")],
        "Step 3/4 — Distortion diagnostic report (t-SNE + archetypes)",
    )

    # ── Step 4: AI reasoning report ───────────────────────────────────────────
    script      = _REPORT_SCRIPTS[version]
    project_root = _PIPE_DIR.parent
    extra = ["--no-vlm"] if no_vlm else []
    _run(
        [sys.executable, str(script),
         "--logs-dir", str(logs_dir),
         "--report",   str(reports_dir / "distortion_report.json"),
         "--output",   str(project_root / f"ai_reasoning_summary{'_' + version if version != 'v1' else ''}.md"),
         ] + extra,
        f"Step 4/4 — AI reasoning report ({version.upper()})"
        + (" [stats-only / no VLM]" if no_vlm else " [VLM enabled — server must be running]"),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 SimpleCNN and optionally run the full analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test.py                        # training only
  python3 test.py --version v3           # full pipeline with v3 report
  python3 test.py --version v2 --no-vlm  # pipeline, skip VLM calls
""",
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3"], default=None,
        help="After training, run the full pipeline with this report version. "
             "Omit to run training only.",
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Pass --no-vlm to the report script (stats-only, no GPU/server needed). "
             "Only relevant when --version is set.",
    )
    args = parser.parse_args()

    # ── Step 1: training ──────────────────────────────────────────────────────
    print("="*70)
    print("Step 1/4 — Training SimpleCNN on CIFAR-10" if args.version else
          "Training SimpleCNN on CIFAR-10")
    print("="*70)

    print("\n📊 Loading CIFAR-10 dataset...")
    train_loader, val_loader = setup_data_loaders(batch_size=32)
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples:   {len(val_loader.dataset)}")

    print("\n🔧 Initializing SimpleCNN model...")
    model = SimpleCNN(learning_rate=0.001)
    print("✓ Model initialized")

    print("\n📝 Initializing DebugLogger...")
    best_pt_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    debug_logger = DebugLogger(save_dir='./logs', distortion_model_path=best_pt_path)
    print("✓ DebugLogger initialized")

    print("\n🚀 Creating Trainer...")
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[debug_logger],
        enable_progress_bar=True,
        logger=False,
    )
    print("✓ Trainer created")

    print("\n🎯 Starting training...\n")
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "="*70)
    print("✅ Training completed!")
    print("="*70)
    print("\n📁 logs/training_log_{timestamp}.json written.")

    if args.version is None:
        print("\nTip: run with --version v3 to continue the full analysis pipeline.")
        print("="*70)
        return

    # ── Steps 2-4: pipeline ───────────────────────────────────────────────────
    run_pipeline(args.version, args.no_vlm)

    print("\n" + "="*70)
    print("✅ Full pipeline complete!")
    print("="*70)


if __name__ == "__main__":
    main()
