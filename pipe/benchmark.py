#!/usr/bin/env python3
"""
Benchmark: SimpleCNN vs ImprovedCNN on M3 Air (MPS)
Trains both models for 3 epochs on CIFAR-10 and compares timing + accuracy.
"""
import os
import sys
import time
from pathlib import Path

# Setup paths
_pipe = str(Path(__file__).resolve().parent)
if _pipe not in sys.path:
    sys.path.insert(0, _pipe)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 64

print("=" * 65)
print("  🏎️  BENCHMARK: SimpleCNN vs ImprovedCNN")
print(f"  Device: {DEVICE.upper()} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std  = (0.2470, 0.2435, 0.2616)

train_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
val_basic = train_basic

train_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value='random'),
])
val_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])


def get_loaders(train_tf, val_tf, bs=BATCH_SIZE):
    tr = datasets.CIFAR10('./data', train=True,  transform=train_tf, download=True)
    va = datasets.CIFAR10('./data', train=False, transform=val_tf)
    return (DataLoader(tr, bs, shuffle=True,  num_workers=0),
            DataLoader(va, bs, shuffle=False, num_workers=0))


# ═══════════════════════════════════════════════════════════════════
# MODEL 1: SimpleCNN (baseline)
# ═══════════════════════════════════════════════════════════════════

class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)
        self.drop  = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log('val_loss', F.cross_entropy(logits, y))
        self.log('accuracy', (logits.argmax(1) == y).float().mean(), prog_bar=True)


# ═══════════════════════════════════════════════════════════════════
# MODEL 2: ImprovedCNN (v5 architecture)
# ═══════════════════════════════════════════════════════════════════

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
    def __init__(self):
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
        opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, EPOCHS, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def training_step(self, batch, _):
        x, y = batch; loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss, prog_bar=True); return loss

    def validation_step(self, batch, _):
        x, y = batch; logits = self(x)
        self.log('val_loss', self.loss_fn(logits, y))
        self.log('accuracy', (logits.argmax(1) == y).float().mean(), prog_bar=True)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_benchmark(name, model_cls, train_tf, val_tf):
    print(f"\n{'─' * 65}")
    print(f"  🔧 {name}")
    print(f"{'─' * 65}")

    train_ld, val_ld = get_loaders(train_tf, val_tf)
    model = model_cls()
    n = count_params(model)
    print(f"  Parameters: {n:,}")

    accelerator = "mps" if DEVICE == "mps" else "cpu"
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )

    t0 = time.time()
    trainer.fit(model, train_ld, val_ld)
    elapsed = time.time() - t0

    # Get final metrics
    metrics = trainer.callback_metrics
    acc = metrics.get("accuracy", torch.tensor(0)).item()
    val_loss = metrics.get("val_loss", torch.tensor(0)).item()
    train_loss = metrics.get("train_loss", torch.tensor(0)).item()

    result = {
        "name": name,
        "params": n,
        "time_sec": elapsed,
        "time_per_epoch": elapsed / EPOCHS,
        "accuracy": acc,
        "val_loss": val_loss,
        "train_loss": train_loss,
    }

    print(f"\n  ✅ {name} complete:")
    print(f"     Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
    print(f"     Val Loss:       {val_loss:.4f}")
    print(f"     Total Time:     {elapsed:.1f}s")
    print(f"     Per Epoch:      {elapsed/EPOCHS:.1f}s")
    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n📥 Downloading CIFAR-10 (if needed)...")
    # Pre-download
    datasets.CIFAR10('./data', train=True, download=True)
    datasets.CIFAR10('./data', train=False, download=True)

    results = []

    # Benchmark 1: SimpleCNN (baseline)
    r1 = run_benchmark("SimpleCNN (baseline)", SimpleCNN, train_basic, val_basic)
    results.append(r1)

    # Benchmark 2: ImprovedCNN
    r2 = run_benchmark("ImprovedCNN (v5)", ImprovedCNN, train_aug, val_aug)
    results.append(r2)

    # ── Final Comparison ──────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  📊 FINAL COMPARISON — M3 Air, 3 Epochs")
    print("═" * 65)
    print(f"  {'Metric':<25} {'SimpleCNN':>15} {'ImprovedCNN':>15}")
    print(f"  {'─'*25} {'─'*15} {'─'*15}")
    print(f"  {'Parameters':<25} {r1['params']:>15,} {r2['params']:>15,}")
    print(f"  {'Total Time':<25} {r1['time_sec']:>14.1f}s {r2['time_sec']:>14.1f}s")
    print(f"  {'Per Epoch':<25} {r1['time_per_epoch']:>14.1f}s {r2['time_per_epoch']:>14.1f}s")
    print(f"  {'Accuracy':<25} {r1['accuracy']*100:>14.2f}% {r2['accuracy']*100:>14.2f}%")
    print(f"  {'Val Loss':<25} {r1['val_loss']:>15.4f} {r2['val_loss']:>15.4f}")

    delta_acc = (r2['accuracy'] - r1['accuracy']) * 100
    speedup = r1['time_sec'] / r2['time_sec'] if r2['time_sec'] > 0 else 0
    print(f"\n  📈 Accuracy gain:  {'+' if delta_acc >= 0 else ''}{delta_acc:.2f} percentage points")
    print(f"  ⏱️  Time ratio:    ImprovedCNN is {r2['time_sec']/r1['time_sec']:.1f}x {'slower' if r2['time_sec'] > r1['time_sec'] else 'faster'}")
    print(f"  🧠 Param ratio:   ImprovedCNN has {r2['params']/r1['params']:.1f}x more parameters")
    print("═" * 65)
