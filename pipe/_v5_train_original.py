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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss)
        self.log("accuracy", (preds == y).float().mean())
        return loss

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
    model = SimpleCNN()
    best_pt = os.path.join(os.path.dirname(__file__), 'best.pt')
    logger = DebugLogger(save_dir='./logs', distortion_model_path=best_pt)
    trainer = pl.Trainer(max_epochs=3, callbacks=[logger],
                         enable_progress_bar=True, logger=False)
    trainer.fit(model, train_ld, val_ld)
    print("TRAINING_COMPLETE")
