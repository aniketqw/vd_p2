"""
v3/data_loader.py
=================
Load all pipeline artifacts from disk.
Responsibility: JSON parsing and summary building only — no LLM, no images.

Public API
----------
load_training_summary(log_path)       → Dict
load_misclassified_stats(mc_path)     → Dict
load_distortion_report(report_path)  → Dict
load_all_runs(logs_dir)              → List[Dict]   ← NEW: all training logs
find_latest_file(directory, pattern) → Path | None
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import CIFAR10_CLASSES, DISTORTION_TYPES


# ── file discovery ─────────────────────────────────────────────────────────────

def find_latest_file(directory: Path, glob_pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching glob_pattern, or None."""
    matches = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def find_all_files(directory: Path, glob_pattern: str) -> List[Path]:
    """Return all files matching glob_pattern, sorted oldest→newest."""
    return sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)


# ── single-run loaders ─────────────────────────────────────────────────────────

def load_training_summary(log_path: Path) -> Dict:
    """Parse a training_log JSON and return a compact summary dict."""
    with open(log_path) as f:
        raw = json.load(f)
    summary = raw.get("summary", {})
    epochs  = raw.get("epochs", [])
    return {
        "path":                 str(log_path),
        "best_accuracy":        summary.get("best_accuracy"),
        "best_epoch":           summary.get("best_epoch"),
        "total_epochs":         summary.get("total_epochs"),
        "total_misclassified":  summary.get("total_misclassified"),
        "final_train_loss":     summary.get("final_train_loss"),
        "final_val_loss":       summary.get("final_val_loss"),
        "epochs": [
            {
                "epoch":             e.get("epoch"),
                "accuracy":          e.get("accuracy") or 0,
                "overall_accuracy":  e.get("overall_accuracy") or 0,
                "f1_score":          e.get("f1_score") or 0,
                "num_misclassified": e.get("num_misclassified") or 0,
            }
            for e in epochs
        ],
        "dataset_classes": summary.get("dataset_info", {}).get("classes", CIFAR10_CLASSES),
    }


def load_misclassified_stats(mc_path: Path) -> Dict:
    """Parse misclassified JSON; return per-distortion counts and confusion data."""
    with open(mc_path) as f:
        raw = json.load(f)

    samples = raw.get("misclassified_samples", [])
    by_dist: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        dt = str(s.get("distortion_predicted") or "unknown")
        by_dist[dt].append(s)

    stats: Dict = {
        "path":                 str(mc_path),
        "n_misclassified":      raw.get("n_misclassified", len(samples)),
        "by_distortion":        {},
        "top_confusion_pairs":  [],
    }

    for dt in DISTORTION_TYPES + ["unknown"]:
        group = by_dist.get(dt, [])
        if not group:
            continue
        confs     = [s.get("distortion_confidence") or 0 for s in group]
        true_cnt  = Counter(s.get("true_label")      for s in group)
        pred_cnt  = Counter(s.get("predicted_label") for s in group)
        epoch_cnt = Counter(s.get("epoch")           for s in group)
        stats["by_distortion"][dt] = {
            "count":           len(group),
            "avg_confidence":  round(sum(confs) / len(confs), 4) if confs else 0,
            "top_true_labels": [
                (CIFAR10_CLASSES.get(lbl, str(lbl)), cnt)
                for lbl, cnt in true_cnt.most_common(5)
            ],
            "top_pred_labels": [
                (CIFAR10_CLASSES.get(lbl, str(lbl)), cnt)
                for lbl, cnt in pred_cnt.most_common(5)
            ],
            "epoch_distribution": dict(sorted(epoch_cnt.items())),
        }

    pair_cnt = Counter(
        (s.get("true_label"), s.get("predicted_label")) for s in samples
    )
    stats["top_confusion_pairs"] = [
        {
            "true":  CIFAR10_CLASSES.get(t, str(t)),
            "pred":  CIFAR10_CLASSES.get(p, str(p)),
            "count": cnt,
        }
        for (t, p), cnt in pair_cnt.most_common(20)
    ]
    stats["_hash_to_meta"] = {
        s.get("hash"): s for s in samples if s.get("hash")
    }
    # store raw samples for RAG indexing
    stats["_raw_samples"] = samples
    return stats


def load_distortion_report(report_path: Path) -> Dict:
    """Parse distortion_report.json (archetype/outlier image paths + distances)."""
    with open(report_path) as f:
        return json.load(f)


# ── multi-run loader (NEW in v3) ───────────────────────────────────────────────

def load_all_runs(logs_dir: Path) -> List[Dict]:
    """
    Load ALL training_log_*.json files in logs_dir, oldest first.
    Returns a list of compact summary dicts — same shape as load_training_summary()
    but with an added 'run_index' key for ordering.

    Used by:
      - RAG store: build a cross-run failure history
      - Longitudinal context injected into Turn 2
    """
    log_files = find_all_files(logs_dir, "training_log_*.json")
    runs = []
    for idx, path in enumerate(log_files):
        try:
            summary = load_training_summary(path)
            summary["run_index"] = idx
            runs.append(summary)
        except Exception as e:
            # Don't crash if one log is malformed
            runs.append({
                "path": str(path),
                "run_index": idx,
                "error": str(e),
            })
    return runs


def load_all_misclassified(logs_dir: Path) -> List[Dict]:
    """
    Load ALL misclassified_*.json files in logs_dir, oldest first.
    Returns a list of stats dicts — same shape as load_misclassified_stats()
    with an added 'run_index' key.

    Used by:
      - RAG store: index all past failures for retrieval
    """
    mc_files = find_all_files(logs_dir, "misclassified_*.json")
    all_stats = []
    for idx, path in enumerate(mc_files):
        try:
            stats = load_misclassified_stats(path)
            stats["run_index"] = idx
            all_stats.append(stats)
        except Exception as e:
            all_stats.append({
                "path": str(path),
                "run_index": idx,
                "error": str(e),
            })
    return all_stats
