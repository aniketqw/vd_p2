"""
v3/tools.py
===========
Agent tool functions callable during the verify node of the agentic loop.

Each tool takes mc_stats (loaded from misclassified JSON) and returns a
plain Python dict. The graph.py verify_node calls these tools based on
the hypothesis formed in hypothesise_node and injects the results back
into the agent state before the next loop iteration.

Public API
----------
query_confusion_count(true_class, pred_class, mc_stats)  → Dict
get_epoch_trend(dist_type, mc_stats)                     → Dict
get_distortion_stats(dist_type, mc_stats)                → Dict
get_top_confusion_for_distortion(dist_type, mc_stats, n) → Dict
TOOL_REGISTRY                                            → Dict[str, callable]
"""

from typing import Dict


# ── individual tools ───────────────────────────────────────────────────────────

def query_confusion_count(
    true_class: str,
    pred_class: str,
    mc_stats:   Dict,
) -> Dict:
    """
    Return how many times true_class was predicted as pred_class.
    Also returns rank among all confusion pairs (1 = most common).

    Used by the agent to verify hypotheses like:
      "blur on cats causes dog predictions because silhouette is similar"
      → tool confirms: cat→dog is the #1 confusion pair (237 occurrences)
    """
    pairs = mc_stats.get("top_confusion_pairs", [])
    count = 0
    rank  = None
    for i, pair in enumerate(pairs, 1):
        if pair["true"] == true_class and pair["pred"] == pred_class:
            count = pair["count"]
            rank  = i
            break

    total = mc_stats.get("n_misclassified", 1) or 1
    return {
        "true_class":  true_class,
        "pred_class":  pred_class,
        "count":       count,
        "rank":        rank,
        "pct_of_total": round(count / total * 100, 2),
        "interpretation": (
            f"{true_class}→{pred_class} appears {count} times "
            f"({'rank #' + str(rank) if rank else 'not in top 20'} confusion pair, "
            f"{count/total*100:.1f}% of all failures)."
        ),
    }


def get_epoch_trend(dist_type: str, mc_stats: Dict) -> Dict:
    """
    Return how failures for dist_type are distributed across epochs.
    Tells the agent whether failures are concentrated early (undertrained)
    or persist late (structural model weakness).

    Interpretation logic:
      - If >70% of failures in epoch 0 → model never learned to handle this distortion
      - If failures spread evenly → persistent failure across training
      - If failures INCREASE in later epochs → catastrophic forgetting signal
    """
    by_dist = mc_stats.get("by_distortion", {})
    info    = by_dist.get(dist_type, {})
    if not info:
        return {"dist_type": dist_type, "error": "No data for this distortion type."}

    epoch_dist = info.get("epoch_distribution", {})
    total_dt   = info.get("count", 1) or 1
    pcts       = {
        f"epoch_{k}": round(v / total_dt * 100, 1)
        for k, v in epoch_dist.items()
    }

    # Auto-interpret the trend
    if epoch_dist:
        epochs_sorted = sorted(epoch_dist.keys())
        first_epoch_pct = epoch_dist.get(epochs_sorted[0], 0) / total_dt * 100
        last_epoch_pct  = epoch_dist.get(epochs_sorted[-1], 0) / total_dt * 100

        if first_epoch_pct > 70:
            interpretation = (
                f"{first_epoch_pct:.0f}% of {dist_type} failures in epoch 0 — "
                "model never learned to handle this distortion type. "
                "Strong signal: add this distortion to training augmentation."
            )
        elif last_epoch_pct > first_epoch_pct:
            interpretation = (
                f"Failures INCREASED in later epochs ({first_epoch_pct:.0f}% → "
                f"{last_epoch_pct:.0f}%) — possible catastrophic forgetting or "
                "overfitting to clean images."
            )
        else:
            interpretation = (
                f"Failures declined across epochs ({first_epoch_pct:.0f}% → "
                f"{last_epoch_pct:.0f}%) — model is learning but not converging fully."
            )
    else:
        interpretation = "No epoch distribution data available."

    return {
        "dist_type":        dist_type,
        "epoch_pcts":       pcts,
        "total_failures":   total_dt,
        "interpretation":   interpretation,
    }


def get_distortion_stats(dist_type: str, mc_stats: Dict) -> Dict:
    """
    Return a summary of failure statistics for a distortion type.
    Used by the agent to ground its hypothesis in the actual numbers.
    """
    by_dist   = mc_stats.get("by_distortion", {})
    info      = by_dist.get(dist_type, {})
    total_all = mc_stats.get("n_misclassified", 1) or 1

    if not info:
        return {"dist_type": dist_type, "error": "No data for this distortion type."}

    count   = info.get("count", 0)
    pct     = round(count / total_all * 100, 1)
    avg_conf = info.get("avg_confidence", 0)

    top_true = [f"{lbl}({cnt})" for lbl, cnt in info.get("top_true_labels", [])]
    top_pred = [f"{lbl}({cnt})" for lbl, cnt in info.get("top_pred_labels", [])]

    return {
        "dist_type":          dist_type,
        "count":              count,
        "pct_of_total":       pct,
        "avg_conf":           avg_conf,
        "top_true_classes":   top_true,
        "top_pred_classes":   top_pred,
        "interpretation": (
            f"{dist_type.capitalize()} accounts for {pct}% of all failures "
            f"({count} images). Most commonly misclassified: {', '.join(top_true[:3])}. "
            f"Most common wrong predictions: {', '.join(top_pred[:3])}. "
            f"Avg distortion classifier confidence: {avg_conf:.4f}."
        ),
    }


def get_top_confusion_for_distortion(
    dist_type: str,
    mc_stats:  Dict,
    n:         int = 5,
) -> Dict:
    """
    Return the top N confusion pairs within a specific distortion type.
    More specific than the global confusion pairs in mc_stats.
    """
    by_dist = mc_stats.get("by_distortion", {})
    info    = by_dist.get(dist_type, {})
    if not info:
        return {"dist_type": dist_type, "error": "No data."}

    top_true = info.get("top_true_labels", [])
    top_pred = info.get("top_pred_labels", [])

    pairs_str = []
    for (true_lbl, _), (pred_lbl, cnt) in zip(top_true[:n], top_pred[:n]):
        pairs_str.append(f"{true_lbl}→{pred_lbl} ({cnt} cases)")

    return {
        "dist_type":     dist_type,
        "top_pairs":     pairs_str,
        "interpretation": (
            f"Top confusion pairs for {dist_type}: " + ", ".join(pairs_str)
        ),
    }


# ── tool registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "query_confusion_count":           query_confusion_count,
    "get_epoch_trend":                 get_epoch_trend,
    "get_distortion_stats":            get_distortion_stats,
    "get_top_confusion_for_distortion": get_top_confusion_for_distortion,
}
