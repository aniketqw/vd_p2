"""
v3/image_sampler.py
===================
Image sampling with semantic deduplication.

Improvements over v2:
  - semantic_deduplicate() ensures the N images sent to the VLM are
    maximally visually diverse, preventing the VLM from analysing
    near-identical images twice.

Public API
----------
resolve_image_path(raw_path)                               → Path | None
gather_images_for_distortion(dist_type, report, mc_stats,
                              images_dir, n_samples, seed)  → List[Dict]
semantic_deduplicate(items, threshold)                     → List[Dict]
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import DEDUP_SIMILARITY_THRESHOLD
from .embeddings import extract_embedding, cosine_similarity


# ── path resolution ────────────────────────────────────────────────────────────

def resolve_image_path(raw_path: str) -> Optional[Path]:
    """
    Resolve an image path that may be absolute /mnt/... or relative.
    Falls back to the symlinked home path if /mnt/... does not exist.
    """
    p = Path(raw_path)
    if p.exists():
        return p
    try:
        rel = p.relative_to("/mnt/data/vision_dev_project")
        alt = Path("/home/pratik2/vision_dev_project") / rel
        if alt.exists():
            return alt
    except ValueError:
        pass
    return None


# ── primary sampler ────────────────────────────────────────────────────────────

def gather_images_for_distortion(
    dist_type:  str,
    report:     Dict,
    mc_stats:   Dict,
    images_dir: Optional[Path],
    n_samples:  int,
    seed:       int = 42,
) -> List[Dict]:
    """
    Return up to n_samples image items for the given distortion type.
    Priority: typical archetype → outlier archetype → random from images_dir.

    Each item:
      {"path": Path, "role": str, "distance": float|None, "meta": dict|None}
    """
    rng          = random.Random(seed)
    collected:   List[Dict] = []
    seen_paths   = set()
    hash_to_meta = mc_stats.get("_hash_to_meta", {})
    archetypes   = report.get("archetypes", {}).get(dist_type, {})

    def _add(raw_path: str, role: str, distance: Optional[float]):
        if len(collected) >= n_samples:
            return
        p = resolve_image_path(raw_path)
        if p is None or str(p) in seen_paths:
            return
        seen_paths.add(str(p))
        meta = hash_to_meta.get(p.stem)
        collected.append({"path": p, "role": role, "distance": distance, "meta": meta})

    for item in archetypes.get("typical", []):
        _add(item["file"], "typical", item.get("distance"))
    for item in archetypes.get("outlier", []):
        _add(item["file"], "outlier", item.get("distance"))

    # fill remaining slots from random images folder
    if len(collected) < n_samples and images_dir is not None:
        folder = images_dir / dist_type
        if folder.exists():
            all_imgs = [
                p for p in folder.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                and str(p) not in seen_paths
            ]
            rng.shuffle(all_imgs)
            for p in all_imgs:
                if len(collected) >= n_samples:
                    break
                meta = hash_to_meta.get(p.stem)
                collected.append({"path": p, "role": "random", "distance": None, "meta": meta})

    return collected


# ── semantic deduplication (NEW in v3) ─────────────────────────────────────────

def semantic_deduplicate(
    items:     List[Dict],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> List[Dict]:
    """
    Remove near-duplicate images from a sampled list using cosine similarity
    on pixel-level embeddings.

    Algorithm:
      Greedily keep the first image, then for each subsequent image only keep
      it if its cosine similarity to ALL already-kept images is below threshold.

    This guarantees the final set is maximally visually diverse — preventing
    the VLM from seeing the same failure mode twice in one batch.

    Args:
        items     : list of image items from gather_images_for_distortion()
        threshold : cosine similarity above which two images are "duplicates"
                    (default 0.96 — very high, catches near-identical images)

    Returns:
        Filtered list of items. Deduped items are logged but not silently dropped.
    """
    if len(items) <= 1:
        return items

    kept:       List[Dict]       = []
    kept_embs:  List[np.ndarray] = []
    dropped = 0

    for item in items:
        try:
            emb = extract_embedding(item["path"])
        except Exception:
            # If we can't embed it, keep it (don't silently drop on error)
            kept.append(item)
            continue

        if not kept_embs:
            kept.append(item)
            kept_embs.append(emb)
            continue

        # Check similarity against all kept embeddings
        matrix = np.stack(kept_embs)
        sims   = cosine_similarity(emb, matrix)
        if float(sims.max()) >= threshold:
            dropped += 1
            continue  # near-duplicate — skip

        kept.append(item)
        kept_embs.append(emb)

    if dropped:
        print(f"  [dedup] Removed {dropped} near-duplicate image(s) "
              f"(threshold={threshold:.2f}). Kept {len(kept)}/{len(items)}.")

    return kept
