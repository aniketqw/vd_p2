"""
v3/rag.py
=========
RAG (Retrieval-Augmented Generation) store for past failure history.

What it does:
  Builds a searchable index of every misclassified image across ALL training
  runs. At query time, given an embedding of the current images being analysed,
  it retrieves the K most similar past failures and formats them as a
  "historical precedent" context block for the VLM prompt.

  This gives the VLM cross-run memory: it can say "this blur/cat failure is
  structurally identical to 12 prior cases — and their root cause was X."

Storage:
  Index is persisted as a .npz file (numpy compressed archive) alongside
  the distortion report. Rebuilt automatically if missing or stale.

Public API
----------
RAGStore.build(all_mc_stats, index_path)      → RAGStore
RAGStore.load(index_path)                     → RAGStore
RAGStore.retrieve(query_emb, dist_type, k)    → List[Dict]
RAGStore.format_context(retrieved)            → str
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (
    CIFAR10_CLASSES, DISTORTION_TYPES,
    RAG_TOP_K, RAG_SIMILARITY_THRESHOLD,
)
from .embeddings import extract_embedding_b64, cosine_similarity, top_k_indices


class RAGStore:
    """
    Vector store of misclassified image embeddings across all training runs.

    Attributes:
        embeddings : np.ndarray  shape (N, 3072) — one row per past failure
        metadata   : List[Dict]  — parallel list of metadata dicts
    """

    def __init__(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.embeddings = embeddings   # (N, 3072) float32, L2-normalised
        self.metadata   = metadata     # [{distortion, true_label, pred_label,
                                       #   confidence, run_index, hash}, ...]

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        all_mc_stats: List[Dict],
        index_path:   Optional[Path] = None,
    ) -> "RAGStore":
        """
        Build the RAG index from a list of misclassified stats dicts
        (one per training run, as returned by data_loader.load_all_misclassified).

        Extracts pixel embeddings from the base64 images stored in each
        run's JSON. Persists to index_path if provided.
        """
        all_embs:  List[np.ndarray] = []
        all_meta:  List[Dict]       = []

        for run_stats in all_mc_stats:
            if "error" in run_stats:
                continue
            run_idx = run_stats.get("run_index", 0)
            samples = run_stats.get("_raw_samples", [])

            print(f"  [RAG] Indexing run {run_idx}: {len(samples)} samples…")
            for sample in samples:
                b64 = sample.get("image_base64")
                if not b64:
                    continue
                try:
                    emb = extract_embedding_b64(b64)
                    all_embs.append(emb)
                    all_meta.append({
                        "distortion":  str(sample.get("distortion_predicted") or "unknown"),
                        "true_label":  CIFAR10_CLASSES.get(
                                           sample.get("true_label"),
                                           str(sample.get("true_label", "?"))),
                        "pred_label":  CIFAR10_CLASSES.get(
                                           sample.get("predicted_label"),
                                           str(sample.get("predicted_label", "?"))),
                        "confidence":  sample.get("distortion_confidence") or 0.0,
                        "epoch":       sample.get("epoch"),
                        "run_index":   run_idx,
                        "hash":        sample.get("hash", ""),
                    })
                except Exception:
                    continue  # skip unembeddable images silently

        if not all_embs:
            print("  [RAG] Warning: no embeddings extracted — RAG will return empty context.")
            return cls(np.zeros((0, 3072), dtype=np.float32), [])

        matrix = np.stack(all_embs).astype(np.float32)
        store  = cls(matrix, all_meta)

        if index_path is not None:
            store.save(index_path)
            print(f"  [RAG] Index saved → {index_path}  ({len(all_meta)} vectors)")

        return store

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, index_path: Path):
        """Persist embeddings + metadata to a .npz file."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            index_path,
            embeddings=self.embeddings,
            metadata=np.array(
                [json.dumps(m) for m in self.metadata], dtype=object
            ),
        )

    @classmethod
    def load(cls, index_path: Path) -> "RAGStore":
        """Load a previously saved RAG index from a .npz file."""
        data     = np.load(index_path, allow_pickle=True)
        embs     = data["embeddings"].astype(np.float32)
        metadata = [json.loads(m) for m in data["metadata"]]
        print(f"  [RAG] Index loaded ← {index_path}  ({len(metadata)} vectors)")
        return cls(embs, metadata)

    @classmethod
    def load_or_build(
        cls,
        index_path:   Path,
        all_mc_stats: List[Dict],
        force_rebuild: bool = False,
    ) -> "RAGStore":
        """
        Load from disk if the index exists and is not stale, else rebuild.
        Staleness check: if the most recent misclassified JSON is newer than
        the index file, rebuild automatically.
        """
        if index_path.exists() and not force_rebuild:
            # Check staleness: newest source file vs index mtime
            newest_src = max(
                (Path(s["path"]).stat().st_mtime
                 for s in all_mc_stats if "path" in s and Path(s["path"]).exists()),
                default=0.0,
            )
            index_mtime = index_path.stat().st_mtime
            if index_mtime >= newest_src:
                return cls.load(index_path)
            print("  [RAG] Index is stale — rebuilding…")

        return cls.build(all_mc_stats, index_path)

    # ── retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_emb: np.ndarray,
        dist_type: str,
        k:         int = RAG_TOP_K,
        threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> List[Dict]:
        """
        Retrieve the k most similar past failures for the given distortion type.

        Filters the index to the given distortion type first, then runs
        cosine similarity search on that subset.

        Returns a list of dicts: {true_label, pred_label, confidence,
                                   run_index, similarity, distortion}
        Excludes entries below threshold.
        """
        if self.embeddings.shape[0] == 0:
            return []

        # Filter to same distortion type
        dt_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("distortion") == dist_type
        ]
        if not dt_indices:
            return []

        sub_matrix = self.embeddings[dt_indices]
        sims       = cosine_similarity(query_emb, sub_matrix)

        k_actual = min(k, len(dt_indices))
        top_local = np.argsort(sims)[::-1][:k_actual].tolist()

        results = []
        for local_idx in top_local:
            sim = float(sims[local_idx])
            if sim < threshold:
                continue
            global_idx = dt_indices[local_idx]
            entry = dict(self.metadata[global_idx])
            entry["similarity"] = round(sim, 4)
            results.append(entry)

        return results

    # ── context formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_context(retrieved: List[Dict]) -> str:
        """
        Format retrieved RAG results as a readable context block
        for injection into the Turn 2 VLM prompt.
        """
        if not retrieved:
            return ""

        lines = [
            "━━━ RETRIEVED SIMILAR FAILURES FROM PAST RUNS ━━━",
            f"(top {len(retrieved)} most visually similar past failures — "
            "same distortion type, cosine similarity on pixel embeddings)",
            "",
        ]
        for i, r in enumerate(retrieved, 1):
            lines.append(
                f"  Past Case {i}  [run {r.get('run_index', '?')}, "
                f"sim={r.get('similarity', 0):.3f}]:  "
                f"true={r['true_label']}  →  predicted={r['pred_label']}  "
                f"(dist_conf={r.get('confidence', 0):.4f})"
            )
        lines += [
            "",
            "Do these past cases share the same failure mechanism as the "
            "current images, or does the current batch represent a novel pattern?",
        ]
        return "\n".join(lines)
