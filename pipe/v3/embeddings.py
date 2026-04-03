"""
v3/embeddings.py
================
Pixel-level image embeddings and cosine similarity search.
No deep-learning dependency required — uses normalized pixel vectors
which are surprisingly effective for 32×32 CIFAR-10 similarity search.

Public API
----------
extract_embedding(image_path)            → np.ndarray  shape (3072,)
extract_embedding_b64(b64_str)           → np.ndarray  shape (3072,)
build_matrix(paths_or_b64, mode)         → np.ndarray  shape (N, 3072)
cosine_similarity(query, matrix)         → np.ndarray  shape (N,)
top_k_indices(query, matrix, k)          → List[int]
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Union

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _load_image_array(source: Union[str, Path, bytes]) -> np.ndarray:
    """
    Load an image from a file path, base64 string, or raw bytes.
    Returns a (32, 32, 3) uint8 numpy array, resized if necessary.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for image embedding. pip install Pillow")

    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    else:
        # raw bytes
        img = Image.open(BytesIO(source)).convert("RGB")

    # Ensure 32×32 (CIFAR-10 native size; resize gracefully if needed)
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)

    return np.asarray(img, dtype=np.uint8)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector. Returns zero vector if norm is zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_embedding(image_path: Union[str, Path]) -> np.ndarray:
    """
    Extract a 3072-dim L2-normalized pixel embedding from an image file.
    Vector = flatten(HWC float32 / 255), then L2-normalize.
    """
    arr = _load_image_array(image_path)
    vec = arr.astype(np.float32).flatten() / 255.0
    return _normalize(vec)


def extract_embedding_b64(b64_str: str) -> np.ndarray:
    """
    Extract a 3072-dim L2-normalized pixel embedding from a base64 PNG string.
    """
    raw_bytes = base64.b64decode(b64_str)
    arr = _load_image_array(raw_bytes)
    vec = arr.astype(np.float32).flatten() / 255.0
    return _normalize(vec)


def build_matrix(sources: list, mode: str = "path") -> np.ndarray:
    """
    Build an (N, 3072) embedding matrix from a list of image sources.

    Args:
        sources : list of file paths (mode='path') or base64 strings (mode='b64')
        mode    : 'path' | 'b64'

    Returns:
        np.ndarray of shape (N, 3072), dtype float32.
        Rows where extraction fails are filled with zeros.
    """
    rows = []
    for src in sources:
        try:
            if mode == "b64":
                rows.append(extract_embedding_b64(src))
            else:
                rows.append(extract_embedding(src))
        except Exception:
            rows.append(np.zeros(3072, dtype=np.float32))
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, 3072), dtype=np.float32)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and every row of matrix.
    Both are assumed to be L2-normalized already.
    Returns shape (N,) float32.
    """
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    return (matrix @ query).astype(np.float32)


def top_k_indices(query: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    """
    Return the indices of the k most similar rows in matrix to query.
    """
    sims = cosine_similarity(query, matrix)
    if len(sims) == 0:
        return []
    k = min(k, len(sims))
    return np.argsort(sims)[::-1][:k].tolist()
