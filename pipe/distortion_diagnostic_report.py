"""Generate a diagnostic report for distortion misclassifications.

Expects a directory structure like:

  /path/to/misclassified/
      blur/
      jpeg/
      pixelate/
      noise/

Each folder should contain image files (PNG/JPEG) representing misclassified
images annotated with that distortion type.

Outputs:
  - A scatter plot of all images embedded in 2-D space (PCA → t-SNE).
  - 3 "typical" (closest-to-centroid) and 3 "outlier" (furthest-from-centroid)
    images selected per distortion category.
  - A JSON report describing those selected images and their centroid distances.

Usage:
  python distortion_diagnostic_report.py \\
    --base-dir /path/to/misclassified \\
    --output report.json \\
    --plot output_plot.png

Dry-run (processes whatever images are available):
  python distortion_diagnostic_report.py --dry-run \\
    --base-dir /path/to/misclassified
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import base64
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES: List[str] = ["blur", "jpeg", "pixelate", "noise"]  # Default/fallback
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Fallback: CIFAR predicted class → distortion archetype.
# Mirrors DebugLogger._CIFAR_TO_DISTORTION for legacy logs where
# distortion_predicted is "unknown" or None.
_CIFAR_NAME_TO_DISTORTION = {
    "airplane": "jpeg",   "automobile": "jpeg",
    "bird":     "noise",  "cat":        "blur",
    "deer":     "noise",  "dog":        "blur",
    "frog":     "noise",  "horse":      "pixelate",
    "ship":     "pixelate", "truck":    "jpeg",
}
_CIFAR_IDX_TO_DISTORTION = {
    "0": "jpeg",  "1": "jpeg",  "2": "noise", "3": "blur",
    "4": "noise", "5": "blur",  "6": "noise", "7": "pixelate",
    "8": "pixelate", "9": "jpeg",
}


def _resolve_distortion(sample: dict) -> str:
    """Return the distortion label for a sample, with CIFAR fallback."""
    dt_raw = sample.get("distortion_predicted")
    if dt_raw and dt_raw not in ("unknown", "None"):
        return str(dt_raw)
    cifar_val = str(sample.get("distortion_type", ""))
    return (
        _CIFAR_NAME_TO_DISTORTION.get(cifar_val)
        or _CIFAR_IDX_TO_DISTORTION.get(cifar_val)
        or "unknown"
    )


# ── Image I/O helpers ─────────────────────────────────────────────────────────

def _get_image_paths(base_dir: Path, categories: List[str]) -> Dict[str, List[Path]]:
    """Return a dict mapping each category to sorted image paths under base_dir."""
    out: Dict[str, List[Path]] = {}
    for cat in categories:
        folder = base_dir / cat
        if not folder.is_dir():
            out[cat] = []
        else:
            out[cat] = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
    return out


def _load_and_preprocess(img_path: Path, transform) -> torch.Tensor:
    """Open an image file and apply the given transform."""
    return transform(Image.open(img_path).convert("RGB"))


def _extract_images_from_log(
    json_path: Path,
    out_dir: Path,
    categories: List[str],
) -> Path:
    """Decode base64-encoded images from a misclassified JSON log onto disk.

    Images are placed in subfolders under *out_dir* named after the predicted
    distortion label (falls back to 'unknown' for unrecognised labels).

    Returns the resolved *out_dir* path.
    """
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    samples = data.get("misclassified_samples", [])
    if not samples:
        raise SystemExit(f"No misclassified samples found in: {json_path}")

    # Pre-create all category subfolders
    for cat in categories + ["unknown"]:
        (out_dir / cat).mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        b64 = sample.get("image_base64")
        if not b64:
            continue

        distortion = _resolve_distortion(sample)
        if distortion not in categories:
            distortion = "unknown"

        fname = sample.get("hash") or f"img_{idx:06d}"
        out_path = out_dir / distortion / f"{fname}.png"
        if out_path.exists():
            continue

        try:
            out_path.write_bytes(base64.b64decode(b64))
        except Exception:
            continue  # skip corrupt entries

    return out_dir


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(
    image_paths: List[Path],
    model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run images through *model* in batches and return stacked feature vectors."""
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = torch.stack(
                [_load_and_preprocess(p, transform) for p in image_paths[i : i + batch_size]]
            ).to(device)
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features, axis=0) if features else np.zeros((0,))


# ── Archetype selection ───────────────────────────────────────────────────────

def _select_archetypes(
    embeddings: np.ndarray,
    labels: List[str],
    categories: List[str],
    paths: List[Path],
    choose_k: int = 3,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """For each category, pick the *choose_k* most-typical and most-outlier samples.

    Typicality is measured by distance to the per-category centroid in embedding
    space: smaller distance → more typical; larger → more outlier.
    """
    report: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for cat in categories:
        cat_idx = [i for i, lbl in enumerate(labels) if lbl == cat]
        if not cat_idx:
            report[cat] = {"typical": [], "outlier": []}
            continue

        cat_emb = embeddings[cat_idx]
        centroid = cat_emb.mean(axis=0)
        dists = np.linalg.norm(cat_emb - centroid, axis=1)
        order = np.argsort(dists)
        k = min(choose_k, len(order))

        report[cat] = {
            "typical": [(str(paths[cat_idx[i]]), float(dists[i])) for i in order[:k]],
            "outlier":  [(str(paths[cat_idx[i]]), float(dists[i])) for i in order[-k:][::-1]],
        }
    return report


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_cluster_plot(
    embeddings: np.ndarray,
    labels: List[str],
    archetypes: Dict[str, Dict[str, List[Tuple[str, float]]]],
    path_to_index: Dict[Path, int],
    output_path: Path,
    dpi: int = 200,
) -> None:
    """Save a 2-D scatter plot of the t-SNE embeddings with archetype markers."""
    palette = plt.get_cmap("tab10")
    unique_labels = sorted(set(labels))
    label_to_color_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    xs, ys = embeddings[:, 0], embeddings[:, 1]
    cs = [palette(label_to_color_idx[lbl] % 10) for lbl in labels]

    plt.figure(figsize=(12, 10), dpi=dpi)
    plt.scatter(xs, ys, c=cs, s=18, alpha=0.7, edgecolors="none")

    # Overlay archetype markers: ★ = typical, ✕ = outlier
    for cat, ar in archetypes.items():
        color = [palette(label_to_color_idx.get(cat, 0) % 10)]
        for path_str, _ in ar.get("typical", []):
            idx = path_to_index.get(Path(path_str))
            if idx is not None:
                plt.scatter(xs[idx], ys[idx], marker="*", s=150, c=color, edgecolor="k", linewidth=0.7)
        for path_str, _ in ar.get("outlier", []):
            idx = path_to_index.get(Path(path_str))
            if idx is not None:
                plt.scatter(xs[idx], ys[idx], marker="X", s=150, c=color, edgecolor="k", linewidth=0.7)

    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title("Distortion cluster map (t-SNE on ResNet18 features)")
    plt.legend(unique_labels, title="Distortion", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ── Report serialisation ──────────────────────────────────────────────────────

def _build_report(
    archetypes: Dict[str, Dict[str, List[Tuple[str, float]]]],
) -> Dict:
    """Convert archetype tuples into a JSON-serialisable dict."""
    summary = {}
    for cat, ar in archetypes.items():
        summary[cat] = {
            "typical": [{"file": p, "distance": d} for p, d in ar.get("typical", [])],
            "outlier":  [{"file": p, "distance": d} for p, d in ar.get("outlier", [])],
            "note": (
                "Typical images are closest to the cluster centroid. "
                "Outliers deviate the most and may be mislabelled or show mixed distortions."
            ),
        }
    return {"archetypes": summary, "generated_at": datetime.datetime.now().isoformat()}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a distortion diagnostic report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", required=True, type=Path,
        help="Distortion subfolder root (blur/jpeg/pixelate/noise), or a misclassified JSON log",
    )
    parser.add_argument("--output",      required=True, type=Path, help="Output JSON report path")
    parser.add_argument("--plot",        required=True, type=Path, help="Output cluster plot (PNG)")
    parser.add_argument("--dry-run",     action="store_true",       help="Process whatever images are available")
    parser.add_argument("--max-samples", type=int, default=1000,    help="Max total images to process")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)

    global CATEGORIES
    # If a JSON log is supplied, decode its embedded images onto disk first
    if args.base_dir.is_file() and args.base_dir.suffix.lower() == ".json":
        with open(args.base_dir) as f:
            data = json.load(f)
        samples = data.get("misclassified_samples", [])
        cats = set()
        for s in samples:
            cats.add(_resolve_distortion(s))
        # Only keep known distortion categories; discard "unknown" as a category
        known_cats = cats & set(CATEGORIES)
        if known_cats:
            CATEGORIES = sorted(list(known_cats))
            
        extracted_dir = args.base_dir.parent / f"{args.base_dir.stem}_images"
        print(f"Extracting images from log: {args.base_dir} → {extracted_dir}")
        args.base_dir = _extract_images_from_log(args.base_dir, extracted_dir, CATEGORIES)
    elif args.base_dir.is_dir():
        CATEGORIES = sorted([d.name for d in args.base_dir.iterdir() if d.is_dir()])

    # ── Collect image paths ───────────────────────────────────────────────────
    images = _get_image_paths(args.base_dir, CATEGORIES)

    # Distribute max_samples evenly across categories so no single category
    # monopolises the budget when processed alphabetically.
    all_paths: List[Path] = []
    all_labels: List[str] = []
    per_cat_limit = (
        max(1, args.max_samples // max(1, len(CATEGORIES)))
        if args.max_samples else None
    )
    for cat in CATEGORIES:
        cat_paths = images.get(cat, [])
        if per_cat_limit is not None:
            cat_paths = cat_paths[:per_cat_limit]
        all_paths.extend(cat_paths)
        all_labels.extend([cat] * len(cat_paths))

    if not all_paths:
        raise SystemExit("No images found. Check that the directory structure and image files exist.")

    # ── Feature extraction ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  images: {len(all_paths)}")

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # strip classifier head
    backbone.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Extracting ResNet18 features…")
    features = _extract_features(all_paths, backbone, transform, device).reshape(len(all_paths), -1)

    # ── Dimensionality reduction: PCA → t-SNE ─────────────────────────────────
    print("Running PCA (50 components) + t-SNE…")
    pca_feats  = PCA(n_components=50).fit_transform(features)
    embeddings = TSNE(n_components=2, init="pca", random_state=42, learning_rate="auto").fit_transform(pca_feats)

    # ── Archetype selection + plot ────────────────────────────────────────────
    path_to_index = {p: i for i, p in enumerate(all_paths)}
    archetypes    = _select_archetypes(embeddings, all_labels, CATEGORIES, all_paths)
    _draw_cluster_plot(embeddings, all_labels, archetypes, path_to_index, args.plot)

    # ── Write JSON report ─────────────────────────────────────────────────────
    report = _build_report(archetypes)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"Report  → {args.output}")
    print(f"Cluster plot → {args.plot}")


if __name__ == "__main__":
    main()
