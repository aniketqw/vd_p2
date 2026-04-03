"""
AI Vision Reasoning Report Generator
=====================================
Reads distortion pipeline artifacts (training logs, misclassified JSON,
distortion cluster report) and uses the locally-hosted Qwen2.5-VL vision-language
model (via vLLM on port 8000) to visually inspect representative misclassified
images and produce a structured reasoning report in Markdown.

Usage (run from project root):
  python3 pipe/vision_reasoning_report.py

  python3 pipe/vision_reasoning_report.py \\
      --logs-dir     pipe/logs \\
      --report       pipe/reports/distortion_report.json \\
      --output       ai_reasoning_summary.md \\
      --samples      3

Optional overrides:
  --training-log   pipe/logs/training_log_20260317_012558.json
  --misclassified  pipe/logs/misclassified_20260317_005331.json
  --port           8000
  --model          Qwen/Qwen2.5-VL-7B-Instruct

The script auto-discovers the latest training_log_*.json and
misclassified_*.json inside --logs-dir when those flags are omitted.

vLLM must be running before executing this script:
  VLLM_USE_V1=0 env HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \\
  python3 -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-VL-7B-Instruct \\
    --quantization bitsandbytes \\
    --gpu-memory-utilization 0.4 \\
    --max-model-len 2048 \\
    --enforce-eager \\
    --port 8000
"""

import argparse
import base64
import datetime
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger_setup import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── constants ─────────────────────────────────────────────────────────────────
DISTORTION_TYPES = ["blur", "jpeg", "pixelate", "noise"]
CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image(image_path: Path) -> str:
    """Return base64-encoded string for an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_latest_file(directory: Path, glob_pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching glob_pattern, or None."""
    matches = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


# ── data loading ──────────────────────────────────────────────────────────────

def load_training_summary(log_path: Path) -> Dict:
    """Parse a training_log JSON and return a compact summary dict."""
    with open(log_path) as f:
        raw = json.load(f)
    summary = raw.get("summary", {})
    epochs = raw.get("epochs", [])
    return {
        "path": str(log_path),
        "best_accuracy": summary.get("best_accuracy"),
        "best_epoch": summary.get("best_epoch"),
        "total_epochs": summary.get("total_epochs"),
        "total_misclassified": summary.get("total_misclassified"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "epochs": [
            {
                "epoch": e.get("epoch"),
                "accuracy": e.get("accuracy") or 0,
                "overall_accuracy": e.get("overall_accuracy") or 0,
                "f1_score": e.get("f1_score") or 0,
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

    stats = {
        "path": str(mc_path),
        "n_misclassified": raw.get("n_misclassified", len(samples)),
        "by_distortion": {},
        "top_confusion_pairs": [],
    }

    for dt in DISTORTION_TYPES + ["unknown"]:
        group = by_dist.get(dt, [])
        if not group:
            continue
        confs = [s.get("distortion_confidence") or 0 for s in group]
        true_cnt = Counter(s.get("true_label") for s in group)
        pred_cnt = Counter(s.get("predicted_label") for s in group)
        epoch_cnt = Counter(s.get("epoch") for s in group)
        stats["by_distortion"][dt] = {
            "count": len(group),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0,
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
            "true": CIFAR10_CLASSES.get(t, str(t)),
            "pred": CIFAR10_CLASSES.get(p, str(p)),
            "count": cnt,
        }
        for (t, p), cnt in pair_cnt.most_common(20)
    ]

    # Build a hash → metadata lookup for image sampling
    stats["_hash_to_meta"] = {
        s.get("hash"): s for s in samples if s.get("hash")
    }
    return stats


def load_distortion_report(report_path: Path) -> Dict:
    """Parse distortion_report.json (archetype/outlier image paths)."""
    with open(report_path) as f:
        return json.load(f)


# ── image sampling ────────────────────────────────────────────────────────────

def resolve_image_path(raw_path: str) -> Optional[Path]:
    """
    Resolve a path that may be absolute /mnt/... or relative.
    Falls back to the symlinked home path if /mnt/... does not exist.
    """
    p = Path(raw_path)
    if p.exists():
        return p
    # try swapping /mnt/data/vision_dev_project → /home/pratik2/vision_dev_project
    try:
        rel = p.relative_to("/mnt/data/vision_dev_project")
        alt = Path("/home/pratik2/vision_dev_project") / rel
        if alt.exists():
            return alt
    except ValueError:
        pass
    return None


def gather_images_for_distortion(
    dist_type: str,
    report: Dict,
    mc_stats: Dict,
    images_dir: Optional[Path],
    n_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Return up to n_samples images (with metadata) for the given distortion type.
    Priority: typical archetype → outlier archetype → random from images_dir.
    Each item: {"path": Path, "role": str, "distance": float|None, "meta": dict|None}
    """
    rng = random.Random(seed)
    collected: List[Dict] = []
    seen_paths = set()

    archetypes = report.get("archetypes", {}).get(dist_type, {})
    hash_to_meta = mc_stats.get("_hash_to_meta", {})

    def _add(raw_path: str, role: str, distance: Optional[float]):
        if len(collected) >= n_samples:
            return
        p = resolve_image_path(raw_path)
        if p is None or str(p) in seen_paths:
            return
        seen_paths.add(str(p))
        stem = p.stem
        meta = hash_to_meta.get(stem)
        collected.append({"path": p, "role": role, "distance": distance, "meta": meta})

    for item in archetypes.get("typical", []):
        _add(item["file"], "typical", item.get("distance"))

    for item in archetypes.get("outlier", []):
        _add(item["file"], "outlier", item.get("distance"))

    # fill remaining slots with random samples from the images folder
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
                stem = p.stem
                meta = hash_to_meta.get(stem)
                collected.append({"path": p, "role": "random", "distance": None, "meta": meta})

    return collected


# ── VLM server readiness ──────────────────────────────────────────────────────

def wait_for_server(port: int, timeout: int = 300, poll_interval: int = 5) -> bool:
    """
    Poll http://localhost:{port}/health until it returns 200 or timeout expires.
    Prints a live countdown so the user knows the server is still loading.
    Returns True if server became ready, False if timed out.
    """
    import urllib.request
    import urllib.error
    import time

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    attempt = 0

    print(f"\n⏳ Waiting for vLLM server on port {port} (timeout={timeout}s)…")
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"✅ Server ready (after ~{attempt * poll_interval}s)\n")
                    logger.info("vLLM server is ready.")
                    return True
        except Exception:
            remaining = int(deadline - time.time())
            print(f"   not ready yet — retrying in {poll_interval}s "
                  f"(~{remaining}s remaining) …", end="\r", flush=True)
            time.sleep(poll_interval)

    print(f"\n❌ Server did not become ready within {timeout}s.")
    logger.error("vLLM server readiness timeout.")
    return False


# ── VLM analysis ──────────────────────────────────────────────────────────────

_VLM_PROMPT_TEXT = """\
You are an expert computer vision failure analyst.

You are given a 32×32 CIFAR-10 image that was MISCLASSIFIED by a neural network.
Study the image very carefully before answering — every pixel detail matters at this resolution.

━━━ IMAGE METADATA ━━━
Distortion applied        : {distortion_type}
True class (ground truth) : {true_label}
Predicted class (wrong)   : {pred_label}
Distortion confidence     : {dist_confidence}
Sample role               : {role}
  (typical = near cluster centroid — representative failure)
  (outlier  = far from centroid   — unusual/edge-case failure)

━━━ YOUR TASK ━━━
Respond using EXACTLY these seven labeled sections. Each section must start on its own line \
with the label shown below followed by a colon. Be specific — reference actual colors, \
shapes, regions, and pixel-level observations you see in the image.

DISTORTION ARTIFACTS:
Describe the specific visual artifacts the {distortion_type} distortion has introduced \
(e.g. blur halos around edges, 8×8 JPEG block grid, mosaic rectangles, noise speckle pattern). \
Point to exact regions (top-left, fur area, background, etc.).

SURVIVING TRUE-CLASS FEATURES:
What features in the image still correctly suggest this is a {true_label}? \
(color blobs, silhouette shape, structural layout that survived the distortion)

WHAT MISLED THE MODEL:
Which specific distortion artifact or degraded region makes this look like a {pred_label}? \
Be precise — name the visual pattern and where it appears.

MODEL REASONING CORRECT:
What did the model partially get right when looking at this image? \
(any features it correctly associated with the scene content)

MODEL REASONING INCORRECT:
What specific mistake did the model make? Which artifact or pattern triggered the wrong \
class activation, and why does it visually resemble {pred_label}?

CONFIDENCE ASSESSMENT:
The distortion detection confidence was {dist_confidence}. Based on what you see, \
explain why the confidence is this value — does the distortion look obvious or subtle? \
Is the image ambiguous even to a human eye?

ROOT CAUSE:
One sentence only: the single core reason why {distortion_type} distortion \
on a {true_label} image specifically triggers a {pred_label} prediction.\
"""


def build_vlm_chain(model_name: str, port: int) -> object:
    """Build and return the LangChain vision chain."""
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key="local-3090",
        openai_api_base=f"http://localhost:{port}/v1",
        temperature=0.1,
        max_tokens=900,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {
                "type": "text",
                "text": _VLM_PROMPT_TEXT,
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,{image_data}"},
            },
        ])
    ])
    return prompt | llm | StrOutputParser()


def parse_vlm_sections(raw: str) -> Dict[str, str]:
    """
    Parse the VLM response into a dict keyed by section label.
    Expects lines starting with 'LABEL:' as section boundaries.
    Falls back to storing everything under '_raw' if no sections found.
    """
    section_keys = [
        "DISTORTION ARTIFACTS",
        "SURVIVING TRUE-CLASS FEATURES",
        "WHAT MISLED THE MODEL",
        "MODEL REASONING CORRECT",
        "MODEL REASONING INCORRECT",
        "CONFIDENCE ASSESSMENT",
        "ROOT CAUSE",
    ]
    result: Dict[str, str] = {}
    current_key = None
    current_lines: List[str] = []

    for line in raw.splitlines():
        matched = False
        for key in section_keys:
            if line.upper().startswith(key + ":"):
                if current_key:
                    result[current_key] = " ".join(current_lines).strip()
                current_key = key
                # grab any text after the colon on the same line
                after = line[len(key) + 1:].strip()
                current_lines = [after] if after else []
                matched = True
                break
        if not matched and current_key:
            current_lines.append(line.strip())

    if current_key:
        result[current_key] = " ".join(current_lines).strip()

    if not result:
        result["_raw"] = raw.strip()

    return result


def analyze_image(chain, item: Dict) -> Optional[Dict]:
    """
    Call the VLM chain for one image item.
    Returns a dict of parsed sections (or {'_raw': ...}) or None on failure.
    """
    p: Path = item["path"]
    meta = item.get("meta") or {}
    dist_type = item.get("dist_type", "unknown")

    true_lbl = CIFAR10_CLASSES.get(meta.get("true_label"), str(meta.get("true_label", "?")))
    pred_lbl = CIFAR10_CLASSES.get(meta.get("predicted_label"), str(meta.get("predicted_label", "?")))
    role = item.get("role", "random")
    dist_conf = meta.get("distortion_confidence") or 0.0

    try:
        img_b64 = encode_image(p)
        logger.info(f"  → VLM call: {p.name}  true={true_lbl}  pred={pred_lbl}  conf={dist_conf:.4f}")
        raw_response = chain.invoke({
            "distortion_type": dist_type,
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "role": role,
            "dist_confidence": f"{dist_conf:.4f}",
            "image_data": img_b64,
        })
        sections = parse_vlm_sections(raw_response.strip())
        logger.info(f"     sections found: {list(sections.keys())}")
        return sections
    except Exception as exc:
        logger.warning(f"VLM call failed for {p}: {exc}")
        return None


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_report(
    training: Dict,
    mc_stats: Dict,
    per_type_analyses: Dict[str, List[Dict]],
    output_path: Path,
):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_mc = mc_stats["n_misclassified"]
    best_acc = training.get("best_accuracy", 0)

    lines = [
        f"# AI Vision Reasoning Report",
        f"**Generated:** {now}  ",
        f"**Training log:** `{Path(training['path']).name}`  ",
        f"**Misclassified log:** `{Path(mc_stats['path']).name}`  ",
        f"**Model:** CIFAR-10 · 32×32 px · {training.get('total_epochs', '?')} epochs",
        "",
        "---",
        "",
        "## 1. Training Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best Accuracy | **{best_acc:.2%}** |",
        f"| Best Epoch | {training.get('best_epoch', '?')} |",
        f"| Final Train Loss | {training.get('final_train_loss', 0):.4f} |",
        f"| Final Val Loss | {training.get('final_val_loss', 0):.4f} |",
        f"| Total Misclassified | {training.get('total_misclassified', '?')} |",
        "",
        "**Epoch-level progression:**",
        "",
        "| Epoch | Accuracy | Overall Acc | F1 | Misclassified |",
        "|-------|----------|-------------|-----|---------------|",
    ]
    for ep in training.get("epochs", []):
        lines.append(
            f"| {ep['epoch']} | {ep['accuracy']:.4f} | {ep['overall_accuracy']:.4f} "
            f"| {ep['f1_score']:.4f} | {ep['num_misclassified']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Failure Distribution by Distortion Type",
        "",
        "| Distortion | Failures | % of Total | Avg Distortion Confidence |",
        "|------------|---------|------------|--------------------------|",
    ]
    for dt in DISTORTION_TYPES + ["unknown"]:
        d = mc_stats["by_distortion"].get(dt)
        if not d:
            continue
        pct = d["count"] / total_mc * 100 if total_mc else 0
        lines.append(
            f"| **{dt}** | {d['count']} | {pct:.1f}% | {d['avg_confidence']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Top Confusion Pairs (all distortions)",
        "",
        "| True Class | Predicted Class | Count |",
        "|-----------|----------------|-------|",
    ]
    for pair in mc_stats["top_confusion_pairs"][:15]:
        lines.append(f"| {pair['true']} | {pair['pred']} | {pair['count']} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Per-Distortion Visual Analysis (Qwen2.5-VL)",
        "",
        "> Each section below shows the model's blind spot analysis backed by",
        "> direct visual inspection of representative misclassified images.",
        "",
    ]

    # emoji + label pairs for each parsed section
    _SECTION_DISPLAY = [
        ("DISTORTION ARTIFACTS",          "🔬 Distortion Artifacts Observed"),
        ("SURVIVING TRUE-CLASS FEATURES",  "✅ Surviving True-Class Features"),
        ("WHAT MISLED THE MODEL",          "❌ What Misled the Model"),
        ("MODEL REASONING CORRECT",        "🧠 Model Reasoning — Correct Part"),
        ("MODEL REASONING INCORRECT",      "🚫 Model Reasoning — Incorrect Part"),
        ("CONFIDENCE ASSESSMENT",          "📊 Confidence Assessment"),
        ("ROOT CAUSE",                     "🎯 Root Cause"),
    ]

    blind_spot_summaries = []

    for dt in DISTORTION_TYPES:
        d = mc_stats["by_distortion"].get(dt)
        analyses = per_type_analyses.get(dt, [])
        pct = (d["count"] / total_mc * 100) if (d and total_mc) else 0

        lines += [
            f"### 4.{DISTORTION_TYPES.index(dt)+1} {dt.capitalize()} "
            f"— {d['count'] if d else 0} failures ({pct:.1f}%)",
            "",
        ]

        if d:
            true_labels = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in d["top_true_labels"])
            pred_labels = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in d["top_pred_labels"])
            ep_dist = " · ".join(f"ep{k}:{v}" for k, v in d["epoch_distribution"].items())
            lines += [
                f"**Most misclassified true classes:** {true_labels}  ",
                f"**Most common wrong predictions:** {pred_labels}  ",
                f"**Avg distortion confidence:** {d['avg_confidence']:.4f}  ",
                f"**Epoch distribution:** {ep_dist}",
                "",
            ]

        if not analyses:
            lines.append("_No images were analysed for this distortion type._\n")
            blind_spot_summaries.append((dt, "_No visual analysis available._"))
            continue

        root_causes = []
        any_vlm = False

        for img_idx, item in enumerate(analyses, 1):
            sections = item.get("vlm_response")   # now a dict after analyze_image change
            meta = item.get("meta") or {}
            true_lbl = CIFAR10_CLASSES.get(meta.get("true_label"), "?")
            pred_lbl = CIFAR10_CLASSES.get(meta.get("predicted_label"), "?")
            role = item.get("role", "random")
            dist_conf = meta.get("distortion_confidence") or 0.0
            cluster_dist = item.get("distance")
            fname = Path(item["path"]).name

            # ── image header card ──
            role_icon = {"typical": "⭕", "outlier": "⚠️"}.get(role, "🔹")
            cluster_str = f" · cluster dist={cluster_dist:.3f}" if cluster_dist is not None else ""
            lines += [
                f"#### Image {img_idx} — `{fname}`",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| Role | {role_icon} **{role}**{cluster_str} |",
                f"| True class | **{true_lbl}** |",
                f"| Predicted (wrong) | **{pred_lbl}** |",
                f"| Distortion confidence | `{dist_conf:.4f}` |",
                "",
            ]

            if not sections:
                lines.append("_VLM call failed for this image — is the server running?_\n")
                continue

            any_vlm = True

            # ── fallback: raw response ──
            if "_raw" in sections:
                lines += [
                    "> _(response not structured — displaying raw output)_",
                    "",
                    f"> {sections['_raw']}",
                    "",
                ]
                root_causes.append(sections["_raw"][:200])
                continue

            # ── structured sections ──
            for key, label in _SECTION_DISPLAY:
                text = sections.get(key, "").strip()
                if not text:
                    continue
                lines += [
                    f"**{label}**",
                    "",
                    f"> {text}",
                    "",
                ]

            rc = sections.get("ROOT CAUSE", "").strip()
            if rc:
                root_causes.append(rc)

        # ── blind spot summary for this distortion type ──
        if root_causes:
            blind_spot_summaries.append((dt, " | ".join(root_causes)))
        elif any_vlm:
            blind_spot_summaries.append((dt, "_Root cause not extracted from VLM response._"))
        else:
            blind_spot_summaries.append((dt, "_VLM calls failed — check server._"))

    lines += [
        "---",
        "",
        "## 5. Model Blind Spot Summary",
        "",
        "| Distortion | Visual Blind Spot (synthesised from VLM analysis) |",
        "|------------|--------------------------------------------------|",
    ]
    for dt, summary in blind_spot_summaries:
        safe = summary.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| **{dt}** | {safe} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Actionable Recommendations",
        "",
        "Based on the failure distribution and VLM visual analysis above:",
        "",
        "| Priority | Recommendation | Addresses |",
        "|----------|---------------|-----------|",
        "| 🔴 High | **Multi-distortion augmentation** — add blur (σ=0.5–3), JPEG sim (q=10–70), pixelation (block 2–8px), Gaussian noise (σ=0.05–0.25) to training dataloader | Blur (40%), JPEG (30%), Pixelate (15%) |",
        "| 🔴 High | **Dual-pathway backbone** — add a low-pass feature stream (shape/silhouette) alongside the standard high-freq stream; merge with channel attention (SE-Net/CBAM) | Blur, Pixelate |",
        "| 🟡 Medium | **Gradient-reversal distortion-adversarial head** — auxiliary head predicts distortion type with GRL; forces backbone to learn distortion-invariant features | All types |",
        "| 🟡 Medium | **Fine-grained contrastive loss** — add SupCon or NT-Xent loss for the cat/dog/deer/bird confusion cluster; enforce larger embedding margins between visually similar classes | JPEG, Blur |",
        "| 🟢 Low | **Train longer + cosine LR decay** — current 3-epoch run shows monotonically falling loss; model has not converged. Val loss > train loss indicates early overfitting | All types |",
        "",
        "---",
        "",
        f"*Report generated by `pipe/vision_reasoning_report.py` · {now}*",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written → {output_path}")
    print(f"\n✅  Report written to: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    # Paths default relative to this script's location (pipe/) so the script
    # works whether invoked from the project root or from inside pipe/.
    _script_dir = Path(__file__).resolve().parent   # …/pipe
    _project_root = _script_dir.parent              # …/vision_dev_project

    p = argparse.ArgumentParser(
        description="Generate an AI vision reasoning report using Qwen2.5-VL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # run from inside pipe/  (no arguments needed)\n"
            "  python3 vision_reasoning_report.py\n\n"
            "  # run from project root\n"
            "  python3 pipe/vision_reasoning_report.py\n\n"
            "  # stats only (no vLLM server required)\n"
            "  python3 vision_reasoning_report.py --no-vlm\n"
        ),
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=_script_dir / "logs",
        help="Directory containing training_log_*.json and misclassified_*.json "
             f"(default: {_script_dir / 'logs'})",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_script_dir / "reports" / "distortion_report.json",
        help="Path to distortion_report.json "
             f"(default: {_script_dir / 'reports' / 'distortion_report.json'})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_project_root / "ai_reasoning_summary.md",
        help="Output Markdown report path "
             f"(default: {_project_root / 'ai_reasoning_summary.md'})",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of images per distortion type to send to the VLM",
    )
    p.add_argument(
        "--training-log",
        type=Path,
        default=None,
        help="Override: explicit path to a training_log JSON",
    )
    p.add_argument(
        "--misclassified",
        type=Path,
        default=None,
        help="Override: explicit path to a misclassified JSON",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="vLLM model name",
    )
    p.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM calls (stats-only report, useful for testing without GPU)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logs_dir = args.logs_dir.resolve()

    # ── 1. Discover / validate input files ────────────────────────────────────
    training_log_path = args.training_log or find_latest_file(logs_dir, "training_log_*.json")
    if training_log_path is None or not training_log_path.exists():
        sys.exit(f"ERROR: No training_log_*.json found in {logs_dir}. Use --training-log.")
    logger.info(f"Training log  : {training_log_path}")

    mc_path = args.misclassified or find_latest_file(logs_dir, "misclassified_*.json")
    if mc_path is None or not mc_path.exists():
        sys.exit(f"ERROR: No misclassified_*.json found in {logs_dir}. Use --misclassified.")
    logger.info(f"Misclassified : {mc_path}")

    report_path = args.report.resolve()
    if not report_path.exists():
        sys.exit(f"ERROR: distortion_report.json not found at {report_path}. Use --report.")
    logger.info(f"Distortion report: {report_path}")

    # Derive the images folder (e.g. misclassified_20260317_005331_images/)
    images_dir = mc_path.parent / f"{mc_path.stem}_images"
    if not images_dir.exists():
        logger.warning(f"Images folder not found: {images_dir} — random sampling disabled.")
        images_dir = None

    # ── 2. Load data ───────────────────────────────────────────────────────────
    logger.info("Loading training summary…")
    training = load_training_summary(training_log_path)

    logger.info("Loading misclassified stats…")
    mc_stats = load_misclassified_stats(mc_path)

    logger.info("Loading distortion report…")
    dist_report = load_distortion_report(report_path)

    # ── 3. Build VLM chain (unless --no-vlm) ──────────────────────────────────
    chain = None
    if not args.no_vlm:
        ready = wait_for_server(args.port, timeout=300, poll_interval=5)
        if not ready:
            print("⚠️  Continuing without VLM — all image analyses will be skipped.")
            print("   Start the vLLM server first and re-run, or use --no-vlm for stats only.\n")
        else:
            logger.info(f"Building VLM chain → http://localhost:{args.port}/v1  model={args.model}")
            chain = build_vlm_chain(args.model, args.port)

    # ── 4. Per-distortion image analysis ──────────────────────────────────────
    per_type_analyses: Dict[str, List[Dict]] = {}

    for dt in DISTORTION_TYPES:
        logger.info(f"\n{'─'*60}\nDistortion: {dt.upper()}")
        items = gather_images_for_distortion(
            dist_type=dt,
            report=dist_report,
            mc_stats=mc_stats,
            images_dir=images_dir,
            n_samples=args.samples,
            seed=args.seed,
        )
        logger.info(f"  Sampled {len(items)} image(s) for VLM analysis.")

        for item in items:
            item["dist_type"] = dt
            if chain is not None:
                resp = analyze_image(chain, item)
                item["vlm_response"] = resp
            else:
                item["vlm_response"] = None

        per_type_analyses[dt] = items

    # ── 5. Render report ───────────────────────────────────────────────────────
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_report(training, mc_stats, per_type_analyses, output_path)


if __name__ == "__main__":
    main()
