"""
AI Vision Reasoning Report Generator  —  v2
============================================
Upgraded from v1 with three key changes:

  1. MULTI-IMAGE BATCHING
     All sampled images for one distortion type are sent in a single VLM call
     (Qwen2.5-VL natively supports multiple images per message).
     Reduces VLM calls from N×4 → 2×4 while enabling cross-image comparison.

  2. TWO-TURN RECURSIVE REASONING
     Turn 1 — Visual grounding  (images shown, labels withheld):
       The model describes distortion artifacts and surviving structure purely from
       pixel observation, before any class metadata can bias it.
     Turn 2 — Causal reasoning  (Turn 1 output + labels injected):
       Given its own prior observations and now knowing the true/predicted labels,
       the model synthesises WHAT specifically caused the wrong prediction and WHY
       the typical vs outlier cases may differ.

  3. CROSS-IMAGE SYNTHESIS
     Turn 2 produces a single synthesised analysis covering all images together
     (shared failure pattern, typical-vs-outlier comparison, one root cause),
     replacing the v1 per-image string-join approach.

Usage (identical to v1):
  python3 pipe/vision_reasoning_report_v2.py
  python3 pipe/vision_reasoning_report_v2.py --no-vlm
  python3 pipe/vision_reasoning_report_v2.py --samples 3 --seed 42 --port 8000
  python3 pipe/vision_reasoning_report_v2.py \\
      --logs-dir  pipe/logs \\
      --report    pipe/reports/distortion_report.json \\
      --output    ai_reasoning_summary_v2.md \\
      --samples   3
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
from typing import Dict, List, Optional

# ── make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger_setup import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# ── constants ─────────────────────────────────────────────────────────────────
DISTORTION_TYPES = ["blur", "jpeg", "pixelate", "noise"]
CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer",     5: "dog",        6: "frog", 7: "horse",
    8: "ship",     9: "truck",
}

# ── Turn 1 prompt — visual grounding, NO labels ───────────────────────────────
_TURN1_PROMPT = """\
You are an expert computer vision analyst.

You are given {n_images} CIFAR-10 image(s), each 32×32 pixels, all affected by \
{distortion_type} distortion. Study every image carefully — at this resolution \
every pixel cluster matters.

IMAGE ROLES:
{image_roles}
  (typical = near cluster centroid — representative sample of this failure type)
  (outlier  = far from centroid   — unusual or edge-case sample)

━━━ YOUR TASK — visual observation only, NO class guessing ━━━
Do NOT name or guess object classes. Purely describe what you see.

For EACH image (label them Image 1, Image 2 …):

  ARTIFACTS: Describe the exact {distortion_type} degradation — name the specific \
region (top-left corner, center, background, fur area, edges) and the type of \
artifact (e.g. blur halos, 8×8 block grid, mosaic rectangles, speckle pattern).

  SURVIVING STRUCTURE: What visual information survived — color blobs, silhouette \
outlines, spatial layout, dominant hue?

  MOST DEGRADED: Which region or feature has been most destroyed by the distortion?

Finally, answer this one question:

  COMMON PATTERN: What single visual element or degradation characteristic is \
present across ALL {n_images} image(s)?\
"""

# ── Turn 2 prompt — causal reasoning, labels revealed ────────────────────────
_TURN2_PROMPT = """\
You are an expert computer vision failure analyst.

━━━ STEP 1 — YOUR PRIOR VISUAL OBSERVATIONS ━━━
{turn1_output}

━━━ GROUND TRUTH LABELS (now revealed) ━━━
{image_metadata}

These {n_images} CIFAR-10 image(s) were MISCLASSIFIED by a neural network under \
{distortion_type} distortion.

Using your Step 1 visual observations as evidence, respond with EXACTLY these \
five labeled sections. Each section must begin on its own line with the label \
shown below followed by a colon.

SHARED FAILURE PATTERN:
Which specific artifact or degraded region — that you observed in Step 1 — is \
present in ALL images and most directly explains the wrong prediction? Reference \
your exact Step 1 observations (region names, texture descriptions you used).

TYPICAL VS OUTLIER:
Compare the typical image(s) against the outlier. Do they fail for the same \
reason, or does the outlier represent a different failure mechanism? \
What is structurally different about how the distortion damaged the outlier?

WHAT MISLED THE MODEL:
Name the exact visual pattern that activated the wrong class. Why does this \
specific degraded region visually resemble the predicted (wrong) class rather \
than the true class?

CONFIDENCE ASSESSMENT:
Given your visual observations, do these {distortion_type} distortions look \
obvious or subtle at 32×32 resolution? Is the distortion detectable even to a \
human? Explain why the distortion confidence values are at the levels shown.

ROOT CAUSE:
One sentence only — the single core mechanism by which {distortion_type} \
distortion on these images triggers the specific wrong-class prediction shown.\
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image(image_path: Path) -> str:
    """Return base64-encoded PNG string for an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_latest_file(directory: Path, glob_pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching glob_pattern, or None."""
    matches = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


# ── data loading (unchanged from v1) ─────────────────────────────────────────

def load_training_summary(log_path: Path) -> Dict:
    with open(log_path) as f:
        raw = json.load(f)
    summary = raw.get("summary", {})
    epochs  = raw.get("epochs", [])
    return {
        "path": str(log_path),
        "best_accuracy":        summary.get("best_accuracy"),
        "best_epoch":           summary.get("best_epoch"),
        "total_epochs":         summary.get("total_epochs"),
        "total_misclassified":  summary.get("total_misclassified"),
        "final_train_loss":     summary.get("final_train_loss"),
        "final_val_loss":       summary.get("final_val_loss"),
        "epochs": [
            {
                "epoch":            e.get("epoch"),
                "accuracy":         e.get("accuracy") or 0,
                "overall_accuracy": e.get("overall_accuracy") or 0,
                "f1_score":         e.get("f1_score") or 0,
                "num_misclassified":e.get("num_misclassified") or 0,
            }
            for e in epochs
        ],
        "dataset_classes": summary.get("dataset_info", {}).get("classes", CIFAR10_CLASSES),
    }


def load_misclassified_stats(mc_path: Path) -> Dict:
    with open(mc_path) as f:
        raw = json.load(f)

    samples = raw.get("misclassified_samples", [])
    by_dist: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        dt = str(s.get("distortion_predicted") or "unknown")
        by_dist[dt].append(s)

    stats = {
        "path":              str(mc_path),
        "n_misclassified":   raw.get("n_misclassified", len(samples)),
        "by_distortion":     {},
        "top_confusion_pairs": [],
    }

    for dt in DISTORTION_TYPES + ["unknown"]:
        group = by_dist.get(dt, [])
        if not group:
            continue
        confs    = [s.get("distortion_confidence") or 0 for s in group]
        true_cnt = Counter(s.get("true_label") for s in group)
        pred_cnt = Counter(s.get("predicted_label") for s in group)
        epoch_cnt= Counter(s.get("epoch") for s in group)
        stats["by_distortion"][dt] = {
            "count":            len(group),
            "avg_confidence":   round(sum(confs) / len(confs), 4) if confs else 0,
            "top_true_labels":  [
                (CIFAR10_CLASSES.get(lbl, str(lbl)), cnt)
                for lbl, cnt in true_cnt.most_common(5)
            ],
            "top_pred_labels":  [
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
    return stats


def load_distortion_report(report_path: Path) -> Dict:
    with open(report_path) as f:
        return json.load(f)


# ── image sampling (unchanged from v1) ───────────────────────────────────────

def resolve_image_path(raw_path: str) -> Optional[Path]:
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

    archetypes  = report.get("archetypes", {}).get(dist_type, {})
    hash_to_meta = mc_stats.get("_hash_to_meta", {})

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


# ── VLM server readiness (unchanged from v1) ─────────────────────────────────

def wait_for_server(port: int, timeout: int = 300, poll_interval: int = 5) -> bool:
    import urllib.request, urllib.error, time
    url      = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    attempt  = 0
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
            print(
                f"   not ready yet — retrying in {poll_interval}s "
                f"(~{remaining}s remaining) …", end="\r", flush=True,
            )
            time.sleep(poll_interval)
    print(f"\n❌ Server did not become ready within {timeout}s.")
    logger.error("vLLM server readiness timeout.")
    return False


# ── LLM construction ──────────────────────────────────────────────────────────

def build_llm(model_name: str, port: int) -> ChatOpenAI:
    """
    Return a ChatOpenAI pointed at the local vLLM server.
    max_tokens is raised to 1500 (vs 900 in v1) because Turn 2 must cover
    multiple images plus five structured sections.
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key="local-3090",
        openai_api_base=f"http://localhost:{port}/v1",
        temperature=0.1,
        max_tokens=1500,
    )


# ── message builders ──────────────────────────────────────────────────────────

def _build_image_roles_str(items: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        role = item.get("role", "random")
        dist = item.get("distance")
        suffix = f" (cluster dist = {dist:.3f})" if dist is not None else ""
        lines.append(f"  Image {i}: {role}{suffix}")
    return "\n".join(lines)


def _build_image_metadata_str(items: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        meta      = item.get("meta") or {}
        true_lbl  = CIFAR10_CLASSES.get(meta.get("true_label"), str(meta.get("true_label", "?")))
        pred_lbl  = CIFAR10_CLASSES.get(meta.get("predicted_label"), str(meta.get("predicted_label", "?")))
        conf      = meta.get("distortion_confidence") or 0.0
        role      = item.get("role", "random")
        lines.append(
            f"  Image {i} ({role}):  "
            f"true class = {true_lbl}  |  "
            f"predicted (wrong) = {pred_lbl}  |  "
            f"distortion confidence = {conf:.4f}"
        )
    return "\n".join(lines)


def build_turn1_message(items: List[Dict], dist_type: str) -> List[HumanMessage]:
    """
    Construct the Turn 1 HumanMessage:
      text prompt (no labels) + all images as base64 image_url blocks.
    """
    text = _TURN1_PROMPT.format(
        n_images      = len(items),
        distortion_type = dist_type,
        image_roles   = _build_image_roles_str(items),
    )
    content = [{"type": "text", "text": text}]
    for item in items:
        b64 = encode_image(item["path"])
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [HumanMessage(content=content)]


def build_turn2_message(
    items: List[Dict],
    dist_type: str,
    turn1_output: str,
) -> List[HumanMessage]:
    """
    Construct the Turn 2 HumanMessage:
      text prompt (Turn 1 output + labels) + same images resent for reference.
    Images are resent because each .invoke() call is stateless.
    """
    text = _TURN2_PROMPT.format(
        turn1_output    = turn1_output.strip(),
        image_metadata  = _build_image_metadata_str(items),
        n_images        = len(items),
        distortion_type = dist_type,
    )
    content = [{"type": "text", "text": text}]
    for item in items:
        b64 = encode_image(item["path"])
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return [HumanMessage(content=content)]


# ── section parser for Turn 2 ─────────────────────────────────────────────────

_TURN2_SECTION_KEYS = [
    "SHARED FAILURE PATTERN",
    "TYPICAL VS OUTLIER",
    "WHAT MISLED THE MODEL",
    "CONFIDENCE ASSESSMENT",
    "ROOT CAUSE",
]


def parse_turn2_sections(raw: str) -> Dict[str, str]:
    """
    Parse the Turn 2 VLM response into a dict keyed by section label.
    Falls back to {'_raw': ...} if no recognised section headers are found.
    """
    result: Dict[str, str] = {}
    current_key   = None
    current_lines: List[str] = []

    for line in raw.splitlines():
        matched = False
        for key in _TURN2_SECTION_KEYS:
            if line.upper().startswith(key + ":"):
                if current_key:
                    result[current_key] = " ".join(current_lines).strip()
                current_key   = key
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


# ── two-turn batch analysis ───────────────────────────────────────────────────

def analyze_distortion_batch(
    llm: ChatOpenAI,
    items: List[Dict],
    dist_type: str,
) -> Dict:
    """
    Run the two-turn recursive analysis for all images of one distortion type.

    Returns:
      {
        "turn1_output":   str  — raw Turn 1 visual observations
        "turn2_sections": dict — parsed Turn 2 structured sections
        "items":          list — the original image items (with meta)
        "error":          str | None
      }
    """
    result = {
        "turn1_output":   None,
        "turn2_sections": None,
        "items":          items,
        "error":          None,
    }

    if not items:
        result["error"] = "No images provided."
        return result

    parser = StrOutputParser()

    # ── Turn 1: visual grounding ──────────────────────────────────────────────
    try:
        logger.info(
            f"  [Turn 1] Sending {len(items)} image(s) for {dist_type} "
            f"— visual grounding (no labels)"
        )
        turn1_msgs   = build_turn1_message(items, dist_type)
        turn1_output = (llm | parser).invoke(turn1_msgs)
        result["turn1_output"] = turn1_output.strip()
        logger.info(f"  [Turn 1] Response received ({len(turn1_output)} chars)")
    except Exception as exc:
        logger.warning(f"  [Turn 1] FAILED for {dist_type}: {exc}")
        result["error"] = f"Turn 1 failed: {exc}"
        return result

    # ── Turn 2: causal reasoning ──────────────────────────────────────────────
    try:
        logger.info(
            f"  [Turn 2] Sending {len(items)} image(s) + Turn 1 output "
            f"— causal reasoning (labels revealed)"
        )
        turn2_msgs    = build_turn2_message(items, dist_type, result["turn1_output"])
        turn2_output  = (llm | parser).invoke(turn2_msgs)
        result["turn2_sections"] = parse_turn2_sections(turn2_output.strip())
        found_keys = list(result["turn2_sections"].keys())
        logger.info(f"  [Turn 2] Sections parsed: {found_keys}")
    except Exception as exc:
        logger.warning(f"  [Turn 2] FAILED for {dist_type}: {exc}")
        result["error"] = f"Turn 2 failed: {exc}"

    return result


# ── Markdown rendering ────────────────────────────────────────────────────────

# Display labels for each Turn 2 section
_SECTION_DISPLAY = [
    ("SHARED FAILURE PATTERN", "🔗 Shared Failure Pattern"),
    ("TYPICAL VS OUTLIER",     "⭕↔⚠️  Typical vs Outlier"),
    ("WHAT MISLED THE MODEL",  "❌ What Misled the Model"),
    ("CONFIDENCE ASSESSMENT",  "📊 Confidence Assessment"),
    ("ROOT CAUSE",             "🎯 Root Cause"),
]


def render_report(
    training: Dict,
    mc_stats: Dict,
    per_type_results: Dict[str, Dict],
    output_path: Path,
):
    now       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_mc  = mc_stats["n_misclassified"]
    best_acc  = training.get("best_accuracy", 0)

    lines = [
        "# AI Vision Reasoning Report  —  v2",
        f"**Generated:** {now}  ",
        f"**Strategy:** Multi-image batching + Two-turn recursive reasoning  ",
        f"**Training log:** `{Path(training['path']).name}`  ",
        f"**Misclassified log:** `{Path(mc_stats['path']).name}`  ",
        f"**Model:** CIFAR-10 · 32×32 px · {training.get('total_epochs', '?')} epochs",
        "",
        "---",
        "",
        "## 1. Training Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
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
        "## 4. Per-Distortion Synthesised Analysis  (Qwen2.5-VL · Two-Turn)",
        "",
        "> **Methodology:** Each section below reflects a *two-turn* VLM analysis.",
        "> Turn 1 grounded the model visually (no labels). Turn 2 injected labels",
        "> and asked for cross-image synthesis — shared failure pattern, typical-vs-outlier",
        "> comparison, and a single synthesised root cause.",
        "",
    ]

    blind_spot_summaries = []

    for dt in DISTORTION_TYPES:
        d       = mc_stats["by_distortion"].get(dt)
        result  = per_type_results.get(dt, {})
        pct     = (d["count"] / total_mc * 100) if (d and total_mc) else 0
        items   = result.get("items", [])

        lines += [
            f"### 4.{DISTORTION_TYPES.index(dt)+1}  {dt.capitalize()} "
            f"— {d['count'] if d else 0} failures ({pct:.1f}%)",
            "",
        ]

        if d:
            true_labels = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in d["top_true_labels"])
            pred_labels = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in d["top_pred_labels"])
            ep_dist     = " · ".join(f"ep{k}:{v}" for k, v in d["epoch_distribution"].items())
            lines += [
                f"**Most misclassified true classes:** {true_labels}  ",
                f"**Most common wrong predictions:** {pred_labels}  ",
                f"**Avg distortion confidence:** {d['avg_confidence']:.4f}  ",
                f"**Epoch distribution:** {ep_dist}",
                "",
            ]

        # ── images table ──────────────────────────────────────────────────────
        if items:
            lines += [
                "**Images analysed:**",
                "",
                "| # | File | Role | True | Predicted | Dist. Conf. |",
                "|---|------|------|------|-----------|-------------|",
            ]
            for i, item in enumerate(items, 1):
                meta      = item.get("meta") or {}
                true_lbl  = CIFAR10_CLASSES.get(meta.get("true_label"), "?")
                pred_lbl  = CIFAR10_CLASSES.get(meta.get("predicted_label"), "?")
                conf      = meta.get("distortion_confidence") or 0.0
                role      = item.get("role", "random")
                dist_str  = item.get("distance")
                role_icon = {"typical": "⭕", "outlier": "⚠️"}.get(role, "🔹")
                dist_cell = f"{dist_str:.3f}" if dist_str is not None else "—"
                fname     = Path(item["path"]).name
                lines.append(
                    f"| {i} | `{fname}` | {role_icon} {role} · d={dist_cell} "
                    f"| {true_lbl} | {pred_lbl} | `{conf:.4f}` |"
                )
            lines.append("")

        # ── error fallback ────────────────────────────────────────────────────
        err = result.get("error")
        if err:
            lines.append(f"_VLM analysis failed: {err}_\n")
            blind_spot_summaries.append((dt, f"_Analysis failed: {err}_"))
            continue

        # ── Turn 1 observations ───────────────────────────────────────────────
        turn1 = result.get("turn1_output")
        if not turn1:
            lines.append("_No Turn 1 output available (--no-vlm mode or call failed)._\n")
            blind_spot_summaries.append((dt, "_No VLM analysis available._"))
            continue

        lines += [
            "#### Turn 1 — Visual Observations (labels withheld)",
            "",
            "> " + turn1.replace("\n", "\n> "),
            "",
        ]

        # ── Turn 2 synthesised sections ───────────────────────────────────────
        sections = result.get("turn2_sections")
        if not sections:
            lines.append("_Turn 2 call failed — Turn 1 observations above may still be useful._\n")
            blind_spot_summaries.append((dt, "_Turn 2 synthesis failed._"))
            continue

        lines += ["#### Turn 2 — Synthesised Failure Analysis (labels revealed)", ""]

        # raw fallback
        if "_raw" in sections:
            lines += [
                "> _(response not structured — displaying raw output)_",
                "",
                f"> {sections['_raw']}",
                "",
            ]
            blind_spot_summaries.append((dt, sections["_raw"][:200]))
            continue

        # structured sections
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
        blind_spot_summaries.append((dt, rc if rc else "_Root cause not extracted._"))

    # ── Section 5: Blind Spot Summary ─────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 5. Model Blind Spot Summary  _(synthesised by VLM across images)_",
        "",
        "| Distortion | Synthesised Root Cause |",
        "|------------|------------------------|",
    ]
    for dt, summary in blind_spot_summaries:
        safe = summary.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| **{dt}** | {safe} |")

    # ── Section 6: Recommendations ────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 6. Actionable Recommendations",
        "",
        "Based on the failure distribution and VLM two-turn visual analysis above:",
        "",
        "| Priority | Recommendation | Addresses |",
        "|----------|---------------|-----------|",
        "| 🔴 High | **Multi-distortion augmentation** — blur (σ=0.5–3), JPEG sim (q=10–70), pixelation (block 2–8px), Gaussian noise (σ=0.05–0.25) | Blur (40%), JPEG (30%), Pixelate (15%) |",
        "| 🔴 High | **Dual-pathway backbone** — low-pass silhouette stream + high-freq stream merged via SE-Net/CBAM channel attention | Blur, Pixelate |",
        "| 🟡 Medium | **Gradient-reversal distortion-adversarial head** — auxiliary head predicts distortion type via GRL; forces distortion-invariant features | All types |",
        "| 🟡 Medium | **Fine-grained contrastive loss** — SupCon / NT-Xent for the cat/dog/deer/bird confusion cluster | JPEG, Blur |",
        "| 🟢 Low | **Train longer + cosine LR decay** — 3 epochs is underfit; val loss > train loss = early overfitting | All types |",
        "",
        "---",
        "",
        f"*Report generated by `pipe/vision_reasoning_report_v2.py` · {now}*  ",
        "*Strategy: multi-image batching + two-turn recursive reasoning*",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written → {output_path}")
    print(f"\n✅  Report written to: {output_path}")


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    _script_dir  = Path(__file__).resolve().parent
    _project_root = _script_dir.parent

    p = argparse.ArgumentParser(
        description="Generate a v2 AI vision reasoning report (multi-image + two-turn).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 pipe/vision_reasoning_report_v2.py\n"
            "  python3 pipe/vision_reasoning_report_v2.py --no-vlm\n"
            "  python3 pipe/vision_reasoning_report_v2.py --samples 3 --seed 42\n"
        ),
    )
    p.add_argument("--logs-dir",      type=Path, default=_script_dir / "logs")
    p.add_argument("--report",        type=Path, default=_script_dir / "reports" / "distortion_report.json")
    p.add_argument("--output",        type=Path, default=_project_root / "ai_reasoning_summary_v2.md")
    p.add_argument("--samples",       type=int,  default=3,
                   help="Images per distortion type sent to the VLM (default: 3)")
    p.add_argument("--training-log",  type=Path, default=None)
    p.add_argument("--misclassified", type=Path, default=None)
    p.add_argument("--port",          type=int,  default=8000)
    p.add_argument("--model",         type=str,  default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--no-vlm",        action="store_true",
                   help="Skip VLM calls entirely — stats-only mode, no server required")
    p.add_argument("--seed",          type=int,  default=42)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    logs_dir = args.logs_dir.resolve()

    # ── 1. Discover input files ────────────────────────────────────────────────
    training_log_path = args.training_log or find_latest_file(logs_dir, "training_log_*.json")
    if training_log_path is None or not training_log_path.exists():
        sys.exit(f"ERROR: No training_log_*.json found in {logs_dir}.")
    logger.info(f"Training log  : {training_log_path}")

    mc_path = args.misclassified or find_latest_file(logs_dir, "misclassified_*.json")
    if mc_path is None or not mc_path.exists():
        sys.exit(f"ERROR: No misclassified_*.json found in {logs_dir}.")
    logger.info(f"Misclassified : {mc_path}")

    report_path = args.report.resolve()
    if not report_path.exists():
        sys.exit(f"ERROR: distortion_report.json not found at {report_path}.")
    logger.info(f"Distortion report: {report_path}")

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

    # ── 3. Build LLM (unless --no-vlm) ────────────────────────────────────────
    llm = None
    if not args.no_vlm:
        ready = wait_for_server(args.port, timeout=300, poll_interval=5)
        if not ready:
            print("⚠️  Continuing without VLM — all image analyses will be skipped.")
            print("   Start the vLLM server first and re-run, or use --no-vlm.\n")
        else:
            logger.info(f"Building LLM → http://localhost:{args.port}/v1  model={args.model}")
            llm = build_llm(args.model, args.port)

    # ── 4. Per-distortion two-turn batch analysis ──────────────────────────────
    per_type_results: Dict[str, Dict] = {}
    total_vlm_calls = 0

    for dt in DISTORTION_TYPES:
        logger.info(f"\n{'─'*60}\nDistortion: {dt.upper()}")

        items = gather_images_for_distortion(
            dist_type  = dt,
            report     = dist_report,
            mc_stats   = mc_stats,
            images_dir = images_dir,
            n_samples  = args.samples,
            seed       = args.seed,
        )
        logger.info(f"  Sampled {len(items)} image(s).")

        if llm is not None and items:
            # 2 VLM calls per distortion type (Turn 1 + Turn 2)
            result = analyze_distortion_batch(llm, items, dt)
            total_vlm_calls += 2
        else:
            result = {"turn1_output": None, "turn2_sections": None, "items": items, "error": None}

        per_type_results[dt] = result

    logger.info(f"\nTotal VLM calls made: {total_vlm_calls}  "
                f"(vs {args.samples * len(DISTORTION_TYPES)} in v1 with same --samples)")

    # ── 5. Render report ───────────────────────────────────────────────────────
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_report(training, mc_stats, per_type_results, output_path)


if __name__ == "__main__":
    main()
