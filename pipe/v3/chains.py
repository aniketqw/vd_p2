"""
v3/chains.py
============
LLM construction and prompt message builders.

Separated from the graph so prompts can be tested independently.
All LangChain imports are isolated here — no other v3 module touches LangChain.

Public API
----------
build_llm(model_name, port)                          → ChatOpenAI
encode_image(path)                                   → str (base64)
build_turn1_message(items, dist_type)                → List[HumanMessage]
build_turn2_message(items, dist_type,
                    turn1_output, rag_context,
                    tool_results, format_instructions) → List[HumanMessage]
build_recommendations_message(stats_summary,
                               root_causes,
                               format_instructions)  → List[HumanMessage]
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .config import (
    DEFAULT_MODEL, DEFAULT_PORT,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_TOKENS_REC,
    CIFAR10_CLASSES,
)


# ── LLM factory ───────────────────────────────────────────────────────────────

def build_llm(
    model_name: str = DEFAULT_MODEL,
    port:       int = DEFAULT_PORT,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI pointed at the local vLLM server.
    max_tokens=1500: Turn 2 covers multiple images + five structured sections
                     + RAG context + tool results.
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key="local-3090",
        openai_api_base=f"http://localhost:{port}/v1",
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


def build_rec_llm(
    model_name: str = DEFAULT_MODEL,
    port:       int = DEFAULT_PORT,
) -> ChatOpenAI:
    """Lighter LLM for the recommendations call (text-only, no images)."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key="local-3090",
        openai_api_base=f"http://localhost:{port}/v1",
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS_REC,
    )


# ── image encoding ─────────────────────────────────────────────────────────────

def encode_image(image_path: Path) -> str:
    """Return base64-encoded PNG string for an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── helper formatters ──────────────────────────────────────────────────────────

def _image_roles_str(items: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        role = item.get("role", "random")
        dist = item.get("distance")
        suffix = f" (cluster dist = {dist:.3f})" if dist is not None else ""
        lines.append(f"  Image {i}: {role}{suffix}")
    return "\n".join(lines)


def _image_metadata_str(items: List[Dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        meta     = item.get("meta") or {}
        true_lbl = CIFAR10_CLASSES.get(meta.get("true_label"),
                                        str(meta.get("true_label", "?")))
        pred_lbl = CIFAR10_CLASSES.get(meta.get("predicted_label"),
                                        str(meta.get("predicted_label", "?")))
        conf     = meta.get("distortion_confidence") or 0.0
        role     = item.get("role", "random")
        lines.append(
            f"  Image {i} ({role}):  "
            f"true = {true_lbl}  |  predicted (wrong) = {pred_lbl}  "
            f"|  distortion confidence = {conf:.4f}"
        )
    return "\n".join(lines)


# ── Turn 1 prompt ─────────────────────────────────────────────────────────────

_TURN1_PROMPT = """\
You are an expert computer vision analyst.

You are given {n_images} CIFAR-10 image(s), each 32×32 pixels, all affected by \
{distortion_type} distortion. Study every image carefully — at this resolution \
every pixel cluster matters.

IMAGE ROLES:
{image_roles}
  (typical = near cluster centroid — representative sample)
  (outlier  = far from centroid   — unusual/edge-case sample)

━━━ YOUR TASK — visual observation only, NO class guessing ━━━
Do NOT name or guess object classes. Purely describe what you see.

For EACH image (label them Image 1, Image 2 …):

  ARTIFACTS: Describe the exact {distortion_type} degradation — name the specific \
region (top-left, center, background, edges) and the artifact type \
(e.g. blur halos, 8×8 block grid, mosaic tiles, speckle pattern).

  SURVIVING STRUCTURE: What visual information survived — color blobs, \
silhouette outlines, spatial layout, dominant hue?

  MOST DEGRADED: Which region or feature has been most destroyed?

Finally:
  COMMON PATTERN: What single visual element is present across ALL {n_images} image(s)?\
"""


def build_turn1_message(items: List[Dict], dist_type: str) -> List[HumanMessage]:
    """Construct the Turn 1 HumanMessage: text (no labels) + all images."""
    text = _TURN1_PROMPT.format(
        n_images        = len(items),
        distortion_type = dist_type,
        image_roles     = _image_roles_str(items),
    )
    content = [{"type": "text", "text": text}]
    for item in items:
        b64 = encode_image(item["path"])
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [HumanMessage(content=content)]


# ── Turn 2 prompt ─────────────────────────────────────────────────────────────

_TURN2_PROMPT = """\
You are an expert computer vision failure analyst.

━━━ STEP 1 — YOUR PRIOR VISUAL OBSERVATIONS ━━━
{turn1_output}

{rag_context}

{tool_results_block}

━━━ GROUND TRUTH LABELS (now revealed) ━━━
{image_metadata}

These {n_images} CIFAR-10 image(s) were MISCLASSIFIED by a neural network under \
{distortion_type} distortion. Using your Step 1 observations — and any historical \
context above — answer the five structured sections below.

{format_instructions}\
"""


def build_turn2_message(
    items:               List[Dict],
    dist_type:           str,
    turn1_output:        str,
    rag_context:         str = "",
    tool_results:        Optional[List[Dict]] = None,
    format_instructions: str = "",
) -> List[HumanMessage]:
    """
    Construct the Turn 2 HumanMessage:
      text (Turn 1 output + RAG context + tool results + labels) + same images.
    Images are resent because each .invoke() call is stateless.
    """
    # Format tool results block
    if tool_results:
        tool_lines = ["━━━ STATISTICAL VERIFICATION (from your dataset) ━━━"]
        for tr in tool_results:
            interp = tr.get("interpretation", str(tr))
            tool_lines.append(f"  • {interp}")
        tool_block = "\n".join(tool_lines)
    else:
        tool_block = ""

    text = _TURN2_PROMPT.format(
        turn1_output        = turn1_output.strip(),
        rag_context         = rag_context,
        tool_results_block  = tool_block,
        image_metadata      = _image_metadata_str(items),
        n_images            = len(items),
        distortion_type     = dist_type,
        format_instructions = format_instructions,
    )
    content = [{"type": "text", "text": text}]
    for item in items:
        b64 = encode_image(item["path"])
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [HumanMessage(content=content)]


# ── Recommendations prompt ────────────────────────────────────────────────────

_REC_PROMPT = """\
You are a machine learning improvement advisor for computer vision models.

A CIFAR-10 classifier (32×32 images, 3-epoch training run) produced the following
failure analysis:

━━━ FAILURE DISTRIBUTION ━━━
{failure_stats}

━━━ ROOT CAUSES BY DISTORTION TYPE (from VLM visual analysis) ━━━
{root_causes}

━━━ TOP CONFUSION PAIRS ━━━
{confusion_pairs}

Based on this SPECIFIC data — not generic advice — generate exactly 5 prioritised
recommendations. Each must:
  1. Target a specific failure pattern identified above
  2. Include a concrete implementation technique
  3. Estimate the % of failures it would address

{format_instructions}\
"""


def build_recommendations_message(
    stats_summary:       Dict,
    root_causes:         Dict[str, str],
    format_instructions: str = "",
) -> List[HumanMessage]:
    """
    Build the recommendations prompt from live failure stats and VLM root causes.
    Text-only (no images) — uses build_rec_llm for lower token budget.
    """
    by_dist = stats_summary.get("by_distortion", {})
    total   = stats_summary.get("n_misclassified", 1) or 1

    # Failure distribution block
    dist_lines = []
    for dt, info in by_dist.items():
        pct = round(info["count"] / total * 100, 1)
        dist_lines.append(
            f"  {dt}: {info['count']} failures ({pct}%), "
            f"avg distortion conf={info['avg_confidence']:.4f}"
        )
    failure_stats_str = "\n".join(dist_lines) if dist_lines else "  No data."

    # Root causes block
    rc_lines = [
        f"  {dt}: {rc}" for dt, rc in root_causes.items() if rc
    ]
    root_causes_str = "\n".join(rc_lines) if rc_lines else "  No VLM analysis available."

    # Top confusion pairs block
    pairs = stats_summary.get("top_confusion_pairs", [])[:10]
    pair_lines = [
        f"  {p['true']}→{p['pred']}: {p['count']}" for p in pairs
    ]
    confusion_str = "\n".join(pair_lines) if pair_lines else "  No data."

    text = _REC_PROMPT.format(
        failure_stats       = failure_stats_str,
        root_causes         = root_causes_str,
        confusion_pairs     = confusion_str,
        format_instructions = format_instructions,
    )
    return [HumanMessage(content=[{"type": "text", "text": text}])]
