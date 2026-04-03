# 📌 `pipe/` — v2 Pipeline Guide

This document covers `vision_reasoning_report_v2.py`, the upgraded AI reasoning
report generator. All other scripts (`test.py`, `debug_logger.py`,
`export_misclassified.py`, `distortion_diagnostic_report.py`) are unchanged from
v1. Only the final reporting step is replaced.

---

## 🔄 What changed from v1

| | v1 | v2 |
|---|---|---|
| Images per VLM call | 1 | All samples for a distortion type together |
| Turns per distortion type | 1 | 2 (visual grounding → causal reasoning) |
| Labels shown to model | Before image analysis | Withheld in Turn 1, revealed in Turn 2 |
| Root cause | Per-image string, joined with `\|` | Synthesised across all images in one call |
| Total VLM calls (`--samples 3`) | 12 | 8 |
| Cross-image comparison | ❌ | ✅ |
| Typical vs outlier reasoning | ❌ | ✅ |

### Why two turns?

**Turn 1 — Visual grounding (no labels)**
The model receives all sampled images for one distortion type and is asked to
describe only what it sees: exact artifacts, surviving structure, and what has
been degraded — without knowing the class labels. This forces genuine pixel-level
observation before any class metadata can bias the analysis.

**Turn 2 — Causal reasoning (labels revealed)**
Turn 1's observations are injected as evidence. The model now receives the true
and predicted class labels for each image and is asked to answer five structured
questions using its prior observations as grounding:

| Section | What it answers |
|---------|----------------|
| 🔗 Shared Failure Pattern | Which artifact from Turn 1 is present in ALL images and caused the wrong prediction |
| ⭕↔⚠️ Typical vs Outlier | Do typical and outlier images fail for the same reason, or different mechanisms? |
| ❌ What Misled the Model | The exact degraded region that activated the wrong class |
| 📊 Confidence Assessment | Why the distortion confidence is at that level; obvious vs subtle at 32×32 |
| 🎯 Root Cause | One sentence: the core mechanism of this class of failure |

---

## 🧠 What's implemented (full pipeline)

### ✅ Training + misclassification logging
Unchanged from v1. See `help.md`.

### ✅ Misclassified export
Unchanged from v1. See `help.md`.

### ✅ Diagnostic report (clustering + archetypes)
Unchanged from v1. See `help.md`.

---

### ✅ AI Vision Reasoning Report v2 (`vision_reasoning_report_v2.py`)

Reads training logs + misclassified images + distortion cluster report, then
runs a **two-turn recursive analysis** using a locally-hosted
**Qwen2.5-VL-7B-Instruct** vision-language model (via vLLM on port 8000).

For each distortion type the script makes exactly **2 VLM calls**:

```
Turn 1  →  all N images sent together, labels withheld
           model describes artifacts, surviving structure, common pattern

Turn 2  →  same N images + Turn 1 output + labels revealed
           model synthesises shared failure pattern, typical-vs-outlier
           comparison, and a single root cause sentence
```

Output → `ai_reasoning_summary_v2.md` (project root by default)

**Do I need to start the VLM server manually?**
**Yes.** The script calls the VLM over HTTP on port 8000. Start the server in a
separate terminal first and wait for `Application startup complete` before
running the script. See Mode A below.

---

#### Mode A — Full deep report (with VLM visual analysis) 🖼️

**Terminal 1 — Start the vLLM server and keep it running:**

```bash
cd ~/vision_dev_project
source venv_vision/bin/activate

VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
```

Wait until this line appears in Terminal 1:
```
INFO:     Application startup complete.
```

First run downloads the model (~5.8 GB to `/mnt/data/pratik_models`).

**Terminal 2 — Run the v2 report:**

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report_v2.py
```

No arguments needed — auto-discovers the latest logs.

The script **automatically waits** for the server to be ready (polls `/health`
every 5 s, up to 5 minutes):

```
⏳ Waiting for vLLM server on port 8000 (timeout=300s)…
   not ready yet — retrying in 5s (~270s remaining) …
✅ Server ready (after ~30s)
```

If the server never becomes ready within 5 minutes the script continues without
VLM calls — no crash, no lost work. Stats sections of the report are still
written.

Output → `~/vision_dev_project/ai_reasoning_summary_v2.md`

---

#### Mode B — Stats-only report (no GPU / no VLM server) 📊

No server, no GPU needed. Produces failure statistics, confusion tables, and
recommendations without the per-distortion-type VLM visual analysis sections.

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report_v2.py --no-vlm
```

---

#### Optional flags

```
--samples N        Images per distortion type sent to VLM in one batch (default: 3)
--port PORT        vLLM server port (default: 8000)
--model NAME       Model name served by vLLM (default: Qwen/Qwen2.5-VL-7B-Instruct)
--logs-dir PATH    Override logs folder (default: pipe/logs)
--report PATH      Override distortion_report.json path (default: pipe/reports/distortion_report.json)
--output PATH      Override output .md path (default: ai_reasoning_summary_v2.md in project root)
--training-log     Override: use a specific training_log JSON instead of auto-discovering latest
--misclassified    Override: use a specific misclassified JSON instead of auto-discovering latest
--no-vlm           Skip VLM calls entirely — stats-only mode, no server required
--seed N           Random seed for image sampling (default: 42)
```

Example with all flags explicit (run from inside `pipe/`):

```bash
python3 vision_reasoning_report_v2.py \
  --logs-dir  logs \
  --report    reports/distortion_report.json \
  --output    ../ai_reasoning_summary_v2.md \
  --samples   3 \
  --model     Qwen/Qwen2.5-VL-7B-Instruct \
  --port      8000 \
  --seed      42
```

---

#### Prerequisites check

Before running, confirm all four conditions are met:

```
1. pipe/logs/training_log_*.json        → run test.py first
2. pipe/logs/misclassified_*.json       → produced by debug_logger.py during training
3. pipe/reports/distortion_report.json  → run distortion_diagnostic_report.py first
4. (Mode A only) vLLM server running    → Terminal 1 shows "Application startup complete"
```

---

## ▶️ Full recommended flow (all steps)

```bash
# ── Terminal 1: keep this running for step 3 ──────────────────────────────────
cd ~/vision_dev_project
source venv_vision/bin/activate
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000
# wait for: INFO:     Application startup complete.


# ── Terminal 2: pipeline steps ────────────────────────────────────────────────
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate

# Step 1 — train and log misclassified samples
python3 test.py
# produces: logs/training_log_*.json  +  logs/misclassified_*.json

# Step 2 — cluster diagnostic (t-SNE + archetype selection)
python3 distortion_diagnostic_report.py \
  --base-dir logs/misclassified_YYYYMMDD_HHMMSS.json \
  --output   reports/distortion_report.json \
  --plot     reports/distortion_clusters.png
# produces: reports/distortion_report.json  +  reports/distortion_clusters.png

# Step 3 — v2 deep AI reasoning report (requires Terminal 1 running)
python3 vision_reasoning_report_v2.py
# produces: ../ai_reasoning_summary_v2.md  (project root)

# Step 3 (alt) — stats-only, no VLM server needed
python3 vision_reasoning_report_v2.py --no-vlm
```

---

## 📂 Directory structure (what you should see)

```
pipe/
  best.pt                           # distortion classifier (YOLO) — Git LFS
  test.py                           # train + logging
  debug_logger.py                   # Lightning callback (logging + distortion prediction)
  export_misclassified.py
  distortion_diagnostic_report.py
  vision_reasoning_report.py        # v1 — single-turn per-image analysis
  vision_reasoning_report_v2.py     # v2 — multi-image batching + two-turn reasoning ← NEW

  data/                             # CIFAR-10 data (downloaded on first run)
  logs/
    training_log_*.json
    misclassified_*.json
    misclassified_*_images/         # extracted images (blur/jpeg/pixelate/noise/unknown)
  reports/
    distortion_report.json
    distortion_clusters.png

ai_reasoning_summary.md             # v1 report output (project root)
ai_reasoning_summary_v2.md          # v2 report output (project root) ← NEW
```

---

## 📋 Report structure (v2 output)

The generated `ai_reasoning_summary_v2.md` contains six sections:

```
1. Training Summary
   Accuracy, loss, epoch progression table

2. Failure Distribution by Distortion Type
   Count, % of total, avg distortion confidence per type

3. Top Confusion Pairs
   The 15 most common true→predicted misclassification pairs

4. Per-Distortion Synthesised Analysis  (one subsection per distortion type)
   ├── Stats: most misclassified classes, epoch distribution
   ├── Images analysed table (file, role, true, predicted, confidence)
   ├── Turn 1 block: raw visual observations (no labels)
   └── Turn 2 structured sections:
       🔗 Shared Failure Pattern
       ⭕↔⚠️ Typical vs Outlier
       ❌ What Misled the Model
       📊 Confidence Assessment
       🎯 Root Cause

5. Model Blind Spot Summary
   One-row-per-distortion table of synthesised root causes

6. Actionable Recommendations
   Priority-ranked improvement suggestions
```

---

## 🧪 Notes / troubleshooting

**VLM call count**
With `--samples 3` (default), v2 makes **8 VLM calls** (2 turns × 4 distortion
types) compared to 12 in v1. Each call is larger (multiple images + longer
prompts) so per-call latency is slightly higher — expect ~3–8 s per call.
Total runtime: roughly 1–2 minutes once the server is warm.

**`max_tokens` is 1500 in v2 (vs 900 in v1)**
Turn 2 must cover multiple images and five structured sections. If responses
are being cut off mid-sentence, the server's `--max-model-len` may be too low.
Raise it to `4096` in the server start command.

**Images are sent twice (once per turn)**
Turn 2 resends the same images alongside Turn 1's text output because each
`.invoke()` call is stateless — the server has no session memory. This is
intentional and correct. It doubles image encoding work but adds negligible
latency at 32×32 px.

**Turn 1 fails but Turn 2 does not run**
If Turn 1 raises a connection error, the batch for that distortion type is
skipped entirely and the report section notes the failure. Start the vLLM
server first (`Application startup complete`) before running the script.

**Turn 2 response has no section headers**
If the VLM does not follow the five-section format, the full response is stored
as `_raw` and displayed verbatim in the report. This can happen with very short
`--max-model-len` values that truncate the prompt. Raise `--max-model-len 4096`
in the server start command.

**Only 1–2 images available for a distortion type**
The script handles any number of images ≥ 1. The typical/outlier comparison
section will note that only one image was available and skip the contrast
analysis naturally — no crash.

**`distortion_diagnostic_report.py` — no archetypes in report**
If `distortion_report.json` has empty archetype lists for a distortion type,
image sampling falls back to random files from the `misclassified_*_images/`
folder. Pass the `.json` log to `distortion_diagnostic_report.py` (not an empty
folder) to ensure archetypes are populated.

**VRAM**
Qwen2.5-VL at 4-bit quantization uses ~5.8 GB. Sending 3 images per call
adds minimal VRAM overhead. If you hit OOM, reduce
`--gpu-memory-utilization 0.3` in the server start command, or drop
`--samples` to 2.
