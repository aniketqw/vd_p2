# 📌 `pipe/` — What this folder does

This folder contains the training/analysis pipeline used to:

- Train a CIFAR-10 model and log misclassified samples
- Predict the type of distortion on each misclassified image (blur/jpeg/pixelate/noise)
- Export misclassified images and run clustering diagnostics

---

## 🧠 What’s implemented (current capabilities)

### ✅ Training + misclassification logging
- `test.py` runs a PyTorch Lightning training loop on CIFAR-10.
- `pipe/debug_logger.py` is a Lightning callback that:
  - records per-epoch stats (loss/accuracy/time)
  - collects misclassified samples (image base64 + true/pred labels)
  - runs a distortion classifier (Ultralytics YOLO `best.pt`) and stores:
    - `distortion_predicted` (blur/jpeg/pixelate/noise)
    - `distortion_confidence`

Output:
- `logs/training_log_*.json` (full training metrics + misclassified samples)
- `logs/misclassified_*.json` (exported misclassified samples only)

---

### ✅ Misclassified export
- `export_misclassified.py` reads a training log JSON and writes a smaller JSON with just the misclassified samples.

Example:
```bash
cd pipe
python export_misclassified.py \
  --input logs/training_log_20260317_005331.json \
  --output logs/misclassified_20260317_005331.json
```

---

### ✅ Diagnostic report (clustering + archetypes)
- `distortion_diagnostic_report.py` creates a report from misclassified images.
- It can **take either**:
  1) a directory with per-distortion subfolders (blur/jpeg/pixelate/noise)
  2) a misclassified JSON log (`logs/misclassified_*.json`) — it auto-extracts the images

It produces:
- `reports/distortion_report.json` — archetypes (typical/outlier images + distances)
- `reports/distortion_clusters.png` — t-SNE scatterplot (colored by distortion)

---


### ✅ AI Vision Reasoning Report (`vision_reasoning_report.py`)

Reads training logs + misclassified images, sends representative images to a
locally-hosted **Qwen2.5-VL-7B-Instruct** vision-language model, and writes a
deep structured Markdown reasoning report (`ai_reasoning_summary.md`).

For each misclassified image the VLM is forced to produce **7 labelled sections**:

| Section | What it answers |
|---------|----------------|
| 🔬 Distortion Artifacts | Exact artifacts visible — blur halos, JPEG 8×8 blocks, mosaic tiles, noise speckle — with region references |
| ✅ Surviving True-Class Features | What still looks correct despite the distortion |
| ❌ What Misled the Model | The specific artifact / region that triggered the wrong prediction |
| 🧠 Model Reasoning — Correct | What the model partially got right |
| 🚫 Model Reasoning — Incorrect | The exact mistake — which degraded pattern fired the wrong class |
| 📊 Confidence Assessment | Why distortion confidence is that specific number; is the distortion obvious or subtle even to a human? |
| 🎯 Root Cause | One sentence: the core mechanism of this failure |

**Do I need to start the VLM server manually?**
**Yes.** The script calls the VLM over HTTP on port 8000. Start the server in a
separate terminal first and wait for `Application startup complete` before running
the script. See Mode A below.

---

#### Mode A — Full deep report (with VLM visual analysis) 🖼️

**Terminal 1 — Start the vLLM server and keep it running:**

```bash
cd ~/vision_dev_project
source venv_vision/bin/activate

VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE=”/mnt/data/pratik_models” \
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

**Terminal 2 — Run the report:**

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report.py
```

No arguments needed — auto-discovers the latest logs.

The script **automatically waits** for the server to be ready (polls `/health`
every 5 s, up to 5 minutes). You will see a live countdown:

```
⏳ Waiting for vLLM server on port 8000 (timeout=300s)…
   not ready yet — retrying in 5s (~270s remaining) …
✅ Server ready (after ~30s)
```

If the server never becomes ready within 5 minutes the script falls back to
stats-only mode automatically — no crash, no lost work.

Output → `~/vision_dev_project/ai_reasoning_summary.md`

---

#### Mode B — Stats-only report (no GPU / no VLM server) 📊

No server, no GPU needed. Produces failure statistics, confusion tables, and
recommendations — but without the per-image VLM visual analysis sections.

```bash
cd ~/vision_dev_project/pipe
source ../venv_vision/bin/activate
python3 vision_reasoning_report.py --no-vlm
```

---

#### Optional flags

```
--samples N        Images per distortion type sent to the VLM (default: 3)
--port PORT        vLLM server port (default: 8000)
--model NAME       Model name served by vLLM (default: Qwen/Qwen2.5-VL-7B-Instruct)
--logs-dir PATH    Override logs folder (default: pipe/logs)
--report PATH      Override distortion_report.json path (default: pipe/reports/distortion_report.json)
--output PATH      Override output .md path (default: ai_reasoning_summary.md in project root)
--training-log     Override: use a specific training_log JSON instead of auto-discovering latest
--misclassified    Override: use a specific misclassified JSON instead of auto-discovering latest
--no-vlm           Skip VLM calls entirely — stats-only mode, no server required
--seed N           Random seed for image sampling (default: 42)
```

Example with all flags explicit (run from inside `pipe/`):

```bash
python3 vision_reasoning_report.py \
  --logs-dir  logs \
  --report    reports/distortion_report.json \
  --output    ../ai_reasoning_summary.md \
  --samples   3 \
  --model     Qwen/Qwen2.5-VL-7B-Instruct \
  --port      8000
```

---

#### Prerequisites check

Before running, confirm all four conditions are met:

```
1. pipe/logs/training_log_*.json        → run test.py first
2. pipe/logs/misclassified_*.json       → produced by debug_logger.py during training
3. pipe/reports/distortion_report.json  → run distortion_diagnostic_report.py first
4. (Mode A only) vLLM server running    → Terminal 1 shows “Application startup complete”
```

---

## ▶️ Full recommended flow (all steps)

```bash
# ── Terminal 1: keep this running for step 4 ──────────────────────────────────
cd ~/vision_dev_project
source venv_vision/bin/activate
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE=”/mnt/data/pratik_models” \
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

# Step 3 — deep AI visual reasoning report (requires Terminal 1 running)
python3 vision_reasoning_report.py
# produces: ../ai_reasoning_summary.md  (project root)

# Step 3 (alt) — stats-only, no VLM server needed
python3 vision_reasoning_report.py --no-vlm
```

---

## 📂 Directory structure (what you should see)

```
pipe/
  best.pt                       # distortion classifier (YOLO) — Git LFS
  test.py                       # train + logging
  debug_logger.py               # Lightning callback (logging + distortion prediction)
  export_misclassified.py
  distortion_diagnostic_report.py
  vision_reasoning_report.py    # AI VLM reasoning report ← NEW

  data/                         # CIFAR-10 data (downloaded on first run)
  logs/
    training_log_*.json
    misclassified_*.json
    misclassified_*_images/     # extracted images (blur/jpeg/pixelate/noise/unknown)
  reports/
    distortion_report.json
    distortion_clusters.png

ai_reasoning_summary.md         # final report (written to project root)
```

---

## 🧪 Notes / troubleshooting

**distortion_diagnostic_report.py**
- If you pass a folder with no images it will say “No images found.” — pass the `.json` log file instead; images are extracted automatically.
- ResNet18 weights are downloaded on the first run and cached under `~/.cache/torch/hub`.

**vision_reasoning_report.py**
- Run from inside `pipe/` or from the project root — paths resolve correctly either way.
- If the vLLM server is not running you will see `Connection refused` for every image. Switch to `--no-vlm` or start the server first.
- The server takes **30–90 seconds to load the model**. The script polls `/health` automatically so you no longer need to wait and watch Terminal 1 — just run the script and it will block until the server is ready.
- VRAM at 4-bit quantization: Qwen2.5-VL uses ~5.8 GB. If you hit OOM reduce `--gpu-memory-utilization 0.3` in the server command.
- Each image makes one VLM call (~2–5 s each). With `--samples 3` and 4 distortion types that is 12 calls total (~1 min).
- If the VLM response does not contain the expected section labels (`DISTORTION ARTIFACTS:`, `ROOT CAUSE:`, etc.) it is stored as `_raw` and displayed verbatim — the report still generates cleanly.
- The distortion predictor (`best.pt`) is run by `debug_logger.py` during training automatically — you do not need to call it manually.
