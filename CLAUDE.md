# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📋 Project Overview

**vd_p2** is an AI vision reasoning system that analyzes misclassified images from CIFAR-10 model training runs. It generates comprehensive markdown reports explaining model blind spots, then automatically diagnoses and fixes the model code using an agentic LLM loop.

The system has **two entry points** and **three pipeline stages**:

| Entry point | What it runs |
|-------------|-------------|
| `unified_pipeline.py` ← **start here** | Stage A + B (+ optional C) end-to-end |
| `pipe/test.py --version v3` | Stage A only (V3 analysis) |

**Pipeline stages:**
- **Stage A (V3)**: Train CNN → export failures → cluster distortions → VLM reasoning → `ai_reasoning_summary_v3.md`
- **Stage B (Debug)**: Parse report → diagnose root causes → generate fix → human review → run → compare real metrics → improvement report
- **Stage C (optional)**: Re-run Stage A on fixed code → `ai_reasoning_summary_v3_fixed.md`

**For deep-dive architecture, see [`info.md`](./info.md)**

---

## 🏗️ Architecture Overview

```
unified_pipeline.py                         # ← SINGLE ENTRY POINT (recommended)
  │
  ├── STAGE A  (pipe/test.py --version v3)
  │     ├── pipe/test.py                        Train SimpleCNN + orchestrate
  │     ├── pipe/export_misclassified.py        Export failure samples
  │     ├── pipe/distortion_diagnostic_report.py  Cluster + archetype (t-SNE)
  │     └── pipe/v3/vision_reasoning_report_v3.py
  │           ├── config.py, data_loader.py, embeddings.py
  │           ├── image_sampler.py (semantic dedup), rag.py (cross-run memory)
  │           ├── schemas.py (Pydantic), tools.py (stat callables)
  │           ├── chains.py (LLM + prompts), graph.py (observe→hypothesise→verify→conclude)
  │           ├── renderer.py (report writer + dynamic recs)
  │           └── server_utils.py (vLLM health check)
  │
  ├── STAGE B  (agentic_debugger.py — imported directly)
  │     ├── Node 1: parse_report()         Markdown → structured findings
  │     ├── Node 2: analyze_code()         Map findings to lines  (Local LLM)
  │     ├── Node 3: diagnose_root_causes() Root cause + confidence (Groq)
  │     ├── Node 4: generate_fix()         Fixed code with comments (Groq)
  │     ├── [HUMAN REVIEW]                approve / reject
  │     ├── Node 5: run_fixed_code()       Execute → new training_log_*.json
  │     ├── Node 6: compare_metrics()      Before vs after (no LLM)
  │     └── Node 7/8: generate_report()   Markdown with REAL results (Groq)
  │
  └── STAGE C  (optional, re-runs Stage A on fixed code)
        └── → ai_reasoning_summary_v3_fixed.md

Artifacts (project root):
  ai_reasoning_summary_v3.md         Stage A output
  master_pipeline_report.md          Unified output (A + B + C)
  ai_reasoning_summary_v3_fixed.md   Stage C output (if run)

Artifacts (pipe/logs/):
  training_log_*.json                Baseline metrics (Stage A)
  misclassified_*.json               Failure metadata (Stage A)

Artifacts (pipe/reports/):
  distortion_report.json             Clustering output (Stage A)
  rag_index.npz                      RAG vector index (Stage A, cached)
```

### Full Data Flow

```
CIFAR-10 (torch/vision)
  ↓ [Stage A]
test.py → training_log_*.json + misclassified_*.json
  ↓
distortion_diagnostic_report.py → distortion_report.json
  ↓
vision_reasoning_report_v3.py (vLLM + RAG + LangGraph)
  ↓
ai_reasoning_summary_v3.md
  ↓ [Stage B]
agentic_debugger: parse → diagnose → fix → validate
  ↓
test_fixed.py (generated) + training_log_fixed_*.json
  ↓
master_pipeline_report.md  (with REAL before/after metrics)
  ↓ [Stage C, optional]
vision_reasoning_report_v3.py on fixed run
  ↓
ai_reasoning_summary_v3_fixed.md
  ↓
ai_reasoning_summary_v3.md (analysis report)
```

### Key Concepts

**Distortion Types**: blur, brightness, contrast, fog, frosted_glass, etc.
- Applied to training images to test model robustness
- V3 analyzes failure patterns per distortion type

**RAG (Retrieval-Augmented Generation)**:
- Vector index of all misclassified images from ALL past training runs
- Cached in `rag_index.npz` (built once, reused)
- Injected into analysis prompts as historical context

**Agentic Loop** (V3 only):
- 4-node state machine: observe → hypothesise → verify → conclude
- `verify` calls stat tools to validate hypotheses against data
- Revises if hypothesis contradicts statistics (max 2 iterations)

**Pydantic Structured Output** (V3 only):
- Turn 2 response parsed into strict JSON schema
- Explicit validation failures (not silent degradation)

---

## 🚀 Common Commands

### ⭐ Recommended: Unified Pipeline (all stages)

```bash
# Terminal 1: Start vLLM server (for Stage A visual analysis)
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000

# Terminal 2: Unified pipeline (Stage A + B)
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY
# Outputs: ai_reasoning_summary_v3.md + master_pipeline_report.md

# With re-analysis (Stage A + B + C)
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY --stage-c

# No GPU / no vLLM (stats-only analysis, still runs full debug loop)
python3 unified_pipeline.py --no-vlm --groq-api-key $GROQ_API_KEY

# Non-interactive (CI / headless)
python3 unified_pipeline.py --no-vlm --auto-approve --groq-api-key $GROQ_API_KEY
```

### Stage A Only (V3 Analysis — unchanged from before)

```bash
# Via unified pipeline
python3 unified_pipeline.py --only-stage a

# Direct (original command, still works)
cd pipe && python3 test.py --version v3
```

### Stage B Only (Debug an existing report)

```bash
# Assumes ai_reasoning_summary_v3.md already exists
python3 unified_pipeline.py --only-stage b \
  --groq-api-key $GROQ_API_KEY \
  --max-iterations 2

# Point at a custom report
python3 unified_pipeline.py --only-stage b \
  --report path/to/my_report.md \
  --code pipe/test.py
```

### V3 Pass-Through Flags (unified_pipeline.py forwards these to Stage A)

```bash
--no-vlm            # Skip vLLM (stats + RAG only, no GPU)
--no-rag            # Skip RAG index (faster cold start)
--rebuild-rag       # Force rebuild RAG even if cached
--samples N         # Images per distortion type (default: 3)
--seed N            # Random seed (default: 42)
--no-tool-trace     # Hide tool verification results from V3 report
```

### Stage B / Debugger Flags

```bash
--groq-api-key KEY          # Groq API key (or set $GROQ_API_KEY)
--local-llm-port PORT       # Local LLM server port (default: 11434)
--local-llm-format FORMAT   # API format: 'ollama' (default) or 'openai'
                            #   'ollama'  — Ollama /api/generate, port 11434
                            #   'openai'  — OpenAI-compatible /v1/chat/completions
                            #             Use for LM Studio, LocalAI, port 8081, etc.
--groq-rate-limit N         # Max Groq calls/min (default: 100)
--debug-llm-provider        # auto | groq | local
--max-iterations N          # Max fix attempts (default: 3)
--auto-approve              # Skip human review (non-interactive)
--stage-c                   # Re-run V3 on fixed code after success
```

**Using the port 8081 OpenAI-compatible server for Stage B:**
```bash
python3 unified_pipeline.py --only-stage b \
  --local-llm-port 8081 --local-llm-format openai \
  --auto-approve
```

### Testing / Debugging

```bash
# List available distortion types
python3 -c "from pipe.v3.config import DISTORTION_TYPES; print(DISTORTION_TYPES)"

# Quick test run (3 epoch training, 1 sample per distortion)
python3 pipe/test.py --version v3 --no-vlm --samples 1 --seed 42

# Check if vLLM server is running
curl http://localhost:8000/v1/models
```

---

## 🎯 Key Files to Know

### Entry Points
- **`pipe/test.py`**: Orchestrates full pipeline (train → export → cluster → report)
- **`pipe/v3/vision_reasoning_report_v3.py`**: V3 report generation (requires vLLM server)

### V3 Modules (One responsibility each)
| File | Purpose |
|------|---------|
| `config.py` | Constants, paths, LLM settings |
| `data_loader.py` | Parse training_log_*.json, misclassified_*.json |
| `embeddings.py` | Extract pixel-level CNN features |
| `image_sampler.py` | Sample + semantic deduplication |
| `rag.py` | Build/load vector index of past failures |
| `schemas.py` | Pydantic models for structured output |
| `tools.py` | Stat callables for agentic verify node |
| `chains.py` | LLM setup + prompt templating |
| `graph.py` | LangGraph state machine (observe → hypothesise → verify → conclude) |
| `renderer.py` | Markdown report writing + dynamic recommendations |
| `server_utils.py` | vLLM health checking |

### Generated Artifacts
- `pipe/logs/training_log_*.json` — Training metrics (epoch, accuracy, loss)
- `pipe/logs/misclassified_*.json` — Misclassified samples with metadata
- `pipe/reports/distortion_report.json` — Clustering/archetype analysis
- `pipe/reports/rag_index.npz` — Cached RAG vector index
- `ai_reasoning_summary_v3.md` — Final analysis report (project root)

---

## 🔧 Architecture Patterns

### Dependency Injection
- LLMChain, RAGStore, and graph all passed to functions
- Easy to mock for testing or swap implementations

### Modular Configuration
- All constants in `v3/config.py`
- Single source of truth for tuning (model name, port, distortion types, iteration limits)

### Explicit Error Handling
- Pydantic validation failures shown verbatim (not silent)
- RAG unavailability gracefully degrades (continues without cross-run context)
- vLLM server health-checked before use (timeout 300s)

### Caching Strategy
- RAG index built once, cached in `rag_index.npz`
- Automatic staleness detection: rebuilds if newer misclassified JSON found
- Per-distortion results stored in `per_type_states` dict

### State Machine Pattern
- LangGraph 4-node pipeline: observe, hypothesise, verify, conclude
- Revision loop: if verify fails, jump back to hypothesise with tool results
- Explicit state transitions with typed `AnalysisState` Pydantic model

---

## ⚙️ Setup & Dependencies

### Python & Virtual Environment
```bash
# Activate venv (adjust path as needed)
source venv_vision/bin/activate

# Key dependencies (implied by code):
# - torch, torchvision (CNN training)
# - pytorch-lightning (training orchestration)
# - langchain, langgraph (agentic loops + chains)
# - pydantic (structured output validation)
# - vllm (model server, run separately)
# - faiss-cpu or faiss-gpu (vector search for RAG)
# - numpy, scipy (embeddings, similarity)
```

### vLLM Server
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (vision language model)
- Quantization: `bitsandbytes` (4-bit)
- GPU memory: 0.4 utilization (adjust if OOM)
- Context length: 2048 tokens (raise to 4096 if Turn 2 prompt truncates)
- Port: 8000 (default, customizable via `--port`)

---

## 📖 Documentation Reference

- **`pipe/help_v3.md`**: User-facing pipeline guide (running, troubleshooting)
- **`info.md`**: Complete technical analysis of V3 (refer here for deep dives)
- **`pipe/README.md`**: Original project documentation

For questions about V3 design, architecture, or troubleshooting, **start with `info.md`**.

---

## 🐛 Debugging Tips

### vLLM Server Health
```bash
# Check if running
curl http://localhost:8000/v1/models

# Check logs (adjust vLLM server terminal)
# Look for: "INFO:     Application startup complete."
```

### RAG Index Issues
```python
# Force rebuild
python3 pipe/vision_reasoning_report_v3.py --rebuild-rag

# Skip RAG (faster)
python3 pipe/vision_reasoning_report_v3.py --no-rag
```

### Pydantic Parse Failures
- Check vLLM `--max-model-len` (should be ≥2048, ideally 4096)
- Raw Turn 2 text is logged if JSON parsing fails
- Verify data_loader correctly loads test JSON files

### Long Execution
- RAG index build: first run 1–2 min, cached runs <1s
- Per-distortion analysis: ~30–60s (depends on image count)
- Total: 5–10 min with warm vLLM server
- Use `--no-vlm` for stats-only (no GPU)

---

## 🔄 Typical Workflows

### Developing V3 Features
1. Modify module in `pipe/v3/` (e.g., `rag.py`)
2. Test with `--no-vlm` first (faster, no GPU)
3. Run full pipeline once with vLLM to verify end-to-end

### Running Analysis on New Training
1. Train: `python3 pipe/test.py` (generates logs/)
2. Cluster: Built into test.py pipeline
3. Report: `python3 pipe/vision_reasoning_report_v3.py --no-vlm` (or with vLLM)

### Tuning Agent Behavior
1. Edit `pipe/v3/config.py` (AGENT_MAX_ITERATIONS, distortion types, etc.)
2. Rerun: `python3 pipe/vision_reasoning_report_v3.py`
3. Check `ai_reasoning_summary_v3.md` for changes

### Adding New Tools
1. Define callable in `pipe/v3/tools.py`
2. Register in `build_graph()` in `pipe/v3/graph.py`
3. Include in verify node's tool call

---

## 📌 Design Constraints

- **CIFAR-10 only**: Dataset hardcoded in training script
- **Distortion-centric**: Analysis grouped by distortion type
- **vLLM requirement**: Turn 1 & 2 analysis need OpenAI-compatible server
- **Single-node**: No distributed training (torch.nn parallel only)
- **Deterministic (mostly)**: Seed controls randomness; RAG index is sorted

---

## 🚨 Common Pitfalls

1. **vLLM server not running**: Graph will wait 300s then continue without VLM (stats only)
2. **Model weights not found**: Training auto-downloads CIFAR-10; vLLM downloads model from HuggingFace
3. **Images folder missing**: Pipeline disables image-based sampling; continues with text-only analysis
4. **JSON parsing fails**: Check if vLLM output is malformed (usually from context overflow)
5. **RAG rebuild takes forever**: Embedding extraction slow with many images; use `--no-rag` if not needed
