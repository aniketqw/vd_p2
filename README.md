# 🤖 Unified Vision AI Pipeline

A complete end-to-end system that automatically trains CNN models on CIFAR-10, analyzes failures using Vision-Language Models (VLMs) with RAG, diagnoses root causes with agentic reasoning, generates and validates fixes, and re-analyzes to measure real improvements.

**Status:** ✅ Production Ready | **Version:** 1.0 | **Last Updated:** 2026-04-21

---

## 📋 Quick Navigation

| I want to... | Read this | Time |
|---|---|---|
| **Get started immediately** | [Quick Start](#-quick-start) | 2 min |
| **Understand the system** | [CLAUDE.md](./CLAUDE.md) | 5 min |
| **Run the pipeline** | [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md) | 10 min |
| **Learn technical details** | [info.md](./info.md) | 30 min |
| **Understand design patterns** | [AGENTIC_SYSTEM_DESIGN.md](./AGENTIC_SYSTEM_DESIGN.md) | 20 min |

---

## 🚀 Quick Start

### Option 1: Full Pipeline (GPU + vLLM)

```bash
# Terminal 1: Start vLLM server
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --port 8000

# Terminal 2: Run pipeline
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY
```

**Output:** `master_pipeline_report.md` with before/after metrics  
**Time:** ~8-12 minutes

---

### Option 2: No GPU (Stats-only)

```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY --no-vlm
```

**Output:** Same as Option 1 (VLM analysis skipped)  
**Time:** ~3-5 minutes

---

### Option 3: Non-Interactive (CI/CD)

```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --auto-approve \
  --no-vlm
```

**Features:** Skips human review, auto-approves fixes  
**Time:** ~3-5 minutes

---

## 🎯 What This System Does

```
STAGE A: V3 Analysis (Train + Analyze)
  ├─ Train SimpleCNN on CIFAR-10
  ├─ Export misclassified samples
  ├─ Cluster failures by distortion type
  └─ VLM reasoning with RAG + LangGraph
     Output: ai_reasoning_summary_v3.md

STAGE B: Agentic Debugging (Diagnose + Fix + Validate)
  ├─ Parse findings from V3 report
  ├─ Diagnose root causes
  ├─ Generate fixed code
  ├─ Human review & approval
  ├─ Execute & capture metrics
  └─ Compare real before/after results
     Output: master_pipeline_report.md + test_fixed.py

STAGE C: Re-Analysis (Optional - Validate Fix)
  ├─ Re-run V3 on fixed code
  └─ Compare original vs fixed analysis
     Output: ai_reasoning_summary_v3_fixed.md
```

### Key Features

✅ **Real metrics only** — Uses actual training_log_*.json files  
✅ **Human-in-the-loop** — Interactive approval before execution  
✅ **Flexible LLM backends** — Groq + Ollama with auto-fallback  
✅ **Automatic iteration** — Refines diagnosis (max 3 attempts)  
✅ **No code modifications** — All original code unchanged  

---

## 📚 Documentation Structure

### For Users (Getting Started)

**Start here:** [CLAUDE.md](./CLAUDE.md)
- Project structure overview
- Entry points and quick reference
- Common commands for all workflows
- Setup & dependencies

**Complete guide:** [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md)
- Installation & setup (vLLM, Groq, Ollama)
- 8 detailed usage examples:
  1. Full end-to-end pipeline
  2. Skip vLLM (no GPU)
  3. Non-interactive (CI/CD)
  4. Stage A only (V3 analysis)
  5. Stage B only (debug existing report)
  6. Full pipeline + re-analysis
  7. Custom LLM backends
  8. Tune stage-specific flags
- Stage breakdown with configuration
- Output artifacts reference
- Comprehensive troubleshooting (8 common issues)
- Performance benchmarks & FAQ

### For Developers (Technical Details)

**Deep-dive analysis:** [info.md](./info.md)
- Complete technical analysis of V3 system
- All 10 V3 modules explained
- Data flow and dependencies
- Dependency injection patterns
- Configuration & constants
- Caching strategy
- State machine pattern
- Debugging tips

**Design patterns:** [AGENTIC_SYSTEM_DESIGN.md](./AGENTIC_SYSTEM_DESIGN.md)
- Agentic system design
- LLM strategy for each phase
- Rate limiting & fallback logic
- Output file formats
- Integration points
- Future enhancements

---

## 📁 Project Structure

```
vd_p2/
├── 📄 README.md ..................... This file
├── 📄 CLAUDE.md ..................... Quick reference & project map
├── 📄 UNIFIED_PIPELINE_GUIDE.md ..... Complete usage guide ← Start here for running
├── 📄 info.md ....................... Technical deep-dive
├── 📄 AGENTIC_SYSTEM_DESIGN.md ...... Design patterns & architecture
│
├── 🐍 unified_pipeline.py ........... Main orchestrator (Stage A+B+C)
├── 🐍 agentic_debugger.py ........... Stage B: Agentic debugging loop
│
├── 📂 pipe/ ......................... V3 analysis pipeline
│   ├── test.py ...................... Training orchestrator
│   ├── 📂 v3/ ....................... V3 analysis modules
│   │   ├── config.py ................ Constants & defaults
│   │   ├── graph.py ................. LangGraph state machine
│   │   ├── vision_reasoning_report_v3.py
│   │   ├── embeddings.py, rag.py, chains.py, tools.py
│   │   ├── schemas.py, renderer.py, server_utils.py
│   │   └── ... (and 3 more support modules)
│   │
│   ├── 📂 logs/ ..................... Generated training logs
│   ├── 📂 reports/ .................. Generated reports
│   └── ... (other original pipeline scripts)
│
└── 📂 data/ ......................... CIFAR-10 dataset (auto-downloaded)
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU, optional if using --no-vlm)
- Virtual environment with dependencies installed

### Dependencies (already in venv_vision/)

```bash
# Core
torch torchvision pytorch-lightning

# LLM & Reasoning
langchain langchain-core pydantic

# Vector search (for RAG)
faiss-cpu  # or faiss-gpu for GPU

# External services
vllm       # for VLM analysis (optional)
groq       # for Groq API (optional)
```

### External Services Setup

**vLLM Server (for Stage A with VLM):**
```bash
# Already installed, just start it:
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --port 8000
```

**Groq API (for Stage B):**
```bash
# 1. Sign up: https://console.groq.com
# 2. Get API key
# 3. Set environment variable
export GROQ_API_KEY="gsk_..."
```

**Ollama (for local LLM fallback):**
```bash
# Optional, for auto-fallback if Groq unavailable
ollama pull mistral
ollama serve  # runs on port 11434
```

---

## 📊 Example Outputs

### Master Pipeline Report (`master_pipeline_report.md`)

The main output file combining all stages:

```markdown
# Unified Vision AI Pipeline — Master Report

## Stage A — V3 Analysis Summary
- Best Accuracy: 68.11%
- Total Misclassified: 10,083
- Final Val Loss: 0.8987

## Stage B — Agentic Debugging Results
✅ Status: SUCCESS

### Measured Improvements (ACTUAL results)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Best Accuracy | 68.11% | 73.44% | +5.33% |
| Total Misclassified | 10,083 | 6,587 | -3,496 |
| Final Val Loss | 0.8987 | 0.7234 | -0.1753 |

### Root Cause Analysis
3 root causes identified (confidence 92%):
- Missing BatchNorm2d
- No L2 regularization
- Unregularized conv layers

### Code Changes Applied
- Add BatchNorm2d after Conv2d
- Add L2 regularization (weight_decay=1e-4)
- Add Dropout2d in backbone
```

### Other Output Files

| File | Source | Purpose |
|------|--------|---------|
| `ai_reasoning_summary_v3.md` | Stage A | V3 analysis report with findings |
| `master_pipeline_report.md` | Stage B | Main report (before/after metrics) |
| `ai_reasoning_summary_v3_fixed.md` | Stage C | Re-analysis of fixed code |
| `pipe/test_fixed.py` | Stage B | Generated fixed code |
| `pipe/logs/training_log_*.json` | Stage A | Baseline training metrics |
| `pipe/logs/training_log_fixed_*.json` | Stage B | Fixed code training metrics |

---

## 🎓 Common Workflows

### Workflow 1: Full End-to-End
```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY
```
→ Trains model → Analyzes → Diagnoses → Fixes → Validates → Reports

### Workflow 2: Debug Existing Report
```bash
python3 unified_pipeline.py --only-stage b --groq-api-key $GROQ_API_KEY
```
→ Skips training, debugs existing `ai_reasoning_summary_v3.md`

### Workflow 3: V3 Analysis Only
```bash
python3 unified_pipeline.py --only-stage a
```
→ Trains and analyzes, no debugging loop

### Workflow 4: Full + Re-validation
```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY --stage-c
```
→ Includes re-analysis on fixed code

See [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md) for 8 complete examples with detailed explanations.

---

## 🔍 Troubleshooting

**vLLM not responding?**
```bash
# Use stats-only mode
python3 unified_pipeline.py --no-vlm --groq-api-key $GROQ_API_KEY
```

**Groq rate limit exceeded?**
```bash
# Increase rate limit or wait 1 minute
python3 unified_pipeline.py --groq-rate-limit 200 --groq-api-key $GROQ_API_KEY
```

**Training execution failed?**
```bash
# Check if venv is activated and dependencies installed
pip install -r requirements.txt  # if exists
```

For 8 more troubleshooting scenarios, see [UNIFIED_PIPELINE_GUIDE.md - Troubleshooting](./UNIFIED_PIPELINE_GUIDE.md#troubleshooting).

---

## 📖 Documentation Reference

| Document | Purpose | Audience | Read time |
|----------|---------|----------|-----------|
| **CLAUDE.md** | Quick reference & project structure | Users & developers | 5 min |
| **UNIFIED_PIPELINE_GUIDE.md** | Complete usage guide with 8 examples | Users & ops | 10 min |
| **info.md** | Technical deep-dive of V3 system | Developers | 30 min |
| **AGENTIC_SYSTEM_DESIGN.md** | Design patterns & architecture | Developers & architects | 20 min |

---

## 🤝 How to Use This Project

### For First-Time Users

1. **Read** [CLAUDE.md](./CLAUDE.md) (2 min)
   - Understand the system architecture

2. **Pick a workflow** from [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md) (5 min)
   - Option 1: Full pipeline with vLLM
   - Option 2: No GPU (stats-only)
   - Option 3: Non-interactive (CI/CD)
   - Or one of 5 other examples

3. **Run the pipeline** (3-12 min depending on option)
   ```bash
   python3 unified_pipeline.py [flags]
   ```

4. **Review output** in `master_pipeline_report.md`
   - Check before/after metrics
   - Review root cause analysis
   - Examine code changes

5. **Iterate** as needed
   - Adjust flags and re-run
   - Consult [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md) for advanced options

### For Developers

1. **Architecture** → [CLAUDE.md](./CLAUDE.md) (project map)
2. **Technical details** → [info.md](./info.md) (V3 deep-dive)
3. **Design patterns** → [AGENTIC_SYSTEM_DESIGN.md](./AGENTIC_SYSTEM_DESIGN.md)
4. **Source code** → Check docstrings and inline comments

---

## 🚀 Key Features

| Feature | Benefit |
|---------|---------|
| **No code modifications** | Original pipeline unchanged, pure orchestration |
| **Real metrics only** | Uses actual training_log_*.json (not speculative) |
| **Human-in-the-loop** | Interactive approval of generated fixes |
| **LLM flexibility** | Groq + Ollama with auto-fallback |
| **Automatic iteration** | Refines diagnosis if fix doesn't work (max 3x) |
| **Comprehensive reports** | Before/after with explanation of why changes work |

---

## 📊 Performance

| Stage | Duration | GPU Required |
|-------|----------|--------------|
| **A (V3 Analysis)** | 6-10 min | Only for VLM (optional) |
| **B (Debugging)** | 2-5 min | No |
| **C (Re-analysis)** | 6-10 min | Only for VLM (optional) |
| **A+B (Total)** | 8-15 min | Only for VLM (optional) |

Use `--no-vlm` to run without GPU (~3-5 min for A+B).

---

## 🔗 Integration

This system integrates:
- **Stage A:** V3 analysis pipeline (unchanged from original)
- **Stage B:** Agentic debugging loop (new)
- **Stage C:** Optional re-analysis (new)
- **unified_pipeline.py:** Orchestrator (new)

All in a single, coherent workflow via `unified_pipeline.py`.

---

## ✅ Verification

```bash
# Verify syntax
python3 -m py_compile unified_pipeline.py
python3 -m py_compile agentic_debugger.py

# Check help
python3 unified_pipeline.py --help
```

---

## 📝 File Sizes

| File | Size | Lines |
|------|------|-------|
| unified_pipeline.py | 33.3 KB | 848 |
| agentic_debugger.py | 32.5 KB | 905 |
| CLAUDE.md | 14 KB | 376 |
| UNIFIED_PIPELINE_GUIDE.md | 18 KB | 759 |
| info.md | 18 KB | 473 |
| AGENTIC_SYSTEM_DESIGN.md | 14 KB | 435 |

---

## 🎯 Next Steps

1. **Setup:** Follow steps in [UNIFIED_PIPELINE_GUIDE.md - Installation](./UNIFIED_PIPELINE_GUIDE.md#installation--setup)
2. **Choose workflow:** Pick from 8 examples in [UNIFIED_PIPELINE_GUIDE.md - Usage Examples](./UNIFIED_PIPELINE_GUIDE.md#usage-examples)
3. **Run:** `python3 unified_pipeline.py [flags]`
4. **Review:** Check `master_pipeline_report.md` output

---

## 📞 Questions?

| Question | Answer In |
|----------|-----------|
| How do I run it? | [UNIFIED_PIPELINE_GUIDE.md](./UNIFIED_PIPELINE_GUIDE.md) |
| What does it do? | [CLAUDE.md](./CLAUDE.md) |
| How does it work? | [info.md](./info.md) |
| Why is it designed this way? | [AGENTIC_SYSTEM_DESIGN.md](./AGENTIC_SYSTEM_DESIGN.md) |
| Something broke | [UNIFIED_PIPELINE_GUIDE.md - Troubleshooting](./UNIFIED_PIPELINE_GUIDE.md#troubleshooting) |

---

**Status:** ✅ Production Ready  
**Version:** 1.0  
**Last Updated:** 2026-04-21

