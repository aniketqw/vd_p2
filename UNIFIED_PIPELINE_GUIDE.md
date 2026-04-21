# Unified Vision AI Pipeline — Complete Guide

**Status:** Production Ready  
**Version:** 1.0  
**Last Updated:** 2026-04-21

---

## 📋 Table of Contents

1. [What Is It?](#what-is-it)
2. [Quick Start](#quick-start)
3. [Installation & Setup](#installation--setup)
4. [Usage Examples](#usage-examples)
5. [Stage Breakdown](#stage-breakdown)
6. [Output Artifacts](#output-artifacts)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

---

## What Is It?

The **Unified Vision AI Pipeline** is an end-to-end system that:

1. **Trains a CNN model** on CIFAR-10 with distortion augmentations
2. **Analyzes failures** using Vision-Language Models (VLMs) and RAG
3. **Diagnoses root causes** of misclassification
4. **Automatically generates and validates fixes** using an agentic LLM loop
5. **Re-analyzes** the fixed model to measure real improvements
6. **Generates reports** with actual (not speculative) metrics

```
Training Data
    ↓
STAGE A: V3 Analysis (unchanged from original)
    ├── Train → Export failures → Cluster distortions
    ├── VLM reasoning (with RAG + LangGraph + Pydantic)
    └── ai_reasoning_summary_v3.md
         ↓
STAGE B: Agentic Debugging (NEW)
    ├── Parse findings → Map to code → Diagnose → Generate fix
    ├── Human review → Execute → Compare metrics
    └── master_pipeline_report.md (with REAL improvements)
         ↓
STAGE C: Re-Analysis (optional)
    ├── Re-run V3 on fixed code
    └── ai_reasoning_summary_v3_fixed.md
```

### Key Principles

- **No speculative metrics**: Uses actual training logs for comparison
- **Human-in-the-loop**: Fixes require explicit approval before execution
- **Real results only**: All improvements grounded in actual training runs
- **Iteration support**: Automatically refines fixes (max 3 attempts)
- **Flexible LLM backends**: Supports Groq + Ollama with auto-fallback

---

## Quick Start

### Minimal Setup (5 minutes)

```bash
# 1. Ensure venv is active
source venv_vision/bin/activate

# 2. Start vLLM server (in a separate terminal)
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000

# 3. Run the pipeline (in another terminal)
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY
```

**Result:** Generates `master_pipeline_report.md` with before/after metrics.

### No GPU? (Stats-only, no VLM)

```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY --no-vlm
```

Takes ~2 minutes. Skips VLM analysis but still runs full debug loop.

---

## Installation & Setup

### Dependencies

Already included in `venv_vision/`:
- `torch`, `torchvision` (CNN training)
- `pytorch-lightning` (training orchestration)
- `langchain`, `langchain-core` (agentic loops)
- `pydantic` (structured output validation)
- `faiss-cpu` (RAG vector search)

### External Services

**vLLM Server** (required for Stage A with VLM):
```bash
# Install vLLM (if not already installed)
pip install vllm

# Start server (runs on port 8000)
VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --enforce-eager \
  --port 8000

# Verify it's running (in another terminal)
curl http://localhost:8000/v1/models
```

**Groq API** (for Stage B LLM calls):
```bash
# 1. Sign up: https://console.groq.com
# 2. Create API key
# 3. Set environment variable
export GROQ_API_KEY="gsk_..."
```

**Ollama** (optional, for local LLM fallback):
```bash
# Download from: https://ollama.ai
# Install and start
ollama pull mistral
ollama serve  # runs on port 11434
```

---

## Usage Examples

### Example 1: Full End-to-End (Recommended)

```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY
```

**What happens:**
1. Stage A: Trains CNN, clusters failures, analyzes with VLM → `ai_reasoning_summary_v3.md`
2. Stage B: Reads report, diagnoses, generates fix, tests it → `master_pipeline_report.md`
3. (Optional) Stage C: Re-analyzes fixed code

**Output files:**
- `pipe/logs/training_log_*.json` (Stage A baseline metrics)
- `pipe/logs/training_log_fixed_*.json` (Stage B fixed code metrics)
- `ai_reasoning_summary_v3.md` (Stage A analysis)
- `master_pipeline_report.md` ← **Main report with improvements**

**Time:** ~8-12 minutes (depends on training + VLM server speed)

---

### Example 2: Skip VLM (No GPU)

```bash
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY --no-vlm
```

**Differences:**
- Stage A uses statistics only (no visual analysis)
- Still runs full agentic debugging loop
- Faster on CPU-only machines

**Time:** ~3-5 minutes

---

### Example 3: Non-Interactive (CI/Headless)

```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --auto-approve \
  --no-vlm
```

**Features:**
- `--auto-approve`: Skip human review, auto-approve fixes
- Useful for CI/CD pipelines
- Still generates full report with real metrics

---

### Example 4: Stage A Only (V3 Analysis)

```bash
python3 unified_pipeline.py --only-stage a
```

Generates `ai_reasoning_summary_v3.md` without debugging.  
Same as: `python3 pipe/test.py --version v3`

---

### Example 5: Stage B Only (Debug Existing Report)

```bash
python3 unified_pipeline.py --only-stage b --groq-api-key $GROQ_API_KEY
```

Assumes `ai_reasoning_summary_v3.md` already exists.  
Runs the full debugging loop without re-training.

---

### Example 6: Full Pipeline + Re-Analysis

```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --stage-c
```

Includes Stage C: re-runs V3 on the fixed code.  
Outputs: `ai_reasoning_summary_v3_fixed.md` for side-by-side comparison.

---

### Example 7: Tune LLM Backend

```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --local-llm-port 11434 \
  --debug-llm-provider auto
```

Options for `--debug-llm-provider`:
- `auto` (default): Use Groq if available, fallback to Ollama
- `groq`: Force Groq API only
- `local`: Force Ollama only

---

### Example 8: Custom Paths

```bash
python3 unified_pipeline.py \
  --report /path/to/custom_report.md \
  --code /path/to/custom_model.py \
  --logs-dir /path/to/logs \
  --output /path/to/output_report.md
```

---

## Stage Breakdown

### Stage A: V3 Analysis

**Entry:** `pipe/test.py --version v3`  
**Duration:** ~6-10 minutes

**Sub-steps:**
1. **A1 — Train**: SimpleCNN on CIFAR-10 with distortions → `training_log_*.json`
2. **A2 — Export**: Extract misclassified samples → `misclassified_*.json`
3. **A3 — Cluster**: t-SNE + archetypes by distortion type → `distortion_report.json`
4. **A4 — Analyze**: VLM reasoning with RAG + LangGraph → `ai_reasoning_summary_v3.md`

**Configuration** (passed via unified_pipeline.py):
```bash
--no-vlm           # Skip vLLM (stats-only)
--no-rag           # Skip RAG index
--rebuild-rag      # Force rebuild RAG
--samples 3        # Images per distortion type
--seed 42          # Random seed
```

**Output:**
- `ai_reasoning_summary_v3.md`: Markdown report with findings per distortion
- `pipe/logs/training_log_*.json`: Training metrics (epoch, accuracy, loss)
- `pipe/reports/distortion_report.json`: Clustering analysis
- `pipe/reports/rag_index.npz`: Cached RAG vector index

---

### Stage B: Agentic Debugging

**Entry:** Imported from `agentic_debugger.py`  
**Duration:** ~2-5 minutes (depends on iterations)

**Sub-steps:**

1. **B1 — Parse Report**: Extract findings from markdown
2. **B2 — Analyze Code**: Map findings to code layers (Local LLM)
3. **B3 — Diagnose**: Root cause analysis with confidence (Groq)
4. **B4 — Generate Fix**: Create fixed code with comments (Groq)
5. **B5 — Human Review**: Interactive approval gate
6. **B6 — Execute**: Run fixed code, capture new metrics
7. **B7 — Compare**: Measure improvements vs baseline
8. **B8 — Report**: Generate detailed improvement report

**Success Criteria:**
- Accuracy improvement > **3%**, OR
- Misclassified reduction > **20%**

**Iteration Behavior:**
- If fix doesn't meet criteria: retry with refined diagnosis (max 3 times)
- If execution fails: retry with different diagnosis
- If human rejects: abort gracefully

**Output:**
- `test_fixed.py`: Generated fixed code (in `pipe/` directory)
- `pipe/logs/training_log_fixed_*.json`: New training metrics
- `agentic_debug_report.json`: Detailed iteration history
- `master_pipeline_report.md`: Combined report with real improvements

---

### Stage C: Re-Analysis (Optional)

**Entry:** `run_stage_c()` in unified_pipeline.py  
**Duration:** ~6-10 minutes  
**Enabled by:** `--stage-c` flag

**Purpose:** Re-run Stage A on the fixed code to validate improvements.

**Output:**
- `ai_reasoning_summary_v3_fixed.md`: V3 analysis of fixed code
- Appended to `master_pipeline_report.md` as comparison section

**Useful for:**
- Validating that improvements are consistent
- Understanding what's still failing after fix
- Side-by-side distortion analysis before/after

---

## Output Artifacts

### Master Report: `master_pipeline_report.md`

**Contains:**
- Stage A summary (original training metrics)
- Stage B improvements (before/after comparison with real numbers)
- Stage C comparison (if enabled)
- Iteration history
- Root cause analysis
- Code changes applied
- Recommended next steps

**Example sections:**
```markdown
## Stage B — Agentic Debugging Results

### ✅ Status: SUCCESS

### Measured Improvements (ACTUAL results)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Best Accuracy | 68.11% | 73.44% | +5.33% |
| Total Misclassified | 10,083 | 6,587 | -3,496 |
| Final Val Loss | 0.8987 | 0.7234 | -0.1753 |

### Root Cause Analysis

3 root causes identified (confidence 92%):
- Missing BatchNorm2d (confidence: 95%)
- No L2 regularization (confidence: 92%)
- Unregularized conv layers (confidence: 88%)

### Code Changes Applied

- Add BatchNorm2d after Conv2d layers
- Add L2 regularization (weight_decay=1e-4) to optimizer
- Add Dropout2d in conv backbone
```

### Training Metrics: `pipe/logs/training_log_*.json`

**Format:**
```json
{
  "timestamp": "20260421_150030",
  "epochs": [
    {
      "epoch": 1,
      "overall_accuracy": 0.6511,
      "per_class_accuracy": {...},
      "misclassified_samples": [...]
    },
    ...
  ],
  "best_accuracy": 0.7344,
  "total_misclassified": 6587
}
```

### Report Metrics: `ai_reasoning_summary_v3.md`

**Key tables:**
- Distortion-wise accuracy
- Top confusion pairs per distortion
- Trend analysis (improving/regressing)
- Failure distribution

---

## Troubleshooting

### vLLM Server Issues

**Problem:** `Failed to connect to vLLM at localhost:8000`

**Solutions:**
```bash
# 1. Check if running
ps aux | grep vllm

# 2. Verify port is open
curl http://localhost:8000/v1/models

# 3. Start it manually
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.4 \
  --max-model-len 2048 \
  --port 8000

# 4. Use --no-vlm to skip VLM entirely
python3 unified_pipeline.py --no-vlm --groq-api-key $GROQ_API_KEY
```

---

### Groq Rate Limit Exceeded

**Problem:** `Groq API rate limit exceeded`

**Solutions:**
```bash
# 1. Wait a minute and retry
sleep 60
python3 unified_pipeline.py --groq-api-key $GROQ_API_KEY

# 2. Increase rate limit
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --groq-rate-limit 200

# 3. Use local Ollama instead
python3 unified_pipeline.py --debug-llm-provider local
```

---

### Training Execution Failed

**Problem:** `Code execution failed: error running fixed code`

**Solutions:**
```bash
# 1. Check if test_fixed.py was generated
ls -la pipe/test_fixed.py

# 2. Test execution manually
cd pipe && python3 test_fixed.py --version v3

# 3. Check pipe/logs permissions
ls -la pipe/logs/

# 4. Review Stage B error message for details
# The master_pipeline_report.md includes stderr output
```

---

### Pydantic Parse Failure (V3 Only)

**Problem:** V3 Turn 2 output failed Pydantic validation

**Diagnosis:**
- Usually due to vLLM context overflow
- Check if output was truncated

**Solutions:**
```bash
# 1. Increase max-model-len in vLLM
# (modify vLLM server startup command)
--max-model-len 4096

# 2. Reduce samples
python3 unified_pipeline.py --samples 2

# 3. Skip VLM entirely
python3 unified_pipeline.py --no-vlm
```

---

### RAG Index Issues

**Problem:** `RAG retrieval failed` or very slow

**Solutions:**
```bash
# 1. Skip RAG (faster cold start)
python3 unified_pipeline.py --no-rag

# 2. Force rebuild RAG
python3 unified_pipeline.py --rebuild-rag

# 3. Check if rag_index.npz is corrupt
rm pipe/reports/rag_index.npz
python3 unified_pipeline.py --rebuild-rag
```

---

### Human Review Timeout

**Problem:** Waits for user input indefinitely

**Solutions:**
```bash
# 1. Use --auto-approve to skip review
python3 unified_pipeline.py --auto-approve

# 2. Run in non-interactive mode with input piped
echo "y" | python3 unified_pipeline.py

# 3. Review manually, then run Stage B only
python3 unified_pipeline.py --only-stage b --auto-approve
```

---

### Out of Memory (OOM)

**Problem:** GPU/CPU runs out of memory

**Solutions:**
```bash
# 1. Reduce vLLM GPU utilization
--gpu-memory-utilization 0.3

# 2. Reduce batch size in training (edit pipe/test.py)
batch_size=16  # reduced from 32

# 3. Skip VLM entirely
python3 unified_pipeline.py --no-vlm

# 4. Reduce samples
python3 unified_pipeline.py --samples 1
```

---

## Advanced Configuration

### Custom LLM Providers

The system auto-detects available LLM backends:

```python
# In your code, if customizing:
from agentic_debugger import LLMClient, LLMProvider

llm = LLMClient(
    local_port = 11434,           # Ollama
    groq_api_key = "gsk_...",     # Groq
    preferred_provider = LLMProvider.AUTO,  # or GROQ, LOCAL
)
```

**Provider selection order:**
1. If `preferred_provider == AUTO`:
   - Try Groq first (if API key provided)
   - Fall back to Ollama (if available on local_port)
   - Fall back to Ollama with default prompt (no external calls)
2. If `preferred_provider == GROQ`: Force Groq, error if unavailable
3. If `preferred_provider == LOCAL`: Force Ollama, error if unavailable

---

### Environment Variables

```bash
# Groq API key
export GROQ_API_KEY="gsk_..."

# Optional: Override LLM provider
export AGENTIC_LLM_PROVIDER="groq"  # or "local" or "auto"

# Optional: Override iteration limit
export AGENTIC_MAX_ITERATIONS=2

# Optional: Override Ollama port
export AGENTIC_LOCAL_LLM_PORT=9999
```

---

### Command Chaining Examples

**Full pipeline with all customizations:**
```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --local-llm-port 11434 \
  --debug-llm-provider auto \
  --max-iterations 3 \
  --no-vlm \
  --no-rag \
  --samples 2 \
  --seed 0 \
  --stage-c \
  --output /tmp/my_report.md
```

**CI/CD-safe version:**
```bash
python3 unified_pipeline.py \
  --only-stage a \
  --no-vlm \
  --auto-approve \
  && python3 unified_pipeline.py \
  --only-stage b \
  --groq-api-key $GROQ_API_KEY \
  --auto-approve
```

---

### Testing & Validation

**Quick validation run (1 min):**
```bash
python3 unified_pipeline.py \
  --only-stage a \
  --no-vlm \
  --samples 1 \
  --seed 42
```

**Full test with fixed code (5 min):**
```bash
python3 unified_pipeline.py \
  --groq-api-key $GROQ_API_KEY \
  --no-vlm \
  --max-iterations 1 \
  --auto-approve
```

**Dry-run Stage B (no execution):**
```bash
python3 unified_pipeline.py \
  --only-stage b \
  --groq-api-key $GROQ_API_KEY \
  # Manual review will show the fix without executing
```

---

## FAQ

### Q: Can I run Stage B on a report from a different project?

**A:** Yes. Use:
```bash
python3 unified_pipeline.py \
  --only-stage b \
  --report /path/to/custom_report.md \
  --code /path/to/custom_code.py \
  --logs-dir /path/to/logs
```

The code analysis will map to your custom code, not test.py.

---

### Q: How do I skip the human review?

**A:** Use `--auto-approve`:
```bash
python3 unified_pipeline.py --auto-approve
```

**Warning:** Bypasses visual inspection. Use only in CI/CD or testing.

---

### Q: What if Stage B fails to improve metrics?

**A:** The system:
1. Retries with refined diagnosis (up to 3 iterations)
2. If still no improvement, escalates to human with full report
3. Generates "best-effort" report explaining why

Check `master_pipeline_report.md` for the escalation reason.

---

### Q: Can I run just the training part?

**A:** Yes:
```bash
cd pipe && python3 test.py
```

This trains but doesn't run any analysis or debugging.

---

### Q: How do I preserve the original test.py while testing fixes?

**A:** The system automatically generates `test_fixed.py`. Keep both:
- `test.py`: Original (baseline)
- `test_fixed.py`: Generated fix (validated)

You can manually compare or diff them.

---

## Performance Notes

| Operation | Duration | Notes |
|-----------|----------|-------|
| Training (Stage A1) | 2-3 min | CNN with 3 epochs |
| Clustering (Stage A3) | 1-2 min | t-SNE on ~1000 images |
| VLM Analysis (Stage A4) | 3-5 min | vLLM server + RAG |
| Diagnosis (Stage B3) | 30-60 sec | LLM reasoning |
| Code Generation (Stage B4) | 30-60 sec | Groq API |
| Execution (Stage B6) | 2-3 min | New training run |
| Comparison (Stage B7) | <1 sec | Metrics arithmetic |
| Report Gen (Stage B8) | 30 sec | LLM + markdown |
| Total (A+B) | 8-15 min | Varies by backend |

---

## Architecture Reference

For detailed architecture information, see:
- **`CLAUDE.md`**: Project structure and quick reference
- **`info.md`**: Complete technical deep-dive
- **`AGENTIC_SYSTEM_DESIGN.md`**: Agentic loop design patterns

---

**Status:** ✅ Production Ready  
**Last Updated:** 2026-04-21

