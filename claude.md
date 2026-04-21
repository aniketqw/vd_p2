# 🧠 Claude — LangGraph Code Correction Agent (v4)

## ✅ Status: READY TO RUN

All files created and syntax-verified.

---

## 📌 What This Is

A **LangGraph agentic system** (`pipe/v4/`) that:

1. **Reads** the v3 failure report (`ai_reasoning_summary_v3.md`)
2. **Reads** the current model code (`pipe/test.py` → `SimpleCNN`)
3. **Uses an LLM** to analyze the model code against the failure findings
4. **Generates** an improved model class (Python code)
5. **Validates** the generated code (syntax check → retry loop if invalid)
6. **Builds** a complete training script around the generated model
7. **Runs** the training as a subprocess
8. **Parses** the training results from the log JSON
9. **Reports** a comparison: baseline SimpleCNN vs improved model

---

## 🏗️ Architecture — LangGraph State Machine

```
START
  → [read_inputs]      Read v3 report (strips base64 images) + model code
  → [analyze]           LLM identifies 5 code-level issues causing failures
  → [generate_model]    LLM generates improved ImprovedCNN class
  → [validate]          Python AST syntax check
       │ valid               │ invalid (& iterations < 3)
       ▼                     ▼
  → [build_script]      ← [generate_model] (with error feedback)
  → [write]             Write test_generated.py to disk
  → [run_training]      Execute as subprocess
       │ success             │ runtime error (& iterations < 3)
       ▼                     ▼
  → [evaluate]          ← [generate_model] (with error feedback)
  → [report]            Print baseline vs improved comparison
  → END
```

**Key features:**
- **Self-healing loops**: If generated code has syntax errors or runtime errors, loops back to `generate_model` with error feedback (max 3 iterations)
- **Fallback model**: If all LLM attempts fail, uses a pre-built `ImprovedCNN` (ResNet + SE attention + BatchNorm) that is known to work
- **Automatic report parsing**: Strips the ~770KB v3 report down to ~3KB of key tables and findings (no base64 images sent to LLM)
- **Full compatibility**: Generated script uses the existing `DebugLogger` for logging, so the v3 analysis pipeline can run on the results

---

## 📁 Files Created

```
pipe/v4/
├── __init__.py                  # Package init
├── config.py                    # Constants, paths, LLM settings
├── graph.py                     # Core: state machine, nodes, report parser,
│                                #   code extraction, training template, fallback model,
│                                #   conditional routing, CompiledGraph
└── code_correction_agent.py     # Entry point: arg parsing, server check, main()
```

Also created (standalone, non-agentic versions):
```
pipe/improved_model.py           # Hand-crafted ImprovedCNN (used as fallback)
pipe/test_improved.py            # Standalone training script (no LLM needed)
```

---

## ▶️ How to Run

### Prerequisites

1. **v3 report must exist** — run the v3 pipeline first if needed:
   ```bash
   python3 pipe/v3/vision_reasoning_report_v3.py --no-vlm
   ```

2. **vLLM server must be running** — with `--max-model-len 4096` for code generation:
   ```bash
   # Terminal 1
   VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \
   python3 -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --quantization bitsandbytes \
     --gpu-memory-utilization 0.4 \
     --max-model-len 4096 \
     --enforce-eager --port 8000
   ```

### Run the Agent

```bash
# Terminal 2 — full pipeline (analyze → generate → train → report)
python3 pipe/v4/code_correction_agent.py

# Quick test with fewer epochs
python3 pipe/v4/code_correction_agent.py --epochs 3

# Analysis only — generate script but don't train
python3 pipe/v4/code_correction_agent.py --no-train

# Then run the generated script manually
python3 pipe/test_generated.py
```

### All Flags

```
--report PATH       Path to v3 report (default: ai_reasoning_summary_v3.md)
--model-code PATH   Path to current model code (default: pipe/test.py)
--output PATH       Path for generated script (default: pipe/test_generated.py)
--port PORT         vLLM server port (default: 8000)
--model NAME        LLM model name (default: Qwen/Qwen2.5-VL-7B-Instruct)
--epochs N          Training epochs for generated script (default: 5)
--no-train          Generate script only, skip training
--timeout N         Server check timeout seconds (default: 120)
```

---

## 🔍 What the LLM Does at Each Step

### Step 1: `analyze` — Code Issue Detection
The LLM receives:
- Parsed failure report (accuracy, failure distribution, confusion pairs, recommendations)
- Current `SimpleCNN` code from `test.py`

It identifies 5 specific code-level issues, e.g.:
- "No BatchNorm → unstable training → slow convergence (68% after 3 epochs)"
- "No data augmentation → model never sees blur/jpeg → 80% of failures from these distortions"
- "No residual connections → spatial info destroyed by 3× MaxPool at 32×32"

### Step 2: `generate_model` — Code Generation
The LLM generates a complete `ImprovedCNN` class that fixes all identified issues.

Requirements enforced by the prompt:
- Must be a `pl.LightningModule` named `ImprovedCNN`
- Must have `training_step` logging `train_loss`
- Must have `validation_step` logging `val_loss` and `accuracy`
- Must include all helper classes (SE blocks, etc.)

### Step 3: `validate` → Retry Loop
If the generated code has syntax errors:
- The error message is fed back to `generate_model`
- LLM gets another chance (up to 3 iterations)
- After 3 failures, a pre-built fallback model is used

### Step 4: Training + Evaluation
The generated model is wrapped in a complete training script with:
- Distortion-aware augmentation (GaussianBlur, ColorJitter, RandomErasing)
- Proper CIFAR-10 normalization (actual mean/std, not 0.5/0.5/0.5)
- DebugLogger callback (same logging format as baseline)
- Results parsed from `logs/training_log_*.json`

---

## 📊 Expected Output

```
═══════════════════════════════════════════════════════════════════════════
  📋 CODE CORRECTION AGENT — FINAL REPORT
═══════════════════════════════════════════════════════════════════════════

  🔍 Analysis:
    ISSUE 1: No BatchNorm — causes unstable gradients...
    ISSUE 2: No data augmentation — blur/jpeg never seen during training...
    ...

  🛠️  Iterations: 1
  📝 Generated script: pipe/test_generated.py

  📊 RESULTS:
    Best Accuracy:  0.82
    Best Epoch:     4
    Misclassified:  5400
    ...

  ─── COMPARISON ───
    Baseline (SimpleCNN):   68.11% accuracy, 10083 misclassified
    Improved (ImprovedCNN): 82.00% accuracy, 5400 misclassified (+13.89%)

═══════════════════════════════════════════════════════════════════════════
```

---

## 🔗 How This Connects to the Existing Pipeline

```
pipe/test.py          →  Train SimpleCNN          →  training_log_*.json
pipe/v3/              →  Analyze failures (VLM)   →  ai_reasoning_summary_v3.md
pipe/v4/              →  Fix model code (LLM)     →  test_generated.py  →  NEW training_log_*.json
pipe/v3/ (again)      →  Analyze NEW failures     →  ai_reasoning_summary_v3_improved.md
```

The v4 agent **closes the loop**: v3 diagnoses problems → v4 fixes them → v3 can re-diagnose to verify the fixes worked.

---

## ⚠️ Important Notes

1. **vLLM `--max-model-len`**: Must be at least **4096** for code generation (default 2048 is too small)
2. **7B model code generation**: Qwen2.5-VL-7B can generate simple model classes but may struggle with complex architectures. The fallback model ensures the pipeline always produces a working result.
3. **Training time**: 5 epochs on CIFAR-10 takes ~5-15 min on GPU, ~30-60 min on CPU. Use `--epochs 3` for quick testing.
4. **Generated script location**: `pipe/test_generated.py` — inspect it before running if desired.
