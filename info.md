# V3 AI Vision Reasoning Report — Complete Technical Analysis

## Overview

V3 is a sophisticated AI vision reasoning system that analyzes misclassified images from CIFAR-10 training runs. It combines **Retrieval-Augmented Generation (RAG)**, **LangGraph agentic loops**, **Pydantic structured output**, and **dynamic LLM-generated recommendations** to provide deep insights into model blind spots.

---

## 🎯 What V3 Does

The V3 pipeline takes a trained model's failed predictions and generates a comprehensive markdown report (`ai_reasoning_summary_v3.md`) that:

1. **Analyzes per-distortion patterns**: For each image distortion type (blur, brightness, contrast, fog, etc.), examines misclassified samples
2. **Compares with historical data**: Uses a vector index (RAG) of past failures to identify novel vs known patterns
3. **Verifies hypotheses**: Runs an agentic loop that checks model hypotheses against statistical data
4. **Generates structured insights**: Produces JSON-validated analysis with root causes, typical vs outlier failures
5. **Provides actionable recommendations**: Suggests improvements based on actual failure distributions

---

## 🔄 Evolution: V1 → V2 → V3

### V1 — Basic Single-Image Analysis
- **1 image → 1 prompt → 1 response**
- Each misclassified image analyzed in isolation
- 7 static sections per image (structure hardcoded)
- No cross-image comparison
- **VLM calls**: 12 (for 4 distortion types × 3 samples)

### V2 — Multi-Image Two-Turn Reasoning
- **All samples for a type → 2-turn conversation**
- Turn 1: Withholds labels (forces visual grounding)
- Turn 2: Reveals labels, requests cross-image synthesis
- Model can compare typical vs outlier failures
- Synthesized "blind spot summary" (not pipe-joined strings)
- **VLM calls**: 8 (4 distortion types × 2 turns)

### V3 — Full Agentic Pipeline with Memory
- **Semantic deduplication**: Removes near-duplicate images before analysis
- **RAG (cross-run memory)**: Vector index of all past failures
- **LangGraph agentic loop**: 4-node state machine with verification
- **Pydantic structured output**: Validated JSON schema responses
- **Dynamic recommendations**: LLM-generated from live failure data
- **Modular architecture**: Each feature in separate v3/ module
- **VLM calls**: 8–16 (depends on hypothesis revisions)

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| Cross-image comparison | ❌ | ✅ | ✅ |
| Semantic deduplication | ❌ | ❌ | ✅ |
| RAG (cross-run memory) | ❌ | ❌ | ✅ |
| Agentic verify loop | ❌ | ❌ | ✅ |
| Stat tool calls | ❌ | ❌ | ✅ |
| Pydantic structured output | ❌ | ❌ | ✅ |
| Dynamic recommendations | ❌ | ❌ | ✅ (LLM) |
| Architecture | Monolithic | Monolithic | Modular |

---

## 🏗️ V3 Architecture

### Entry Point
**`pipe/v3/vision_reasoning_report_v3.py`**
- Thin orchestrator: wires together all v3/ modules
- Command-line argument parsing and validation
- 7-step pipeline execution

### Core Modules

#### 1. **config.py** — Configuration & Constants
```
# M3 Mac defaults (Ollama — models confirmed installed)
DEFAULT_PORT          = 11434      # Ollama server port
DEFAULT_MODEL         = "llava"    # Vision model for Stage A (llava:latest)
DEFAULT_STAGE_B_MODEL = "qwen2.5" # Text/code model for Stage B (qwen2.5:latest)

# vLLM GPU server (named constants for override)
VLLM_PORT  = 8000
VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Port 8081 OpenAI-compatible fallback
LOCAL_OPENAI_PORT         = 8081
LOCAL_OPENAI_MODEL        = "gpt-3.5-turbo"
LOCAL_OPENAI_MODEL_VISION = "llava"

DEFAULT_LOGS_DIR = "pipe/logs"
DEFAULT_RAG_INDEX = "pipe/reports/rag_index.npz"
DISTORTION_TYPES = ["blur", "jpeg", "pixelate", "noise"]
AGENT_MAX_ITERATIONS = 2
```
- All hardcoded paths, model names, thresholds
- Single source of truth for tuning

#### 2. **data_loader.py** — JSON Parsing
- `load_training_summary()`: extracts epoch, accuracy, loss from training_log_*.json
- `load_misclassified_stats()`: parses misclassified_*.json for per-distortion failure counts
- `load_distortion_report()`: reads distortion_report.json (from prior clustering step)
- `load_all_misclassified()`: loads ALL misclassified JSONs from logs/ (for RAG)
- `find_latest_file()`: auto-discovers most recent files

#### 3. **embeddings.py** — Pixel-Level Feature Extraction
- Decodes base64-encoded image data from JSON
- Uses pre-trained CNN (ResNet or similar) to extract pixel embeddings
- Returns dense vectors per image
- Used by RAG for similarity search

#### 4. **image_sampler.py** — Sampling & Deduplication
- `gather_images_for_distortion()`: samples N images per distortion type from report
- **Semantic deduplication**: computes cosine similarity between pixel embeddings
  - Removes near-duplicates (threshold: 0.96 similarity)
  - Ensures batch has maximally diverse failures
  - Example: if 3 blur samples are 99% identical pixels, keeps only 1
- Returns diverse, representative samples ready for VLM

#### 5. **rag.py** — Retrieval-Augmented Generation
```python
class RAGStore:
  embeddings        # shape: (N_images_all_runs, embedding_dim)
  index_path        # pipe/reports/rag_index.npz
  
  load_or_build()   # builds or loads from cache
  retrieve(query_embedding, k=5)  # returns top-K similar past failures
```
- Builds vector index from ALL misclassified images across all training runs
- Cached in `rag_index.npz` (built once, reused)
- Automatic staleness detection: rebuilds if new misclassified JSON found
- At analysis time: retrieves K most similar past failures for each distortion type
- Injects into Turn 2 prompt as historical precedent

#### 6. **schemas.py** — Pydantic Models
```python
class Turn2Analysis(BaseModel):
  shared_failure_pattern: str
  typical_vs_outlier: str
  what_misled_the_model: str
  confidence_assessment: str
  root_cause: str
  rag_novel_vs_known: str
```
- Enforces strict JSON structure on Turn 2 response
- Field-level validation
- Parse failures are explicit (not silent `_raw` degradation)
- Enables downstream processing without string parsing

#### 7. **tools.py** — Agentic Stat Tools
VLM-callable tools for the `verify` node:
- `query_confusion_count(label_pair)`: how many times was X misclassified as Y?
- `query_epoch_trend()`: did this distortion improve over epochs?
- `query_top_confusion_pairs()`: what are the worst label-pair confusions for this type?
- `query_misclassification_stats()`: overall failure rate by distortion

#### 8. **chains.py** — LLM Factory & Prompts
```python
build_llm(model_name, port)  # → LLMChain object
```
- Configures connection to vLLM server (OpenAI-compatible API)
- Builds Turn 1 and Turn 2 prompts with RAG context injected
- Handles prompt templating and message formatting
- Supports streaming token-by-token output

#### 9. **graph.py** — LangGraph Agentic State Machine
```
State:
  dist_type → string
  items → list of images
  
observe (Turn 1)
  ↓
hypothesise (Turn 1 response)
  ↓
verify (stat tools)
  ↓
conclude (Turn 2 response)
  
(if verify fails, loop: verify → hypothesise with tool results)
```
- **4-node agentic loop**:
  1. **observe**: sends Turn 1 (images + metadata, no labels)
  2. **hypothesise**: extracts hypothesis from Turn 1 response
  3. **verify**: calls stat tools to check consistency
  4. **conclude**: sends Turn 2 (reveal labels, inject tool results)
  
- **Revision loop**: if hypothesis inconsistent with stats, jump back to hypothesise (max 2 iterations)
- **Output**: `AnalysisState` with structured_output (Pydantic-parsed Turn 2)

#### 10. **renderer.py** — Report Generation
- `render_report()`: writes markdown with 6 sections
- `generate_dynamic_recommendations()`: calls LLM to synthesize 5 prioritized recommendations from failure data
- `_fallback_recommendations()`: data-aware fallback if LLM call fails

---

## 🔄 Full Pipeline Execution

### Step 0: Setup
```bash
# Terminal 1: Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization bitsandbytes \
  --port 8000

# Terminal 2: Run v3
python3 pipe/v3/vision_reasoning_report_v3.py
```

### Step 1: Argument Parsing
- Read CLI flags: `--samples 3 --seed 42 --no-rag` etc.
- Resolve file paths relative to pipe/

### Step 2: Discover Input Files
- Find latest `training_log_*.json` in logs/
- Find latest `misclassified_*.json` in logs/
- Find `distortion_report.json` in reports/
- Find image folder `misclassified_*_images/` (per-distortion organized images)

### Step 3: Load Data
- `training`: epoch, final accuracy, loss
- `mc_stats`: failure count per distortion type
- `dist_report`: archetype clustering info (from prior distortion_diagnostic_report.py step)

### Step 4: Build RAG Index (if `--no-rag` not set)
- Load ALL misclassified JSONs from logs/
- Extract base64 images, compute pixel embeddings
- Build faiss vector index, cache in `rag_index.npz`
- Status: ✅ RAG index ready: N vectors

### Step 5: Build LLM + Agentic Graph (if `--no-vlm` not set)
- Health-check vLLM server at `localhost:8000` (with timeout)
- Initialize LLMChain connected to vLLM
- Compile LangGraph state machine with tools

### Step 6: Per-Distortion Agent Runs
**For each distortion type (blur, brightness, ...) in DISTORTION_TYPES:**

a. **Sample & Deduplicate**
   - `gather_images_for_distortion()`: get N images from report
   - `semantic_deduplicate()`: remove near-duplicates by cosine similarity
   
b. **Run Agentic Loop** (if graph compiled)
   - Initialize `AnalysisState(dist_type=dt, items=deduplicated_images)`
   - Call `graph.run(initial_state)` → executes observe → hypothesise → verify → conclude
   - If verify fails consistency check, revises hypothesis (up to 2 iterations)
   - Returns `final_state` with:
     - `structured_output`: Pydantic-parsed Turn 2 response
     - `hypothesis_text`: raw Turn 1 hypothesis
     - `iterations`: how many revisions occurred
   
c. **Store Results**
   - `per_type_states[dt] = final_state`

### Step 7: Dynamic Recommendations
- Extract root_causes from structured outputs
- Call LLM: "Given these failure stats and root causes, generate 5 prioritized recommendations"
- If LLM call fails, fall back to data-aware ranking (by failure count)

### Step 8: Render Report
- Combine training summary, per-distortion analyses, and recommendations
- Write markdown to `ai_reasoning_summary_v3.md`

### Step 9: Log & Exit
- Print summary stats (total VLM calls, etc.)
- Status: ✅ Report generated

---

## 📋 Report Output Structure

The generated `ai_reasoning_summary_v3.md` contains:

```markdown
# AI Reasoning Report — V3

## 1. Training Summary
- Epochs, accuracy, loss
- Data split info

## 2. Failure Distribution by Distortion Type
- Table: distortion type → count → percentage

## 3. Top Confusion Pairs
- Worst misclassification pairs (e.g., "cat → dog: 45 cases")

## 4. Per-Distortion Agent Analysis
(One subsection for each distortion type: blur, brightness, contrast, etc.)

### Distortion: BLUR
- **Stats**: N failures, % of total
- **Images Analyzed** (post-dedup): sample list
- **🗃️ RAG Context**: K most similar past failures from history
- **Turn 1 — Visual Observations**: VLM's observation without labels
- **🔧 Tool Verification**: epoch trend, confusion count for this type
- **Turn 2 — Structured Analysis**:
  - 🔗 Shared Failure Pattern
  - ⭕↔⚠️ Typical vs Outlier
  - ❌ What Misled the Model
  - 📊 Confidence Assessment
  - 🎯 Root Cause
  - 🗃️ RAG — Novel vs Known Pattern

## 5. Model Blind Spot Summary
- Synthesized across all distortions
- Tool-verified insights

## 6. Actionable Recommendations
- 5 prioritized recommendations (LLM-generated from live data)
- If LLM call fails: data-aware fallback (ranked by failure count)
```

---

## 🎛️ Command-Line Flags

### Inherited from V1/V2
```
--logs-dir PATH         Logs folder (default: pipe/logs)
--report PATH           distortion_report.json path
--output PATH           Output .md path (default: ai_reasoning_summary_v3.md)
--samples N             Images per distortion type (default: 3)
--training-log PATH     Override specific training_log JSON
--misclassified PATH    Override specific misclassified JSON
--port PORT             vLLM server port (default: 8000)
--model NAME            vLLM model name
--seed N                Random seed (default: 42)
--no-vlm                Skip all VLM calls (stats + RAG only)
```

### New in V3
```
--no-rag                Skip RAG index build/load (faster cold start)
--rebuild-rag           Force rebuild RAG index even if cached
--rag-index PATH        Override path to rag_index.npz
--no-tool-trace         Hide tool verification results from report
```

---

## 🔑 Key Improvements Over V2

### 1. **Semantic Deduplication**
- **Problem**: 3 sampled images might be 99% identical pixels (redundant)
- **Solution**: Compute pixel-level embeddings, remove near-duplicates (cosine > 0.96)
- **Benefit**: Batch always contains diverse failure modes

### 2. **RAG — Cross-Run Memory**
- **Problem**: Each run analyzed in isolation; repeated patterns not recognized
- **Solution**: Build vector index of ALL past failures; retrieve K most similar
- **Benefit**: Model can say "this pattern matches 8 prior cases" or "this is novel"

### 3. **Agentic Verification Loop**
- **Problem**: VLM might hypothesize something contradicted by stats
- **Solution**: Call stat tools to verify; if inconsistent, loop back with corrective context
- **Benefit**: Reasoning is grounded in data, not hallucinated

### 4. **Pydantic Structured Output**
- **Problem**: Turn 2 response is unstructured text; hard to process downstream
- **Solution**: Use `PydanticOutputParser(Turn2Analysis)` to enforce JSON schema
- **Benefit**: Guaranteed structured fields; validation errors are explicit

### 5. **Dynamic Recommendations**
- **Problem**: Section 6 is a hardcoded table
- **Solution**: Call LLM with actual failure distribution + root causes
- **Benefit**: Recommendations are data-specific and prioritized by impact

---

## 🚀 Performance Characteristics

### VLM Calls
- **V1**: 12 calls (4 types × 3 images)
- **V2**: 8 calls (4 types × 2 turns)
- **V3**: 8–16 calls (depends on revisions + recommendations)
  - `4 types × 2 turns = 8` (base)
  - `+ up to 2 revisions per type = +up to 8`
  - `+ 1 recommendations call = +1`
  - Total: 8–17 (typically 10–12)

### Latency
- **RAG index build**: 1–2 minutes (first run only, cached)
- **Per-distortion agent run**: ~30–60 seconds (depends on image count + revisions)
- **Total end-to-end**: 5–10 minutes (with warm vLLM server)

### Memory
- **RAG embeddings**: ~50 MB (for ~1000 images)
- **vLLM model**: 7B parameters = ~14 GB (with quantization)

---

## ⚡ Troubleshooting

### RAG index takes too long
- Embedding extraction processes every base64 image in every misclassified JSON
- With thousands of failures, this can take 1–2 minutes
- **Solution**: Index is cached; subsequent runs load in <1 second
- **Alternative**: Use `--no-rag` to skip entirely

### Pydantic parse fails
- Raw VLM text is displayed verbatim if JSON parsing fails
- **Cause**: vLLM `--max-model-len` too short for combined Turn 2 prompt (RAG + tools + images)
- **Solution**: Increase `--max-model-len` from 2048 to 4096

### Agent loops twice then produces no structured output
- `verify → hypothesise` revision runs max 2 times (`AGENT_MAX_ITERATIONS=2`)
- After that, `conclude` runs with whatever hypothesis_text was produced
- **Solution**: Increase limit in `v3/config.py` if needed

### vLLM server unreachable
- Default `--port 8000`; ensure server is running in Terminal 1
- **Solution**: Pass `--port N` if using different port

### Images folder not found
- Pipeline auto-detects `misclassified_*_images/` folder
- If not found, random sampling is disabled
- **Solution**: Ensure image folder exists or re-run distortion_diagnostic_report.py

---

## 🤖 LLM Providers

### M3 Mac Setup (recommended default)

```bash
brew install ollama && ollama serve
ollama pull llava-llama3   # Stage A vision model (~5 GB)
ollama pull llama3.2:3b    # Stage B code model  (~2 GB)
```

| Model | Stage | RAM | Ollama tag |
|-------|-------|-----|-----------|
| LLaVA 1.6 | A — vision analysis | ~4.7 GB | `llava` |
| Qwen 2.5 | B — code debugging | ~4.7 GB | `qwen2.5` |

Total: ~9.4 GB on a 16 GB M3 Mac (models load on demand, not simultaneously).

### Stage A (Vision Analysis)
Uses `ChatOpenAI` (LangChain) via OpenAI-compatible API. Auto-tries three servers in order:

| Priority | Server | Port | Model | Condition |
|----------|--------|------|-------|-----------|
| 1st | Ollama (M3 default) | 11434 | `llava-llama3` | always tried first |
| 2nd | vLLM (GPU server) | 8000 | `Qwen/Qwen2.5-VL-7B-Instruct` | if 11434 not ready |
| 3rd | Port 8081 (any OpenAI-compat) | 8081 | `llava` | if 8000 not ready |
| fallback | — | — | — | skip VLM, stats-only |

Override with `--vlm-port` / `--vlm-model` flags.

### Stage B (Agentic Debugger)
`LLMClient` in `agentic_debugger.py` supports four backends. AUTO priority: GROQ → LOCAL_OPENAI → LOCAL → port 8081 last-resort.

| Provider | Format | Default Port | Default Model | How to activate |
|----------|--------|-------------|--------------|-----------------|
| `LOCAL` (Ollama) | `/api/generate` | 11434 | `llama3.2:3b` | default |
| `LOCAL_OPENAI` | `/v1/chat/completions` | configurable | `gpt-3.5-turbo` | `--local-llm-format openai` |
| `GROQ` | Groq cloud API | — | `mixtral-8x7b-32768` | `--groq-api-key KEY` |
| port 8081 (last resort) | `/v1/chat/completions` | 8081 | `gpt-3.5-turbo` | auto when others unavailable |

**Use port 8081 OpenAI-compatible server for Stage B:**
```bash
python3 unified_pipeline.py --only-stage b \
  --local-llm-port 8081 --local-llm-format openai
```

---

## 🧩 Integration Points

### Inputs (Data Dependencies)
1. **training_log_*.json** — Training metrics (epoch, accuracy, loss)
2. **misclassified_*.json** — Per-image metadata (true label, predicted, confidence, distortion)
3. **distortion_report.json** — Clustering/archetype info (from clustering step)
4. **misclassified_*_images/** — Actual image files (base64 encoded in JSON, also as files)
5. **vLLM server** — Running at localhost:PORT, OpenAI-compatible API

### Outputs (Artifacts)
1. **ai_reasoning_summary_v3.md** — Main report (markdown)
2. **rag_index.npz** — Cached vector index (built once, reused)

### Optional Inputs
- `--training-log`, `--misclassified`, `--report`: override auto-discovered files
- `--rag-index`: override cached index path

---

## 🔗 File Dependencies Graph

```
vision_reasoning_report_v3.py (entry point)
  ├── config.py (constants)
  ├── data_loader.py (load JSONs)
  ├── image_sampler.py (dedup)
  │   └── embeddings.py (pixel features)
  ├── rag.py (vector index)
  │   └── embeddings.py (pixel features)
  ├── chains.py (LLM setup)
  ├── graph.py (agentic loop)
  │   ├── tools.py (stat callables)
  │   ├── chains.py (prompt builders)
  │   └── schemas.py (Pydantic models)
  ├── renderer.py (report writing)
  │   ├── chains.py (LLM for recommendations)
  │   └── schemas.py (data structures)
  └── server_utils.py (vLLM health check)
```

---

## 📚 Related Files

- **help_v3.md**: User-facing pipeline guide (running instructions, troubleshooting)
- **ai_reasoning_summary_v3.md**: Generated report (output artifact, human-readable)
- **pipe/test.py**: CIFAR-10 training script (generates training_log_*.json)
- **pipe/distortion_diagnostic_report.py**: Clustering & archetype analysis (generates distortion_report.json)

---

## 🎓 Design Principles

1. **One responsibility per module**: Each v3/ file handles one concern
2. **Fail loudly, not silently**: Pydantic validation, explicit error messages
3. **Cache aggressively**: RAG index built once, reused; staleness auto-detected
4. **Tool-grounded reasoning**: Agentic loop verifies VLM hypotheses against data
5. **Data-driven recommendations**: Section 6 generated from actual failure patterns, not hardcoded

---

## 🔮 Future Enhancements

- Multi-turn revisions: loop beyond 2 iterations based on verification quality
- Fine-tuned embeddings: custom pixel embedding model trained on distortion-specific data
- Confidence thresholds: dynamic sampling based on analysis complexity per distortion
- Cross-distortion patterns: identify interactions between distortion types
- Temporal analysis: track failure evolution across training epochs
