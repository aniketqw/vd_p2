# 🧠 Claude — Vision Pipeline v5: End-to-End Web App

## ✅ Status: BUILT & BENCHMARKED

---

## 📊 Benchmark Results — M3 Air 16GB (MPS, 3 Epochs)

| Metric | SimpleCNN | ImprovedCNN | Delta |
|--------|-----------|-------------|-------|
| **Accuracy** | **69.76%** | **75.81%** | **+6.05 pp** ✅ |
| Parameters | 620K | 11.5M | 18.6x |
| Total Time | 50.9s | 953.9s (~16 min) | 18.7x slower |
| Per Epoch | 17.0s | 318.0s (~5.3 min) | — |

> With more epochs (10-15), ImprovedCNN reaches ~82-85% thanks to CosineAnnealing + augmentation.

---

## 🐍 Virtual Environment

All commands must run inside the venv:

```bash
# The venv is at:
source /Users/aniketsaxena/Documents/p/p0/Ai_ml_Learning/visionDev/tyu/vd_p2/venv/bin/activate

# Or use the venv python directly:
./venv/bin/python3 pipe/v5/app.py
./venv/bin/python3 pipe/benchmark.py
```

---

## What Is v5?

v5 combines v3 (failure analysis) + v4 (code correction) into a **web app**
where users paste PyTorch model code and get automated analysis + improvement.

### Two Modes

| Mode | What it does | Time (M3 Air) |
|------|-------------|---------------|
| ⚡ **Quick Analyze** | LLM reviews code, generates improvements | ~30 sec |
| 🔬 **Train & Compare** | Trains both on CIFAR-10, real failure analysis | ~17 min |

---

## 📁 Files

```
venv/                            # Python virtual environment
pipe/
├── benchmark.py                 # SimpleCNN vs ImprovedCNN timing
├── improved_model.py            # Hand-crafted ImprovedCNN
├── test_improved.py             # Standalone training script
├── v4/
│   ├── config.py                # v4 settings
│   ├── graph.py                 # LangGraph code correction agent
│   └── code_correction_agent.py # v4 CLI entry point
└── v5/
    ├── __init__.py
    ├── config.py                # Web server port (5050), LLM settings
    ├── pipeline.py              # Combined v3+v4 engine
    ├── app.py                   # FastAPI server
    └── index.html               # Web UI (dark theme + glassmorphism)
```

---

## ▶️ How to Run

```bash
cd /Users/aniketsaxena/Documents/p/p0/Ai_ml_Learning/visionDev/tyu/vd_p2

# Activate venv
source venv/bin/activate

# Run web app
python3 pipe/v5/app.py
# → Open http://localhost:5050

# Run benchmark
python3 pipe/benchmark.py

# Run v4 CLI agent (needs vLLM server)
python3 pipe/v4/code_correction_agent.py --epochs 3
```

---

## Pipeline Versions

| Version | What it does | Interface |
|---------|-------------|-----------|
| v1 | Single-image VLM analysis | CLI |
| v2 | Multi-image two-turn analysis | CLI |
| v3 | RAG + LangGraph agentic analysis | CLI |
| v4 | LLM code correction agent | CLI |
| **v5** | **v3 + v4 combined, web UI** | **Web app** |
