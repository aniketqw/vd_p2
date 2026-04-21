"""
v5/config.py
============
Configuration for the v5 end-to-end web pipeline.
"""
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
V5_DIR       = Path(__file__).resolve().parent
PIPE_DIR     = V5_DIR.parent
PROJECT_ROOT = PIPE_DIR.parent
LOGS_DIR     = PIPE_DIR / "logs"
REPORTS_DIR  = PIPE_DIR / "reports"

# ── web server ────────────────────────────────────────────────────────────────
WEB_HOST = "0.0.0.0"
WEB_PORT = 5050

# ── LLM — Ollama (OpenAI-compatible API on port 11434) ───────────────────────
LLM_MODEL       = "qwen2.5:latest"
LLM_PORT        = 11434
LLM_API_BASE    = "http://localhost:11434/v1"
LLM_API_KEY     = "ollama"              # Ollama doesn't need a real key
LLM_TEMPERATURE = 0.2
ANALYZE_TOKENS   = 1200
GENERATE_TOKENS  = 2500

# ── training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS   = 3
TRAINING_TIMEOUT = 1800   # 30 min max per training run
MAX_FIX_RETRIES  = 3

# ── CIFAR-10 class names ──────────────────────────────────────────────────────
CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird",  3: "cat",   4: "deer",
    5: "dog",      6: "frog",      7: "horse", 8: "ship",  9: "truck",
}
