"""
v3/config.py
============
All shared constants and default settings in one place.
Import from here — never hardcode these elsewhere.
"""
from pathlib import Path

# ── dataset constants ─────────────────────────────────────────────────────────
DISTORTION_TYPES = ["blur", "jpeg", "pixelate", "noise"]

CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird",  3: "cat",
    4: "deer",     5: "dog",        6: "frog",  7: "horse",
    8: "ship",     9: "truck",
}

# ── default paths (relative to pipe/) ────────────────────────────────────────
DEFAULT_LOGS_DIR    = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
DEFAULT_RAG_INDEX   = DEFAULT_REPORTS_DIR / "rag_index.npz"

# ── LLM settings ──────────────────────────────────────────────────────────────
DEFAULT_MODEL       = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PORT        = 8000

# Alternative: OpenAI-compatible local server (e.g. LM Studio, LocalAI, text-gen-webui)
LOCAL_OPENAI_PORT   = 8081
LOCAL_OPENAI_MODEL  = "gpt-3.5-turbo"  # model identifier accepted by the server
LLM_TEMPERATURE     = 0.1
LLM_MAX_TOKENS      = 1500        # Turn 1 + Turn 2 cover multiple images
LLM_MAX_TOKENS_REC  = 800         # Recommendations call is text-only, shorter

# ── RAG settings ──────────────────────────────────────────────────────────────
RAG_TOP_K           = 5           # How many similar past failures to retrieve
RAG_SIMILARITY_THRESHOLD = 0.80   # Min cosine similarity to include in context

# ── agent settings ────────────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS = 2          # Max hypothesis-verify loops before forcing conclude

# ── semantic dedup settings ───────────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.96 # Images above this cosine sim are considered duplicates
