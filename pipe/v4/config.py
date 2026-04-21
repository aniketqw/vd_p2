"""
v4/config.py
============
Configuration for the LangGraph code correction agent.
"""
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PIPE_DIR     = Path(__file__).resolve().parent.parent          # pipe/
PROJECT_ROOT = PIPE_DIR.parent                                 # vd_p2/

DEFAULT_REPORT     = PROJECT_ROOT / "ai_reasoning_summary_v3.md"
DEFAULT_MODEL_CODE = PIPE_DIR / "test.py"
GENERATED_SCRIPT   = PIPE_DIR / "test_generated.py"
LOGS_DIR           = PIPE_DIR / "logs"

# ── LLM settings ──────────────────────────────────────────────────────────────
# NOTE: For code generation, restart your vLLM server with --max-model-len 4096
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PORT      = 8000
LLM_TEMPERATURE   = 0.2       # slightly creative for code generation
ANALYZE_MAX_TOKENS  = 1000     # output tokens for analysis call
GENERATE_MAX_TOKENS = 2500     # output tokens for code generation call

# ── agent settings ─────────────────────────────────────────────────────────────
MAX_FIX_ITERATIONS   = 3       # max generate→validate loops
DEFAULT_TRAIN_EPOCHS = 5       # default epochs for generated scripts (quick)
TRAINING_TIMEOUT     = 3600    # max seconds for training subprocess (1 hour)
