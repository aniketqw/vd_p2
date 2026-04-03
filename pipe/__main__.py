"""
Makes `pipe` a runnable package.

From the project root you can call any version of the pipeline directly:

    python3 -m pipe                        # training only
    python3 -m pipe --version v1           # train → v1 report
    python3 -m pipe --version v2           # train → v2 report
    python3 -m pipe --version v3           # train → v3 report (RAG + agent)
    python3 -m pipe --version v3 --no-vlm  # stats only, no GPU needed
"""
import sys
from pathlib import Path

# Ensure the pipe/ directory is on sys.path so that bare imports inside
# test.py (e.g. `from debug_logger import DebugLogger`) resolve correctly.
_pipe_dir = str(Path(__file__).resolve().parent)
if _pipe_dir not in sys.path:
    sys.path.insert(0, _pipe_dir)

from .test import main

if __name__ == "__main__":
    main()
