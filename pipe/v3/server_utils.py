"""
v3/server_utils.py
==================
vLLM server health polling — identical to v2, isolated here as its own module.
"""
import time
import urllib.error
import urllib.request

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger_setup import logger


def wait_for_server(port: int, timeout: int = 300, poll_interval: int = 5) -> bool:
    """
    Poll http://localhost:{port} until HTTP 200 or timeout.
    Tries /health first (vLLM), then /v1/models (standard OpenAI-compatible servers).
    Prints a live countdown. Returns True if ready, False if timed out.
    """
    candidates = [
        f"http://localhost:{port}/health",    # vLLM
        f"http://localhost:{port}/v1/models", # standard OpenAI-compatible servers
    ]
    deadline = time.time() + timeout
    attempt  = 0

    print(f"\n⏳ Waiting for server on port {port} (timeout={timeout}s)…")
    while time.time() < deadline:
        attempt += 1
        for url in candidates:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"✅ Server ready (after ~{attempt * poll_interval}s)\n")
                        logger.info("Server is ready.")
                        return True
            except Exception:
                continue
        remaining = int(deadline - time.time())
        print(
            f"   not ready yet — retrying in {poll_interval}s "
            f"(~{remaining}s remaining) …",
            end="\r", flush=True,
        )
        time.sleep(poll_interval)

    print(f"\n❌ Server did not become ready within {timeout}s.")
    logger.error("Server readiness timeout.")
    return False
