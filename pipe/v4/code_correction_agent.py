"""
v4/code_correction_agent.py
============================
Entry point for the LangGraph Code Correction Agent.

Reads the v3 failure report, analyses the model code using an LLM,
generates an improved model, runs training, and reports results — all
automatically via a LangGraph agentic state machine.

Prerequisites
-------------
  1. ai_reasoning_summary_v3.md must exist (run v3 pipeline first)
  2. vLLM server must be running:
       VLLM_USE_V1=0 HUGGINGFACE_HUB_CACHE="/mnt/data/pratik_models" \\
       python3 -m vllm.entrypoints.openai.api_server \\
         --model Qwen/Qwen2.5-VL-7B-Instruct \\
         --quantization bitsandbytes \\
         --gpu-memory-utilization 0.4 \\
         --max-model-len 4096 \\
         --enforce-eager --port 8000

Usage
-----
  python3 pipe/v4/code_correction_agent.py
  python3 pipe/v4/code_correction_agent.py --epochs 3       # quick test
  python3 pipe/v4/code_correction_agent.py --port 8001      # custom port
  python3 pipe/v4/code_correction_agent.py --no-train       # analysis only, skip training
"""

import argparse
import sys
import time
from pathlib import Path

# ── make pipe/ and project root importable ────────────────────────────────────
_V4_DIR      = Path(__file__).resolve().parent          # pipe/v4/
_PIPE_DIR    = _V4_DIR.parent                           # pipe/
_PROJECT_ROOT = _PIPE_DIR.parent                        # vd_p2/

for _p in (str(_PROJECT_ROOT), str(_PIPE_DIR), str(_PIPE_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from v4.config import (
    DEFAULT_LLM_MODEL, DEFAULT_PORT,
    DEFAULT_REPORT, DEFAULT_MODEL_CODE, GENERATED_SCRIPT,
    DEFAULT_TRAIN_EPOCHS,
)
from v4.graph import CodeCorrectionState, build_graph


# ── server check ──────────────────────────────────────────────────────────────

def check_server(port: int, timeout: int = 120) -> bool:
    """Poll for vLLM server readiness."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    attempt = 0

    print(f"\n⏳ Checking vLLM server on port {port}...")
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"   ✅ Server ready\n")
                    return True
        except Exception:
            remaining = int(deadline - time.time())
            print(
                f"   Waiting... ({remaining}s remaining)    ",
                end="\r", flush=True,
            )
            time.sleep(5)

    print(f"\n   ❌ Server not available on port {port}.")
    return False


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "LangGraph Code Correction Agent — reads v3 failure report, "
            "analyses model code, generates improved model, trains it, and reports results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                    help="Path to ai_reasoning_summary_v3.md")
    p.add_argument("--model-code", type=Path, default=DEFAULT_MODEL_CODE,
                    help="Path to current model code (test.py)")
    p.add_argument("--output", type=Path, default=GENERATED_SCRIPT,
                    help="Path for generated training script")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="vLLM server port")
    p.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                    help="LLM model name on vLLM server")
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS,
                    help=f"Training epochs for generated script (default: {DEFAULT_TRAIN_EPOCHS})")
    p.add_argument("--no-train", action="store_true",
                    help="Generate the improved script but don't run training")
    p.add_argument("--timeout", type=int, default=120,
                    help="Server check timeout in seconds")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 70)
    print("  🧠 LangGraph Code Correction Agent — v4")
    print("  Analyze → Generate → Validate → Train → Report")
    print("═" * 70)

    # Validate inputs
    if not args.report.exists():
        sys.exit(
            f"\n❌ Report not found: {args.report}\n"
            f"   Run the v3 pipeline first:\n"
            f"     python3 pipe/v3/vision_reasoning_report_v3.py\n"
        )

    if not args.model_code.exists():
        sys.exit(f"\n❌ Model code not found: {args.model_code}\n")

    # Check server
    server_ready = check_server(args.port, timeout=args.timeout)
    if not server_ready:
        sys.exit(
            f"\n❌ vLLM server not running on port {args.port}.\n"
            f"   Start it with:\n"
            f"     VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \\\n"
            f"       --model {args.model} \\\n"
            f"       --max-model-len 4096 --port {args.port}\n"
        )

    # Print configuration
    print(f"  📄 Report:      {args.report}")
    print(f"  📝 Model code:  {args.model_code}")
    print(f"  💾 Output:      {args.output}")
    print(f"  🤖 LLM:        {args.model} (port {args.port})")
    print(f"  🔄 Epochs:      {args.epochs}")
    print(f"  🏃 Train:       {'Yes' if not args.no_train else 'No (analysis only)'}")

    # Build initial state
    initial_state = CodeCorrectionState(
        report_path=args.report.resolve(),
        model_code_path=args.model_code.resolve(),
        output_script=args.output.resolve(),
        train_epochs=args.epochs,
    )

    # If --no-train, we'll modify state after build_script to skip execution
    # (We still build the graph normally but the training node will skip)

    # Build and run the graph
    print("\n🔗 Building agent graph...")
    graph = build_graph(model_name=args.model, port=args.port)
    print("   ✅ Graph compiled: read_inputs → analyze → generate → validate → build → write → train → evaluate → report\n")

    start_time = time.time()

    if args.no_train:
        # Run only up to write_node by manually stepping through
        from v4.graph import (
            read_inputs_node, analyze_node, generate_model_node,
            validate_node, build_script_node, write_node, report_node,
            build_llm, ANALYZE_MAX_TOKENS, GENERATE_MAX_TOKENS,
            MAX_FIX_ITERATIONS, FALLBACK_MODEL_CODE,
        )

        state = initial_state
        state = read_inputs_node(state)

        analyze_llm = build_llm(args.model, args.port, ANALYZE_MAX_TOKENS)
        state = analyze_node(state, analyze_llm)

        generate_llm = build_llm(args.model, args.port, GENERATE_MAX_TOKENS)

        for _ in range(MAX_FIX_ITERATIONS):
            state = generate_model_node(state, generate_llm)
            state = validate_node(state)
            if state.code_valid:
                break
        else:
            # Fallback
            print(f"\n⚠️  Using fallback model after {MAX_FIX_ITERATIONS} failed attempts.")
            state.generated_model = FALLBACK_MODEL_CODE
            state.code_valid = True

        state = build_script_node(state)
        state = write_node(state)

        print(f"\n✅ Script generated at: {state.output_script}")
        print(f"   Run it manually with: python3 {state.output_script}")

        state = report_node(state)
    else:
        # Full graph execution
        final_state = graph.run(initial_state)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("═" * 70)


if __name__ == "__main__":
    main()
