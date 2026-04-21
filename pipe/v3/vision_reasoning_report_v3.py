"""
AI Vision Reasoning Report Generator  —  v3
============================================
Entry point. Thin orchestrator: wires v3/ modules together and runs
the full pipeline. All logic lives in the modules under v3/.

Improvements over v2
---------------------
  ① Semantic dedup      images_sampler.semantic_deduplicate()
                        Ensures the N images per batch are maximally diverse.

  ② RAG                 v3/rag.py  RAGStore
                        Builds a vector index of ALL past failures across all
                        training runs. Retrieves similar historical cases and
                        injects them into the Turn 2 prompt.

  ③ LangGraph agentic   v3/graph.py  build_graph()
                        4-node state machine: observe → hypothesise → verify
                        → conclude. verify calls stat tools; if the hypothesis
                        is inconsistent with the data, the graph loops back to
                        hypothesise with the tool results as corrective context.

  ④ Pydantic output     v3/schemas.py  Turn2Analysis
                        Forces structured JSON output. Makes failures loud
                        (retry/fallback) rather than silently degrading to _raw.

  ⑤ Dynamic recs        v3/renderer.py  generate_dynamic_recommendations()
                        Section 6 is now an LLM call against the live failure
                        stats + root causes — not a hardcoded table.

Usage (same flags as v1/v2)
---------------------------
  python3 pipe/v3/vision_reasoning_report_v3.py
  python3 pipe/v3/vision_reasoning_report_v3.py --no-vlm
  python3 pipe/v3/vision_reasoning_report_v3.py --no-rag
  python3 pipe/v3/vision_reasoning_report_v3.py --samples 3 --seed 42
  # or via test.py:
  python3 pipe/test.py --version v3

Optional flags (new in v3)
--------------------------
  --no-rag            Skip RAG index build/load (faster cold start)
  --rebuild-rag       Force rebuild of RAG index even if cached
  --no-tool-trace     Hide tool verification results from the report
  --rag-index PATH    Override path to rag_index.npz
"""

import argparse
import sys
from pathlib import Path

# ── make pipe/ and project root importable ────────────────────────────────────
# File lives at  pipe/v3/vision_reasoning_report_v3.py
#   parent       → pipe/v3/
#   parent.parent→ pipe/          (needs to be on sys.path so `from v3.config` works)
#   parent×3     → vision_dev_project/  (needs to be on sys.path for logger_setup)
_PIPE_DIR    = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PIPE_DIR.parent
for _p in (str(PROJECT_ROOT), str(_PIPE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logger_setup import logger

from v3.config import (
    DEFAULT_LOGS_DIR, DEFAULT_REPORTS_DIR, DEFAULT_RAG_INDEX,
    DEFAULT_MODEL, DEFAULT_PORT, DISTORTION_TYPES,
    LOCAL_OPENAI_PORT, LOCAL_OPENAI_MODEL_VISION,
    VLLM_PORT, VLLM_MODEL,
)
from v3.data_loader import (
    find_latest_file,
    load_training_summary,
    load_misclassified_stats,
    load_distortion_report,
    load_all_misclassified,
)
from v3.image_sampler import gather_images_for_distortion, semantic_deduplicate
from v3.rag import RAGStore
from v3.chains import build_llm
from v3.graph import AnalysisState, build_graph
from v3.renderer import generate_dynamic_recommendations, render_report, _fallback_recommendations
from v3.server_utils import wait_for_server


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    # File lives at pipe/v3/ — defaults must anchor to pipe/, not v3/
    _pipe_dir     = Path(__file__).resolve().parent.parent   # pipe/
    _project_root = _pipe_dir.parent                         # vision_dev_project/

    p = argparse.ArgumentParser(
        description="v3 AI vision reasoning report: RAG + LangGraph + Pydantic + dynamic recs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--logs-dir",      type=Path, default=_pipe_dir / "logs")
    p.add_argument("--report",        type=Path,
                   default=_pipe_dir / "reports" / "distortion_report.json")
    p.add_argument("--output",        type=Path,
                   default=_project_root / "ai_reasoning_summary_v3.md")
    p.add_argument("--samples",       type=int,  default=3,
                   help="Images per distortion type (default: 3)")
    p.add_argument("--training-log",  type=Path, default=None)
    p.add_argument("--misclassified", type=Path, default=None)
    p.add_argument("--port",          type=int,  default=DEFAULT_PORT)
    p.add_argument("--model",         type=str,  default=DEFAULT_MODEL)
    p.add_argument("--seed",          type=int,  default=42)
    # v3-specific flags
    p.add_argument("--no-vlm",        action="store_true",
                   help="Skip all VLM calls (stats + RAG index only)")
    p.add_argument("--no-rag",        action="store_true",
                   help="Skip RAG index build/load")
    p.add_argument("--rebuild-rag",   action="store_true",
                   help="Force rebuild RAG index even if cached")
    p.add_argument("--rag-index",     type=Path, default=None,
                   help="Override path to rag_index.npz")
    p.add_argument("--no-tool-trace", action="store_true",
                   help="Hide tool verification results from the report")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    logs_dir = args.logs_dir.resolve()

    print("\n" + "═" * 60)
    print("  AI Vision Reasoning Report  —  v3")
    print("  RAG + LangGraph + Pydantic + Dynamic Recommendations")
    print("═" * 60 + "\n")

    # ── 1. Discover input files ────────────────────────────────────────────────
    training_log_path = args.training_log or find_latest_file(logs_dir, "training_log_*.json")
    if training_log_path is None or not training_log_path.exists():
        sys.exit(f"ERROR: No training_log_*.json found in {logs_dir}.")
    logger.info(f"Training log  : {training_log_path}")

    mc_path = args.misclassified or find_latest_file(logs_dir, "misclassified_*.json")
    if mc_path is None or not mc_path.exists():
        sys.exit(f"ERROR: No misclassified_*.json found in {logs_dir}.")
    logger.info(f"Misclassified : {mc_path}")

    report_path = args.report.resolve()
    if not report_path.exists():
        sys.exit(f"ERROR: distortion_report.json not found at {report_path}.")
    logger.info(f"Distortion report: {report_path}")

    images_dir = mc_path.parent / f"{mc_path.stem}_images"
    if not images_dir.exists():
        logger.warning(f"Images folder not found: {images_dir} — random sampling disabled.")
        images_dir = None

    # ── 2. Load data ───────────────────────────────────────────────────────────
    logger.info("Loading training summary…")
    training = load_training_summary(training_log_path)

    logger.info("Loading misclassified stats…")
    mc_stats = load_misclassified_stats(mc_path)

    logger.info("Loading distortion report…")
    dist_report = load_distortion_report(report_path)

    # ── 3. Build RAG index ────────────────────────────────────────────────────
    rag_store = None
    if not args.no_rag:
        logger.info("Building / loading RAG index…")
        print("🗃️  RAG: indexing all training runs for cross-run failure memory…")
        try:
            all_mc = load_all_misclassified(logs_dir)
            rag_index_path = (args.rag_index or DEFAULT_RAG_INDEX).resolve()
            rag_store = RAGStore.load_or_build(
                index_path    = rag_index_path,
                all_mc_stats  = all_mc,
                force_rebuild = args.rebuild_rag,
            )
            print(f"   ✅ RAG index ready: {rag_store.embeddings.shape[0]} vectors\n")
        except Exception as exc:
            logger.warning(f"RAG setup failed: {exc} — continuing without RAG.")
            print(f"   ⚠️ RAG unavailable: {exc}\n")

    # ── 4. Build LLM + agentic graph ──────────────────────────────────────────
    llm   = None
    graph = None
    if not args.no_vlm:
        active_port  = args.port   # default: 11434 (Ollama)
        active_model = args.model  # default: llava-llama3

        # 1st try: configured port (Ollama 11434 by default)
        ready = wait_for_server(active_port, timeout=15, poll_interval=3)

        # 2nd try: vLLM port 8000 (GPU server)
        if not ready and active_port != VLLM_PORT:
            print(f"⚠️  Port {active_port} not ready — trying vLLM port {VLLM_PORT}…")
            ready = wait_for_server(VLLM_PORT, timeout=15, poll_interval=3)
            if ready:
                active_port  = VLLM_PORT
                active_model = VLLM_MODEL

        # 3rd try: port 8081 OpenAI-compatible server (llava)
        if not ready and active_port != LOCAL_OPENAI_PORT:
            print(f"⚠️  Port {VLLM_PORT} not ready — trying fallback port {LOCAL_OPENAI_PORT} (llava)…")
            ready = wait_for_server(LOCAL_OPENAI_PORT, timeout=10, poll_interval=2)
            if ready:
                active_port  = LOCAL_OPENAI_PORT
                active_model = LOCAL_OPENAI_MODEL_VISION

        if not ready:
            print("⚠️  Continuing without VLM — analyses will be skipped.\n")
        else:
            logger.info(f"Building LLM → http://localhost:{active_port}/v1  model={active_model}")
            llm   = build_llm(active_model, active_port)
            graph = build_graph(llm, rag_store, mc_stats)
            print("✅ Agent graph compiled (observe→hypothesise→verify→conclude)\n")

    # ── 5. Per-distortion agent runs ───────────────────────────────────────────
    per_type_states: dict = {}
    total_vlm_calls = 0

    for dt in DISTORTION_TYPES:
        print(f"{'─'*50}\n🔍 Distortion: {dt.upper()}")
        logger.info(f"Distortion: {dt.upper()}")

        # Sample images
        items = gather_images_for_distortion(
            dist_type  = dt,
            report     = dist_report,
            mc_stats   = mc_stats,
            images_dir = images_dir,
            n_samples  = args.samples,
            seed       = args.seed,
        )

        # Semantic deduplication (improvement ①)
        items = semantic_deduplicate(items)
        logger.info(f"  {len(items)} image(s) after dedup.")

        initial_state = AnalysisState(dist_type=dt, items=items)

        if graph is not None and items:
            final_state = graph.run(initial_state)
            # Count VLM calls: 2 per iteration (Turn1 + Turn2) + possible revision
            total_vlm_calls += 1 + final_state.iterations  # observe=1, hypothesise=iterations
        else:
            final_state = initial_state

        per_type_states[dt] = final_state

    logger.info(f"\nTotal VLM calls: {total_vlm_calls}")
    print(f"\n✅ Agent runs complete. Total VLM calls: {total_vlm_calls}")

    # ── 6. Dynamic recommendations ─────────────────────────────────────────────
    root_causes = {}
    for dt, state in per_type_states.items():
        if state.structured_output:
            root_causes[dt] = state.structured_output.root_cause
        elif state.hypothesis_text:
            root_causes[dt] = state.hypothesis_text[:200]

    if llm is not None:
        print("\n📝 Generating dynamic recommendations from live failure data…")
        recommendations = generate_dynamic_recommendations(
            llm         = llm,
            mc_stats    = mc_stats,
            root_causes = root_causes,
            model_name  = args.model,
            port        = args.port,
        )
    else:
        recommendations = _fallback_recommendations(mc_stats)
        print("   (Using data-aware fallback recommendations — no VLM)")

    # ── 7. Render report ────────────────────────────────────────────────────────
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_report(
        training        = training,
        mc_stats        = mc_stats,
        per_type_states = per_type_states,
        recommendations = recommendations,
        output_path     = output_path,
        tool_trace      = not args.no_tool_trace,
        logs_dir        = logs_dir,
        report_path     = args.report,
    )


if __name__ == "__main__":
    main()
