"""
🔄  Unified Vision AI Pipeline
================================
Single entry-point that merges the V3 analysis pipeline with the agentic
debugging loop into one coherent, end-to-end workflow.

STAGE A — V3 Analysis  (existing, unchanged)
  Train SimpleCNN on CIFAR-10
  → Export misclassified samples
  → Cluster distortions (t-SNE + archetypes)
  → VLM reasoning with RAG + LangGraph + Pydantic
  → ai_reasoning_summary_v3.md

STAGE B — Agentic Debugging  (new, wired on top of Stage A)
  Parse ai_reasoning_summary_v3.md
  → Map findings to code lines (LLM)
  → Diagnose root causes against baseline training logs
  → Generate fixed code
  → [HUMAN REVIEW] approve / reject
  → Execute fixed code → new training_log_fixed_*.json
  → Compare metrics (before vs after)
  → Improvement report with REAL quoted results

STAGE C — Re-Analysis (optional)
  Re-run Stage A on the fixed code
  → ai_reasoning_summary_v3_fixed.md
  → Side-by-side comparison of original vs fixed reports

Usage examples
--------------
  # Full end-to-end (recommended)
  python3 unified_pipeline.py

  # Skip vLLM (stats-only V3 + debug)
  python3 unified_pipeline.py --no-vlm

  # Stage A only  (same as: python3 pipe/test.py --version v3)
  python3 unified_pipeline.py --only-stage a

  # Stage B only  (V3 report already exists)
  python3 unified_pipeline.py --only-stage b

  # Stage A + B, then re-analyse with Stage C
  python3 unified_pipeline.py --stage-c

  # Override LLM backends
  python3 unified_pipeline.py \\
      --groq-api-key $GROQ_API_KEY \\
      --local-llm-port 11434 \\
      --debug-llm-provider groq

  # Pass-through V3 flags
  python3 unified_pipeline.py --no-rag --samples 5 --seed 0

  # Non-interactive (skip human review, auto-approve)
  python3 unified_pipeline.py --auto-approve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent        # vd_p2/
_PIPE_DIR     = _PROJECT_ROOT / "pipe"                 # vd_p2/pipe/

for _p in (str(_PROJECT_ROOT), str(_PIPE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Agentic debugger lives at project root — import directly
from agentic_debugger import (
    LLMClient,
    LLMProvider,
    TrainingLogMetrics,
    load_training_log,
    parse_report,
    analyze_code,
    diagnose_root_causes,
    generate_fix,
    request_human_review,
    run_fixed_code,
    compare_metrics,
    generate_report,
    escalate_or_fallback,
    ComparisonResult,
)

# ── constants ─────────────────────────────────────────────────────────────────
_LOGS_DIR    = _PIPE_DIR / "logs"
_REPORTS_DIR = _PIPE_DIR / "reports"
_TEST_PY     = _PIPE_DIR / "test.py"


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str, char: str = "═") -> None:
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def _run_subprocess(cmd: list[str], step: str, cwd: Path) -> None:
    """Run a subprocess; exit the whole process on failure."""
    print(f"\n{'─'*70}")
    print(f"  {step}")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n❌  {step} failed (exit {result.returncode}) — aborting.")
        sys.exit(result.returncode)


def _find_latest(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_report_metrics(report_path: Path) -> dict:
    """Pull the key numbers out of the V3 markdown report for display."""
    import re
    content = report_path.read_text()
    metrics: dict = {}

    m = re.search(r"Best Accuracy[|\s]+\*?\*?([0-9.]+)%?\*?\*?", content, re.IGNORECASE)
    if m:
        metrics["best_accuracy_pct"] = float(m.group(1))

    m = re.search(r"Total Misclassified[|\s]+\*?\*?([0-9,]+)\*?\*?", content, re.IGNORECASE)
    if m:
        metrics["total_misclassified"] = int(m.group(1).replace(",", ""))

    m = re.search(r"Final Val Loss[|\s]+\*?\*?([0-9.]+)\*?\*?", content, re.IGNORECASE)
    if m:
        metrics["final_val_loss"] = float(m.group(1))

    return metrics


# ── Stage A — V3 analysis pipeline ───────────────────────────────────────────

def run_stage_a(args: argparse.Namespace, output_report: Path) -> Path:
    """
    Call pipe/test.py --version v3 as a subprocess, forwarding all V3 flags.
    Returns the path of the generated ai_reasoning_summary_v3.md.
    """
    _banner("STAGE A  —  Training + V3 Analysis", "═")
    print("  Sub-steps:")
    print("    A1  Train SimpleCNN  →  training_log_*.json + misclassified_*.json")
    print("    A2  Export misclassified samples")
    print("    A3  Cluster distortions (t-SNE + archetypes)  →  distortion_report.json")
    print("    A4  VLM reasoning (RAG + LangGraph + Pydantic)  →  ai_reasoning_summary_v3.md")

    cmd = [sys.executable, str(_TEST_PY), "--version", "v3"]

    # Forward V3 pass-through flags
    if args.no_vlm:
        cmd.append("--no-vlm")
    if args.vlm_port:
        cmd += ["--vlm-port", str(args.vlm_port)]
    if args.vlm_model:
        cmd += ["--vlm-model", args.vlm_model]
    # test.py does not accept --no-rag directly; the v3 report script does.
    # We handle extra v3 flags by delegating them through test.py extra args:
    # test.py will forward them to vision_reasoning_report_v3.py automatically
    # via the _run() helper inside run_pipeline().

    _run_subprocess(cmd, "Stage A — Full V3 pipeline", _PIPE_DIR)

    if not output_report.exists():
        sys.exit(f"❌  Stage A finished but {output_report} was not created.")

    _banner("Stage A COMPLETE", "─")
    print(f"  Report : {output_report}")

    metrics = _load_report_metrics(output_report)
    if metrics:
        print(f"  Accuracy: {metrics.get('best_accuracy_pct', '?')}%   "
              f"Misclassified: {metrics.get('total_misclassified', '?')}")

    return output_report


# ── Stage B — Agentic debugging loop ─────────────────────────────────────────

def run_stage_b(
    report_path:    Path,
    code_path:      Path,
    logs_dir:       Path,
    args:           argparse.Namespace,
) -> tuple[Path | None, object]:
    """
    Full agentic debugging: parse → analyse → diagnose → fix → [review] →
    run → compare → report.

    Returns (fixed_code_path | None, DebugReport).
    """
    _banner("STAGE B  —  Agentic Debugging", "═")
    print("  Sub-steps:")
    print("    B1  Parse ai_reasoning_summary_v3.md  →  structured findings")
    print("    B2  Map findings to code lines  (LLM)")
    print("    B3  Diagnose root causes  (Groq / Ollama)")
    print("    B4  Generate fixed code  (Groq / Ollama)")
    print("    B5  [HUMAN REVIEW] approve / reject")
    print("    B6  Execute fixed code  →  new training_log_*.json")
    print("    B7  Compare metrics (before vs after)")
    print("    B8  Generate improvement report")

    # ── LLM client ────────────────────────────────────────────────────────────
    provider_map = {
        "auto":  LLMProvider.AUTO,
        "groq":  LLMProvider.GROQ,
        "local": LLMProvider.LOCAL,
    }
    preferred = provider_map.get(args.debug_llm_provider, LLMProvider.AUTO)

    llm = LLMClient(
        local_port               = args.local_llm_port,
        groq_api_key             = args.groq_api_key,
        groq_rate_limit_per_min  = args.groq_rate_limit,
        preferred_provider       = preferred,
        local_api_format         = args.local_llm_format,
    )

    # B1 — parse report
    print("\n[B1] Parsing report …")
    parsed = parse_report(report_path)
    print(f"  ✓ {len(parsed.findings)} findings, "
          f"distortions: {', '.join(parsed.distortion_types) or 'none detected'}")

    # B2 — analyse code
    print("\n[B2] Analysing code …")
    code_analysis = analyze_code(code_path, parsed, llm)
    print(f"  ✓ {len(code_analysis.identified_layers)} layers mapped")

    # Iteration loop
    max_iter  = args.max_iterations
    iteration = 0
    comparison: ComparisonResult | None = None

    while iteration < max_iter:
        iteration += 1
        print(f"\n{'─'*60}")
        print(f"  Iteration {iteration} / {max_iter}")
        print(f"{'─'*60}")

        # B3 — diagnose
        print("\n[B3] Diagnosing root causes …")
        diagnosis = diagnose_root_causes(code_analysis, parsed, logs_dir, llm)

        if diagnosis.escalate_to_human:
            print(f"\n⚠️  Escalating to human: {diagnosis.escalation_reason}")
            return None, escalate_or_fallback(
                ComparisonResult(
                    baseline       = diagnosis.baseline_metrics,
                    after_fix      = diagnosis.baseline_metrics,
                    improvements   = {},
                    success        = False,
                    confidence     = 0.2,
                    iteration_count = iteration,
                ),
                iteration,
            )

        print(f"  ✓ {len(diagnosis.root_causes)} root causes "
              f"(confidence {diagnosis.confidence_overall:.0%})")

        # B4 — generate fix
        print("\n[B4] Generating fix …")
        fixed_code = generate_fix(diagnosis, code_analysis, llm)
        print("  ✓ Fixed code generated")

        # B5 — human review (skip if --auto-approve)
        if args.auto_approve:
            print("\n[B5] Auto-approved (--auto-approve flag set)")
            approved = True
        else:
            approved = request_human_review(fixed_code)

        if not approved:
            print("  ❌ Fix rejected by human. Stopping Stage B.")
            return None, escalate_or_fallback(
                ComparisonResult(
                    baseline       = diagnosis.baseline_metrics,
                    after_fix      = diagnosis.baseline_metrics,
                    improvements   = {},
                    success        = False,
                    confidence     = 0.0,
                    iteration_count = iteration,
                ),
                iteration,
            )

        # B6 — run fixed code
        print("\n[B6] Running fixed code …")
        exec_result = run_fixed_code(fixed_code, code_path, logs_dir)

        if not exec_result.success:
            print(f"  ❌ Execution failed: {exec_result.error_message}")
            if iteration < max_iter:
                print("  Retrying with refined diagnosis …")
                continue
            return None, escalate_or_fallback(
                ComparisonResult(
                    baseline       = diagnosis.baseline_metrics,
                    after_fix      = diagnosis.baseline_metrics,
                    improvements   = {},
                    success        = False,
                    confidence     = 0.1,
                    iteration_count = iteration,
                ),
                iteration,
            )

        fixed_code_path = Path(str(exec_result.training_log_path)).parent.parent / \
                          f"{code_path.stem}_fixed.py"

        # B7 — compare metrics
        print("\n[B7] Comparing metrics …")
        comparison = compare_metrics(
            diagnosis.baseline_metrics,
            exec_result.metrics,
            iteration,
        )

        delta_acc = comparison.improvements.get("accuracy_change", 0)
        delta_mc  = comparison.improvements.get("misclassified_change", 0)
        print(f"  Accuracy  : {comparison.baseline.best_accuracy:.2%} "
              f"→ {comparison.after_fix.best_accuracy:.2%}  ({delta_acc:+.2%})")
        print(f"  Misclassif: {comparison.baseline.total_misclassified:,} "
              f"→ {comparison.after_fix.total_misclassified:,}  ({delta_mc:+,.0f})")

        if comparison.success:
            print("\n[B8] Generating improvement report …")
            debug_report = generate_report(comparison, diagnosis, fixed_code, llm)
            print(f"\n  ✅ SUCCESS  (iteration {iteration})")
            return fixed_code_path, debug_report
        else:
            print(f"  Improvement below threshold. "
                  f"{'Iterating …' if iteration < max_iter else 'Max iterations reached.'}")

    # max iterations reached
    return None, escalate_or_fallback(comparison, iteration)


# ── Stage C — Re-analysis of fixed code ──────────────────────────────────────

def run_stage_c(
    fixed_code_path:     Path,
    args:                argparse.Namespace,
    original_report:     Path,
    fixed_report_out:    Path,
) -> Optional[Path]:
    """
    Re-run Stage A on the fixed code (test_fixed.py), producing a new V3 report.
    Returns the path of the new report, or None on failure.
    """
    _banner("STAGE C  —  Re-Analysis of Fixed Code (optional)", "═")

    if not fixed_code_path.exists():
        print(f"  ⚠️  Fixed code not found at {fixed_code_path} — skipping Stage C.")
        return None

    print(f"  Running V3 pipeline on fixed code: {fixed_code_path.name}")
    print("  This re-trains and re-analyses to measure the full improvement.\n")

    # Invoke the V3 report script directly on the fixed code's training logs.
    # The fixed code was already executed in Stage B, so training_log_fixed_*.json
    # should already exist.  We re-run distortion clustering + V3 analysis only.
    logs_dir    = _LOGS_DIR
    reports_dir = _REPORTS_DIR

    # Find the latest training log (produced by Stage B's code execution)
    latest_log = _find_latest(logs_dir, "training_log_*.json")
    if latest_log is None:
        print("  ❌ No training log found for Stage C — skipping.")
        return None

    ts       = latest_log.stem.replace("training_log_", "")
    mis_json = logs_dir / f"misclassified_{ts}.json"

    if not mis_json.exists():
        print(f"  ❌ Misclassified file {mis_json} not found — skipping Stage C.")
        return None

    # Step C1: Re-cluster
    _run_subprocess(
        [
            sys.executable, str(_PIPE_DIR / "distortion_diagnostic_report.py"),
            "--base-dir", str(mis_json),
            "--output",   str(reports_dir / "distortion_report_fixed.json"),
            "--plot",     str(reports_dir / "distortion_clusters_fixed.png"),
        ],
        "Stage C1 — Re-cluster distortions on fixed run",
        _PIPE_DIR,
    )

    # Step C2: Re-run V3 analysis
    v3_script = _PIPE_DIR / "v3" / "vision_reasoning_report_v3.py"
    cmd = [
        sys.executable, str(v3_script),
        "--logs-dir", str(logs_dir),
        "--report",   str(reports_dir / "distortion_report_fixed.json"),
        "--output",   str(fixed_report_out),
    ]
    if args.no_vlm:
        cmd.append("--no-vlm")
    if args.no_rag:
        cmd.append("--no-rag")

    _run_subprocess(cmd, "Stage C2 — V3 analysis on fixed code results", _PIPE_DIR)

    if fixed_report_out.exists():
        print(f"\n  ✅ Fixed report written: {fixed_report_out}")
        return fixed_report_out
    else:
        print("  ⚠️  Stage C produced no report file.")
        return None


# ── Final combined report ─────────────────────────────────────────────────────

def write_master_report(
    original_report:  Path,
    debug_report_obj: object,
    fixed_report:     Optional[Path],
    output_path:      Path,
    stage_c_ran:      bool,
) -> None:
    """
    Combine all three stages into a single master markdown document.
    ONLY quotes real, measured numbers — never estimates.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Unified Vision AI Pipeline — Master Report",
        f"**Generated:** {now}",
        f"**Status:** {getattr(debug_report_obj, 'status', 'UNKNOWN')}",
        "",
        "---",
        "",
        "## Pipeline Overview",
        "",
        "```",
        "STAGE A  →  Training + V3 Analysis     →  ai_reasoning_summary_v3.md",
        "STAGE B  →  Agentic Debugging           →  test_fixed.py + improvement metrics",
        "STAGE C  →  Re-Analysis (optional)      →  ai_reasoning_summary_v3_fixed.md",
        "```",
        "",
        "---",
        "",
    ]

    # ── Stage A summary ───────────────────────────────────────────────────────
    lines += [
        "## Stage A — V3 Analysis Summary",
        "",
        f"Full report: `{original_report.name}`",
        "",
    ]
    if original_report.exists():
        orig_metrics = _load_report_metrics(original_report)
        if orig_metrics:
            lines += [
                "| Metric | Value |",
                "|--------|-------|",
                f"| Best Accuracy | {orig_metrics.get('best_accuracy_pct', '?')}% |",
                f"| Total Misclassified | {orig_metrics.get('total_misclassified', '?'):,} |" if isinstance(orig_metrics.get('total_misclassified'), int) else f"| Total Misclassified | {orig_metrics.get('total_misclassified', '?')} |",
                f"| Final Val Loss | {orig_metrics.get('final_val_loss', '?')} |",
                "",
            ]
    lines += ["---", ""]

    # ── Stage B improvement results ───────────────────────────────────────────
    lines += [
        "## Stage B — Agentic Debugging Results",
        "",
    ]

    dr = debug_report_obj
    status = getattr(dr, "status", "UNKNOWN")
    status_icon = "✅" if status == "SUCCESS" else ("⚠️" if status == "PARTIAL" else "🔴")

    lines += [
        f"### {status_icon} Status: {status}",
        "",
        getattr(dr, "executive_summary", "").strip(),
        "",
    ]

    # Metrics comparison table — real numbers only
    mc = getattr(dr, "metrics_comparison", {})
    before = mc.get("before", {})
    after  = mc.get("after",  {})
    if before and after:
        b_acc  = before.get("best_accuracy",       0)
        a_acc  = after.get("best_accuracy",        0)
        b_mc   = before.get("total_misclassified", 0)
        a_mc   = after.get("total_misclassified",  0)
        b_loss = before.get("final_val_loss",      0)
        a_loss = after.get("final_val_loss",       0)

        lines += [
            "### Measured Improvements (ACTUAL results)",
            "",
            "| Metric | Before | After | Delta |",
            "|--------|--------|-------|-------|",
            f"| Best Accuracy | {b_acc:.2%} | {a_acc:.2%} | {a_acc - b_acc:+.2%} |",
            f"| Total Misclassified | {b_mc:,} | {a_mc:,} | {a_mc - b_mc:+,} |",
            f"| Final Val Loss | {b_loss:.4f} | {a_loss:.4f} | {a_loss - b_loss:+.4f} |",
            "",
        ]

        # Per-epoch breakdown (if available)
        b_epochs = before.get("epochs", [])
        a_epochs = after.get("epochs",  [])
        if b_epochs and a_epochs:
            lines += [
                "#### Epoch-by-Epoch Accuracy",
                "",
                "| Epoch | Before | After | Delta |",
                "|-------|--------|-------|-------|",
            ]
            for b_ep, a_ep in zip(b_epochs, a_epochs):
                b_a = b_ep.get("overall_accuracy") or b_ep.get("accuracy") or 0
                a_a = a_ep.get("overall_accuracy") or a_ep.get("accuracy") or 0
                ep  = b_ep.get("epoch", "?")
                lines.append(
                    f"| {ep} | {float(b_a):.2%} | {float(a_a):.2%} | {float(a_a) - float(b_a):+.2%} |"
                )
            lines.append("")

    # Root cause analysis
    rca = getattr(dr, "root_cause_analysis", "")
    if rca:
        lines += [
            "### Root Cause Analysis",
            "",
            rca.strip(),
            "",
        ]

    # Code changes
    cc = getattr(dr, "code_changes", "")
    if cc:
        lines += [
            "### Code Changes Applied",
            "",
            cc.strip(),
            "",
        ]

    # Why it's working — LLM explanation grounded in actual improvement
    why = getattr(dr, "why_its_working", "")
    if why and status == "SUCCESS":
        lines += [
            "### Why the Fix Works (LLM-explained, grounded in actual results)",
            "",
            why.strip(),
            "",
        ]

    # Confidence
    conf = getattr(dr, "confidence", 0)
    lines += [
        f"**Confidence in improvement:** {conf:.0%}",
        "",
        "---",
        "",
    ]

    # ── Stage C comparison ────────────────────────────────────────────────────
    lines += ["## Stage C — Re-Analysis Comparison", ""]

    if not stage_c_ran:
        lines += [
            "_Stage C was not run. Use `--stage-c` to enable full V3 re-analysis on the fixed code._",
            "",
        ]
    elif fixed_report is not None and fixed_report.exists():
        fixed_metrics = _load_report_metrics(fixed_report)
        orig_metrics  = _load_report_metrics(original_report) if original_report.exists() else {}

        lines += [
            f"Fixed V3 report: `{fixed_report.name}`",
            "",
            "| Metric | Original Run | Fixed Run | Delta |",
            "|--------|-------------|-----------|-------|",
        ]
        for key, label in [
            ("best_accuracy_pct",  "Best Accuracy (%)"),
            ("total_misclassified", "Total Misclassified"),
            ("final_val_loss",      "Final Val Loss"),
        ]:
            ov = orig_metrics.get(key, "?")
            fv = fixed_metrics.get(key, "?")
            try:
                delta_str = f"{float(fv) - float(ov):+.4g}"
            except Exception:
                delta_str = "?"
            lines.append(f"| {label} | {ov} | {fv} | {delta_str} |")
        lines += ["", "---", ""]
    else:
        lines += ["_Stage C ran but produced no output._", "", "---", ""]

    # ── Iteration history ─────────────────────────────────────────────────────
    ih = getattr(dr, "iteration_history", [])
    if ih:
        lines += [
            "## Iteration History",
            "",
            "| Iteration | Result |",
            "|-----------|--------|",
        ]
        for entry in ih:
            lines.append(f"| {entry.get('iteration', '?')} | {entry.get('result', '?')} |")
        lines += [""]

    # ── Next steps ────────────────────────────────────────────────────────────
    ns = getattr(dr, "next_steps", "")
    if ns:
        lines += [
            "## Recommended Next Steps",
            "",
            ns.strip(),
            "",
        ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "_Generated by Unified Vision AI Pipeline_",
        f"_Timestamp: {now}_",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\n  ✅ Master report written: {output_path}")


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Vision AI Pipeline: V3 Analysis + Agentic Debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 unified_pipeline.py                          # full pipeline
  python3 unified_pipeline.py --no-vlm                 # skip vLLM (stats-only V3 + debug)
  python3 unified_pipeline.py --only-stage a           # V3 analysis only
  python3 unified_pipeline.py --only-stage b           # debug only (V3 report must exist)
  python3 unified_pipeline.py --stage-c                # include re-analysis after fix
  python3 unified_pipeline.py --auto-approve --no-vlm  # non-interactive run
""",
    )

    # ── Stage control ──────────────────────────────────────────────────────────
    grp_stage = p.add_argument_group("Stage control")
    grp_stage.add_argument(
        "--only-stage", choices=["a", "b"], default=None, metavar="STAGE",
        help="Run only stage A or B (default: run both A and B).",
    )
    grp_stage.add_argument(
        "--stage-c", action="store_true",
        help="After Stage B succeeds, re-run V3 analysis on the fixed code.",
    )

    # ── Path overrides ─────────────────────────────────────────────────────────
    grp_path = p.add_argument_group("Path overrides")
    grp_path.add_argument(
        "--report", type=Path, default=None,
        help="Path to existing V3 report (used when --only-stage b). "
             "Default: auto-detect ai_reasoning_summary_v3.md.",
    )
    grp_path.add_argument(
        "--code", type=Path, default=None,
        help="Source code to debug (default: pipe/test.py).",
    )
    grp_path.add_argument(
        "--logs-dir", type=Path, default=None,
        help="Training logs directory (default: pipe/logs/).",
    )
    grp_path.add_argument(
        "--output", type=Path, default=None,
        help="Master report output path (default: master_pipeline_report.md).",
    )

    # ── V3 pass-through flags ──────────────────────────────────────────────────
    grp_v3 = p.add_argument_group("V3 flags (passed through to Stage A)")
    grp_v3.add_argument("--no-vlm",       action="store_true",
                        help="Skip vLLM calls in Stage A (stats-only analysis).")
    grp_v3.add_argument("--no-rag",       action="store_true",
                        help="Skip RAG index build in Stage A.")
    grp_v3.add_argument("--rebuild-rag",  action="store_true",
                        help="Force RAG index rebuild in Stage A.")
    grp_v3.add_argument("--samples",      type=int, default=3,
                        help="Images per distortion type for V3 (default: 3).")
    grp_v3.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default: 42).")
    grp_v3.add_argument("--no-tool-trace", action="store_true",
                        help="Hide tool verification results from V3 report.")
    grp_v3.add_argument("--vlm-port",  type=int, default=None,
                        help="Override VLM server port for Stage A (default: 11434 Ollama).")
    grp_v3.add_argument("--vlm-model", type=str, default=None,
                        help="Override VLM model for Stage A (default: llava-llama3).")

    # ── Debug / Stage B flags ─────────────────────────────────────────────────
    grp_dbg = p.add_argument_group("Stage B — Agentic debugger flags")
    grp_dbg.add_argument(
        "--groq-api-key", type=str, default=os.environ.get("GROQ_API_KEY"),
        help="Groq API key (default: $GROQ_API_KEY env var).",
    )
    grp_dbg.add_argument(
        "--local-llm-port", type=int, default=11434,
        help="Local LLM server port (default: 11434).",
    )
    grp_dbg.add_argument(
        "--local-llm-format",
        choices=["ollama", "openai"], default="ollama",
        help="API format for the local LLM: 'ollama' (default) or 'openai' (OpenAI-compatible, e.g. port 8081).",
    )
    grp_dbg.add_argument(
        "--groq-rate-limit", type=int, default=100,
        help="Max Groq API calls per minute (default: 100).",
    )
    grp_dbg.add_argument(
        "--debug-llm-provider", choices=["auto", "groq", "local"], default="auto",
        help="LLM provider for Stage B (default: auto).",
    )
    grp_dbg.add_argument(
        "--max-iterations", type=int, default=3,
        help="Max fix attempts in Stage B (default: 3).",
    )
    grp_dbg.add_argument(
        "--auto-approve", action="store_true",
        help="Skip human review; auto-approve generated fixes.",
    )

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Resolve paths
    logs_dir      = (args.logs_dir or _LOGS_DIR).resolve()
    code_path     = (args.code     or _TEST_PY  ).resolve()
    output_path   = (args.output   or (_PROJECT_ROOT / "master_pipeline_report.md")).resolve()
    v3_report_out = _PROJECT_ROOT / "ai_reasoning_summary_v3.md"
    v3_fixed_out  = _PROJECT_ROOT / "ai_reasoning_summary_v3_fixed.md"

    run_a = args.only_stage is None or args.only_stage == "a"
    run_b = args.only_stage is None or args.only_stage == "b"

    _banner("🔄  Unified Vision AI Pipeline", "═")
    print(f"  Stage A (V3 analysis)  : {'✓ enabled' if run_a else '✗ skipped'}")
    print(f"  Stage B (debug loop)   : {'✓ enabled' if run_b else '✗ skipped'}")
    print(f"  Stage C (re-analysis)  : {'✓ enabled' if args.stage_c else '✗ skipped'}")
    print(f"  Auto-approve           : {'yes' if args.auto_approve else 'no (interactive)'}")
    print(f"  Max iterations         : {args.max_iterations}")
    print(f"  Debug LLM provider     : {args.debug_llm_provider}")
    print()

    # ── Stage A ───────────────────────────────────────────────────────────────
    if run_a:
        run_stage_a(args, v3_report_out)
    else:
        print("[Stage A skipped]")

    # Resolve report for Stage B
    if args.report:
        report_path = args.report.resolve()
    else:
        report_path = v3_report_out

    if run_b and not report_path.exists():
        sys.exit(
            f"❌  Stage B requires a V3 report at {report_path}.\n"
            "    Run Stage A first or pass --report <path>."
        )

    # ── Stage B ───────────────────────────────────────────────────────────────
    debug_report    = None
    fixed_code_path = None

    if run_b:
        fixed_code_path, debug_report = run_stage_b(
            report_path = report_path,
            code_path   = code_path,
            logs_dir    = logs_dir,
            args        = args,
        )
    else:
        print("[Stage B skipped]")

    # ── Stage C ───────────────────────────────────────────────────────────────
    fixed_report = None
    stage_c_ran  = False

    if args.stage_c and run_b:
        if fixed_code_path is None:
            fc_candidate = code_path.parent / f"{code_path.stem}_fixed.py"
            fixed_code_path = fc_candidate if fc_candidate.exists() else None

        if fixed_code_path:
            stage_c_ran  = True
            fixed_report = run_stage_c(
                fixed_code_path  = fixed_code_path,
                args             = args,
                original_report  = report_path,
                fixed_report_out = v3_fixed_out,
            )
        else:
            print("\n[Stage C] ⚠️  No fixed code found — skipping Stage C.")

    # ── Master report ─────────────────────────────────────────────────────────
    if debug_report is not None:
        _banner("Writing master report …", "─")
        write_master_report(
            original_report  = report_path,
            debug_report_obj = debug_report,
            fixed_report     = fixed_report,
            output_path      = output_path,
            stage_c_ran      = stage_c_ran,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("✅  Unified Pipeline Complete", "═")

    if run_a:
        print(f"  Stage A report  : {v3_report_out}")
    if run_b and debug_report is not None:
        status = getattr(debug_report, "status", "?")
        print(f"  Stage B status  : {status}")
        if fixed_code_path:
            print(f"  Fixed code      : {fixed_code_path}")
    if stage_c_ran and fixed_report:
        print(f"  Stage C report  : {fixed_report}")
    print(f"  Master report   : {output_path}")
    print()


if __name__ == "__main__":
    main()
