"""
v3/graph.py
===========
LangGraph-style agentic state machine for per-distortion-type analysis.

Architecture
------------
Follows LangGraph patterns (nodes, edges, conditional routing, compiled graph)
implemented in pure Python — no langgraph library required, but the API shape
mirrors it so migration is straightforward if langgraph is available.

State machine:
                    ┌──────────────────────────────┐
                    │  (loop back if revision_needed│
                    │   AND iterations < MAX)       │
                    ▼                               │
  START → [observe] → [hypothesise] → [verify] ────┘
                                         │
                                         └──(done)──→ [conclude] → END

Node responsibilities
---------------------
  observe      Turn 1: send all images, no labels, pure visual grounding.
  hypothesise  Turn 2 (first pass OR revised): form structured root cause
               hypothesis. Injects RAG context + tool results from verify.
  verify       Call stat tools to check if hypothesis is supported by data.
               Sets revision_needed=True if hypothesis contradicts the stats.
  conclude     Finalise the structured Turn2Analysis output. No new LLM call
               if hypothesise already produced valid structured output.

Public API
----------
AnalysisState                — TypedDict for the agent state
build_graph(llm, rag_store, mc_stats) → CompiledGraph
CompiledGraph.run(state)     → AnalysisState
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from .config import AGENT_MAX_ITERATIONS, CIFAR10_CLASSES
from .chains import build_turn1_message, build_turn2_message
from .tools import (
    get_distortion_stats,
    get_epoch_trend,
    get_top_confusion_for_distortion,
    query_confusion_count,
)
from .embeddings import extract_embedding
from .schemas import Turn2Analysis, get_turn2_parser, TURN2_EXAMPLE_FORMAT

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from logger_setup import logger


# ── agent state ───────────────────────────────────────────────────────────────

@dataclass
class AnalysisState:
    """All state carried between nodes for one distortion type's analysis."""
    dist_type:        str
    items:            List[Dict]

    # populated by observe
    turn1_output:     str                  = ""

    # populated/updated by hypothesise
    hypothesis_text:  str                  = ""
    structured_output: Optional[Turn2Analysis] = None

    # populated by verify
    tool_results:     List[Dict]           = field(default_factory=list)
    rag_context:      str                  = ""
    revision_needed:  bool                 = False
    revision_reason:  str                  = ""

    # loop control
    iterations:       int                  = 0

    # error tracking
    error:            Optional[str]        = None


# ── JSON extraction helper ────────────────────────────────────────────────────

def _parse_turn2(text: str, dist_type: str) -> Optional[Turn2Analysis]:
    """
    Try multiple strategies to extract a Turn2Analysis from raw LLM text.
    1. Direct JSON parse of the whole response
    2. Extract first {...} block via regex (handles prose + JSON)
    3. Validate extracted dict with Pydantic
    Returns None if all strategies fail (raw text kept in hypothesis_text).
    """
    required_keys = {"shared_failure_pattern", "typical_vs_outlier",
                     "what_misled_the_model", "confidence_assessment", "root_cause"}

    def _try_build(d: dict) -> Optional[Turn2Analysis]:
        if not required_keys.issubset(d.keys()):
            return None
        try:
            return Turn2Analysis(**{k: d[k] for k in Turn2Analysis.model_fields if k in d})
        except Exception:
            return None

    # Strategy 1: whole text is JSON
    try:
        d = json.loads(text)
        result = _try_build(d)
        if result:
            logger.info(f"  [graph:hypothesise] Pydantic parse succeeded (strategy 1).")
            return result
    except Exception:
        pass

    # Strategy 2: pull out the largest {...} block
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            d = json.loads(match.group())
            result = _try_build(d)
            if result:
                logger.info(f"  [graph:hypothesise] Pydantic parse succeeded (strategy 2).")
                return result
    except Exception:
        pass

    # Strategy 3: greedy — find outermost { ... } allowing nested braces
    try:
        start = text.index("{")
        depth, end = 0, start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        d = json.loads(text[start:end + 1])
        result = _try_build(d)
        if result:
            logger.info(f"  [graph:hypothesise] Pydantic parse succeeded (strategy 3).")
            return result
    except Exception:
        pass

    logger.warning(f"  [graph:hypothesise] All parse strategies failed for {dist_type} — keeping raw text.")
    return None


# ── nodes ──────────────────────────────────────────────────────────────────────

def observe_node(state: AnalysisState, llm, parser) -> AnalysisState:
    """
    Turn 1 — visual grounding.
    Sends all images with no labels. Populates state.turn1_output.
    """
    if not state.items:
        state.error = "No images to analyse."
        return state

    try:
        logger.info(f"  [graph:observe] {state.dist_type} — Turn 1 ({len(state.items)} images)")
        msgs = build_turn1_message(state.items, state.dist_type)
        state.turn1_output = (llm | parser).invoke(msgs).strip()
        logger.info(f"  [graph:observe] Turn 1 complete ({len(state.turn1_output)} chars)")
    except Exception as exc:
        state.error = f"observe_node failed: {exc}"
        logger.warning(f"  [graph:observe] ERROR: {exc}")

    return state


def hypothesise_node(
    state:      AnalysisState,
    llm,
    parser,
    rag_store,
) -> AnalysisState:
    """
    Turn 2 — causal reasoning with structured output.
    Injects RAG context and any tool results from the verify node.
    Populates state.structured_output (Turn2Analysis) or falls back
    to storing raw text in state.hypothesis_text.
    """
    if state.error or not state.turn1_output:
        return state

    state.iterations += 1
    logger.info(
        f"  [graph:hypothesise] {state.dist_type} — "
        f"Turn 2 iteration {state.iterations}"
    )

    # Build RAG context from query embedding of first item
    rag_ctx = state.rag_context
    if rag_store is not None and not rag_ctx and state.items:
        try:
            query_emb = extract_embedding(state.items[0]["path"])
            retrieved = rag_store.retrieve(query_emb, state.dist_type)
            rag_ctx   = rag_store.format_context(retrieved)
            state.rag_context = rag_ctx
        except Exception as exc:
            logger.warning(f"  [graph:hypothesise] RAG retrieval failed: {exc}")

    # Use example-based format instructions — smaller models echo JSON Schema back
    # instead of filling it in, so we give a concrete filled-in example instead.
    format_instructions = TURN2_EXAMPLE_FORMAT

    try:
        msgs = build_turn2_message(
            items               = state.items,
            dist_type           = state.dist_type,
            turn1_output        = state.turn1_output,
            rag_context         = rag_ctx,
            tool_results        = state.tool_results if state.tool_results else None,
            format_instructions = format_instructions,
        )
        raw_response = (llm | StrOutputParser()).invoke(msgs).strip()
        state.hypothesis_text = raw_response

        # Try to parse structured output with multiple fallback strategies
        state.structured_output = _parse_turn2(raw_response, state.dist_type)

    except Exception as exc:
        state.error = f"hypothesise_node failed: {exc}"
        logger.warning(f"  [graph:hypothesise] ERROR: {exc}")

    return state


def verify_node(
    state:    AnalysisState,
    mc_stats: Dict,
) -> AnalysisState:
    """
    Verification node — calls stat tools to check whether the hypothesis
    formed in hypothesise_node is consistent with the dataset.

    Tool calls made:
      1. get_distortion_stats      — anchors analysis in actual failure %
      2. get_epoch_trend           — checks if failures are early/late/persistent
      3. get_top_confusion_for_distortion — validates confusion-pair claims

    Sets state.revision_needed = True if a significant inconsistency is found
    (e.g. the VLM claimed a rare confusion pair but the data shows it's #1).
    """
    if state.error:
        return state

    logger.info(f"  [graph:verify] {state.dist_type} — running tool calls")

    results = []

    # Tool 1: overall distortion stats
    try:
        r1 = get_distortion_stats(state.dist_type, mc_stats)
        results.append(r1)
    except Exception as e:
        results.append({"error": str(e)})

    # Tool 2: epoch trend
    try:
        r2 = get_epoch_trend(state.dist_type, mc_stats)
        results.append(r2)
    except Exception as e:
        results.append({"error": str(e)})

    # Tool 3: top confusion pairs for this distortion
    try:
        r3 = get_top_confusion_for_distortion(state.dist_type, mc_stats, n=3)
        results.append(r3)
    except Exception as e:
        results.append({"error": str(e)})

    # Tool 4: if structured output has a root_cause referencing specific pair,
    #         verify that pair's count
    if state.structured_output and state.structured_output.root_cause:
        rc_text = state.structured_output.root_cause.lower()
        by_dist = mc_stats.get("by_distortion", {}).get(state.dist_type, {})
        top_true  = [lbl for lbl, _ in by_dist.get("top_true_labels", [])]
        top_pred  = [lbl for lbl, _ in by_dist.get("top_pred_labels", [])]
        # try to find class names mentioned in root cause
        for true_c in top_true[:3]:
            for pred_c in top_pred[:3]:
                if true_c.lower() in rc_text and pred_c.lower() in rc_text:
                    try:
                        r4 = query_confusion_count(true_c, pred_c, mc_stats)
                        results.append(r4)
                    except Exception:
                        pass
                    break

    state.tool_results = results

    # Decide if revision is needed:
    # Only request revision if this is the first pass (avoid infinite loops)
    # and if the epoch trend shows a notable inconsistency
    if state.iterations < AGENT_MAX_ITERATIONS:
        for r in results:
            interpretation = r.get("interpretation", "")
            # If >70% early failures — strong signal the VLM should know about
            if "never learned" in interpretation and state.iterations == 1:
                state.revision_needed = True
                state.revision_reason = interpretation
                logger.info(
                    f"  [graph:verify] Revision requested: {state.revision_reason[:80]}"
                )
                break
    else:
        state.revision_needed = False

    return state


def conclude_node(state: AnalysisState) -> AnalysisState:
    """
    Conclusion node — no new LLM call.
    Simply marks the analysis as final. If structured_output is already
    populated by hypothesise_node, it's kept as-is.
    Logs a summary of what was produced.
    """
    if state.error:
        logger.warning(f"  [graph:conclude] {state.dist_type} finished with error: {state.error}")
        return state

    if state.structured_output:
        logger.info(
            f"  [graph:conclude] {state.dist_type} — "
            f"structured output ready. Root cause: "
            f"{state.structured_output.root_cause[:80]}…"
        )
    else:
        logger.info(
            f"  [graph:conclude] {state.dist_type} — "
            f"raw hypothesis text only ({len(state.hypothesis_text)} chars)"
        )
    return state


# ── router ────────────────────────────────────────────────────────────────────

def _route_after_verify(state: AnalysisState) -> str:
    """
    Conditional edge after verify_node.
    Returns 'hypothesise' to loop, or 'conclude' to finish.
    """
    if state.revision_needed and state.iterations < AGENT_MAX_ITERATIONS:
        return "hypothesise"
    return "conclude"


# ── compiled graph ─────────────────────────────────────────────────────────────

class CompiledGraph:
    """
    Executable compiled graph. Mirrors LangGraph's compiled graph interface.
    Call .run(initial_state) to execute the full state machine.
    """
    def __init__(
        self,
        nodes:      Dict[str, Callable],
        edges:      Dict[str, str],
        cond_edges: Dict[str, Callable],
        entry:      str,
        finish:     str,
    ):
        self._nodes      = nodes
        self._edges      = edges       # static edges: node → next_node
        self._cond_edges = cond_edges  # conditional edges: node → router_fn
        self._entry      = entry
        self._finish     = finish

    def run(self, state: AnalysisState) -> AnalysisState:
        current = self._entry
        while current != self._finish:
            if state.error and current not in ("conclude",):
                # Short-circuit to conclude on unrecoverable error
                current = "conclude"
                continue
            node_fn = self._nodes.get(current)
            if node_fn is None:
                state.error = f"Unknown node: {current}"
                break
            state = node_fn(state)
            # Route to next node
            if current in self._cond_edges:
                current = self._cond_edges[current](state)
            elif current in self._edges:
                current = self._edges[current]
            else:
                break  # no outgoing edge → done

        # Run finish node
        finish_fn = self._nodes.get(self._finish)
        if finish_fn:
            state = finish_fn(state)

        return state


def build_graph(llm, rag_store, mc_stats: Dict) -> CompiledGraph:
    """
    Construct and compile the analysis graph for one distortion type.

    Args:
        llm       : ChatOpenAI instance
        rag_store : RAGStore instance (or None if --no-rag)
        mc_stats  : misclassified stats dict (used by verify_node tools)

    Returns:
        CompiledGraph ready for .run(initial_state)
    """
    str_parser = StrOutputParser()

    nodes = {
        "observe":     lambda s: observe_node(s, llm, str_parser),
        "hypothesise": lambda s: hypothesise_node(s, llm, str_parser, rag_store),
        "verify":      lambda s: verify_node(s, mc_stats),
        "conclude":    lambda s: conclude_node(s),
    }
    edges = {
        "observe":     "hypothesise",
        "hypothesise": "verify",
    }
    cond_edges = {
        "verify": _route_after_verify,
    }

    return CompiledGraph(
        nodes      = nodes,
        edges      = edges,
        cond_edges = cond_edges,
        entry      = "observe",
        finish     = "conclude",
    )
