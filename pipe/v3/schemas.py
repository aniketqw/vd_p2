"""
v3/schemas.py
=============
Pydantic models for structured LLM output (improvement #5).

Why Pydantic:
  v1 and v2 parse raw text with a line-scanner. If the VLM adds an extra
  colon, capitalises differently, or skips a section, output silently degrades
  to `_raw`. Pydantic + LangChain's structured output forces JSON-schema
  compliance and makes failures loud (retry) rather than silent.

Public API
----------
Turn2Analysis        — structured output model for the two-turn analysis
Recommendation       — one prioritised recommendation
RecommendationList   — container for dynamic recommendations
get_turn2_parser()   — returns a PydanticOutputParser[Turn2Analysis]
get_rec_parser()     — returns a PydanticOutputParser[RecommendationList]
"""

from typing import List, Literal

from pydantic import BaseModel, Field


# ── Turn 2 structured output ───────────────────────────────────────────────────

class Turn2Analysis(BaseModel):
    """
    Structured output for the Turn 2 causal reasoning call.
    Every field maps to a section in the VLM response.
    """
    shared_failure_pattern: str = Field(
        description=(
            "Which visual artifact is present in ALL images and most directly "
            "explains the wrong prediction. Must reference observations from Turn 1."
        )
    )
    typical_vs_outlier: str = Field(
        description=(
            "Comparison of typical (cluster-centroid) images against the outlier. "
            "Do they fail for the same reason or a different mechanism?"
        )
    )
    what_misled_the_model: str = Field(
        description=(
            "The exact degraded region or artifact pattern that activated the "
            "wrong class. Why does it visually resemble the predicted class?"
        )
    )
    confidence_assessment: str = Field(
        description=(
            "Whether the distortion looks obvious or subtle at 32×32 resolution. "
            "Explanation of why the distortion confidence values are at the shown levels."
        )
    )
    root_cause: str = Field(
        description=(
            "ONE sentence only: the single core mechanism by which this distortion "
            "type on these images triggers the specific wrong-class prediction."
        )
    )
    rag_novel_pattern: str = Field(
        default="",
        description=(
            "Optional: whether this batch represents a NEW failure pattern "
            "not seen in retrieved historical cases, or matches a known pattern."
        )
    )


# ── Dynamic recommendations ────────────────────────────────────────────────────

class Recommendation(BaseModel):
    """One prioritised recommendation generated from the failure analysis."""
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Impact priority: HIGH | MEDIUM | LOW"
    )
    title: str = Field(
        description="Short title (5–10 words)"
    )
    description: str = Field(
        description="Concrete implementation approach (1–3 sentences)"
    )
    addresses: str = Field(
        description="Which distortion type(s) and failure count this targets"
    )
    impact_estimate: str = Field(
        description="Rough estimate of % failure reduction if applied"
    )


class RecommendationList(BaseModel):
    """Container for the full set of dynamic recommendations."""
    recommendations: List[Recommendation] = Field(
        description="Exactly 5 prioritised recommendations, ordered HIGH→LOW"
    )


# ── Parser factories ───────────────────────────────────────────────────────────

def get_turn2_parser():
    """Return a PydanticOutputParser for Turn2Analysis."""
    from langchain_core.output_parsers import PydanticOutputParser
    return PydanticOutputParser(pydantic_object=Turn2Analysis)


def get_rec_parser():
    """Return a PydanticOutputParser for RecommendationList."""
    from langchain_core.output_parsers import PydanticOutputParser
    return PydanticOutputParser(pydantic_object=RecommendationList)
