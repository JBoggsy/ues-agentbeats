"""Data models for the CriteriaJudge.

This module defines Pydantic models used by the CriteriaJudge for
LLM-based criterion evaluation. These models enable structured output
parsing from the evaluation LLM.

Classes:
    LLMEvaluationResult: Structured output from LLM-based criterion evaluation.

Design Notes:
    - Uses Pydantic for structured output parsing with LangChain
    - The model is designed to be used with `.with_structured_output()`
    - Fields are carefully typed to ensure valid responses
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# =============================================================================
# LLM Evaluation Result Model
# =============================================================================


class LLMEvaluationResult(BaseModel):
    """Structured output from LLM-based criterion evaluation.

    This model is used with LangChain's `.with_structured_output()` to
    parse the LLM's evaluation response into a typed object.

    Attributes:
        score: The score assigned by the LLM (0 to max_score).
        explanation: Human-readable explanation of the score.
        strengths: List of things the assistant did well.
        weaknesses: List of areas for improvement.

    Example:
        >>> result = LLMEvaluationResult(
        ...     score=7.5,
        ...     explanation="The assistant handled most tasks well but missed one urgent email.",
        ...     strengths=["Prompt responses", "Professional tone"],
        ...     weaknesses=["Missed urgent email from vendor"]
        ... )
    """

    score: float = Field(
        ...,
        ge=0,
        description="Score from 0 to the criterion's max_score",
    )
    explanation: str = Field(
        ...,
        min_length=10,
        description="Clear explanation of why this score was assigned",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Specific things the assistant did well",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific areas for improvement",
    )
