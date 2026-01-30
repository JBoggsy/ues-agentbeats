"""Tests for the judge_models module.

Tests cover:
- LLMEvaluationResult model validation
- Serialization and deserialization
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.green.judge_models import LLMEvaluationResult


# =============================================================================
# Tests for LLMEvaluationResult
# =============================================================================


class TestLLMEvaluationResult:
    """Tests for LLMEvaluationResult model."""

    def test_create_with_required_fields(self) -> None:
        """Should create result with required fields."""
        result = LLMEvaluationResult(
            score=8.5,
            explanation="Good performance overall.",
        )
        assert result.score == 8.5
        assert result.explanation == "Good performance overall."
        assert result.strengths == []
        assert result.weaknesses == []

    def test_create_with_all_fields(self) -> None:
        """Should create result with all fields."""
        result = LLMEvaluationResult(
            score=7.0,
            explanation="Solid work with room for improvement.",
            strengths=["Quick responses", "Professional tone"],
            weaknesses=["Missed one email", "No follow-up"],
        )
        assert result.score == 7.0
        assert result.explanation == "Solid work with room for improvement."
        assert result.strengths == ["Quick responses", "Professional tone"]
        assert result.weaknesses == ["Missed one email", "No follow-up"]

    def test_score_can_be_zero(self) -> None:
        """Score of 0 should be valid."""
        result = LLMEvaluationResult(
            score=0.0,
            explanation="No criteria met during evaluation.",
        )
        assert result.score == 0.0

    def test_score_can_be_float(self) -> None:
        """Score should support decimal values."""
        result = LLMEvaluationResult(
            score=8.75,
            explanation="Very good performance.",
        )
        assert result.score == 8.75

    def test_negative_score_rejected(self) -> None:
        """Negative score should be rejected."""
        with pytest.raises(ValidationError):
            LLMEvaluationResult(
                score=-1.0,
                explanation="Invalid score.",
            )

    def test_short_explanation_rejected(self) -> None:
        """Explanation shorter than 10 characters should be rejected."""
        with pytest.raises(ValidationError):
            LLMEvaluationResult(
                score=5.0,
                explanation="Short",  # Less than 10 chars
            )

    def test_empty_explanation_rejected(self) -> None:
        """Empty explanation should be rejected."""
        with pytest.raises(ValidationError):
            LLMEvaluationResult(
                score=5.0,
                explanation="",
            )

    def test_missing_score_rejected(self) -> None:
        """Missing score should be rejected."""
        with pytest.raises(ValidationError):
            LLMEvaluationResult(
                explanation="Valid explanation text.",
            )

    def test_missing_explanation_rejected(self) -> None:
        """Missing explanation should be rejected."""
        with pytest.raises(ValidationError):
            LLMEvaluationResult(
                score=5.0,
            )

    def test_serialization_to_dict(self) -> None:
        """Should serialize to dict correctly."""
        result = LLMEvaluationResult(
            score=8.0,
            explanation="Well done overall.",
            strengths=["Good work"],
            weaknesses=["Minor issue"],
        )
        data = result.model_dump()

        assert data["score"] == 8.0
        assert data["explanation"] == "Well done overall."
        assert data["strengths"] == ["Good work"]
        assert data["weaknesses"] == ["Minor issue"]

    def test_deserialization_from_dict(self) -> None:
        """Should deserialize from dict correctly."""
        data = {
            "score": 9.5,
            "explanation": "Excellent performance throughout.",
            "strengths": ["A", "B"],
            "weaknesses": ["C"],
        }
        result = LLMEvaluationResult.model_validate(data)

        assert result.score == 9.5
        assert result.explanation == "Excellent performance throughout."
        assert result.strengths == ["A", "B"]
        assert result.weaknesses == ["C"]

    def test_serialization_round_trip(self) -> None:
        """Should survive serialization round trip."""
        original = LLMEvaluationResult(
            score=7.25,
            explanation="Good work with some areas to improve.",
            strengths=["Fast", "Accurate"],
            weaknesses=["Verbose responses"],
        )
        data = original.model_dump()
        restored = LLMEvaluationResult.model_validate(data)

        assert original == restored

    def test_strengths_defaults_to_empty_list(self) -> None:
        """Strengths should default to empty list."""
        result = LLMEvaluationResult(
            score=5.0,
            explanation="Average performance.",
        )
        assert result.strengths == []
        assert isinstance(result.strengths, list)

    def test_weaknesses_defaults_to_empty_list(self) -> None:
        """Weaknesses should default to empty list."""
        result = LLMEvaluationResult(
            score=5.0,
            explanation="Average performance.",
        )
        assert result.weaknesses == []
        assert isinstance(result.weaknesses, list)

    def test_json_serialization(self) -> None:
        """Should serialize to JSON correctly."""
        result = LLMEvaluationResult(
            score=8.0,
            explanation="Solid performance.",
        )
        json_str = result.model_dump_json()

        assert '"score": 8.0' in json_str or '"score":8.0' in json_str
        assert "Solid performance" in json_str
