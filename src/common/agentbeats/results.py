"""AgentBeats assessment results models.

This module defines Pydantic models for assessment results that the Green agent
produces after evaluating a Purple agent's performance. These results are
returned as A2A artifacts at the end of an assessment.

Models:
    - CriterionResult: Result for a single evaluation criterion
    - DimensionScore: Aggregated score for a scoring dimension
    - OverallScore: Overall assessment score
    - Scores: Container for all scores
    - ActionLogEntry: Action log entry with turn context (built from UES events)
    - AssessmentResults: Complete assessment results artifact

Scoring Dimensions:
    - accuracy: Correctness of actions and responses
    - instruction_following: Adherence to user instructions
    - efficiency: Resource usage and action economy
    - safety: Avoidance of harmful or risky actions
    - politeness: Appropriate tone and communication style

Design Note:
    All result models include a `message_type` field with a fixed literal string
    value for consistent parsing. The AssessmentResults model is designed to be
    serialized as a JSON artifact in the A2A response.

Example:
    >>> from src.common.agentbeats.results import AssessmentResults, Scores
    >>> results = AssessmentResults(
    ...     assessment_id="assess-123",
    ...     scenario_id="email_triage_basic",
    ...     participant="purple-agent-1",
    ...     status="completed",
    ...     duration_seconds=1234.5,
    ...     turns_taken=10,
    ...     actions_taken=25,
    ...     scores=scores,
    ...     criteria_results=[...],
    ...     action_log=[...]
    ... )
    >>> artifact_data = results.model_dump(mode="json")
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Scoring Dimension Type
# =============================================================================

# All valid scoring dimensions
ScoringDimension = Literal[
    "accuracy",
    "instruction_following",
    "efficiency",
    "safety",
    "politeness",
]

# List of all dimensions for validation
ALL_DIMENSIONS: list[ScoringDimension] = [
    "accuracy",
    "instruction_following",
    "efficiency",
    "safety",
    "politeness",
]


# =============================================================================
# Score Models
# =============================================================================


class DimensionScore(BaseModel):
    """Aggregated score for a single scoring dimension.

    Each dimension (accuracy, efficiency, etc.) has its own score that is
    computed by aggregating the scores of all criteria in that dimension.

    Attributes:
        message_type: Fixed identifier for this message type.
        score: The achieved score for this dimension.
        max_score: The maximum possible score for this dimension.

    Example:
        >>> dim_score = DimensionScore(score=8, max_score=10)
        >>> dim_score.percentage
        80.0
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["dimension_score"] = "dimension_score"
    score: int = Field(..., ge=0, description="Achieved score")
    max_score: int = Field(..., ge=0, description="Maximum possible score")

    @property
    def percentage(self) -> float:
        """Calculate the score as a percentage.

        Returns:
            The score as a percentage (0-100), or 0.0 if max_score is 0.
        """
        if self.max_score == 0:
            return 0.0
        return (self.score / self.max_score) * 100.0

    @model_validator(mode="after")
    def validate_score_not_exceeds_max(self) -> "DimensionScore":
        """Ensure score does not exceed max_score."""
        if self.score > self.max_score:
            raise ValueError(
                f"score ({self.score}) cannot exceed max_score ({self.max_score})"
            )
        return self


class OverallScore(BaseModel):
    """Overall assessment score aggregated across all dimensions.

    The overall score is computed by summing all dimension scores.

    Attributes:
        message_type: Fixed identifier for this message type.
        score: The total achieved score.
        max_score: The maximum possible total score.

    Example:
        >>> overall = OverallScore(score=35, max_score=50)
        >>> overall.percentage
        70.0
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["overall_score"] = "overall_score"
    score: int = Field(..., ge=0, description="Total achieved score")
    max_score: int = Field(..., ge=0, description="Maximum possible total score")

    @property
    def percentage(self) -> float:
        """Calculate the score as a percentage.

        Returns:
            The score as a percentage (0-100), or 0.0 if max_score is 0.
        """
        if self.max_score == 0:
            return 0.0
        return (self.score / self.max_score) * 100.0

    @model_validator(mode="after")
    def validate_score_not_exceeds_max(self) -> "OverallScore":
        """Ensure score does not exceed max_score."""
        if self.score > self.max_score:
            raise ValueError(
                f"score ({self.score}) cannot exceed max_score ({self.max_score})"
            )
        return self


class Scores(BaseModel):
    """Container for all assessment scores.

    Contains the overall score and individual dimension scores. The overall
    score should equal the sum of all dimension scores.

    Attributes:
        message_type: Fixed identifier for this message type.
        overall: The overall assessment score.
        dimensions: Mapping of dimension name to dimension score.

    Example:
        >>> scores = Scores(
        ...     overall=OverallScore(score=35, max_score=50),
        ...     dimensions={
        ...         "accuracy": DimensionScore(score=18, max_score=20),
        ...         "efficiency": DimensionScore(score=17, max_score=30),
        ...     }
        ... )
        >>> scores.message_type
        'scores'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["scores"] = "scores"
    overall: OverallScore = Field(..., description="Overall assessment score")
    dimensions: dict[str, DimensionScore] = Field(
        ..., description="Scores by dimension"
    )

    @model_validator(mode="after")
    def validate_overall_matches_dimensions(self) -> "Scores":
        """Validate that overall score matches sum of dimension scores."""
        total_score = sum(d.score for d in self.dimensions.values())
        total_max = sum(d.max_score for d in self.dimensions.values())

        if self.overall.score != total_score:
            raise ValueError(
                f"overall.score ({self.overall.score}) must equal sum of "
                f"dimension scores ({total_score})"
            )
        if self.overall.max_score != total_max:
            raise ValueError(
                f"overall.max_score ({self.overall.max_score}) must equal sum of "
                f"dimension max_scores ({total_max})"
            )
        return self


# =============================================================================
# Criterion Result
# =============================================================================


class CriterionResult(BaseModel):
    """Result for a single evaluation criterion.

    Each criterion belongs to a scoring dimension and contributes to that
    dimension's total score. The `details` field can carry structured
    information about individual evaluations (e.g., per-item scores).

    Attributes:
        message_type: Fixed identifier for this message type.
        criterion_id: Unique identifier for this criterion.
        name: Human-readable name of the criterion.
        dimension: The scoring dimension this criterion belongs to.
        score: The achieved score for this criterion.
        max_score: The maximum possible score for this criterion.
        explanation: Human-readable explanation of the score.
        details: Optional structured breakdown of the evaluation.

    Example:
        >>> result = CriterionResult(
        ...     criterion_id="email_politeness",
        ...     name="Email Politeness",
        ...     dimension="politeness",
        ...     score=8,
        ...     max_score=10,
        ...     explanation="Most emails were polite, but one was too curt.",
        ...     details={"per_email_scores": {"email_1": 10, "email_2": 6}}
        ... )
        >>> result.percentage
        80.0
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["criterion_result"] = "criterion_result"
    criterion_id: str = Field(..., min_length=1, description="Unique criterion ID")
    name: str = Field(..., min_length=1, description="Human-readable criterion name")
    dimension: ScoringDimension = Field(
        ..., description="Scoring dimension this criterion belongs to"
    )
    score: int = Field(..., ge=0, description="Achieved score")
    max_score: int = Field(..., ge=1, description="Maximum possible score (must be > 0)")
    explanation: str = Field(..., description="Explanation of the score")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured evaluation details"
    )

    @property
    def percentage(self) -> float:
        """Calculate the score as a percentage.

        Returns:
            The score as a percentage (0-100).
        """
        return (self.score / self.max_score) * 100.0

    @model_validator(mode="after")
    def validate_score_not_exceeds_max(self) -> "CriterionResult":
        """Ensure score does not exceed max_score."""
        if self.score > self.max_score:
            raise ValueError(
                f"score ({self.score}) cannot exceed max_score ({self.max_score})"
            )
        return self


# =============================================================================
# Action Log
# =============================================================================


class ActionLogEntry(BaseModel):
    """Action log entry for assessment results.

    The Green agent builds the action log by querying UES for executed events
    after each turn. Only events from the Purple agent (identified by agent_id)
    are included in the assessment results action log.

    Attributes:
        message_type: Fixed identifier for this message type.
        turn: The turn number when this action was executed.
        timestamp: When the action was performed (event executed_at time).
        action: Action identifier derived from event modality and data.
        parameters: Action-specific parameters from event data.
        success: Whether the action succeeded (event status).
        error_message: Error message if the event failed.

    Example:
        >>> entry = ActionLogEntry(
        ...     turn=3,
        ...     timestamp=datetime.now(tz=timezone.utc),
        ...     action="email.send",
        ...     parameters={"to": ["alice@example.com"]},
        ...     success=True
        ... )
        >>> entry.turn
        3
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["action_log_entry"] = "action_log_entry"
    turn: int = Field(..., ge=1, description="Turn number (1-indexed)")
    timestamp: datetime = Field(..., description="When the action was performed")
    action: str = Field(
        ...,
        min_length=1,
        description="Action identifier (e.g., 'email.send')",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action-specific parameters"
    )
    success: bool = Field(..., description="Whether the action succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if success=False"
    )


# =============================================================================
# Assessment Results
# =============================================================================

# Valid assessment status values
AssessmentStatus = Literal["completed", "failed", "timeout"]


class AssessmentResults(BaseModel):
    """Complete assessment results artifact.

    This is the main artifact returned by the Green agent at the end of an
    assessment. It contains all scoring information, criterion results, and
    the complete action log.

    Attributes:
        message_type: Fixed identifier for this message type.
        assessment_id: Unique identifier for this assessment run.
        scenario_id: Identifier of the scenario that was run.
        participant: Identifier of the Purple agent that was assessed.
        status: Final status of the assessment.
        duration_seconds: Total duration of the assessment in seconds.
        turns_taken: Number of turns completed.
        actions_taken: Total number of actions taken by the Purple agent.
        scores: All scoring information.
        criteria_results: Results for each evaluation criterion.
        action_log: Complete log of actions taken during the assessment.

    Example:
        >>> results = AssessmentResults(
        ...     assessment_id="assess-123",
        ...     scenario_id="email_triage_basic",
        ...     participant="purple-agent-1",
        ...     status="completed",
        ...     duration_seconds=1234.5,
        ...     turns_taken=10,
        ...     actions_taken=25,
        ...     scores=scores,
        ...     criteria_results=[criterion_result],
        ...     action_log=[action_entry]
        ... )
        >>> results.message_type
        'assessment_results'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["assessment_results"] = "assessment_results"
    assessment_id: str = Field(..., min_length=1, description="Unique assessment ID")
    scenario_id: str = Field(..., min_length=1, description="Scenario ID")
    participant: str = Field(..., min_length=1, description="Purple agent identifier")
    status: AssessmentStatus = Field(..., description="Final assessment status")
    duration_seconds: float = Field(
        ..., ge=0, description="Assessment duration in seconds"
    )
    turns_taken: int = Field(..., ge=0, description="Number of turns completed")
    actions_taken: int = Field(..., ge=0, description="Total actions taken")
    scores: Scores = Field(..., description="All scoring information")
    criteria_results: list[CriterionResult] = Field(
        ..., description="Results for each criterion"
    )
    action_log: list[ActionLogEntry] = Field(
        ..., description="Complete action log"
    )

    @model_validator(mode="after")
    def validate_actions_match_log(self) -> "AssessmentResults":
        """Validate that actions_taken matches action_log length."""
        if self.actions_taken != len(self.action_log):
            raise ValueError(
                f"actions_taken ({self.actions_taken}) must match "
                f"action_log length ({len(self.action_log)})"
            )
        return self

    @model_validator(mode="after")
    def validate_criteria_dimensions_in_scores(self) -> "AssessmentResults":
        """Validate that all criterion dimensions exist in scores.dimensions."""
        for criterion in self.criteria_results:
            if criterion.dimension not in self.scores.dimensions:
                raise ValueError(
                    f"Criterion '{criterion.criterion_id}' has dimension "
                    f"'{criterion.dimension}' which is not in scores.dimensions"
                )
        return self


# =============================================================================
# Result Parsing
# =============================================================================

# Mapping from message_type string to model class
RESULT_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "dimension_score": DimensionScore,
    "overall_score": OverallScore,
    "scores": Scores,
    "criterion_result": CriterionResult,
    "action_log_entry": ActionLogEntry,
    "assessment_results": AssessmentResults,
}


def parse_result(data: dict[str, Any]) -> BaseModel:
    """Parse a dictionary into the appropriate result model type.

    Uses the `message_type` field to determine which model to instantiate.

    Args:
        data: Dictionary containing result data, must include `message_type`.

    Returns:
        The appropriate result model instance.

    Raises:
        ValueError: If `message_type` is missing or unrecognized.

    Example:
        >>> data = {"message_type": "dimension_score", "score": 8, "max_score": 10}
        >>> result = parse_result(data)
        >>> isinstance(result, DimensionScore)
        True
    """
    message_type = data.get("message_type")
    if message_type is None:
        raise ValueError("Result data must include 'message_type' field")

    model_class = RESULT_TYPE_REGISTRY.get(message_type)
    if model_class is None:
        valid_types = ", ".join(sorted(RESULT_TYPE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown message_type '{message_type}'. Valid types: {valid_types}"
        )

    return model_class.model_validate(data)
