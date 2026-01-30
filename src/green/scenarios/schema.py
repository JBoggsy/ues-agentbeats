"""Scenario schema definitions for the Green Agent.

This module defines Pydantic models for scenario configuration. Scenarios
describe the assessment environment including characters, timing, initial
state, and evaluation criteria.

Models:
    ResponseTiming: Timing configuration for character responses
    CharacterProfile: Profile for a simulated character
    EvaluationCriterion: Single evaluation criterion definition
    ScenarioConfig: Complete scenario configuration

Design Note:
    All models use Pydantic v2 with strict validation. Time durations use
    ISO 8601 format (e.g., "PT1H" for 1 hour, "PT30M" for 30 minutes).
    Datetime fields must be timezone-aware.

Example:
    >>> from src.green.scenarios.schema import ScenarioConfig
    >>> config = ScenarioConfig.model_validate(json_data)
    >>> print(config.scenario_id)
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


if TYPE_CHECKING:
    # Avoid circular import - these types are only needed for type hints
    from ues.client import AsyncUESClient

    # Forward reference for ActionLogEntry from results module
    from src.common.agentbeats.results import ActionLogEntry


# =============================================================================
# Constants
# =============================================================================

# ISO 8601 duration regex pattern (simplified for common cases)
# Matches: P[n]Y[n]M[n]DT[n]H[n]M[n]S, PT[n]H[n]M[n]S, P[n]D, etc.
ISO_8601_DURATION_PATTERN = re.compile(
    r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)

# Valid scoring dimensions (matches results.py)
ScoringDimension = Literal[
    "accuracy",
    "instruction_following",
    "efficiency",
    "safety",
    "politeness",
]


# =============================================================================
# Evaluator Types
# =============================================================================


@dataclass
class EvalResult:
    """Result returned by an evaluator function.

    Follows UES agent_testing pattern. Evaluators return this; CriteriaJudge
    converts to CriterionResult (adding criterion_id, name, dimension).

    Attributes:
        score: Score awarded for this evaluation.
        max_score: Maximum possible score for this evaluation.
        explanation: Human-readable explanation of the score.
        details: Optional structured details about the evaluation.

    Example:
        >>> result = EvalResult(
        ...     score=15,
        ...     max_score=20,
        ...     explanation="Responded to 3 of 4 urgent emails within time limit",
        ...     details={"urgent_emails": 4, "timely_responses": 3}
        ... )
    """

    score: float
    max_score: float
    explanation: str
    details: dict[str, Any] | None = None


@dataclass
class AgentBeatsEvalContext:
    """Context passed to evaluator functions.

    Extends UES EvalContext pattern with AgentBeats-specific fields:
    action_log, initial_state, final_state, user_prompt.

    Attributes:
        client: Async UES client for querying environment state.
        scenario_config: Scenario configuration dictionary.
        action_log: List of actions taken by the Purple agent.
        initial_state: Modality snapshots before assessment.
        final_state: Modality snapshots after assessment.
        user_prompt: The task description given to the Purple agent.

    Note:
        The client has proctor-level access for evaluation purposes,
        allowing queries that wouldn't be available to the Purple agent.
    """

    client: AsyncUESClient
    scenario_config: dict[str, Any]
    action_log: list[ActionLogEntry]
    initial_state: dict[str, Any]
    final_state: dict[str, Any]
    user_prompt: str

    async def get_state(self, modality: str) -> Any:
        """Get current modality state from UES.

        Args:
            modality: The modality to query ('email', 'sms', 'calendar', 'chat').

        Returns:
            The current state of the requested modality.

        Raises:
            ValueError: If an unknown modality is requested.
        """
        modality_clients = {
            "email": self.client.email,
            "sms": self.client.sms,
            "calendar": self.client.calendar,
            "chat": self.client.chat,
        }
        if modality not in modality_clients:
            raise ValueError(f"Unknown modality: {modality}")
        return await modality_clients[modality].get_state()

    async def get_time(self) -> datetime:
        """Get current simulation time.

        Returns:
            The current simulation time from UES.
        """
        time_state = await self.client.time.get_state()
        return time_state.current_time


# Type alias for evaluator functions
# Signature: (context, params) -> EvalResult
EvaluatorFunc = Callable[[AgentBeatsEvalContext, dict[str, Any]], Awaitable[EvalResult]]

# Type alias for evaluator registry (mapping evaluator_id -> function)
EvaluatorRegistry = dict[str, EvaluatorFunc]


# =============================================================================
# Helper Functions
# =============================================================================


def parse_iso8601_duration(duration: str) -> timedelta:
    """Parse an ISO 8601 duration string to a timedelta.

    Supports the following format: P[n]Y[n]M[n]DT[n]H[n]M[n]S
    Note: Years and months are approximated (365 days/year, 30 days/month).

    Args:
        duration: ISO 8601 duration string (e.g., "PT1H", "P1DT12H", "PT30M").

    Returns:
        A timedelta representing the duration.

    Raises:
        ValueError: If the duration string is not valid ISO 8601 format.

    Example:
        >>> parse_iso8601_duration("PT1H30M")
        timedelta(hours=1, minutes=30)
        >>> parse_iso8601_duration("P1D")
        timedelta(days=1)
    """
    match = ISO_8601_DURATION_PATTERN.match(duration)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration: {duration}")

    years = int(match.group(1) or 0)
    months = int(match.group(2) or 0)
    days = int(match.group(3) or 0)
    hours = int(match.group(4) or 0)
    minutes = int(match.group(5) or 0)
    seconds = float(match.group(6) or 0)

    # Approximate years and months
    total_days = years * 365 + months * 30 + days

    return timedelta(
        days=total_days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
    )


def validate_iso8601_duration(value: str) -> str:
    """Validate that a string is a valid ISO 8601 duration.

    Args:
        value: The string to validate.

    Returns:
        The validated string (unchanged).

    Raises:
        ValueError: If the string is not valid ISO 8601 format.
    """
    if not ISO_8601_DURATION_PATTERN.match(value):
        raise ValueError(f"Invalid ISO 8601 duration format: {value}")
    return value


# =============================================================================
# Response Timing Model
# =============================================================================


class ResponseTiming(BaseModel):
    """Timing configuration for character responses.

    Defines how long a character typically takes to respond, with variance
    to simulate realistic human behavior. The actual response delay is
    calculated as: base_delay + random(-variance, +variance).

    Attributes:
        base_delay: Base response delay in ISO 8601 duration format.
        variance: Response time variance in ISO 8601 duration format.

    Example:
        >>> timing = ResponseTiming(base_delay="PT30M", variance="PT10M")
        >>> timing.base_delay_timedelta
        timedelta(minutes=30)
    """

    model_config = ConfigDict(frozen=True)

    base_delay: str = Field(
        ...,
        description="Base response delay (ISO 8601 duration, e.g., 'PT30M')",
    )
    variance: str = Field(
        ...,
        description="Response time variance (ISO 8601 duration, e.g., 'PT10M')",
    )

    @field_validator("base_delay", "variance")
    @classmethod
    def validate_duration_format(cls, value: str) -> str:
        """Validate that the field is a valid ISO 8601 duration."""
        return validate_iso8601_duration(value)

    @property
    def base_delay_timedelta(self) -> timedelta:
        """Get the base delay as a timedelta.

        Returns:
            The base_delay converted to a timedelta.
        """
        return parse_iso8601_duration(self.base_delay)

    @property
    def variance_timedelta(self) -> timedelta:
        """Get the variance as a timedelta.

        Returns:
            The variance converted to a timedelta.
        """
        return parse_iso8601_duration(self.variance)


# =============================================================================
# Character Profile Model
# =============================================================================


class CharacterProfile(BaseModel):
    """Profile for a simulated character in the scenario.

    Characters are simulated contacts that the Purple agent interacts with.
    Each character has a defined personality and response patterns that the
    Green agent uses to generate realistic responses.

    Attributes:
        name: The character's display name.
        relationships: Mapping of character names to relationship descriptions.
            Describes this character's relationships to other characters in the
            scenario (e.g., {"Alex": "direct report", "Bob": "team member"}).
        personality: Description of the character's personality and communication style.
        email: The character's email address (if they communicate via email).
        phone: The character's phone number (if they communicate via SMS).
        response_timing: Timing configuration for this character's responses.
        special_instructions: Additional instructions for response generation.
        config: Character-specific configuration (e.g., pricing for vendors).

    Example:
        >>> from datetime import timedelta
        >>> character = CharacterProfile(
        ...     name="Alice Chen",
        ...     relationships={"Alex Thompson": "direct report"},
        ...     personality="Professional but friendly. Prefers concise communication.",
        ...     email="alice.chen@company.com",
        ...     response_timing=ResponseTiming(base_delay="PT15M", variance="PT5M"),
        ... )
        >>> character.name
        'Alice Chen'
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Character's display name",
    )
    relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of character name to relationship description",
    )
    personality: str = Field(
        ...,
        min_length=1,
        description="Description of personality and communication style",
    )
    email: str | None = Field(
        default=None,
        description="Character's email address",
    )
    phone: str | None = Field(
        default=None,
        description="Character's phone number",
    )
    response_timing: ResponseTiming = Field(
        ...,
        description="Timing configuration for this character's responses",
    )
    special_instructions: str | None = Field(
        default=None,
        description="Additional instructions for response generation",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Character-specific configuration (e.g., pricing data)",
    )

    @model_validator(mode="after")
    def validate_has_contact_method(self) -> "CharacterProfile":
        """Ensure at least one contact method (email or phone) is provided."""
        if self.email is None and self.phone is None:
            raise ValueError("Character must have at least one contact method (email or phone)")
        return self


# =============================================================================
# Evaluation Criterion Model
# =============================================================================


class EvaluationCriterion(BaseModel):
    """Definition of a single evaluation criterion.

    Criteria define how the Green agent evaluates Purple agent performance.
    Each criterion contributes to a scoring dimension. Evaluation can be
    performed either programmatically (via evaluator_id) or using an LLM
    (via evaluation_prompt).

    At least one of `evaluator_id` or `evaluation_prompt` must be provided.

    Attributes:
        criterion_id: Unique identifier for this criterion.
        name: Human-readable name for display.
        description: Detailed description of what this criterion evaluates.
        dimension: The scoring dimension this criterion contributes to.
        max_score: Maximum achievable score for this criterion.
        evaluator_id: Reference to a registered programmatic evaluator function.
        evaluation_prompt: Prompt template for LLM-based evaluation.
        params: Additional parameters passed to the evaluator.

    Example:
        >>> criterion = EvaluationCriterion(
        ...     criterion_id="email_response_accuracy",
        ...     name="Email Response Accuracy",
        ...     description="Evaluates whether sent emails address the original query.",
        ...     dimension="accuracy",
        ...     max_score=10,
        ...     evaluator_id="check_email_response_content",
        ...     params={"required_keywords": ["confirmation", "schedule"]},
        ... )
    """

    model_config = ConfigDict(frozen=True)

    criterion_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique identifier (lowercase, underscore-separated)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable name",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of what this criterion evaluates",
    )
    dimension: ScoringDimension = Field(
        ...,
        description="The scoring dimension this criterion contributes to",
    )
    max_score: int = Field(
        ...,
        ge=1,
        le=100,
        description="Maximum achievable score (1-100)",
    )
    evaluator_id: str | None = Field(
        default=None,
        description="Reference to a registered programmatic evaluator function",
    )
    evaluation_prompt: str | None = Field(
        default=None,
        description="Prompt template for LLM-based evaluation",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Additional parameters passed to the evaluator",
    )

    @model_validator(mode="after")
    def validate_has_evaluation_method(self) -> "EvaluationCriterion":
        """Ensure at least one evaluation method is specified."""
        if self.evaluator_id is None and self.evaluation_prompt is None:
            raise ValueError(
                "Criterion must have at least one evaluation method "
                "(evaluator_id or evaluation_prompt)"
            )
        return self


# =============================================================================
# Scenario Configuration Model
# =============================================================================


class ScenarioConfig(BaseModel):
    """Complete scenario configuration.

    A scenario defines everything needed to run an assessment:
    - Basic metadata (ID, name, description)
    - Timing parameters (start/end time, default time step)
    - User prompt (the task given to the Purple agent)
    - Character profiles (contacts to simulate)
    - Initial UES state (emails, calendar events, etc.)
    - Evaluation criteria (how to score performance)
    - Early completion conditions (optional)

    Attributes:
        scenario_id: Unique identifier for the scenario.
        name: Human-readable scenario name.
        description: Detailed description of the scenario.
        start_time: Assessment start time (timezone-aware).
        end_time: Assessment end time (timezone-aware).
        default_time_step: Default time advance per turn (ISO 8601 duration).
        user_prompt: The task description given to the Purple agent via chat.
        user_character: Key of the character in `characters` that represents the user.
        characters: Mapping of character ID to CharacterProfile.
        initial_state: UES scenario export (imported at assessment start).
        criteria: List of evaluation criteria.
        early_completion_conditions: Optional conditions that end assessment early.

    Example:
        >>> config = ScenarioConfig(
        ...     scenario_id="email_triage_basic",
        ...     name="Basic Email Triage",
        ...     description="Handle incoming emails appropriately.",
        ...     start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
        ...     end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
        ...     default_time_step="PT1H",
        ...     user_prompt="Please triage my inbox and respond to urgent emails.",
        ...     user_character="alex",
        ...     characters={"alex": user_profile, "alice": alice_profile},
        ...     initial_state={...},
        ...     criteria=[criterion1, criterion2],
        ... )
    """

    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique identifier (lowercase, underscore-separated)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable scenario name",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Detailed description of the scenario",
    )
    start_time: datetime = Field(
        ...,
        description="Assessment start time (must be timezone-aware)",
    )
    end_time: datetime = Field(
        ...,
        description="Assessment end time (must be timezone-aware)",
    )
    default_time_step: str = Field(
        ...,
        description="Default time advance per turn (ISO 8601 duration)",
    )
    user_prompt: str = Field(
        ...,
        min_length=1,
        description="Task description given to Purple agent via chat",
    )
    user_character: str = Field(
        ...,
        min_length=1,
        description="Key of the character in 'characters' that represents the user being assisted",
    )
    characters: dict[str, CharacterProfile] = Field(
        ...,
        min_length=1,
        description="Mapping of character ID to profile",
    )
    initial_state: dict[str, Any] = Field(
        ...,
        description="UES scenario export (imported at assessment start)",
    )
    criteria: list[EvaluationCriterion] = Field(
        ...,
        min_length=1,
        description="List of evaluation criteria",
    )
    early_completion_conditions: list[str] | None = Field(
        default=None,
        description="Conditions that end assessment early (optional)",
    )

    @field_validator("default_time_step")
    @classmethod
    def validate_time_step_format(cls, value: str) -> str:
        """Validate that default_time_step is a valid ISO 8601 duration."""
        return validate_iso8601_duration(value)

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure datetime fields are timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_time_ordering(self) -> "ScenarioConfig":
        """Ensure start_time is before end_time."""
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        return self

    @model_validator(mode="after")
    def validate_unique_criterion_ids(self) -> "ScenarioConfig":
        """Ensure all criterion IDs are unique."""
        ids = [c.criterion_id for c in self.criteria]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            raise ValueError(f"Duplicate criterion IDs: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_unique_character_emails(self) -> "ScenarioConfig":
        """Ensure all character emails are unique (if present)."""
        emails = [c.email for c in self.characters.values() if c.email is not None]
        if len(emails) != len(set(emails)):
            duplicates = [e for e in emails if emails.count(e) > 1]
            raise ValueError(f"Duplicate character emails: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_unique_character_phones(self) -> "ScenarioConfig":
        """Ensure all character phone numbers are unique (if present)."""
        phones = [c.phone for c in self.characters.values() if c.phone is not None]
        if len(phones) != len(set(phones)):
            duplicates = [p for p in phones if phones.count(p) > 1]
            raise ValueError(f"Duplicate character phones: {set(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_user_character_exists(self) -> "ScenarioConfig":
        """Ensure user_character refers to a valid character in the characters dict."""
        if self.user_character not in self.characters:
            available = list(self.characters.keys())
            raise ValueError(
                f"user_character '{self.user_character}' not found in characters. "
                f"Available characters: {available}"
            )
        return self

    @property
    def default_time_step_timedelta(self) -> timedelta:
        """Get the default time step as a timedelta.

        Returns:
            The default_time_step converted to a timedelta.
        """
        return parse_iso8601_duration(self.default_time_step)

    @property
    def duration(self) -> timedelta:
        """Get the total scenario duration.

        Returns:
            The time between start_time and end_time.
        """
        return self.end_time - self.start_time

    def get_character_by_email(self, email: str) -> CharacterProfile | None:
        """Find a character by their email address.

        Args:
            email: The email address to search for.

        Returns:
            The CharacterProfile if found, None otherwise.
        """
        for character in self.characters.values():
            if character.email == email:
                return character
        return None

    def get_character_by_phone(self, phone: str) -> CharacterProfile | None:
        """Find a character by their phone number.

        Args:
            phone: The phone number to search for.

        Returns:
            The CharacterProfile if found, None otherwise.
        """
        for character in self.characters.values():
            if character.phone == phone:
                return character
        return None

    def get_criteria_by_dimension(
        self, dimension: ScoringDimension
    ) -> list[EvaluationCriterion]:
        """Get all criteria for a specific scoring dimension.

        Args:
            dimension: The dimension to filter by.

        Returns:
            List of criteria belonging to that dimension.
        """
        return [c for c in self.criteria if c.dimension == dimension]

    def get_total_max_score(self) -> int:
        """Get the total maximum score across all criteria.

        Returns:
            Sum of max_score for all criteria.
        """
        return sum(c.max_score for c in self.criteria)

    def get_max_score_by_dimension(self) -> dict[ScoringDimension, int]:
        """Get the maximum score for each dimension.

        Returns:
            Mapping of dimension to total max_score for that dimension.
        """
        result: dict[ScoringDimension, int] = {}
        for criterion in self.criteria:
            dim = criterion.dimension
            result[dim] = result.get(dim, 0) + criterion.max_score
        return result

    def get_user_character_profile(self) -> CharacterProfile:
        """Get the CharacterProfile for the user being assisted.

        Returns:
            The CharacterProfile for the user_character.

        Note:
            This is guaranteed to succeed because the model validator
            ensures user_character exists in characters.
        """
        return self.characters[self.user_character]
