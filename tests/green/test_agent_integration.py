"""Integration tests for GreenAgent.

These tests verify the GreenAgent class works correctly with a real UES server
and a mock Purple agent. They test the full assessment lifecycle including:

- UES server lifecycle (startup, shutdown)
- Scenario loading and state capture
- Turn orchestration with mock Purple responses
- Response generation and scheduling
- Evaluation and results assembly

Tests are marked with ``pytest.mark.integration`` for selective execution:

    # Run only integration tests
    uv run pytest tests/green/test_agent_integration.py -m integration -v

    # Skip integration tests
    uv run pytest -m "not integration"

Note:
    These tests start real UES server processes and may take several seconds
    to run. They are designed to be run in isolation or as part of a full
    test suite with appropriate timeout settings.

See Also:
    - ``docs/design/GREEN_AGENT_DESIGN_PLAN.md`` § 16.3 (Testing Strategy)
    - ``tests/green/core/test_ues_server.py`` for UESServerManager unit tests
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.messages import (
    CalendarSummary,
    ChatSummary,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    SMSSummary,
    TurnCompleteMessage,
)
from src.common.agentbeats.results import (
    AssessmentResults,
    CriterionResult,
    DimensionScore,
    OverallScore,
    Scores,
)
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.agent import GreenAgent
from src.green.assessment.models import EndOfTurnResult, TurnResult
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    EvaluatorRegistry,
    ResponseTiming,
    ScenarioConfig,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockA2AClientWrapper:
    """Mock A2A client for Purple agent communication.

    Provides configurable responses for testing different scenarios.
    """

    agent_url: str = "http://mock-purple:8001"
    responses: list[Any] = field(default_factory=list)
    _response_index: int = 0
    calls: list[Any] = field(default_factory=list)

    async def send_message(self, message: Any, blocking: bool = True) -> Any:
        """Record the call and return the next configured response."""
        self.calls.append(message)
        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response
        # Default to TurnCompleteMessage if no more responses configured
        return self._create_default_response()

    def _create_default_response(self) -> Any:
        """Create a default TurnCompleteMessage response."""
        return MagicMock(
            parts=[
                MagicMock(
                    root=MagicMock(
                        data={
                            "message_type": "turn_complete",
                            "notes": "Default mock response",
                            "time_step": "PT1H",
                        }
                    )
                )
            ]
        )

    def reset(self) -> None:
        """Reset response index and call history."""
        self._response_index = 0
        self.calls.clear()


@dataclass
class MockTaskUpdater:
    """Mock TaskUpdater for capturing emitted updates."""

    events: list[Any] = field(default_factory=list)

    async def update_status(
        self, state: Any = None, message: Any = None
    ) -> None:
        """Record the update event."""
        self.events.append({"state": state, "message": message})


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def green_config() -> GreenAgentConfig:
    """Create a GreenAgentConfig for testing."""
    return GreenAgentConfig(
        ues_base_port=9100,
        default_max_turns=5,
        default_turn_timeout=30.0,
        response_generator_model="gpt-4o-mini",
        evaluation_model="gpt-4o-mini",
    )


@pytest.fixture
def response_timing() -> ResponseTiming:
    """Create a response timing configuration."""
    return ResponseTiming(base_delay="PT30M", variance="PT10M")


@pytest.fixture
def user_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create a user character profile."""
    return CharacterProfile(
        name="Test User",
        relationships={},
        personality="The user being assisted",
        email="testuser@example.com",
        phone="+15550001111",
        response_timing=response_timing,
    )


@pytest.fixture
def assistant_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create an assistant character profile."""
    return CharacterProfile(
        name="Alice Assistant",
        relationships={"Test User": "colleague"},
        personality="A helpful colleague who responds promptly.",
        email="alice@example.com",
        phone="+15550002222",
        response_timing=response_timing,
    )


@pytest.fixture
def sample_criterion() -> EvaluationCriterion:
    """Create a sample evaluation criterion."""
    return EvaluationCriterion(
        criterion_id="test_criterion",
        name="Test Criterion",
        description="A test evaluation criterion",
        dimension="accuracy",
        max_score=10,
        evaluator_id="test_evaluator",
    )


@pytest.fixture
def simple_scenario(
    user_character: CharacterProfile,
    assistant_character: CharacterProfile,
    sample_criterion: EvaluationCriterion,
) -> ScenarioConfig:
    """Create a simple scenario for testing."""
    return ScenarioConfig(
        scenario_id="test_simple",
        name="Simple Test Scenario",
        description="A minimal scenario for integration testing",
        start_time=datetime(2026, 2, 9, 9, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 2, 9, 17, 0, tzinfo=timezone.utc),
        default_time_step="PT1H",
        user_prompt="Complete the simple test task.",
        user_character="user",
        characters={
            "user": user_character,
            "alice": assistant_character,
        },
        initial_state={"environment": {}},
        criteria=[sample_criterion],
    )


@pytest.fixture
def mock_evaluator_registry() -> EvaluatorRegistry:
    """Create a mock evaluator registry."""

    async def mock_evaluator(context: Any) -> CriterionResult:
        return CriterionResult(
            criterion_id="test_criterion",
            name="Test Criterion",
            dimension="accuracy",
            score=8.0,
            max_score=10.0,
            explanation="Mock evaluation passed.",
        )

    return {"test_evaluator": mock_evaluator}


@pytest.fixture
def mock_purple_client() -> MockA2AClientWrapper:
    """Create a mock Purple agent client."""
    return MockA2AClientWrapper()


@pytest.fixture
def mock_task_updater() -> MockTaskUpdater:
    """Create a mock task updater."""
    return MockTaskUpdater()


@pytest.fixture
def mock_emitter(mock_task_updater: MockTaskUpdater) -> TaskUpdateEmitter:
    """Create a TaskUpdateEmitter with mock updater."""
    return TaskUpdateEmitter(mock_task_updater)  # type: ignore


# =============================================================================
# Helper Functions
# =============================================================================


def create_turn_complete_response(
    notes: str = "Turn complete",
    time_step: str = "PT1H",
) -> MagicMock:
    """Create a mock A2A response with TurnCompleteMessage data."""
    return MagicMock(
        parts=[
            MagicMock(
                root=MagicMock(
                    data={
                        "message_type": "turn_complete",
                        "notes": notes,
                        "time_step": time_step,
                    }
                )
            )
        ]
    )


def create_early_completion_response(
    reason: str = "Task completed early",
) -> MagicMock:
    """Create a mock A2A response with EarlyCompletionMessage data."""
    return MagicMock(
        parts=[
            MagicMock(
                root=MagicMock(
                    data={
                        "message_type": "early_completion",
                        "reason": reason,
                    }
                )
            )
        ]
    )


# =============================================================================
# Construction Tests
# =============================================================================


@pytest.mark.integration
class TestGreenAgentConstruction:
    """Tests for GreenAgent initialization."""

    def test_init_stores_port_and_config(
        self, green_config: GreenAgentConfig
    ) -> None:
        """GreenAgent stores port and config during init."""
        # This test will fail until __init__ is implemented
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    def test_init_creates_llm_factory_instances(
        self, green_config: GreenAgentConfig
    ) -> None:
        """GreenAgent creates LLM instances during init."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Lifecycle Tests
# =============================================================================


@pytest.mark.integration
class TestGreenAgentLifecycle:
    """Tests for GreenAgent lifecycle management (startup/shutdown)."""

    @pytest.mark.asyncio
    async def test_startup_starts_ues_server(
        self, green_config: GreenAgentConfig
    ) -> None:
        """startup() starts the UES server subprocess."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.startup()

    @pytest.mark.asyncio
    async def test_shutdown_stops_ues_server(
        self, green_config: GreenAgentConfig
    ) -> None:
        """shutdown() stops the UES server subprocess."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.startup()
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_startup_creates_ues_client(
        self, green_config: GreenAgentConfig
    ) -> None:
        """startup() creates an AsyncUESClient with proctor key."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.startup()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self, green_config: GreenAgentConfig
    ) -> None:
        """shutdown() can be called multiple times safely."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.shutdown()
            await agent.shutdown()


# =============================================================================
# Run Method Tests
# =============================================================================


@pytest.mark.integration
class TestGreenAgentRun:
    """Tests for the main run() assessment method."""

    @pytest.mark.asyncio
    async def test_run_raises_without_startup(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """run() raises RuntimeError if startup() wasn't called."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.run(
                task_id="test-task",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore
                assessment_config={},
            )

    @pytest.mark.asyncio
    async def test_run_returns_assessment_results(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """run() returns an AssessmentResults object."""
        # Set up mock responses for each turn
        mock_purple_client.responses = [
            create_turn_complete_response(notes=f"Turn {i}")
            for i in range(1, 6)
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.startup()
            try:
                results = await agent.run(
                    task_id="test-task",
                    emitter=mock_emitter,
                    scenario=simple_scenario,
                    evaluators=mock_evaluator_registry,
                    purple_client=mock_purple_client,  # type: ignore
                    assessment_config={},
                )
                assert isinstance(results, AssessmentResults)
            finally:
                await agent.shutdown()


# =============================================================================
# Turn Loop Tests
# =============================================================================


@pytest.mark.integration
class TestTurnLoop:
    """Tests for the turn loop orchestration."""

    @pytest.mark.asyncio
    async def test_max_turns_reached(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment stops when max_turns is reached."""
        # Configure enough responses for all turns
        mock_purple_client.responses = [
            create_turn_complete_response()
            for _ in range(green_config.default_max_turns + 2)
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_early_completion_stops_loop(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment stops when Purple signals early completion."""
        mock_purple_client.responses = [
            create_turn_complete_response(notes="Turn 1"),
            create_turn_complete_response(notes="Turn 2"),
            create_early_completion_response("All tasks done"),
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Cancellation Tests
# =============================================================================


@pytest.mark.integration
class TestCancellation:
    """Tests for assessment cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(
        self, green_config: GreenAgentConfig
    ) -> None:
        """cancel() sets the cancellation flag for the matching task."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.cancel("test-task")

    @pytest.mark.asyncio
    async def test_cancel_mismatched_task_is_noop(
        self, green_config: GreenAgentConfig
    ) -> None:
        """cancel() with wrong task_id does not affect running assessment."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.cancel("wrong-task")


# =============================================================================
# State Management Tests
# =============================================================================


@pytest.mark.integration
class TestStateManagement:
    """Tests for UES state management methods."""

    @pytest.mark.asyncio
    async def test_capture_state_snapshot(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_capture_state_snapshot returns modality states."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_build_initial_state_summary(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_build_initial_state_summary returns InitialStateSummary."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# API Key Management Tests
# =============================================================================


@pytest.mark.integration
class TestAPIKeyManagement:
    """Tests for Purple agent API key management."""

    @pytest.mark.asyncio
    async def test_create_user_api_key(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_create_user_api_key returns (secret, key_id) tuple."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_revoke_user_api_key(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_revoke_user_api_key removes the key from UES."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key_is_silent(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_revoke_user_api_key silently handles 404 for missing keys."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Time Advancement Tests
# =============================================================================


@pytest.mark.integration
class TestTimeAdvancement:
    """Tests for UES time advancement methods."""

    @pytest.mark.asyncio
    async def test_advance_time_parses_duration(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_advance_time correctly parses ISO 8601 duration."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_advance_remainder_computes_correctly(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_advance_remainder computes (time_step - apply_seconds)."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_advance_remainder_handles_zero(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_advance_remainder returns zero-event response when time_step <= 1s."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Response Scheduling Tests
# =============================================================================


@pytest.mark.integration
class TestResponseScheduling:
    """Tests for character response scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_email_response(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_schedule_email_response injects email via UES."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_schedule_sms_response(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_schedule_sms_response injects SMS via UES."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_schedule_calendar_response(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_schedule_calendar_response injects RSVP via UES."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_schedule_unknown_modality_raises(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_schedule_response raises ValueError for unknown modality."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Purple Agent Communication Tests
# =============================================================================


@pytest.mark.integration
class TestPurpleCommunication:
    """Tests for Purple agent A2A communication."""

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_timeout(
        self,
        green_config: GreenAgentConfig,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """_send_and_wait_purple raises TimeoutError when Purple doesn't respond."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_extract_response_data_from_turn_complete(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data extracts TurnCompleteMessage data."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_extract_response_data_from_early_completion(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data extracts EarlyCompletionMessage data."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_extract_response_data_missing_data_part_raises(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data raises ValueError when no DataPart found."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Health Monitoring Tests
# =============================================================================


@pytest.mark.integration
class TestHealthMonitoring:
    """Tests for UES server health monitoring."""

    @pytest.mark.asyncio
    async def test_check_ues_health_when_running(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_check_ues_health returns True when UES is healthy."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_check_ues_health_when_down(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_check_ues_health returns False when UES is not running."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Result Building Tests
# =============================================================================


@pytest.mark.integration
class TestResultBuilding:
    """Tests for assessment result assembly."""

    def test_build_results_creates_valid_artifact(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_build_results creates a valid AssessmentResults object."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# End-to-End Integration Tests (with Real UES)
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWithRealUES:
    """End-to-end integration tests using a real UES server.

    These tests are marked as slow because they start actual UES processes.
    They require the UES package to be installed and working.

    Run with: uv run pytest -m "integration and slow" -v
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full GreenAgent implementation")
    async def test_full_assessment_lifecycle(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Test complete assessment lifecycle with real UES."""
        # Configure Purple to complete after 3 turns
        mock_purple_client.responses = [
            create_turn_complete_response(notes="Turn 1", time_step="PT30M"),
            create_turn_complete_response(notes="Turn 2", time_step="PT30M"),
            create_early_completion_response("All tasks done"),
        ]

        agent = GreenAgent(ues_port=9100, config=green_config)
        try:
            await agent.startup()

            results = await agent.run(
                task_id="e2e-test",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore
                assessment_config={"max_turns": 10},
            )

            # Verify results structure
            assert isinstance(results, AssessmentResults)
            assert results.assessment_id is not None
            assert results.scenario_id == "test_simple"
            assert results.status in ("completed", "failed", "timeout")
            assert results.turns_taken <= 10

        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full GreenAgent implementation")
    async def test_scenario_loads_initial_state(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Test that scenario initial state is loaded into UES."""
        mock_purple_client.responses = [
            create_early_completion_response("Immediate completion")
        ]

        agent = GreenAgent(ues_port=9101, config=green_config)
        try:
            await agent.startup()

            # Capture initial state before and after scenario load
            await agent.run(
                task_id="state-test",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore
                assessment_config={},
            )

            # Verify scenario was loaded (via emitter events)
            # Look for scenario_loaded update in mock_task_updater.events

        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires full GreenAgent implementation")
    async def test_api_key_lifecycle(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Test that API key is created and revoked during assessment."""
        mock_purple_client.responses = [
            create_early_completion_response("Quick completion")
        ]

        agent = GreenAgent(ues_port=9102, config=green_config)
        try:
            await agent.startup()

            await agent.run(
                task_id="key-test",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore
                assessment_config={},
            )

            # After run completes, the key should be revoked
            # We can't directly verify this without access to UES key list

        finally:
            await agent.shutdown()


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_ues_crash_during_assessment(
        self, green_config: GreenAgentConfig
    ) -> None:
        """Assessment handles UES server crash gracefully."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_purple_invalid_response_type(
        self,
        green_config: GreenAgentConfig,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment handles unexpected Purple response type."""
        # Configure Purple to send an unknown message type
        mock_purple_client.responses = [
            MagicMock(
                parts=[
                    MagicMock(
                        root=MagicMock(
                            data={
                                "message_type": "unknown_type",
                                "data": "garbage",
                            }
                        )
                    )
                ]
            )
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_llm_failure_during_response_generation(
        self, green_config: GreenAgentConfig
    ) -> None:
        """Assessment continues when LLM fails during response generation."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_llm_failure_during_evaluation(
        self, green_config: GreenAgentConfig
    ) -> None:
        """Assessment handles LLM failure during criterion evaluation."""
        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge case scenarios."""

    @pytest.mark.asyncio
    async def test_empty_scenario_no_actions(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        user_character: CharacterProfile,
        sample_criterion: EvaluationCriterion,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment handles scenario with no required actions."""
        empty_scenario = ScenarioConfig(
            scenario_id="empty_test",
            name="Empty Scenario",
            description="A scenario with no initial state",
            start_time=datetime(2026, 2, 9, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 2, 9, 17, 0, tzinfo=timezone.utc),
            default_time_step="PT1H",
            user_prompt="Do nothing.",
            user_character="user",
            characters={"user": user_character},
            initial_state={"environment": {}},
            criteria=[sample_criterion],
        )

        mock_purple_client.responses = [
            create_early_completion_response("Nothing to do")
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_single_turn_assessment(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment with max_turns=1 completes correctly."""
        mock_purple_client.responses = [
            create_turn_complete_response(notes="Only turn")
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_zero_time_step(
        self,
        green_config: GreenAgentConfig,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment handles Purple requesting zero time advance."""
        mock_purple_client.responses = [
            create_turn_complete_response(time_step="PT0S")
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)

    @pytest.mark.asyncio
    async def test_very_large_time_step(
        self,
        green_config: GreenAgentConfig,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Assessment handles Purple requesting very large time advance."""
        mock_purple_client.responses = [
            create_turn_complete_response(time_step="P7D")  # 7 days
        ]

        with pytest.raises(NotImplementedError):
            agent = GreenAgent(ues_port=9100, config=green_config)
