"""Integration tests for GreenAgent.

These tests verify the GreenAgent class works correctly with mocked
dependencies. They test the full assessment lifecycle including:

- Construction and initialization
- UES server lifecycle (startup, shutdown)
- Turn orchestration with mock Purple responses
- State management and API key lifecycle
- Time advancement
- Response scheduling
- Purple agent communication
- Health monitoring, result building, error handling

Tests are marked with ``pytest.mark.integration`` for selective
execution::

    # Run only integration tests
    uv run pytest tests/green/test_agent_integration.py -m integration -v

    # Skip integration tests
    uv run pytest -m "not integration"

Note:
    These tests mock ``LLMFactory`` and ``UESServerManager`` to avoid
    needing real API keys or UES server processes. End-to-end tests
    requiring real UES instances are marked ``@pytest.mark.skip`` at
    the bottom of this file.

See Also:
    - ``docs/design/GREEN_AGENT_DESIGN_PLAN.md`` § 16.3 (Testing)
    - ``tests/green/test_agent.py`` for more granular unit tests
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
from src.green.assessment.models import TurnResult
from src.green.response.models import ScheduledResponse
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    EvaluatorRegistry,
    ResponseTiming,
    ScenarioConfig,
)


# ================================================================
# Mock Objects
# ================================================================


@dataclass
class MockA2AClientWrapper:
    """Mock A2A client for Purple agent communication.

    Provides configurable responses for testing various scenarios.
    """

    agent_url: str = "http://mock-purple:8001"
    responses: list[Any] = field(default_factory=list)
    _response_index: int = 0
    calls: list[Any] = field(default_factory=list)
    sent_data_calls: list[Any] = field(default_factory=list)

    async def send_message(
        self, message: Any, blocking: bool = True
    ) -> Any:
        """Record the call and return the next configured response."""
        self.calls.append(message)
        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response
        return self._create_default_response()

    async def send_data(
        self, data: Any = None, blocking: bool = False, **kwargs: Any
    ) -> None:
        """Record data sent (fire-and-forget messages)."""
        self.sent_data_calls.append(
            {"data": data, "blocking": blocking}
        )

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
        self.sent_data_calls.clear()


@dataclass
class MockTaskUpdater:
    """Mock TaskUpdater for capturing emitted updates."""

    events: list[Any] = field(default_factory=list)

    async def update_status(
        self, state: Any = None, message: Any = None
    ) -> None:
        """Record the update event."""
        self.events.append({"state": state, "message": message})


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def green_config() -> GreenAgentConfig:
    """Create a ``GreenAgentConfig`` for testing."""
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
def assistant_character(
    response_timing: ResponseTiming,
) -> CharacterProfile:
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
        initial_state={
            "metadata": {
                "ues_version": "0.2.1",
                "scenario_version": "1",
                "created_at": "2026-02-09T00:00:00+00:00",
                "description": "Simple test scenario",
            },
            "environment": {
                "time_state": {
                    "current_time": "2026-02-09T09:00:00+00:00",
                    "time_scale": 1.0,
                    "is_paused": False,
                    "auto_advance": False,
                    "last_wall_time_update": "2026-02-09T09:00:00+00:00",
                },
                "modality_states": {
                    "email": {
                        "modality_type": "email",
                        "last_updated": "2026-02-09T09:00:00+00:00",
                        "update_count": 0,
                        "emails": {},
                        "threads": {},
                        "folders": {
                            "inbox": [],
                            "sent": [],
                            "drafts": [],
                            "trash": [],
                        },
                        "labels": {},
                        "drafts": {},
                        "user_email_address": "user@test.com",
                    },
                    "sms": {
                        "modality_type": "sms",
                        "last_updated": "2026-02-09T09:00:00+00:00",
                        "update_count": 0,
                        "messages": {},
                        "conversations": {},
                        "max_messages_per_conversation": 10000,
                        "user_phone_number": "+15550000000",
                    },
                    "calendar": {
                        "modality_type": "calendar",
                        "last_updated": "2026-02-09T09:00:00+00:00",
                        "update_count": 0,
                        "calendars": {
                            "primary": {
                                "calendar_id": "primary",
                                "name": "Personal",
                                "color": "#4285f4",
                                "visible": True,
                                "created_at": "2026-02-09T09:00:00+00:00",
                                "updated_at": "2026-02-09T09:00:00+00:00",
                                "event_ids": [],
                                "default_reminders": [],
                            },
                        },
                        "events": {},
                        "default_calendar_id": "primary",
                        "user_timezone": "UTC",
                    },
                    "chat": {
                        "modality_type": "chat",
                        "last_updated": "2026-02-09T09:00:00+00:00",
                        "update_count": 0,
                        "messages": [],
                        "conversations": {},
                        "max_history_size": 1000,
                        "default_conversation_id": "default",
                    },
                },
            },
            "events": {
                "events": [],
            },
        },
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
    """Create a ``TaskUpdateEmitter`` with mock updater."""
    return TaskUpdateEmitter(mock_task_updater)  # type: ignore[arg-type]


@pytest.fixture
def mock_ues_server_manager() -> MagicMock:
    """Create a mock ``UESServerManager``."""
    manager = MagicMock()
    manager.admin_api_key = "test-admin-key-" + "x" * 50
    manager.base_url = "http://127.0.0.1:9100"
    manager.port = 9100
    manager.is_running = True
    manager.start = AsyncMock()
    manager.stop = AsyncMock(return_value=0)
    manager.check_health = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock ``AsyncUESClient``."""
    client = AsyncMock()

    # Mock time client
    client.time.get_state = AsyncMock(
        return_value=MagicMock(
            current_time=datetime(
                2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc
            ),
            model_dump=MagicMock(return_value={"mocked": True}),
        )
    )
    client.time.advance = AsyncMock(
        return_value=MagicMock(events_executed=0)
    )

    # Mock email client
    client.email.get_state = AsyncMock(
        return_value=MagicMock(
            total_email_count=10,
            threads=[MagicMock(), MagicMock()],
            unread_count=3,
            folders={"drafts": []},
            model_dump=MagicMock(return_value={"mocked": True}),
        )
    )
    client.email.receive = AsyncMock()

    # Mock SMS client
    client.sms.get_state = AsyncMock(
        return_value=MagicMock(
            total_message_count=5,
            conversations=[MagicMock()],
            unread_count=1,
            model_dump=MagicMock(return_value={"mocked": True}),
        )
    )
    client.sms.receive = AsyncMock()

    # Mock calendar client
    client.calendar.get_state = AsyncMock(
        return_value=MagicMock(
            events={},
            calendars=[MagicMock()],
            model_dump=MagicMock(return_value={"mocked": True}),
        )
    )
    client.calendar.respond_to_event = AsyncMock()

    # Mock chat client
    client.chat.get_state = AsyncMock(
        return_value=MagicMock(
            total_message_count=2,
            conversation_count=1,
            model_dump=MagicMock(return_value={"mocked": True}),
        )
    )

    # Mock simulation / events / close
    client.simulation.clear = AsyncMock()
    client.simulation.start = AsyncMock()
    client.events.list_events = AsyncMock(
        return_value=MagicMock(events=[])
    )
    client.close = AsyncMock()

    return client


# ================================================================
# Helper Functions
# ================================================================


def _make_agent(
    green_config: GreenAgentConfig,
    mock_ues_server_manager: MagicMock | None = None,
) -> tuple[GreenAgent, MagicMock]:
    """Build a ``GreenAgent`` with ``LLMFactory`` mocked.

    Returns:
        Tuple of (agent, mock_llm_factory).
    """
    factory_patcher = patch("src.green.agent.LLMFactory")
    mock_factory = factory_patcher.start()
    mock_factory.create.return_value = MagicMock()

    patches = [factory_patcher]

    if mock_ues_server_manager is not None:
        mgr_patcher = patch(
            "src.green.agent.UESServerManager",
            return_value=mock_ues_server_manager,
        )
        mgr_patcher.start()
        patches.append(mgr_patcher)

    agent = GreenAgent(ues_port=9100, config=green_config)

    # Stop patches after construction
    for p in patches:
        p.stop()

    return agent, mock_factory


def _make_scores(**kwargs: Any) -> Scores:
    """Create a minimal ``Scores`` for result building."""
    return Scores(
        overall=OverallScore(
            score=kwargs.get("score", 8),
            max_score=kwargs.get("max_score", 10),
        ),
        dimensions={
            "accuracy": DimensionScore(
                score=kwargs.get("score", 8),
                max_score=kwargs.get("max_score", 10),
            )
        },
    )


def _mock_judge(
    scores: Scores | None = None,
    results: list[CriterionResult] | None = None,
    dimensions: list[str] | None = None,
) -> MagicMock:
    """Create a mock ``CriteriaJudge``."""
    judge = MagicMock()
    judge.get_dimensions.return_value = dimensions or ["accuracy"]
    judge.evaluate_all = AsyncMock(return_value=results or [])
    judge.aggregate_scores.return_value = scores or _make_scores()
    return judge


def _turn_result(
    turn: int = 1,
    early: bool = False,
    error: str | None = None,
    **kwargs: Any,
) -> TurnResult:
    """Create a ``TurnResult`` with sensible defaults."""
    return TurnResult(
        turn_number=turn,
        actions_taken=kwargs.get("actions_taken", 1),
        time_step=kwargs.get("time_step", "PT1H"),
        events_processed=kwargs.get("events_processed", 2),
        early_completion=early,
        notes=kwargs.get("notes"),
        error=error,
    )


async def _setup_agent_for_run(
    agent: GreenAgent,
    mock_ues_client: AsyncMock,
) -> None:
    """Inject mocks for methods called during ``run()`` setup phase."""
    agent.ues_client = mock_ues_client
    agent._setup_ues = AsyncMock()
    agent._create_user_api_key = AsyncMock(
        return_value=("secret-key", "key-id-123")
    )
    agent._revoke_user_api_key = AsyncMock()
    agent._send_assessment_start = AsyncMock()
    agent._send_assessment_complete = AsyncMock()


# ================================================================
# Construction Tests
# ================================================================


@pytest.mark.integration
class TestGreenAgentConstruction:
    """Tests for GreenAgent initialization."""

    def test_init_stores_port_and_config(
        self, green_config: GreenAgentConfig
    ) -> None:
        """GreenAgent stores port and config during init."""
        agent, _ = _make_agent(green_config)
        assert agent.ues_port == 9100
        assert agent.config == green_config
        assert agent.ues_client is None
        assert agent._current_task_id is None
        assert agent._cancelled is False

    def test_init_creates_two_llm_instances(
        self, green_config: GreenAgentConfig
    ) -> None:
        """GreenAgent creates response and evaluation LLMs."""
        with patch("src.green.agent.LLMFactory") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.create.return_value = mock_llm
            agent = GreenAgent(ues_port=9100, config=green_config)

            assert mock_factory.create.call_count == 2
            mock_factory.create.assert_any_call(
                "gpt-4o-mini", temperature=0.7
            )
            mock_factory.create.assert_any_call(
                "gpt-4o-mini", temperature=0.0
            )
            assert agent.response_llm is mock_llm
            assert agent.evaluation_llm is mock_llm


# ================================================================
# Lifecycle Tests
# ================================================================


@pytest.mark.integration
class TestGreenAgentLifecycle:
    """Tests for GreenAgent lifecycle management."""

    @pytest.mark.asyncio
    async def test_startup_starts_ues_server(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """startup() starts the UES server subprocess."""
        with (
            patch("src.green.agent.LLMFactory"),
            patch(
                "src.green.agent.UESServerManager",
                return_value=mock_ues_server_manager,
            ),
            patch("ues.client.AsyncUESClient") as mock_cls,
        ):
            mock_cls.return_value = AsyncMock()
            agent = GreenAgent(ues_port=9100, config=green_config)
            await agent.startup()
            mock_ues_server_manager.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_creates_async_ues_client(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """startup() creates an AsyncUESClient with proctor key."""
        with (
            patch("src.green.agent.LLMFactory"),
            patch(
                "src.green.agent.UESServerManager",
                return_value=mock_ues_server_manager,
            ),
            patch("ues.client.AsyncUESClient") as mock_cls,
        ):
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            agent = GreenAgent(ues_port=9100, config=green_config)

            await agent.startup()

            mock_cls.assert_called_once_with(
                base_url=mock_ues_server_manager.base_url,
                api_key=mock_ues_server_manager.admin_api_key,
            )
            assert agent.ues_client is mock_client

    @pytest.mark.asyncio
    async def test_shutdown_stops_server_and_clears_client(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """shutdown() stops UES and closes AsyncUESClient."""
        agent, _ = _make_agent(green_config, mock_ues_server_manager)
        agent.ues_client = mock_ues_client

        await agent.shutdown()

        mock_ues_server_manager.stop.assert_awaited_once()
        mock_ues_client.close.assert_awaited_once()
        assert agent.ues_client is None

    @pytest.mark.asyncio
    async def test_shutdown_without_client_is_safe(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """shutdown() is safe when no UES client exists."""
        agent, _ = _make_agent(green_config, mock_ues_server_manager)
        assert agent.ues_client is None

        # Should not raise
        await agent.shutdown()
        mock_ues_server_manager.stop.assert_awaited_once()


# ================================================================
# Run Method Tests
# ================================================================


@pytest.mark.integration
class TestGreenAgentRun:
    """Tests for the main ``run()`` assessment method."""

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
        agent, _ = _make_agent(green_config)
        # agent.ues_client is None (no startup)
        with pytest.raises(RuntimeError, match="startup"):
            await agent.run(
                task_id="test-task",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
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
        mock_ues_client: AsyncMock,
    ) -> None:
        """run() returns an AssessmentResults with correct metadata."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        # Early completion on first turn
        agent._run_turn = AsyncMock(
            return_value=_turn_result(early=True)
        )

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge()

            results = await agent.run(
                task_id="test-task",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

        assert isinstance(results, AssessmentResults)
        assert results.scenario_id == "test_simple"
        assert results.status == "completed"
        assert results.turns_taken >= 1


# ================================================================
# Turn Loop Tests
# ================================================================


@pytest.mark.integration
class TestTurnLoop:
    """Tests for turn loop orchestration."""

    @pytest.mark.asyncio
    async def test_max_turns_stops_loop(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment stops when max_turns is reached."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        agent._run_turn = AsyncMock(return_value=_turn_result())

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge()

            results = await agent.run(
                task_id="t",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={"max_turns": 3},
            )

        assert agent._run_turn.await_count == 3
        assert results.turns_taken == 3

    @pytest.mark.asyncio
    async def test_early_completion_stops_loop(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment stops when Purple signals early completion."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        agent._run_turn = AsyncMock(
            side_effect=[
                _turn_result(turn=1),
                _turn_result(turn=2, early=True, notes="Done"),
            ]
        )

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge()

            results = await agent.run(
                task_id="t",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={"max_turns": 10},
            )

        assert agent._run_turn.await_count == 2
        assert results.turns_taken == 2


# ================================================================
# Cancellation Tests
# ================================================================


@pytest.mark.integration
class TestCancellation:
    """Tests for assessment cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(
        self, green_config: GreenAgentConfig
    ) -> None:
        """cancel() sets the cancellation flag for matching task."""
        agent, _ = _make_agent(green_config)
        agent._current_task_id = "task-1"
        agent._cancelled = False

        await agent.cancel("task-1")
        assert agent._cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_wrong_task_is_noop(
        self, green_config: GreenAgentConfig
    ) -> None:
        """cancel() with wrong task_id does not set flag."""
        agent, _ = _make_agent(green_config)
        agent._current_task_id = "real-task"
        agent._cancelled = False

        await agent.cancel("wrong-task")
        assert agent._cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_mid_turn_loop_produces_cancelled_status(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Cancellation during the turn loop produces cancelled status.

        Simulates cancellation by setting ``_cancelled = True`` as a
        side-effect of the first ``_run_turn`` call.  The loop should
        exit after that turn and the result status should be
        ``"cancelled"``.
        """
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        def _cancel_on_first_turn(**kwargs: Any) -> TurnResult:
            agent._cancelled = True
            return _turn_result(turn=1)

        agent._run_turn = AsyncMock(side_effect=_cancel_on_first_turn)

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge()

            result = await agent.run(
                task_id="cancel-mid",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={"max_turns": 5},
            )

        # Only one turn should have executed
        assert agent._run_turn.await_count == 1
        assert result.turns_taken == 1
        # "cancelled" maps to "failed" in _build_results status_map
        assert result.status == "failed"

        # Assessment complete message should still have been sent
        agent._send_assessment_complete.assert_awaited_once()

        # API key should still be revoked
        agent._revoke_user_api_key.assert_awaited_once()


# ================================================================
# State Management Tests
# ================================================================


@pytest.mark.integration
class TestStateManagement:
    """Tests for UES state management methods."""

    @pytest.mark.asyncio
    async def test_capture_state_snapshot_returns_all_modalities(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_capture_state_snapshot includes all modality states."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        snapshot = await agent._capture_state_snapshot()

        assert "email" in snapshot
        assert "sms" in snapshot
        assert "calendar" in snapshot
        assert "chat" in snapshot
        assert "time" in snapshot

    @pytest.mark.asyncio
    async def test_build_initial_state_summary_populates_counts(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_build_initial_state_summary returns correct summary counts."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        summary = await agent._build_initial_state_summary()

        assert isinstance(summary, InitialStateSummary)
        assert summary.email.total_emails == 10
        assert summary.sms.total_messages == 5
        assert summary.chat.conversation_count == 1


# ================================================================
# API Key Management Tests
# ================================================================


@pytest.mark.integration
class TestAPIKeyManagement:
    """Tests for Purple agent API key management."""

    @pytest.mark.asyncio
    async def test_create_user_api_key_returns_tuple(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """_create_user_api_key returns (secret, key_id)."""
        agent, _ = _make_agent(green_config, mock_ues_server_manager)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "secret": "purple-key",
            "key_id": "key-123",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            secret, key_id = await agent._create_user_api_key("a-1")

        assert secret == "purple-key"
        assert key_id == "key-123"

    @pytest.mark.asyncio
    async def test_revoke_user_api_key_calls_delete(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """_revoke_user_api_key issues an HTTP DELETE."""
        agent, _ = _make_agent(green_config, mock_ues_server_manager)

        mock_response = MagicMock(status_code=200)
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.delete = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            await agent._revoke_user_api_key("key-123")
            mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_key_does_not_raise(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """_revoke_user_api_key handles 404 silently."""
        agent, _ = _make_agent(green_config, mock_ues_server_manager)

        mock_response = MagicMock(status_code=404)
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.delete = AsyncMock(return_value=mock_response)
            MockClient.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            MockClient.return_value.__aexit__ = AsyncMock(
                return_value=False
            )

            # Should not raise
            await agent._revoke_user_api_key("nonexistent")


# ================================================================
# Time Advancement Tests
# ================================================================


@pytest.mark.integration
class TestTimeAdvancement:
    """Tests for UES time advancement methods."""

    @pytest.mark.asyncio
    async def test_advance_time_one_hour(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_advance_time(PT1H) calls advance(seconds=3600)."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        await agent._advance_time("PT1H")

        mock_ues_client.time.advance.assert_awaited_once_with(
            seconds=3600
        )

    @pytest.mark.asyncio
    async def test_advance_remainder_subtracts_apply(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_advance_remainder(PT1H, 1) advances by 3599 seconds."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        await agent._advance_remainder("PT1H", apply_seconds=1)

        mock_ues_client.time.advance.assert_awaited_once_with(
            seconds=3599
        )

    @pytest.mark.asyncio
    async def test_advance_remainder_zero_skips_advance(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_advance_remainder(PT1S, 1) returns zero—no UES call."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        result = await agent._advance_remainder("PT1S", apply_seconds=1)

        mock_ues_client.time.advance.assert_not_awaited()
        assert result.events_executed == 0


# ================================================================
# Response Scheduling Tests
# ================================================================


@pytest.mark.integration
class TestResponseScheduling:
    """Tests for character response scheduling."""

    @pytest.fixture
    def _now(self) -> datetime:
        return datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_schedule_email_response(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
        _now: datetime,
    ) -> None:
        """_schedule_email_response calls email.receive."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        scheduled = ScheduledResponse(
            modality="email",
            character_name="Alice",
            character_email="alice@example.com",
            scheduled_time=_now,
            content="Hello!",
            recipients=["user@example.com"],
            subject="Test",
        )
        await agent._schedule_email_response(scheduled)

        mock_ues_client.email.receive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_schedule_sms_response(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
        _now: datetime,
    ) -> None:
        """_schedule_sms_response calls sms.receive."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        scheduled = ScheduledResponse(
            modality="sms",
            character_name="Bob",
            character_phone="+15551234567",
            scheduled_time=_now,
            content="Hi!",
            recipients=["+15559876543"],
        )
        await agent._schedule_sms_response(scheduled)

        mock_ues_client.sms.receive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_schedule_calendar_response(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
        _now: datetime,
    ) -> None:
        """_schedule_calendar_response calls calendar.respond_to_event."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        scheduled = ScheduledResponse(
            modality="calendar",
            character_name="Carol",
            character_email="carol@example.com",
            scheduled_time=_now,
            event_id="event-1",
            rsvp_status="accepted",
        )
        await agent._schedule_calendar_response(scheduled)

        mock_ues_client.calendar.respond_to_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_schedule_unknown_modality_raises(
        self,
        green_config: GreenAgentConfig,
    ) -> None:
        """_schedule_response raises ValueError for unknown modality."""
        agent, _ = _make_agent(green_config)
        scheduled = MagicMock()
        scheduled.modality = "fax"

        with pytest.raises(ValueError, match="Unknown modality"):
            await agent._schedule_response(scheduled)


# ================================================================
# Purple Communication Tests
# ================================================================


@pytest.mark.integration
class TestPurpleCommunication:
    """Tests for Purple agent A2A communication."""

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_timeout(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_send_and_wait_purple raises on timeout."""
        agent, _ = _make_agent(green_config)
        mock_client = AsyncMock()

        async def slow_send(*a: Any, **kw: Any) -> None:
            await asyncio.sleep(10)

        mock_client.send_message = AsyncMock(side_effect=slow_send)
        message = MagicMock()
        message.model_dump.return_value = {"message_type": "turn_start"}

        with pytest.raises(asyncio.TimeoutError):
            await agent._send_and_wait_purple(
                purple_client=mock_client,
                message=message,
                timeout=0.01,
            )

    @pytest.mark.asyncio
    async def test_extract_turn_complete_data(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data extracts TurnComplete data."""
        from a2a.types import DataPart, Part, Task

        agent, _ = _make_agent(green_config)
        expected = {
            "message_type": "turn_complete",
            "time_step": "PT1H",
        }
        mock_task = MagicMock(spec=Task)
        mock_task.artifacts = [
            MagicMock(parts=[Part(root=DataPart(data=expected))])
        ]
        mock_task.history = None

        result = agent._extract_response_data(mock_task)
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_early_completion_data(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data extracts EarlyCompletion data."""
        from a2a.types import DataPart, Part, Task

        agent, _ = _make_agent(green_config)
        expected = {
            "message_type": "early_completion",
            "reason": "All done",
        }
        mock_task = MagicMock(spec=Task)
        mock_task.artifacts = [
            MagicMock(parts=[Part(root=DataPart(data=expected))])
        ]
        mock_task.history = None

        result = agent._extract_response_data(mock_task)
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_response_data_no_data_part_raises(
        self, green_config: GreenAgentConfig
    ) -> None:
        """_extract_response_data raises ValueError without DataPart."""
        from a2a.types import Part, Task, TextPart

        agent, _ = _make_agent(green_config)
        mock_task = MagicMock(spec=Task)
        mock_task.artifacts = [
            MagicMock(parts=[Part(root=TextPart(text="no data"))])
        ]
        mock_task.history = []

        with pytest.raises(ValueError, match="No DataPart found"):
            agent._extract_response_data(mock_task)


# ================================================================
# Health Monitoring Tests
# ================================================================


@pytest.mark.integration
class TestHealthMonitoring:
    """Tests for UES server health monitoring."""

    @pytest.mark.asyncio
    async def test_check_health_when_running(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """_check_ues_health returns True when UES is healthy."""
        mock_ues_server_manager.check_health = AsyncMock(
            return_value=True
        )
        agent, _ = _make_agent(green_config, mock_ues_server_manager)

        result = await agent._check_ues_health()

        assert result is True
        mock_ues_server_manager.check_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_health_when_down(
        self,
        green_config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """_check_ues_health returns False when UES is down."""
        mock_ues_server_manager.check_health = AsyncMock(
            return_value=False
        )
        agent, _ = _make_agent(green_config, mock_ues_server_manager)

        result = await agent._check_ues_health()

        assert result is False


# ================================================================
# Result Building Tests
# ================================================================


@pytest.mark.integration
class TestResultBuilding:
    """Tests for assessment result assembly."""

    def test_build_results_creates_valid_object(
        self,
        green_config: GreenAgentConfig,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_build_results creates a complete AssessmentResults."""
        agent, _ = _make_agent(green_config)

        result = agent._build_results(
            assessment_id="a-1",
            scenario=simple_scenario,
            scores=_make_scores(),
            criteria_results=[],
            action_log=[],
            turns_completed=3,
            duration=120.0,
            status="completed",
            participant="http://purple.example.com",
        )

        assert isinstance(result, AssessmentResults)
        assert result.assessment_id == "a-1"
        assert result.scenario_id == "test_simple"
        assert result.status == "completed"
        assert result.turns_taken == 3
        assert result.duration_seconds == 120.0
        assert result.participant == "http://purple.example.com"


# ================================================================
# Error Handling Tests
# ================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_ues_crash_during_setup(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment propagates error on UES crash."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client
        agent._setup_ues = AsyncMock(
            side_effect=ConnectionError("UES crashed")
        )
        agent._create_user_api_key = AsyncMock(
            return_value=("s", "k")
        )
        agent._revoke_user_api_key = AsyncMock()

        with pytest.raises(ConnectionError, match="UES crashed"):
            await agent.run(
                task_id="crash",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

        # Error update was emitted
        updater = mock_emitter._updater  # type: ignore[attr-defined]
        assert len(updater.events) > 0

    @pytest.mark.asyncio
    async def test_purple_invalid_message_type(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment raises ValueError for unknown Purple message."""
        from a2a.types import DataPart, Part, Task

        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        bad_task = MagicMock(spec=Task)
        bad_task.artifacts = [
            MagicMock(
                parts=[
                    Part(
                        root=DataPart(
                            data={
                                "message_type": "unknown_type",
                                "data": "garbage",
                            }
                        )
                    )
                ]
            )
        ]
        bad_task.history = None

        mock_purple = AsyncMock()
        mock_purple.agent_url = "http://mock-purple:8001"
        mock_purple.send_message = AsyncMock(return_value=bad_task)
        mock_purple.send_data = AsyncMock()

        with pytest.raises(ValueError, match="Unexpected message type"):
            await agent.run(
                task_id="bad-type",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple,  # type: ignore[arg-type]
                assessment_config={"max_turns": 1},
            )

    @pytest.mark.asyncio
    async def test_run_turn_error_propagates(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Error in _run_turn propagates and emits error update."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)
        agent._run_turn = AsyncMock(
            side_effect=RuntimeError("Response gen failed")
        )

        mock_purple = AsyncMock()
        mock_purple.agent_url = "http://p:8001"
        mock_purple.send_data = AsyncMock()

        with pytest.raises(RuntimeError, match="Response gen failed"):
            await agent.run(
                task_id="fail",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple,  # type: ignore[arg-type]
                assessment_config={"max_turns": 1},
            )

    @pytest.mark.asyncio
    async def test_evaluation_error_propagates(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Error in evaluation propagates and emits error update."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)

        agent._run_turn = AsyncMock(
            return_value=_turn_result(early=True)
        )

        mock_purple = AsyncMock()
        mock_purple.agent_url = "http://p:8001"
        mock_purple.send_data = AsyncMock()

        with patch("src.green.agent.CriteriaJudge") as jcls:
            judge = MagicMock()
            judge.get_dimensions.return_value = ["accuracy"]
            judge.evaluate_all = AsyncMock(
                side_effect=RuntimeError("LLM eval failed")
            )
            jcls.return_value = judge

            with pytest.raises(RuntimeError, match="LLM eval failed"):
                await agent.run(
                    task_id="eval-fail",
                    emitter=mock_emitter,
                    scenario=simple_scenario,
                    evaluators=mock_evaluator_registry,
                    purple_client=mock_purple,  # type: ignore[arg-type]
                    assessment_config={"max_turns": 1},
                )

    @pytest.mark.asyncio
    async def test_ues_crash_during_turn_loop(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_ues_client: AsyncMock,
    ) -> None:
        """UES ConnectionError mid-turn-loop triggers cleanup.

        When the UES subprocess dies (or becomes unreachable) during
        the turn loop, the exception should propagate, the user API
        key revocation should be attempted, and an error update should
        be emitted.
        """
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)
        agent._run_turn = AsyncMock(
            side_effect=ConnectionError("UES process died")
        )

        mock_purple = AsyncMock()
        mock_purple.agent_url = "http://p:8001"
        mock_purple.send_data = AsyncMock()

        with pytest.raises(ConnectionError, match="UES process died"):
            await agent.run(
                task_id="crash-mid",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple,  # type: ignore[arg-type]
                assessment_config={"max_turns": 3},
            )

        # API key revocation should have been attempted in except block
        agent._revoke_user_api_key.assert_awaited_once_with("key-id-123")

        # Error update should have been emitted
        updater = mock_emitter._updater  # type: ignore[attr-defined]
        assert len(updater.events) > 0

    @pytest.mark.asyncio
    async def test_key_creation_failure_skips_revocation(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """When _create_user_api_key raises, revocation is not attempted.

        If API key creation fails, ``user_key_id`` remains ``None``
        and the ``except`` block should skip the revocation call.
        """
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client
        agent._setup_ues = AsyncMock()
        agent._create_user_api_key = AsyncMock(
            side_effect=RuntimeError("Admin key invalid")
        )
        agent._revoke_user_api_key = AsyncMock()

        with pytest.raises(RuntimeError, match="Admin key invalid"):
            await agent.run(
                task_id="key-fail",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

        # Revocation should NOT have been called since key_id is None
        agent._revoke_user_api_key.assert_not_awaited()

        # Error update should still be emitted
        updater = mock_emitter._updater  # type: ignore[attr-defined]
        assert len(updater.events) > 0


# ================================================================
# Edge Case Tests
# ================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Tests for edge case scenarios."""

    @pytest.mark.asyncio
    async def test_empty_scenario_zero_actions(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        user_character: CharacterProfile,
        sample_criterion: EvaluationCriterion,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment handles scenario with zero actions."""
        empty = ScenarioConfig(
            scenario_id="empty",
            name="Empty Scenario",
            description="No initial state",
            start_time=datetime(
                2026, 2, 9, 9, 0, tzinfo=timezone.utc
            ),
            end_time=datetime(
                2026, 2, 9, 17, 0, tzinfo=timezone.utc
            ),
            default_time_step="PT1H",
            user_prompt="Do nothing.",
            user_character="user",
            characters={"user": user_character},
            initial_state={
                "metadata": {
                    "ues_version": "0.2.1",
                    "scenario_version": "1",
                    "created_at": "2026-02-09T00:00:00+00:00",
                    "description": "Empty test scenario",
                },
                "environment": {
                    "time_state": {
                        "current_time": "2026-02-09T09:00:00+00:00",
                        "time_scale": 1.0,
                        "is_paused": True,
                        "auto_advance": False,
                        "last_wall_time_update": "2026-02-09T09:00:00+00:00",
                    },
                    "modality_states": {
                        "email": {"modality_type": "email"},
                        "sms": {"modality_type": "sms"},
                        "calendar": {"modality_type": "calendar"},
                        "chat": {"modality_type": "chat"},
                    },
                },
                "events": {
                    "events": [],
                },
            },
            criteria=[sample_criterion],
        )

        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)
        agent._run_turn = AsyncMock(
            return_value=_turn_result(
                early=True, actions_taken=0, notes="Nothing to do"
            )
        )

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge()

            result = await agent.run(
                task_id="empty",
                emitter=mock_emitter,
                scenario=empty,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

        assert result.actions_taken == 0
        assert result.turns_taken == 1

    @pytest.mark.asyncio
    async def test_single_turn_assessment(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_ues_client: AsyncMock,
    ) -> None:
        """Assessment with max_turns=1 completes in one turn."""
        agent, _ = _make_agent(green_config)
        await _setup_agent_for_run(agent, mock_ues_client)
        agent._run_turn = AsyncMock(
            return_value=_turn_result(actions_taken=2)
        )

        with patch("src.green.agent.CriteriaJudge") as jcls:
            jcls.return_value = _mock_judge(
                scores=_make_scores(score=0, max_score=0),
                dimensions=[],
            )

            result = await agent.run(
                task_id="one",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={"max_turns": 1},
            )

        assert result.turns_taken == 1
        assert agent._run_turn.await_count == 1

    @pytest.mark.asyncio
    async def test_zero_time_step_no_advance(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_advance_remainder(PT0S, 1) returns zero events."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        result = await agent._advance_remainder("PT0S", apply_seconds=1)

        assert result.events_executed == 0
        mock_ues_client.time.advance.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_large_time_step(
        self,
        green_config: GreenAgentConfig,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_advance_time(P7D) advances by 604800 seconds."""
        agent, _ = _make_agent(green_config)
        agent.ues_client = mock_ues_client

        await agent._advance_time("P7D")

        mock_ues_client.time.advance.assert_awaited_once_with(
            seconds=604800
        )


# ================================================================
# End-to-End Tests (Real UES)
# ================================================================


def _early_completion_response(
    reason: str = "Done",
) -> MagicMock:
    """Build a mock A2A response that duck-types as early completion.

    The response is navigated via the duck-typing fallback path in
    ``_extract_response_data`` (it is not a real ``Task`` or
    ``Message`` instance).
    """
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


def _turn_complete_response(
    notes: str = "Turn done",
    time_step: str = "PT1H",
) -> MagicMock:
    """Build a mock A2A response that duck-types as turn complete."""
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


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWithRealUES:
    """End-to-end tests against a real UES server.

    These tests start a real UES subprocess, load scenario state, run
    the assessment loop with a mock Purple client, and verify results.

    The mock Purple client provides canned responses so no real Purple
    agent is needed. LLM calls use the OpenAI key from ``.env``.

    Run with::

        uv run pytest -m "integration and slow" -v
    """

    @pytest.mark.asyncio
    async def test_full_assessment_lifecycle(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Full lifecycle: startup → run (2 turns + early) → shutdown."""
        mock_purple_client.responses = [
            _turn_complete_response("Turn 1", "PT30M"),
            _turn_complete_response("Turn 2", "PT30M"),
            _early_completion_response("All tasks done"),
        ]

        agent = GreenAgent(ues_port=9200, config=green_config)
        try:
            await agent.startup()

            results = await agent.run(
                task_id="e2e-lifecycle",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={"max_turns": 5},
            )

            assert isinstance(results, AssessmentResults)
            assert results.scenario_id == "test_simple"
            assert results.status == "completed"
            assert results.turns_taken == 3
            assert results.duration_seconds > 0
            assert results.scores is not None
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_scenario_loads_initial_state(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """Scenario initial state is loaded and UES is queryable."""
        mock_purple_client.responses = [
            _early_completion_response("Immediate completion"),
        ]

        agent = GreenAgent(ues_port=9201, config=green_config)
        try:
            await agent.startup()

            results = await agent.run(
                task_id="e2e-state",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

            assert isinstance(results, AssessmentResults)
            assert results.turns_taken == 1

            # Verify UES is still queryable after assessment
            time_state = await agent.ues_client.time.get_state()
            assert time_state.current_time is not None
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_api_key_lifecycle(
        self,
        green_config: GreenAgentConfig,
        mock_emitter: TaskUpdateEmitter,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
    ) -> None:
        """API key is created for Purple and revoked after assessment."""
        mock_purple_client.responses = [
            _early_completion_response("Quick completion"),
        ]

        agent = GreenAgent(ues_port=9202, config=green_config)
        try:
            await agent.startup()

            # Capture the assessment start data to verify API key was sent
            results = await agent.run(
                task_id="e2e-keys",
                emitter=mock_emitter,
                scenario=simple_scenario,
                evaluators=mock_evaluator_registry,
                purple_client=mock_purple_client,  # type: ignore[arg-type]
                assessment_config={},
            )

            assert isinstance(results, AssessmentResults)

            # Verify assessment start message included API key
            assert len(mock_purple_client.sent_data_calls) >= 1
            start_data = mock_purple_client.sent_data_calls[0]["data"]
            assert "api_key" in start_data
            assert len(start_data["api_key"]) > 0

            # Verify assessment complete was also sent
            assert len(mock_purple_client.sent_data_calls) >= 2
            complete_data = mock_purple_client.sent_data_calls[1]["data"]
            assert complete_data["message_type"] == "assessment_complete"
        finally:
            await agent.shutdown()
