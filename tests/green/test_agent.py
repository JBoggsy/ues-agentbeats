"""Tests for src.green.agent module.

This module provides comprehensive unit tests for the GreenAgent class,
which is the core orchestrator for AgentBeats assessments.

Tests cover:
- Initialization and lifecycle (startup, shutdown, cancel)
- API key management (create, revoke)
- UES setup and state management
- Time advancement
- Response scheduling
- Purple agent communication
- Turn orchestration
- Full assessment flow
- Error handling

Test Strategy:
    Most tests use mocking to isolate the GreenAgent from external
    dependencies (UES server, LLMs, A2A clients). Integration tests
    requiring real UES instances are in a separate test file.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.messages import (
    CalendarSummary,
    ChatSummary,
    DEFAULT_ASSESSMENT_INSTRUCTIONS,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    SMSSummary,
    TurnCompleteMessage,
)
from src.common.agentbeats.results import (
    ActionLogEntry,
    AssessmentResults,
    CriterionResult,
    DimensionScore,
    OverallScore,
    Scores,
)
from src.green.agent import USER_PERMISSIONS, GreenAgent
from src.green.assessment.models import EndOfTurnResult, TurnResult
from src.green.response.models import ScheduledResponse


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> GreenAgentConfig:
    """Create a test GreenAgentConfig."""
    return GreenAgentConfig(
        port=8000,
        ues_base_port=8100,
        default_max_turns=10,
        default_turn_timeout=60.0,
        response_generator_model="gpt-4o-mini",
        summarization_model="gpt-4o-mini",
        evaluation_model="gpt-4o",
    )


@pytest.fixture
def now() -> datetime:
    """Return a timezone-aware datetime for testing."""
    return datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_ues_server_manager() -> MagicMock:
    """Create a mock UESServerManager."""
    manager = MagicMock()
    manager.admin_api_key = "test-admin-key-" + "x" * 50
    manager.base_url = "http://127.0.0.1:8100"
    manager.is_running = True
    manager.start = AsyncMock()
    manager.stop = AsyncMock(return_value=0)
    manager.check_health = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock AsyncUESClient."""
    client = AsyncMock()

    # Mock time client
    client.time.get_state = AsyncMock(
        return_value=MagicMock(current_time=datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc))
    )
    client.time.advance = AsyncMock(return_value=MagicMock(events_executed=0))

    # Mock email client
    client.email.get_state = AsyncMock(
        return_value=MagicMock(
            total_email_count=10,
            threads=[MagicMock(), MagicMock()],
            unread_count=3,
            drafts=[],
        )
    )
    client.email.receive = AsyncMock()

    # Mock SMS client
    client.sms.get_state = AsyncMock(
        return_value=MagicMock(
            total_messages=5,
            conversations=[MagicMock()],
            unread_count=1,
        )
    )
    client.sms.receive = AsyncMock()

    # Mock calendar client
    client.calendar.get_state = AsyncMock(
        return_value=MagicMock(
            events=[],
            calendars=[MagicMock()],
        )
    )
    client.calendar.respond_to_event = AsyncMock()

    # Mock chat client
    client.chat.get_state = AsyncMock(
        return_value=MagicMock(
            total_message_count=2,
            conversation_count=1,
        )
    )

    # Mock simulation client
    client.simulation.clear = AsyncMock()
    client.simulation.start = AsyncMock()

    # Mock events client
    client.events.list_events = AsyncMock(return_value=MagicMock(events=[]))

    return client


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.invoke = MagicMock()
    return llm


@pytest.fixture
def mock_scenario() -> MagicMock:
    """Create a mock ScenarioConfig."""
    scenario = MagicMock()
    scenario.scenario_id = "test-scenario"
    scenario.user_prompt = "Process all unread emails"
    scenario.criteria = []
    scenario.initial_state = {"email": {}, "sms": {}, "calendar": {}, "chat": {}}
    scenario.model_dump = MagicMock(return_value={"scenario_id": "test-scenario"})
    return scenario


@pytest.fixture
def mock_purple_client() -> AsyncMock:
    """Create a mock A2AClientWrapper."""
    client = AsyncMock()
    client.agent_url = "http://localhost:8001"
    client.send_message = AsyncMock()
    return client


@pytest.fixture
def mock_emitter() -> AsyncMock:
    """Create a mock TaskUpdateEmitter."""
    emitter = AsyncMock()
    emitter.assessment_started = AsyncMock()
    emitter.turn_started = AsyncMock()
    emitter.turn_completed = AsyncMock()
    emitter.action_observed = AsyncMock()
    emitter.responses_generated = AsyncMock()
    emitter.evaluation_started = AsyncMock()
    emitter.assessment_completed = AsyncMock()
    return emitter


# =============================================================================
# USER_PERMISSIONS Constant Tests
# =============================================================================


class TestUserPermissions:
    """Tests for USER_PERMISSIONS constant."""

    def test_contains_email_state(self) -> None:
        """USER_PERMISSIONS should include email:state."""
        assert "email:state" in USER_PERMISSIONS

    def test_contains_email_send(self) -> None:
        """USER_PERMISSIONS should include email:send."""
        assert "email:send" in USER_PERMISSIONS

    def test_does_not_contain_email_receive(self) -> None:
        """USER_PERMISSIONS should NOT include email:receive (proctor-only)."""
        assert "email:receive" not in USER_PERMISSIONS

    def test_does_not_contain_time_advance(self) -> None:
        """USER_PERMISSIONS should NOT include time:advance (proctor-only)."""
        assert "time:advance" not in USER_PERMISSIONS

    def test_contains_time_read(self) -> None:
        """USER_PERMISSIONS should include time:read."""
        assert "time:read" in USER_PERMISSIONS

    def test_does_not_contain_keys_admin(self) -> None:
        """USER_PERMISSIONS should NOT include keys:* (proctor-only)."""
        assert not any(p.startswith("keys:") for p in USER_PERMISSIONS)

    def test_does_not_contain_simulation_control(self) -> None:
        """USER_PERMISSIONS should NOT include simulation:* (proctor-only)."""
        assert not any(p.startswith("simulation:") for p in USER_PERMISSIONS)


# =============================================================================
# GreenAgent Initialization Tests
# =============================================================================


class TestGreenAgentInit:
    """Tests for GreenAgent.__init__."""

    def test_initializes_agent_with_config(self, config: GreenAgentConfig) -> None:
        """__init__ should initialize agent with config and LLMs."""
        with patch("src.green.agent.LLMFactory") as mock_factory:
            mock_llm = MagicMock()
            mock_factory.create.return_value = mock_llm

            agent = GreenAgent(ues_port=8100, config=config)

            assert agent.config == config
            assert agent.ues_port == 8100
            assert agent.response_llm == mock_llm
            assert agent.evaluation_llm == mock_llm
            assert agent.ues_client is None
            assert agent._current_task_id is None
            assert agent._cancelled is False

    def test_creates_ues_server_manager(self, config: GreenAgentConfig) -> None:
        """__init__ should create UESServerManager with correct port."""
        with patch("src.green.agent.LLMFactory") as mock_factory:
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8200, config=config)

            assert agent._ues_server.port == 8200
            assert agent._ues_port == 8200
            assert agent._proctor_api_key == agent._ues_server.admin_api_key

    def test_creates_llms_with_correct_params(self, config: GreenAgentConfig) -> None:
        """__init__ should create LLMs with correct model names and temperatures."""
        with patch("src.green.agent.LLMFactory") as mock_factory:
            mock_factory.create.return_value = MagicMock()

            GreenAgent(ues_port=8100, config=config)

            # Check response LLM creation
            mock_factory.create.assert_any_call(
                config.response_generator_model,
                temperature=0.7,
            )
            # Check evaluation LLM creation
            mock_factory.create.assert_any_call(
                config.evaluation_model,
                temperature=0.0,
            )


# =============================================================================
# GreenAgent Lifecycle Tests
# =============================================================================


class TestGreenAgentLifecycle:
    """Tests for GreenAgent lifecycle methods (startup, shutdown, cancel)."""

    @pytest.mark.asyncio
    async def test_startup_starts_ues_and_creates_client(
        self,
        config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """startup() should start UES server and create client."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager", return_value=mock_ues_server_manager),
            patch("ues.client.AsyncUESClient") as mock_client_cls,
        ):
            mock_factory.create.return_value = MagicMock()
            mock_client = AsyncMock()
            mock_client_cls.return_value = mock_client

            agent = GreenAgent(ues_port=8100, config=config)
            await agent.startup()

            mock_ues_server_manager.start.assert_awaited_once()
            mock_client_cls.assert_called_once_with(
                base_url=mock_ues_server_manager.base_url,
                api_key=mock_ues_server_manager.admin_api_key,
            )
            assert agent.ues_client == mock_client

    @pytest.mark.asyncio
    async def test_shutdown_stops_ues_and_closes_client(
        self,
        config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """shutdown() should stop UES server and close client."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager", return_value=mock_ues_server_manager),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            agent.ues_client = mock_ues_client

            await agent.shutdown()

            mock_ues_client.close.assert_awaited_once()
            mock_ues_server_manager.stop.assert_awaited_once()
            assert agent.ues_client is None

    @pytest.mark.asyncio
    async def test_shutdown_safe_to_call_multiple_times(
        self,
        config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """shutdown() should be safe to call multiple times."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager", return_value=mock_ues_server_manager),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            agent.ues_client = None  # Already None

            # Should not raise
            await agent.shutdown()
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_sets_cancelled_flag(
        self,
        config: GreenAgentConfig,
    ) -> None:
        """cancel() should set the cancelled flag for matching task."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager"),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            agent._current_task_id = "task-123"
            agent._cancelled = False

            await agent.cancel("task-123")

            assert agent._cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_ignores_non_matching_task(
        self,
        config: GreenAgentConfig,
    ) -> None:
        """cancel() should ignore requests for non-matching task IDs."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager"),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            agent._current_task_id = "task-123"
            agent._cancelled = False

            await agent.cancel("other-task")

            assert agent._cancelled is False


# =============================================================================
# GreenAgent.run() Tests
# =============================================================================


class TestGreenAgentRun:
    """Tests for GreenAgent.run() main entry point."""

    @pytest.mark.asyncio
    async def test_run_raises_runtime_error_without_startup(
        self,
        mock_emitter: AsyncMock,
        mock_scenario: MagicMock,
        mock_purple_client: AsyncMock,
    ) -> None:
        """run() should raise RuntimeError if startup() was not called."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = None
            with pytest.raises(RuntimeError, match="startup"):
                await agent.run(
                    task_id="task-123",
                    emitter=mock_emitter,
                    scenario=mock_scenario,
                    evaluators={},
                    purple_client=mock_purple_client,
                    assessment_config={},
                )


# =============================================================================
# Turn Orchestration Tests
# =============================================================================


class TestTurnOrchestration:
    """Tests for turn orchestration methods (_run_turn, _process_turn_end)."""

    @pytest.mark.asyncio
    async def test_run_turn_sends_turn_start_and_waits(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_run_turn() should send TurnStart, wait for reply, and process turn end."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent.config = GreenAgentConfig(
                default_turn_timeout=60.0,
            )

            # Purple replies with TurnCompleteMessage
            turn_complete = TurnCompleteMessage(time_step="PT1H")
            agent._send_and_wait_purple = AsyncMock(return_value=turn_complete)
            agent._process_turn_end = AsyncMock(
                return_value=EndOfTurnResult(
                    actions_taken=2, total_events=5, responses_generated=1
                )
            )

            result = await agent._run_turn(
                turn=1,
                emitter=mock_emitter,
                purple_client=mock_purple_client,
                action_log_builder=MagicMock(),
                message_collector=MagicMock(),
                response_generator=MagicMock(),
            )

            assert result.turn_number == 1
            assert result.actions_taken == 2
            assert result.events_processed == 5
            assert result.early_completion is False
            mock_emitter.turn_started.assert_awaited_once()
            mock_emitter.turn_completed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_turn_handles_early_completion(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_run_turn() should return early_completion=True on EarlyCompletionMessage."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent.config = GreenAgentConfig(default_turn_timeout=60.0)

            early = EarlyCompletionMessage(reason="All tasks done")
            agent._send_and_wait_purple = AsyncMock(return_value=early)

            result = await agent._run_turn(
                turn=1,
                emitter=mock_emitter,
                purple_client=mock_purple_client,
                action_log_builder=MagicMock(),
                message_collector=MagicMock(),
                response_generator=MagicMock(),
            )

            assert result.early_completion is True
            assert result.notes == "All tasks done"

    @pytest.mark.asyncio
    async def test_run_turn_handles_timeout(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_run_turn() should return error='timeout' on asyncio.TimeoutError."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent.config = GreenAgentConfig(default_turn_timeout=60.0)

            agent._send_and_wait_purple = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )

            result = await agent._run_turn(
                turn=1,
                emitter=mock_emitter,
                purple_client=mock_purple_client,
                action_log_builder=MagicMock(),
                message_collector=MagicMock(),
                response_generator=MagicMock(),
            )

            assert result.error == "timeout"
            assert result.early_completion is False

    @pytest.mark.asyncio
    async def test_process_turn_end_orchestrates_all_phases(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_process_turn_end() should advance time, collect events, generate responses."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            # Mock the time advancement methods
            agent._advance_time = AsyncMock(
                return_value=MagicMock(events_executed=2)
            )
            agent._advance_remainder = AsyncMock(
                return_value=MagicMock(events_executed=3)
            )
            agent._schedule_response = AsyncMock()

            # Mock action_log_builder
            mock_action_log = MagicMock()
            mock_purple_entry = MagicMock(
                timestamp=now, action="email.send",
                parameters={}, success=True, error_message=None,
            )
            mock_action_log.add_events_from_turn.return_value = (
                [mock_purple_entry], []
            )

            # Mock message_collector
            mock_message_collector = MagicMock()
            mock_message_collector.collect = AsyncMock(return_value=[])

            # Mock response_generator
            mock_response_gen = MagicMock()
            mock_response_gen.process_new_messages = AsyncMock(return_value=[])

            result = await agent._process_turn_end(
                turn=1,
                turn_start_time=now,
                time_step="PT1H",
                emitter=mock_emitter,
                action_log_builder=mock_action_log,
                message_collector=mock_message_collector,
                response_generator=mock_response_gen,
            )

            assert result.actions_taken == 1
            assert result.total_events == 5  # 2 + 3
            assert result.responses_generated == 0
            agent._advance_time.assert_awaited_once_with("PT1S")
            agent._advance_remainder.assert_awaited_once_with(
                time_step="PT1H", apply_seconds=1
            )


# =============================================================================
# Time Management Tests
# =============================================================================


class TestTimeManagement:
    """Tests for time management methods (_advance_time, _advance_remainder)."""

    @pytest.mark.asyncio
    async def test_advance_time_parses_duration_and_advances(self) -> None:
        """_advance_time() parses ISO 8601 duration and calls UES client."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            # Mock the UES client
            mock_client = MagicMock()
            mock_advance_result = MagicMock()
            mock_advance_result.events_executed = 5
            mock_client.time.advance = AsyncMock(return_value=mock_advance_result)
            agent.ues_client = mock_client

            result = await agent._advance_time("PT1H")

            # Verify correct seconds were passed (1 hour = 3600 seconds)
            mock_client.time.advance.assert_called_once_with(seconds=3600)
            assert result.events_executed == 5

    @pytest.mark.asyncio
    async def test_advance_time_handles_various_durations(self) -> None:
        """_advance_time() correctly parses various ISO 8601 durations."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_client = MagicMock()
            mock_client.time.advance = AsyncMock(return_value=MagicMock(events_executed=0))
            agent.ues_client = mock_client

            # Test various durations
            test_cases = [
                ("PT1S", 1),
                ("PT30M", 1800),
                ("PT1H30M", 5400),
                ("P1D", 86400),
            ]
            for duration_str, expected_seconds in test_cases:
                mock_client.time.advance.reset_mock()
                await agent._advance_time(duration_str)
                mock_client.time.advance.assert_called_once_with(seconds=expected_seconds)

    @pytest.mark.asyncio
    async def test_advance_remainder_calculates_correctly(self) -> None:
        """_advance_remainder() computes (time_step - apply_seconds) and advances."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_client = MagicMock()
            mock_advance_result = MagicMock()
            mock_advance_result.events_executed = 3
            mock_client.time.advance = AsyncMock(return_value=mock_advance_result)
            agent.ues_client = mock_client

            result = await agent._advance_remainder("PT1H", apply_seconds=1)

            # 1 hour (3600s) - 1s apply = 3599s remainder
            mock_client.time.advance.assert_called_once_with(seconds=3599)
            assert result.events_executed == 3

    @pytest.mark.asyncio
    async def test_advance_remainder_returns_zero_event_for_small_timestep(self) -> None:
        """_advance_remainder() returns zero-event result when remainder <= 0."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_client = MagicMock()
            mock_client.time.advance = AsyncMock()
            agent.ues_client = mock_client

            # PT1S = 1 second, apply_seconds=1 → remainder=0
            result = await agent._advance_remainder("PT1S", apply_seconds=1)

            # Should NOT call ues_client.time.advance
            mock_client.time.advance.assert_not_called()
            # Should return a zero-event placeholder
            assert result.events_executed == 0
            assert result.events_failed == 0

    @pytest.mark.asyncio
    async def test_advance_remainder_handles_negative_remainder(self) -> None:
        """_advance_remainder() returns zero-event result for negative remainder."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_client = MagicMock()
            mock_client.time.advance = AsyncMock()
            agent.ues_client = mock_client

            # PT0S = 0 seconds, apply_seconds=1 → remainder=-1 (negative)
            result = await agent._advance_remainder("PT0S", apply_seconds=1)

            mock_client.time.advance.assert_not_called()
            assert result.events_executed == 0


# =============================================================================
# Purple Agent Communication Tests
# =============================================================================


class TestPurpleCommunication:
    """Tests for Purple agent communication methods."""

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_parses_turn_complete(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_and_wait_purple() should parse TurnCompleteMessage responses."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Part, Task

            response_data = {"message_type": "turn_complete", "time_step": "PT30M"}
            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = [
                MagicMock(parts=[Part(root=DataPart(data=response_data))])
            ]
            mock_task.history = None
            mock_purple_client.send_message = AsyncMock(return_value=mock_task)

            message = MagicMock()
            message.model_dump.return_value = {"message_type": "turn_start"}

            result = await agent._send_and_wait_purple(
                purple_client=mock_purple_client,
                message=message,
                timeout=60.0,
            )

            assert isinstance(result, TurnCompleteMessage)
            assert result.time_step == "PT30M"

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_parses_early_completion(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_and_wait_purple() should parse EarlyCompletionMessage responses."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Part, Task

            response_data = {
                "message_type": "early_completion",
                "reason": "All done",
            }
            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = [
                MagicMock(parts=[Part(root=DataPart(data=response_data))])
            ]
            mock_task.history = None
            mock_purple_client.send_message = AsyncMock(return_value=mock_task)

            message = MagicMock()
            message.model_dump.return_value = {"message_type": "turn_start"}

            result = await agent._send_and_wait_purple(
                purple_client=mock_purple_client,
                message=message,
                timeout=60.0,
            )

            assert isinstance(result, EarlyCompletionMessage)
            assert result.reason == "All done"

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_raises_on_unexpected_type(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_and_wait_purple() should raise ValueError on unknown message_type."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Part, Task

            response_data = {"message_type": "unknown_type"}
            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = [
                MagicMock(parts=[Part(root=DataPart(data=response_data))])
            ]
            mock_task.history = None
            mock_purple_client.send_message = AsyncMock(return_value=mock_task)

            message = MagicMock()
            message.model_dump.return_value = {"message_type": "turn_start"}

            with pytest.raises(ValueError, match="Unexpected message type"):
                await agent._send_and_wait_purple(
                    purple_client=mock_purple_client,
                    message=message,
                    timeout=60.0,
                )

    @pytest.mark.asyncio
    async def test_send_and_wait_purple_raises_on_timeout(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_and_wait_purple() should propagate TimeoutError."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            async def slow_send(*args, **kwargs):
                await asyncio.sleep(10)

            mock_purple_client.send_message = AsyncMock(side_effect=slow_send)

            message = MagicMock()
            message.model_dump.return_value = {"message_type": "turn_start"}

            with pytest.raises(asyncio.TimeoutError):
                await agent._send_and_wait_purple(
                    purple_client=mock_purple_client,
                    message=message,
                    timeout=0.01,
                )

    def test_extract_response_data_from_task_with_artifacts(self) -> None:
        """_extract_response_data() should extract data from Task artifacts."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Part, Task

            expected_data = {"message_type": "turn_complete", "time_step": "PT1H"}
            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = [
                MagicMock(parts=[Part(root=DataPart(data=expected_data))])
            ]
            mock_task.history = None

            result = agent._extract_response_data(mock_task)
            assert result == expected_data

    def test_extract_response_data_from_task_history(self) -> None:
        """_extract_response_data() should extract data from Task history."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Message, Part, Role, Task

            expected_data = {"message_type": "turn_complete"}
            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = []
            agent_msg = Message(
                role=Role.agent,
                parts=[Part(root=DataPart(data=expected_data))],
                message_id="msg-1",
            )
            mock_task.history = [agent_msg]

            result = agent._extract_response_data(mock_task)
            assert result == expected_data

    def test_extract_response_data_raises_no_data_part(self) -> None:
        """_extract_response_data() should raise ValueError when no DataPart found."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import Part, Task, TextPart

            mock_task = MagicMock(spec=Task)
            mock_task.artifacts = [
                MagicMock(parts=[Part(root=TextPart(text="not data"))])
            ]
            mock_task.history = []

            with pytest.raises(ValueError, match="No DataPart found"):
                agent._extract_response_data(mock_task)

    @pytest.mark.asyncio
    async def test_send_assessment_start_sends_correct_message(
        self,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_send_assessment_start() should send AssessmentStartMessage via purple_client."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            initial_summary = InitialStateSummary(
                email=EmailSummary(
                    total_emails=0, total_threads=0, unread=0, draft_count=0
                ),
                calendar=CalendarSummary(
                    event_count=0, calendar_count=0, events_today=0
                ),
                sms=SMSSummary(
                    total_messages=0, total_conversations=0, unread=0
                ),
                chat=ChatSummary(total_messages=0, conversation_count=0),
            )

            await agent._send_assessment_start(
                purple_client=mock_purple_client,
                initial_summary=initial_summary,
                ues_url="http://127.0.0.1:8100",
                api_key="test-key",
            )

            mock_purple_client.send_data.assert_awaited_once()
            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["message_type"] == "assessment_start"
            assert sent_data["ues_url"] == "http://127.0.0.1:8100"
            assert sent_data["api_key"] == "test-key"
            assert (
                sent_data["assessment_instructions"]
                == DEFAULT_ASSESSMENT_INSTRUCTIONS
            )

    @pytest.mark.asyncio
    async def test_send_assessment_complete_sends_correct_message(
        self,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should send AssessmentCompleteMessage."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            await agent._send_assessment_complete(
                purple_client=mock_purple_client,
                reason="scenario_complete",
            )

            mock_purple_client.send_data.assert_awaited_once()
            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["message_type"] == "assessment_complete"
            assert sent_data["reason"] == "scenario_complete"

    @pytest.mark.asyncio
    async def test_send_assessment_complete_maps_cancelled_to_timeout(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should map 'cancelled' to 'timeout'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            await agent._send_assessment_complete(
                purple_client=mock_purple_client,
                reason="cancelled",
            )

            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["reason"] == "timeout"


# =============================================================================
# API Key Management Tests
# =============================================================================


class TestAPIKeyManagement:
    """Tests for API key management methods."""

    @pytest.mark.asyncio
    async def test_create_user_api_key_returns_secret_and_id(self) -> None:
        """_create_user_api_key() should POST to UES and return (secret, key_id)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent._ues_server = MagicMock()
            agent._ues_server.base_url = "http://127.0.0.1:8100"
            agent._ues_server.admin_api_key = "admin-key"

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "secret": "purple-secret-key",
                "key_id": "key-abc",
            }
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                MockClient.return_value.__aenter__ = AsyncMock(
                    return_value=mock_client
                )
                MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

                secret, key_id = await agent._create_user_api_key("assess-123")

                assert secret == "purple-secret-key"
                assert key_id == "key-abc"
                mock_client.post.assert_awaited_once()
                call_args = mock_client.post.call_args
                assert "/keys" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_revoke_user_api_key_deletes_key(self) -> None:
        """_revoke_user_api_key() should DELETE the key via UES."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent._ues_server = MagicMock()
            agent._ues_server.base_url = "http://127.0.0.1:8100"
            agent._ues_server.admin_api_key = "admin-key"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_client.delete = AsyncMock(return_value=mock_response)
                MockClient.return_value.__aenter__ = AsyncMock(
                    return_value=mock_client
                )
                MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

                await agent._revoke_user_api_key("key-abc")

                mock_client.delete.assert_awaited_once()
                call_args = mock_client.delete.call_args
                assert "key-abc" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_revoke_user_api_key_ignores_404(self) -> None:
        """_revoke_user_api_key() should silently ignore 404."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent._ues_server = MagicMock()
            agent._ues_server.base_url = "http://127.0.0.1:8100"
            agent._ues_server.admin_api_key = "admin-key"

            mock_response = MagicMock()
            mock_response.status_code = 404

            with patch("httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_client.delete = AsyncMock(return_value=mock_response)
                MockClient.return_value.__aenter__ = AsyncMock(
                    return_value=mock_client
                )
                MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

                # Should not raise
                await agent._revoke_user_api_key("nonexistent-key")


# =============================================================================
# UES Setup Tests
# =============================================================================


class TestUESSetup:
    """Tests for UES setup and state management methods."""

    @pytest.mark.asyncio
    async def test_setup_ues_clears_and_loads_state(
        self,
        mock_scenario: MagicMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_setup_ues() should clear UES, import state, and start simulation."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent._ues_port = 8100
            agent._proctor_api_key = "admin-key"

            await agent._setup_ues(mock_scenario)

            # Should clear state first
            mock_ues_client.simulation.clear.assert_awaited_once()
            # Should import scenario state via client library
            mock_ues_client.scenario.import_full.assert_awaited_once_with(
                scenario=mock_scenario.initial_state,
                strict_modalities=False,
            )
            # Should start simulation
            mock_ues_client.simulation.start.assert_awaited_once_with(
                auto_advance=False
            )


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management methods."""

    @pytest.mark.asyncio
    async def test_capture_state_snapshot_returns_all_modalities(
        self,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_capture_state_snapshot() should return dict with all modality states."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            # Mock model_dump on all state objects
            for sub in [
                mock_ues_client.email.get_state,
                mock_ues_client.sms.get_state,
                mock_ues_client.calendar.get_state,
                mock_ues_client.chat.get_state,
                mock_ues_client.time.get_state,
            ]:
                state = sub.return_value
                state.model_dump = MagicMock(return_value={"mocked": True})

            result = await agent._capture_state_snapshot()

            assert "email" in result
            assert "sms" in result
            assert "calendar" in result
            assert "chat" in result
            assert "time" in result
            for key in ["email", "sms", "calendar", "chat", "time"]:
                assert result[key] == {"mocked": True}

    @pytest.mark.asyncio
    async def test_build_initial_state_summary_returns_summary(
        self,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_build_initial_state_summary() should return an InitialStateSummary."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            # Configure email state to have expected attributes
            email_state = mock_ues_client.email.get_state.return_value
            email_state.total_email_count = 10
            email_state.threads = [MagicMock(), MagicMock()]
            email_state.unread_count = 3
            email_state.folders = {"drafts": [MagicMock()]}

            # Configure sms state
            sms_state = mock_ues_client.sms.get_state.return_value
            sms_state.total_message_count = 5
            sms_state.conversations = [MagicMock()]
            sms_state.unread_count = 1

            # Configure calendar state
            cal_state = mock_ues_client.calendar.get_state.return_value
            cal_state.events = {}  # dict for _count_events_today
            cal_state.calendars = [MagicMock()]

            # Configure chat state
            chat_state = mock_ues_client.chat.get_state.return_value
            chat_state.total_message_count = 8
            chat_state.conversation_count = 2

            result = await agent._build_initial_state_summary()

            assert isinstance(result, InitialStateSummary)
            assert result.email.total_emails == 10
            assert result.email.total_threads == 2
            assert result.email.unread == 3
            assert result.email.draft_count == 1
            assert result.sms.total_messages == 5
            assert result.sms.total_conversations == 1
            assert result.sms.unread == 1
            assert result.calendar.event_count == 0  # empty dict
            assert result.calendar.calendar_count == 1
            assert result.chat.total_messages == 8
            assert result.chat.conversation_count == 2

    def test_count_events_today_with_matching_events(self, now: datetime) -> None:
        """_count_events_today() should count events on the current date."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            # Create mock calendar state with events dict
            cal_state = MagicMock()
            event_today = MagicMock()
            event_today.start = now  # Same date
            event_tomorrow = MagicMock()
            event_tomorrow.start = now + timedelta(days=1)
            cal_state.events = {
                "evt-1": event_today,
                "evt-2": event_tomorrow,
            }

            result = agent._count_events_today(cal_state, now)
            assert result == 1

    def test_count_events_today_with_no_events(self, now: datetime) -> None:
        """_count_events_today() should return 0 for empty events."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            cal_state = MagicMock()
            cal_state.events = {}

            result = agent._count_events_today(cal_state, now)
            assert result == 0

    def test_count_events_today_with_string_time(self) -> None:
        """_count_events_today() should handle string current_time."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            cal_state = MagicMock()
            event = MagicMock()
            event.start = datetime(2026, 2, 9, 14, 0, 0, tzinfo=timezone.utc)
            cal_state.events = {"evt-1": event}

            result = agent._count_events_today(
                cal_state, "2026-02-09T10:00:00+00:00"
            )
            assert result == 1


# =============================================================================
# Response Scheduling Tests
# =============================================================================


class TestResponseScheduling:
    """Tests for response scheduling methods."""

    @pytest.mark.asyncio
    async def test_schedule_response_dispatches_email(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_response() should dispatch email to _schedule_email_response."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent._schedule_email_response = AsyncMock()
            agent._schedule_sms_response = AsyncMock()
            agent._schedule_calendar_response = AsyncMock()

            scheduled = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello!",
                recipients=["user@example.com"],
            )
            await agent._schedule_response(scheduled)

            agent._schedule_email_response.assert_awaited_once_with(scheduled)
            agent._schedule_sms_response.assert_not_awaited()
            agent._schedule_calendar_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_schedule_response_dispatches_sms(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_response() should dispatch sms to _schedule_sms_response."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent._schedule_email_response = AsyncMock()
            agent._schedule_sms_response = AsyncMock()
            agent._schedule_calendar_response = AsyncMock()

            scheduled = ScheduledResponse(
                modality="sms",
                character_name="Bob",
                character_phone="+15551234567",
                scheduled_time=now,
                content="Hi!",
                recipients=["+15559876543"],
            )
            await agent._schedule_response(scheduled)

            agent._schedule_sms_response.assert_awaited_once_with(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_response_dispatches_calendar(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_response() should dispatch calendar to _schedule_calendar_response."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent._schedule_email_response = AsyncMock()
            agent._schedule_sms_response = AsyncMock()
            agent._schedule_calendar_response = AsyncMock()

            scheduled = ScheduledResponse(
                modality="calendar",
                character_name="Carol",
                character_email="carol@example.com",
                scheduled_time=now,
                event_id="event-123",
                rsvp_status="accepted",
            )
            await agent._schedule_response(scheduled)

            agent._schedule_calendar_response.assert_awaited_once_with(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_response_raises_on_unknown_modality(
        self,
        now: datetime,
    ) -> None:
        """_schedule_response() should raise ValueError for unknown modality."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            # Use MagicMock to bypass ScheduledResponse validation
            scheduled = MagicMock()
            scheduled.modality = "fax"

            with pytest.raises(ValueError, match="Unknown modality"):
                await agent._schedule_response(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_email_response_calls_ues_client(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_email_response() should call ues_client.email.receive."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            scheduled = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello!",
                recipients=["user@example.com"],
                subject="Re: Test",
                thread_id="thread-1",
            )
            await agent._schedule_email_response(scheduled)

            mock_ues_client.email.receive.assert_awaited_once()
            call_kwargs = mock_ues_client.email.receive.call_args.kwargs
            assert call_kwargs["from_address"] == "alice@example.com"
            assert call_kwargs["to_addresses"] == ["user@example.com"]
            assert call_kwargs["subject"] == "Re: Test"
            assert call_kwargs["body_text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_schedule_sms_response_calls_ues_client(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_sms_response() should call ues_client.sms.receive."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            scheduled = ScheduledResponse(
                modality="sms",
                character_name="Bob",
                character_phone="+15551234567",
                scheduled_time=now,
                content="Hi there!",
                recipients=["+15559876543"],
            )
            await agent._schedule_sms_response(scheduled)

            mock_ues_client.sms.receive.assert_awaited_once()
            call_kwargs = mock_ues_client.sms.receive.call_args.kwargs
            assert call_kwargs["from_number"] == "+15551234567"
            assert call_kwargs["to_numbers"] == ["+15559876543"]
            assert call_kwargs["body"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_schedule_calendar_response_calls_ues_client(
        self,
        now: datetime,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_schedule_calendar_response() should call ues_client.calendar.respond_to_event."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            scheduled = ScheduledResponse(
                modality="calendar",
                character_name="Carol",
                character_email="carol@example.com",
                scheduled_time=now,
                event_id="event-123",
                rsvp_status="accepted",
                rsvp_comment="Sounds good!",
            )
            await agent._schedule_calendar_response(scheduled)

            mock_ues_client.calendar.respond_to_event.assert_awaited_once()
            call_kwargs = mock_ues_client.calendar.respond_to_event.call_args.kwargs
            assert call_kwargs["event_id"] == "event-123"
            assert call_kwargs["attendee_email"] == "carol@example.com"
            assert call_kwargs["response"] == "accepted"
            assert call_kwargs["comment"] == "Sounds good!"


# =============================================================================
# Result Building Tests
# =============================================================================


class TestResultBuilding:
    """Tests for result building methods."""

    def test_build_results_returns_assessment_results(
        self,
        mock_scenario: MagicMock,
    ) -> None:
        """_build_results() should return a valid AssessmentResults."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scores = Scores(
                overall=OverallScore(score=10, max_score=20),
                dimensions={"accuracy": DimensionScore(score=10, max_score=20)},
            )
            criterion_result = CriterionResult(
                criterion_id="test-criterion",
                name="Test Criterion",
                dimension="accuracy",
                score=10,
                max_score=20,
                explanation="Test passed",
            )
            action_entry = ActionLogEntry(
                turn=1,
                timestamp=datetime.now(timezone.utc),
                action="email.send",
                parameters={"to": ["test@example.com"]},
                success=True,
            )

            result = agent._build_results(
                assessment_id="assess-123",
                scenario=mock_scenario,
                scores=scores,
                criteria_results=[criterion_result],
                action_log=[action_entry],
                turns_completed=5,
                duration=100.0,
                status="completed",
                participant="http://purple.example.com",
            )

            assert isinstance(result, AssessmentResults)
            assert result.assessment_id == "assess-123"
            assert result.scenario_id == mock_scenario.scenario_id
            assert result.participant == "http://purple.example.com"
            assert result.status == "completed"
            assert result.duration_seconds == 100.0
            assert result.turns_taken == 5
            assert result.actions_taken == 1
            assert result.scores == scores
            assert len(result.criteria_results) == 1
            assert len(result.action_log) == 1

    def test_build_results_maps_cancelled_to_failed(
        self,
        mock_scenario: MagicMock,
    ) -> None:
        """_build_results() should map 'cancelled' status to 'failed'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scores = Scores(
                overall=OverallScore(score=0, max_score=20),
                dimensions={"accuracy": DimensionScore(score=0, max_score=20)},
            )

            result = agent._build_results(
                assessment_id="assess-456",
                scenario=mock_scenario,
                scores=scores,
                criteria_results=[],
                action_log=[],
                turns_completed=2,
                duration=50.0,
                status="cancelled",
                participant="http://purple.example.com",
            )

            assert result.status == "failed"

    def test_build_results_empty_action_log(
        self,
        mock_scenario: MagicMock,
    ) -> None:
        """_build_results() should handle empty action log."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scores = Scores(
                overall=OverallScore(score=5, max_score=10),
                dimensions={"efficiency": DimensionScore(score=5, max_score=10)},
            )

            result = agent._build_results(
                assessment_id="assess-789",
                scenario=mock_scenario,
                scores=scores,
                criteria_results=[],
                action_log=[],
                turns_completed=0,
                duration=10.0,
                status="completed",
                participant="http://purple.example.com",
            )

            assert result.actions_taken == 0
            assert len(result.action_log) == 0


# =============================================================================
# Health Monitoring Tests
# =============================================================================


class TestHealthMonitoring:
    """Tests for health monitoring methods."""

    @pytest.mark.asyncio
    async def test_check_ues_health_delegates_to_server_manager(self) -> None:
        """_check_ues_health() delegates to UESServerManager.check_health()."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_server = MagicMock()
            mock_server.check_health = AsyncMock(return_value=True)
            agent._ues_server = mock_server

            result = await agent._check_ues_health()

            assert result is True
            mock_server.check_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_ues_health_returns_false_when_unhealthy(self) -> None:
        """_check_ues_health() returns False when UES server is unhealthy."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            mock_server = MagicMock()
            mock_server.check_health = AsyncMock(return_value=False)
            agent._ues_server = mock_server

            result = await agent._check_ues_health()

            assert result is False
            mock_server.check_health.assert_awaited_once()


# =============================================================================
# TurnResult Model Tests
# =============================================================================


class TestTurnResult:
    """Tests for the TurnResult dataclass."""

    def test_create_turn_result(self) -> None:
        """Should create a TurnResult with all required fields."""
        result = TurnResult(
            turn_number=1,
            actions_taken=5,
            time_step="PT1H",
            events_processed=10,
            early_completion=False,
        )
        assert result.turn_number == 1
        assert result.actions_taken == 5
        assert result.time_step == "PT1H"
        assert result.events_processed == 10
        assert result.early_completion is False
        assert result.notes is None
        assert result.error is None

    def test_turn_result_with_optional_fields(self) -> None:
        """Should create a TurnResult with optional fields."""
        result = TurnResult(
            turn_number=3,
            actions_taken=0,
            time_step="PT30M",
            events_processed=5,
            early_completion=True,
            notes="Purple signaled early completion",
            error=None,
        )
        assert result.early_completion is True
        assert result.notes == "Purple signaled early completion"

    def test_turn_result_with_error(self) -> None:
        """Should create a TurnResult with error field."""
        result = TurnResult(
            turn_number=2,
            actions_taken=0,
            time_step="PT0S",
            events_processed=0,
            early_completion=False,
            error="timeout",
        )
        assert result.error == "timeout"


# =============================================================================
# EndOfTurnResult Model Tests
# =============================================================================


class TestEndOfTurnResult:
    """Tests for the EndOfTurnResult dataclass."""

    def test_create_end_of_turn_result(self) -> None:
        """Should create an EndOfTurnResult with all fields."""
        result = EndOfTurnResult(
            actions_taken=3,
            total_events=15,
            responses_generated=2,
        )
        assert result.actions_taken == 3
        assert result.total_events == 15
        assert result.responses_generated == 2

    def test_end_of_turn_result_zero_values(self) -> None:
        """Should handle zero values correctly."""
        result = EndOfTurnResult(
            actions_taken=0,
            total_events=0,
            responses_generated=0,
        )
        assert result.actions_taken == 0
        assert result.total_events == 0
        assert result.responses_generated == 0


# =============================================================================
# ScheduledResponse Validation Tests
# =============================================================================


class TestScheduledResponseValidation:
    """Tests for ScheduledResponse validation in scheduling context."""

    def test_email_response_requires_email(self, now: datetime) -> None:
        """Email response should require character_email."""
        with pytest.raises(ValueError, match="character_email"):
            ScheduledResponse(
                modality="email",
                character_name="Alice",
                scheduled_time=now,
                content="Hello!",
                recipients=["user@example.com"],
            )

    def test_email_response_requires_content(self, now: datetime) -> None:
        """Email response should require content."""
        with pytest.raises(ValueError, match="content"):
            ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                recipients=["user@example.com"],
            )

    def test_email_response_requires_recipients(self, now: datetime) -> None:
        """Email response should require at least one recipient."""
        with pytest.raises(ValueError, match="recipient"):
            ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello!",
            )

    def test_sms_response_requires_phone(self, now: datetime) -> None:
        """SMS response should require character_phone."""
        with pytest.raises(ValueError, match="character_phone"):
            ScheduledResponse(
                modality="sms",
                character_name="Bob",
                scheduled_time=now,
                content="Hi!",
                recipients=["+15559876543"],
            )

    def test_calendar_response_requires_event_id(self, now: datetime) -> None:
        """Calendar response should require event_id."""
        with pytest.raises(ValueError, match="event_id"):
            ScheduledResponse(
                modality="calendar",
                character_name="Carol",
                character_email="carol@example.com",
                scheduled_time=now,
                rsvp_status="accepted",
            )

    def test_calendar_response_requires_rsvp_status(self, now: datetime) -> None:
        """Calendar response should require rsvp_status."""
        with pytest.raises(ValueError, match="rsvp_status"):
            ScheduledResponse(
                modality="calendar",
                character_name="Carol",
                character_email="carol@example.com",
                scheduled_time=now,
                event_id="event-123",
            )

    def test_valid_email_response(self, now: datetime) -> None:
        """Should create a valid email ScheduledResponse."""
        response = ScheduledResponse(
            modality="email",
            character_name="Alice",
            character_email="alice@example.com",
            scheduled_time=now,
            content="Hello!",
            recipients=["user@example.com"],
            subject="Re: Test",
            thread_id="thread-123",
        )
        assert response.modality == "email"
        assert response.character_name == "Alice"
        assert response.content == "Hello!"

    def test_valid_sms_response(self, now: datetime) -> None:
        """Should create a valid SMS ScheduledResponse."""
        response = ScheduledResponse(
            modality="sms",
            character_name="Bob",
            character_phone="+15551234567",
            scheduled_time=now,
            content="Hi there!",
            recipients=["+15559876543"],
        )
        assert response.modality == "sms"
        assert response.character_phone == "+15551234567"

    def test_valid_calendar_response(self, now: datetime) -> None:
        """Should create a valid calendar ScheduledResponse."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Carol",
            character_email="carol@example.com",
            scheduled_time=now,
            event_id="event-123",
            rsvp_status="accepted",
            rsvp_comment="Looking forward to it!",
        )
        assert response.modality == "calendar"
        assert response.rsvp_status == "accepted"


# =============================================================================
# Message Model Tests
# =============================================================================


class TestMessageModels:
    """Tests for message models used in Purple communication."""

    def test_turn_complete_message_defaults(self) -> None:
        """TurnCompleteMessage should have default time_step of PT1H."""
        msg = TurnCompleteMessage()
        assert msg.message_type == "turn_complete"
        assert msg.time_step == "PT1H"
        assert msg.notes is None

    def test_turn_complete_message_custom_time_step(self) -> None:
        """TurnCompleteMessage should accept custom time_step."""
        msg = TurnCompleteMessage(time_step="PT30M", notes="Custom step")
        assert msg.time_step == "PT30M"
        assert msg.notes == "Custom step"

    def test_early_completion_message(self) -> None:
        """EarlyCompletionMessage should have correct message_type."""
        msg = EarlyCompletionMessage(reason="All tasks completed")
        assert msg.message_type == "early_completion"
        assert msg.reason == "All tasks completed"


# =============================================================================
# InitialStateSummary Tests
# =============================================================================


class TestInitialStateSummary:
    """Tests for InitialStateSummary construction."""

    def test_create_initial_state_summary(self) -> None:
        """Should create a valid InitialStateSummary."""
        summary = InitialStateSummary(
            email=EmailSummary(total_emails=10, total_threads=5, unread=3, draft_count=1),
            calendar=CalendarSummary(event_count=20, calendar_count=2, events_today=4),
            sms=SMSSummary(total_messages=15, total_conversations=3, unread=2),
            chat=ChatSummary(total_messages=8, conversation_count=1),
        )
        assert summary.message_type == "initial_state_summary"
        assert summary.email.total_emails == 10
        assert summary.calendar.events_today == 4
        assert summary.sms.unread == 2
        assert summary.chat.conversation_count == 1

    def test_email_summary_message_type(self) -> None:
        """EmailSummary should have correct message_type."""
        summary = EmailSummary(total_emails=10, total_threads=5, unread=3, draft_count=1)
        assert summary.message_type == "email_summary"

    def test_calendar_summary_message_type(self) -> None:
        """CalendarSummary should have correct message_type."""
        summary = CalendarSummary(event_count=20, calendar_count=2, events_today=4)
        assert summary.message_type == "calendar_summary"


# =============================================================================
# Config Tests
# =============================================================================


class TestGreenAgentConfig:
    """Tests for GreenAgentConfig used by GreenAgent."""

    def test_default_config(self) -> None:
        """Should create config with default values."""
        config = GreenAgentConfig()
        assert config.port == 8000
        assert config.ues_base_port == 8080
        assert config.default_max_turns == 100
        assert config.default_turn_timeout == 300.0

    def test_custom_config(self) -> None:
        """Should create config with custom values."""
        config = GreenAgentConfig(
            port=9000,
            ues_base_port=9100,
            default_max_turns=50,
            default_turn_timeout=120.0,
        )
        assert config.port == 9000
        assert config.ues_base_port == 9100
        assert config.default_max_turns == 50
        assert config.default_turn_timeout == 120.0

    def test_llm_model_config(self) -> None:
        """Should configure LLM models."""
        config = GreenAgentConfig(
            response_generator_model="gpt-4o",
            summarization_model="gpt-3.5-turbo",
            evaluation_model="claude-3-opus",
        )
        assert config.response_generator_model == "gpt-4o"
        assert config.summarization_model == "gpt-3.5-turbo"
        assert config.evaluation_model == "claude-3-opus"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestRunFullFlow:
    """Tests for the full run() assessment flow with mocked internals."""

    def _make_agent_with_mocks(
        self,
        mock_ues_client: AsyncMock,
        config: GreenAgentConfig | None = None,
    ) -> GreenAgent:
        """Helper: create a GreenAgent with all internals mocked."""
        agent = GreenAgent.__new__(GreenAgent)
        agent.config = config or GreenAgentConfig()
        agent.ues_port = 8100
        agent.ues_client = mock_ues_client
        agent._ues_server = MagicMock()
        agent._ues_server.base_url = "http://127.0.0.1:8100"
        agent._ues_server.admin_api_key = "admin-key"
        agent._ues_port = 8100
        agent._proctor_api_key = "admin-key"
        agent._current_task_id = None
        agent._cancelled = False
        agent.response_llm = MagicMock()
        agent.evaluation_llm = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_run_completes_full_assessment(
        self,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
        mock_scenario: MagicMock,
    ) -> None:
        """run() should complete a full assessment and return results."""
        agent = self._make_agent_with_mocks(mock_ues_client)

        # Mock setup methods
        agent._setup_ues = AsyncMock()
        agent._create_user_api_key = AsyncMock(
            return_value=("secret-key", "key-id")
        )
        agent._revoke_user_api_key = AsyncMock()
        agent._capture_state_snapshot = AsyncMock(return_value={"email": {}})
        agent._build_initial_state_summary = AsyncMock(
            return_value=InitialStateSummary(
                email=EmailSummary(
                    total_emails=0, total_threads=0, unread=0, draft_count=0
                ),
                calendar=CalendarSummary(
                    event_count=0, calendar_count=0, events_today=0
                ),
                sms=SMSSummary(
                    total_messages=0, total_conversations=0, unread=0
                ),
                chat=ChatSummary(total_messages=0, conversation_count=0),
            )
        )
        agent._send_assessment_start = AsyncMock()
        agent._send_assessment_complete = AsyncMock()

        # Make _run_turn signal early completion on first turn
        agent._run_turn = AsyncMock(
            return_value=TurnResult(
                turn_number=1,
                actions_taken=0,
                time_step="PT1H",
                events_processed=0,
                early_completion=True,
                notes="All done",
            )
        )

        # Mock scenario criteria
        mock_scenario.criteria = []
        mock_scenario.characters = []
        mock_scenario.name = "Test Scenario"

        # Mock CriteriaJudge and helpers
        with (
            patch("src.green.agent.ActionLogBuilder") as mock_alb_cls,
            patch("src.green.agent.NewMessageCollector") as mock_nmc_cls,
            patch("src.green.agent.ResponseGenerator") as mock_rg_cls,
            patch("src.green.agent.CriteriaJudge") as mock_judge_cls,
        ):
            mock_judge = MagicMock()
            mock_judge.get_dimensions.return_value = ["accuracy"]
            mock_judge.evaluate_all = AsyncMock(return_value=[])
            mock_judge.aggregate_scores.return_value = Scores(
                overall=OverallScore(score=10, max_score=20),
                dimensions={"accuracy": DimensionScore(score=10, max_score=20)},
            )
            mock_judge_cls.return_value = mock_judge

            mock_alb = MagicMock()
            mock_alb.get_log.return_value = []
            mock_alb_cls.return_value = mock_alb

            mock_nmc = MagicMock()
            mock_nmc.initialize = AsyncMock()
            mock_nmc_cls.return_value = mock_nmc

            mock_rg_cls.return_value = MagicMock()

            result = await agent.run(
                task_id="task-1",
                emitter=mock_emitter,
                scenario=mock_scenario,
                evaluators={},
                purple_client=mock_purple_client,
                assessment_config={},
            )

            assert isinstance(result, AssessmentResults)
            assert result.status == "completed"
            assert result.scores.overall.score == 10
            agent._setup_ues.assert_awaited_once()
            agent._create_user_api_key.assert_awaited_once()
            agent._revoke_user_api_key.assert_awaited_once_with("key-id")
            agent._send_assessment_start.assert_awaited_once()
            agent._send_assessment_complete.assert_awaited_once()
            mock_emitter.assessment_started.assert_awaited_once()
            mock_emitter.assessment_completed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_max_turns_reached(
        self,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
        mock_scenario: MagicMock,
    ) -> None:
        """run() should stop at max_turns and set reason to max_turns_reached."""
        agent = self._make_agent_with_mocks(mock_ues_client)
        agent._setup_ues = AsyncMock()
        agent._create_user_api_key = AsyncMock(
            return_value=("secret-key", "key-id")
        )
        agent._revoke_user_api_key = AsyncMock()
        agent._capture_state_snapshot = AsyncMock(return_value={})
        agent._build_initial_state_summary = AsyncMock(
            return_value=InitialStateSummary(
                email=EmailSummary(
                    total_emails=0, total_threads=0, unread=0, draft_count=0
                ),
                calendar=CalendarSummary(
                    event_count=0, calendar_count=0, events_today=0
                ),
                sms=SMSSummary(
                    total_messages=0, total_conversations=0, unread=0
                ),
                chat=ChatSummary(total_messages=0, conversation_count=0),
            )
        )
        agent._send_assessment_start = AsyncMock()
        agent._send_assessment_complete = AsyncMock()

        # Never signal early completion
        agent._run_turn = AsyncMock(
            return_value=TurnResult(
                turn_number=1,
                actions_taken=1,
                time_step="PT1H",
                events_processed=2,
                early_completion=False,
            )
        )
        mock_scenario.criteria = []
        mock_scenario.characters = []
        mock_scenario.name = "Test"

        with (
            patch("src.green.agent.ActionLogBuilder") as mock_alb_cls,
            patch("src.green.agent.NewMessageCollector") as mock_nmc_cls,
            patch("src.green.agent.ResponseGenerator"),
            patch("src.green.agent.CriteriaJudge") as mock_judge_cls,
        ):
            mock_judge = MagicMock()
            mock_judge.get_dimensions.return_value = []
            mock_judge.evaluate_all = AsyncMock(return_value=[])
            mock_judge.aggregate_scores.return_value = Scores(
                overall=OverallScore(score=0, max_score=0),
                dimensions={},
            )
            mock_judge_cls.return_value = mock_judge
            mock_alb = MagicMock()
            mock_alb.get_log.return_value = []
            mock_alb_cls.return_value = mock_alb
            mock_nmc = MagicMock()
            mock_nmc.initialize = AsyncMock()
            mock_nmc_cls.return_value = mock_nmc

            result = await agent.run(
                task_id="task-1",
                emitter=mock_emitter,
                scenario=mock_scenario,
                evaluators={},
                purple_client=mock_purple_client,
                assessment_config={"max_turns": 3},
            )

            assert agent._run_turn.await_count == 3
            assert result.turns_taken == 3

    @pytest.mark.asyncio
    async def test_run_clears_task_id_in_finally(
        self,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
        mock_scenario: MagicMock,
    ) -> None:
        """run() should always reset _current_task_id in finally block."""
        agent = self._make_agent_with_mocks(mock_ues_client)
        agent._setup_ues = AsyncMock(side_effect=Exception("setup boom"))

        mock_scenario.criteria = []
        mock_scenario.characters = []
        mock_scenario.name = "Test"

        with (
            patch("src.green.agent.ActionLogBuilder"),
            patch("src.green.agent.NewMessageCollector"),
            patch("src.green.agent.ResponseGenerator"),
            patch("src.green.agent.CriteriaJudge"),
        ):
            with pytest.raises(Exception, match="setup boom"):
                await agent.run(
                    task_id="task-99",
                    emitter=mock_emitter,
                    scenario=mock_scenario,
                    evaluators={},
                    purple_client=mock_purple_client,
                    assessment_config={},
                )

        # _current_task_id should be cleared by finally
        assert agent._current_task_id is None

    @pytest.mark.asyncio
    async def test_run_revokes_api_key_on_error(
        self,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
        mock_scenario: MagicMock,
    ) -> None:
        """run() should revoke API key even when an error occurs."""
        agent = self._make_agent_with_mocks(mock_ues_client)
        agent._setup_ues = AsyncMock()
        agent._create_user_api_key = AsyncMock(
            return_value=("secret", "key-to-revoke")
        )
        agent._revoke_user_api_key = AsyncMock()
        agent._capture_state_snapshot = AsyncMock(side_effect=Exception("snap boom"))

        mock_scenario.criteria = []
        mock_scenario.characters = []
        mock_scenario.name = "Test"

        with (
            patch("src.green.agent.ActionLogBuilder"),
            patch("src.green.agent.NewMessageCollector") as mock_nmc_cls,
            patch("src.green.agent.ResponseGenerator"),
            patch("src.green.agent.CriteriaJudge"),
        ):
            mock_nmc = MagicMock()
            mock_nmc.initialize = AsyncMock()
            mock_nmc_cls.return_value = mock_nmc

            with pytest.raises(Exception, match="snap boom"):
                await agent.run(
                    task_id="task-1",
                    emitter=mock_emitter,
                    scenario=mock_scenario,
                    evaluators={},
                    purple_client=mock_purple_client,
                    assessment_config={},
                )

        agent._revoke_user_api_key.assert_awaited_with("key-to-revoke")


class TestShutdownErrorHandling:
    """Tests for shutdown edge cases."""

    @pytest.mark.asyncio
    async def test_shutdown_handles_client_close_error(
        self,
        config: GreenAgentConfig,
        mock_ues_server_manager: MagicMock,
    ) -> None:
        """shutdown() should handle client close() errors gracefully."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch(
                "src.green.agent.UESServerManager",
                return_value=mock_ues_server_manager,
            ),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            mock_client = AsyncMock()
            mock_client.close = AsyncMock(side_effect=Exception("close error"))
            agent.ues_client = mock_client

            # Should not raise
            await agent.shutdown()

            assert agent.ues_client is None
            mock_ues_server_manager.stop.assert_awaited_once()


class TestCancelEdgeCases:
    """Tests for cancel edge cases."""

    @pytest.mark.asyncio
    async def test_cancel_with_no_current_task(
        self,
        config: GreenAgentConfig,
    ) -> None:
        """cancel() should be a no-op when no task is running."""
        with (
            patch("src.green.agent.LLMFactory") as mock_factory,
            patch("src.green.agent.UESServerManager"),
        ):
            mock_factory.create.return_value = MagicMock()

            agent = GreenAgent(ues_port=8100, config=config)
            assert agent._current_task_id is None
            assert agent._cancelled is False

            await agent.cancel("some-task")

            assert agent._cancelled is False


class TestProcessTurnEndWithResponses:
    """Tests for _process_turn_end when responses are generated."""

    @pytest.mark.asyncio
    async def test_process_turn_end_schedules_generated_responses(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_process_turn_end() should schedule each generated response."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client

            agent._advance_time = AsyncMock(
                return_value=MagicMock(events_executed=0)
            )
            agent._advance_remainder = AsyncMock(
                return_value=MagicMock(events_executed=0)
            )
            agent._schedule_response = AsyncMock()

            mock_action_log = MagicMock()
            mock_action_log.add_events_from_turn.return_value = ([], [])

            mock_message_collector = MagicMock()
            mock_message_collector.collect = AsyncMock(return_value=[])

            response1 = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Reply 1",
                recipients=["user@example.com"],
            )
            response2 = ScheduledResponse(
                modality="sms",
                character_name="Bob",
                character_phone="+15551234567",
                scheduled_time=now,
                content="Reply 2",
                recipients=["+15559876543"],
            )
            mock_response_gen = MagicMock()
            mock_response_gen.process_new_messages = AsyncMock(
                return_value=[response1, response2]
            )

            result = await agent._process_turn_end(
                turn=1,
                turn_start_time=now,
                time_step="PT1H",
                emitter=mock_emitter,
                action_log_builder=mock_action_log,
                message_collector=mock_message_collector,
                response_generator=mock_response_gen,
            )

            assert result.responses_generated == 2
            assert agent._schedule_response.await_count == 2
            mock_emitter.responses_generated.assert_awaited_once()
            call_kwargs = mock_emitter.responses_generated.call_args.kwargs
            assert call_kwargs["responses_count"] == 2
            assert set(call_kwargs["characters_involved"]) == {"Alice", "Bob"}


class TestExtractResponseDataEdgeCases:
    """Tests for _extract_response_data with various response types."""

    def test_extract_from_message_response(self) -> None:
        """_extract_response_data() should extract data from Message response."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            from a2a.types import DataPart, Message, Part, Role

            expected = {"message_type": "turn_complete"}
            msg = Message(
                role=Role.agent,
                parts=[Part(root=DataPart(data=expected))],
                message_id="msg-1",
            )

            result = agent._extract_response_data(msg)
            assert result == expected

    def test_extract_from_duck_typed_response(self) -> None:
        """_extract_response_data() should handle duck-typed objects with parts."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            expected = {"message_type": "turn_complete"}
            mock_part = MagicMock()
            mock_part.root = MagicMock()
            mock_part.root.data = expected

            mock_response = MagicMock()
            # Not a Task or Message instance, but has parts
            mock_response.__class__ = type("CustomResponse", (), {})
            mock_response.artifacts = []  # empty list, not None
            mock_response.parts = [mock_part]

            result = agent._extract_response_data(mock_response)
            assert result == expected


class TestSendAssessmentCompleteReasonMapping:
    """Tests for _send_assessment_complete reason mapping edge cases."""

    @pytest.mark.asyncio
    async def test_maps_max_turns_reached_to_timeout(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should map 'max_turns_reached' to 'timeout'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            await agent._send_assessment_complete(
                purple_client=mock_purple_client,
                reason="max_turns_reached",
            )

            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_maps_error_to_error(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should map 'error' to 'error'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            await agent._send_assessment_complete(
                purple_client=mock_purple_client,
                reason="error",
            )

            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["reason"] == "error"

    @pytest.mark.asyncio
    async def test_maps_unknown_reason_to_timeout(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should map unknown reasons to 'timeout'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            await agent._send_assessment_complete(
                purple_client=mock_purple_client,
                reason="something_unexpected",
            )

            call_kwargs = mock_purple_client.send_data.call_args
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            assert sent_data["reason"] == "timeout"


class TestBuildResultsStatusMapping:
    """Tests for _build_results status mapping edge cases."""

    def test_maps_timeout_status(self, mock_scenario: MagicMock) -> None:
        """_build_results() should map 'timeout' status correctly."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            scores = Scores(
                overall=OverallScore(score=0, max_score=10),
                dimensions={"d": DimensionScore(score=0, max_score=10)},
            )
            result = agent._build_results(
                assessment_id="a1",
                scenario=mock_scenario,
                scores=scores,
                criteria_results=[],
                action_log=[],
                turns_completed=0,
                duration=0,
                status="timeout",
                participant="http://p.example.com",
            )
            assert result.status == "timeout"

    def test_maps_unknown_status_to_failed(
        self, mock_scenario: MagicMock
    ) -> None:
        """_build_results() should map unknown status to 'failed'."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            scores = Scores(
                overall=OverallScore(score=0, max_score=10),
                dimensions={"d": DimensionScore(score=0, max_score=10)},
            )
            result = agent._build_results(
                assessment_id="a1",
                scenario=mock_scenario,
                scores=scores,
                criteria_results=[],
                action_log=[],
                turns_completed=0,
                duration=0,
                status="something_weird",
                participant="http://p.example.com",
            )
            assert result.status == "failed"


class TestCountEventsTodayEdgeCases:
    """Tests for _count_events_today edge cases."""

    def test_count_multiple_events_same_day(self, now: datetime) -> None:
        """_count_events_today() should count all events on the same date."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            cal_state = MagicMock()
            evt1 = MagicMock()
            evt1.start = now.replace(hour=8)
            evt2 = MagicMock()
            evt2.start = now.replace(hour=14)
            evt3 = MagicMock()
            evt3.start = now.replace(hour=23)
            cal_state.events = {"a": evt1, "b": evt2, "c": evt3}

            result = agent._count_events_today(cal_state, now)
            assert result == 3

    def test_count_events_with_string_event_start(self, now: datetime) -> None:
        """_count_events_today() should handle events with string start times."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)

            cal_state = MagicMock()
            evt = MagicMock()
            evt.start = "2026-02-09T15:00:00+00:00"
            cal_state.events = {"a": evt}

            result = agent._count_events_today(cal_state, now)
            assert result == 1


class TestRunTurnTimeStepDefault:
    """Tests for _run_turn time_step handling."""

    @pytest.mark.asyncio
    async def test_run_turn_uses_default_time_step(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_run_turn() should use default PT1H time_step from TurnCompleteMessage."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent.config = GreenAgentConfig(default_turn_timeout=60.0)

            # TurnComplete with default time_step (PT1H)
            turn_complete = TurnCompleteMessage()
            agent._send_and_wait_purple = AsyncMock(return_value=turn_complete)
            agent._process_turn_end = AsyncMock(
                return_value=EndOfTurnResult(
                    actions_taken=0, total_events=0, responses_generated=0
                )
            )

            result = await agent._run_turn(
                turn=1,
                emitter=mock_emitter,
                purple_client=mock_purple_client,
                action_log_builder=MagicMock(),
                message_collector=MagicMock(),
                response_generator=MagicMock(),
            )

            assert result.time_step == "PT1H"
            agent._process_turn_end.assert_awaited_once()
            call_kwargs = agent._process_turn_end.call_args.kwargs
            assert call_kwargs["time_step"] == "PT1H"

    @pytest.mark.asyncio
    async def test_run_turn_uses_custom_time_step(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
    ) -> None:
        """_run_turn() should use custom time_step from TurnCompleteMessage."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            agent.ues_client = mock_ues_client
            agent.config = GreenAgentConfig(default_turn_timeout=60.0)

            turn_complete = TurnCompleteMessage(time_step="PT30M")
            agent._send_and_wait_purple = AsyncMock(return_value=turn_complete)
            agent._process_turn_end = AsyncMock(
                return_value=EndOfTurnResult(
                    actions_taken=0, total_events=0, responses_generated=0
                )
            )

            result = await agent._run_turn(
                turn=1,
                emitter=mock_emitter,
                purple_client=mock_purple_client,
                action_log_builder=MagicMock(),
                message_collector=MagicMock(),
                response_generator=MagicMock(),
            )

            assert result.time_step == "PT30M"
            call_kwargs = agent._process_turn_end.call_args.kwargs
            assert call_kwargs["time_step"] == "PT30M"
