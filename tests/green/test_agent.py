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
    async def test_run_raises_not_implemented(
        self,
        config: GreenAgentConfig,
        mock_emitter: AsyncMock,
        mock_scenario: MagicMock,
        mock_purple_client: AsyncMock,
    ) -> None:
        """run() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
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
    async def test_run_turn_raises_not_implemented(
        self,
        mock_emitter: AsyncMock,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_run_turn() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._run_turn(
                    turn=1,
                    emitter=mock_emitter,
                    purple_client=mock_purple_client,
                    action_log_builder=MagicMock(),
                    message_collector=MagicMock(),
                    response_generator=MagicMock(),
                )

    @pytest.mark.asyncio
    async def test_process_turn_end_raises_not_implemented(
        self,
        now: datetime,
        mock_emitter: AsyncMock,
    ) -> None:
        """_process_turn_end() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._process_turn_end(
                    turn=1,
                    turn_start_time=now,
                    time_step="PT1H",
                    emitter=mock_emitter,
                    action_log_builder=MagicMock(),
                    message_collector=MagicMock(),
                    response_generator=MagicMock(),
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
    async def test_send_and_wait_purple_raises_not_implemented(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_and_wait_purple() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._send_and_wait_purple(
                    purple_client=mock_purple_client,
                    message=MagicMock(),
                    timeout=60.0,
                )

    def test_extract_response_data_raises_not_implemented(self) -> None:
        """_extract_response_data() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                agent._extract_response_data(MagicMock())

    @pytest.mark.asyncio
    async def test_send_assessment_start_raises_not_implemented(
        self,
        mock_purple_client: AsyncMock,
        mock_scenario: MagicMock,
    ) -> None:
        """_send_assessment_start() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            initial_summary = InitialStateSummary(
                email=EmailSummary(total_emails=0, total_threads=0, unread=0, draft_count=0),
                calendar=CalendarSummary(event_count=0, calendar_count=0, events_today=0),
                sms=SMSSummary(total_messages=0, total_conversations=0, unread=0),
                chat=ChatSummary(total_messages=0, conversation_count=0),
            )
            with pytest.raises(NotImplementedError):
                await agent._send_assessment_start(
                    purple_client=mock_purple_client,
                    scenario=mock_scenario,
                    initial_summary=initial_summary,
                    ues_url="http://127.0.0.1:8100",
                    api_key="test-key",
                )

    @pytest.mark.asyncio
    async def test_send_assessment_complete_raises_not_implemented(
        self,
        mock_purple_client: AsyncMock,
    ) -> None:
        """_send_assessment_complete() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._send_assessment_complete(
                    purple_client=mock_purple_client,
                    reason="scenario_complete",
                )


# =============================================================================
# API Key Management Tests
# =============================================================================


class TestAPIKeyManagement:
    """Tests for API key management methods."""

    @pytest.mark.asyncio
    async def test_create_user_api_key_raises_not_implemented(self) -> None:
        """_create_user_api_key() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._create_user_api_key("assessment-123")

    @pytest.mark.asyncio
    async def test_revoke_user_api_key_raises_not_implemented(self) -> None:
        """_revoke_user_api_key() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._revoke_user_api_key("key-123")


# =============================================================================
# UES Setup Tests
# =============================================================================


class TestUESSetup:
    """Tests for UES setup and state management methods."""

    @pytest.mark.asyncio
    async def test_setup_ues_raises_not_implemented(
        self,
        mock_scenario: MagicMock,
    ) -> None:
        """_setup_ues() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._setup_ues(mock_scenario)


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management methods."""

    @pytest.mark.asyncio
    async def test_capture_state_snapshot_raises_not_implemented(self) -> None:
        """_capture_state_snapshot() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._capture_state_snapshot()

    @pytest.mark.asyncio
    async def test_build_initial_state_summary_raises_not_implemented(self) -> None:
        """_build_initial_state_summary() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._build_initial_state_summary()

    def test_count_events_today_raises_not_implemented(self) -> None:
        """_count_events_today() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                agent._count_events_today(MagicMock())


# =============================================================================
# Response Scheduling Tests
# =============================================================================


class TestResponseScheduling:
    """Tests for response scheduling methods."""

    @pytest.mark.asyncio
    async def test_schedule_response_raises_not_implemented(
        self,
        now: datetime,
    ) -> None:
        """_schedule_response() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scheduled = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello!",
                recipients=["user@example.com"],
            )
            with pytest.raises(NotImplementedError):
                await agent._schedule_response(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_email_response_raises_not_implemented(
        self,
        now: datetime,
    ) -> None:
        """_schedule_email_response() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scheduled = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello!",
                recipients=["user@example.com"],
            )
            with pytest.raises(NotImplementedError):
                await agent._schedule_email_response(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_sms_response_raises_not_implemented(
        self,
        now: datetime,
    ) -> None:
        """_schedule_sms_response() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scheduled = ScheduledResponse(
                modality="sms",
                character_name="Bob",
                character_phone="+15551234567",
                scheduled_time=now,
                content="Hi there!",
                recipients=["+15559876543"],
            )
            with pytest.raises(NotImplementedError):
                await agent._schedule_sms_response(scheduled)

    @pytest.mark.asyncio
    async def test_schedule_calendar_response_raises_not_implemented(
        self,
        now: datetime,
    ) -> None:
        """_schedule_calendar_response() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scheduled = ScheduledResponse(
                modality="calendar",
                character_name="Carol",
                character_email="carol@example.com",
                scheduled_time=now,
                event_id="event-123",
                rsvp_status="accepted",
            )
            with pytest.raises(NotImplementedError):
                await agent._schedule_calendar_response(scheduled)


# =============================================================================
# Result Building Tests
# =============================================================================


class TestResultBuilding:
    """Tests for result building methods."""

    def test_build_results_raises_not_implemented(
        self,
        mock_scenario: MagicMock,
    ) -> None:
        """_build_results() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            scores = Scores(
                overall=OverallScore(score=10, max_score=20),
                dimensions={"accuracy": DimensionScore(score=10, max_score=20)},
            )
            with pytest.raises(NotImplementedError):
                agent._build_results(
                    assessment_id="assess-123",
                    scenario=mock_scenario,
                    scores=scores,
                    criteria_results=[],
                    action_log=[],
                    turns_completed=5,
                    duration=100.0,
                    status="completed",
                )


# =============================================================================
# Health Monitoring Tests
# =============================================================================


class TestHealthMonitoring:
    """Tests for health monitoring methods."""

    @pytest.mark.asyncio
    async def test_check_ues_health_raises_not_implemented(self) -> None:
        """_check_ues_health() should raise NotImplementedError (stubbed)."""
        with patch.object(GreenAgent, "__init__", lambda self, **kwargs: None):
            agent = GreenAgent.__new__(GreenAgent)
            with pytest.raises(NotImplementedError):
                await agent._check_ues_health()


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
