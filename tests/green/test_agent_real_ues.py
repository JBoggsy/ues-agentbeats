"""Real UES server tests for GreenAgent.

These tests exercise the GreenAgent against a real UES server subprocess
with mocked LLMs and mock Purple clients. They validate correctness of
the GreenAgent↔UES interaction layer — the most complex and error-prone
part of the system — without requiring LLM API keys.

Test categories:
    - Two-phase time advancement with event execution
    - API key permissions enforcement (user vs proctor)
    - Response scheduling creates visible modality state
    - Event listing with agent_id filtering
    - Scenario import, clear, and restart cycle
    - Full assessment flow with real UES + mocked LLMs

All tests are marked ``@pytest.mark.slow`` and ``@pytest.mark.integration``
for selective execution::

    # Run only real UES tests
    uv run pytest tests/green/test_agent_real_ues.py -m "slow" -v

    # Skip slow tests
    uv run pytest -m "not slow"

Note:
    These tests start real UES server subprocesses. Each test class uses
    a unique port range (9300+) to avoid conflicts with other tests.
    The ``ues_server`` fixtures handle startup and teardown automatically.

See Also:
    - ``docs/design/GREEN_AGENT_DESIGN_PLAN.md`` § 16.3 (Testing)
    - ``tests/green/test_agent_integration.py`` for mocked-UES tests
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.messages import (
    InitialStateSummary,
)
from src.common.agentbeats.results import (
    AssessmentResults,
    CriterionResult,
    DimensionScore,
    OverallScore,
    Scores,
)
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.agent import GreenAgent, USER_PERMISSIONS
from src.green.core.ues_server import UESServerManager
from src.green.response.models import ScheduledResponse
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    EvaluatorRegistry,
    ResponseTiming,
    ScenarioConfig,
)


# ================================================================
# Shared Constants
# ================================================================

# Port range for real UES tests — avoids conflicts with integration tests
_BASE_PORT = 9300


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
        ues_base_port=_BASE_PORT,
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
    """Create a simple scenario with initial state for real UES testing."""
    return ScenarioConfig(
        scenario_id="test_real_ues",
        name="Real UES Test Scenario",
        description="A scenario for testing against a real UES instance",
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
                "description": "Real UES test scenario",
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
                        "user_email_address": "testuser@example.com",
                    },
                    "sms": {
                        "modality_type": "sms",
                        "last_updated": "2026-02-09T09:00:00+00:00",
                        "update_count": 0,
                        "messages": {},
                        "conversations": {},
                        "max_messages_per_conversation": 10000,
                        "user_phone_number": "+15550001111",
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

    async def mock_evaluator(context: Any, params: Any = None) -> Any:
        from src.green.scenarios.schema import EvalResult

        return EvalResult(
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
def mock_emitter() -> TaskUpdateEmitter:
    """Create a ``TaskUpdateEmitter`` with mock updater."""
    updater = MockTaskUpdater()
    return TaskUpdateEmitter(updater)  # type: ignore[arg-type]


# ================================================================
# UES Server Fixture
# ================================================================


@pytest.fixture
async def ues_server(request: pytest.FixtureRequest) -> UESServerManager:
    """Start a real UES server and yield the manager.

    Each test gets a unique port based on its parameter marker index
    to avoid conflicts. Falls back to _BASE_PORT if no index given.

    Yields:
        A running ``UESServerManager`` instance.
    """
    # Use a unique port per test to avoid conflicts
    port = getattr(request, "param", _BASE_PORT)
    manager = UESServerManager(port=port)
    await manager.start(ready_timeout=30.0)
    yield manager
    await manager.stop()


def _make_agent_with_real_ues(
    config: GreenAgentConfig,
    ues_server: UESServerManager,
) -> GreenAgent:
    """Build a ``GreenAgent`` with a real UES server but mocked LLMs.

    The LLMFactory is patched during construction so no API keys are
    needed. The agent's UES server manager is replaced with the
    fixture-provided one.

    Returns:
        A ``GreenAgent`` ready for ``startup()``.
    """
    with patch("src.green.agent.LLMFactory") as mock_factory:
        mock_factory.create.return_value = MagicMock()

        # Patch UESServerManager to use the fixture-provided one
        with patch(
            "src.green.agent.UESServerManager",
            return_value=ues_server,
        ):
            agent = GreenAgent(ues_port=ues_server.port, config=config)

    return agent


def _make_scores(**kwargs: Any) -> Scores:
    """Create minimal ``Scores``."""
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


def _early_completion_response(
    reason: str = "Done",
) -> MagicMock:
    """Build a mock A2A response that duck-types as early completion."""
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


# ================================================================
# Test: Scenario Import, Clear, and Restart Cycle
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestScenarioImportClearRestart:
    """Verify _setup_ues() with a real UES server.

    Tests that clear → import → start works correctly and state
    matches the scenario. Also tests that re-clearing resets state.
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9300)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_setup_ues_loads_scenario_state(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_setup_ues() loads scenario and UES state matches."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Verify time was set correctly
            time_state = await agent.ues_client.time.get_state()
            assert time_state.current_time == datetime(
                2026, 2, 9, 9, 0, tzinfo=timezone.utc
            )

            # Verify email modality was loaded
            email_state = await agent.ues_client.email.get_state()
            assert email_state.user_email_address == "testuser@example.com"

            # Verify SMS modality was loaded
            sms_state = await agent.ues_client.sms.get_state()
            assert sms_state.user_phone_number == "+15550001111"

            # Verify calendar modality was loaded
            cal_state = await agent.ues_client.calendar.get_state()
            assert len(cal_state.calendars) >= 1
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_setup_ues_can_clear_and_reimport(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """Calling _setup_ues() twice resets state each time."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            # First setup
            await agent._setup_ues(simple_scenario)

            # Inject some extra state
            await agent.ues_client.email.receive(
                from_address="extra@example.com",
                to_addresses=["testuser@example.com"],
                subject="Extra email",
                body_text="This should be cleared",
            )

            # Verify extra email exists
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count >= 1

            # Second setup — should clear everything
            await agent._setup_ues(simple_scenario)

            # Verify extra email is gone
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count == 0
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_initial_state_summary_matches_scenario(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_build_initial_state_summary() reflects loaded scenario."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            summary = await agent._build_initial_state_summary()

            assert isinstance(summary, InitialStateSummary)
            # Empty scenario — all counts should be zero
            assert summary.email.total_emails == 0
            assert summary.email.unread == 0
            assert summary.sms.total_messages == 0
            assert summary.calendar.event_count == 0
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: Two-Phase Time Advancement
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestTwoPhaseTimeAdvancement:
    """Verify the two-phase time advancement strategy with real UES.

    The core mechanic:
    1. Create events at time T+0.5s (simulating Purple actions)
    2. Apply 1s advance → events should fire
    3. Create "character response" events at T+30m
    4. Advance by remainder → character responses should fire
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9301)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_apply_advance_fires_immediate_events(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """1s apply advance fires events scheduled between T and T+1s."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Current time is 09:00. Schedule an email.receive at 09:00:00
            # (immediate via the email API — this is a proctor action).
            await agent.ues_client.email.receive(
                from_address="test@example.com",
                to_addresses=["testuser@example.com"],
                subject="Apply test",
                body_text="Should be visible after 1s advance",
            )

            # Advance by 1 second (the "apply" phase)
            apply_result = await agent._advance_time("PT1S")

            # Email should now be visible in state
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count >= 1

            # Time should have advanced by 1 second
            time_state = await agent.ues_client.time.get_state()
            assert time_state.current_time == datetime(
                2026, 2, 9, 9, 0, 1, tzinfo=timezone.utc
            )
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_remainder_advance_fires_scheduled_events(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """Remainder advance fires events scheduled during the gap."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Phase 1: Apply advance (1s)
            await agent._advance_time("PT1S")

            # Schedule a "character response" email at T+30m
            response_time = datetime(
                2026, 2, 9, 9, 30, 0, tzinfo=timezone.utc
            )
            await agent.ues_client.events.create(
                scheduled_time=response_time,
                modality="email",
                data={
                    "operation": "receive",
                    "from_address": "alice@example.com",
                    "to_addresses": ["testuser@example.com"],
                    "subject": "Character response",
                    "body_text": "Response from Alice",
                },
            )

            # Phase 2: Remainder advance (PT1H minus 1s = 3599s)
            remainder_result = await agent._advance_remainder(
                "PT1H", apply_seconds=1
            )

            # The scheduled event should have fired
            assert remainder_result.events_executed >= 1

            # Verify the email is now in the mailbox
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count >= 1

            # Time should be at T+1H total
            time_state = await agent.ues_client.time.get_state()
            assert time_state.current_time == datetime(
                2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc
            )
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_two_phase_full_cycle(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """Full two-phase cycle: apply fires Purple events, remainder fires responses."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Simulate Purple sending an email (happens before time advance)
            await agent.ues_client.email.send(
                from_address="testuser@example.com",
                to_addresses=["alice@example.com"],
                subject="Purple action",
                body_text="Hello Alice",
            )

            # Phase 1: Apply (1s) — Purple's events fire
            await agent._advance_time("PT1S")

            # Purple's sent email should exist
            email_state = await agent.ues_client.email.get_state()
            sent_count = len(email_state.folders.get("sent", []))
            assert sent_count >= 1

            # Green schedules a character response at T+30m
            response_time = datetime(
                2026, 2, 9, 9, 30, 0, tzinfo=timezone.utc
            )
            await agent.ues_client.email.receive(
                from_address="alice@example.com",
                to_addresses=["testuser@example.com"],
                subject="Re: Purple action",
                body_text="Hi! Got your email.",
                sent_at=response_time,
            )

            # Phase 2: Remainder advance
            await agent._advance_remainder("PT1H", apply_seconds=1)

            # Both emails should exist
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count >= 2
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: API Key Permissions Enforcement
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestAPIKeyPermissionsEnforcement:
    """Verify that Purple agent API keys have correct permissions.

    Security-critical: validates that a key created with USER_PERMISSIONS
    can perform user operations but cannot perform proctor operations.
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9302)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_user_key_can_read_time(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key can read simulation time (time:read permission)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Create a user-level key
            secret, key_id = await agent._create_user_api_key("perm-test")

            # Use the user key to read time
            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                time_state = await user_client.time.get_state()
                assert time_state.current_time is not None
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_can_send_email(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key can send emails (email:send permission)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-2")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                # email:send is in USER_PERMISSIONS
                result = await user_client.email.send(
                    from_address="testuser@example.com",
                    to_addresses=["alice@example.com"],
                    subject="Permission test",
                    body_text="Testing email:send permission",
                )
                # No exception means permission was granted
                assert result is not None
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_can_query_email(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key can query email state (email:state permission)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-3")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                email_state = await user_client.email.get_state()
                assert email_state is not None
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_cannot_advance_time(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key CANNOT advance time (time:advance is proctor-only)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-4")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                with pytest.raises(Exception):
                    # time:advance is NOT in USER_PERMISSIONS
                    await user_client.time.advance(seconds=60)
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_cannot_receive_email(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key CANNOT inject emails (email:receive is proctor-only)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-5")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                with pytest.raises(Exception):
                    # email:receive is NOT in USER_PERMISSIONS
                    await user_client.email.receive(
                        from_address="attacker@example.com",
                        to_addresses=["testuser@example.com"],
                        subject="Spoofed email",
                        body_text="This should be forbidden",
                    )
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_cannot_clear_simulation(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key CANNOT clear simulation (simulation:* is proctor-only)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-6")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                with pytest.raises(Exception):
                    await user_client.simulation.clear()
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_user_key_can_create_calendar_event(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """User key can create calendar events (calendar:create permission)."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("perm-test-7")

            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                result = await user_client.calendar.create(
                    title="Test Event",
                    start=datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 2, 9, 15, 0, tzinfo=timezone.utc),
                )
                assert result is not None
            finally:
                await user_client.close()
                await agent._revoke_user_api_key(key_id)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_revoked_key_cannot_access_api(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """A revoked key cannot be used to access the API."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            secret, key_id = await agent._create_user_api_key("revoke-test")

            # Revoke the key
            await agent._revoke_user_api_key(key_id)

            # Try to use the revoked key
            user_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=secret,
            )
            try:
                with pytest.raises(Exception):
                    await user_client.time.get_state()
            finally:
                await user_client.close()
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: Response Scheduling Creates Visible State
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestResponseSchedulingVisibility:
    """Verify that scheduled responses actually appear in UES state.

    Tests that email.receive(), sms.receive(), and
    calendar.respond_to_event() inject content that is visible
    when querying modality state.
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9303)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_email_receive_visible_in_state(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_schedule_email_response() makes email visible in inbox."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            now = datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)
            scheduled = ScheduledResponse(
                modality="email",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                content="Hello from Alice!",
                recipients=["testuser@example.com"],
                subject="Test email",
            )
            await agent._schedule_email_response(scheduled)

            # Verify email is in the inbox
            email_state = await agent.ues_client.email.get_state()
            assert email_state.total_email_count >= 1

            # Verify email content via query
            query_result = await agent.ues_client.email.query(
                from_address="alice@example.com"
            )
            assert len(query_result.emails) >= 1
            email = query_result.emails[0]
            assert email.subject == "Test email"
            assert "Hello from Alice" in email.body_text
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_sms_receive_visible_in_state(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_schedule_sms_response() makes SMS visible in conversations."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            now = datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)
            scheduled = ScheduledResponse(
                modality="sms",
                character_name="Bob",
                character_phone="+15551234567",
                scheduled_time=now,
                content="Hey, this is Bob!",
                recipients=["+15550001111"],
            )
            await agent._schedule_sms_response(scheduled)

            # Verify SMS is visible
            sms_state = await agent.ues_client.sms.get_state()
            assert sms_state.total_message_count >= 1

            # Query for the message
            query_result = await agent.ues_client.sms.query(
                from_number="+15551234567"
            )
            assert len(query_result.messages) >= 1
            msg = query_result.messages[0]
            assert "Hey, this is Bob" in msg.body
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_calendar_respond_visible_in_state(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_schedule_calendar_response() calls respond_to_event.

        Verifies the Green Agent correctly calls the calendar respond
        API and the event is created in the calendar.
        """
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Create a calendar event with an attendee
            await agent.ues_client.calendar.create(
                title="Team Meeting",
                start=datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc),
                end=datetime(2026, 2, 9, 15, 0, tzinfo=timezone.utc),
                attendees=[
                    {
                        "email": "alice@example.com",
                        "name": "Alice",
                        "response": "needs-action",
                    }
                ],
            )

            # Find the calendar event ID from state
            cal_state = await agent.ues_client.calendar.get_state()
            cal_event_id = None
            for eid, evt in cal_state.events.items():
                if evt.title == "Team Meeting":
                    cal_event_id = eid
                    break
            assert cal_event_id is not None

            # Schedule a calendar RSVP via the Green Agent method
            now = datetime(2026, 2, 9, 10, 0, 0, tzinfo=timezone.utc)
            scheduled = ScheduledResponse(
                modality="calendar",
                character_name="Alice",
                character_email="alice@example.com",
                scheduled_time=now,
                event_id=cal_event_id,
                rsvp_status="accepted",
                rsvp_comment="I'll be there!",
            )

            # Should not raise — the API call completes successfully
            await agent._schedule_calendar_response(scheduled)

            # Verify the calendar event still exists (not corrupted)
            cal_state = await agent.ues_client.calendar.get_state()
            event = cal_state.events.get(cal_event_id)
            assert event is not None
            assert event.title == "Team Meeting"
            assert len(event.attendees) == 1
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: Event Listing and Filtering
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestEventListingAndFiltering:
    """Verify event listing with time and status filters.

    Tests the events.list_events() call that _process_turn_end()
    relies on for building the action log.
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9304)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_list_events_filters_by_time_range(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """events.list_events() correctly filters by start_time/end_time."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Create and execute some events by sending emails
            await agent.ues_client.email.send(
                from_address="testuser@example.com",
                to_addresses=["alice@example.com"],
                subject="Event 1",
                body_text="First email",
            )

            # Advance time to execute events
            start_time = datetime(
                2026, 2, 9, 9, 0, 0, tzinfo=timezone.utc
            )
            await agent.ues_client.time.advance(seconds=60)
            end_time = datetime(
                2026, 2, 9, 9, 1, 0, tzinfo=timezone.utc
            )

            # List executed events in the time range
            events = await agent.ues_client.events.list_events(
                start_time=start_time,
                end_time=end_time,
                status="executed",
            )

            # Should have at least one executed event
            assert events.total >= 1
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_list_events_status_filter(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """events.list_events(status='pending') returns only pending events."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Schedule an event in the future
            future_time = datetime(
                2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc
            )
            await agent.ues_client.events.create(
                scheduled_time=future_time,
                modality="email",
                data={
                    "operation": "receive",
                    "from_address": "future@example.com",
                    "to_addresses": ["testuser@example.com"],
                    "subject": "Future email",
                    "body_text": "Scheduled for later",
                },
            )

            # Listing pending events should include it
            pending_events = await agent.ues_client.events.list_events(
                status="pending"
            )
            assert pending_events.total >= 1

            # Listing executed events should not include it
            executed_events = await agent.ues_client.events.list_events(
                status="executed"
            )
            # The future event should not be in executed
            future_in_executed = any(
                e.scheduled_time >= future_time
                for e in executed_events.events
            )
            assert not future_in_executed
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_events_model_dump_includes_expected_fields(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """Event responses include fields expected by ActionLogBuilder.

        Verifies that ``model_dump()`` includes all fields needed by
        ``ActionLogBuilder.add_events_from_turn()``, including
        ``agent_id`` (previously missing — see
        ``docs/UPSTREAM_BUG_EVENT_AGENT_ID.md``).
        """
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Create an action
            await agent.ues_client.email.send(
                from_address="testuser@example.com",
                to_addresses=["alice@example.com"],
                subject="Field test",
                body_text="Testing event fields",
            )

            # Advance to execute
            await agent.ues_client.time.advance(seconds=1)

            # List executed events
            events = await agent.ues_client.events.list_events(
                status="executed"
            )
            assert len(events.events) >= 1

            # Check that model_dump includes expected fields
            event = events.events[0]
            event_dict = event.model_dump()
            assert "event_id" in event_dict
            assert "modality" in event_dict
            assert "status" in event_dict
            assert "scheduled_time" in event_dict
            assert "data" in event_dict
            # agent_id is now returned (upstream fix verified)
            assert "agent_id" in event_dict
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: State Snapshot Capture with Real UES
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestStateSnapshotWithRealUES:
    """Verify _capture_state_snapshot() returns valid, serializable data."""

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9305)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_snapshot_captures_all_modalities(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """_capture_state_snapshot returns dict with all modality keys."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            snapshot = await agent._capture_state_snapshot()

            assert isinstance(snapshot, dict)
            assert "email" in snapshot
            assert "sms" in snapshot
            assert "calendar" in snapshot
            assert "chat" in snapshot
            assert "time" in snapshot

            # All values should be serializable dicts (from model_dump)
            for key in ("email", "sms", "calendar", "chat", "time"):
                assert isinstance(snapshot[key], dict)
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_snapshot_reflects_injected_content(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
    ) -> None:
        """State snapshot changes when content is injected into UES."""
        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            await agent._setup_ues(simple_scenario)

            # Snapshot before injection
            before = await agent._capture_state_snapshot()

            # Inject an email
            await agent.ues_client.email.receive(
                from_address="test@example.com",
                to_addresses=["testuser@example.com"],
                subject="Snapshot test",
                body_text="Testing snapshot",
            )

            # Snapshot after injection
            after = await agent._capture_state_snapshot()

            # Email count should differ
            assert after["email"] != before["email"]
        finally:
            await agent.ues_client.close()


# ================================================================
# Test: Full Assessment Flow with Real UES + Mocked LLMs
# ================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestFullAssessmentFlowRealUES:
    """Full assessment flow against a real UES server with mocked LLMs.

    Unlike the existing TestEndToEndWithRealUES which uses real LLMs,
    these tests mock LLMFactory + CriteriaJudge + ResponseGenerator
    so they can run without API keys and are deterministic.
    """

    @pytest.fixture
    async def ues_server(self) -> UESServerManager:
        """Dedicated UES server for this test class."""
        manager = UESServerManager(port=9306)
        await manager.start(ready_timeout=30.0)
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_full_run_with_early_completion(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_emitter: TaskUpdateEmitter,
    ) -> None:
        """Full run(): startup → run (early completion) → shutdown."""
        mock_purple_client.responses = [
            _early_completion_response("All tasks done"),
        ]

        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            # Mock the per-turn processing that involves LLMs
            with patch("src.green.agent.CriteriaJudge") as jcls:
                jcls.return_value = _mock_judge()

                # Mock ResponseGenerator too (it uses LLMs)
                with patch("src.green.agent.ResponseGenerator") as rgcls:
                    mock_rg = MagicMock()
                    mock_rg.process_new_messages = AsyncMock(
                        return_value=[]
                    )
                    rgcls.return_value = mock_rg

                    results = await agent.run(
                        task_id="real-ues-e2e-1",
                        emitter=mock_emitter,
                        scenario=simple_scenario,
                        evaluators=mock_evaluator_registry,
                        purple_client=mock_purple_client,  # type: ignore[arg-type]
                        assessment_config={"max_turns": 5},
                    )

            assert isinstance(results, AssessmentResults)
            assert results.scenario_id == "test_real_ues"
            assert results.status == "completed"
            assert results.turns_taken == 1
            assert results.duration_seconds > 0
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_full_run_multiple_turns(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_emitter: TaskUpdateEmitter,
    ) -> None:
        """Full run(): 2 turns + early completion with time advances."""
        mock_purple_client.responses = [
            _turn_complete_response("Turn 1 done", "PT30M"),
            _turn_complete_response("Turn 2 done", "PT30M"),
            _early_completion_response("All done"),
        ]

        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            with patch("src.green.agent.CriteriaJudge") as jcls:
                jcls.return_value = _mock_judge()

                with patch("src.green.agent.ResponseGenerator") as rgcls:
                    mock_rg = MagicMock()
                    mock_rg.process_new_messages = AsyncMock(
                        return_value=[]
                    )
                    rgcls.return_value = mock_rg

                    results = await agent.run(
                        task_id="real-ues-e2e-2",
                        emitter=mock_emitter,
                        scenario=simple_scenario,
                        evaluators=mock_evaluator_registry,
                        purple_client=mock_purple_client,  # type: ignore[arg-type]
                        assessment_config={"max_turns": 10},
                    )

            assert isinstance(results, AssessmentResults)
            assert results.turns_taken == 3
            assert results.status == "completed"

            # Verify time actually advanced in UES
            time_state = await agent.ues_client.time.get_state()
            # After 2 turns of PT30M + 1 early completion turn:
            # the time should have advanced from 09:00
            assert time_state.current_time > datetime(
                2026, 2, 9, 9, 0, tzinfo=timezone.utc
            )
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_full_run_api_key_revoked_after(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_emitter: TaskUpdateEmitter,
    ) -> None:
        """Purple API key is revoked after assessment completes."""
        mock_purple_client.responses = [
            _early_completion_response("Quick"),
        ]

        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            # Capture the API key that gets sent to Purple
            with patch("src.green.agent.CriteriaJudge") as jcls:
                jcls.return_value = _mock_judge()

                with patch("src.green.agent.ResponseGenerator") as rgcls:
                    mock_rg = MagicMock()
                    mock_rg.process_new_messages = AsyncMock(
                        return_value=[]
                    )
                    rgcls.return_value = mock_rg

                    results = await agent.run(
                        task_id="real-ues-e2e-3",
                        emitter=mock_emitter,
                        scenario=simple_scenario,
                        evaluators=mock_evaluator_registry,
                        purple_client=mock_purple_client,  # type: ignore[arg-type]
                        assessment_config={},
                    )

            # Get the API key from the start message sent to Purple
            assert len(mock_purple_client.sent_data_calls) >= 1
            start_data = mock_purple_client.sent_data_calls[0]["data"]
            purple_key = start_data["api_key"]

            # Try to use the revoked key — should fail
            revoked_client = AsyncUESClient(
                base_url=ues_server.base_url,
                api_key=purple_key,
            )
            try:
                with pytest.raises(Exception):
                    await revoked_client.time.get_state()
            finally:
                await revoked_client.close()
        finally:
            await agent.ues_client.close()

    @pytest.mark.asyncio
    async def test_full_run_max_turns_reached(
        self,
        green_config: GreenAgentConfig,
        ues_server: UESServerManager,
        simple_scenario: ScenarioConfig,
        mock_evaluator_registry: EvaluatorRegistry,
        mock_purple_client: MockA2AClientWrapper,
        mock_emitter: TaskUpdateEmitter,
    ) -> None:
        """Full run() exits correctly when max_turns is reached."""
        # All responses are turn_complete (no early completion)
        mock_purple_client.responses = [
            _turn_complete_response("Turn 1", "PT30M"),
            _turn_complete_response("Turn 2", "PT30M"),
            _turn_complete_response("Turn 3", "PT30M"),
        ]

        agent = _make_agent_with_real_ues(green_config, ues_server)

        from ues.client import AsyncUESClient

        agent.ues_client = AsyncUESClient(
            base_url=ues_server.base_url,
            api_key=ues_server.admin_api_key,
        )

        try:
            with patch("src.green.agent.CriteriaJudge") as jcls:
                jcls.return_value = _mock_judge()

                with patch("src.green.agent.ResponseGenerator") as rgcls:
                    mock_rg = MagicMock()
                    mock_rg.process_new_messages = AsyncMock(
                        return_value=[]
                    )
                    rgcls.return_value = mock_rg

                    results = await agent.run(
                        task_id="real-ues-e2e-4",
                        emitter=mock_emitter,
                        scenario=simple_scenario,
                        evaluators=mock_evaluator_registry,
                        purple_client=mock_purple_client,  # type: ignore[arg-type]
                        assessment_config={"max_turns": 3},
                    )

            assert results.turns_taken == 3
            assert results.status == "completed"
        finally:
            await agent.ues_client.close()
