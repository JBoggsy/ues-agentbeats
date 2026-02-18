"""Tests for GreenAgentExecutor and its helper types.

Organized by implementation step:
- TestAssessmentRequest: Step 1 — request validation model
- TestPortAllocator: Step 2 — sequential port allocator
- TestGreenAgentExecutorInit: Step 3 — executor construction
- TestParseRequest: Step 4 — request parsing from RequestContext
- TestGetOrCreateAgent: Step 5 — agent lifecycle management
- TestExecute: Step 6 — core A2A handler
- TestCancel: Step 7 — cancellation handling
- TestShutdown: Step 8 — cleanup
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import DataPart, Part, TaskState, TextPart
from pydantic import ValidationError

from src.common.a2a.messages import create_data_message, create_text_message
from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.results import (
    AssessmentResults,
    OverallScore,
    Scores,
)
from src.green.executor import AssessmentRequest, GreenAgentExecutor, _PortAllocator
from src.green.scenarios.loader import (
    EvaluatorLoadError,
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
)


# =============================================================================
# Step 1: AssessmentRequest
# =============================================================================


class TestAssessmentRequest:
    """Tests for the AssessmentRequest Pydantic model."""

    def test_valid_request(self) -> None:
        """Happy path with all fields populated."""
        request = AssessmentRequest(
            participants={"personal_assistant": "http://purple:8001"},
            config={"scenario_id": "email_triage_basic", "seed": 42},
        )
        assert request.participants == {
            "personal_assistant": "http://purple:8001"
        }
        assert request.config == {
            "scenario_id": "email_triage_basic",
            "seed": 42,
        }
        assert request.scenario_id == "email_triage_basic"
        assert request.purple_agent_url == "http://purple:8001"

    def test_defaults_empty_config(self) -> None:
        """Config defaults to empty dict when not provided."""
        request = AssessmentRequest(
            participants={"agent": "http://localhost:9000"},
        )
        assert request.config == {}

    def test_scenario_id_missing_raises(self) -> None:
        """Raises ValueError when scenario_id is not in config."""
        request = AssessmentRequest(
            participants={"agent": "http://localhost:9000"},
            config={"other_key": "value"},
        )
        with pytest.raises(ValueError, match="Missing 'scenario_id'"):
            _ = request.scenario_id

    def test_scenario_id_empty_string_raises(self) -> None:
        """Raises ValueError when scenario_id is an empty string."""
        request = AssessmentRequest(
            participants={"agent": "http://localhost:9000"},
            config={"scenario_id": ""},
        )
        with pytest.raises(ValueError, match="Missing 'scenario_id'"):
            _ = request.scenario_id

    def test_no_participants_raises(self) -> None:
        """Raises ValueError when participants dict is empty."""
        request = AssessmentRequest(
            participants={},
            config={"scenario_id": "email_triage_basic"},
        )
        with pytest.raises(ValueError, match="No participants"):
            _ = request.purple_agent_url

    def test_frozen(self) -> None:
        """Model is immutable (frozen=True)."""
        request = AssessmentRequest(
            participants={"agent": "http://localhost:9000"},
            config={"scenario_id": "test"},
        )
        with pytest.raises(ValidationError):
            request.participants = {"other": "http://other:8000"}  # type: ignore[misc]

    def test_multiple_participants_returns_first(self) -> None:
        """purple_agent_url returns the first participant value."""
        request = AssessmentRequest(
            participants={
                "first_agent": "http://first:8001",
                "second_agent": "http://second:8002",
            },
            config={"scenario_id": "test"},
        )
        # dict preserves insertion order in Python 3.7+
        assert request.purple_agent_url == "http://first:8001"

    def test_participants_required(self) -> None:
        """participants field is required (no default)."""
        with pytest.raises(ValidationError):
            AssessmentRequest(config={"scenario_id": "test"})  # type: ignore[call-arg]

    def test_extra_config_fields_preserved(self) -> None:
        """Extra fields in config dict are preserved."""
        request = AssessmentRequest(
            participants={"agent": "http://localhost:9000"},
            config={
                "scenario_id": "email_triage_basic",
                "seed": 12345,
                "custom_param": "value",
            },
        )
        assert request.config["seed"] == 12345
        assert request.config["custom_param"] == "value"


# =============================================================================
# Step 2: _PortAllocator
# =============================================================================


class TestPortAllocator:
    """Tests for the _PortAllocator helper."""

    def test_sequential_allocation(self) -> None:
        """Allocate 3 ports, verify sequential increment."""
        allocator = _PortAllocator(base_port=8080)
        assert allocator.allocate() == 8080
        assert allocator.allocate() == 8081
        assert allocator.allocate() == 8082

    def test_custom_base_port(self) -> None:
        """Non-default base port starts allocation correctly."""
        allocator = _PortAllocator(base_port=19000)
        assert allocator.allocate() == 19000
        assert allocator.allocate() == 19001

    def test_state_independent_instances(self) -> None:
        """Separate allocator instances have independent counters."""
        alloc_a = _PortAllocator(base_port=8080)
        alloc_b = _PortAllocator(base_port=9000)
        assert alloc_a.allocate() == 8080
        assert alloc_b.allocate() == 9000
        assert alloc_a.allocate() == 8081
        assert alloc_b.allocate() == 9001


# =============================================================================
# Step 3: GreenAgentExecutor skeleton
# =============================================================================


class TestGreenAgentExecutorInit:
    """Tests for executor construction."""

    @pytest.fixture()
    def config(self) -> GreenAgentConfig:
        """GreenAgentConfig with test defaults."""
        return GreenAgentConfig(ues_base_port=19000)

    @pytest.fixture()
    def scenario_manager(self) -> MagicMock:
        """Mock ScenarioManager."""
        return MagicMock(spec=ScenarioManager)

    @pytest.fixture()
    def executor(
        self, config: GreenAgentConfig, scenario_manager: MagicMock
    ) -> GreenAgentExecutor:
        """GreenAgentExecutor with test dependencies."""
        return GreenAgentExecutor(config, scenario_manager)

    def test_executor_init(
        self, executor: GreenAgentExecutor, config: GreenAgentConfig,
        scenario_manager: MagicMock,
    ) -> None:
        """All instance variables are initialized correctly."""
        assert executor._config is config
        assert executor._scenario_manager is scenario_manager
        assert executor._agents == {}
        assert executor._context_locks == {}
        # Port allocator starts from config's ues_base_port
        assert executor._port_allocator.allocate() == 19000

    def test_executor_is_agent_executor(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Executor is an instance of the A2A AgentExecutor ABC."""
        assert isinstance(executor, AgentExecutor)


# =============================================================================
# Shared helpers for Steps 4–8
# =============================================================================


def _make_context(
    task_id: str = "task-1",
    context_id: str = "ctx-1",
    message_data: dict[str, Any] | None = None,
    has_message: bool = True,
) -> MagicMock:
    """Create a mock RequestContext with configurable properties.

    Args:
        task_id: Task ID for the context.
        context_id: Context ID for the context.
        message_data: If provided, creates a message with a DataPart
            containing this dict. If None and has_message is True,
            creates a text-only message.
        has_message: If False, context.message returns None.

    Returns:
        A MagicMock configured like a RequestContext.
    """
    ctx = MagicMock(spec=RequestContext)
    ctx.task_id = task_id
    ctx.context_id = context_id

    if not has_message:
        ctx.message = None
    elif message_data is not None:
        ctx.message = create_data_message(message_data)
    else:
        ctx.message = create_text_message("hello")

    return ctx


def _valid_request_data(
    scenario_id: str = "email_triage_basic",
    purple_url: str = "http://purple:8001",
) -> dict[str, Any]:
    """Return a valid assessment request payload."""
    return {
        "participants": {"personal_assistant": purple_url},
        "config": {"scenario_id": scenario_id},
    }


def _make_mock_results() -> AssessmentResults:
    """Create a minimal AssessmentResults for testing."""
    return AssessmentResults(
        message_type="assessment_results",
        scenario_id="email_triage_basic",
        assessment_id="assess-1",
        participant="purple-agent-1",
        status="completed",
        duration_seconds=10.0,
        turns_taken=2,
        actions_taken=0,
        scores=Scores(
            overall=OverallScore(score=0, max_score=0),
            dimensions={},
        ),
        criteria_results=[],
        action_log=[],
    )


# =============================================================================
# Step 4: _parse_request
# =============================================================================


class TestParseRequest:
    """Tests for _parse_request()."""

    @pytest.fixture()
    def executor(self) -> GreenAgentExecutor:
        """Executor for parse_request tests."""
        config = GreenAgentConfig(ues_base_port=19000)
        return GreenAgentExecutor(config, MagicMock(spec=ScenarioManager))

    def test_parse_request_valid(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Valid data part returns AssessmentRequest."""
        ctx = _make_context(message_data=_valid_request_data())
        request = executor._parse_request(ctx)

        assert isinstance(request, AssessmentRequest)
        assert request.scenario_id == "email_triage_basic"
        assert request.purple_agent_url == "http://purple:8001"

    def test_parse_request_no_message(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Missing message raises ValueError."""
        ctx = _make_context(has_message=False)
        with pytest.raises(ValueError, match="No message"):
            executor._parse_request(ctx)

    def test_parse_request_no_data_part(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Message with only text parts raises ValueError."""
        ctx = _make_context()  # text-only message
        with pytest.raises(ValueError, match="No data content"):
            executor._parse_request(ctx)

    def test_parse_request_missing_scenario_id(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Data with no scenario_id raises ValueError."""
        data = {
            "participants": {"agent": "http://purple:8001"},
            "config": {},
        }
        ctx = _make_context(message_data=data)
        with pytest.raises(ValueError, match="scenario_id"):
            executor._parse_request(ctx)

    def test_parse_request_empty_participants(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Empty participants raises ValueError."""
        data = {
            "participants": {},
            "config": {"scenario_id": "test"},
        }
        ctx = _make_context(message_data=data)
        with pytest.raises(ValueError, match="No participants"):
            executor._parse_request(ctx)


# =============================================================================
# Step 5: _get_or_create_agent
# =============================================================================


class TestGetOrCreateAgent:
    """Tests for _get_or_create_agent()."""

    @pytest.fixture()
    def executor(self) -> GreenAgentExecutor:
        """Executor for agent lifecycle tests."""
        config = GreenAgentConfig(ues_base_port=19000)
        return GreenAgentExecutor(config, MagicMock(spec=ScenarioManager))

    @pytest.mark.asyncio
    @patch("src.green.executor.GreenAgent")
    async def test_creates_new_agent(
        self, mock_agent_cls: MagicMock, executor: GreenAgentExecutor
    ) -> None:
        """First call creates agent and calls startup()."""
        mock_agent = AsyncMock()
        mock_agent_cls.return_value = mock_agent

        agent = await executor._get_or_create_agent("ctx-1")

        assert agent is mock_agent
        mock_agent_cls.assert_called_once_with(
            ues_port=19000, config=executor._config
        )
        mock_agent.startup.assert_awaited_once()
        assert executor._agents["ctx-1"] is mock_agent

    @pytest.mark.asyncio
    @patch("src.green.executor.GreenAgent")
    async def test_reuses_existing_agent(
        self, mock_agent_cls: MagicMock, executor: GreenAgentExecutor
    ) -> None:
        """Second call returns cached agent, no second startup()."""
        mock_agent = AsyncMock()
        mock_agent_cls.return_value = mock_agent

        agent1 = await executor._get_or_create_agent("ctx-1")
        agent2 = await executor._get_or_create_agent("ctx-1")

        assert agent1 is agent2
        # Only one GreenAgent created, one startup() called
        mock_agent_cls.assert_called_once()
        mock_agent.startup.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("src.green.executor.GreenAgent")
    async def test_unique_ports_per_context(
        self, mock_agent_cls: MagicMock, executor: GreenAgentExecutor
    ) -> None:
        """Different context_ids get different ports."""
        agents = []
        mock_agent_cls.side_effect = lambda **kwargs: AsyncMock(
            _port=kwargs["ues_port"]
        )

        await executor._get_or_create_agent("ctx-1")
        await executor._get_or_create_agent("ctx-2")

        calls = mock_agent_cls.call_args_list
        assert calls[0].kwargs["ues_port"] == 19000
        assert calls[1].kwargs["ues_port"] == 19001

    @pytest.mark.asyncio
    @patch("src.green.executor.GreenAgent")
    async def test_startup_failure_not_cached(
        self, mock_agent_cls: MagicMock, executor: GreenAgentExecutor
    ) -> None:
        """startup() failure means agent is not cached."""
        mock_agent = AsyncMock()
        mock_agent.startup.side_effect = RuntimeError("UES failed")
        mock_agent_cls.return_value = mock_agent

        with pytest.raises(RuntimeError, match="UES failed"):
            await executor._get_or_create_agent("ctx-1")

        assert "ctx-1" not in executor._agents


# =============================================================================
# Step 6: execute
# =============================================================================


class TestExecute:
    """Tests for execute() — the core A2A handler."""

    @pytest.fixture()
    def scenario_manager(self) -> MagicMock:
        """Mock ScenarioManager that returns test fixtures."""
        mgr = MagicMock(spec=ScenarioManager)
        mgr.get_scenario.return_value = MagicMock(name="mock_scenario")
        mgr.get_evaluators.return_value = MagicMock(name="mock_evaluators")
        return mgr

    @pytest.fixture()
    def executor(
        self, scenario_manager: MagicMock
    ) -> GreenAgentExecutor:
        """Executor with mocked dependencies."""
        config = GreenAgentConfig(ues_base_port=19000)
        return GreenAgentExecutor(config, scenario_manager)

    @pytest.fixture()
    def event_queue(self) -> AsyncMock:
        """Mock EventQueue."""
        return AsyncMock(spec=EventQueue)

    @pytest.mark.asyncio
    @patch("src.green.executor.A2AClientWrapper")
    @patch("src.green.executor.GreenAgent")
    async def test_happy_path(
        self,
        mock_agent_cls: MagicMock,
        mock_client_cls: MagicMock,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """Full successful flow emits artifact and completed state."""
        # Setup mock agent
        mock_agent = AsyncMock()
        mock_results = _make_mock_results()
        mock_agent.run.return_value = mock_results
        mock_agent_cls.return_value = mock_agent

        # Setup mock A2A client
        mock_purple = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_purple
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(
            return_value=False
        )

        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        # Agent was created and started
        mock_agent.startup.assert_awaited_once()
        # Assessment was run
        mock_agent.run.assert_awaited_once()

        # Verify events were enqueued (artifact + completed status)
        enqueued = event_queue.enqueue_event.call_args_list
        assert len(enqueued) >= 2  # artifact + status

        # Last event should be completed status
        last_event = enqueued[-1].args[0]
        assert last_event.status.state == TaskState.completed

    @pytest.mark.asyncio
    async def test_bad_request_emits_failed(
        self,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """Invalid message emits TaskState.failed."""
        ctx = _make_context()  # text-only, no data part
        await executor.execute(ctx, event_queue)

        event_queue.enqueue_event.assert_called_once()
        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed
        assert "Assessment failed" in event.status.message.parts[0].root.text

    @pytest.mark.asyncio
    async def test_scenario_not_found_emits_failed(
        self,
        executor: GreenAgentExecutor,
        scenario_manager: MagicMock,
        event_queue: AsyncMock,
    ) -> None:
        """Unknown scenario_id emits TaskState.failed."""
        from pathlib import Path

        scenario_manager.get_scenario.side_effect = ScenarioNotFoundError(
            "email_triage_basic", Path("/tmp/scenarios")
        )
        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed

    @pytest.mark.asyncio
    async def test_scenario_validation_error_emits_failed(
        self,
        executor: GreenAgentExecutor,
        scenario_manager: MagicMock,
        event_queue: AsyncMock,
    ) -> None:
        """Invalid scenario emits TaskState.failed."""
        scenario_manager.get_scenario.side_effect = ScenarioValidationError(
            "email_triage_basic", ["bad schema"]
        )
        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed

    @pytest.mark.asyncio
    async def test_evaluator_load_error_emits_failed(
        self,
        executor: GreenAgentExecutor,
        scenario_manager: MagicMock,
        event_queue: AsyncMock,
    ) -> None:
        """Evaluator load failure emits TaskState.failed."""
        scenario_manager.get_evaluators.side_effect = EvaluatorLoadError(
            "email_triage_basic", "bad evaluator"
        )
        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed

    @pytest.mark.asyncio
    @patch("src.green.executor.GreenAgent")
    async def test_agent_startup_failure_emits_failed(
        self,
        mock_agent_cls: MagicMock,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """startup() failure emits TaskState.failed."""
        mock_agent = AsyncMock()
        mock_agent.startup.side_effect = RuntimeError("UES crash")
        mock_agent_cls.return_value = mock_agent

        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed
        assert "UES crash" in event.status.message.parts[0].root.text

    @pytest.mark.asyncio
    @patch("src.green.executor.A2AClientWrapper")
    @patch("src.green.executor.GreenAgent")
    async def test_run_failure_emits_failed_agent_still_cached(
        self,
        mock_agent_cls: MagicMock,
        mock_client_cls: MagicMock,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """agent.run() failure emits failed but agent remains cached."""
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("run exploded")
        mock_agent_cls.return_value = mock_agent

        mock_purple = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_purple
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(
            return_value=False
        )

        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        # Failed status emitted
        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed

        # But agent is still cached (can serve future requests)
        assert "ctx-1" in executor._agents

    @pytest.mark.asyncio
    @patch("src.green.executor.A2AClientWrapper")
    @patch("src.green.executor.GreenAgent")
    async def test_emitter_passed_to_run(
        self,
        mock_agent_cls: MagicMock,
        mock_client_cls: MagicMock,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """TaskUpdateEmitter is passed to agent.run()."""
        mock_agent = AsyncMock()
        mock_agent.run.return_value = _make_mock_results()
        mock_agent_cls.return_value = mock_agent

        mock_purple = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_purple
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(
            return_value=False
        )

        ctx = _make_context(message_data=_valid_request_data())
        await executor.execute(ctx, event_queue)

        # Check that emitter kwarg was a TaskUpdateEmitter
        from src.common.agentbeats.updates import TaskUpdateEmitter

        run_kwargs = mock_agent.run.call_args.kwargs
        assert isinstance(run_kwargs["emitter"], TaskUpdateEmitter)

    @pytest.mark.asyncio
    @patch("src.green.executor.A2AClientWrapper")
    @patch("src.green.executor.GreenAgent")
    async def test_concurrent_same_context_serialized(
        self,
        mock_agent_cls: MagicMock,
        mock_client_cls: MagicMock,
        executor: GreenAgentExecutor,
    ) -> None:
        """Two concurrent requests for same context_id serialize."""
        execution_order: list[str] = []

        async def slow_run(**kwargs: Any) -> AssessmentResults:
            execution_order.append("start")
            await asyncio.sleep(0.05)
            execution_order.append("end")
            return _make_mock_results()

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = slow_run
        mock_agent_cls.return_value = mock_agent

        mock_purple = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_purple
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(
            return_value=False
        )

        ctx1 = _make_context(
            task_id="t1", context_id="ctx-same",
            message_data=_valid_request_data(),
        )
        ctx2 = _make_context(
            task_id="t2", context_id="ctx-same",
            message_data=_valid_request_data(),
        )
        eq1 = AsyncMock(spec=EventQueue)
        eq2 = AsyncMock(spec=EventQueue)

        await asyncio.gather(
            executor.execute(ctx1, eq1),
            executor.execute(ctx2, eq2),
        )

        # Serialized: start-end-start-end, not start-start-end-end
        assert execution_order == ["start", "end", "start", "end"]

    @pytest.mark.asyncio
    @patch("src.green.executor.A2AClientWrapper")
    @patch("src.green.executor.GreenAgent")
    async def test_concurrent_different_contexts_parallel(
        self,
        mock_agent_cls: MagicMock,
        mock_client_cls: MagicMock,
        executor: GreenAgentExecutor,
    ) -> None:
        """Different context_ids run in parallel."""
        execution_log: list[tuple[str, str]] = []

        async def tracked_run(**kwargs: Any) -> AssessmentResults:
            tid = kwargs["task_id"]
            execution_log.append((tid, "start"))
            await asyncio.sleep(0.05)
            execution_log.append((tid, "end"))
            return _make_mock_results()

        mock_agent_cls.side_effect = lambda **kw: AsyncMock(
            run=AsyncMock(side_effect=tracked_run)
        )

        mock_purple = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_purple
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(
            return_value=False
        )

        ctx1 = _make_context(
            task_id="t1", context_id="ctx-a",
            message_data=_valid_request_data(),
        )
        ctx2 = _make_context(
            task_id="t2", context_id="ctx-b",
            message_data=_valid_request_data(),
        )
        eq1 = AsyncMock(spec=EventQueue)
        eq2 = AsyncMock(spec=EventQueue)

        await asyncio.gather(
            executor.execute(ctx1, eq1),
            executor.execute(ctx2, eq2),
        )

        # Parallel: both starts happen before both ends
        starts = [e for e in execution_log if e[1] == "start"]
        ends = [e for e in execution_log if e[1] == "end"]
        # Both started before the first one ended
        first_end_idx = execution_log.index(ends[0])
        assert len([e for e in execution_log[:first_end_idx] if e[1] == "start"]) == 2

    @pytest.mark.asyncio
    async def test_no_message_emits_failed(
        self,
        executor: GreenAgentExecutor,
        event_queue: AsyncMock,
    ) -> None:
        """Context with no message emits TaskState.failed."""
        ctx = _make_context(has_message=False)
        await executor.execute(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.failed


# =============================================================================
# Step 7: cancel
# =============================================================================


class TestCancel:
    """Tests for cancel()."""

    @pytest.fixture()
    def executor(self) -> GreenAgentExecutor:
        """Executor for cancel tests."""
        config = GreenAgentConfig(ues_base_port=19000)
        return GreenAgentExecutor(config, MagicMock(spec=ScenarioManager))

    @pytest.fixture()
    def event_queue(self) -> AsyncMock:
        """Mock EventQueue."""
        return AsyncMock(spec=EventQueue)

    @pytest.mark.asyncio
    async def test_cancel_existing_agent(
        self, executor: GreenAgentExecutor, event_queue: AsyncMock
    ) -> None:
        """Agent exists and task_id present: cancel() called, canceled emitted."""
        mock_agent = AsyncMock()
        executor._agents["ctx-1"] = mock_agent

        ctx = _make_context(task_id="task-1", context_id="ctx-1")
        await executor.cancel(ctx, event_queue)

        mock_agent.cancel.assert_awaited_once_with("task-1")
        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.canceled

    @pytest.mark.asyncio
    async def test_cancel_no_agent(
        self, executor: GreenAgentExecutor, event_queue: AsyncMock
    ) -> None:
        """No agent for context: canceled still emitted."""
        ctx = _make_context(task_id="task-1", context_id="ctx-unknown")
        await executor.cancel(ctx, event_queue)

        event = event_queue.enqueue_event.call_args.args[0]
        assert event.status.state == TaskState.canceled

    @pytest.mark.asyncio
    async def test_cancel_no_task_id(
        self, executor: GreenAgentExecutor, event_queue: AsyncMock
    ) -> None:
        """task_id is None: early return, no events emitted."""
        mock_agent = AsyncMock()
        executor._agents["ctx-1"] = mock_agent

        ctx = _make_context(task_id=None, context_id="ctx-1")
        await executor.cancel(ctx, event_queue)

        mock_agent.cancel.assert_not_awaited()
        event_queue.enqueue_event.assert_not_called()


# =============================================================================
# Step 8: shutdown
# =============================================================================


class TestShutdown:
    """Tests for shutdown()."""

    @pytest.fixture()
    def executor(self) -> GreenAgentExecutor:
        """Executor for shutdown tests."""
        config = GreenAgentConfig(ues_base_port=19000)
        return GreenAgentExecutor(config, MagicMock(spec=ScenarioManager))

    @pytest.mark.asyncio
    async def test_shutdown_all_agents(
        self, executor: GreenAgentExecutor
    ) -> None:
        """All cached agents have shutdown() called."""
        agent_a = AsyncMock()
        agent_b = AsyncMock()
        executor._agents["ctx-a"] = agent_a
        executor._agents["ctx-b"] = agent_b

        await executor.shutdown()

        agent_a.shutdown.assert_awaited_once()
        agent_b.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_error_continues(
        self, executor: GreenAgentExecutor
    ) -> None:
        """One agent's shutdown() failure doesn't prevent others."""
        agent_a = AsyncMock()
        agent_a.shutdown.side_effect = RuntimeError("cleanup failed")
        agent_b = AsyncMock()
        executor._agents["ctx-a"] = agent_a
        executor._agents["ctx-b"] = agent_b

        # Should not raise
        await executor.shutdown()

        agent_a.shutdown.assert_awaited_once()
        agent_b.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_clears_caches(
        self, executor: GreenAgentExecutor
    ) -> None:
        """After shutdown, agents and locks caches are empty."""
        executor._agents["ctx-1"] = AsyncMock()
        executor._context_locks["ctx-1"] = asyncio.Lock()

        await executor.shutdown()

        assert executor._agents == {}
        assert executor._context_locks == {}

    @pytest.mark.asyncio
    async def test_shutdown_empty_executor(
        self, executor: GreenAgentExecutor
    ) -> None:
        """Shutdown on executor with no agents is a no-op."""
        await executor.shutdown()  # Should not raise
        assert executor._agents == {}
        assert executor._context_locks == {}
