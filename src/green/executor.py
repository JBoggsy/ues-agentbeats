"""GreenAgentExecutor — bridge between A2A protocol and GreenAgent.

This module provides the ``GreenAgentExecutor`` class, which implements
the ``AgentExecutor`` interface from the A2A Python SDK, translating
A2A ``RequestContext`` / ``EventQueue`` objects into the higher-level
abstractions that ``GreenAgent.run()`` expects.

It also defines helper types used by the executor:

- ``AssessmentRequest``: Pydantic model for validating incoming A2A
  assessment requests.
- ``_PortAllocator``: Simple sequential port allocator for UES servers.

Architecture::

    A2A HTTP request
          │
          ▼
    DefaultRequestHandler (a2a-python SDK)
          │ calls execute() / cancel()
          ▼
    GreenAgentExecutor        ← THIS MODULE
          │ creates / manages
          ▼
    GreenAgent(s)             one per context_id

See Also:
    - ``docs/design/GREEN_EXECUTOR_DESIGN.md`` for full design rationale.
    - ``docs/design/GREEN_EXECUTOR_IMPLEMENTATION_PLAN.md`` for the
      step-by-step implementation plan.
    - ``src/green/agent.py`` for the ``GreenAgent`` orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Role, TaskState
from pydantic import BaseModel, ConfigDict, Field

from src.common.a2a.artifacts import create_json_artifact
from src.common.a2a.client import A2AClientWrapper
from src.common.a2a.messages import create_text_message, get_data_content
from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.agent import GreenAgent
from src.green.scenarios.loader import (
    EvaluatorLoadError,
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AssessmentRequest
# =============================================================================


class AssessmentRequest(BaseModel):
    """Parsed assessment request from the AgentBeats platform.

    Validates incoming A2A assessment requests containing participant
    endpoints and assessment configuration.

    Attributes:
        participants: Mapping of role names to A2A endpoint URLs.
            Must contain at least one entry.
        config: Assessment-specific configuration. May contain
            ``scenario_id`` and other scenario parameters.

    Example:
        >>> request = AssessmentRequest(
        ...     participants={"personal_assistant": "http://purple:8001"},
        ...     config={"scenario_id": "email_triage_basic"},
        ... )
        >>> request.scenario_id
        'email_triage_basic'
        >>> request.purple_agent_url
        'http://purple:8001'
    """

    model_config = ConfigDict(frozen=True)

    participants: dict[str, str]
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        """Get the scenario ID from the config.

        Returns:
            The scenario identifier string.

        Raises:
            ValueError: If ``scenario_id`` is not present in config.
        """
        scenario_id = self.config.get("scenario_id")
        if not scenario_id:
            raise ValueError(
                "Missing 'scenario_id' in assessment config"
            )
        return scenario_id

    @property
    def purple_agent_url(self) -> str:
        """Get the Purple agent URL from participants.

        Returns the first participant URL. In the current competition
        format there is exactly one Purple agent per assessment.

        Returns:
            The Purple agent's A2A endpoint URL.

        Raises:
            ValueError: If participants is empty.
        """
        if not self.participants:
            raise ValueError(
                "No participants in assessment request"
            )
        return next(iter(self.participants.values()))


# =============================================================================
# _PortAllocator
# =============================================================================


class _PortAllocator:
    """Simple sequential port allocator.

    Allocates ports starting from a base port, incrementing by one
    for each allocation. Used by ``GreenAgentExecutor`` to assign
    unique UES ports to each ``GreenAgent`` instance.

    Attributes:
        _next_port: The next port number to allocate.

    Example:
        >>> allocator = _PortAllocator(base_port=8080)
        >>> allocator.allocate()
        8080
        >>> allocator.allocate()
        8081
    """

    def __init__(self, base_port: int) -> None:
        """Initialize the port allocator.

        Args:
            base_port: The first port number to allocate.
        """
        self._next_port = base_port

    def allocate(self) -> int:
        """Allocate the next available port.

        Returns:
            The allocated port number.
        """
        port = self._next_port
        self._next_port += 1
        return port


# =============================================================================
# GreenAgentExecutor
# =============================================================================


class GreenAgentExecutor(AgentExecutor):
    """Bridge between A2A protocol and GreenAgent assessment orchestrator.

    Implements the ``AgentExecutor`` interface from the A2A Python SDK,
    managing ``GreenAgent`` instances (one per ``context_id``) and
    translating A2A requests into assessment orchestration calls.

    Responsibilities:
        - Parse and validate incoming A2A assessment requests
        - Manage ``GreenAgent`` lifecycle (create, cache, shutdown)
        - Allocate unique UES ports for each agent
        - Wire ``TaskUpdater`` / ``TaskUpdateEmitter`` for event emission
        - Dispatch ``GreenAgent.run()`` and emit results as artifacts
        - Handle cancellation and error conditions

    Attributes:
        _config: Green agent configuration.
        _scenario_manager: Loads scenario configs and evaluators.
        _port_allocator: Assigns unique UES ports to agents.
        _agents: Cache of ``GreenAgent`` instances keyed by context_id.
        _context_locks: Per-context locks serializing assessments.

    Example:
        >>> from src.common.agentbeats.config import GreenAgentConfig
        >>> from src.green.scenarios.loader import ScenarioManager
        >>> config = GreenAgentConfig()
        >>> manager = ScenarioManager(Path("scenarios"))
        >>> executor = GreenAgentExecutor(config, manager)
    """

    def __init__(
        self,
        config: GreenAgentConfig,
        scenario_manager: ScenarioManager,
    ) -> None:
        """Initialize the executor.

        Args:
            config: Green agent configuration with UES base port,
                LLM model names, and other parameters.
            scenario_manager: Pre-built scenario manager for loading
                scenario configs and evaluators.
        """
        self._config = config
        self._scenario_manager = scenario_manager
        self._port_allocator = _PortAllocator(config.ues_base_port)
        self._agents: dict[str, GreenAgent] = {}
        self._context_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_request(self, context: RequestContext) -> AssessmentRequest:
        """Extract and validate the assessment request from an A2A message.

        Reads the first ``DataPart`` from the incoming message, validates
        it against the ``AssessmentRequest`` model, and triggers early
        validation of required properties (``scenario_id``,
        ``purple_agent_url``).

        Args:
            context: The incoming A2A request context.

        Returns:
            Validated AssessmentRequest.

        Raises:
            ValueError: If the message is missing, contains no data part,
                or fails validation.
        """
        if context.message is None:
            raise ValueError("No message in request context")

        data = get_data_content(context.message)
        if data is None:
            raise ValueError("No data content in request message")

        request = AssessmentRequest(**data)

        # Trigger early validation — these raise ValueError if missing
        _ = request.scenario_id
        _ = request.purple_agent_url

        return request

    async def _get_or_create_agent(self, context_id: str) -> GreenAgent:
        """Get an existing GreenAgent for context_id or create a new one.

        Creates a new ``GreenAgent`` if one doesn't exist for this
        ``context_id``, allocates a UES port, and calls ``startup()``.
        If ``startup()`` raises, the agent is **not** cached — the
        exception propagates to the caller.

        Args:
            context_id: The A2A context ID.

        Returns:
            A started GreenAgent instance.

        Raises:
            Exception: If ``GreenAgent.startup()`` fails (e.g.,
                UES server fails to start).
        """
        existing = self._agents.get(context_id)
        if existing is not None:
            return existing

        port = self._port_allocator.allocate()
        agent = GreenAgent(ues_port=port, config=self._config)

        # If startup() raises, don't cache the agent
        await agent.startup()

        self._agents[context_id] = agent
        logger.info(
            "Created GreenAgent for context_id=%s on port %d",
            context_id,
            port,
        )
        return agent

    # ------------------------------------------------------------------
    # AgentExecutor interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute an assessment for the given A2A request.

        Parses the incoming request, loads the scenario, creates or
        reuses a ``GreenAgent``, runs the assessment, emits results
        as an A2A artifact, and sets the terminal task state.

        Error handling:
            - Known errors (bad request, missing scenario) emit
              ``TaskState.failed`` with a descriptive message.
            - Unexpected errors emit ``TaskState.failed`` with the
              exception message and log the full traceback.

        Args:
            context: The incoming A2A request context.
            event_queue: Queue for publishing task status and artifact
                events.
        """
        task_id = context.task_id
        context_id = context.context_id

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        try:
            # 1. Parse and validate request
            request = self._parse_request(context)

            # 2. Load scenario + evaluators (read-only, before lock)
            scenario = self._scenario_manager.get_scenario(
                request.scenario_id
            )
            evaluators = self._scenario_manager.get_evaluators(
                request.scenario_id
            )

            # 3. Acquire per-context lock (serializes assessments
            #    within a context)
            lock = self._context_locks.setdefault(
                context_id, asyncio.Lock()
            )
            async with lock:
                # 4. Get or create agent
                agent = await self._get_or_create_agent(context_id)

                # 5. Create emitter for this task
                emitter = TaskUpdateEmitter(updater)

                # 6. Create Purple client and run assessment
                async with A2AClientWrapper(
                    request.purple_agent_url
                ) as purple_client:
                    results = await agent.run(
                        task_id=task_id,
                        emitter=emitter,
                        scenario=scenario,
                        evaluators=evaluators,
                        purple_client=purple_client,
                        assessment_config=request.config,
                    )

                # 7. Emit results artifact
                artifact = create_json_artifact(
                    data=results.model_dump(mode="json"),
                    name="assessment_results",
                    description=(
                        f"Assessment results for scenario "
                        f"{request.scenario_id}"
                    ),
                )
                await updater.add_artifact(
                    parts=artifact.parts,
                    name=artifact.name,
                )

                # 8. Executor owns terminal state
                await updater.update_status(
                    state=TaskState.completed
                )

        except (
            ValueError,
            ScenarioNotFoundError,
            ScenarioValidationError,
            EvaluatorLoadError,
        ) as exc:
            logger.warning("Assessment request failed: %s", exc)
            msg = create_text_message(
                text=f"Assessment failed: {exc}",
                role=Role.agent,
            )
            await updater.update_status(
                state=TaskState.failed, message=msg
            )

        except Exception as exc:
            logger.exception(
                "Assessment failed with unexpected error: %s", exc
            )
            msg = create_text_message(
                text=f"Assessment failed: {exc}",
                role=Role.agent,
            )
            await updater.update_status(
                state=TaskState.failed, message=msg
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an ongoing assessment.

        Looks up the ``GreenAgent`` for the given context and forwards
        the cancellation request. Always emits ``TaskState.canceled``
        regardless of whether an agent was found.

        If ``task_id`` is None (no task to cancel), logs a warning and
        returns without emitting any event.

        Args:
            context: The request context with the task ID to cancel.
            event_queue: Queue for publishing the cancellation status.
        """
        context_id = context.context_id
        task_id = context.task_id

        if task_id is None:
            logger.warning(
                "Cancel called with no task_id for context=%s",
                context_id,
            )
            return

        agent = self._agents.get(context_id)
        if agent is not None:
            await agent.cancel(task_id)
            logger.info(
                "Forwarded cancellation for task=%s context=%s",
                task_id,
                context_id,
            )
        else:
            logger.debug(
                "No agent to cancel for context=%s task=%s",
                context_id,
                task_id,
            )

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        await updater.update_status(state=TaskState.canceled)

    async def shutdown(self) -> None:
        """Shut down all managed GreenAgent instances.

        Calls ``shutdown()`` on every cached agent, catching per-agent
        exceptions to ensure all agents are cleaned up. Clears the
        agent and lock caches afterwards.
        """
        for context_id, agent in self._agents.items():
            try:
                await agent.shutdown()
                logger.info(
                    "Shut down GreenAgent for context_id=%s", context_id
                )
            except Exception:
                logger.exception(
                    "Error shutting down GreenAgent for context_id=%s",
                    context_id,
                )

        self._agents.clear()
        self._context_locks.clear()
        logger.info("GreenAgentExecutor shutdown complete")
