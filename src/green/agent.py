"""GreenAgent — core orchestrator for AgentBeats assessments.

This module provides the ``GreenAgent`` class, which is the main orchestrator
for running AgentBeats assessments. Each instance:

- Owns and manages its own UES server instance via ``UESServerManager``
- Runs assessments for a single ``context_id`` (Purple agent session)
- Coordinates response generation, action logging, and evaluation
- Produces assessment results as A2A artifacts

Lifecycle:
    1. The executor creates a ``GreenAgent`` with an allocated port and config.
    2. The executor calls ``startup()`` to spin up the UES server.
    3. The executor calls ``run()`` for each assessment task.
    4. The executor calls ``shutdown()`` when the context is no longer needed.

Architecture:
    The ``GreenAgent`` sits between the ``GreenAgentExecutor`` (which handles
    A2A protocol details) and the per-assessment helper classes
    (``ActionLogBuilder``, ``NewMessageCollector``, ``ResponseGenerator``,
    ``CriteriaJudge``). Long-lived resources (UES server, LLMs) persist
    across assessments; per-assessment helpers are created fresh each time.

Constants:
    USER_PERMISSIONS: Permissions granted to Purple agent API keys.

Classes:
    GreenAgent: High-level orchestrator for assessments within a context.

See Also:
    - ``docs/design/GREEN_AGENT_DESIGN_PLAN.md`` for full design rationale.
    - ``src/green/core/ues_server.py`` for UES subprocess management.
    - ``src/green/assessment/models.py`` for ``TurnResult``/``EndOfTurnResult``.

Example:
    >>> from src.green.agent import GreenAgent
    >>> from src.common.agentbeats.config import GreenAgentConfig
    >>>
    >>> config = GreenAgentConfig()
    >>> agent = GreenAgent(ues_port=8100, config=config)
    >>> await agent.startup()
    >>> # ... run assessments via agent.run() ...
    >>> await agent.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.common.agentbeats.config import GreenAgentConfig
from src.common.agentbeats.messages import (
    AssessmentCompleteMessage,
    AssessmentStartMessage,
    CalendarSummary,
    ChatSummary,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    SMSSummary,
    TurnCompleteMessage,
    TurnStartMessage,
)
from src.common.agentbeats.results import (
    AssessmentResults,
    CriterionResult,
    Scores,
)
from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.assessment.models import EndOfTurnResult, TurnResult
from src.green.core.action_log import ActionLogBuilder
from src.green.core.llm_config import LLMFactory
from src.green.core.message_collector import NewMessageCollector
from src.green.core.ues_server import UESServerManager
from src.green.evaluation.judge import CriteriaJudge
from src.green.response.generator import ResponseGenerator
from src.green.response.models import ScheduledResponse
from src.green.scenarios.schema import (
    AgentBeatsEvalContext,
    EvaluatorRegistry,
    ScenarioConfig,
)

if TYPE_CHECKING:
    from ues.client import AsyncUESClient

    from src.common.a2a.client import A2AClientWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

USER_PERMISSIONS: list[str] = [
    # State & Query
    "email:state",
    "email:query",
    "sms:state",
    "sms:query",
    "calendar:state",
    "calendar:query",
    "chat:state",
    "chat:query",
    "time:read",
    # Email user-side actions
    "email:send",
    "email:read",
    "email:unread",
    "email:star",
    "email:unstar",
    "email:archive",
    "email:delete",
    "email:label",
    "email:unlabel",
    "email:move",
    # SMS user-side actions
    "sms:send",
    "sms:read",
    "sms:unread",
    "sms:delete",
    "sms:react",
    # Calendar user-side actions
    "calendar:create",
    "calendar:update",
    "calendar:delete",
    "calendar:respond",
    # Chat user-side actions
    "chat:send",
]
"""Permissions granted to Purple agent API keys.

These permissions allow user-level actions (sending messages, reading
state, managing calendar events) but forbid proctor-level operations
(receiving messages, advancing time, managing simulations, API keys,
webhooks, etc.).
"""


# =============================================================================
# GreenAgent
# =============================================================================


class GreenAgent:
    """High-level orchestrator for assessments within a context.

    Each ``GreenAgent`` instance owns its own UES server and can run
    multiple sequential assessments. The executor creates one
    ``GreenAgent`` per ``context_id``.

    Long-lived resources (UES server, LLM instances) persist across
    assessments within the same context. Per-assessment helpers
    (``ActionLogBuilder``, ``NewMessageCollector``, ``ResponseGenerator``,
    ``CriteriaJudge``) are created fresh for each ``run()`` call.

    Lifecycle:
        1. Executor creates ``GreenAgent`` with allocated port.
        2. Executor calls ``startup()`` to spin up UES server and LLMs.
        3. Executor calls ``run()`` for each assessment task.
        4. Executor calls ``shutdown()`` when context is no longer needed.

    Attributes:
        config: Green agent configuration.
        ues_port: Port for this agent's UES server.
        ues_client: Async client for UES API calls (proctor-level).
            Set during ``startup()``.
        response_llm: LLM for generating character responses.
            Created during ``__init__``.
        evaluation_llm: LLM for criteria evaluation (temperature=0).
            Created during ``__init__``.

    Example:
        >>> config = GreenAgentConfig()
        >>> agent = GreenAgent(ues_port=8100, config=config)
        >>> await agent.startup()
        >>> results = await agent.run(
        ...     task_id="task-1",
        ...     emitter=emitter,
        ...     scenario=scenario,
        ...     evaluators=evaluators,
        ...     purple_client=purple_client,
        ...     assessment_config={},
        ... )
        >>> await agent.shutdown()
    """

    def __init__(self, ues_port: int, config: GreenAgentConfig) -> None:
        """Initialize GreenAgent (does not start UES server).

        Creates LLM instances and the ``UESServerManager`` but does not
        start the UES server subprocess. Call ``startup()`` to start
        the server.

        Args:
            ues_port: Port number for the UES server this agent will own.
            config: Green agent configuration with LLM model names,
                default turn settings, and other parameters.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start UES server and create the async UES client.

        Starts the UES server subprocess via ``UESServerManager.start()``
        and creates an ``AsyncUESClient`` configured with the proctor-level
        admin API key.

        Must be called after ``__init__`` and before any ``run()`` calls.

        Raises:
            UESServerError: If the UES server fails to start.
            TimeoutError: If the UES server doesn't become ready in time.
        """
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Stop UES server and clean up all resources.

        Stops the UES server subprocess, closes the async UES client,
        and cancels any background tasks. Safe to call multiple times.

        After ``shutdown()``, this ``GreenAgent`` instance should not
        be reused.
        """
        raise NotImplementedError

    async def cancel(self, task_id: str) -> None:
        """Request cancellation of an ongoing assessment.

        Sets a cancellation flag that the turn loop checks between
        iterations. The assessment will complete its current turn
        and then exit gracefully with a ``"cancelled"`` status.

        If the given ``task_id`` does not match the currently running
        assessment, this method is a no-op.

        Args:
            task_id: The task ID of the assessment to cancel. Must match
                the ``task_id`` passed to the active ``run()`` call.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        task_id: str,
        emitter: TaskUpdateEmitter,
        scenario: ScenarioConfig,
        evaluators: EvaluatorRegistry,
        purple_client: A2AClientWrapper,
        assessment_config: dict[str, Any],
    ) -> AssessmentResults:
        """Run a complete assessment.

        This is the main entry point for executing a single assessment.
        It orchestrates the entire assessment lifecycle:

        1. **Setup**: Reset UES, load scenario state, create Purple API key,
           instantiate per-assessment helpers.
        2. **Start**: Build initial state summary, send ``AssessmentStartMessage``
           to the Purple agent.
        3. **Turn Loop**: Iterate turns until max turns reached, Purple
           signals early completion, or cancellation is requested.
        4. **Evaluation**: Capture final state, run ``CriteriaJudge``,
           aggregate scores.
        5. **Completion**: Send ``AssessmentCompleteMessage``, revoke
           Purple's API key, build and return results.

        Args:
            task_id: Unique task ID for this assessment (from A2A).
            emitter: Task update emitter for observability. Used to emit
                structured updates at each phase of the assessment.
            scenario: Scenario configuration with initial state, characters,
                evaluation criteria, and timing parameters.
            evaluators: Registry mapping evaluator IDs to async evaluator
                functions. Used by ``CriteriaJudge`` for programmatic
                criterion evaluation.
            purple_client: A2A client wrapper for communicating with the
                Purple agent. Used to send turn messages and receive
                responses.
            assessment_config: Additional assessment configuration overrides
                (e.g., ``max_turns``, ``time_step``). These override
                scenario defaults.

        Returns:
            Complete assessment results including scores, criterion results,
            and the full action log.

        Raises:
            RuntimeError: If ``startup()`` has not been called.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Turn orchestration (private)
    # ------------------------------------------------------------------

    async def _run_turn(
        self,
        turn: int,
        emitter: TaskUpdateEmitter,
        purple_client: A2AClientWrapper,
        action_log_builder: ActionLogBuilder,
        message_collector: NewMessageCollector,
        response_generator: ResponseGenerator,
    ) -> TurnResult:
        """Execute a single assessment turn.

        A turn consists of:
        1. Sending a ``TurnStartMessage`` to Purple.
        2. Waiting for Purple's ``TurnCompleteMessage`` (or
           ``EarlyCompletionMessage``).
        3. Running end-of-turn processing via ``_process_turn_end()``.

        Time is advanced in two phases within ``_process_turn_end()``:
        first by 1 second (to apply Purple's scheduled events), then
        by the remainder of the time step (to fire character responses).

        Args:
            turn: The 1-based turn number.
            emitter: Task update emitter for turn-level observability.
            purple_client: A2A client for communicating with Purple.
            action_log_builder: Builder tracking Purple agent actions.
            message_collector: Collector for new messages from UES.
            response_generator: Generator for character responses.

        Returns:
            ``TurnResult`` describing the turn outcome, including
            actions taken, events processed, and whether early
            completion was requested.
        """
        raise NotImplementedError

    async def _process_turn_end(
        self,
        turn: int,
        turn_start_time: datetime,
        time_step: str,
        emitter: TaskUpdateEmitter,
        action_log_builder: ActionLogBuilder,
        message_collector: NewMessageCollector,
        response_generator: ResponseGenerator,
    ) -> EndOfTurnResult:
        """Process end-of-turn: apply events, generate responses, advance time.

        This method handles all processing after Purple completes its turn
        and before the next turn begins. It is intentionally separated from
        ``_run_turn()`` to allow the processing strategy to be changed
        independently of the turn orchestration logic.

        The current implementation uses a batch strategy:
          1. **Apply advance** (1s): Advance UES time by 1 second to make
             Purple's submitted events fire and become visible.
          2. **Collect events**: Retrieve executed events from UES and
             update the action log.
          3. **Collect messages**: Gather new messages from UES modality
             states (emails, SMS, calendar events).
          4. **Generate responses**: Use ``ResponseGenerator`` to create
             character responses to new messages.
          5. **Schedule responses**: Inject responses into UES via
             proctor-level API calls.
          6. **Remainder advance**: Advance UES time by the rest of the
             time step to fire character responses.

        Future alternatives (e.g., event-by-event processing) can replace
        this method without touching ``_run_turn()``.

        Args:
            turn: Current turn number (1-based).
            turn_start_time: Simulation time at the start of this turn.
            time_step: ISO 8601 duration string for total time advancement
                this turn (e.g., ``"PT1H"``).
            emitter: Task update emitter for observability.
            action_log_builder: Builder tracking Purple agent actions.
            message_collector: Collector for new messages from UES.
            response_generator: Generator for character responses.

        Returns:
            ``EndOfTurnResult`` with counts of actions, events, and
            responses generated.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Time management (private)
    # ------------------------------------------------------------------

    async def _advance_time(self, time_step: str) -> Any:
        """Advance UES simulation time by a given duration.

        Parses the ISO 8601 duration string and calls the UES client's
        ``time.advance()`` method.

        Args:
            time_step: ISO 8601 duration string (e.g., ``"PT1H"``,
                ``"PT1S"``, ``"PT30M"``).

        Returns:
            The ``AdvanceTimeResponse`` from UES, which includes the
            number of events executed during the advance.
        """
        raise NotImplementedError

    async def _advance_remainder(
        self,
        time_step: str,
        apply_seconds: int = 1,
    ) -> Any:
        """Advance UES simulation time by the remainder after apply advance.

        Computes ``time_step - apply_seconds`` and advances by that amount.
        If the remainder is zero or negative (``time_step <= apply_seconds``),
        no advance is performed and a zero-event response is returned.

        This is used in the two-phase time advancement strategy: after the
        initial 1-second apply advance fires Purple's events, this method
        advances by the remaining time to fire character responses.

        Args:
            time_step: Original ISO 8601 duration string (e.g., ``"PT1H"``).
            apply_seconds: Seconds already advanced in the apply phase
                (default: 1).

        Returns:
            The ``AdvanceTimeResponse`` from UES, or a zero-event
            placeholder if no advancement was needed.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Purple agent communication (private)
    # ------------------------------------------------------------------

    async def _send_and_wait_purple(
        self,
        purple_client: A2AClientWrapper,
        message: BaseModel,
        timeout: float,
    ) -> TurnCompleteMessage | EarlyCompletionMessage:
        """Send a message to Purple agent and wait for a response.

        Serializes the given Pydantic model to a JSON ``DataPart`` in an
        A2A ``Message``, sends it via the A2A client, and waits for the
        Purple agent's response (blocking mode).

        The response is parsed based on its ``message_type`` field:
        ``"turn_complete"`` → ``TurnCompleteMessage``,
        ``"early_completion"`` → ``EarlyCompletionMessage``.

        Args:
            purple_client: A2A client wrapper for the Purple agent.
            message: Pydantic model to send (e.g., ``TurnStartMessage``).
                Serialized to JSON via ``model_dump()``.
            timeout: Maximum seconds to wait for Purple's response.

        Returns:
            Parsed response message — either a ``TurnCompleteMessage``
            or an ``EarlyCompletionMessage``.

        Raises:
            TimeoutError: If Purple doesn't respond within ``timeout``.
            ValueError: If the response ``message_type`` is unexpected.
        """
        from a2a.types import DataPart, Message, Part, Role

        # Create A2A message with data payload
        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data=message.model_dump(mode="json")))],
            message_id=str(uuid.uuid4()),
        )

        # Send and wait for response (blocking=True)
        response = await asyncio.wait_for(
            purple_client.send_message(a2a_message, blocking=True),
            timeout=timeout,
        )

        # Parse response
        response_data = self._extract_response_data(response)
        message_type = response_data.get("message_type")

        if message_type == "turn_complete":
            return TurnCompleteMessage.model_validate(response_data)
        elif message_type == "early_completion":
            return EarlyCompletionMessage.model_validate(response_data)
        else:
            raise ValueError(f"Unexpected message type from Purple: {message_type}")

    def _extract_response_data(self, response: Any) -> dict[str, Any]:
        """Extract the data payload from an A2A response.

        Navigates the A2A ``Task`` or ``Message`` response structure to
        find and return the ``DataPart`` payload as a dictionary.

        Args:
            response: A2A ``Task`` or ``Message`` returned by
                ``purple_client.send_message()``.

        Returns:
            Dictionary of the data payload from the response.

        Raises:
            ValueError: If no ``DataPart`` is found in the response.
        """
        from a2a.types import DataPart, Message, Task

        # Handle Task response (most common for blocking=True)
        if isinstance(response, Task):
            # Task has artifacts or history - check artifacts first
            if response.artifacts:
                for artifact in response.artifacts:
                    if artifact.parts:
                        for part in artifact.parts:
                            # Part is a discriminated union with root attribute
                            part_data = part.root if hasattr(part, "root") else part
                            if isinstance(part_data, DataPart):
                                return part_data.data
            # Check history messages
            if response.history:
                for msg in reversed(response.history):
                    if msg.role.value == "agent" and msg.parts:
                        for part in msg.parts:
                            part_data = part.root if hasattr(part, "root") else part
                            if isinstance(part_data, DataPart):
                                return part_data.data
            raise ValueError(
                "No DataPart found in Task response artifacts or history"
            )

        # Handle Message response
        if isinstance(response, Message):
            if response.parts:
                for part in response.parts:
                    part_data = part.root if hasattr(part, "root") else part
                    if isinstance(part_data, DataPart):
                        return part_data.data
            raise ValueError("No DataPart found in Message response parts")

        # Try duck typing for unknown response types
        if hasattr(response, "artifacts"):
            for artifact in response.artifacts:
                if hasattr(artifact, "parts") and artifact.parts:
                    for part in artifact.parts:
                        part_data = part.root if hasattr(part, "root") else part
                        if hasattr(part_data, "data"):
                            return part_data.data

        if hasattr(response, "parts"):
            for part in response.parts:
                part_data = part.root if hasattr(part, "root") else part
                if hasattr(part_data, "data"):
                    return part_data.data

        raise ValueError(
            f"Unable to extract DataPart from response of type {type(response).__name__}"
        )

    async def _send_assessment_start(
        self,
        purple_client: A2AClientWrapper,
        scenario: ScenarioConfig,
        initial_summary: InitialStateSummary,
        ues_url: str,
        api_key: str,
    ) -> None:
        """Send the ``AssessmentStartMessage`` to the Purple agent.

        Constructs and sends the initial message that kicks off an
        assessment, providing Purple with the scenario context, UES
        connection details, and initial state summary.

        Args:
            purple_client: A2A client for the Purple agent.
            scenario: Scenario configuration with user prompt and metadata.
            initial_summary: Summary of UES initial state (email counts,
                calendar events, etc.).
            ues_url: Base URL for the UES server (e.g.,
                ``"http://127.0.0.1:8100"``).
            api_key: User-level API key secret for Purple to authenticate
                with UES.
        """
        # Get current simulation time
        time_state = await self.ues_client.time.get_state()

        message = AssessmentStartMessage(
            ues_url=ues_url,
            api_key=api_key,
            assessment_instructions=scenario.user_prompt,
            current_time=time_state.current_time,
            initial_state_summary=initial_summary,
        )

        # Send as a fire-and-forget message (no response expected)
        await purple_client.send_data(
            data=message.model_dump(mode="json"),
            blocking=False,
        )

        logger.debug(
            "Sent AssessmentStartMessage to Purple: ues_url=%s, current_time=%s",
            ues_url,
            time_state.current_time,
        )

    async def _send_assessment_complete(
        self,
        purple_client: A2AClientWrapper,
        reason: str,
    ) -> None:
        """Send the ``AssessmentCompleteMessage`` to the Purple agent.

        Notifies Purple that the assessment has ended and provides the
        completion reason.

        Args:
            purple_client: A2A client for the Purple agent.
            reason: Why the assessment ended. One of
                ``"scenario_complete"``, ``"early_completion"``,
                ``"max_turns_reached"``, or ``"cancelled"``.
        """
        # Map "max_turns_reached" or "cancelled" to valid AssessmentCompleteMessage reasons
        # Valid reasons: "scenario_complete", "early_completion", "timeout", "error"
        reason_mapping = {
            "scenario_complete": "scenario_complete",
            "early_completion": "early_completion",
            "max_turns_reached": "timeout",
            "cancelled": "timeout",
            "timeout": "timeout",
            "error": "error",
        }
        mapped_reason = reason_mapping.get(reason, "timeout")

        message = AssessmentCompleteMessage(reason=mapped_reason)

        # Send as fire-and-forget (no response expected)
        await purple_client.send_data(
            data=message.model_dump(mode="json"),
            blocking=False,
        )

        logger.debug("Sent AssessmentCompleteMessage to Purple: reason=%s", reason)

    # ------------------------------------------------------------------
    # API key management (private)
    # ------------------------------------------------------------------

    async def _create_user_api_key(
        self,
        assessment_id: str,
    ) -> tuple[str, str]:
        """Create a user-level API key for the Purple agent.

        Creates a new API key via the UES ``/keys`` endpoint with
        ``USER_PERMISSIONS``. The returned ``key_id`` serves a dual
        purpose: identifying the key for later revocation, and acting
        as the ``agent_id`` for all UES events created by Purple
        (enabling event filtering in the action log).

        Args:
            assessment_id: Unique assessment ID, used in the key name
                for identification.

        Returns:
            A tuple of ``(api_key_secret, key_id)``. The secret is given
            to Purple for authentication; the ``key_id`` is used as
            Purple's ``agent_id`` for event attribution.
        """
        raise NotImplementedError

    async def _revoke_user_api_key(self, key_id: str) -> None:
        """Revoke a user API key after an assessment ends.

        Deletes the key via the UES ``/keys/{key_id}`` endpoint.
        Silently ignores 404 responses (key already revoked or
        doesn't exist).

        Args:
            key_id: The key ID to revoke (returned by
                ``_create_user_api_key``).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # UES setup (private)
    # ------------------------------------------------------------------

    async def _setup_ues(self, scenario: ScenarioConfig) -> None:
        """Reset UES and load scenario initial state.

        Clears all existing UES state, imports the scenario's initial
        state via the ``/scenario/import/full`` endpoint, and starts
        the simulation with ``auto_advance=False`` (Green controls time).

        Args:
            scenario: Scenario configuration containing the
                ``initial_state`` to load into UES.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State management (private)
    # ------------------------------------------------------------------

    async def _capture_state_snapshot(self) -> dict[str, Any]:
        """Capture a snapshot of the current UES modality states.

        Queries all modalities (email, SMS, calendar, chat, time) and
        returns their state as serialized dictionaries. Used to capture
        initial and final state for evaluation context.

        Returns:
            Dictionary with keys ``"email"``, ``"sms"``, ``"calendar"``,
            ``"chat"``, and ``"time"``, each containing the serialized
            modality state.
        """
        raise NotImplementedError

    async def _build_initial_state_summary(self) -> InitialStateSummary:
        """Build a summary of the initial UES state for the Purple agent.

        Queries UES modality states and constructs an
        ``InitialStateSummary`` with counts for each modality. This
        summary is included in the ``AssessmentStartMessage`` so Purple
        has an overview without needing to query each modality.

        Returns:
            An ``InitialStateSummary`` with email, SMS, calendar, and
            chat summaries.
        """
        raise NotImplementedError

    def _count_events_today(self, calendar_state: Any) -> int:
        """Count the number of calendar events scheduled for today.

        Compares each event's start time against the current simulation
        date to determine which events fall on the current day.

        Args:
            calendar_state: The calendar modality state object from UES.

        Returns:
            Number of events whose start time falls on today's date.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Response scheduling (private)
    # ------------------------------------------------------------------

    async def _schedule_response(
        self,
        scheduled: ScheduledResponse,
    ) -> None:
        """Schedule a character response in UES.

        Dispatches to the modality-specific scheduling method based on
        ``scheduled.modality``. Uses proctor-level API calls to inject
        responses as if they came from external sources.

        Args:
            scheduled: The response to schedule, containing modality,
                content, timing, and contact information.

        Raises:
            ValueError: If ``scheduled.modality`` is not a recognized
                modality (``"email"``, ``"sms"``, or ``"calendar"``).
        """
        raise NotImplementedError

    async def _schedule_email_response(
        self,
        scheduled: ScheduledResponse,
    ) -> None:
        """Schedule an email response via UES.

        Uses ``ues_client.email.receive()`` (proctor-level) to inject
        an email from the character into the UES mailbox.

        Args:
            scheduled: The email response to schedule. Must have
                ``character_email``, ``recipients``, ``subject``,
                ``content``, and threading fields populated.
        """
        raise NotImplementedError

    async def _schedule_sms_response(
        self,
        scheduled: ScheduledResponse,
    ) -> None:
        """Schedule an SMS response via UES.

        Uses ``ues_client.sms.receive()`` (proctor-level) to inject
        an SMS message from the character into the UES conversation.

        Args:
            scheduled: The SMS response to schedule. Must have
                ``character_phone``, ``recipients``, and ``content``
                populated.
        """
        raise NotImplementedError

    async def _schedule_calendar_response(
        self,
        scheduled: ScheduledResponse,
    ) -> None:
        """Schedule a calendar RSVP response via UES.

        Uses ``ues_client.calendar.respond_to_event()`` (proctor-level)
        to inject a calendar RSVP from the character.

        Args:
            scheduled: The calendar response to schedule. Must have
                ``character_email``, ``event_id``, ``rsvp_status``,
                and optionally ``rsvp_comment`` populated.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Result building (private)
    # ------------------------------------------------------------------

    def _build_results(
        self,
        assessment_id: str,
        scenario: ScenarioConfig,
        scores: Scores,
        criteria_results: list[CriterionResult],
        action_log: list[Any],
        turns_completed: int,
        duration: float,
        status: str,
    ) -> AssessmentResults:
        """Build the final ``AssessmentResults`` artifact.

        Assembles all assessment data into a single ``AssessmentResults``
        Pydantic model suitable for serialization as an A2A artifact.

        Args:
            assessment_id: Unique assessment ID.
            scenario: Scenario configuration used for this assessment.
            scores: Aggregated scores from ``CriteriaJudge``.
            criteria_results: Individual criterion evaluation results.
            action_log: List of ``ActionLogEntry`` objects from the
                action log builder.
            turns_completed: Number of turns completed before the
                assessment ended.
            duration: Total assessment duration in seconds.
            status: Final assessment status (``"completed"``,
                ``"cancelled"``, ``"failed"``).

        Returns:
            A fully-populated ``AssessmentResults`` instance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Health monitoring (private)
    # ------------------------------------------------------------------

    async def _check_ues_health(self) -> bool:
        """Check if the UES server is still running and responsive.

        Checks both the subprocess status (via ``UESServerManager``) and
        HTTP health (via the ``/health`` endpoint).

        Returns:
            ``True`` if the UES process is alive and the ``/health``
            endpoint returns HTTP 200; ``False`` otherwise.
        """
        raise NotImplementedError
