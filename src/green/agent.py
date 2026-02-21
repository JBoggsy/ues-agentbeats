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
        self.config = config
        self.ues_port = ues_port

        # Create LLM instances (long-lived, persist across assessments)
        self.response_llm: BaseChatModel = LLMFactory.create(
            config.response_generator_model,
            temperature=0.7,
        )
        self.evaluation_llm: BaseChatModel = LLMFactory.create(
            config.evaluation_model,
            temperature=0.0,
        )

        # UES client is set during startup()
        self.ues_client: AsyncUESClient | None = None

        # UES server manager (handles subprocess lifecycle)
        self._ues_server = UESServerManager(port=ues_port)
        self._ues_port = ues_port
        self._proctor_api_key = self._ues_server.admin_api_key

        # Cancellation tracking
        self._current_task_id: str | None = None
        self._cancelled = False

        logger.info(
            "GreenAgent initialized: ues_port=%d, response_model=%s, eval_model=%s",
            ues_port,
            config.response_generator_model,
            config.evaluation_model,
        )

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
        from ues.client import AsyncUESClient

        logger.info("Starting UES server on port %d", self.ues_port)

        # Start UES server subprocess
        await self._ues_server.start()

        # Create async UES client with proctor-level API key
        self.ues_client = AsyncUESClient(
            base_url=self._ues_server.base_url,
            api_key=self._proctor_api_key,
        )

        logger.info("GreenAgent startup complete: ues_url=%s", self._ues_server.base_url)

    async def shutdown(self) -> None:
        """Stop UES server and clean up all resources.

        Stops the UES server subprocess, closes the async UES client,
        and cancels any background tasks. Safe to call multiple times.

        After ``shutdown()``, this ``GreenAgent`` instance should not
        be reused.
        """
        logger.info("Shutting down GreenAgent (ues_port=%d)", self.ues_port)

        # Close the UES client if it exists
        if self.ues_client is not None:
            try:
                await self.ues_client.close()
            except Exception:
                logger.exception("Error closing UES client")
            self.ues_client = None

        # Stop the UES server subprocess
        await self._ues_server.stop()

        logger.info("GreenAgent shutdown complete")

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
        if self._current_task_id == task_id:
            logger.info("Cancellation requested for task %s", task_id)
            self._cancelled = True
        else:
            logger.debug(
                "Ignoring cancellation request for task %s (current task: %s)",
                task_id,
                self._current_task_id,
            )

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
        if self.ues_client is None:
            raise RuntimeError("startup() must be called before run()")

        # Track start time for duration calculation
        start_time = datetime.now(timezone.utc)
        assessment_id = str(uuid.uuid4())

        # Mark this assessment as active for cancellation support
        self._current_task_id = task_id
        self._cancelled = False

        user_key_id: str | None = None

        try:
            # === Phase 1: Setup ===
            await emitter.assessment_started(
                assessment_id=assessment_id,
                scenario_id=scenario.scenario_id,
                participant_url=purple_client.agent_url,
                start_time=start_time,
            )

            # Reset UES and load scenario
            await self._setup_ues(scenario)

            # Inject the scenario's user task into chat before Purple starts.
            await self._inject_user_prompt_chat(scenario.user_prompt)

            # Create user API key for Purple
            user_api_key, user_key_id = await self._create_user_api_key(assessment_id)

            # Create per-assessment helpers
            action_log_builder = ActionLogBuilder(purple_agent_id=user_key_id)
            message_collector = NewMessageCollector(client=self.ues_client)
            response_generator = ResponseGenerator(
                client=self.ues_client,
                scenario_config=scenario,
                response_llm=self.response_llm,
                summarization_llm=self.response_llm,
            )
            criteria_judge = CriteriaJudge(
                llm=self.evaluation_llm,
                criteria=scenario.criteria,
                evaluators=evaluators,
                emitter=emitter,
            )

            # Initialize message collector
            time_state = await self.ues_client.time.get_state()
            await message_collector.initialize(time_state.current_time)

            # Capture initial state
            initial_state = await self._capture_state_snapshot()

            # Emit scenario loaded update
            await emitter.scenario_loaded(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                criteria_count=len(scenario.criteria),
                character_count=len(scenario.characters),
            )

            # === Phase 2: Send Start to Purple ===
            initial_summary = await self._build_initial_state_summary()
            ues_url = f"http://127.0.0.1:{self.ues_port}"
            await self._send_assessment_start(
                purple_client=purple_client,
                initial_summary=initial_summary,
                ues_url=ues_url,
                api_key=user_api_key,
            )

            # === Phase 3: Turn Loop ===
            max_turns = assessment_config.get("max_turns", self.config.default_max_turns)
            turn = 0
            completion_reason = "scenario_complete"

            while turn < max_turns and not self._cancelled:
                turn += 1

                turn_result = await self._run_turn(
                    turn=turn,
                    emitter=emitter,
                    purple_client=purple_client,
                    action_log_builder=action_log_builder,
                    message_collector=message_collector,
                    response_generator=response_generator,
                )

                if turn_result.early_completion:
                    completion_reason = "early_completion"
                    break

                if turn_result.error:
                    logger.warning(
                        "Turn %d encountered error: %s", turn, turn_result.error
                    )

            if self._cancelled:
                completion_reason = "cancelled"
            elif turn >= max_turns and completion_reason == "scenario_complete":
                completion_reason = "max_turns_reached"

            # === Phase 4: Evaluation ===
            dimensions = criteria_judge.get_dimensions()
            await emitter.evaluation_started(
                criteria_count=len(scenario.criteria),
                dimensions=dimensions,
            )

            final_state = await self._capture_state_snapshot()

            eval_context = AgentBeatsEvalContext(
                client=self.ues_client,
                scenario_config=scenario.model_dump(),
                action_log=action_log_builder.get_log(),
                initial_state=initial_state,
                final_state=final_state,
                user_prompt=scenario.user_prompt,
            )

            criteria_results = await criteria_judge.evaluate_all(eval_context)
            scores = criteria_judge.aggregate_scores(criteria_results)

            # === Phase 5: Completion ===
            # Map completion_reason to valid AssessmentCompleteMessage reason
            complete_reason_map = {
                "scenario_complete": "scenario_complete",
                "early_completion": "early_completion",
                "max_turns_reached": "scenario_complete",
                "cancelled": "error",
            }
            await self._send_assessment_complete(
                purple_client, complete_reason_map.get(completion_reason, "error")
            )

            # Revoke user API key
            if user_key_id:
                await self._revoke_user_api_key(user_key_id)

            # Build results
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            action_log = action_log_builder.get_log()

            results = self._build_results(
                assessment_id=assessment_id,
                scenario=scenario,
                scores=scores,
                criteria_results=criteria_results,
                action_log=action_log,
                turns_completed=turn,
                duration=duration,
                status="completed" if completion_reason != "cancelled" else "cancelled",
                participant=purple_client.agent_url,
            )

            # Map completion_reason to valid emitter reason
            emitter_reason_map = {
                "scenario_complete": "scenario_complete",
                "early_completion": "early_completion",
                "max_turns_reached": "scenario_complete",
                "cancelled": "error",
            }

            await emitter.assessment_completed(
                reason=emitter_reason_map.get(completion_reason, "scenario_complete"),
                total_turns=turn,
                total_actions=len(action_log),
                duration_seconds=duration,
                overall_score=scores.overall.score,
                max_score=scores.overall.max_score,
            )

            return results

        except Exception as e:
            logger.exception("Assessment failed with error: %s", e)

            # Try to revoke API key on error
            if user_key_id:
                try:
                    await self._revoke_user_api_key(user_key_id)
                except Exception:
                    pass

            # Emit error update
            await emitter.error_occurred(
                error_type="internal_error",
                error_message=str(e),
                recoverable=False,
                context={"assessment_id": assessment_id, "task_id": task_id},
            )

            raise

        finally:
            self._current_task_id = None

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
        # Get current time before turn
        time_state = await self.ues_client.time.get_state()
        turn_start_time = time_state.current_time

        # Step 1: Send TurnStart to Purple
        await emitter.turn_started(
            turn_number=turn,
            current_time=turn_start_time,
            events_pending=0,
        )

        turn_start_msg = TurnStartMessage(
            turn_number=turn,
            current_time=turn_start_time,
            events_processed=0,
        )

        # Step 2: Wait for Purple's response
        try:
            response = await self._send_and_wait_purple(
                purple_client=purple_client,
                message=turn_start_msg,
                timeout=self.config.default_turn_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Turn {turn} timed out waiting for Purple")
            await emitter.turn_completed(
                turn_number=turn,
                actions_taken=0,
                time_advanced="PT0S",
                early_completion_requested=False,
            )
            return TurnResult(
                turn_number=turn,
                actions_taken=0,
                time_step="PT0S",
                events_processed=0,
                early_completion=False,
                error="timeout",
            )

        # Check for early completion
        if isinstance(response, EarlyCompletionMessage):
            await emitter.turn_completed(
                turn_number=turn,
                actions_taken=0,
                time_advanced="PT0S",
                early_completion_requested=True,
            )
            return TurnResult(
                turn_number=turn,
                actions_taken=0,
                time_step="PT0S",
                events_processed=0,
                early_completion=True,
                notes=response.reason,
            )

        # Response is TurnCompleteMessage
        turn_complete_msg = response
        time_step = turn_complete_msg.time_step or "PT1H"

        # Step 3: End-of-turn processing
        end_of_turn = await self._process_turn_end(
            turn=turn,
            turn_start_time=turn_start_time,
            time_step=time_step,
            emitter=emitter,
            action_log_builder=action_log_builder,
            message_collector=message_collector,
            response_generator=response_generator,
        )

        await emitter.turn_completed(
            turn_number=turn,
            actions_taken=end_of_turn.actions_taken,
            time_advanced=time_step,
            early_completion_requested=False,
        )

        return TurnResult(
            turn_number=turn,
            actions_taken=end_of_turn.actions_taken,
            time_step=time_step,
            events_processed=end_of_turn.total_events,
            early_completion=False,
            notes=turn_complete_msg.notes,
        )

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
        # == Phase 1: Apply advance — advance by 1s to apply Purple's events ==
        apply_result = await self._advance_time("PT1S")

        # Get time after apply advance
        time_state = await self.ues_client.time.get_state()
        apply_time = time_state.current_time

        # == Phase 2: Collect and process events ==
        action_log_builder.start_turn(turn)

        events = await self.ues_client.events.list_events(
            start_time=turn_start_time,
            end_time=apply_time,
            status="executed",
        )

        purple_entries, _green_events = action_log_builder.add_events_from_turn(
            [e.model_dump() if hasattr(e, "model_dump") else e for e in events.events]
        )
        action_log_builder.end_turn()

        # Emit action updates for observability
        for entry in purple_entries:
            await emitter.action_observed(
                turn_number=turn,
                timestamp=entry.timestamp,
                action=entry.action,
                parameters=entry.parameters,
                success=entry.success,
                error_message=entry.error_message,
            )

        # == Phase 3: Collect new messages ==
        new_messages = await message_collector.collect(apply_time)

        # == Phase 4: Generate character responses ==
        responses = await response_generator.process_new_messages(
            new_messages=new_messages,
            current_time=apply_time,
        )

        # == Phase 5: Schedule responses in UES ==
        for scheduled in responses:
            await self._schedule_response(scheduled)

        await emitter.responses_generated(
            turn_number=turn,
            responses_count=len(responses),
            characters_involved=[r.character_name for r in responses],
        )

        # == Phase 6: Remainder advance — advance by (time_step - 1s) ==
        remainder_result = await self._advance_remainder(
            time_step=time_step, apply_seconds=1
        )

        return EndOfTurnResult(
            actions_taken=len(purple_entries),
            total_events=(
                apply_result.events_executed + remainder_result.events_executed
            ),
            responses_generated=len(responses),
        )

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
        from src.green.scenarios.schema import parse_iso8601_duration

        duration = parse_iso8601_duration(time_step)
        seconds = int(duration.total_seconds())

        return await self.ues_client.time.advance(seconds=seconds)

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
        from dataclasses import dataclass

        from src.green.scenarios.schema import parse_iso8601_duration

        duration = parse_iso8601_duration(time_step)
        total_seconds = int(duration.total_seconds())
        remainder_seconds = total_seconds - apply_seconds

        if remainder_seconds <= 0:
            # No advancement needed — return a zero-event placeholder
            @dataclass
            class ZeroAdvanceResult:
                """Placeholder result when no time advancement is needed."""

                events_executed: int = 0
                events_failed: int = 0

            return ZeroAdvanceResult()

        return await self.ues_client.time.advance(seconds=remainder_seconds)

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
        initial_summary: InitialStateSummary,
        ues_url: str,
        api_key: str,
    ) -> None:
        """Send the ``AssessmentStartMessage`` to the Purple agent.

        Constructs and sends the initial message that kicks off an
        assessment, providing Purple with fixed high-level protocol
        instructions, UES connection details, and initial state summary.
        Scenario-specific tasks are delivered separately via chat.

        Args:
            purple_client: A2A client for the Purple agent.
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
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._ues_server.base_url}/keys",
                headers={"X-API-Key": self._ues_server.admin_api_key},
                json={
                    "name": f"Purple Agent ({assessment_id})",
                    "permissions": USER_PERMISSIONS,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["secret"], data["key_id"]

    async def _revoke_user_api_key(self, key_id: str) -> None:
        """Revoke a user API key after an assessment ends.

        Deletes the key via the UES ``/keys/{key_id}`` endpoint.
        Silently ignores 404 responses (key already revoked or
        doesn't exist).

        Args:
            key_id: The key ID to revoke (returned by
                ``_create_user_api_key``).
        """
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self._ues_server.base_url}/keys/{key_id}",
                headers={"X-API-Key": self._ues_server.admin_api_key},
                timeout=10.0,
            )
            # Ignore 404 (key already revoked or doesn't exist)
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()

    # ------------------------------------------------------------------
    # UES setup (private)
    # ------------------------------------------------------------------

    async def _setup_ues(self, scenario: ScenarioConfig) -> None:
        """Reset UES and load scenario initial state.

        Clears all existing UES state, imports the scenario's initial
        state via the UES client's scenario import method, and starts
        the simulation with ``auto_advance=False`` (Green controls time).

        Args:
            scenario: Scenario configuration containing the
                ``initial_state`` to load into UES. Must be a full
                UES scenario export dict with ``metadata``,
                ``environment``, and ``events`` sections.

        Raises:
            Exception: If the scenario import fails (e.g., validation
                error from UES).
        """
        # Clear UES state (removes all events and modality states)
        await self.ues_client.simulation.clear()

        # Load scenario initial state via client library.
        # initial_state may be the raw request body (with "scenario" and
        # "strict_modalities" wrappers) or just the inner scenario dict.
        # Unwrap if needed — import_full() adds its own wrapper.
        state = scenario.initial_state
        if "scenario" in state and "metadata" not in state:
            # Wrapped format: {"scenario": {...}, "strict_modalities": ...}
            strict = state.get("strict_modalities", False)
            state = state["scenario"]
        else:
            strict = False

        await self.ues_client.scenario.import_full(
            scenario=state,
            strict_modalities=strict,
        )

        # Start simulation with manual time control (Green advances time)
        await self.ues_client.simulation.start(auto_advance=False)

    async def _inject_user_prompt_chat(self, user_prompt: str) -> None:
        """Inject the scenario user prompt into chat as an immediate user message.

        This is called at assessment start, before the first Purple turn,
        so scenario-specific tasks are always delivered through the chat
        modality.

        Args:
            user_prompt: Scenario task instructions to inject into chat.
        """
        await self.ues_client.chat.send(
            role="user",
            content=user_prompt,
            conversation_id="user-assistant",
        )

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
        email_state, sms_state, calendar_state, chat_state, time_state = (
            await asyncio.gather(
                self.ues_client.email.get_state(),
                self.ues_client.sms.get_state(),
                self.ues_client.calendar.get_state(),
                self.ues_client.chat.get_state(),
                self.ues_client.time.get_state(),
            )
        )
        return {
            "email": email_state.model_dump(mode="json"),
            "sms": sms_state.model_dump(mode="json"),
            "calendar": calendar_state.model_dump(mode="json"),
            "chat": chat_state.model_dump(mode="json"),
            "time": time_state.model_dump(mode="json"),
        }

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
        email_state, sms_state, calendar_state, chat_state, time_state = (
            await asyncio.gather(
                self.ues_client.email.get_state(),
                self.ues_client.sms.get_state(),
                self.ues_client.calendar.get_state(),
                self.ues_client.chat.get_state(),
                self.ues_client.time.get_state(),
            )
        )

        return InitialStateSummary(
            email=EmailSummary(
                total_emails=email_state.total_email_count,
                total_threads=len(email_state.threads),
                unread=email_state.unread_count,
                draft_count=len(email_state.folders.get("drafts", [])),
            ),
            sms=SMSSummary(
                total_messages=sms_state.total_message_count,
                total_conversations=len(sms_state.conversations),
                unread=sms_state.unread_count,
            ),
            calendar=CalendarSummary(
                event_count=len(calendar_state.events),
                calendar_count=len(calendar_state.calendars),
                events_today=self._count_events_today(
                    calendar_state, time_state.current_time
                ),
            ),
            chat=ChatSummary(
                total_messages=chat_state.total_message_count,
                conversation_count=chat_state.conversation_count,
            ),
        )

    def _count_events_today(
        self, calendar_state: Any, current_time: datetime | str
    ) -> int:
        """Count the number of calendar events scheduled for today.

        Compares each event's start time against the current simulation
        date to determine which events fall on the current day.

        Args:
            calendar_state: The calendar modality state object from UES.
                Expected to have an ``events`` attribute (dict of event_id
                to CalendarEvent) where each event has a ``start`` datetime.
            current_time: The current simulation time from UES. Used to
                determine what "today" means for the count.

        Returns:
            Number of events whose start time falls on today's date.
        """
        if not calendar_state.events:
            return 0

        # Parse current_time if it's a string
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time.replace("Z", "+00:00"))

        today_date = current_time.date()
        today_count = 0

        for event in calendar_state.events.values():
            event_start = event.start
            if isinstance(event_start, str):
                event_start = datetime.fromisoformat(event_start.replace("Z", "+00:00"))

            if event_start.date() == today_date:
                today_count += 1

        return today_count

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
        if scheduled.modality == "email":
            await self._schedule_email_response(scheduled)
        elif scheduled.modality == "sms":
            await self._schedule_sms_response(scheduled)
        elif scheduled.modality == "calendar":
            await self._schedule_calendar_response(scheduled)
        else:
            raise ValueError(f"Unknown modality: {scheduled.modality}")

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
        logger.debug(
            "Scheduling email response from %s to %s: %s",
            scheduled.character_email,
            scheduled.recipients,
            scheduled.subject,
        )
        await self.ues_client.email.receive(
            from_address=scheduled.character_email,
            to_addresses=scheduled.recipients,
            subject=scheduled.subject or "",
            body_text=scheduled.content or "",
            cc_addresses=scheduled.cc_recipients or None,
            thread_id=scheduled.thread_id,
            in_reply_to=scheduled.in_reply_to,
            references=scheduled.references or None,
            sent_at=scheduled.scheduled_time,
        )

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
        logger.debug(
            "Scheduling SMS response from %s to %s",
            scheduled.character_phone,
            scheduled.recipients,
        )
        await self.ues_client.sms.receive(
            from_number=scheduled.character_phone,
            to_numbers=scheduled.recipients,
            body=scheduled.content or "",
            replied_to_message_id=scheduled.original_message_id,
            sent_at=scheduled.scheduled_time,
        )

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
        logger.debug(
            "Scheduling calendar RSVP from %s for event %s: %s",
            scheduled.character_email,
            scheduled.event_id,
            scheduled.rsvp_status,
        )
        await self.ues_client.calendar.respond_to_event(
            event_id=scheduled.event_id,
            attendee_email=scheduled.character_email,
            response=scheduled.rsvp_status,
            comment=scheduled.rsvp_comment,
        )

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
        participant: str,
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
                ``"cancelled"``, ``"failed"``). Mapped to valid
                ``AssessmentStatus`` values.
            participant: Identifier for the Purple agent that was
                assessed (e.g., the agent's URL).

        Returns:
            A fully-populated ``AssessmentResults`` instance.
        """
        # Map status to valid AssessmentStatus ("completed", "failed", "timeout")
        status_map = {
            "completed": "completed",
            "cancelled": "failed",
            "failed": "failed",
            "timeout": "timeout",
        }
        mapped_status = status_map.get(status, "failed")

        return AssessmentResults(
            assessment_id=assessment_id,
            scenario_id=scenario.scenario_id,
            participant=participant,
            status=mapped_status,  # type: ignore[arg-type]
            duration_seconds=duration,
            turns_taken=turns_completed,
            actions_taken=len(action_log),
            scores=scores,
            criteria_results=criteria_results,
            action_log=action_log,
        )

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
        return await self._ues_server.check_health()
