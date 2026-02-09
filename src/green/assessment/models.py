"""Assessment data models for turn orchestration.

This module defines data models used by the GreenAgent during assessment
execution. These models communicate results between the turn loop and
its helper methods.

Classes:
    TurnResult: Outcome of a single assessment turn.
    EndOfTurnResult: Outcome of end-of-turn processing.

Design Notes:
    - Both models are dataclasses for simplicity and consistency with
      other internal models (e.g., ScheduledResponse).
    - TurnResult is returned by ``_run_turn()`` to the main ``run()`` loop.
    - EndOfTurnResult is returned by ``_process_turn_end()`` to ``_run_turn()``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnResult:
    """Result from executing a single assessment turn.

    Returned by ``_run_turn()`` to communicate the turn outcome back to
    the main assessment loop in ``run()``.

    Attributes:
        turn_number: The 1-based turn index.
        actions_taken: Number of Purple agent actions observed this turn.
        time_step: ISO 8601 duration string for this turn's time step.
        events_processed: Total UES events processed across both time
            advances (apply + remainder).
        early_completion: Whether Purple signaled early completion.
        notes: Optional free-text notes about the turn (e.g., warnings).
        error: Error message if the turn failed; ``None`` on success.
    """

    turn_number: int
    actions_taken: int
    time_step: str
    events_processed: int
    early_completion: bool
    notes: str | None = None
    error: str | None = None


@dataclass
class EndOfTurnResult:
    """Result from end-of-turn processing.

    Returned by ``_process_turn_end()`` to communicate processing outcome
    back to ``_run_turn()``. Captures the aggregate counts from applying
    Purple's events, collecting actions, and generating character responses.

    Attributes:
        actions_taken: Number of Purple agent actions observed.
        total_events: Total UES events executed across both time
            advances (apply + remainder).
        responses_generated: Number of character responses scheduled.
    """

    actions_taken: int
    total_events: int
    responses_generated: int
