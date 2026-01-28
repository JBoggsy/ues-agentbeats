"""Task management utilities for A2A protocol.

This module provides helper functions for managing A2A task state,
including state checks and task update event creation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from a2a.types import (
    Artifact,
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


# Terminal states where no further updates are expected
TERMINAL_STATES = frozenset({
    TaskState.completed,
    TaskState.failed,
    TaskState.canceled,
    TaskState.rejected,
})


def is_terminal_state(task: Task) -> bool:
    """Check if a task is in a terminal state.

    Terminal states are: completed, failed, canceled, rejected.

    Args:
        task: The Task to check.

    Returns:
        True if the task is in a terminal state.

    Example:
        >>> # Assuming task.status.state == TaskState.completed
        >>> is_terminal_state(task)
        True
    """
    return task.status.state in TERMINAL_STATES


def is_completed(task: Task) -> bool:
    """Check if a task has completed successfully.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'completed'.

    Example:
        >>> is_completed(task)  # True if task.status.state == TaskState.completed
        True
    """
    return task.status.state == TaskState.completed


def is_failed(task: Task) -> bool:
    """Check if a task has failed.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'failed'.
    """
    return task.status.state == TaskState.failed


def is_canceled(task: Task) -> bool:
    """Check if a task was canceled.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'canceled'.
    """
    return task.status.state == TaskState.canceled


def is_rejected(task: Task) -> bool:
    """Check if a task was rejected.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'rejected'.
    """
    return task.status.state == TaskState.rejected


def is_input_required(task: Task) -> bool:
    """Check if a task is waiting for user input.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'input_required'.
    """
    return task.status.state == TaskState.input_required


def is_auth_required(task: Task) -> bool:
    """Check if a task is waiting for authentication.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'auth_required'.
    """
    return task.status.state == TaskState.auth_required


def is_working(task: Task) -> bool:
    """Check if a task is currently being processed.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'working'.
    """
    return task.status.state == TaskState.working


def is_submitted(task: Task) -> bool:
    """Check if a task has been submitted but not yet started.

    Args:
        task: The Task to check.

    Returns:
        True if the task state is 'submitted'.
    """
    return task.status.state == TaskState.submitted


def get_task_state(task: Task) -> TaskState:
    """Get the current state of a task.

    Args:
        task: The Task to get the state from.

    Returns:
        The TaskState enum value.
    """
    return task.status.state


def create_status_update(
    task_id: str,
    context_id: str,
    state: TaskState,
    message: Message | None = None,
    final: bool = False,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    """Create a task status update event.

    Args:
        task_id: The ID of the task being updated.
        context_id: The context ID for the task.
        state: The new TaskState.
        message: Optional message to include with the update.
        final: Whether this is the final update for the task.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskStatusUpdateEvent ready to be enqueued.

    Example:
        >>> event = create_status_update(
        ...     task_id="task-123",
        ...     context_id="ctx-456",
        ...     state=TaskState.working,
        ...     message=create_text_message("Processing request..."),
        ... )
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    return TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=TaskStatus(
            state=state,
            message=message,
            timestamp=timestamp,
        ),
        final=final,
        metadata=metadata,
    )


def create_artifact_update(
    task_id: str,
    context_id: str,
    artifact: Artifact,
    append: bool = False,
    last_chunk: bool = True,
    metadata: dict[str, Any] | None = None,
) -> TaskArtifactUpdateEvent:
    """Create a task artifact update event.

    Args:
        task_id: The ID of the task being updated.
        context_id: The context ID for the task.
        artifact: The Artifact to add or update.
        append: Whether to append to an existing artifact.
        last_chunk: Whether this is the last chunk of the artifact.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskArtifactUpdateEvent ready to be enqueued.

    Example:
        >>> from src.common.a2a.artifacts import create_json_artifact
        >>> artifact = create_json_artifact({"result": "success"})
        >>> event = create_artifact_update(
        ...     task_id="task-123",
        ...     context_id="ctx-456",
        ...     artifact=artifact,
        ... )
    """
    return TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=append,
        last_chunk=last_chunk,
        metadata=metadata,
    )


def create_completed_status(
    task_id: str,
    context_id: str,
    message: Message | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    """Create a completion status update event.

    Convenience function for marking a task as completed.

    Args:
        task_id: The ID of the task.
        context_id: The context ID for the task.
        message: Optional completion message.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskStatusUpdateEvent with state=completed and final=True.
    """
    return create_status_update(
        task_id=task_id,
        context_id=context_id,
        state=TaskState.completed,
        message=message,
        final=True,
        metadata=metadata,
    )


def create_failed_status(
    task_id: str,
    context_id: str,
    message: Message | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    """Create a failure status update event.

    Convenience function for marking a task as failed.

    Args:
        task_id: The ID of the task.
        context_id: The context ID for the task.
        message: Optional failure message describing the error.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskStatusUpdateEvent with state=failed and final=True.
    """
    return create_status_update(
        task_id=task_id,
        context_id=context_id,
        state=TaskState.failed,
        message=message,
        final=True,
        metadata=metadata,
    )


def create_working_status(
    task_id: str,
    context_id: str,
    message: Message | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    """Create a working status update event.

    Convenience function for indicating a task is being processed.

    Args:
        task_id: The ID of the task.
        context_id: The context ID for the task.
        message: Optional status message.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskStatusUpdateEvent with state=working.
    """
    return create_status_update(
        task_id=task_id,
        context_id=context_id,
        state=TaskState.working,
        message=message,
        final=False,
        metadata=metadata,
    )


def create_input_required_status(
    task_id: str,
    context_id: str,
    message: Message | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskStatusUpdateEvent:
    """Create an input-required status update event.

    Convenience function for indicating a task needs user input.

    Args:
        task_id: The ID of the task.
        context_id: The context ID for the task.
        message: Optional message describing what input is needed.
        metadata: Optional metadata dictionary.

    Returns:
        A TaskStatusUpdateEvent with state=input_required.
    """
    return create_status_update(
        task_id=task_id,
        context_id=context_id,
        state=TaskState.input_required,
        message=message,
        final=False,
        metadata=metadata,
    )
