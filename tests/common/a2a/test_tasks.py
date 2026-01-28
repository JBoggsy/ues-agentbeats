"""Tests for tasks.py."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from a2a.types import Task, TaskState, TaskStatus

from src.common.a2a.tasks import (
    create_artifact_update,
    create_completed_status,
    create_failed_status,
    create_input_required_status,
    create_status_update,
    create_working_status,
    get_task_state,
    is_auth_required,
    is_canceled,
    is_completed,
    is_failed,
    is_input_required,
    is_rejected,
    is_submitted,
    is_terminal_state,
    is_working,
)


def make_task(state: TaskState) -> Task:
    """Create a mock Task with the given state."""
    return Task(
        id="task-123",
        context_id="ctx-456",
        status=TaskStatus(
            state=state,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


class TestTaskStateChecks:
    """Tests for task state check functions."""

    def test_is_terminal_state_completed(self):
        """Test that completed is a terminal state."""
        task = make_task(TaskState.completed)
        assert is_terminal_state(task) is True

    def test_is_terminal_state_failed(self):
        """Test that failed is a terminal state."""
        task = make_task(TaskState.failed)
        assert is_terminal_state(task) is True

    def test_is_terminal_state_canceled(self):
        """Test that canceled is a terminal state."""
        task = make_task(TaskState.canceled)
        assert is_terminal_state(task) is True

    def test_is_terminal_state_rejected(self):
        """Test that rejected is a terminal state."""
        task = make_task(TaskState.rejected)
        assert is_terminal_state(task) is True

    def test_is_terminal_state_working(self):
        """Test that working is not a terminal state."""
        task = make_task(TaskState.working)
        assert is_terminal_state(task) is False

    def test_is_terminal_state_submitted(self):
        """Test that submitted is not a terminal state."""
        task = make_task(TaskState.submitted)
        assert is_terminal_state(task) is False

    def test_is_completed(self):
        """Test is_completed function."""
        assert is_completed(make_task(TaskState.completed)) is True
        assert is_completed(make_task(TaskState.working)) is False

    def test_is_failed(self):
        """Test is_failed function."""
        assert is_failed(make_task(TaskState.failed)) is True
        assert is_failed(make_task(TaskState.completed)) is False

    def test_is_canceled(self):
        """Test is_canceled function."""
        assert is_canceled(make_task(TaskState.canceled)) is True
        assert is_canceled(make_task(TaskState.working)) is False

    def test_is_rejected(self):
        """Test is_rejected function."""
        assert is_rejected(make_task(TaskState.rejected)) is True
        assert is_rejected(make_task(TaskState.working)) is False

    def test_is_input_required(self):
        """Test is_input_required function."""
        assert is_input_required(make_task(TaskState.input_required)) is True
        assert is_input_required(make_task(TaskState.working)) is False

    def test_is_auth_required(self):
        """Test is_auth_required function."""
        assert is_auth_required(make_task(TaskState.auth_required)) is True
        assert is_auth_required(make_task(TaskState.working)) is False

    def test_is_working(self):
        """Test is_working function."""
        assert is_working(make_task(TaskState.working)) is True
        assert is_working(make_task(TaskState.completed)) is False

    def test_is_submitted(self):
        """Test is_submitted function."""
        assert is_submitted(make_task(TaskState.submitted)) is True
        assert is_submitted(make_task(TaskState.working)) is False

    def test_get_task_state(self):
        """Test get_task_state function."""
        task = make_task(TaskState.working)
        assert get_task_state(task) == TaskState.working


class TestCreateStatusUpdate:
    """Tests for status update creation functions."""

    def test_create_status_update_basic(self):
        """Test creating a basic status update."""
        event = create_status_update(
            task_id="task-123",
            context_id="ctx-456",
            state=TaskState.working,
        )

        assert event.task_id == "task-123"
        assert event.context_id == "ctx-456"
        assert event.status.state == TaskState.working
        assert event.status.message is None
        assert event.final is False
        assert event.status.timestamp is not None

    def test_create_status_update_with_message(self):
        """Test creating a status update with a message."""
        from src.common.a2a.messages import create_text_message

        message = create_text_message("Processing...")
        event = create_status_update(
            task_id="task-123",
            context_id="ctx-456",
            state=TaskState.working,
            message=message,
        )

        assert event.status.message is not None

    def test_create_status_update_final(self):
        """Test creating a final status update."""
        event = create_status_update(
            task_id="task-123",
            context_id="ctx-456",
            state=TaskState.completed,
            final=True,
        )

        assert event.final is True

    def test_create_status_update_with_metadata(self):
        """Test creating a status update with metadata."""
        event = create_status_update(
            task_id="task-123",
            context_id="ctx-456",
            state=TaskState.working,
            metadata={"progress": 50},
        )

        assert event.metadata == {"progress": 50}

    def test_create_completed_status(self):
        """Test the convenience function for completed status."""
        event = create_completed_status(
            task_id="task-123",
            context_id="ctx-456",
        )

        assert event.status.state == TaskState.completed
        assert event.final is True

    def test_create_failed_status(self):
        """Test the convenience function for failed status."""
        event = create_failed_status(
            task_id="task-123",
            context_id="ctx-456",
        )

        assert event.status.state == TaskState.failed
        assert event.final is True

    def test_create_working_status(self):
        """Test the convenience function for working status."""
        event = create_working_status(
            task_id="task-123",
            context_id="ctx-456",
        )

        assert event.status.state == TaskState.working
        assert event.final is False

    def test_create_input_required_status(self):
        """Test the convenience function for input_required status."""
        event = create_input_required_status(
            task_id="task-123",
            context_id="ctx-456",
        )

        assert event.status.state == TaskState.input_required
        assert event.final is False


class TestCreateArtifactUpdate:
    """Tests for artifact update creation."""

    def test_create_artifact_update_basic(self):
        """Test creating a basic artifact update."""
        from src.common.a2a.artifacts import create_json_artifact

        artifact = create_json_artifact({"result": "success"})
        event = create_artifact_update(
            task_id="task-123",
            context_id="ctx-456",
            artifact=artifact,
        )

        assert event.task_id == "task-123"
        assert event.context_id == "ctx-456"
        assert event.artifact is artifact
        assert event.append is False
        assert event.last_chunk is True

    def test_create_artifact_update_append(self):
        """Test creating an append artifact update."""
        from src.common.a2a.artifacts import create_text_artifact

        artifact = create_text_artifact("More content")
        event = create_artifact_update(
            task_id="task-123",
            context_id="ctx-456",
            artifact=artifact,
            append=True,
            last_chunk=False,
        )

        assert event.append is True
        assert event.last_chunk is False

    def test_create_artifact_update_with_metadata(self):
        """Test creating an artifact update with metadata."""
        from src.common.a2a.artifacts import create_json_artifact

        artifact = create_json_artifact({"data": 42})
        event = create_artifact_update(
            task_id="task-123",
            context_id="ctx-456",
            artifact=artifact,
            metadata={"chunk_index": 0},
        )

        assert event.metadata == {"chunk_index": 0}
