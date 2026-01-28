"""Tests for messages.py."""

import base64

import pytest
from a2a.types import DataPart, Message, Role, TextPart

from src.common.a2a.messages import (
    create_data_message,
    create_multipart_message,
    create_text_message,
    data_part,
    file_part,
    get_all_data_parts,
    get_all_text_parts,
    get_data_content,
    get_text_content,
    new_agent_text_message,
    new_user_text_message,
    text_part,
)


def unwrap_part(part):
    """Unwrap a Part wrapper to get the underlying TextPart/DataPart.

    The A2A SDK wraps parts in a Part(root=...) container.
    """
    if hasattr(part, "root"):
        return part.root
    return part


class TestTextPart:
    """Tests for the text_part function."""

    def test_create_text_part(self):
        """Test creating a TextPart."""
        part = text_part("Hello, world!")
        assert isinstance(part, TextPart)
        assert part.text == "Hello, world!"
        assert part.metadata is None

    def test_create_text_part_with_metadata(self):
        """Test creating a TextPart with metadata."""
        part = text_part("Hello", metadata={"key": "value"})
        assert part.text == "Hello"
        assert part.metadata == {"key": "value"}


class TestDataPart:
    """Tests for the data_part function."""

    def test_create_data_part(self):
        """Test creating a DataPart."""
        part = data_part({"status": "ok"})
        assert isinstance(part, DataPart)
        assert part.data == {"status": "ok"}
        assert part.metadata is None

    def test_create_data_part_with_metadata(self):
        """Test creating a DataPart with metadata."""
        part = data_part({"a": 1}, metadata={"source": "test"})
        assert part.data == {"a": 1}
        assert part.metadata == {"source": "test"}


class TestFilePart:
    """Tests for the file_part function."""

    def test_create_file_part(self):
        """Test creating a FilePart."""
        content = b"file content"
        part = file_part(content, "test.txt", "text/plain")

        assert part.file.name == "test.txt"
        assert part.file.mime_type == "text/plain"
        # Verify content is base64 encoded
        decoded = base64.standard_b64decode(part.file.bytes)
        assert decoded == content

    def test_create_file_part_binary(self):
        """Test creating a FilePart with binary content."""
        content = bytes([0, 1, 2, 255, 254, 253])
        part = file_part(content, "binary.dat", "application/octet-stream")

        decoded = base64.standard_b64decode(part.file.bytes)
        assert decoded == content


class TestCreateTextMessage:
    """Tests for the create_text_message function."""

    def test_create_text_message_minimal(self):
        """Test creating a text message with minimal arguments."""
        msg = create_text_message("Hello!")
        assert isinstance(msg, Message)
        assert msg.role == Role.user
        assert len(msg.parts) == 1
        inner = unwrap_part(msg.parts[0])
        assert isinstance(inner, TextPart)
        assert inner.text == "Hello!"
        assert msg.message_id is not None

    def test_create_text_message_full(self):
        """Test creating a text message with all arguments."""
        msg = create_text_message(
            text="Hello!",
            role=Role.agent,
            message_id="msg-123",
            task_id="task-456",
            context_id="ctx-789",
            metadata={"source": "test"},
        )
        assert msg.role == Role.agent
        assert msg.message_id == "msg-123"
        assert msg.task_id == "task-456"
        assert msg.context_id == "ctx-789"
        assert msg.metadata == {"source": "test"}


class TestCreateDataMessage:
    """Tests for the create_data_message function."""

    def test_create_data_message_minimal(self):
        """Test creating a data message with minimal arguments."""
        msg = create_data_message({"key": "value"})
        assert isinstance(msg, Message)
        assert msg.role == Role.user
        assert len(msg.parts) == 1
        inner = unwrap_part(msg.parts[0])
        assert isinstance(inner, DataPart)
        assert inner.data == {"key": "value"}

    def test_create_data_message_as_agent(self):
        """Test creating a data message from agent."""
        msg = create_data_message({"result": 42}, role=Role.agent)
        assert msg.role == Role.agent
        inner = unwrap_part(msg.parts[0])
        assert inner.data == {"result": 42}


class TestCreateMultipartMessage:
    """Tests for the create_multipart_message function."""

    def test_create_multipart_message(self):
        """Test creating a message with multiple parts."""
        parts = [
            text_part("Here is the data:"),
            data_part({"value": 42}),
        ]
        msg = create_multipart_message(parts)

        assert len(msg.parts) == 2
        inner0 = unwrap_part(msg.parts[0])
        inner1 = unwrap_part(msg.parts[1])
        assert isinstance(inner0, TextPart)
        assert isinstance(inner1, DataPart)


class TestGetTextContent:
    """Tests for the get_text_content function."""

    def test_get_text_content_single_part(self):
        """Test extracting text from a single-part message."""
        msg = create_text_message("Hello!")
        assert get_text_content(msg) == "Hello!"

    def test_get_text_content_multiple_parts(self):
        """Test extracting text from a multi-part message."""
        msg = create_multipart_message([
            text_part("Hello "),
            data_part({}),
            text_part("World!"),
        ])
        assert get_text_content(msg) == "Hello World!"

    def test_get_text_content_no_text(self):
        """Test extracting text when there's no TextPart."""
        msg = create_data_message({"key": "value"})
        assert get_text_content(msg) == ""


class TestGetDataContent:
    """Tests for the get_data_content function."""

    def test_get_data_content_single_part(self):
        """Test extracting data from a single-part message."""
        msg = create_data_message({"key": "value"})
        assert get_data_content(msg) == {"key": "value"}

    def test_get_data_content_multiple_parts(self):
        """Test extracting data from a multi-part message (returns first)."""
        msg = create_multipart_message([
            data_part({"first": 1}),
            data_part({"second": 2}),
        ])
        assert get_data_content(msg) == {"first": 1}

    def test_get_data_content_no_data(self):
        """Test extracting data when there's no DataPart."""
        msg = create_text_message("Hello!")
        assert get_data_content(msg) is None


class TestGetAllTextParts:
    """Tests for the get_all_text_parts function."""

    def test_get_all_text_parts(self):
        """Test extracting all text parts."""
        msg = create_multipart_message([
            text_part("Hello"),
            data_part({}),
            text_part("World"),
        ])
        texts = get_all_text_parts(msg)
        assert texts == ["Hello", "World"]

    def test_get_all_text_parts_empty(self):
        """Test extracting text parts when there are none."""
        msg = create_data_message({"key": "value"})
        assert get_all_text_parts(msg) == []


class TestGetAllDataParts:
    """Tests for the get_all_data_parts function."""

    def test_get_all_data_parts(self):
        """Test extracting all data parts."""
        msg = create_multipart_message([
            data_part({"a": 1}),
            text_part("text"),
            data_part({"b": 2}),
        ])
        data_list = get_all_data_parts(msg)
        assert data_list == [{"a": 1}, {"b": 2}]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_new_agent_text_message(self):
        """Test creating an agent text message."""
        msg = new_agent_text_message("I can help you!")
        assert msg.role == Role.agent
        assert get_text_content(msg) == "I can help you!"

    def test_new_agent_text_message_with_task_id(self):
        """Test creating an agent text message with task ID."""
        msg = new_agent_text_message("Response", task_id="task-123")
        assert msg.role == Role.agent
        assert msg.task_id == "task-123"

    def test_new_user_text_message(self):
        """Test creating a user text message."""
        msg = new_user_text_message("Help me please")
        assert msg.role == Role.user
        assert get_text_content(msg) == "Help me please"

    def test_new_user_text_message_with_task_id(self):
        """Test creating a user text message with task ID."""
        msg = new_user_text_message("Question", task_id="task-456")
        assert msg.role == Role.user
        assert msg.task_id == "task-456"
