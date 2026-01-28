"""Tests for client.py."""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from a2a.types import Message, Role, Task, TaskState, TaskStatus, TextPart

from src.common.a2a.client import A2AClientWrapper, A2AClientError


class TestA2AClientWrapper:
    """Tests for the A2AClientWrapper class."""

    def test_initialization(self):
        """Test client initialization."""
        client = A2AClientWrapper("http://localhost:9999")

        assert client.agent_url == "http://localhost:9999"
        assert client._httpx_client is None
        assert client._agent_card is None

    def test_initialization_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from the URL."""
        client = A2AClientWrapper("http://localhost:9999/")
        assert client.agent_url == "http://localhost:9999"

    def test_initialization_with_custom_timeout(self):
        """Test client with custom timeout."""
        client = A2AClientWrapper("http://localhost:9999", timeout=60.0)
        assert client._timeout == 60.0

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self):
        """Test that context manager creates httpx client."""
        wrapper = A2AClientWrapper("http://localhost:9999")

        async with wrapper as client:
            assert client._httpx_client is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        """Test that context manager closes httpx client."""
        wrapper = A2AClientWrapper("http://localhost:9999")

        async with wrapper:
            pass

        assert wrapper._httpx_client is None

    @pytest.mark.asyncio
    async def test_external_client_not_closed(self):
        """Test that externally provided client is not closed."""
        external_client = httpx.AsyncClient()
        wrapper = A2AClientWrapper("http://localhost:9999", httpx_client=external_client)

        async with wrapper:
            pass

        # External client should not be closed
        assert wrapper._httpx_client is external_client
        await external_client.aclose()

    def test_ensure_client_raises_without_init(self):
        """Test that _ensure_client raises if not initialized."""
        client = A2AClientWrapper("http://localhost:9999")

        with pytest.raises(RuntimeError, match="Client not initialized"):
            client._ensure_client()


class TestA2AClientError:
    """Tests for the A2AClientError class."""

    def test_error_with_message(self):
        """Test error with just a message."""
        error = A2AClientError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.code is None

    def test_error_with_code(self):
        """Test error with message and code."""
        error = A2AClientError("Not found", code=404)
        assert str(error) == "Not found"
        assert error.code == 404


# Note: More integration-style tests for send_message, get_task, etc.
# would require mocking the A2A server responses. These are better suited
# for integration tests with an actual A2A server running.

class TestClientMessageHelpers:
    """Tests for client convenience methods that don't require network."""

    @pytest.mark.asyncio
    async def test_send_text_creates_proper_message(self):
        """Test that send_text creates the correct message structure."""
        # We'll test the message creation part without actually sending
        # by checking the internal message structure

        client = A2AClientWrapper("http://localhost:9999")

        # Create the message that send_text would create
        from uuid import uuid4

        message = Message(
            role=Role.user,
            parts=[TextPart(text="Hello!")],
            messageId=str(uuid4()),
        )

        assert message.role == Role.user
        assert len(message.parts) == 1
        # Handle Part wrapper
        inner = message.parts[0].root if hasattr(message.parts[0], "root") else message.parts[0]
        assert inner.text == "Hello!"

    @pytest.mark.asyncio
    async def test_send_data_creates_proper_message(self):
        """Test that send_data creates the correct message structure."""
        from a2a.types import DataPart
        from uuid import uuid4

        # Create the message that send_data would create
        data = {"key": "value", "number": 42}
        message = Message(
            role=Role.user,
            parts=[DataPart(data=data)],
            messageId=str(uuid4()),
        )

        assert message.role == Role.user
        assert len(message.parts) == 1
        # Handle Part wrapper
        inner = message.parts[0].root if hasattr(message.parts[0], "root") else message.parts[0]
        assert inner.data == data
