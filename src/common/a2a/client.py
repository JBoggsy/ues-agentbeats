"""A2A Client utilities.

This module provides a wrapper around the A2A SDK client for simplified
communication with A2A agents.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskIdParams,
    TaskQueryParams,
    TextPart,
)

logger = logging.getLogger(__name__)


class A2AClientWrapper:
    """Client for communicating with A2A agents.

    This class wraps the A2A SDK's client with a simplified interface
    for common operations.

    Example:
        >>> async with A2AClientWrapper("http://localhost:9999") as client:
        ...     card = await client.get_agent_card()
        ...     print(f"Connected to: {card.name}")
        ...
        ...     response = await client.send_text("Hello, agent!")
        ...     print(response)
    """

    def __init__(
        self,
        agent_url: str,
        httpx_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            agent_url: Base URL of the A2A agent (e.g., "http://localhost:9999").
            httpx_client: Optional httpx AsyncClient to use. If not provided,
                one will be created.
            timeout: Request timeout in seconds (default: 30.0).
        """
        self.agent_url = agent_url.rstrip("/")
        self._timeout = timeout
        self._external_client = httpx_client is not None
        self._httpx_client = httpx_client
        self._agent_card: AgentCard | None = None
        self._a2a_client: A2AClient | None = None

    async def __aenter__(self) -> A2AClientWrapper:
        """Async context manager entry."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if not self._external_client and self._httpx_client is not None:
            await self._httpx_client.aclose()
            self._httpx_client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure httpx client is available.

        Returns:
            The httpx AsyncClient.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._httpx_client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager or "
                "call __aenter__ first."
            )
        return self._httpx_client

    async def get_agent_card(self, force_refresh: bool = False) -> AgentCard:
        """Fetch the agent card from the server.

        Args:
            force_refresh: If True, fetches the card even if already cached.

        Returns:
            The AgentCard describing the remote agent.
        """
        if self._agent_card is not None and not force_refresh:
            return self._agent_card

        httpx_client = self._ensure_client()
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=self.agent_url,
        )
        self._agent_card = await resolver.get_agent_card()
        return self._agent_card

    async def _get_a2a_client(self) -> A2AClient:
        """Get or create the A2A client.

        Returns:
            The initialized A2AClient.
        """
        if self._a2a_client is None:
            httpx_client = self._ensure_client()
            agent_card = await self.get_agent_card()
            self._a2a_client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card,
            )
        return self._a2a_client

    async def send_message(
        self,
        message: Message,
        blocking: bool = False,
        history_length: int | None = None,
    ) -> Task | Message:
        """Send a message to the agent.

        Args:
            message: The Message to send.
            blocking: If True, waits for task completion before returning.
            history_length: Optional number of history messages to include.

        Returns:
            A Task or Message response from the agent.
        """
        client = await self._get_a2a_client()

        config: dict[str, Any] = {}
        if blocking:
            config["blocking"] = True
        if history_length is not None:
            config["history_length"] = history_length

        params_dict: dict[str, Any] = {"message": message}
        if config:
            params_dict["configuration"] = config

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**params_dict),
        )

        response = await client.send_message(request)

        # Handle the response - extract result from success response
        if hasattr(response, "result"):
            return response.result
        if hasattr(response, "error"):
            raise A2AClientError(
                f"Error from agent: {response.error.message}",
                code=response.error.code,
            )
        return response

    async def send_text(
        self,
        text: str,
        role: Role = Role.user,
        task_id: str | None = None,
        context_id: str | None = None,
        blocking: bool = False,
    ) -> Task | Message:
        """Send a text message to the agent.

        Convenience method for sending simple text messages.

        Args:
            text: The text content to send.
            role: Message role (default: Role.user).
            task_id: Optional task ID for continuing a conversation.
            context_id: Optional context ID for the message.
            blocking: If True, waits for task completion before returning.

        Returns:
            A Task or Message response from the agent.
        """
        message = Message(
            role=role,
            parts=[TextPart(text=text)],
            message_id=str(uuid4()),
            task_id=task_id,
            context_id=context_id,
        )
        return await self.send_message(message, blocking=blocking)

    async def send_data(
        self,
        data: dict[str, Any],
        role: Role = Role.user,
        task_id: str | None = None,
        context_id: str | None = None,
        blocking: bool = False,
    ) -> Task | Message:
        """Send a data message to the agent.

        Convenience method for sending structured data messages.

        Args:
            data: The data dictionary to send.
            role: Message role (default: Role.user).
            task_id: Optional task ID for continuing a conversation.
            context_id: Optional context ID for the message.
            blocking: If True, waits for task completion before returning.

        Returns:
            A Task or Message response from the agent.
        """
        from a2a.types import DataPart

        message = Message(
            role=role,
            parts=[DataPart(data=data)],
            message_id=str(uuid4()),
            task_id=task_id,
            context_id=context_id,
        )
        return await self.send_message(message, blocking=blocking)

    async def send_streaming_message(
        self,
        message: Message,
        history_length: int | None = None,
    ) -> AsyncIterator:
        """Send a message and receive streaming responses.

        Args:
            message: The Message to send.
            history_length: Optional number of history messages to include.

        Yields:
            Streaming response events from the agent.
        """
        client = await self._get_a2a_client()

        params_dict: dict[str, Any] = {"message": message}
        if history_length is not None:
            params_dict["configuration"] = {"history_length": history_length}

        request = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**params_dict),
        )

        async for event in client.send_message_streaming(request):
            yield event

    async def send_text_streaming(
        self,
        text: str,
        role: Role = Role.user,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator:
        """Send a text message and receive streaming responses.

        Convenience method for streaming text messages.

        Args:
            text: The text content to send.
            role: Message role (default: Role.user).
            task_id: Optional task ID for continuing a conversation.
            context_id: Optional context ID for the message.

        Yields:
            Streaming response events from the agent.
        """
        message = Message(
            role=role,
            parts=[TextPart(text=text)],
            message_id=str(uuid4()),
            task_id=task_id,
            context_id=context_id,
        )
        async for event in self.send_streaming_message(message):
            yield event

    async def get_task(
        self,
        task_id: str,
        history_length: int | None = None,
    ) -> Task:
        """Get the current state of a task.

        Args:
            task_id: The ID of the task to retrieve.
            history_length: Optional number of history messages to include.

        Returns:
            The Task object with current state.
        """
        client = await self._get_a2a_client()

        params_dict: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params_dict["history_length"] = history_length

        request = GetTaskRequest(
            id=str(uuid4()),
            params=TaskQueryParams(**params_dict),
        )

        response = await client.get_task(request)

        if hasattr(response, "result"):
            return response.result
        if hasattr(response, "error"):
            raise A2AClientError(
                f"Error getting task: {response.error.message}",
                code=response.error.code,
            )
        return response

    async def cancel_task(self, task_id: str) -> Task:
        """Request cancellation of a task.

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            The Task object with updated state.
        """
        client = await self._get_a2a_client()

        request = CancelTaskRequest(
            id=str(uuid4()),
            params=TaskIdParams(id=task_id),
        )

        response = await client.cancel_task(request)

        if hasattr(response, "result"):
            return response.result
        if hasattr(response, "error"):
            raise A2AClientError(
                f"Error canceling task: {response.error.message}",
                code=response.error.code,
            )
        return response


class A2AClientError(Exception):
    """Exception raised for A2A client errors.

    Attributes:
        message: Error message.
        code: Optional error code from the A2A protocol.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error message.
            code: Optional error code.
        """
        super().__init__(message)
        self.code = code
