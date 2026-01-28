"""Message creation and parsing utilities for A2A protocol.

This module provides helper functions for creating and parsing A2A messages,
including convenience functions for common message types.
"""

from __future__ import annotations

import base64
from typing import Any
from uuid import uuid4

from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TextPart,
)


def text_part(text: str, metadata: dict[str, Any] | None = None) -> TextPart:
    """Create a TextPart.

    Args:
        text: The text content.
        metadata: Optional metadata dictionary.

    Returns:
        A TextPart object.

    Example:
        >>> part = text_part("Hello, world!")
        >>> part.text
        'Hello, world!'
    """
    return TextPart(text=text, metadata=metadata)


def data_part(data: dict[str, Any], metadata: dict[str, Any] | None = None) -> DataPart:
    """Create a DataPart.

    Args:
        data: The data dictionary.
        metadata: Optional metadata dictionary.

    Returns:
        A DataPart object.

    Example:
        >>> part = data_part({"key": "value"})
        >>> part.data
        {'key': 'value'}
    """
    return DataPart(data=data, metadata=metadata)


def file_part(
    content: bytes,
    filename: str,
    media_type: str,
    metadata: dict[str, Any] | None = None,
) -> FilePart:
    """Create a FilePart with inline bytes.

    Args:
        content: The file content as bytes.
        filename: The filename.
        media_type: The MIME type (e.g., "application/pdf", "image/png").
        metadata: Optional metadata dictionary.

    Returns:
        A FilePart object with base64-encoded content.

    Example:
        >>> part = file_part(b"file content", "test.txt", "text/plain")
        >>> part.file.name
        'test.txt'
    """
    encoded = base64.standard_b64encode(content).decode("ascii")
    return FilePart(
        file=FileWithBytes(
            bytes=encoded,
            name=filename,
            mimeType=media_type,
        ),
        metadata=metadata,
    )


def create_text_message(
    text: str,
    role: Role = Role.user,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create a text message.

    Args:
        text: The text content of the message.
        role: The message role (default: Role.user).
        message_id: Optional unique message ID. Generated if not provided.
        task_id: Optional task ID for associating with a task.
        context_id: Optional context ID for the conversation.
        metadata: Optional metadata dictionary.

    Returns:
        A Message object with a single TextPart.

    Example:
        >>> msg = create_text_message("Hello!", role=Role.user)
        >>> msg.role
        <Role.user: 'user'>
        >>> msg.parts[0].text
        'Hello!'
    """
    return Message(
        role=role,
        parts=[text_part(text)],
        messageId=message_id or str(uuid4()),
        taskId=task_id,
        contextId=context_id,
        metadata=metadata,
    )


def create_data_message(
    data: dict[str, Any],
    role: Role = Role.user,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create a data message.

    Args:
        data: The data dictionary to include in the message.
        role: The message role (default: Role.user).
        message_id: Optional unique message ID. Generated if not provided.
        task_id: Optional task ID for associating with a task.
        context_id: Optional context ID for the conversation.
        metadata: Optional metadata dictionary.

    Returns:
        A Message object with a single DataPart.

    Example:
        >>> msg = create_data_message({"status": "ok"}, role=Role.agent)
        >>> msg.parts[0].data
        {'status': 'ok'}
    """
    return Message(
        role=role,
        parts=[data_part(data)],
        messageId=message_id or str(uuid4()),
        taskId=task_id,
        contextId=context_id,
        metadata=metadata,
    )


def create_multipart_message(
    parts: list[Part],
    role: Role = Role.user,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Create a message with multiple parts.

    Args:
        parts: List of Part objects (TextPart, DataPart, FilePart).
        role: The message role (default: Role.user).
        message_id: Optional unique message ID. Generated if not provided.
        task_id: Optional task ID for associating with a task.
        context_id: Optional context ID for the conversation.
        metadata: Optional metadata dictionary.

    Returns:
        A Message object with the specified parts.

    Example:
        >>> msg = create_multipart_message([
        ...     text_part("Here is the data:"),
        ...     data_part({"value": 42}),
        ... ])
        >>> len(msg.parts)
        2
    """
    return Message(
        role=role,
        parts=parts,
        messageId=message_id or str(uuid4()),
        taskId=task_id,
        contextId=context_id,
        metadata=metadata,
    )


def get_text_content(message: Message) -> str:
    """Extract text content from a message.

    Concatenates all TextPart contents in the message.

    Args:
        message: The Message to extract text from.

    Returns:
        The concatenated text content, or empty string if no text parts.

    Example:
        >>> msg = create_text_message("Hello!")
        >>> get_text_content(msg)
        'Hello!'
    """
    texts = []
    for part in message.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif hasattr(part, "root") and isinstance(part.root, TextPart):
            texts.append(part.root.text)
    return "".join(texts)


def get_data_content(message: Message) -> dict[str, Any] | None:
    """Extract data content from a message.

    Returns the first DataPart's data found in the message.

    Args:
        message: The Message to extract data from.

    Returns:
        The data dictionary from the first DataPart, or None if not found.

    Example:
        >>> msg = create_data_message({"key": "value"})
        >>> get_data_content(msg)
        {'key': 'value'}
    """
    for part in message.parts:
        if isinstance(part, DataPart):
            return part.data
        elif hasattr(part, "root") and isinstance(part.root, DataPart):
            return part.root.data
    return None


def get_all_text_parts(message: Message) -> list[str]:
    """Extract all text parts from a message as a list.

    Args:
        message: The Message to extract text from.

    Returns:
        A list of text strings from all TextParts in the message.

    Example:
        >>> msg = create_multipart_message([
        ...     text_part("Hello"),
        ...     data_part({}),
        ...     text_part("World"),
        ... ])
        >>> get_all_text_parts(msg)
        ['Hello', 'World']
    """
    texts = []
    for part in message.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif hasattr(part, "root") and isinstance(part.root, TextPart):
            texts.append(part.root.text)
    return texts


def get_all_data_parts(message: Message) -> list[dict[str, Any]]:
    """Extract all data parts from a message as a list.

    Args:
        message: The Message to extract data from.

    Returns:
        A list of data dictionaries from all DataParts in the message.

    Example:
        >>> msg = create_multipart_message([
        ...     data_part({"a": 1}),
        ...     text_part("text"),
        ...     data_part({"b": 2}),
        ... ])
        >>> get_all_data_parts(msg)
        [{'a': 1}, {'b': 2}]
    """
    data_list = []
    for part in message.parts:
        if isinstance(part, DataPart):
            data_list.append(part.data)
        elif hasattr(part, "root") and isinstance(part.root, DataPart):
            data_list.append(part.root.data)
    return data_list


def new_agent_text_message(text: str, task_id: str | None = None) -> Message:
    """Create an agent text message (convenience function).

    This is a shorthand for creating messages from the agent's perspective.

    Args:
        text: The text content.
        task_id: Optional task ID.

    Returns:
        A Message with role=agent containing the text.

    Example:
        >>> msg = new_agent_text_message("I can help you with that!")
        >>> msg.role
        <Role.agent: 'agent'>
    """
    return create_text_message(text, role=Role.agent, task_id=task_id)


def new_user_text_message(text: str, task_id: str | None = None) -> Message:
    """Create a user text message (convenience function).

    This is a shorthand for creating messages from the user's perspective.

    Args:
        text: The text content.
        task_id: Optional task ID.

    Returns:
        A Message with role=user containing the text.

    Example:
        >>> msg = new_user_text_message("Help me with my task")
        >>> msg.role
        <Role.user: 'user'>
    """
    return create_text_message(text, role=Role.user, task_id=task_id)
