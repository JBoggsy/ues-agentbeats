"""Artifact creation and parsing utilities for A2A protocol.

This module provides helper functions for creating and working with
A2A artifacts, which are used to represent task outputs.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from a2a.types import Artifact, DataPart, Part, TextPart


def create_artifact(
    artifact_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    parts: list[Part] | None = None,
    text: str | None = None,
    data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create an Artifact with flexible content options.

    You can provide content in multiple ways:
    - As a list of Part objects via `parts`
    - As text via `text` (creates a TextPart)
    - As data via `data` (creates a DataPart)

    Args:
        artifact_id: Optional unique ID. Generated if not provided.
        name: Optional human-readable name for the artifact.
        description: Optional description of the artifact contents.
        parts: Optional list of Part objects.
        text: Optional text content (creates a TextPart).
        data: Optional data dictionary (creates a DataPart).
        metadata: Optional metadata dictionary.

    Returns:
        An Artifact object.

    Raises:
        ValueError: If no content is provided (no parts, text, or data).

    Example:
        >>> # Create with text
        >>> artifact = create_artifact(text="Hello, world!", name="greeting")
        >>>
        >>> # Create with data
        >>> artifact = create_artifact(data={"status": "ok"}, name="result")
        >>>
        >>> # Create with parts
        >>> artifact = create_artifact(
        ...     parts=[TextPart(text="Part 1"), DataPart(data={})],
        ...     name="multipart",
        ... )
    """
    artifact_parts: list[Part] = []

    if parts is not None:
        artifact_parts.extend(parts)

    if text is not None:
        artifact_parts.append(TextPart(text=text))

    if data is not None:
        artifact_parts.append(DataPart(data=data))

    if not artifact_parts:
        raise ValueError("Artifact must have at least one part (provide parts, text, or data)")

    return Artifact(
        artifactId=artifact_id or str(uuid4()),
        name=name,
        description=description,
        parts=artifact_parts,
        metadata=metadata,
    )


def create_json_artifact(
    data: dict[str, Any],
    name: str = "result",
    description: str | None = None,
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create an Artifact containing JSON data.

    This is a convenience function for the common case of returning
    structured data as an artifact.

    Args:
        data: The data dictionary to include.
        name: Name for the artifact (default: "result").
        description: Optional description of the artifact.
        artifact_id: Optional unique ID. Generated if not provided.
        metadata: Optional metadata dictionary.

    Returns:
        An Artifact with a single DataPart containing the data.

    Example:
        >>> artifact = create_json_artifact(
        ...     {"score": 95, "passed": True},
        ...     name="evaluation_result",
        ...     description="Assessment evaluation results",
        ... )
    """
    return Artifact(
        artifactId=artifact_id or str(uuid4()),
        name=name,
        description=description,
        parts=[DataPart(data=data)],
        metadata=metadata,
    )


def create_text_artifact(
    text: str,
    name: str = "output",
    description: str | None = None,
    artifact_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    """Create an Artifact containing text content.

    This is a convenience function for returning text content as an artifact.

    Args:
        text: The text content to include.
        name: Name for the artifact (default: "output").
        description: Optional description of the artifact.
        artifact_id: Optional unique ID. Generated if not provided.
        metadata: Optional metadata dictionary.

    Returns:
        An Artifact with a single TextPart containing the text.

    Example:
        >>> artifact = create_text_artifact(
        ...     "Task completed successfully.",
        ...     name="completion_message",
        ... )
    """
    return Artifact(
        artifactId=artifact_id or str(uuid4()),
        name=name,
        description=description,
        parts=[TextPart(text=text)],
        metadata=metadata,
    )


def extract_json_from_artifact(artifact: Artifact) -> dict[str, Any] | None:
    """Extract JSON data from an artifact.

    Returns the first DataPart's data found in the artifact.

    Args:
        artifact: The Artifact to extract data from.

    Returns:
        The data dictionary from the first DataPart, or None if not found.

    Example:
        >>> artifact = create_json_artifact({"key": "value"})
        >>> extract_json_from_artifact(artifact)
        {'key': 'value'}
    """
    for part in artifact.parts:
        if isinstance(part, DataPart):
            return part.data
        elif hasattr(part, "root") and isinstance(part.root, DataPart):
            return part.root.data
    return None


def extract_text_from_artifact(artifact: Artifact) -> str | None:
    """Extract text content from an artifact.

    Returns the first TextPart's text found in the artifact.

    Args:
        artifact: The Artifact to extract text from.

    Returns:
        The text from the first TextPart, or None if not found.

    Example:
        >>> artifact = create_text_artifact("Hello!")
        >>> extract_text_from_artifact(artifact)
        'Hello!'
    """
    for part in artifact.parts:
        if isinstance(part, TextPart):
            return part.text
        elif hasattr(part, "root") and isinstance(part.root, TextPart):
            return part.root.text
    return None


def extract_all_json_from_artifact(artifact: Artifact) -> list[dict[str, Any]]:
    """Extract all JSON data from an artifact.

    Returns all DataParts' data found in the artifact.

    Args:
        artifact: The Artifact to extract data from.

    Returns:
        A list of data dictionaries from all DataParts.

    Example:
        >>> artifact = create_artifact(parts=[
        ...     DataPart(data={"a": 1}),
        ...     DataPart(data={"b": 2}),
        ... ])
        >>> extract_all_json_from_artifact(artifact)
        [{'a': 1}, {'b': 2}]
    """
    data_list = []
    for part in artifact.parts:
        if isinstance(part, DataPart):
            data_list.append(part.data)
        elif hasattr(part, "root") and isinstance(part.root, DataPart):
            data_list.append(part.root.data)
    return data_list


def extract_all_text_from_artifact(artifact: Artifact) -> list[str]:
    """Extract all text content from an artifact.

    Returns all TextParts' text found in the artifact.

    Args:
        artifact: The Artifact to extract text from.

    Returns:
        A list of text strings from all TextParts.

    Example:
        >>> artifact = create_artifact(parts=[
        ...     TextPart(text="Hello"),
        ...     TextPart(text="World"),
        ... ])
        >>> extract_all_text_from_artifact(artifact)
        ['Hello', 'World']
    """
    texts = []
    for part in artifact.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif hasattr(part, "root") and isinstance(part.root, TextPart):
            texts.append(part.root.text)
    return texts


def artifact_to_json_string(artifact: Artifact, indent: int | None = 2) -> str:
    """Serialize an artifact to a JSON string.

    This is useful for debugging or logging artifact contents.

    Args:
        artifact: The Artifact to serialize.
        indent: Indentation level for formatting (default: 2).

    Returns:
        A JSON string representation of the artifact.

    Example:
        >>> artifact = create_json_artifact({"key": "value"})
        >>> print(artifact_to_json_string(artifact))
        {
          "artifact_id": "...",
          ...
        }
    """
    return artifact.model_dump_json(indent=indent, exclude_none=True)


def artifact_from_json_string(json_string: str) -> Artifact:
    """Deserialize an artifact from a JSON string.

    Args:
        json_string: The JSON string to parse.

    Returns:
        An Artifact object.

    Example:
        >>> json_str = '{"artifact_id": "123", "parts": [{"kind": "text", "text": "Hi"}]}'
        >>> artifact = artifact_from_json_string(json_str)
    """
    data = json.loads(json_string)
    return Artifact.model_validate(data)
