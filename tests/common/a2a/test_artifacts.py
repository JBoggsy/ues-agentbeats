"""Tests for artifacts.py."""

import json

import pytest
from a2a.types import Artifact, DataPart, TextPart

from src.common.a2a.artifacts import (
    artifact_from_json_string,
    artifact_to_json_string,
    create_artifact,
    create_json_artifact,
    create_text_artifact,
    extract_all_json_from_artifact,
    extract_all_text_from_artifact,
    extract_json_from_artifact,
    extract_text_from_artifact,
)


def unwrap_part(part):
    """Unwrap a Part wrapper to get the underlying TextPart/DataPart.

    The A2A SDK wraps parts in a Part(root=...) container.
    """
    if hasattr(part, "root"):
        return part.root
    return part


class TestCreateArtifact:
    """Tests for the create_artifact function."""

    def test_create_artifact_with_text(self):
        """Test creating an artifact with text content."""
        artifact = create_artifact(
            text="Hello, world!",
            name="greeting",
        )

        assert artifact.name == "greeting"
        assert len(artifact.parts) == 1
        inner = unwrap_part(artifact.parts[0])
        assert isinstance(inner, TextPart)
        assert inner.text == "Hello, world!"
        assert artifact.artifact_id is not None

    def test_create_artifact_with_data(self):
        """Test creating an artifact with data content."""
        artifact = create_artifact(
            data={"status": "ok"},
            name="result",
        )

        assert artifact.name == "result"
        assert len(artifact.parts) == 1
        inner = unwrap_part(artifact.parts[0])
        assert isinstance(inner, DataPart)
        assert inner.data == {"status": "ok"}

    def test_create_artifact_with_parts(self):
        """Test creating an artifact with explicit parts."""
        parts = [
            TextPart(text="Part 1"),
            DataPart(data={"key": "value"}),
        ]
        artifact = create_artifact(parts=parts, name="multipart")

        assert len(artifact.parts) == 2
        assert artifact.name == "multipart"

    def test_create_artifact_with_all_content_types(self):
        """Test creating an artifact with text, data, and parts."""
        artifact = create_artifact(
            parts=[TextPart(text="Part 0")],
            text="Text content",
            data={"data": "content"},
            name="mixed",
        )

        # Should have 3 parts: 1 from parts, 1 from text, 1 from data
        assert len(artifact.parts) == 3

    def test_create_artifact_no_content_raises(self):
        """Test that creating an artifact without content raises."""
        with pytest.raises(ValueError, match="at least one part"):
            create_artifact(name="empty")

    def test_create_artifact_with_custom_id(self):
        """Test creating an artifact with a custom ID."""
        artifact = create_artifact(
            text="Content",
            artifact_id="custom-id-123",
        )
        assert artifact.artifact_id == "custom-id-123"

    def test_create_artifact_with_description(self):
        """Test creating an artifact with a description."""
        artifact = create_artifact(
            text="Content",
            description="A detailed description",
        )
        assert artifact.description == "A detailed description"

    def test_create_artifact_with_metadata(self):
        """Test creating an artifact with metadata."""
        artifact = create_artifact(
            text="Content",
            metadata={"key": "value"},
        )
        assert artifact.metadata == {"key": "value"}


class TestCreateJsonArtifact:
    """Tests for the create_json_artifact function."""

    def test_create_json_artifact_basic(self):
        """Test creating a basic JSON artifact."""
        artifact = create_json_artifact({"score": 95, "passed": True})

        assert artifact.name == "result"  # default
        assert len(artifact.parts) == 1
        inner = unwrap_part(artifact.parts[0])
        assert isinstance(inner, DataPart)
        assert inner.data == {"score": 95, "passed": True}

    def test_create_json_artifact_with_name(self):
        """Test creating a JSON artifact with a custom name."""
        artifact = create_json_artifact(
            {"data": 42},
            name="evaluation_result",
        )
        assert artifact.name == "evaluation_result"

    def test_create_json_artifact_with_description(self):
        """Test creating a JSON artifact with a description."""
        artifact = create_json_artifact(
            {"data": 42},
            description="Assessment results",
        )
        assert artifact.description == "Assessment results"


class TestCreateTextArtifact:
    """Tests for the create_text_artifact function."""

    def test_create_text_artifact_basic(self):
        """Test creating a basic text artifact."""
        artifact = create_text_artifact("Task completed successfully.")

        assert artifact.name == "output"  # default
        assert len(artifact.parts) == 1
        inner = unwrap_part(artifact.parts[0])
        assert isinstance(inner, TextPart)
        assert inner.text == "Task completed successfully."

    def test_create_text_artifact_with_name(self):
        """Test creating a text artifact with a custom name."""
        artifact = create_text_artifact(
            "Message content",
            name="completion_message",
        )
        assert artifact.name == "completion_message"


class TestExtractJsonFromArtifact:
    """Tests for the extract_json_from_artifact function."""

    def test_extract_json_from_artifact_single(self):
        """Test extracting JSON from an artifact with one DataPart."""
        artifact = create_json_artifact({"key": "value"})
        data = extract_json_from_artifact(artifact)
        assert data == {"key": "value"}

    def test_extract_json_from_artifact_multiple(self):
        """Test extracting JSON returns first DataPart."""
        artifact = create_artifact(parts=[
            DataPart(data={"first": 1}),
            DataPart(data={"second": 2}),
        ])
        data = extract_json_from_artifact(artifact)
        assert data == {"first": 1}

    def test_extract_json_from_artifact_no_data(self):
        """Test extracting JSON when there's no DataPart."""
        artifact = create_text_artifact("No data here")
        data = extract_json_from_artifact(artifact)
        assert data is None


class TestExtractTextFromArtifact:
    """Tests for the extract_text_from_artifact function."""

    def test_extract_text_from_artifact_single(self):
        """Test extracting text from an artifact with one TextPart."""
        artifact = create_text_artifact("Hello!")
        text = extract_text_from_artifact(artifact)
        assert text == "Hello!"

    def test_extract_text_from_artifact_multiple(self):
        """Test extracting text returns first TextPart."""
        artifact = create_artifact(parts=[
            TextPart(text="First"),
            TextPart(text="Second"),
        ])
        text = extract_text_from_artifact(artifact)
        assert text == "First"

    def test_extract_text_from_artifact_no_text(self):
        """Test extracting text when there's no TextPart."""
        artifact = create_json_artifact({"data": 42})
        text = extract_text_from_artifact(artifact)
        assert text is None


class TestExtractAllJsonFromArtifact:
    """Tests for the extract_all_json_from_artifact function."""

    def test_extract_all_json_from_artifact(self):
        """Test extracting all JSON from an artifact."""
        artifact = create_artifact(parts=[
            DataPart(data={"a": 1}),
            TextPart(text="text"),
            DataPart(data={"b": 2}),
        ])
        data_list = extract_all_json_from_artifact(artifact)
        assert data_list == [{"a": 1}, {"b": 2}]

    def test_extract_all_json_from_artifact_empty(self):
        """Test extracting JSON when there are no DataParts."""
        artifact = create_text_artifact("No data")
        data_list = extract_all_json_from_artifact(artifact)
        assert data_list == []


class TestExtractAllTextFromArtifact:
    """Tests for the extract_all_text_from_artifact function."""

    def test_extract_all_text_from_artifact(self):
        """Test extracting all text from an artifact."""
        artifact = create_artifact(parts=[
            TextPart(text="Hello"),
            DataPart(data={}),
            TextPart(text="World"),
        ])
        texts = extract_all_text_from_artifact(artifact)
        assert texts == ["Hello", "World"]

    def test_extract_all_text_from_artifact_empty(self):
        """Test extracting text when there are no TextParts."""
        artifact = create_json_artifact({"data": 42})
        texts = extract_all_text_from_artifact(artifact)
        assert texts == []


class TestArtifactSerialization:
    """Tests for artifact serialization functions."""

    def test_artifact_to_json_string(self):
        """Test serializing an artifact to JSON string."""
        artifact = create_json_artifact({"key": "value"}, name="test")
        json_str = artifact_to_json_string(artifact)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert "parts" in parsed

    def test_artifact_to_json_string_no_indent(self):
        """Test serializing with no indentation."""
        artifact = create_text_artifact("Test")
        json_str = artifact_to_json_string(artifact, indent=None)

        # Should not have newlines (compact JSON)
        assert "\n" not in json_str

    def test_artifact_from_json_string(self):
        """Test deserializing an artifact from JSON string."""
        original = create_json_artifact({"data": 42}, name="restored")
        json_str = artifact_to_json_string(original)

        restored = artifact_from_json_string(json_str)
        assert restored.name == "restored"

    def test_roundtrip_serialization(self):
        """Test that serialization/deserialization preserves data."""
        original = create_artifact(
            parts=[
                TextPart(text="Hello"),
                DataPart(data={"key": "value"}),
            ],
            name="multipart",
            description="A multipart artifact",
        )

        json_str = artifact_to_json_string(original)
        restored = artifact_from_json_string(json_str)

        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.parts) == len(original.parts)
