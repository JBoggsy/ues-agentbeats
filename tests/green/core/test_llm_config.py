"""Tests for LLM configuration and factory.

Tests cover:
- LLMProvider enum values
- LLMConfig dataclass creation and validation
- UnsupportedModelError exception
- LLMFactory.detect_provider() method
- LLMFactory.create() and create_from_config() methods
- Provider-specific model creation (OpenAI, Anthropic, Google, Ollama)
- Temperature warning for reasoning models
- Base URL override support
- Edge cases and error handling
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.green.core.llm_config import (
    LLMConfig,
    LLMFactory,
    LLMProvider,
    UnsupportedModelError,
    _is_reasoning_model,
)


class TestLLMProvider:
    """Tests for the LLMProvider enum."""

    def test_provider_values(self) -> None:
        """Test that all expected provider values exist."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_provider_count(self) -> None:
        """Test that we have exactly 4 providers."""
        assert len(LLMProvider) == 4


class TestUnsupportedModelError:
    """Tests for the UnsupportedModelError exception."""

    def test_basic_error(self) -> None:
        """Test basic exception creation."""
        error = UnsupportedModelError("unknown-model")
        assert error.model == "unknown-model"
        assert "unknown-model" in str(error)
        assert "Unsupported model identifier" in str(error)

    def test_default_supported_prefixes(self) -> None:
        """Test that default supported prefixes are included in message."""
        error = UnsupportedModelError("bad-model")
        assert "gpt-*" in str(error)
        assert "claude-*" in str(error)
        assert "gemini-*" in str(error)
        assert "ollama/*" in str(error)

    def test_custom_supported_prefixes(self) -> None:
        """Test custom supported prefixes in error message."""
        error = UnsupportedModelError("bad-model", supported_prefixes=["custom-*"])
        assert error.supported_prefixes == ["custom-*"]
        assert "custom-*" in str(error)

    def test_exception_is_raisable(self) -> None:
        """Test that the exception can be raised and caught."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            raise UnsupportedModelError("test-model")
        assert exc_info.value.model == "test-model"


class TestIsReasoningModel:
    """Tests for the _is_reasoning_model helper function."""

    @pytest.mark.parametrize(
        "model",
        [
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o1-2024-12-17",
            "o1-mini-2024-09-12",
            "o3-mini-2025-01-31",
        ],
    )
    def test_reasoning_models_detected(self, model: str) -> None:
        """Test that reasoning models are correctly detected."""
        assert _is_reasoning_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "chatgpt-4o-latest",
            "o1test",  # No dash separator
            "o2-mini",  # Not a reasoning model
            "foo-o1-bar",  # o1 not at start
        ],
    )
    def test_non_reasoning_models(self, model: str) -> None:
        """Test that non-reasoning models are not flagged."""
        assert _is_reasoning_model(model) is False


class TestLLMConfig:
    """Tests for the LLMConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic config creation with defaults."""
        config = LLMConfig(model="gpt-4o")
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.seed is None
        assert config.base_url is None
        assert config.extra_kwargs == {}

    def test_full_creation(self) -> None:
        """Test config creation with all parameters."""
        config = LLMConfig(
            model="claude-3-opus",
            temperature=0.5,
            seed=42,
            base_url="https://custom.api.com",
            extra_kwargs={"max_tokens": 1000},
        )
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.seed == 42
        assert config.base_url == "https://custom.api.com"
        assert config.extra_kwargs == {"max_tokens": 1000}

    def test_frozen(self) -> None:
        """Test that config is immutable."""
        config = LLMConfig(model="gpt-4o")
        with pytest.raises(AttributeError):
            config.model = "different-model"  # type: ignore[misc]

    def test_empty_model_raises(self) -> None:
        """Test that empty model string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            LLMConfig(model="")

    def test_temperature_too_low_raises(self) -> None:
        """Test that temperature below 0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 2.0"):
            LLMConfig(model="gpt-4o", temperature=-0.1)

    def test_temperature_too_high_raises(self) -> None:
        """Test that temperature above 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 2.0"):
            LLMConfig(model="gpt-4o", temperature=2.5)

    def test_temperature_boundary_values(self) -> None:
        """Test temperature at boundary values."""
        config_low = LLMConfig(model="gpt-4o", temperature=0.0)
        assert config_low.temperature == 0.0

        config_high = LLMConfig(model="gpt-4o", temperature=2.0)
        assert config_high.temperature == 2.0


class TestLLMFactoryDetectProvider:
    """Tests for LLMFactory.detect_provider() method."""

    @pytest.mark.parametrize(
        ("model", "expected_provider"),
        [
            # OpenAI models
            ("gpt-4o", LLMProvider.OPENAI),
            ("gpt-4-turbo", LLMProvider.OPENAI),
            ("gpt-3.5-turbo", LLMProvider.OPENAI),
            ("GPT-4o", LLMProvider.OPENAI),  # Case insensitive
            ("o1", LLMProvider.OPENAI),
            ("o1-mini", LLMProvider.OPENAI),
            ("o1-preview", LLMProvider.OPENAI),
            ("o3", LLMProvider.OPENAI),
            ("o3-mini", LLMProvider.OPENAI),
            ("chatgpt-4o-latest", LLMProvider.OPENAI),
            # Anthropic models
            ("claude-3-opus-20240229", LLMProvider.ANTHROPIC),
            ("claude-3-sonnet-20240229", LLMProvider.ANTHROPIC),
            ("claude-3-haiku-20240307", LLMProvider.ANTHROPIC),
            ("claude-2.1", LLMProvider.ANTHROPIC),
            ("CLAUDE-3-opus", LLMProvider.ANTHROPIC),  # Case insensitive
            # Google models
            ("gemini-1.5-pro", LLMProvider.GOOGLE),
            ("gemini-1.5-flash", LLMProvider.GOOGLE),
            ("gemini-pro", LLMProvider.GOOGLE),
            ("GEMINI-1.5-pro", LLMProvider.GOOGLE),  # Case insensitive
            # Ollama models
            ("ollama/llama3.2", LLMProvider.OLLAMA),
            ("ollama/mistral", LLMProvider.OLLAMA),
            ("ollama/codellama", LLMProvider.OLLAMA),
            ("OLLAMA/llama3.2", LLMProvider.OLLAMA),  # Case insensitive
        ],
    )
    def test_detect_provider(self, model: str, expected_provider: LLMProvider) -> None:
        """Test provider detection for various model strings."""
        assert LLMFactory.detect_provider(model) == expected_provider

    def test_unsupported_model_raises(self) -> None:
        """Test that unsupported model raises UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            LLMFactory.detect_provider("unknown-model")
        assert exc_info.value.model == "unknown-model"

    def test_empty_model_raises(self) -> None:
        """Test that empty model string raises UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError):
            LLMFactory.detect_provider("")


class TestLLMFactoryCreate:
    """Tests for LLMFactory.create() and create_from_config() methods."""

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_openai_basic(self, mock_openai: MagicMock) -> None:
        """Test basic OpenAI model creation."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        result = LLMFactory.create("gpt-4o")

        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.7)
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_openai_with_all_params(self, mock_openai: MagicMock) -> None:
        """Test OpenAI model creation with all parameters."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        result = LLMFactory.create(
            "gpt-4o",
            temperature=0.5,
            seed=42,
            base_url="https://custom.api.com",
            max_tokens=1000,
        )

        mock_openai.assert_called_once_with(
            model="gpt-4o",
            temperature=0.5,
            seed=42,
            base_url="https://custom.api.com",
            max_tokens=1000,
        )
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_openai_reasoning_model_ignores_temperature(
        self, mock_openai: MagicMock
    ) -> None:
        """Test that reasoning models don't pass temperature."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        # Default temperature (0.7) - no warning expected
        LLMFactory.create("o1-mini")

        # Temperature should NOT be in the call
        call_kwargs = mock_openai.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["model"] == "o1-mini"

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_openai_reasoning_model_warns_on_custom_temperature(
        self, mock_openai: MagicMock
    ) -> None:
        """Test that custom temperature on reasoning models emits warning."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LLMFactory.create("o1-mini", temperature=0.5)

            assert len(w) == 1
            assert "reasoning model" in str(w[0].message)
            assert "temperature" in str(w[0].message)
            assert "o1-mini" in str(w[0].message)

    @patch("src.green.core.llm_config.ChatAnthropic")
    def test_create_anthropic_basic(self, mock_anthropic: MagicMock) -> None:
        """Test basic Anthropic model creation."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        result = LLMFactory.create("claude-3-opus-20240229")

        mock_anthropic.assert_called_once_with(
            model="claude-3-opus-20240229", temperature=0.7
        )
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatAnthropic")
    def test_create_anthropic_with_base_url(self, mock_anthropic: MagicMock) -> None:
        """Test Anthropic model creation with base_url."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        LLMFactory.create(
            "claude-3-opus", temperature=0.5, base_url="https://custom.anthropic.com"
        )

        mock_anthropic.assert_called_once_with(
            model="claude-3-opus",
            temperature=0.5,
            base_url="https://custom.anthropic.com",
        )

    @patch("src.green.core.llm_config.ChatAnthropic")
    def test_create_anthropic_seed_ignored(self, mock_anthropic: MagicMock) -> None:
        """Test that seed is ignored for Anthropic (logged, not passed)."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        # Should not raise, seed just ignored
        LLMFactory.create("claude-3-opus", seed=42)

        # Seed should NOT be in the call
        call_kwargs = mock_anthropic.call_args[1]
        assert "seed" not in call_kwargs

    @patch("src.green.core.llm_config.ChatGoogleGenerativeAI")
    def test_create_google_basic(self, mock_google: MagicMock) -> None:
        """Test basic Google model creation."""
        mock_instance = MagicMock()
        mock_google.return_value = mock_instance

        result = LLMFactory.create("gemini-1.5-pro")

        mock_google.assert_called_once_with(model="gemini-1.5-pro", temperature=0.7)
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatGoogleGenerativeAI")
    def test_create_google_seed_and_base_url_ignored(
        self, mock_google: MagicMock
    ) -> None:
        """Test that seed and base_url are ignored for Google."""
        mock_instance = MagicMock()
        mock_google.return_value = mock_instance

        LLMFactory.create(
            "gemini-1.5-pro", seed=42, base_url="https://custom.google.com"
        )

        call_kwargs = mock_google.call_args[1]
        assert "seed" not in call_kwargs
        assert "base_url" not in call_kwargs

    @patch("src.green.core.llm_config.ChatOllama")
    def test_create_ollama_basic(self, mock_ollama: MagicMock) -> None:
        """Test basic Ollama model creation."""
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        result = LLMFactory.create("ollama/llama3.2")

        # Note: "ollama/" prefix should be stripped
        mock_ollama.assert_called_once_with(model="llama3.2", temperature=0.7)
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatOllama")
    def test_create_ollama_with_base_url(self, mock_ollama: MagicMock) -> None:
        """Test Ollama model creation with custom base_url."""
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        LLMFactory.create(
            "ollama/mistral",
            temperature=0.3,
            base_url="http://localhost:11434",
        )

        mock_ollama.assert_called_once_with(
            model="mistral",
            temperature=0.3,
            base_url="http://localhost:11434",
        )

    @patch("src.green.core.llm_config.ChatOllama")
    def test_create_ollama_seed_ignored(self, mock_ollama: MagicMock) -> None:
        """Test that seed is ignored for Ollama."""
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        LLMFactory.create("ollama/llama3.2", seed=42)

        call_kwargs = mock_ollama.call_args[1]
        assert "seed" not in call_kwargs

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_from_config(self, mock_openai: MagicMock) -> None:
        """Test create_from_config method."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        config = LLMConfig(model="gpt-4o", temperature=0.5, seed=42)
        result = LLMFactory.create_from_config(config)

        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.5, seed=42)
        assert result is mock_instance

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_from_config_with_extra_kwargs(
        self, mock_openai: MagicMock
    ) -> None:
        """Test that extra_kwargs are passed through."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        config = LLMConfig(
            model="gpt-4o",
            extra_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )
        LLMFactory.create_from_config(config)

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["top_p"] == 0.9

    def test_create_unsupported_model_raises(self) -> None:
        """Test that creating unsupported model raises error."""
        with pytest.raises(UnsupportedModelError):
            LLMFactory.create("unknown-model")


class TestLLMFactoryHelpers:
    """Tests for LLMFactory helper methods."""

    def test_get_supported_providers(self) -> None:
        """Test get_supported_providers returns all providers."""
        providers = LLMFactory.get_supported_providers()
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.ANTHROPIC in providers
        assert LLMProvider.GOOGLE in providers
        assert LLMProvider.OLLAMA in providers
        assert len(providers) == 4

    def test_get_supported_prefixes(self) -> None:
        """Test get_supported_prefixes returns expected prefixes."""
        prefixes = LLMFactory.get_supported_prefixes()
        assert "gpt-" in prefixes
        assert "claude-" in prefixes
        assert "gemini-" in prefixes
        assert "ollama/" in prefixes
        assert "o1" in prefixes or "o1-" in prefixes


class TestLLMFactoryIntegration:
    """Integration-style tests for LLMFactory (still mocked but more realistic)."""

    @patch("src.green.core.llm_config.ChatOpenAI")
    @patch("src.green.core.llm_config.ChatAnthropic")
    def test_multiple_providers_same_factory(
        self, mock_anthropic: MagicMock, mock_openai: MagicMock
    ) -> None:
        """Test creating models from multiple providers."""
        mock_openai.return_value = MagicMock(name="openai_instance")
        mock_anthropic.return_value = MagicMock(name="anthropic_instance")

        llm1 = LLMFactory.create("gpt-4o")
        llm2 = LLMFactory.create("claude-3-opus")

        mock_openai.assert_called_once()
        mock_anthropic.assert_called_once()
        assert llm1 is not llm2

    @patch("src.green.core.llm_config.ChatOpenAI")
    def test_create_multiple_openai_models(self, mock_openai: MagicMock) -> None:
        """Test creating multiple different OpenAI models."""
        mock_openai.return_value = MagicMock()

        LLMFactory.create("gpt-4o")
        LLMFactory.create("gpt-3.5-turbo", temperature=0.0)

        assert mock_openai.call_count == 2
        calls = mock_openai.call_args_list
        assert calls[0][1]["model"] == "gpt-4o"
        assert calls[1][1]["model"] == "gpt-3.5-turbo"
        assert calls[1][1]["temperature"] == 0.0
