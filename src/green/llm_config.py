"""LLM configuration and factory for the Green Agent.

This module provides a factory for creating LangChain chat model instances
supporting multiple providers: OpenAI, Anthropic, Google (Gemini), and
local Ollama. The factory parses model identifier strings and instantiates
the appropriate provider class with consistent configuration.

Key Classes:
    LLMProvider: Enum of supported LLM providers.
    LLMConfig: Configuration dataclass for LLM creation.
    LLMFactory: Factory class for creating LLM instances.
    UnsupportedModelError: Exception for unrecognized model identifiers.

Supported Model Prefixes:
    - OpenAI: ``gpt-*``, ``o1-*``, ``o3-*``, ``chatgpt-*``
    - Anthropic: ``claude-*``
    - Google: ``gemini-*``
    - Ollama: ``ollama/*`` (e.g., ``ollama/llama3.2``)

Example:
    >>> from src.green.llm_config import LLMFactory
    >>> llm = LLMFactory.create("gpt-4o", temperature=0.7)
    >>> llm = LLMFactory.create("claude-3-opus-20240229")
    >>> llm = LLMFactory.create("ollama/llama3.2", base_url="http://localhost:11434")
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers.

    Attributes:
        OPENAI: OpenAI models (GPT-4, GPT-3.5, o1, o3, etc.)
        ANTHROPIC: Anthropic models (Claude family)
        GOOGLE: Google models (Gemini family)
        OLLAMA: Local Ollama models
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class UnsupportedModelError(Exception):
    """Raised when a model identifier is not recognized.

    Attributes:
        model: The unrecognized model identifier.
        supported_prefixes: List of supported model prefixes.
    """

    def __init__(self, model: str, supported_prefixes: list[str] | None = None) -> None:
        """Initialize the exception.

        Args:
            model: The unrecognized model identifier.
            supported_prefixes: Optional list of supported model prefixes for
                the error message.
        """
        self.model = model
        self.supported_prefixes = supported_prefixes or [
            "gpt-*",
            "o1-*",
            "o3-*",
            "chatgpt-*",
            "claude-*",
            "gemini-*",
            "ollama/*",
        ]
        prefix_str = ", ".join(self.supported_prefixes)
        super().__init__(
            f"Unsupported model identifier: '{model}'. "
            f"Supported prefixes: {prefix_str}"
        )


# OpenAI reasoning models that don't support temperature
_OPENAI_REASONING_MODELS = frozenset({"o1", "o1-mini", "o1-preview", "o3", "o3-mini"})


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is an OpenAI reasoning model that doesn't support temperature.

    Args:
        model: The model identifier to check.

    Returns:
        True if the model is a reasoning model (o1/o3 family), False otherwise.
    """
    # Check exact match first (e.g., "o1", "o3-mini")
    if model in _OPENAI_REASONING_MODELS:
        return True
    # Check if it starts with a reasoning model prefix followed by a dash
    # (e.g., "o1-2024-12-17", "o3-mini-2025-01-31")
    for reasoning_model in _OPENAI_REASONING_MODELS:
        if model.startswith(f"{reasoning_model}-"):
            return True
    return False


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for creating an LLM instance.

    This dataclass holds all parameters needed to instantiate a LangChain
    chat model. It is immutable (frozen) to ensure configuration consistency.

    Attributes:
        model: Model identifier string (e.g., "gpt-4o", "claude-3-opus").
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            Note: Ignored for OpenAI reasoning models (o1/o3 family).
        seed: Optional seed for reproducibility. Only supported by some providers
            (OpenAI). Ignored by providers that don't support it.
        base_url: Optional base URL override for API endpoint. Useful for
            OpenAI-compatible APIs (Azure, vLLM) or custom Ollama endpoints.
        extra_kwargs: Additional provider-specific keyword arguments passed
            directly to the LangChain model constructor.

    Example:
        >>> config = LLMConfig(model="gpt-4o", temperature=0.5, seed=42)
        >>> config = LLMConfig(
        ...     model="ollama/llama3.2",
        ...     base_url="http://localhost:11434",
        ... )
    """

    model: str
    temperature: float = 0.7
    seed: int | None = None
    base_url: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.model:
            raise ValueError("Model identifier cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )


class LLMFactory:
    """Factory for creating LangChain chat model instances.

    This factory parses model identifier strings and creates the appropriate
    LangChain chat model instance for the detected provider. It handles
    provider-specific configuration and parameter mapping.

    Model Detection:
        The factory uses prefix matching to determine the provider:
        - ``gpt-*``, ``o1-*``, ``o3-*``, ``chatgpt-*`` → OpenAI
        - ``claude-*`` → Anthropic
        - ``gemini-*`` → Google
        - ``ollama/*`` → Ollama (e.g., ``ollama/llama3.2``)

    Example:
        >>> # Create with model string (uses defaults)
        >>> llm = LLMFactory.create("gpt-4o")

        >>> # Create with full configuration
        >>> config = LLMConfig(model="claude-3-opus", temperature=0.5)
        >>> llm = LLMFactory.create_from_config(config)

        >>> # Create with custom base URL
        >>> llm = LLMFactory.create(
        ...     "gpt-4o",
        ...     base_url="https://my-azure-endpoint.openai.azure.com/",
        ... )
    """

    # Mapping of model prefixes to providers
    _PREFIX_TO_PROVIDER: dict[str, LLMProvider] = {
        "gpt-": LLMProvider.OPENAI,
        "o1-": LLMProvider.OPENAI,
        "o1": LLMProvider.OPENAI,  # Exact match for "o1" without suffix
        "o3-": LLMProvider.OPENAI,
        "o3": LLMProvider.OPENAI,  # Exact match for "o3" without suffix
        "chatgpt-": LLMProvider.OPENAI,
        "claude-": LLMProvider.ANTHROPIC,
        "gemini-": LLMProvider.GOOGLE,
        "ollama/": LLMProvider.OLLAMA,
    }

    @classmethod
    def detect_provider(cls, model: str) -> LLMProvider:
        """Detect the LLM provider from a model identifier string.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus").

        Returns:
            The detected LLMProvider enum value.

        Raises:
            UnsupportedModelError: If the model identifier doesn't match any
                known provider prefix.

        Example:
            >>> LLMFactory.detect_provider("gpt-4o")
            LLMProvider.OPENAI
            >>> LLMFactory.detect_provider("claude-3-opus")
            LLMProvider.ANTHROPIC
        """
        if not model:
            raise UnsupportedModelError(model)

        model_lower = model.lower()

        # Check for exact matches first (e.g., "o1", "o3")
        if model_lower in cls._PREFIX_TO_PROVIDER:
            return cls._PREFIX_TO_PROVIDER[model_lower]

        # Check for prefix matches
        for prefix, provider in cls._PREFIX_TO_PROVIDER.items():
            if model_lower.startswith(prefix):
                return provider

        raise UnsupportedModelError(model)

    @classmethod
    def create(
        cls,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
        base_url: str | None = None,
        **extra_kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM instance from parameters.

        This is a convenience method that creates an LLMConfig internally
        and delegates to create_from_config().

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus").
            temperature: Sampling temperature (0.0 = deterministic).
                Note: Ignored for OpenAI reasoning models (o1/o3 family).
            seed: Optional seed for reproducibility (provider-dependent).
            base_url: Optional base URL override for API endpoint.
            **extra_kwargs: Additional provider-specific arguments.

        Returns:
            Configured LangChain BaseChatModel instance.

        Raises:
            UnsupportedModelError: If the model identifier is not recognized.
            ValueError: If configuration parameters are invalid.

        Example:
            >>> llm = LLMFactory.create("gpt-4o", temperature=0.5, seed=42)
            >>> llm = LLMFactory.create("ollama/llama3.2")
        """
        config = LLMConfig(
            model=model,
            temperature=temperature,
            seed=seed,
            base_url=base_url,
            extra_kwargs=extra_kwargs,
        )
        return cls.create_from_config(config)

    @classmethod
    def create_from_config(cls, config: LLMConfig) -> BaseChatModel:
        """Create an LLM instance from a configuration object.

        Args:
            config: LLMConfig instance with model parameters.

        Returns:
            Configured LangChain BaseChatModel instance.

        Raises:
            UnsupportedModelError: If the model identifier is not recognized.

        Example:
            >>> config = LLMConfig(model="claude-3-opus", temperature=0.5)
            >>> llm = LLMFactory.create_from_config(config)
        """
        provider = cls.detect_provider(config.model)

        if provider == LLMProvider.OPENAI:
            return cls._create_openai(config)
        elif provider == LLMProvider.ANTHROPIC:
            return cls._create_anthropic(config)
        elif provider == LLMProvider.GOOGLE:
            return cls._create_google(config)
        elif provider == LLMProvider.OLLAMA:
            return cls._create_ollama(config)
        else:
            # This should never happen if detect_provider is working correctly
            raise UnsupportedModelError(config.model)

    @classmethod
    def _create_openai(cls, config: LLMConfig) -> ChatOpenAI:
        """Create an OpenAI chat model instance.

        Args:
            config: LLM configuration.

        Returns:
            Configured ChatOpenAI instance.
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            **config.extra_kwargs,
        }

        # Handle reasoning models that don't support temperature
        if _is_reasoning_model(config.model):
            if config.temperature != 0.7:  # Non-default temperature was specified
                warnings.warn(
                    f"Model '{config.model}' is a reasoning model that does not "
                    f"support temperature. The temperature parameter "
                    f"({config.temperature}) will be ignored.",
                    UserWarning,
                    stacklevel=4,  # Points to the caller of create/create_from_config
                )
            # Don't pass temperature for reasoning models
        else:
            kwargs["temperature"] = config.temperature

        if config.seed is not None:
            kwargs["seed"] = config.seed

        if config.base_url is not None:
            kwargs["base_url"] = config.base_url

        logger.debug(f"Creating OpenAI model: {config.model}")
        return ChatOpenAI(**kwargs)

    @classmethod
    def _create_anthropic(cls, config: LLMConfig) -> ChatAnthropic:
        """Create an Anthropic chat model instance.

        Args:
            config: LLM configuration.

        Returns:
            Configured ChatAnthropic instance.
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            **config.extra_kwargs,
        }

        if config.base_url is not None:
            kwargs["base_url"] = config.base_url

        # Note: Anthropic doesn't support seed parameter
        if config.seed is not None:
            logger.debug(
                f"Seed parameter ({config.seed}) ignored for Anthropic model "
                f"'{config.model}' - Anthropic does not support seed."
            )

        logger.debug(f"Creating Anthropic model: {config.model}")
        return ChatAnthropic(**kwargs)

    @classmethod
    def _create_google(cls, config: LLMConfig) -> ChatGoogleGenerativeAI:
        """Create a Google Generative AI chat model instance.

        Args:
            config: LLM configuration.

        Returns:
            Configured ChatGoogleGenerativeAI instance.
        """
        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            **config.extra_kwargs,
        }

        # Note: Google doesn't support seed or base_url in the same way
        if config.seed is not None:
            logger.debug(
                f"Seed parameter ({config.seed}) ignored for Google model "
                f"'{config.model}' - Google does not support seed."
            )

        if config.base_url is not None:
            logger.debug(
                f"Base URL parameter ignored for Google model '{config.model}' - "
                f"Google does not support custom base URLs."
            )

        logger.debug(f"Creating Google model: {config.model}")
        return ChatGoogleGenerativeAI(**kwargs)

    @classmethod
    def _create_ollama(cls, config: LLMConfig) -> ChatOllama:
        """Create an Ollama chat model instance.

        The model string should be in the format "ollama/model-name".
        The "ollama/" prefix is stripped to get the actual model name.

        Args:
            config: LLM configuration.

        Returns:
            Configured ChatOllama instance.
        """
        # Strip the "ollama/" prefix to get the actual model name
        model_name = config.model
        if model_name.lower().startswith("ollama/"):
            model_name = model_name[7:]  # len("ollama/") == 7

        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": config.temperature,
            **config.extra_kwargs,
        }

        if config.base_url is not None:
            kwargs["base_url"] = config.base_url

        # Note: Ollama doesn't support seed in LangChain
        if config.seed is not None:
            logger.debug(
                f"Seed parameter ({config.seed}) ignored for Ollama model "
                f"'{model_name}' - LangChain Ollama integration does not support seed."
            )

        logger.debug(f"Creating Ollama model: {model_name}")
        return ChatOllama(**kwargs)

    @classmethod
    def get_supported_providers(cls) -> list[LLMProvider]:
        """Get a list of all supported LLM providers.

        Returns:
            List of LLMProvider enum values.
        """
        return list(LLMProvider)

    @classmethod
    def get_supported_prefixes(cls) -> list[str]:
        """Get a list of all supported model prefixes.

        Returns:
            List of prefix strings (e.g., ["gpt-", "claude-", "ollama/"]).
        """
        return list(cls._PREFIX_TO_PROVIDER.keys())
