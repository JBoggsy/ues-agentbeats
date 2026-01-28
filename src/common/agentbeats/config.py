"""AgentBeats configuration models.

This module provides Pydantic-based configuration models for Green and Purple
agents, with support for CLI argument parsing and environment variable loading.

Configuration Hierarchy:
    - AgentBeatsConfig: Base configuration shared by all agents
    - GreenAgentConfig: Configuration specific to Green agents (evaluators)
    - PurpleAgentConfig: Configuration specific to Purple agents (participants)

Configuration Sources (in order of precedence, highest first):
    1. Explicit constructor arguments
    2. CLI arguments (via from_cli_args)
    3. Environment variables (automatic via pydantic-settings)
    4. Default values

Environment Variables:
    Environment variables are prefixed with "AGENTBEATS_" for base config,
    "AGENTBEATS_GREEN_" for Green agent config, and "AGENTBEATS_PURPLE_" for
    Purple agent config. Variable names are derived from field names in
    SCREAMING_SNAKE_CASE.

    Examples:
        AGENTBEATS_HOST=0.0.0.0
        AGENTBEATS_PORT=8080
        AGENTBEATS_GREEN_VERBOSE_UPDATES=false
        AGENTBEATS_PURPLE_MODEL=gpt-4-turbo

CLI Arguments:
    Use the from_cli_args() class method to parse command-line arguments.
    Standard arguments supported:
        --host: Server host (default: 0.0.0.0)
        --port: Server port (default: 8000 for Green, 8001 for Purple)
        --card-url: URL where agent card is accessible

Example:
    >>> from src.common.agentbeats.config import GreenAgentConfig
    >>> # From defaults
    >>> config = GreenAgentConfig()
    >>> config.host
    '0.0.0.0'
    >>> # From CLI arguments
    >>> config = GreenAgentConfig.from_cli_args(['--port', '9000'])
    >>> config.port
    9000
    >>> # From environment (set AGENTBEATS_GREEN_VERBOSE_UPDATES=false)
    >>> import os
    >>> os.environ['AGENTBEATS_GREEN_VERBOSE_UPDATES'] = 'false'
    >>> config = GreenAgentConfig()
    >>> config.verbose_updates
    False
"""

from __future__ import annotations

import argparse
from typing import Any, Literal, Self, Sequence

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Base Configuration
# =============================================================================


class AgentBeatsConfig(BaseSettings):
    """Base configuration for all AgentBeats agents.

    This class provides common configuration fields used by both Green and
    Purple agents. Configuration can be loaded from environment variables
    (prefixed with AGENTBEATS_), CLI arguments, or constructor arguments.

    Attributes:
        host: Server host address to bind to.
        port: Server port to listen on.
        card_url: URL where the agent card will be accessible. If None,
            defaults to http://{host}:{port}/.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Example:
        >>> config = AgentBeatsConfig(host="localhost", port=9000)
        >>> config.host
        'localhost'
        >>> config.effective_card_url
        'http://localhost:9000/'
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTBEATS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Server host address to bind to",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port to listen on",
    )
    card_url: str | None = Field(
        default=None,
        description="URL where the agent card will be accessible",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        """Normalize log level to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v

    @property
    def effective_card_url(self) -> str:
        """Get the effective card URL, using default if not explicitly set.

        Returns:
            The configured card_url, or a default based on host and port.
        """
        if self.card_url is not None:
            return self.card_url
        return f"http://{self.host}:{self.port}/"

    @classmethod
    def from_cli_args(
        cls,
        args: Sequence[str] | None = None,
        **overrides: Any,
    ) -> Self:
        """Create configuration from CLI arguments.

        Parses command-line arguments and combines them with environment
        variables and defaults. Explicit overrides take highest precedence.

        Args:
            args: Command-line arguments to parse. If None, uses sys.argv[1:].
            **overrides: Additional keyword arguments that override all other
                sources.

        Returns:
            A new configuration instance.

        Example:
            >>> config = AgentBeatsConfig.from_cli_args(['--port', '9000'])
            >>> config.port
            9000
        """
        parser = cls._create_argument_parser()
        parsed, _ = parser.parse_known_args(args)
        cli_values = cls._parsed_args_to_dict(parsed)

        # Filter out None values so they don't override environment defaults
        cli_values = {k: v for k, v in cli_values.items() if v is not None}

        # Merge CLI values with overrides (overrides take precedence)
        merged = {**cli_values, **overrides}
        return cls(**merged)

    @classmethod
    def _create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create the argument parser for this config class.

        Subclasses can override to add additional arguments.

        Returns:
            An ArgumentParser configured with base arguments.
        """
        parser = argparse.ArgumentParser(
            description="AgentBeats Agent Configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--host",
            type=str,
            default=None,
            help="Server host address to bind to",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="Server port to listen on",
        )
        parser.add_argument(
            "--card-url",
            type=str,
            default=None,
            dest="card_url",
            help="URL where the agent card will be accessible",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=None,
            dest="log_level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )
        return parser

    @classmethod
    def _parsed_args_to_dict(cls, parsed: argparse.Namespace) -> dict[str, Any]:
        """Convert parsed arguments to a dictionary.

        Subclasses can override to handle additional arguments.

        Args:
            parsed: The parsed argument namespace.

        Returns:
            Dictionary of configuration values from CLI arguments.
        """
        return {
            "host": parsed.host,
            "port": parsed.port,
            "card_url": parsed.card_url,
            "log_level": parsed.log_level,
        }


# =============================================================================
# Green Agent Configuration
# =============================================================================


class GreenAgentConfig(AgentBeatsConfig):
    """Configuration specific to Green agents (evaluators).

    Green agents orchestrate assessments, manage the UES environment, and
    evaluate Purple agent performance. This config extends the base config
    with Green-specific settings.

    Attributes:
        port: Server port (defaults to 8000 for Green agents).
        verbose_updates: Whether to emit detailed task updates during
            assessment. When True, emits updates for each turn, action,
            and criterion evaluation. When False, only emits start/end updates.
        ues_url: URL of the UES instance to use for assessments.
        ues_proctor_api_key: API key for proctor-level UES access. Stored
            as SecretStr to prevent accidental logging.
        scenarios_dir: Directory containing scenario definitions.
        default_max_turns: Default maximum turns if not specified in scenario.
        default_turn_timeout: Default timeout per turn in seconds.
        response_generator_model: LLM model for generating character responses.
        evaluation_model: LLM model for evaluating Purple agent performance.

    Example:
        >>> config = GreenAgentConfig(verbose_updates=False)
        >>> config.verbose_updates
        False
        >>> config.port
        8000

    Environment Variables:
        AGENTBEATS_GREEN_VERBOSE_UPDATES: Enable/disable verbose updates
        AGENTBEATS_GREEN_UES_URL: UES instance URL
        AGENTBEATS_GREEN_UES_PROCTOR_API_KEY: UES proctor API key
        AGENTBEATS_GREEN_SCENARIOS_DIR: Scenarios directory path
        AGENTBEATS_GREEN_RESPONSE_GENERATOR_MODEL: LLM for responses
        AGENTBEATS_GREEN_EVALUATION_MODEL: LLM for evaluation
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTBEATS_GREEN_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Override default port for Green agent
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port to listen on",
    )

    # Green-specific settings
    verbose_updates: bool = Field(
        default=True,
        description="Whether to emit detailed task updates during assessment",
    )
    ues_url: str = Field(
        default="http://localhost:8080",
        description="URL of the UES instance to use for assessments",
    )
    ues_proctor_api_key: SecretStr | None = Field(
        default=None,
        description="API key for proctor-level UES access",
    )
    scenarios_dir: str = Field(
        default="scenarios",
        description="Directory containing scenario definitions",
    )
    default_max_turns: int = Field(
        default=100,
        ge=1,
        description="Default maximum turns if not specified in scenario",
    )
    default_turn_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Default timeout per turn in seconds",
    )
    response_generator_model: str = Field(
        default="gpt-4o",
        description="LLM model for generating character responses",
    )
    evaluation_model: str = Field(
        default="gpt-4o",
        description="LLM model for evaluating Purple agent performance",
    )

    @classmethod
    def _create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create the argument parser with Green-specific arguments."""
        parser = super()._create_argument_parser()
        parser.description = "AgentBeats Green Agent Configuration"

        parser.add_argument(
            "--verbose-updates",
            action="store_true",
            default=None,
            dest="verbose_updates",
            help="Emit detailed task updates during assessment",
        )
        parser.add_argument(
            "--no-verbose-updates",
            action="store_false",
            dest="verbose_updates",
            help="Only emit start/end updates",
        )
        parser.add_argument(
            "--ues-url",
            type=str,
            default=None,
            dest="ues_url",
            help="URL of the UES instance",
        )
        parser.add_argument(
            "--scenarios-dir",
            type=str,
            default=None,
            dest="scenarios_dir",
            help="Directory containing scenario definitions",
        )
        parser.add_argument(
            "--default-max-turns",
            type=int,
            default=None,
            dest="default_max_turns",
            help="Default maximum turns per assessment",
        )
        parser.add_argument(
            "--default-turn-timeout",
            type=float,
            default=None,
            dest="default_turn_timeout",
            help="Default timeout per turn in seconds",
        )
        parser.add_argument(
            "--response-model",
            type=str,
            default=None,
            dest="response_generator_model",
            help="LLM model for generating character responses",
        )
        parser.add_argument(
            "--evaluation-model",
            type=str,
            default=None,
            dest="evaluation_model",
            help="LLM model for evaluation",
        )
        return parser

    @classmethod
    def _parsed_args_to_dict(cls, parsed: argparse.Namespace) -> dict[str, Any]:
        """Convert parsed arguments to a dictionary with Green-specific args."""
        base = super()._parsed_args_to_dict(parsed)
        base.update(
            {
                "verbose_updates": parsed.verbose_updates,
                "ues_url": parsed.ues_url,
                "scenarios_dir": parsed.scenarios_dir,
                "default_max_turns": parsed.default_max_turns,
                "default_turn_timeout": parsed.default_turn_timeout,
                "response_generator_model": parsed.response_generator_model,
                "evaluation_model": parsed.evaluation_model,
            }
        )
        return base


# =============================================================================
# Purple Agent Configuration
# =============================================================================


class PurpleAgentConfig(AgentBeatsConfig):
    """Configuration specific to Purple agents (participants).

    Purple agents are AI personal assistants that interact with UES to
    demonstrate their capabilities. This config extends the base config
    with Purple-specific settings.

    Attributes:
        port: Server port (defaults to 8001 for Purple agents).
        model: LLM model for the Purple agent's reasoning.
        max_actions_per_turn: Maximum number of actions allowed per turn.
        temperature: LLM temperature for response generation (0.0-2.0).
        enable_reflection: Whether to include self-reflection in reasoning.
        action_delay: Minimum delay between actions in seconds (for rate
            limiting or human-readable pacing).

    Example:
        >>> config = PurpleAgentConfig(model="gpt-4-turbo")
        >>> config.model
        'gpt-4-turbo'
        >>> config.port
        8001

    Environment Variables:
        AGENTBEATS_PURPLE_MODEL: LLM model name
        AGENTBEATS_PURPLE_MAX_ACTIONS_PER_TURN: Max actions per turn
        AGENTBEATS_PURPLE_TEMPERATURE: LLM temperature
        AGENTBEATS_PURPLE_ENABLE_REFLECTION: Enable self-reflection
        AGENTBEATS_PURPLE_ACTION_DELAY: Delay between actions
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTBEATS_PURPLE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Override default port for Purple agent
    port: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="Server port to listen on",
    )

    # Purple-specific settings
    model: str = Field(
        default="gpt-4o",
        description="LLM model for the Purple agent's reasoning",
    )
    max_actions_per_turn: int = Field(
        default=50,
        ge=1,
        description="Maximum number of actions allowed per turn",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation",
    )
    enable_reflection: bool = Field(
        default=True,
        description="Whether to include self-reflection in reasoning",
    )
    action_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum delay between actions in seconds",
    )

    @classmethod
    def _create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create the argument parser with Purple-specific arguments."""
        parser = super()._create_argument_parser()
        parser.description = "AgentBeats Purple Agent Configuration"

        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="LLM model for the agent's reasoning",
        )
        parser.add_argument(
            "--max-actions-per-turn",
            type=int,
            default=None,
            dest="max_actions_per_turn",
            help="Maximum actions allowed per turn",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="LLM temperature (0.0-2.0)",
        )
        parser.add_argument(
            "--enable-reflection",
            action="store_true",
            default=None,
            dest="enable_reflection",
            help="Enable self-reflection in reasoning",
        )
        parser.add_argument(
            "--no-reflection",
            action="store_false",
            dest="enable_reflection",
            help="Disable self-reflection",
        )
        parser.add_argument(
            "--action-delay",
            type=float,
            default=None,
            dest="action_delay",
            help="Minimum delay between actions (seconds)",
        )
        return parser

    @classmethod
    def _parsed_args_to_dict(cls, parsed: argparse.Namespace) -> dict[str, Any]:
        """Convert parsed arguments to a dictionary with Purple-specific args."""
        base = super()._parsed_args_to_dict(parsed)
        base.update(
            {
                "model": parsed.model,
                "max_actions_per_turn": parsed.max_actions_per_turn,
                "temperature": parsed.temperature,
                "enable_reflection": parsed.enable_reflection,
                "action_delay": parsed.action_delay,
            }
        )
        return base


# =============================================================================
# Configuration Utilities
# =============================================================================


def merge_configs(
    base: AgentBeatsConfig,
    overrides: dict[str, Any],
) -> AgentBeatsConfig:
    """Create a new configuration with overrides applied.

    Creates a new configuration instance of the same type as the base,
    with the override values merged in. The base configuration is not
    modified.

    Args:
        base: The base configuration to copy.
        overrides: Dictionary of values to override.

    Returns:
        A new configuration instance with overrides applied.

    Example:
        >>> base = GreenAgentConfig(port=8000)
        >>> updated = merge_configs(base, {"port": 9000})
        >>> updated.port
        9000
        >>> base.port
        8000
    """
    base_dict = base.model_dump()
    merged = {**base_dict, **overrides}
    return type(base)(**merged)


def validate_config(config: AgentBeatsConfig) -> list[str]:
    """Validate a configuration and return any warnings.

    Performs additional validation beyond Pydantic's built-in validation,
    checking for potential issues that don't prevent operation but might
    cause problems.

    Args:
        config: The configuration to validate.

    Returns:
        A list of warning messages. Empty if no issues found.

    Example:
        >>> config = GreenAgentConfig(default_turn_timeout=10.0)
        >>> warnings = validate_config(config)
        >>> "turn timeout is very short" in warnings[0].lower()
        True
    """
    warnings: list[str] = []

    # Check for potentially problematic values
    if config.port < 1024:
        warnings.append(
            f"Port {config.port} is a privileged port and may require "
            "elevated permissions"
        )

    if isinstance(config, GreenAgentConfig):
        if config.default_turn_timeout < 30.0:
            warnings.append(
                f"Turn timeout of {config.default_turn_timeout}s is very short "
                "and may cause issues with slower LLM responses"
            )
        if config.ues_proctor_api_key is None:
            warnings.append(
                "No UES proctor API key configured. Assessment will fail "
                "without proctor-level access."
            )

    if isinstance(config, PurpleAgentConfig):
        if config.temperature > 1.5:
            warnings.append(
                f"Temperature {config.temperature} is quite high and may "
                "lead to inconsistent behavior"
            )
        if config.max_actions_per_turn > 100:
            warnings.append(
                f"max_actions_per_turn={config.max_actions_per_turn} is high "
                "and may lead to runaway action loops"
            )

    return warnings
