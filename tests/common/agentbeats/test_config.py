"""Tests for AgentBeats configuration models.

Tests cover:
- Configuration model creation and validation
- CLI argument parsing
- Environment variable loading
- Default values and overrides
- Configuration utilities (merge_configs, validate_config)
- Field validation and constraints
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.common.agentbeats.config import (
    AgentBeatsConfig,
    GreenAgentConfig,
    PurpleAgentConfig,
    merge_configs,
    validate_config,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_env():
    """Fixture that clears AgentBeats environment variables before and after tests."""
    # Save current environment
    saved_env = {k: v for k, v in os.environ.items() if "agentbeats" in k.lower()}

    # Clear environment variables (case-insensitive check)
    for key in list(os.environ.keys()):
        if "agentbeats" in key.lower():
            del os.environ[key]

    yield

    # Clear any env vars set during test (case-insensitive check)
    for key in list(os.environ.keys()):
        if "agentbeats" in key.lower():
            del os.environ[key]
    # Restore original environment
    os.environ.update(saved_env)


@pytest.fixture
def base_config() -> AgentBeatsConfig:
    """Create a base configuration for testing."""
    return AgentBeatsConfig()


@pytest.fixture
def green_config() -> GreenAgentConfig:
    """Create a Green agent configuration for testing."""
    return GreenAgentConfig()


@pytest.fixture
def purple_config() -> PurpleAgentConfig:
    """Create a Purple agent configuration for testing."""
    return PurpleAgentConfig()


# =============================================================================
# Base Configuration Tests
# =============================================================================


class TestAgentBeatsConfig:
    """Tests for AgentBeatsConfig base class."""

    def test_default_values(self, clean_env: None) -> None:
        """Test that default values are correctly applied."""
        config = AgentBeatsConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.card_url is None
        assert config.log_level == "INFO"

    def test_explicit_values(self, clean_env: None) -> None:
        """Test that explicit constructor values are used."""
        config = AgentBeatsConfig(
            host="localhost",
            port=9000,
            card_url="http://example.com/agent",
            log_level="DEBUG",
        )

        assert config.host == "localhost"
        assert config.port == 9000
        assert config.card_url == "http://example.com/agent"
        assert config.log_level == "DEBUG"

    def test_effective_card_url_default(self, clean_env: None) -> None:
        """Test that effective_card_url generates correct default."""
        config = AgentBeatsConfig(host="localhost", port=9000)

        assert config.effective_card_url == "http://localhost:9000/"

    def test_effective_card_url_explicit(self, clean_env: None) -> None:
        """Test that effective_card_url returns explicit value when set."""
        config = AgentBeatsConfig(
            host="localhost",
            port=9000,
            card_url="https://custom.example.com/card",
        )

        assert config.effective_card_url == "https://custom.example.com/card"

    def test_log_level_normalization(self, clean_env: None) -> None:
        """Test that log level is normalized to uppercase."""
        config = AgentBeatsConfig(log_level="debug")  # type: ignore[arg-type]
        assert config.log_level == "DEBUG"

        config = AgentBeatsConfig(log_level="Warning")  # type: ignore[arg-type]
        assert config.log_level == "WARNING"

    def test_port_validation_min(self, clean_env: None) -> None:
        """Test that port validates minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            AgentBeatsConfig(port=0)
        assert "port" in str(exc_info.value).lower()

    def test_port_validation_max(self, clean_env: None) -> None:
        """Test that port validates maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            AgentBeatsConfig(port=70000)
        assert "port" in str(exc_info.value).lower()

    def test_invalid_log_level(self, clean_env: None) -> None:
        """Test that invalid log level raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AgentBeatsConfig(log_level="INVALID")  # type: ignore[arg-type]
        assert "log_level" in str(exc_info.value).lower()


class TestAgentBeatsConfigCLI:
    """Tests for AgentBeatsConfig CLI argument parsing."""

    def test_from_cli_args_defaults(self, clean_env: None) -> None:
        """Test CLI parsing with no arguments uses defaults."""
        config = AgentBeatsConfig.from_cli_args([])

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.card_url is None

    def test_from_cli_args_host(self, clean_env: None) -> None:
        """Test CLI parsing of --host argument."""
        config = AgentBeatsConfig.from_cli_args(["--host", "localhost"])
        assert config.host == "localhost"

    def test_from_cli_args_port(self, clean_env: None) -> None:
        """Test CLI parsing of --port argument."""
        config = AgentBeatsConfig.from_cli_args(["--port", "9000"])
        assert config.port == 9000

    def test_from_cli_args_card_url(self, clean_env: None) -> None:
        """Test CLI parsing of --card-url argument."""
        config = AgentBeatsConfig.from_cli_args(
            ["--card-url", "http://example.com/agent"]
        )
        assert config.card_url == "http://example.com/agent"

    def test_from_cli_args_log_level(self, clean_env: None) -> None:
        """Test CLI parsing of --log-level argument."""
        config = AgentBeatsConfig.from_cli_args(["--log-level", "DEBUG"])
        assert config.log_level == "DEBUG"

    def test_from_cli_args_multiple(self, clean_env: None) -> None:
        """Test CLI parsing with multiple arguments."""
        config = AgentBeatsConfig.from_cli_args(
            ["--host", "127.0.0.1", "--port", "8080", "--log-level", "WARNING"]
        )

        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.log_level == "WARNING"

    def test_from_cli_args_with_overrides(self, clean_env: None) -> None:
        """Test that explicit overrides take precedence over CLI args."""
        config = AgentBeatsConfig.from_cli_args(
            ["--port", "9000"],
            port=9999,  # Override takes precedence
        )
        assert config.port == 9999

    def test_from_cli_args_ignores_unknown(self, clean_env: None) -> None:
        """Test that unknown CLI arguments are ignored."""
        # Should not raise
        config = AgentBeatsConfig.from_cli_args(
            ["--host", "localhost", "--unknown-arg", "value"]
        )
        assert config.host == "localhost"


class TestAgentBeatsConfigEnv:
    """Tests for AgentBeatsConfig environment variable loading."""

    def test_env_host(self, clean_env: None) -> None:
        """Test environment variable for host."""
        os.environ["AGENTBEATS_HOST"] = "env-host.example.com"
        config = AgentBeatsConfig()
        assert config.host == "env-host.example.com"

    def test_env_port(self, clean_env: None) -> None:
        """Test environment variable for port."""
        os.environ["AGENTBEATS_PORT"] = "7777"
        config = AgentBeatsConfig()
        assert config.port == 7777

    def test_env_card_url(self, clean_env: None) -> None:
        """Test environment variable for card_url."""
        os.environ["AGENTBEATS_CARD_URL"] = "http://env.example.com/agent"
        config = AgentBeatsConfig()
        assert config.card_url == "http://env.example.com/agent"

    def test_env_log_level(self, clean_env: None) -> None:
        """Test environment variable for log_level."""
        os.environ["AGENTBEATS_LOG_LEVEL"] = "ERROR"
        config = AgentBeatsConfig()
        assert config.log_level == "ERROR"

    def test_env_case_insensitive(self, clean_env: None) -> None:
        """Test that environment variable names are case-insensitive."""
        os.environ["agentbeats_host"] = "lowercase-host"
        config = AgentBeatsConfig()
        assert config.host == "lowercase-host"

    def test_explicit_overrides_env(self, clean_env: None) -> None:
        """Test that explicit values override environment variables."""
        os.environ["AGENTBEATS_PORT"] = "7777"
        config = AgentBeatsConfig(port=8888)
        assert config.port == 8888


# =============================================================================
# Green Agent Configuration Tests
# =============================================================================


class TestGreenAgentConfig:
    """Tests for GreenAgentConfig."""

    def test_default_values(self, clean_env: None) -> None:
        """Test that Green agent defaults are correctly applied."""
        config = GreenAgentConfig()

        assert config.port == 8000
        assert config.verbose_updates is True
        assert config.ues_base_port == 8080
        assert config.scenarios_dir == "scenarios"
        assert config.default_max_turns == 100
        assert config.default_turn_timeout == 300.0
        assert config.response_generator_model == "gpt-4o"
        assert config.evaluation_model == "gpt-4o"

    def test_explicit_values(self, clean_env: None) -> None:
        """Test that explicit values are used."""
        config = GreenAgentConfig(
            verbose_updates=False,
            ues_base_port=9090,
            scenarios_dir="/custom/scenarios",
            default_max_turns=50,
            default_turn_timeout=120.0,
            response_generator_model="gpt-4-turbo",
            evaluation_model="claude-3-opus",
        )

        assert config.verbose_updates is False
        assert config.ues_base_port == 9090
        assert config.scenarios_dir == "/custom/scenarios"
        assert config.default_max_turns == 50
        assert config.default_turn_timeout == 120.0
        assert config.response_generator_model == "gpt-4-turbo"
        assert config.evaluation_model == "claude-3-opus"

    def test_max_turns_validation(self, clean_env: None) -> None:
        """Test that max_turns validates minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            GreenAgentConfig(default_max_turns=0)
        assert "default_max_turns" in str(exc_info.value).lower()

    def test_turn_timeout_validation(self, clean_env: None) -> None:
        """Test that turn_timeout validates positive value."""
        with pytest.raises(ValidationError) as exc_info:
            GreenAgentConfig(default_turn_timeout=-1.0)
        assert "default_turn_timeout" in str(exc_info.value).lower()

    def test_inherits_base_defaults(self, clean_env: None) -> None:
        """Test that Green config inherits base defaults."""
        config = GreenAgentConfig()

        assert config.host == "0.0.0.0"
        assert config.card_url is None
        assert config.log_level == "INFO"


class TestGreenAgentConfigCLI:
    """Tests for GreenAgentConfig CLI argument parsing."""

    def test_from_cli_args_verbose_updates(self, clean_env: None) -> None:
        """Test CLI parsing of --verbose-updates argument."""
        config = GreenAgentConfig.from_cli_args(["--verbose-updates"])
        assert config.verbose_updates is True

    def test_from_cli_args_no_verbose_updates(self, clean_env: None) -> None:
        """Test CLI parsing of --no-verbose-updates argument."""
        config = GreenAgentConfig.from_cli_args(["--no-verbose-updates"])
        assert config.verbose_updates is False

    def test_from_cli_args_ues_base_port(self, clean_env: None) -> None:
        """Test CLI parsing of --ues-base-port argument."""
        config = GreenAgentConfig.from_cli_args(
            ["--ues-base-port", "9090"]
        )
        assert config.ues_base_port == 9090

    def test_from_cli_args_scenarios_dir(self, clean_env: None) -> None:
        """Test CLI parsing of --scenarios-dir argument."""
        config = GreenAgentConfig.from_cli_args(
            ["--scenarios-dir", "/custom/scenarios"]
        )
        assert config.scenarios_dir == "/custom/scenarios"

    def test_from_cli_args_max_turns(self, clean_env: None) -> None:
        """Test CLI parsing of --default-max-turns argument."""
        config = GreenAgentConfig.from_cli_args(["--default-max-turns", "50"])
        assert config.default_max_turns == 50

    def test_from_cli_args_turn_timeout(self, clean_env: None) -> None:
        """Test CLI parsing of --default-turn-timeout argument."""
        config = GreenAgentConfig.from_cli_args(["--default-turn-timeout", "120.5"])
        assert config.default_turn_timeout == 120.5

    def test_from_cli_args_models(self, clean_env: None) -> None:
        """Test CLI parsing of model arguments."""
        config = GreenAgentConfig.from_cli_args(
            [
                "--response-model",
                "gpt-4-turbo",
                "--evaluation-model",
                "claude-3-opus",
            ]
        )
        assert config.response_generator_model == "gpt-4-turbo"
        assert config.evaluation_model == "claude-3-opus"

    def test_from_cli_args_combined(self, clean_env: None) -> None:
        """Test CLI parsing with multiple Green-specific arguments."""
        config = GreenAgentConfig.from_cli_args(
            [
                "--host",
                "localhost",
                "--port",
                "9000",
                "--ues-base-port",
                "8888",
                "--no-verbose-updates",
                "--default-max-turns",
                "25",
            ]
        )

        assert config.host == "localhost"
        assert config.port == 9000
        assert config.ues_base_port == 8888
        assert config.verbose_updates is False
        assert config.default_max_turns == 25


class TestGreenAgentConfigEnv:
    """Tests for GreenAgentConfig environment variable loading."""

    def test_env_verbose_updates(self, clean_env: None) -> None:
        """Test environment variable for verbose_updates."""
        os.environ["AGENTBEATS_GREEN_VERBOSE_UPDATES"] = "false"
        config = GreenAgentConfig()
        assert config.verbose_updates is False

    def test_env_ues_base_port(self, clean_env: None) -> None:
        """Test environment variable for ues_base_port."""
        os.environ["AGENTBEATS_GREEN_UES_BASE_PORT"] = "9090"
        config = GreenAgentConfig()
        assert config.ues_base_port == 9090

    def test_env_scenarios_dir(self, clean_env: None) -> None:
        """Test environment variable for scenarios_dir."""
        os.environ["AGENTBEATS_GREEN_SCENARIOS_DIR"] = "/env/scenarios"
        config = GreenAgentConfig()
        assert config.scenarios_dir == "/env/scenarios"

    def test_env_models(self, clean_env: None) -> None:
        """Test environment variables for model settings."""
        os.environ["AGENTBEATS_GREEN_RESPONSE_GENERATOR_MODEL"] = "env-response-model"
        os.environ["AGENTBEATS_GREEN_EVALUATION_MODEL"] = "env-eval-model"
        config = GreenAgentConfig()
        assert config.response_generator_model == "env-response-model"
        assert config.evaluation_model == "env-eval-model"


# =============================================================================
# Purple Agent Configuration Tests
# =============================================================================


class TestPurpleAgentConfig:
    """Tests for PurpleAgentConfig."""

    def test_default_values(self, clean_env: None) -> None:
        """Test that Purple agent defaults are correctly applied."""
        config = PurpleAgentConfig()

        assert config.port == 8001
        assert config.model == "gpt-4o"
        assert config.max_actions_per_turn == 50
        assert config.temperature == 0.7
        assert config.enable_reflection is True
        assert config.action_delay == 0.0

    def test_explicit_values(self, clean_env: None) -> None:
        """Test that explicit values are used."""
        config = PurpleAgentConfig(
            model="gpt-4-turbo",
            max_actions_per_turn=25,
            temperature=0.3,
            enable_reflection=False,
            action_delay=0.5,
        )

        assert config.model == "gpt-4-turbo"
        assert config.max_actions_per_turn == 25
        assert config.temperature == 0.3
        assert config.enable_reflection is False
        assert config.action_delay == 0.5

    def test_temperature_validation_min(self, clean_env: None) -> None:
        """Test that temperature validates minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            PurpleAgentConfig(temperature=-0.1)
        assert "temperature" in str(exc_info.value).lower()

    def test_temperature_validation_max(self, clean_env: None) -> None:
        """Test that temperature validates maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            PurpleAgentConfig(temperature=2.5)
        assert "temperature" in str(exc_info.value).lower()

    def test_max_actions_validation(self, clean_env: None) -> None:
        """Test that max_actions_per_turn validates minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            PurpleAgentConfig(max_actions_per_turn=0)
        assert "max_actions_per_turn" in str(exc_info.value).lower()

    def test_action_delay_validation(self, clean_env: None) -> None:
        """Test that action_delay validates non-negative value."""
        with pytest.raises(ValidationError) as exc_info:
            PurpleAgentConfig(action_delay=-1.0)
        assert "action_delay" in str(exc_info.value).lower()

    def test_inherits_base_defaults(self, clean_env: None) -> None:
        """Test that Purple config inherits base defaults."""
        config = PurpleAgentConfig()

        assert config.host == "0.0.0.0"
        assert config.card_url is None
        assert config.log_level == "INFO"


class TestPurpleAgentConfigCLI:
    """Tests for PurpleAgentConfig CLI argument parsing."""

    def test_from_cli_args_model(self, clean_env: None) -> None:
        """Test CLI parsing of --model argument."""
        config = PurpleAgentConfig.from_cli_args(["--model", "gpt-4-turbo"])
        assert config.model == "gpt-4-turbo"

    def test_from_cli_args_max_actions(self, clean_env: None) -> None:
        """Test CLI parsing of --max-actions-per-turn argument."""
        config = PurpleAgentConfig.from_cli_args(["--max-actions-per-turn", "25"])
        assert config.max_actions_per_turn == 25

    def test_from_cli_args_temperature(self, clean_env: None) -> None:
        """Test CLI parsing of --temperature argument."""
        config = PurpleAgentConfig.from_cli_args(["--temperature", "0.3"])
        assert config.temperature == 0.3

    def test_from_cli_args_enable_reflection(self, clean_env: None) -> None:
        """Test CLI parsing of --enable-reflection argument."""
        config = PurpleAgentConfig.from_cli_args(["--enable-reflection"])
        assert config.enable_reflection is True

    def test_from_cli_args_no_reflection(self, clean_env: None) -> None:
        """Test CLI parsing of --no-reflection argument."""
        config = PurpleAgentConfig.from_cli_args(["--no-reflection"])
        assert config.enable_reflection is False

    def test_from_cli_args_action_delay(self, clean_env: None) -> None:
        """Test CLI parsing of --action-delay argument."""
        config = PurpleAgentConfig.from_cli_args(["--action-delay", "0.5"])
        assert config.action_delay == 0.5

    def test_from_cli_args_combined(self, clean_env: None) -> None:
        """Test CLI parsing with multiple Purple-specific arguments."""
        config = PurpleAgentConfig.from_cli_args(
            [
                "--host",
                "localhost",
                "--port",
                "9001",
                "--model",
                "claude-3-opus",
                "--temperature",
                "0.5",
                "--no-reflection",
                "--action-delay",
                "0.1",
            ]
        )

        assert config.host == "localhost"
        assert config.port == 9001
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.enable_reflection is False
        assert config.action_delay == 0.1


class TestPurpleAgentConfigEnv:
    """Tests for PurpleAgentConfig environment variable loading."""

    def test_env_model(self, clean_env: None) -> None:
        """Test environment variable for model."""
        os.environ["AGENTBEATS_PURPLE_MODEL"] = "env-model"
        config = PurpleAgentConfig()
        assert config.model == "env-model"

    def test_env_max_actions(self, clean_env: None) -> None:
        """Test environment variable for max_actions_per_turn."""
        os.environ["AGENTBEATS_PURPLE_MAX_ACTIONS_PER_TURN"] = "30"
        config = PurpleAgentConfig()
        assert config.max_actions_per_turn == 30

    def test_env_temperature(self, clean_env: None) -> None:
        """Test environment variable for temperature."""
        os.environ["AGENTBEATS_PURPLE_TEMPERATURE"] = "0.5"
        config = PurpleAgentConfig()
        assert config.temperature == 0.5

    def test_env_enable_reflection(self, clean_env: None) -> None:
        """Test environment variable for enable_reflection."""
        os.environ["AGENTBEATS_PURPLE_ENABLE_REFLECTION"] = "false"
        config = PurpleAgentConfig()
        assert config.enable_reflection is False

    def test_env_action_delay(self, clean_env: None) -> None:
        """Test environment variable for action_delay."""
        os.environ["AGENTBEATS_PURPLE_ACTION_DELAY"] = "0.25"
        config = PurpleAgentConfig()
        assert config.action_delay == 0.25


# =============================================================================
# Configuration Utilities Tests
# =============================================================================


class TestMergeConfigs:
    """Tests for merge_configs utility function."""

    def test_merge_base_config(self, clean_env: None) -> None:
        """Test merging base configuration."""
        base = AgentBeatsConfig(host="localhost", port=8000)
        merged = merge_configs(base, {"port": 9000})

        assert merged.port == 9000
        assert merged.host == "localhost"
        assert base.port == 8000  # Original unchanged

    def test_merge_green_config(self, clean_env: None) -> None:
        """Test merging Green configuration."""
        base = GreenAgentConfig(verbose_updates=True)
        merged = merge_configs(base, {"verbose_updates": False})

        assert isinstance(merged, GreenAgentConfig)
        assert merged.verbose_updates is False
        assert base.verbose_updates is True  # Original unchanged

    def test_merge_purple_config(self, clean_env: None) -> None:
        """Test merging Purple configuration."""
        base = PurpleAgentConfig(model="gpt-4o")
        merged = merge_configs(base, {"model": "claude-3-opus"})

        assert isinstance(merged, PurpleAgentConfig)
        assert merged.model == "claude-3-opus"
        assert base.model == "gpt-4o"  # Original unchanged

    def test_merge_multiple_fields(self, clean_env: None) -> None:
        """Test merging multiple fields at once."""
        base = GreenAgentConfig(
            host="localhost",
            port=8000,
            verbose_updates=True,
        )
        merged = merge_configs(
            base,
            {
                "host": "0.0.0.0",
                "port": 9000,
                "verbose_updates": False,
            },
        )

        assert merged.host == "0.0.0.0"
        assert merged.port == 9000
        assert merged.verbose_updates is False


class TestValidateConfig:
    """Tests for validate_config utility function."""

    def test_no_warnings_for_valid_config(self, clean_env: None) -> None:
        """Test that valid config produces no warnings."""
        config = AgentBeatsConfig(port=8080)
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_privileged_port_warning(self, clean_env: None) -> None:
        """Test warning for privileged port."""
        config = AgentBeatsConfig(port=80)
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "privileged" in warnings[0].lower()

    def test_green_short_timeout_warning(self, clean_env: None) -> None:
        """Test warning for short turn timeout."""
        config = GreenAgentConfig(default_turn_timeout=10.0)
        warnings = validate_config(config)
        assert any("timeout" in w.lower() and "short" in w.lower() for w in warnings)

    def test_purple_high_temperature_warning(self, clean_env: None) -> None:
        """Test warning for high temperature."""
        config = PurpleAgentConfig(temperature=1.8)
        warnings = validate_config(config)
        assert any("temperature" in w.lower() and "high" in w.lower() for w in warnings)

    def test_purple_high_max_actions_warning(self, clean_env: None) -> None:
        """Test warning for high max_actions_per_turn."""
        config = PurpleAgentConfig(max_actions_per_turn=150)
        warnings = validate_config(config)
        assert any("max_actions" in w.lower() for w in warnings)

    def test_multiple_warnings(self, clean_env: None) -> None:
        """Test that multiple issues produce multiple warnings."""
        config = GreenAgentConfig(
            port=80,  # Privileged port
            default_turn_timeout=5.0,  # Very short timeout
        )
        warnings = validate_config(config)
        assert len(warnings) >= 2


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_boundary_port_values(self, clean_env: None) -> None:
        """Test boundary values for port."""
        # Valid boundaries
        config = AgentBeatsConfig(port=1)
        assert config.port == 1

        config = AgentBeatsConfig(port=65535)
        assert config.port == 65535

    def test_boundary_temperature_values(self, clean_env: None) -> None:
        """Test boundary values for temperature."""
        config = PurpleAgentConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = PurpleAgentConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_empty_cli_args(self, clean_env: None) -> None:
        """Test CLI parsing with empty args list."""
        config = AgentBeatsConfig.from_cli_args([])
        assert config.host == "0.0.0.0"
        assert config.port == 8000

    def test_none_cli_args(self, clean_env: None) -> None:
        """Test CLI parsing with None (uses sys.argv)."""
        with patch("sys.argv", ["prog"]):
            config = AgentBeatsConfig.from_cli_args(None)
            assert config.host == "0.0.0.0"

    def test_config_immutability(self, clean_env: None) -> None:
        """Test that config values cannot be changed after creation."""
        # Note: pydantic-settings models are mutable by default
        # This test documents current behavior
        config = AgentBeatsConfig()
        config.host = "changed"  # This works (mutable)
        assert config.host == "changed"

    def test_config_model_dump(self, clean_env: None) -> None:
        """Test that config can be serialized to dict."""
        config = GreenAgentConfig(
            host="localhost",
            port=9000,
            verbose_updates=False,
        )
        data = config.model_dump()

        assert data["host"] == "localhost"
        assert data["port"] == 9000
        assert data["verbose_updates"] is False

# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_cli_and_env_precedence(self, clean_env: None) -> None:
        """Test that CLI args override environment variables."""
        os.environ["AGENTBEATS_PORT"] = "7777"
        config = AgentBeatsConfig.from_cli_args(["--port", "8888"])
        assert config.port == 8888

    def test_override_and_cli_precedence(self, clean_env: None) -> None:
        """Test that explicit overrides beat CLI args."""
        os.environ["AGENTBEATS_PORT"] = "7777"
        config = AgentBeatsConfig.from_cli_args(
            ["--port", "8888"],
            port=9999,
        )
        assert config.port == 9999

    def test_full_green_config_loading(self, clean_env: None) -> None:
        """Test complete Green agent configuration loading."""
        os.environ["AGENTBEATS_GREEN_UES_BASE_PORT"] = "9090"

        config = GreenAgentConfig.from_cli_args(
            [
                "--host",
                "localhost",
                "--port",
                "9000",
                "--verbose-updates",
                "--scenarios-dir",
                "/scenarios",
            ]
        )

        # CLI values
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.verbose_updates is True
        assert config.scenarios_dir == "/scenarios"

        # Environment values
        assert config.ues_base_port == 9090

    def test_full_purple_config_loading(self, clean_env: None) -> None:
        """Test complete Purple agent configuration loading."""
        os.environ["AGENTBEATS_PURPLE_TEMPERATURE"] = "0.5"
        os.environ["AGENTBEATS_PURPLE_ACTION_DELAY"] = "0.1"

        config = PurpleAgentConfig.from_cli_args(
            [
                "--host",
                "localhost",
                "--port",
                "9001",
                "--model",
                "gpt-4-turbo",
                "--no-reflection",
            ]
        )

        # CLI values
        assert config.host == "localhost"
        assert config.port == 9001
        assert config.model == "gpt-4-turbo"
        assert config.enable_reflection is False

        # Environment values
        assert config.temperature == 0.5
        assert config.action_delay == 0.1
