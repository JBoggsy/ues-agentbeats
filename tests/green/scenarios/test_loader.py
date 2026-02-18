"""Tests for scenario loader utilities.

Tests cover:
- ScenarioLoader: Loading from JSON files, external state files
- ScenarioManager: Listing, loading, caching, validation
- Error handling: Missing files, invalid JSON, validation errors
- Edge cases: Empty directories, malformed data
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src.green.scenarios.loader import (
    ScenarioLoader,
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
)
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    ResponseTiming,
    ScenarioConfig,
)


# =============================================================================
# Fixtures - Test Data
# =============================================================================


@pytest.fixture
def sample_character_data() -> dict[str, Any]:
    """Create sample character data for JSON files."""
    return {
        "name": "Alice Chen",
        "relationships": {"Alex Thompson": "direct report"},
        "personality": "Professional but friendly. Prefers concise communication.",
        "email": "alice.chen@company.com",
        "response_timing": {
            "base_delay": "PT30M",
            "variance": "PT10M",
        },
    }


@pytest.fixture
def sample_criterion_data() -> dict[str, Any]:
    """Create sample criterion data for JSON files."""
    return {
        "criterion_id": "email_response_accuracy",
        "name": "Email Response Accuracy",
        "description": "Evaluates whether sent emails address the original query.",
        "dimension": "accuracy",
        "max_score": 10,
        "evaluator_id": "check_email_response_content",
    }


@pytest.fixture
def sample_initial_state() -> dict[str, Any]:
    """Create sample initial state data."""
    return {
        "email": {
            "folders": [
                {"name": "Inbox", "unread_count": 5},
                {"name": "Sent", "unread_count": 0},
            ]
        },
        "calendar": {"events": []},
        "sms": {"conversations": []},
    }


@pytest.fixture
def sample_scenario_data(
    sample_character_data: dict[str, Any],
    sample_criterion_data: dict[str, Any],
    sample_initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Create sample scenario data for JSON files."""
    return {
        "scenario_id": "email_triage_basic",
        "name": "Basic Email Triage",
        "description": "Handle incoming emails appropriately.",
        "start_time": "2026-01-28T09:00:00Z",
        "end_time": "2026-01-28T17:00:00Z",
        "default_time_step": "PT1H",
        "user_prompt": "Please triage my inbox and respond to urgent emails.",
        "user_character": "alice",
        "characters": {"alice": sample_character_data},
        "initial_state": sample_initial_state,
        "criteria": [sample_criterion_data],
    }


# =============================================================================
# Fixtures - Directory Setup
# =============================================================================


@pytest.fixture
def scenarios_dir(tmp_path: Path) -> Path:
    """Create a temporary scenarios directory."""
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    return scenarios


@pytest.fixture
def scenario_with_embedded_state(
    scenarios_dir: Path,
    sample_scenario_data: dict[str, Any],
) -> Path:
    """Create a scenario with embedded initial_state."""
    scenario_dir = scenarios_dir / "embedded_state"
    scenario_dir.mkdir()

    scenario_file = scenario_dir / "scenario.json"
    scenario_file.write_text(json.dumps(sample_scenario_data, indent=2))

    return scenario_dir


@pytest.fixture
def scenario_with_external_state(
    scenarios_dir: Path,
    sample_scenario_data: dict[str, Any],
    sample_initial_state: dict[str, Any],
) -> Path:
    """Create a scenario with external initial_state.json file."""
    scenario_dir = scenarios_dir / "external_state"
    scenario_dir.mkdir()

    # Create scenario.json without initial_state
    data_without_state = {**sample_scenario_data}
    del data_without_state["initial_state"]
    scenario_file = scenario_dir / "scenario.json"
    scenario_file.write_text(json.dumps(data_without_state, indent=2))

    # Create initial_state.json
    state_file = scenario_dir / "initial_state.json"
    state_file.write_text(json.dumps(sample_initial_state, indent=2))

    return scenario_dir


@pytest.fixture
def scenario_with_state_reference(
    scenarios_dir: Path,
    sample_scenario_data: dict[str, Any],
    sample_initial_state: dict[str, Any],
) -> Path:
    """Create a scenario with initial_state as a file path reference."""
    scenario_dir = scenarios_dir / "state_reference"
    scenario_dir.mkdir()

    # Create subdirectory for state file
    data_dir = scenario_dir / "data"
    data_dir.mkdir()

    # Create scenario.json with path reference
    data_with_reference = {**sample_scenario_data}
    data_with_reference["initial_state"] = "data/environment.json"
    scenario_file = scenario_dir / "scenario.json"
    scenario_file.write_text(json.dumps(data_with_reference, indent=2))

    # Create referenced state file
    state_file = data_dir / "environment.json"
    state_file.write_text(json.dumps(sample_initial_state, indent=2))

    return scenario_dir


@pytest.fixture
def multiple_scenarios(
    scenarios_dir: Path,
    sample_scenario_data: dict[str, Any],
) -> Path:
    """Create multiple scenarios for listing tests."""
    for name in ["alpha_scenario", "beta_scenario", "gamma_scenario"]:
        scenario_dir = scenarios_dir / name
        scenario_dir.mkdir()

        data = {**sample_scenario_data, "scenario_id": name}
        scenario_file = scenario_dir / "scenario.json"
        scenario_file.write_text(json.dumps(data, indent=2))

    return scenarios_dir


# =============================================================================
# ScenarioNotFoundError Tests
# =============================================================================


class TestScenarioNotFoundError:
    """Tests for ScenarioNotFoundError exception."""

    def test_error_attributes(self, scenarios_dir: Path):
        """Test that error has correct attributes."""
        error = ScenarioNotFoundError("missing_scenario", scenarios_dir)
        assert error.scenario_id == "missing_scenario"
        assert error.scenarios_dir == scenarios_dir

    def test_error_message(self, scenarios_dir: Path):
        """Test error message format."""
        error = ScenarioNotFoundError("missing_scenario", scenarios_dir)
        message = str(error)
        assert "missing_scenario" in message
        assert str(scenarios_dir) in message


class TestScenarioValidationError:
    """Tests for ScenarioValidationError exception."""

    def test_error_attributes(self):
        """Test that error has correct attributes."""
        errors = ["Error 1", "Error 2"]
        error = ScenarioValidationError("bad_scenario", errors)
        assert error.scenario_id == "bad_scenario"
        assert error.errors == errors

    def test_error_message(self):
        """Test error message format."""
        errors = ["Missing field", "Invalid value"]
        error = ScenarioValidationError("bad_scenario", errors)
        message = str(error)
        assert "bad_scenario" in message
        assert "Missing field" in message
        assert "Invalid value" in message


# =============================================================================
# ScenarioLoader Tests
# =============================================================================


class TestScenarioLoader:
    """Tests for ScenarioLoader class."""

    def test_init_valid_directory(self, scenario_with_embedded_state: Path):
        """Test initialization with valid directory."""
        loader = ScenarioLoader(scenario_with_embedded_state)
        assert loader.scenario_dir == scenario_with_embedded_state

    def test_init_missing_directory_raises(self, tmp_path: Path):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ScenarioLoader(tmp_path / "nonexistent")

    def test_init_file_instead_of_directory_raises(self, tmp_path: Path):
        """Test that passing a file raises NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        with pytest.raises(NotADirectoryError):
            ScenarioLoader(file_path)

    def test_load_embedded_state(self, scenario_with_embedded_state: Path):
        """Test loading scenario with embedded initial_state."""
        loader = ScenarioLoader(scenario_with_embedded_state)
        config = loader.load()

        assert isinstance(config, ScenarioConfig)
        assert config.scenario_id == "email_triage_basic"
        assert "email" in config.initial_state

    def test_load_external_state_default_file(
        self, scenario_with_external_state: Path
    ):
        """Test loading scenario with default initial_state.json file."""
        loader = ScenarioLoader(scenario_with_external_state)
        config = loader.load()

        assert isinstance(config, ScenarioConfig)
        assert "email" in config.initial_state

    def test_load_state_reference(self, scenario_with_state_reference: Path):
        """Test loading scenario with initial_state as file path."""
        loader = ScenarioLoader(scenario_with_state_reference)
        config = loader.load()

        assert isinstance(config, ScenarioConfig)
        assert "email" in config.initial_state

    def test_load_missing_scenario_file_raises(self, tmp_path: Path):
        """Test that missing scenario.json raises FileNotFoundError."""
        scenario_dir = tmp_path / "empty_scenario"
        scenario_dir.mkdir()

        loader = ScenarioLoader(scenario_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        assert "scenario.json" in str(exc_info.value)

    def test_load_missing_state_file_raises(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test that missing referenced state file raises FileNotFoundError."""
        scenario_dir = scenarios_dir / "missing_state"
        scenario_dir.mkdir()

        # Create scenario.json with path reference to non-existent file
        data = {**sample_scenario_data}
        data["initial_state"] = "nonexistent.json"
        scenario_file = scenario_dir / "scenario.json"
        scenario_file.write_text(json.dumps(data, indent=2))

        loader = ScenarioLoader(scenario_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        assert "nonexistent.json" in str(exc_info.value)

    def test_load_no_state_and_no_default_raises(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test that missing initial_state and no default file raises."""
        scenario_dir = scenarios_dir / "no_state"
        scenario_dir.mkdir()

        # Create scenario.json without initial_state
        data = {**sample_scenario_data}
        del data["initial_state"]
        scenario_file = scenario_dir / "scenario.json"
        scenario_file.write_text(json.dumps(data, indent=2))

        loader = ScenarioLoader(scenario_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        assert "initial_state.json" in str(exc_info.value)

    def test_load_invalid_json_raises(self, scenarios_dir: Path):
        """Test that malformed JSON raises JSONDecodeError."""
        scenario_dir = scenarios_dir / "bad_json"
        scenario_dir.mkdir()

        scenario_file = scenario_dir / "scenario.json"
        scenario_file.write_text("{ invalid json }")

        loader = ScenarioLoader(scenario_dir)
        with pytest.raises(json.JSONDecodeError):
            loader.load()

    def test_load_invalid_initial_state_type_raises(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test that invalid initial_state type raises ValueError."""
        scenario_dir = scenarios_dir / "bad_state_type"
        scenario_dir.mkdir()

        # Create scenario.json with invalid initial_state type
        data = {**sample_scenario_data}
        data["initial_state"] = 12345  # Not a string or dict
        scenario_file = scenario_dir / "scenario.json"
        scenario_file.write_text(json.dumps(data, indent=2))

        loader = ScenarioLoader(scenario_dir)
        with pytest.raises(ValueError) as exc_info:
            loader.load()
        assert "must be a string (file path) or dict" in str(exc_info.value)


# =============================================================================
# ScenarioManager Tests
# =============================================================================


class TestScenarioManager:
    """Tests for ScenarioManager class."""

    def test_init_valid_directory(self, scenarios_dir: Path):
        """Test initialization with valid directory."""
        manager = ScenarioManager(scenarios_dir)
        assert manager.scenarios_dir == scenarios_dir

    def test_init_missing_directory_raises(self, tmp_path: Path):
        """Test that missing directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ScenarioManager(tmp_path / "nonexistent")

    def test_init_file_instead_of_directory_raises(self, tmp_path: Path):
        """Test that passing a file raises NotADirectoryError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        with pytest.raises(NotADirectoryError):
            ScenarioManager(file_path)

    # -------------------------------------------------------------------------
    # list_scenarios tests
    # -------------------------------------------------------------------------

    def test_list_scenarios_empty(self, scenarios_dir: Path):
        """Test listing scenarios in empty directory."""
        manager = ScenarioManager(scenarios_dir)
        assert manager.list_scenarios() == []

    def test_list_scenarios_multiple(self, multiple_scenarios: Path):
        """Test listing multiple scenarios."""
        manager = ScenarioManager(multiple_scenarios)
        scenarios = manager.list_scenarios()

        assert len(scenarios) == 3
        assert scenarios == ["alpha_scenario", "beta_scenario", "gamma_scenario"]

    def test_list_scenarios_ignores_non_scenario_dirs(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test that directories without scenario.json are ignored."""
        # Create valid scenario
        valid_dir = scenarios_dir / "valid_scenario"
        valid_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "valid_scenario"}
        (valid_dir / "scenario.json").write_text(json.dumps(data))

        # Create directory without scenario.json
        invalid_dir = scenarios_dir / "not_a_scenario"
        invalid_dir.mkdir()

        # Create file (not directory)
        (scenarios_dir / "some_file.txt").write_text("test")

        manager = ScenarioManager(scenarios_dir)
        scenarios = manager.list_scenarios()

        assert scenarios == ["valid_scenario"]

    # -------------------------------------------------------------------------
    # load_scenario tests
    # -------------------------------------------------------------------------

    def test_load_scenario_valid(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test loading a valid scenario."""
        scenario_dir = scenarios_dir / "test_scenario"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "test_scenario"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("test_scenario")

        assert isinstance(config, ScenarioConfig)
        assert config.scenario_id == "test_scenario"

    def test_load_scenario_not_found_raises(self, scenarios_dir: Path):
        """Test that loading non-existent scenario raises ScenarioNotFoundError."""
        manager = ScenarioManager(scenarios_dir)

        with pytest.raises(ScenarioNotFoundError) as exc_info:
            manager.load_scenario("nonexistent")

        assert exc_info.value.scenario_id == "nonexistent"

    def test_load_scenario_id_mismatch_raises(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test that scenario_id must match directory name."""
        scenario_dir = scenarios_dir / "directory_name"
        scenario_dir.mkdir()
        # Use different scenario_id than directory name
        data = {**sample_scenario_data, "scenario_id": "different_id"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        with pytest.raises(ScenarioValidationError) as exc_info:
            manager.load_scenario("directory_name")

        assert "does not match directory name" in str(exc_info.value)

    def test_load_scenario_invalid_data_raises(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test that invalid scenario data raises ScenarioValidationError."""
        scenario_dir = scenarios_dir / "invalid_scenario"
        scenario_dir.mkdir()
        # Missing required fields
        data = {"scenario_id": "invalid_scenario", "name": "Test"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        with pytest.raises(ScenarioValidationError):
            manager.load_scenario("invalid_scenario")

    # -------------------------------------------------------------------------
    # Caching tests
    # -------------------------------------------------------------------------

    def test_load_scenario_uses_cache(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test that scenarios are cached after loading."""
        scenario_dir = scenarios_dir / "cached_scenario"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "cached_scenario"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        # Load scenario
        config1 = manager.load_scenario("cached_scenario")

        # Verify it's cached
        assert "cached_scenario" in manager.get_cached_scenarios()

        # Load again - should return same object
        config2 = manager.load_scenario("cached_scenario")
        assert config1 is config2

    def test_load_scenario_bypass_cache(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test loading scenario with use_cache=False."""
        scenario_dir = scenarios_dir / "bypass_cache"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "bypass_cache"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        # Load scenario
        config1 = manager.load_scenario("bypass_cache")

        # Load again bypassing cache - should return different object
        config2 = manager.load_scenario("bypass_cache", use_cache=False)
        assert config1 is not config2
        assert config1 == config2  # But equal content

    def test_clear_cache(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test clearing the scenario cache."""
        scenario_dir = scenarios_dir / "clear_cache"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "clear_cache"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        # Load and cache
        manager.load_scenario("clear_cache")
        assert len(manager.get_cached_scenarios()) == 1

        # Clear cache
        manager.clear_cache()
        assert len(manager.get_cached_scenarios()) == 0

    def test_reload_scenario(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test reloading a scenario from disk."""
        scenario_dir = scenarios_dir / "reload_test"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "reload_test"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)

        # Load scenario
        config1 = manager.load_scenario("reload_test")
        assert "reload_test" in manager.get_cached_scenarios()

        # Reload scenario
        config2 = manager.reload_scenario("reload_test")
        assert config1 is not config2
        assert config1 == config2

    # -------------------------------------------------------------------------
    # validate_scenario tests
    # -------------------------------------------------------------------------

    def test_validate_scenario_no_warnings(
        self, scenarios_dir: Path, sample_scenario_data: dict[str, Any]
    ):
        """Test validation with well-formed scenario (minimal warnings)."""
        # Add criteria for multiple dimensions to avoid missing dimension warning
        criteria = [
            sample_scenario_data["criteria"][0],  # accuracy
            {
                "criterion_id": "instruction_test",
                "name": "Instructions",
                "description": "Test.",
                "dimension": "instruction_following",
                "max_score": 10,
                "evaluator_id": "test",
            },
            {
                "criterion_id": "efficiency_test",
                "name": "Efficiency",
                "description": "Test.",
                "dimension": "efficiency",
                "max_score": 10,
                "evaluator_id": "test",
            },
            {
                "criterion_id": "safety_test",
                "name": "Safety",
                "description": "Test.",
                "dimension": "safety",
                "max_score": 10,
                "evaluator_id": "test",
            },
            {
                "criterion_id": "politeness_test",
                "name": "Politeness",
                "description": "Test.",
                "dimension": "politeness",
                "max_score": 10,
                "evaluator_id": "test",
            },
        ]
        data = {**sample_scenario_data, "criteria": criteria}

        scenario_dir = scenarios_dir / "valid_test"
        scenario_dir.mkdir()
        data["scenario_id"] = "valid_test"
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("valid_test")
        warnings = manager.validate_scenario(config)

        assert len(warnings) == 0

    def test_validate_scenario_short_duration_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for very short scenario duration."""
        data = {
            **sample_scenario_data,
            "scenario_id": "short_duration",
            "start_time": "2026-01-28T09:00:00Z",
            "end_time": "2026-01-28T09:30:00Z",  # Only 30 minutes
        }
        scenario_dir = scenarios_dir / "short_duration"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("short_duration")
        warnings = manager.validate_scenario(config)

        assert any("very short" in w for w in warnings)

    def test_validate_scenario_long_duration_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for very long scenario duration."""
        data = {
            **sample_scenario_data,
            "scenario_id": "long_duration",
            "start_time": "2026-01-01T09:00:00Z",
            "end_time": "2026-01-15T09:00:00Z",  # 14 days
        }
        scenario_dir = scenarios_dir / "long_duration"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("long_duration")
        warnings = manager.validate_scenario(config)

        assert any("very long" in w for w in warnings)

    def test_validate_scenario_time_step_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning when time step >= duration."""
        data = {
            **sample_scenario_data,
            "scenario_id": "big_step",
            "start_time": "2026-01-28T09:00:00Z",
            "end_time": "2026-01-28T10:00:00Z",  # 1 hour
            "default_time_step": "P1D",  # 1 day step
        }
        scenario_dir = scenarios_dir / "big_step"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("big_step")
        warnings = manager.validate_scenario(config)

        assert any(">= scenario duration" in w for w in warnings)

    def test_validate_scenario_missing_dimensions_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for missing scoring dimensions."""
        # Only has accuracy dimension
        scenario_dir = scenarios_dir / "missing_dims"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "missing_dims"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("missing_dims")
        warnings = manager.validate_scenario(config)

        assert any("No criteria for dimensions" in w for w in warnings)

    def test_validate_scenario_low_max_score_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for very low total max score."""
        data = {
            **sample_scenario_data,
            "scenario_id": "low_score",
            "criteria": [
                {
                    "criterion_id": "tiny",
                    "name": "Tiny",
                    "description": "Test.",
                    "dimension": "accuracy",
                    "max_score": 1,  # Very low
                    "evaluator_id": "test",
                }
            ],
        }
        scenario_dir = scenarios_dir / "low_score"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("low_score")
        warnings = manager.validate_scenario(config)

        assert any("very low" in w for w in warnings)

    def test_validate_scenario_empty_initial_state_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for empty initial_state."""
        data = {
            **sample_scenario_data,
            "scenario_id": "empty_state",
            "initial_state": {},  # Empty
        }
        scenario_dir = scenarios_dir / "empty_state"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("empty_state")
        warnings = manager.validate_scenario(config)

        assert any("initial_state is empty" in w for w in warnings)

    def test_validate_scenario_short_prompt_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test warning for very short user_prompt."""
        data = {
            **sample_scenario_data,
            "scenario_id": "short_prompt",
            "user_prompt": "Test",  # Very short
        }
        scenario_dir = scenarios_dir / "short_prompt"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("short_prompt")
        warnings = manager.validate_scenario(config)

        assert any("user_prompt is very short" in w for w in warnings)


# =============================================================================
# Integration Tests
# =============================================================================


class TestScenarioManagerIntegration:
    """Integration tests for ScenarioManager with complex scenarios."""

    def test_load_scenario_with_multiple_characters(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
        sample_character_data: dict[str, Any],
    ):
        """Test loading scenario with multiple characters."""
        characters = {
            "alice": sample_character_data,
            "bob": {
                **sample_character_data,
                "name": "Bob Smith",
                "email": "bob.smith@company.com",
            },
            "carol": {
                "name": "Carol Davis",
                "relationships": {"Alex Thompson": "close friend"},
                "personality": "Casual.",
                "phone": "+1-555-123-4567",
                "response_timing": {"base_delay": "PT5M", "variance": "PT2M"},
            },
        }
        data = {
            **sample_scenario_data,
            "scenario_id": "multi_char",
            "characters": characters,
        }
        scenario_dir = scenarios_dir / "multi_char"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("multi_char")

        assert len(config.characters) == 3
        assert "alice" in config.characters
        assert "bob" in config.characters
        assert "carol" in config.characters

    def test_load_scenario_with_multiple_criteria(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test loading scenario with multiple criteria."""
        criteria = [
            {
                "criterion_id": "accuracy_1",
                "name": "Accuracy 1",
                "description": "Test.",
                "dimension": "accuracy",
                "max_score": 10,
                "evaluator_id": "test_accuracy",
            },
            {
                "criterion_id": "efficiency_1",
                "name": "Efficiency 1",
                "description": "Test.",
                "dimension": "efficiency",
                "max_score": 15,
                "evaluation_prompt": "Rate efficiency.",
            },
            {
                "criterion_id": "safety_1",
                "name": "Safety 1",
                "description": "Test.",
                "dimension": "safety",
                "max_score": 20,
                "evaluator_id": "check_safety",
                "params": {"forbidden_words": ["delete", "destroy"]},
            },
        ]
        data = {
            **sample_scenario_data,
            "scenario_id": "multi_criteria",
            "criteria": criteria,
        }
        scenario_dir = scenarios_dir / "multi_criteria"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("multi_criteria")

        assert len(config.criteria) == 3
        assert config.get_total_max_score() == 45
        scores = config.get_max_score_by_dimension()
        assert scores["accuracy"] == 10
        assert scores["efficiency"] == 15
        assert scores["safety"] == 20

    def test_load_scenario_with_early_completion_conditions(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test loading scenario with early completion conditions."""
        data = {
            **sample_scenario_data,
            "scenario_id": "early_complete",
            "early_completion_conditions": [
                "inbox_empty",
                "all_urgent_emails_handled",
            ],
        }
        scenario_dir = scenarios_dir / "early_complete"
        scenario_dir.mkdir()
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("early_complete")

        assert config.early_completion_conditions is not None
        assert len(config.early_completion_conditions) == 2
        assert "inbox_empty" in config.early_completion_conditions

    def test_roundtrip_json_serialization(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test that loaded scenarios can be serialized back to JSON."""
        scenario_dir = scenarios_dir / "roundtrip"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "roundtrip"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.load_scenario("roundtrip")

        # Serialize to JSON
        json_data = config.model_dump(mode="json")

        # Write to new file
        output_dir = scenarios_dir / "roundtrip_output"
        output_dir.mkdir()
        output_file = output_dir / "scenario.json"
        output_data = {**json_data, "scenario_id": "roundtrip_output"}
        output_file.write_text(json.dumps(output_data, indent=2))

        # Load from new file
        manager2 = ScenarioManager(scenarios_dir)
        config2 = manager2.load_scenario("roundtrip_output")

        # Compare (excluding scenario_id which we changed)
        assert config.name == config2.name
        assert config.description == config2.description
        assert config.user_prompt == config2.user_prompt
        assert len(config.characters) == len(config2.characters)
        assert len(config.criteria) == len(config2.criteria)


# =============================================================================
# EvaluatorLoadError Tests
# =============================================================================


class TestEvaluatorLoadError:
    """Tests for EvaluatorLoadError exception."""

    def test_error_attributes(self):
        """Test that error has correct attributes."""
        from src.green.scenarios.loader import EvaluatorLoadError

        error = EvaluatorLoadError("test_scenario", "Module import failed")
        assert error.scenario_id == "test_scenario"
        assert error.reason == "Module import failed"

    def test_error_message(self):
        """Test error message format."""
        from src.green.scenarios.loader import EvaluatorLoadError

        error = EvaluatorLoadError("test_scenario", "Syntax error at line 5")
        message = str(error)
        assert "test_scenario" in message
        assert "Syntax error at line 5" in message


# =============================================================================
# ScenarioLoader Evaluator Tests
# =============================================================================


class TestScenarioLoaderEvaluators:
    """Tests for ScenarioLoader evaluator loading functionality."""

    @pytest.fixture
    def scenario_with_evaluators(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ) -> Path:
        """Create a scenario with evaluators.py file."""
        scenario_dir = scenarios_dir / "with_evaluators"
        scenario_dir.mkdir()

        # Create scenario.json
        data = {**sample_scenario_data, "scenario_id": "with_evaluators"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        # Create evaluators.py
        evaluators_code = '''
"""Test evaluators."""
from src.green.scenarios.schema import AgentBeatsEvalContext, EvalResult

async def check_email_response_content(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
    """Check email response content."""
    return EvalResult(score=10, max_score=10, explanation="All good")

async def check_response_timing(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
    """Check response timing."""
    return EvalResult(score=8, max_score=10, explanation="Mostly timely")

def sync_helper_function(x: int, y: int) -> int:
    """A sync helper that should NOT be loaded as evaluator."""
    return x + y

def _private_function(ctx, params):
    """Private function that should NOT be loaded."""
    pass

async def _private_async(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
    """Private async function that should NOT be loaded."""
    return EvalResult(score=0, max_score=10, explanation="Should not be found")
'''
        (scenario_dir / "evaluators.py").write_text(evaluators_code)

        return scenario_dir

    @pytest.fixture
    def scenario_without_evaluators(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ) -> Path:
        """Create a scenario without evaluators.py file."""
        scenario_dir = scenarios_dir / "no_evaluators"
        scenario_dir.mkdir()

        data = {**sample_scenario_data, "scenario_id": "no_evaluators"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        return scenario_dir

    @pytest.fixture
    def scenario_with_invalid_evaluators(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ) -> Path:
        """Create a scenario with invalid evaluators.py (syntax error)."""
        scenario_dir = scenarios_dir / "invalid_evaluators"
        scenario_dir.mkdir()

        data = {**sample_scenario_data, "scenario_id": "invalid_evaluators"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        # Create invalid evaluators.py (syntax error)
        (scenario_dir / "evaluators.py").write_text("def broken(\n")

        return scenario_dir

    def test_has_evaluators_true(self, scenario_with_evaluators: Path):
        """Test has_evaluators returns True when file exists."""
        loader = ScenarioLoader(scenario_with_evaluators)
        assert loader.has_evaluators() is True

    def test_has_evaluators_false(self, scenario_without_evaluators: Path):
        """Test has_evaluators returns False when file doesn't exist."""
        loader = ScenarioLoader(scenario_without_evaluators)
        assert loader.has_evaluators() is False

    def test_load_evaluators_success(self, scenario_with_evaluators: Path):
        """Test successful loading of evaluators."""
        loader = ScenarioLoader(scenario_with_evaluators)
        evaluators = loader.load_evaluators()

        # Should find the two public async functions
        assert "check_email_response_content" in evaluators
        assert "check_response_timing" in evaluators
        assert len(evaluators) == 2

    def test_load_evaluators_excludes_sync_functions(
        self, scenario_with_evaluators: Path
    ):
        """Test that sync functions are not loaded as evaluators."""
        loader = ScenarioLoader(scenario_with_evaluators)
        evaluators = loader.load_evaluators()

        assert "sync_helper_function" not in evaluators

    def test_load_evaluators_excludes_private_functions(
        self, scenario_with_evaluators: Path
    ):
        """Test that private functions are not loaded as evaluators."""
        loader = ScenarioLoader(scenario_with_evaluators)
        evaluators = loader.load_evaluators()

        assert "_private_function" not in evaluators
        assert "_private_async" not in evaluators

    def test_load_evaluators_callable(self, scenario_with_evaluators: Path):
        """Test that loaded evaluators are callable."""
        import asyncio

        loader = ScenarioLoader(scenario_with_evaluators)
        evaluators = loader.load_evaluators()

        # Check they're callable
        for name, func in evaluators.items():
            assert callable(func), f"{name} should be callable"
            assert asyncio.iscoroutinefunction(func), f"{name} should be async"

    def test_load_evaluators_missing_file_raises(
        self, scenario_without_evaluators: Path
    ):
        """Test that loading evaluators when file doesn't exist raises error."""
        loader = ScenarioLoader(scenario_without_evaluators)

        with pytest.raises(FileNotFoundError):
            loader.load_evaluators()

    def test_load_evaluators_invalid_syntax_raises(
        self, scenario_with_invalid_evaluators: Path
    ):
        """Test that invalid Python syntax raises EvaluatorLoadError."""
        from src.green.scenarios.loader import EvaluatorLoadError

        loader = ScenarioLoader(scenario_with_invalid_evaluators)

        with pytest.raises(EvaluatorLoadError) as exc_info:
            loader.load_evaluators()

        assert "invalid_evaluators" in str(exc_info.value)


# =============================================================================
# ScenarioManager Evaluator Tests
# =============================================================================


class TestScenarioManagerEvaluators:
    """Tests for ScenarioManager evaluator functionality."""

    @pytest.fixture
    def manager_with_evaluators(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ) -> ScenarioManager:
        """Create a ScenarioManager with scenarios having evaluators."""
        # Scenario with evaluators
        scenario_dir = scenarios_dir / "with_evals"
        scenario_dir.mkdir()
        data = {**sample_scenario_data, "scenario_id": "with_evals"}
        (scenario_dir / "scenario.json").write_text(json.dumps(data))
        (scenario_dir / "evaluators.py").write_text('''
from src.green.scenarios.schema import AgentBeatsEvalContext, EvalResult

async def check_email_response_content(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
    return EvalResult(score=10, max_score=10, explanation="Pass")
''')

        # Scenario without evaluators
        scenario_dir2 = scenarios_dir / "no_evals"
        scenario_dir2.mkdir()
        data2 = {**sample_scenario_data, "scenario_id": "no_evals"}
        (scenario_dir2 / "scenario.json").write_text(json.dumps(data2))

        return ScenarioManager(scenarios_dir)

    def test_has_evaluators(self, manager_with_evaluators: ScenarioManager):
        """Test has_evaluators method."""
        assert manager_with_evaluators.has_evaluators("with_evals") is True
        assert manager_with_evaluators.has_evaluators("no_evals") is False

    def test_has_evaluators_not_found_raises(
        self, manager_with_evaluators: ScenarioManager
    ):
        """Test has_evaluators raises for missing scenario."""
        with pytest.raises(ScenarioNotFoundError):
            manager_with_evaluators.has_evaluators("nonexistent")

    def test_load_evaluators_success(self, manager_with_evaluators: ScenarioManager):
        """Test successful loading of evaluators."""
        evaluators = manager_with_evaluators.load_evaluators("with_evals")
        assert "check_email_response_content" in evaluators

    def test_load_evaluators_empty_for_no_file(
        self, manager_with_evaluators: ScenarioManager
    ):
        """Test that load_evaluators returns empty dict when no file."""
        evaluators = manager_with_evaluators.load_evaluators("no_evals")
        assert evaluators == {}

    def test_load_evaluators_not_found_raises(
        self, manager_with_evaluators: ScenarioManager
    ):
        """Test load_evaluators raises for missing scenario."""
        with pytest.raises(ScenarioNotFoundError):
            manager_with_evaluators.load_evaluators("nonexistent")

    def test_get_evaluators_alias(self, manager_with_evaluators: ScenarioManager):
        """Test get_evaluators is an alias for load_evaluators."""
        evaluators1 = manager_with_evaluators.load_evaluators("with_evals")
        evaluators2 = manager_with_evaluators.get_evaluators("with_evals")
        assert evaluators1 == evaluators2

    def test_evaluator_caching(self, manager_with_evaluators: ScenarioManager):
        """Test that evaluators are cached."""
        # First load
        evaluators1 = manager_with_evaluators.load_evaluators("with_evals")
        assert "with_evals" in manager_with_evaluators.get_cached_evaluators()

        # Second load returns cached
        evaluators2 = manager_with_evaluators.load_evaluators("with_evals")
        assert evaluators1 is evaluators2  # Same object

    def test_evaluator_cache_bypass(self, manager_with_evaluators: ScenarioManager):
        """Test bypassing the evaluator cache."""
        evaluators1 = manager_with_evaluators.load_evaluators("with_evals")
        evaluators2 = manager_with_evaluators.load_evaluators(
            "with_evals", use_cache=False
        )
        # Not the same object due to reload
        assert evaluators1 is not evaluators2
        # But same content
        assert set(evaluators1.keys()) == set(evaluators2.keys())

    def test_reload_evaluators(self, manager_with_evaluators: ScenarioManager):
        """Test reload_evaluators bypasses cache."""
        evaluators1 = manager_with_evaluators.load_evaluators("with_evals")
        evaluators2 = manager_with_evaluators.reload_evaluators("with_evals")
        assert evaluators1 is not evaluators2

    def test_clear_cache_clears_evaluators(
        self, manager_with_evaluators: ScenarioManager
    ):
        """Test that clear_cache clears both scenario and evaluator caches."""
        manager_with_evaluators.load_scenario("with_evals")
        manager_with_evaluators.load_evaluators("with_evals")

        assert len(manager_with_evaluators.get_cached_scenarios()) > 0
        assert len(manager_with_evaluators.get_cached_evaluators()) > 0

        manager_with_evaluators.clear_cache()

        assert manager_with_evaluators.get_cached_scenarios() == []
        assert manager_with_evaluators.get_cached_evaluators() == []

    def test_validate_evaluators_no_warnings(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test validate_evaluators with all evaluators present."""
        # Create scenario with matching evaluators
        scenario_dir = scenarios_dir / "complete"
        scenario_dir.mkdir()

        criterion = {
            "criterion_id": "test_criterion",
            "name": "Test",
            "description": "Test criterion",
            "dimension": "accuracy",
            "max_score": 10,
            "evaluator_id": "my_evaluator",
        }
        data = {
            **sample_scenario_data,
            "scenario_id": "complete",
            "criteria": [criterion],
        }
        (scenario_dir / "scenario.json").write_text(json.dumps(data))
        (scenario_dir / "evaluators.py").write_text('''
from src.green.scenarios.schema import AgentBeatsEvalContext, EvalResult

async def my_evaluator(ctx: AgentBeatsEvalContext, params: dict) -> EvalResult:
    return EvalResult(score=10, max_score=10, explanation="OK")
''')

        manager = ScenarioManager(scenarios_dir)
        config = manager.get_scenario("complete")
        evaluators = manager.get_evaluators("complete")

        warnings = manager.validate_evaluators(config, evaluators)
        assert warnings == []

    def test_validate_evaluators_missing_evaluator(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test validate_evaluators warns about missing evaluators."""
        scenario_dir = scenarios_dir / "incomplete"
        scenario_dir.mkdir()

        criterion = {
            "criterion_id": "test_criterion",
            "name": "Test",
            "description": "Test criterion",
            "dimension": "accuracy",
            "max_score": 10,
            "evaluator_id": "missing_evaluator",
        }
        data = {
            **sample_scenario_data,
            "scenario_id": "incomplete",
            "criteria": [criterion],
        }
        (scenario_dir / "scenario.json").write_text(json.dumps(data))
        # No evaluators.py

        manager = ScenarioManager(scenarios_dir)
        config = manager.get_scenario("incomplete")
        evaluators = manager.get_evaluators("incomplete")

        warnings = manager.validate_evaluators(config, evaluators)
        assert len(warnings) == 1
        assert "missing_evaluator" in warnings[0]

    def test_validate_evaluators_llm_criterion_no_warning(
        self,
        scenarios_dir: Path,
        sample_scenario_data: dict[str, Any],
    ):
        """Test validate_evaluators ignores criteria with evaluation_prompt."""
        scenario_dir = scenarios_dir / "llm_only"
        scenario_dir.mkdir()

        criterion = {
            "criterion_id": "llm_criterion",
            "name": "LLM Test",
            "description": "Test criterion using LLM",
            "dimension": "accuracy",
            "max_score": 10,
            "evaluation_prompt": "Evaluate the response quality.",
        }
        data = {
            **sample_scenario_data,
            "scenario_id": "llm_only",
            "criteria": [criterion],
        }
        (scenario_dir / "scenario.json").write_text(json.dumps(data))

        manager = ScenarioManager(scenarios_dir)
        config = manager.get_scenario("llm_only")
        evaluators = manager.get_evaluators("llm_only")

        warnings = manager.validate_evaluators(config, evaluators)
        assert warnings == []


# =============================================================================
# Integration Tests with Real Scenario
# =============================================================================


class TestEvaluatorIntegration:
    """Integration tests using the real email_triage_basic scenario."""

    @pytest.fixture
    def real_scenarios_dir(self) -> Path:
        """Get the real scenarios directory."""
        return Path(__file__).parent.parent.parent.parent / "scenarios"

    def test_load_real_scenario_evaluators(self, real_scenarios_dir: Path):
        """Test loading evaluators from the real email_triage_basic scenario."""
        if not real_scenarios_dir.exists():
            pytest.skip("scenarios directory not found")

        manager = ScenarioManager(real_scenarios_dir)

        if not manager.has_evaluators("email_triage_basic"):
            pytest.skip("email_triage_basic evaluators.py not found")

        evaluators = manager.get_evaluators("email_triage_basic")
        assert len(evaluators) >= 8  # All 8 programmatic evaluators

        # Check expected evaluators match scenario.json criteria
        expected = {
            "noise_exclusion",
            "summary_accuracy",
            "urgency_accuracy",
            "thread_tracking",
            "hourly_summary_delivery",
            "action_economy",
            "timely_processing",
            "no_unauthorized_sends",
        }
        for name in expected:
            assert name in evaluators, f"Missing evaluator: {name}"

    def test_validate_real_scenario_evaluators(self, real_scenarios_dir: Path):
        """Test that real scenario has all required evaluators."""
        if not real_scenarios_dir.exists():
            pytest.skip("scenarios directory not found")

        manager = ScenarioManager(real_scenarios_dir)

        if "email_triage_basic" not in manager.list_scenarios():
            pytest.skip("email_triage_basic scenario not found")

        config = manager.get_scenario("email_triage_basic")
        evaluators = manager.get_evaluators("email_triage_basic")

        warnings = manager.validate_evaluators(config, evaluators)
        assert warnings == [], f"Missing evaluators: {warnings}"

