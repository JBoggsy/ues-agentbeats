"""Scenario loading utilities for the Green Agent.

This module provides classes for loading and managing assessment scenarios.
Scenarios can be loaded from JSON files, which can include references to
external initial state files.

Key Classes:
    ScenarioNotFoundError: Raised when a scenario cannot be found.
    ScenarioValidationError: Raised when a scenario fails validation.
    EvaluatorLoadError: Raised when evaluator loading fails.
    ScenarioLoader: Low-level scenario loading from files.
    ScenarioManager: High-level scenario management.

Directory Structure:
    Each scenario should be in its own subdirectory with the following structure:
    
    scenarios/
    └── email_triage_basic/
        ├── scenario.json      # Main scenario configuration
        ├── initial_state.json # Optional: UES state (can be embedded in scenario.json)
        └── evaluators.py      # Optional: Programmatic evaluators

Example:
    >>> from pathlib import Path
    >>> from src.green.scenarios import ScenarioManager
    >>> manager = ScenarioManager(Path("scenarios"))
    >>> scenarios = manager.list_scenarios()
    >>> config = manager.get_scenario("email_triage_basic")
    >>> evaluators = manager.get_evaluators("email_triage_basic")
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.green.scenarios.schema import EvaluatorFunc, EvaluatorRegistry, ScenarioConfig


if TYPE_CHECKING:
    from types import ModuleType


# =============================================================================
# Module Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ScenarioNotFoundError(Exception):
    """Raised when a requested scenario cannot be found.

    Attributes:
        scenario_id: The ID of the scenario that was not found.
        scenarios_dir: The directory that was searched.
    """

    def __init__(self, scenario_id: str, scenarios_dir: Path) -> None:
        """Initialize the exception.

        Args:
            scenario_id: The ID of the scenario that was not found.
            scenarios_dir: The directory that was searched.
        """
        self.scenario_id = scenario_id
        self.scenarios_dir = scenarios_dir
        super().__init__(
            f"Scenario '{scenario_id}' not found in {scenarios_dir}. "
            f"Expected directory: {scenarios_dir / scenario_id}"
        )


class ScenarioValidationError(Exception):
    """Raised when a scenario fails validation.

    Attributes:
        scenario_id: The ID of the scenario that failed validation.
        errors: List of validation error messages.
    """

    def __init__(self, scenario_id: str, errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            scenario_id: The ID of the scenario that failed validation.
            errors: List of validation error messages.
        """
        self.scenario_id = scenario_id
        self.errors = errors
        errors_str = "\n  - ".join([""] + errors)
        super().__init__(
            f"Scenario '{scenario_id}' failed validation:{errors_str}"
        )


class EvaluatorLoadError(Exception):
    """Raised when evaluator loading fails.

    This can occur when:
    - The evaluators.py file has syntax errors
    - Required dependencies are missing
    - Evaluator functions have invalid signatures

    Attributes:
        scenario_id: The ID of the scenario whose evaluators failed to load.
        reason: Description of why loading failed.
    """

    def __init__(self, scenario_id: str, reason: str) -> None:
        """Initialize the exception.

        Args:
            scenario_id: The ID of the scenario whose evaluators failed to load.
            reason: Description of why loading failed.
        """
        self.scenario_id = scenario_id
        self.reason = reason
        super().__init__(
            f"Failed to load evaluators for scenario '{scenario_id}': {reason}"
        )


# =============================================================================
# Scenario Loader
# =============================================================================


class ScenarioLoader:
    """Low-level loader for scenario configuration files.

    This class handles the mechanics of loading scenario JSON files and
    resolving references to external files (like initial_state.json).

    The loader supports two modes for initial state:
    1. Embedded: The initial_state is included directly in scenario.json
    2. External: The initial_state field contains a file path reference

    Attributes:
        scenario_dir: Path to the scenario's directory.

    Example:
        >>> loader = ScenarioLoader(Path("scenarios/email_triage_basic"))
        >>> config = loader.load()
    """

    # Standard filenames
    SCENARIO_FILE = "scenario.json"
    INITIAL_STATE_FILE = "initial_state.json"
    EVALUATORS_FILE = "evaluators.py"

    def __init__(self, scenario_dir: Path) -> None:
        """Initialize the loader.

        Args:
            scenario_dir: Path to the scenario's directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
        if not scenario_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {scenario_dir}")
        self.scenario_dir = scenario_dir

    def load(self) -> ScenarioConfig:
        """Load the scenario configuration.

        Loads the scenario.json file and resolves any external file references.
        If initial_state is a string, it's treated as a path relative to the
        scenario directory.

        Returns:
            The loaded and validated ScenarioConfig.

        Raises:
            FileNotFoundError: If scenario.json or referenced files don't exist.
            json.JSONDecodeError: If any JSON file is malformed.
            pydantic.ValidationError: If the configuration is invalid.
        """
        scenario_file = self.scenario_dir / self.SCENARIO_FILE
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

        logger.debug("Loading scenario from %s", scenario_file)

        # Load main scenario file
        data = self._load_json(scenario_file)

        # Resolve initial_state if it's a file reference
        data = self._resolve_initial_state(data)

        # Validate and create ScenarioConfig
        return ScenarioConfig.model_validate(data)

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Load a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            The parsed JSON data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is malformed.
        """
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_initial_state(self, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve the initial_state field if it's a file reference.

        If initial_state is a string, treat it as a relative path and load
        the referenced file. If it's already a dict, return unchanged.

        Args:
            data: The scenario data with potential file reference.

        Returns:
            The data with initial_state resolved to a dict.

        Raises:
            FileNotFoundError: If the referenced file doesn't exist.
            ValueError: If initial_state is neither a string nor a dict.
        """
        initial_state = data.get("initial_state")

        if initial_state is None:
            # Check for default initial_state.json file
            default_file = self.scenario_dir / self.INITIAL_STATE_FILE
            if default_file.exists():
                logger.debug("Loading initial state from default file: %s", default_file)
                data["initial_state"] = self._load_json(default_file)
            else:
                raise FileNotFoundError(
                    f"No initial_state provided and no {self.INITIAL_STATE_FILE} found"
                )
        elif isinstance(initial_state, str):
            # Treat as file path relative to scenario directory
            state_file = self.scenario_dir / initial_state
            if not state_file.exists():
                raise FileNotFoundError(f"Initial state file not found: {state_file}")
            logger.debug("Loading initial state from %s", state_file)
            data["initial_state"] = self._load_json(state_file)
        elif isinstance(initial_state, dict):
            # Already a dict, use as-is
            logger.debug("Using embedded initial state")
        else:
            raise ValueError(
                f"initial_state must be a string (file path) or dict, "
                f"got {type(initial_state).__name__}"
            )

        return data

    def has_evaluators(self) -> bool:
        """Check if the scenario has an evaluators.py file.

        Returns:
            True if evaluators.py exists in the scenario directory.
        """
        return (self.scenario_dir / self.EVALUATORS_FILE).exists()

    def load_evaluators(self) -> EvaluatorRegistry:
        """Load evaluator functions from the scenario's evaluators.py file.

        Dynamically imports the evaluators.py module and extracts all
        async functions that match the EvaluatorFunc signature.

        Returns:
            A dict mapping evaluator_id to evaluator function.

        Raises:
            FileNotFoundError: If evaluators.py doesn't exist.
            EvaluatorLoadError: If the module fails to import or has invalid
                evaluators.

        Example:
            >>> loader = ScenarioLoader(Path("scenarios/email_triage_basic"))
            >>> evaluators = loader.load_evaluators()
            >>> "check_urgent_email_responses" in evaluators
            True
        """
        evaluators_file = self.scenario_dir / self.EVALUATORS_FILE
        if not evaluators_file.exists():
            raise FileNotFoundError(f"Evaluators file not found: {evaluators_file}")

        logger.debug("Loading evaluators from %s", evaluators_file)

        # Create a unique module name to avoid conflicts
        module_name = f"_scenario_evaluators_{self.scenario_dir.name}"

        try:
            module = self._import_module(evaluators_file, module_name)
        except Exception as e:
            raise EvaluatorLoadError(
                self.scenario_dir.name,
                f"Failed to import evaluators.py: {e}"
            ) from e

        # Extract evaluator functions
        evaluators = self._extract_evaluators(module)

        if not evaluators:
            logger.warning(
                "No evaluator functions found in %s. Functions should be "
                "async and have signature (AgentBeatsEvalContext, dict) -> EvalResult",
                evaluators_file,
            )

        return evaluators

    def _import_module(self, module_path: Path, module_name: str) -> ModuleType:
        """Dynamically import a Python module from a file path.

        The module's parent directory is temporarily added to ``sys.path``
        so that sibling modules (e.g. ``ground_truth.py``,
        ``_eval_helpers.py``) can be imported with regular ``import``
        statements from within the evaluators module.

        Args:
            module_path: Path to the .py file.
            module_name: Name to give the module in sys.modules.

        Returns:
            The imported module.

        Raises:
            ImportError: If the module cannot be loaded.
        """
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Add the scenario directory to sys.path so sibling imports work
        scenario_dir_str = str(module_path.parent.resolve())
        path_added = False
        if scenario_dir_str not in sys.path:
            sys.path.insert(0, scenario_dir_str)
            path_added = True

        try:
            spec.loader.exec_module(module)
        except Exception:
            # Clean up sys.modules on failure
            sys.modules.pop(module_name, None)
            raise
        finally:
            # Clean up sys.path to avoid polluting the import namespace
            if path_added:
                try:
                    sys.path.remove(scenario_dir_str)
                except ValueError:
                    pass

        return module

    def _extract_evaluators(self, module: ModuleType) -> EvaluatorRegistry:
        """Extract evaluator functions from a module.

        An evaluator function must be:
        - Async (defined with 'async def')
        - Have at least 2 parameters (ctx, params)
        - Not start with underscore (private functions are excluded)

        Args:
            module: The imported module to extract from.

        Returns:
            A dict mapping function name to function.
        """
        evaluators: EvaluatorRegistry = {}

        for name in dir(module):
            # Skip private/dunder attributes
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Must be a callable (function)
            if not callable(obj):
                continue

            # Must be an async function (coroutine function)
            if not asyncio.iscoroutinefunction(obj):
                continue

            # Check signature has at least 2 parameters
            try:
                sig = inspect.signature(obj)
                params = list(sig.parameters.values())
                if len(params) < 2:
                    logger.debug(
                        "Skipping %s: expected at least 2 parameters, got %d",
                        name,
                        len(params),
                    )
                    continue
            except (ValueError, TypeError):
                logger.debug("Skipping %s: cannot inspect signature", name)
                continue

            evaluators[name] = obj
            logger.debug("Found evaluator function: %s", name)

        return evaluators


# =============================================================================
# Scenario Manager
# =============================================================================


class ScenarioManager:
    """High-level manager for assessment scenarios.

    The ScenarioManager provides a convenient interface for discovering,
    loading, and validating scenarios. It maintains a cache of loaded
    scenarios and their evaluators to avoid repeated file I/O.

    Attributes:
        scenarios_dir: Base directory containing scenario subdirectories.

    Example:
        >>> manager = ScenarioManager(Path("scenarios"))
        >>> scenarios = manager.list_scenarios()
        ['email_triage_basic', 'calendar_scheduling']
        >>> config = manager.get_scenario("email_triage_basic")
        >>> evaluators = manager.get_evaluators("email_triage_basic")
        >>> warnings = manager.validate_scenario(config)
    """

    def __init__(self, scenarios_dir: Path) -> None:
        """Initialize the manager.

        Args:
            scenarios_dir: Base directory containing scenario subdirectories.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not scenarios_dir.exists():
            raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
        if not scenarios_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {scenarios_dir}")
        self.scenarios_dir = scenarios_dir
        self._scenario_cache: dict[str, ScenarioConfig] = {}
        self._evaluator_cache: dict[str, EvaluatorRegistry] = {}
        logger.debug("ScenarioManager initialized with dir: %s", scenarios_dir)

    def list_scenarios(self) -> list[str]:
        """List all available scenario IDs.

        Scans the scenarios directory for subdirectories containing a
        scenario.json file.

        Returns:
            List of scenario IDs (directory names) sorted alphabetically.

        Example:
            >>> manager = ScenarioManager(Path("scenarios"))
            >>> manager.list_scenarios()
            ['calendar_scheduling', 'email_triage_basic']
        """
        scenarios: list[str] = []
        for path in self.scenarios_dir.iterdir():
            if path.is_dir() and (path / ScenarioLoader.SCENARIO_FILE).exists():
                scenarios.append(path.name)
        scenarios.sort()
        logger.debug("Found %d scenarios", len(scenarios))
        return scenarios

    def load_scenario(
        self,
        scenario_id: str,
        *,
        use_cache: bool = True,
    ) -> ScenarioConfig:
        """Load a scenario by ID.

        Args:
            scenario_id: The scenario identifier (directory name).
            use_cache: Whether to use cached scenario if available.

        Returns:
            The loaded ScenarioConfig.

        Raises:
            ScenarioNotFoundError: If the scenario doesn't exist.
            ScenarioValidationError: If the scenario fails validation.

        Example:
            >>> config = manager.load_scenario("email_triage_basic")
            >>> config.scenario_id
            'email_triage_basic'
        """
        # Check cache
        if use_cache and scenario_id in self._scenario_cache:
            logger.debug("Returning cached scenario: %s", scenario_id)
            return self._scenario_cache[scenario_id]

        # Check directory exists
        scenario_dir = self.scenarios_dir / scenario_id
        if not scenario_dir.exists():
            raise ScenarioNotFoundError(scenario_id, self.scenarios_dir)

        # Load scenario
        try:
            loader = ScenarioLoader(scenario_dir)
            config = loader.load()
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            raise ScenarioValidationError(scenario_id, [str(e)]) from e

        # Validate scenario_id matches directory name
        if config.scenario_id != scenario_id:
            raise ScenarioValidationError(
                scenario_id,
                [
                    f"scenario_id '{config.scenario_id}' does not match "
                    f"directory name '{scenario_id}'"
                ],
            )

        # Cache and return
        self._scenario_cache[scenario_id] = config
        logger.info("Loaded scenario: %s", scenario_id)
        return config

    def validate_scenario(self, config: ScenarioConfig) -> list[str]:
        """Validate a scenario and return any warnings.

        This method performs additional validation beyond the Pydantic model
        validation. It checks for potential issues that may not cause errors
        but could indicate configuration problems.

        Args:
            config: The ScenarioConfig to validate.

        Returns:
            List of warning messages (empty if no issues found).

        Example:
            >>> warnings = manager.validate_scenario(config)
            >>> if warnings:
            ...     for w in warnings:
            ...         print(f"Warning: {w}")
        """
        warnings: list[str] = []

        # Check scenario duration
        duration = config.duration
        if duration.total_seconds() < 3600:  # Less than 1 hour
            warnings.append(
                f"Scenario duration ({duration}) is very short (< 1 hour)"
            )
        if duration.total_seconds() > 86400 * 7:  # More than 1 week
            warnings.append(
                f"Scenario duration ({duration}) is very long (> 1 week)"
            )

        # Check time step vs duration
        time_step = config.default_time_step_timedelta
        if time_step >= duration:
            warnings.append(
                f"default_time_step ({config.default_time_step}) is >= scenario duration"
            )

        # Check for characters without any criteria that reference them
        character_emails = {
            c.email for c in config.characters.values() if c.email
        }
        character_phones = {
            c.phone for c in config.characters.values() if c.phone
        }
        if not character_emails and not character_phones:
            warnings.append("No characters have contact methods")

        # Check criteria coverage across dimensions
        dimensions_covered = {c.dimension for c in config.criteria}
        all_dimensions = {"accuracy", "instruction_following", "efficiency", "safety", "politeness"}
        missing = all_dimensions - dimensions_covered
        if missing:
            warnings.append(
                f"No criteria for dimensions: {', '.join(sorted(missing))}"
            )

        # Check for very high or very low max scores
        total_max = config.get_total_max_score()
        if total_max < 5:
            warnings.append(
                f"Total max score ({total_max}) is very low"
            )
        if total_max > 1000:
            warnings.append(
                f"Total max score ({total_max}) is very high"
            )

        # Check for empty initial_state
        if not config.initial_state:
            warnings.append("initial_state is empty")

        # Check user_prompt length
        if len(config.user_prompt) < 20:
            warnings.append(
                f"user_prompt is very short ({len(config.user_prompt)} chars)"
            )

        return warnings

    def clear_cache(self) -> None:
        """Clear the scenario and evaluator caches.

        Call this method if scenarios may have been modified on disk and
        you want to reload them.
        """
        self._scenario_cache.clear()
        self._evaluator_cache.clear()
        logger.debug("Scenario and evaluator caches cleared")

    def get_cached_scenarios(self) -> list[str]:
        """Get the IDs of all cached scenarios.

        Returns:
            List of scenario IDs currently in the cache.
        """
        return list(self._scenario_cache.keys())

    def reload_scenario(self, scenario_id: str) -> ScenarioConfig:
        """Reload a scenario from disk, bypassing the cache.

        Args:
            scenario_id: The scenario identifier to reload.

        Returns:
            The freshly loaded ScenarioConfig.

        Raises:
            ScenarioNotFoundError: If the scenario doesn't exist.
            ScenarioValidationError: If the scenario fails validation.
        """
        return self.load_scenario(scenario_id, use_cache=False)

    # Alias for clearer API
    get_scenario = load_scenario

    def has_evaluators(self, scenario_id: str) -> bool:
        """Check if a scenario has an evaluators.py file.

        Args:
            scenario_id: The scenario identifier (directory name).

        Returns:
            True if the scenario has an evaluators.py file.

        Raises:
            ScenarioNotFoundError: If the scenario doesn't exist.
        """
        scenario_dir = self.scenarios_dir / scenario_id
        if not scenario_dir.exists():
            raise ScenarioNotFoundError(scenario_id, self.scenarios_dir)

        loader = ScenarioLoader(scenario_dir)
        return loader.has_evaluators()

    def load_evaluators(
        self,
        scenario_id: str,
        *,
        use_cache: bool = True,
    ) -> EvaluatorRegistry:
        """Load evaluators for a scenario by ID.

        Args:
            scenario_id: The scenario identifier (directory name).
            use_cache: Whether to use cached evaluators if available.

        Returns:
            A dict mapping evaluator_id to evaluator function.
            Returns empty dict if no evaluators.py file exists.

        Raises:
            ScenarioNotFoundError: If the scenario doesn't exist.
            EvaluatorLoadError: If the evaluators.py file fails to load.

        Example:
            >>> evaluators = manager.load_evaluators("email_triage_basic")
            >>> "check_urgent_email_responses" in evaluators
            True
        """
        # Check cache
        if use_cache and scenario_id in self._evaluator_cache:
            logger.debug("Returning cached evaluators: %s", scenario_id)
            return self._evaluator_cache[scenario_id]

        # Check directory exists
        scenario_dir = self.scenarios_dir / scenario_id
        if not scenario_dir.exists():
            raise ScenarioNotFoundError(scenario_id, self.scenarios_dir)

        # Load evaluators (return empty dict if no evaluators.py)
        loader = ScenarioLoader(scenario_dir)
        if not loader.has_evaluators():
            logger.debug("No evaluators.py found for scenario: %s", scenario_id)
            evaluators: EvaluatorRegistry = {}
        else:
            try:
                evaluators = loader.load_evaluators()
            except FileNotFoundError:
                # Shouldn't happen since we checked, but handle gracefully
                evaluators = {}

        # Cache and return
        self._evaluator_cache[scenario_id] = evaluators
        logger.info(
            "Loaded %d evaluators for scenario: %s",
            len(evaluators),
            scenario_id,
        )
        return evaluators

    # Alias for clearer API
    get_evaluators = load_evaluators

    def reload_evaluators(self, scenario_id: str) -> EvaluatorRegistry:
        """Reload evaluators from disk, bypassing the cache.

        Args:
            scenario_id: The scenario identifier to reload.

        Returns:
            The freshly loaded evaluators.

        Raises:
            ScenarioNotFoundError: If the scenario doesn't exist.
            EvaluatorLoadError: If the evaluators.py file fails to load.
        """
        return self.load_evaluators(scenario_id, use_cache=False)

    def get_cached_evaluators(self) -> list[str]:
        """Get the IDs of all scenarios with cached evaluators.

        Returns:
            List of scenario IDs with evaluators in the cache.
        """
        return list(self._evaluator_cache.keys())

    def validate_evaluators(
        self,
        scenario_config: ScenarioConfig,
        evaluators: EvaluatorRegistry,
    ) -> list[str]:
        """Validate that all required evaluators are present.

        Checks that every criterion with an evaluator_id has a corresponding
        function in the evaluator registry.

        Args:
            scenario_config: The scenario configuration.
            evaluators: The loaded evaluators.

        Returns:
            List of warning messages (empty if no issues found).

        Example:
            >>> config = manager.get_scenario("email_triage_basic")
            >>> evaluators = manager.get_evaluators("email_triage_basic")
            >>> warnings = manager.validate_evaluators(config, evaluators)
        """
        warnings: list[str] = []

        for criterion in scenario_config.criteria:
            if criterion.evaluator_id:
                if criterion.evaluator_id not in evaluators:
                    warnings.append(
                        f"Criterion '{criterion.criterion_id}' references "
                        f"evaluator '{criterion.evaluator_id}' which was not found"
                    )

        return warnings
