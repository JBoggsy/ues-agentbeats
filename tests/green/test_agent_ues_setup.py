"""Tests for GreenAgent._setup_ues method.

Tests cover:
- Clearing UES state before scenario import
- Importing scenario initial state via client library
- Starting simulation with auto_advance=False
- Error handling for failed scenario imports
- Correct operation ordering
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.green.agent import GreenAgent
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    ResponseTiming,
    ScenarioConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


def make_minimal_initial_state() -> dict[str, Any]:
    """Create a minimal valid UES initial state for testing."""
    return {
        "metadata": {
            "ues_version": "0.1.0",
            "scenario_version": "1",
            "created_at": "2026-01-28T09:00:00+00:00",
        },
        "environment": {
            "time_state": {
                "current_time": "2026-01-28T09:00:00+00:00",
                "time_scale": 1.0,
                "is_paused": True,
                "auto_advance": False,
                "last_wall_time_update": "2026-01-28T09:00:00+00:00",
            },
            "modality_states": {
                "email": {
                    "modality_type": "email",
                    "last_updated": "2026-01-28T09:00:00+00:00",
                    "update_count": 0,
                    "user_email_address": "user@example.com",
                    "emails": {},
                    "threads": {},
                    "folders": {},
                    "labels": {},
                    "drafts": {},
                },
            },
        },
        "events": {"events": []},
    }


def make_scenario_config() -> ScenarioConfig:
    """Create a minimal ScenarioConfig for testing."""
    return ScenarioConfig(
        scenario_id="test_scenario",
        name="Test Scenario",
        description="A test scenario for unit tests.",
        start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
        default_time_step="PT1H",
        user_prompt="Please handle the test task.",
        user_character="alex",
        characters={
            "alex": CharacterProfile(
                name="Alex Thompson",
                personality="Professional and detail-oriented.",
                email="alex@example.com",
                response_timing=ResponseTiming(
                    base_delay="PT15M", variance="PT5M"
                ),
            ),
        },
        initial_state=make_minimal_initial_state(),
        criteria=[
            EvaluationCriterion(
                criterion_id="test_criterion",
                name="Test Criterion",
                description="A test criterion.",
                dimension="accuracy",
                max_score=10,
                evaluation_prompt="Did the agent complete the test task?",
            ),
        ],
    )


def _make_mock_agent(
    ues_port: int = 8100,
    proctor_key: str = "test-proctor-key",
) -> GreenAgent:
    """Create a GreenAgent with mocked internals for testing _setup_ues."""
    agent = object.__new__(GreenAgent)
    agent._ues_port = ues_port
    agent._proctor_api_key = proctor_key

    # Mock the UES client with all sub-clients as AsyncMocks
    agent.ues_client = AsyncMock()
    agent.ues_client.simulation.clear = AsyncMock()
    agent.ues_client.simulation.start = AsyncMock()
    agent.ues_client.scenario.import_full = AsyncMock()

    return agent


@pytest.fixture
def mock_green_agent() -> GreenAgent:
    """Create a GreenAgent with mocked internals for testing _setup_ues."""
    return _make_mock_agent()


# =============================================================================
# Tests
# =============================================================================


class TestSetupUES:
    """Tests for GreenAgent._setup_ues method."""

    @pytest.mark.asyncio
    async def test_clears_ues_state(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues clears UES state before importing."""
        scenario = make_scenario_config()

        await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.simulation.clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_imports_scenario_via_client(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues imports scenario via client library."""
        scenario = make_scenario_config()

        await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.scenario.import_full.assert_awaited_once_with(
            scenario=scenario.initial_state,
            strict_modalities=False,
        )

    @pytest.mark.asyncio
    async def test_starts_simulation_with_auto_advance_false(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues starts simulation with auto_advance=False."""
        scenario = make_scenario_config()

        await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.simulation.start.assert_awaited_once_with(
            auto_advance=False
        )

    @pytest.mark.asyncio
    async def test_operations_called_in_correct_order(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that clear, import, and start are called in correct order."""
        scenario = make_scenario_config()
        call_order: list[str] = []

        async def track_clear() -> None:
            call_order.append("clear")

        async def track_import(**kwargs: Any) -> None:
            call_order.append("import")

        async def track_start(**kwargs: Any) -> None:
            call_order.append("start")

        mock_green_agent.ues_client.simulation.clear = AsyncMock(
            side_effect=track_clear
        )
        mock_green_agent.ues_client.scenario.import_full = AsyncMock(
            side_effect=track_import
        )
        mock_green_agent.ues_client.simulation.start = AsyncMock(
            side_effect=track_start
        )

        await mock_green_agent._setup_ues(scenario)

        assert call_order == ["clear", "import", "start"]

    @pytest.mark.asyncio
    async def test_raises_on_import_error(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues propagates errors from import_full."""
        scenario = make_scenario_config()

        mock_green_agent.ues_client.scenario.import_full = AsyncMock(
            side_effect=Exception("Import failed")
        )

        with pytest.raises(Exception, match="Import failed"):
            await mock_green_agent._setup_ues(scenario)

    @pytest.mark.asyncio
    async def test_does_not_start_if_import_fails(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that simulation is not started if import fails."""
        scenario = make_scenario_config()

        mock_green_agent.ues_client.scenario.import_full = AsyncMock(
            side_effect=Exception("Import failed")
        )

        with pytest.raises(Exception):
            await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.simulation.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_initial_state_to_import(self) -> None:
        """Test that initial_state from scenario is passed to import_full."""
        custom_state = {
            "metadata": {
                "ues_version": "0.2.1",
                "scenario_version": "42",
                "created_at": "2026-06-15T12:00:00+00:00",
            },
            "environment": {
                "time_state": {
                    "current_time": "2026-06-15T12:00:00+00:00",
                    "time_scale": 2.0,
                    "is_paused": False,
                    "auto_advance": False,
                    "last_wall_time_update": "2026-06-15T12:00:00+00:00",
                },
                "modality_states": {
                    "email": {"modality_type": "email", "emails": {}},
                },
            },
            "events": {"events": [{"event_id": "e1"}]},
        }

        agent = _make_mock_agent()
        scenario = ScenarioConfig(
            scenario_id="custom_state_test",
            name="Custom State Test",
            description="Tests that custom initial_state is forwarded.",
            start_time=datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc),
            default_time_step="PT30M",
            user_prompt="Do the custom task.",
            user_character="alex",
            characters={
                "alex": CharacterProfile(
                    name="Alex",
                    personality="Tester.",
                    email="alex@test.com",
                    response_timing=ResponseTiming(
                        base_delay="PT5M", variance="PT1M"
                    ),
                ),
            },
            initial_state=custom_state,
            criteria=[
                EvaluationCriterion(
                    criterion_id="c1",
                    name="Custom Criterion",
                    description="Test.",
                    dimension="accuracy",
                    max_score=5,
                    evaluation_prompt="Pass?",
                ),
            ],
        )

        await agent._setup_ues(scenario)

        agent.ues_client.scenario.import_full.assert_awaited_once_with(
            scenario=custom_state,
            strict_modalities=False,
        )
