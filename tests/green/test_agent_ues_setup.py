"""Tests for GreenAgent._setup_ues method.

Tests cover:
- Clearing UES state before scenario import
- Importing scenario initial state via HTTP
- Starting simulation with auto_advance=False
- Error handling for failed scenario imports
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
                response_timing=ResponseTiming(base_delay="PT15M", variance="PT5M"),
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


@pytest.fixture
def mock_green_agent() -> GreenAgent:
    """Create a GreenAgent with mocked internals for testing _setup_ues."""
    # Create a mock agent using object.__new__ to bypass __init__
    agent = object.__new__(GreenAgent)

    # Set required attributes that _setup_ues uses
    agent._ues_port = 8100
    agent._proctor_api_key = "test-proctor-key"

    # Mock the UES client
    agent.ues_client = MagicMock()
    agent.ues_client.simulation = MagicMock()
    agent.ues_client.simulation.clear = AsyncMock()
    agent.ues_client.simulation.start = AsyncMock()

    return agent


# =============================================================================
# Tests
# =============================================================================


class TestSetupUES:
    """Tests for GreenAgent._setup_ues method."""

    @pytest.mark.asyncio
    async def test_clears_ues_state(self, mock_green_agent: GreenAgent) -> None:
        """Test that _setup_ues clears UES state before importing."""
        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.simulation.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_imports_scenario_via_http(self, mock_green_agent: GreenAgent) -> None:
        """Test that _setup_ues imports scenario via POST to /scenario/import/full."""
        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await mock_green_agent._setup_ues(scenario)

        # Verify POST was called with correct URL and payload
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:8100/scenario/import/full",
            headers={"X-API-Key": "test-proctor-key"},
            json={"scenario": scenario.initial_state},
            timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_starts_simulation_with_auto_advance_false(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues starts simulation with auto_advance=False."""
        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await mock_green_agent._setup_ues(scenario)

        mock_green_agent.ues_client.simulation.start.assert_called_once_with(
            auto_advance=False
        )

    @pytest.mark.asyncio
    async def test_operations_called_in_correct_order(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that clear, import, and start are called in the correct order."""
        scenario = make_scenario_config()
        call_order: list[str] = []

        # Track call order
        original_clear = mock_green_agent.ues_client.simulation.clear

        async def track_clear() -> None:
            call_order.append("clear")
            return await original_clear()

        mock_green_agent.ues_client.simulation.clear = AsyncMock(side_effect=track_clear)

        original_start = mock_green_agent.ues_client.simulation.start

        async def track_start(**kwargs: Any) -> None:
            call_order.append("start")
            return await original_start(**kwargs)

        mock_green_agent.ues_client.simulation.start = AsyncMock(side_effect=track_start)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            async def track_post(*args: Any, **kwargs: Any) -> MagicMock:
                call_order.append("import")
                return mock_response

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=track_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await mock_green_agent._setup_ues(scenario)

        assert call_order == ["clear", "import", "start"]

    @pytest.mark.asyncio
    async def test_raises_on_import_http_error(
        self, mock_green_agent: GreenAgent
    ) -> None:
        """Test that _setup_ues raises HTTPStatusError on import failure."""
        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server Error",
                    request=MagicMock(),
                    response=mock_response,
                )
            )

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await mock_green_agent._setup_ues(scenario)

    @pytest.mark.asyncio
    async def test_uses_correct_port_in_url(self) -> None:
        """Test that _setup_ues uses the agent's port in the import URL."""
        # Create agent with different port
        agent = object.__new__(GreenAgent)
        agent._ues_port = 9999
        agent._proctor_api_key = "test-key"
        agent.ues_client = MagicMock()
        agent.ues_client.simulation = MagicMock()
        agent.ues_client.simulation.clear = AsyncMock()
        agent.ues_client.simulation.start = AsyncMock()

        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await agent._setup_ues(scenario)

        # Verify the URL uses port 9999
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:9999/scenario/import/full"

    @pytest.mark.asyncio
    async def test_uses_proctor_api_key_in_header(self) -> None:
        """Test that _setup_ues uses the proctor API key in the request header."""
        agent = object.__new__(GreenAgent)
        agent._ues_port = 8100
        agent._proctor_api_key = "my-special-proctor-key"
        agent.ues_client = MagicMock()
        agent.ues_client.simulation = MagicMock()
        agent.ues_client.simulation.clear = AsyncMock()
        agent.ues_client.simulation.start = AsyncMock()

        scenario = make_scenario_config()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            await agent._setup_ues(scenario)

        # Verify the header contains the correct API key
        call_args = mock_client.post.call_args
        assert call_args[1]["headers"]["X-API-Key"] == "my-special-proctor-key"
