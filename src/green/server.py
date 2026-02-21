"""Green Agent A2A server entry point.

This module wires together the Green agent components and starts the A2A
HTTP server. It is the main entry point for running the Green agent,
equivalent to ``server.py`` in the AgentBeats green-agent-template.

Usage::

    uv run src/green/server.py --host 0.0.0.0 --port 9009
    uv run src/green/server.py --port 9009 --card-url http://myhost:9009/

The server accepts the standard AgentBeats CLI arguments:
    --host          Host to bind to (default: 0.0.0.0)
    --port          Port to listen on (default: 8000)
    --card-url      URL to advertise in the agent card
    --log-level     Logging level (default: INFO)

Plus Green-specific arguments:
    --ues-base-port Base port for UES server allocation
    --scenarios-dir Directory containing scenario definitions

Environment variables (prefixed with ``AGENTBEATS_GREEN_``) are also
supported. See ``GreenAgentConfig`` for the full list.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.a2a.agent_card import AgentCardBuilder, create_skill
from src.common.a2a.server import A2AServer
from src.common.agentbeats.config import GreenAgentConfig
from src.green.executor import GreenAgentExecutor
from src.green.scenarios.loader import ScenarioManager


def build_agent_card(config: GreenAgentConfig):
    """Build the AgentCard for the Green agent.

    Args:
        config: Green agent configuration (used for URL).

    Returns:
        An AgentCard describing this Green agent's capabilities.
    """
    skill = create_skill(
        id="ues_assessment",
        name="UES Personal Assistant Assessment",
        description=(
            "Evaluates AI personal assistant agents using the User "
            "Environment Simulator (UES). Provides a simulated environment "
            "with email, calendar, SMS, and chat modalities, then scores "
            "the agent on accuracy, efficiency, and appropriateness."
        ),
        tags=[
            "assessment",
            "evaluation",
            "benchmark",
            "personal-assistant",
            "email",
            "calendar",
            "sms",
        ],
        examples=[
            "Evaluate a personal assistant on email triage tasks",
            "Run a calendar management assessment",
        ],
        input_modes=["application/json"],
        output_modes=["application/json"],
    )

    return (
        AgentCardBuilder()
        .with_name("UES Green Agent")
        .with_description(
            "AgentBeats Green Agent that evaluates AI personal assistant "
            "agents using the User Environment Simulator (UES). Provides "
            "realistic email, calendar, SMS, and chat scenarios with "
            "automated scoring."
        )
        .with_url(config.effective_card_url)
        .with_version("0.1.0")
        .with_skill(skill)
        .with_capabilities(streaming=True)
        .with_default_input_modes(["application/json"])
        .with_default_output_modes(["application/json"])
        .build()
    )


def main() -> None:
    """Parse config, build components, and start the A2A server."""
    config = GreenAgentConfig.from_cli_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load scenarios
    scenarios_dir = Path(config.scenarios_dir)
    if not scenarios_dir.exists():
        logger.warning("Scenarios directory does not exist: %s", scenarios_dir)

    scenario_manager = ScenarioManager(scenarios_dir)
    logger.info(
        "Loaded %d scenarios from %s",
        len(scenario_manager.list_scenarios()),
        scenarios_dir,
    )

    # Build executor
    executor = GreenAgentExecutor(
        config=config,
        scenario_manager=scenario_manager,
    )

    # Build agent card
    agent_card = build_agent_card(config)

    # Build and run server
    server = A2AServer(
        agent_card=agent_card,
        executor=executor,
        host=config.host,
        port=config.port,
    )

    logger.info(
        "Starting Green Agent server at http://%s:%d",
        config.host,
        config.port,
    )
    server.run(log_level=config.log_level.lower())


if __name__ == "__main__":
    main()
