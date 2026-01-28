"""Tests for server.py."""

import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard

from src.common.a2a.agent_card import AgentCardBuilder, create_skill
from src.common.a2a.server import A2AServer


class MockExecutor(AgentExecutor):
    """Mock executor for testing."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle execute request."""
        pass

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle cancel request."""
        pass


def create_test_card() -> AgentCard:
    """Create a test agent card."""
    return (
        AgentCardBuilder()
        .with_name("Test Agent")
        .with_description("A test agent")
        .with_url("http://localhost:8000")
        .with_skill(create_skill("test", "Test", "A test skill"))
        .build()
    )


class TestA2AServer:
    """Tests for the A2AServer class."""

    def test_server_initialization(self):
        """Test basic server initialization."""
        card = create_test_card()
        executor = MockExecutor()

        server = A2AServer(
            agent_card=card,
            executor=executor,
        )

        assert server.agent_card is card
        assert server.executor is executor
        assert server.host == "0.0.0.0"
        assert server.port == 8000

    def test_server_custom_host_port(self):
        """Test server with custom host and port."""
        card = create_test_card()
        executor = MockExecutor()

        server = A2AServer(
            agent_card=card,
            executor=executor,
            host="127.0.0.1",
            port=9000,
        )

        assert server.host == "127.0.0.1"
        assert server.port == 9000

    def test_server_with_extended_card(self):
        """Test server with extended agent card."""
        public_card = create_test_card()
        extended_card = (
            AgentCardBuilder()
            .with_name("Test Agent - Extended")
            .with_description("Extended test agent")
            .with_url("http://localhost:8000")
            .with_skill(create_skill("test", "Test", "A test skill"))
            .with_skill(create_skill("extended", "Extended", "An extended skill"))
            .build()
        )
        executor = MockExecutor()

        server = A2AServer(
            agent_card=public_card,
            executor=executor,
            extended_agent_card=extended_card,
        )

        assert server.extended_agent_card is extended_card

    def test_get_asgi_app(self):
        """Test getting the ASGI application."""
        card = create_test_card()
        executor = MockExecutor()
        server = A2AServer(agent_card=card, executor=executor)

        app = server.get_asgi_app()

        # Should return a Starlette-compatible application
        assert app is not None
        assert callable(app)

    def test_app_property_cached(self):
        """Test that the app property caches the application."""
        card = create_test_card()
        executor = MockExecutor()
        server = A2AServer(agent_card=card, executor=executor)

        app1 = server.app
        app2 = server.app

        assert app1 is app2

    def test_server_creates_default_task_store(self):
        """Test that server creates a default task store if none provided."""
        card = create_test_card()
        executor = MockExecutor()
        server = A2AServer(agent_card=card, executor=executor)

        assert server.task_store is not None

    def test_server_with_custom_task_store(self):
        """Test server with a custom task store."""
        from a2a.server.tasks import InMemoryTaskStore

        card = create_test_card()
        executor = MockExecutor()
        custom_store = InMemoryTaskStore()

        server = A2AServer(
            agent_card=card,
            executor=executor,
            task_store=custom_store,
        )

        assert server.task_store is custom_store
