"""A2A Server utilities.

This module provides a simplified wrapper around the A2A SDK server components
with sensible defaults for setting up A2A-compliant HTTP servers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskStore

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.types import AgentCard

logger = logging.getLogger(__name__)


class A2AServer:
    """Simplified A2A server setup.

    This class wraps the A2A SDK's Starlette application with sensible defaults
    for easy server configuration and startup.

    Example:
        >>> from a2a.server.agent_execution import AgentExecutor
        >>> from a2a.server.events import EventQueue
        >>> from a2a.server.context import RequestContext
        >>>
        >>> class MyExecutor(AgentExecutor):
        ...     async def execute(
        ...         self, context: RequestContext, event_queue: EventQueue
        ...     ) -> None:
        ...         # Handle the request
        ...         pass
        ...
        ...     async def cancel(
        ...         self, context: RequestContext, event_queue: EventQueue
        ...     ) -> None:
        ...         pass
        >>>
        >>> server = A2AServer(
        ...     agent_card=my_card,
        ...     executor=MyExecutor(),
        ...     host="0.0.0.0",
        ...     port=8000,
        ... )
        >>> server.run()  # Blocking
    """

    def __init__(
        self,
        agent_card: AgentCard,
        executor: AgentExecutor,
        host: str = "0.0.0.0",
        port: int = 8000,
        task_store: TaskStore | None = None,
        extended_agent_card: AgentCard | None = None,
    ) -> None:
        """Initialize the A2A server.

        Args:
            agent_card: The public AgentCard describing this agent.
            executor: The AgentExecutor that handles incoming requests.
            host: Host address to bind to (default: "0.0.0.0").
            port: Port number to listen on (default: 8000).
            task_store: Optional task store for managing tasks. If not provided,
                an InMemoryTaskStore will be used.
            extended_agent_card: Optional extended AgentCard for authenticated
                requests with additional capabilities.
        """
        self.agent_card = agent_card
        self.executor = executor
        self.host = host
        self.port = port
        self.task_store = task_store or InMemoryTaskStore()
        self.extended_agent_card = extended_agent_card

        self._app: A2AStarletteApplication | None = None

    def _build_app(self) -> A2AStarletteApplication:
        """Build the Starlette application.

        Returns:
            The configured A2AStarletteApplication.
        """
        request_handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
        )

        return A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=request_handler,
            extended_agent_card=self.extended_agent_card,
        )

    @property
    def app(self) -> A2AStarletteApplication:
        """Get or build the Starlette application.

        Returns:
            The A2AStarletteApplication instance.
        """
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def run(self, log_level: str = "info") -> None:
        """Run the server (blocking).

        This method starts the server and blocks until it's stopped.

        Args:
            log_level: Logging level for uvicorn (default: "info").
        """
        logger.info(f"Starting A2A server at http://{self.host}:{self.port}")
        uvicorn.run(
            self.app.build(),
            host=self.host,
            port=self.port,
            log_level=log_level,
        )

    async def start(self) -> None:
        """Start the server asynchronously.

        This is useful when you need to run the server alongside other
        async tasks. Note: This requires managing the uvicorn server
        lifecycle manually.
        """
        config = uvicorn.Config(
            self.app.build(),
            host=self.host,
            port=self.port,
        )
        server = uvicorn.Server(config)
        await server.serve()

    def get_asgi_app(self):
        """Get the underlying ASGI application.

        This is useful for testing or when you need to mount the A2A
        application within a larger application.

        Returns:
            The Starlette ASGI application.
        """
        return self.app.build()
