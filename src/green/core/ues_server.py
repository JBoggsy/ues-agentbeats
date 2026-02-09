"""UES server lifecycle management for GreenAgent.

This module provides the ``UESServerManager`` class, which handles starting,
monitoring, and stopping a UES (User Environment Simulator) server as a
subprocess. Each ``GreenAgent`` instance owns one ``UESServerManager``
that persists across multiple assessments.

Key Features:
    - Pre-generated admin API key via ``UES_ADMIN_KEY`` environment variable
      (no stdout parsing required).
    - Async health-check polling to wait for server readiness.
    - Background stdout/stderr draining to prevent pipe buffer deadlocks.
    - Graceful shutdown with escalation (``SIGTERM`` → ``SIGKILL``).

Design Notes:
    - The manager generates a random admin key at construction time and
      passes it to the UES subprocess via the ``UES_ADMIN_KEY`` environment
      variable. See UES ``docs/api/AUTHENTICATION.md`` §
      "Programmatic Key Retrieval" for details.
    - Stdout/stderr from the UES process is captured in a bounded
      deque for diagnostic access and continuously drained by a
      background ``asyncio`` task to prevent the subprocess from
      blocking on a full pipe buffer.

Example:
    >>> from src.green.core.ues_server import UESServerManager
    >>> manager = UESServerManager(port=8100)
    >>> await manager.start()       # spins up UES, waits for /health
    >>> manager.admin_api_key       # the pre-generated key
    >>> manager.base_url            # 'http://127.0.0.1:8100'
    >>> await manager.stop()        # graceful shutdown

See Also:
    - ``GREEN_AGENT_DESIGN_PLAN.md`` § 5 (UES Server Management)
    - UES ``docs/api/AUTHENTICATION.md``
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import subprocess
import sys
from collections import deque

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_HEALTH_POLL_INTERVAL = 0.2  # seconds between /health polls
_HEALTH_TIMEOUT = 30.0  # seconds to wait for UES readiness
_SHUTDOWN_TIMEOUT = 5.0  # seconds for graceful SIGTERM before SIGKILL
_LOG_BUFFER_MAXLEN = 2000  # max lines kept in the output buffer
_ADMIN_KEY_BYTES = 32  # secrets.token_hex produces 2× hex chars


class UESServerError(Exception):
    """Raised when UES server operations fail."""


class UESServerManager:
    """Manages the lifecycle of a UES server subprocess.

    Each instance starts a UES server on a dedicated port, generates a
    pre-set admin API key, and provides methods to wait for readiness,
    check health, and shut down cleanly.

    The admin API key is generated at construction time and injected via
    the ``UES_ADMIN_KEY`` environment variable so the server uses it
    directly — no stdout parsing is needed.

    Attributes:
        port: The port the UES server listens on.
        host: The host interface the server binds to.
        admin_api_key: The pre-generated admin API key secret.
        base_url: The full base URL for the UES server.
        is_running: Whether the UES server process is alive.

    Args:
        port: Port number for the UES server.
        host: Host interface to bind to (default ``127.0.0.1``).
        admin_api_key: Optional pre-set admin key. If ``None``, a random
            64-character hex key is generated.
        log_buffer_size: Maximum number of stdout/stderr lines to retain
            in the in-memory ring buffer.
    """

    def __init__(
        self,
        port: int,
        host: str = _DEFAULT_HOST,
        admin_api_key: str | None = None,
        log_buffer_size: int = _LOG_BUFFER_MAXLEN,
    ) -> None:
        self.port = port
        self.host = host
        self.admin_api_key = admin_api_key or secrets.token_hex(
            _ADMIN_KEY_BYTES
        )
        self.base_url = f"http://{self.host}:{self.port}"

        self._process: subprocess.Popen[str] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._output_lines: deque[str] = deque(maxlen=log_buffer_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return ``True`` if the UES server process is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def output_lines(self) -> list[str]:
        """Return a snapshot of buffered server output lines."""
        return list(self._output_lines)

    async def start(
        self,
        ready_timeout: float = _HEALTH_TIMEOUT,
    ) -> None:
        """Start the UES server and wait until it is ready.

        Spawns the UES server as a subprocess with the admin key injected
        via the ``UES_ADMIN_KEY`` environment variable. A background task
        continuously drains the process's combined stdout/stderr. After
        spawning, this method polls the ``/health`` endpoint until it
        responds with HTTP 200 or ``ready_timeout`` is exceeded.

        Args:
            ready_timeout: Maximum seconds to wait for the server to
                become ready.

        Raises:
            UESServerError: If the process fails to start or exits
                prematurely.
            TimeoutError: If the server doesn't respond to health checks
                within ``ready_timeout`` seconds.
        """
        if self.is_running:
            logger.warning(
                "UES server already running on port %d (pid %d)",
                self.port,
                self._process.pid,  # type: ignore[union-attr]
            )
            return

        logger.info(
            "Starting UES server on %s:%d", self.host, self.port,
        )

        self._process = self._spawn_process()

        logger.info(
            "UES server spawned (pid %d) on port %d",
            self._process.pid,
            self.port,
        )

        # Start draining stdout/stderr in the background.
        self._drain_task = asyncio.create_task(
            self._drain_output(),
            name=f"ues-drain-{self.port}",
        )

        await self._wait_for_ready(timeout=ready_timeout)

        logger.info("UES server ready on %s", self.base_url)

    async def stop(
        self,
        timeout: float = _SHUTDOWN_TIMEOUT,
    ) -> int | None:
        """Stop the UES server gracefully.

        Sends ``SIGTERM`` and waits up to ``timeout`` seconds for the
        process to exit. If it doesn't, sends ``SIGKILL``. Also cancels
        the background drain task.

        Args:
            timeout: Seconds to wait for graceful shutdown before
                escalating to ``SIGKILL``.

        Returns:
            The process return code, or ``None`` if no process was
            running.
        """
        if self._process is None:
            return None

        pid = self._process.pid

        if self._process.poll() is None:
            logger.info(
                "Stopping UES server (pid %d) on port %d",
                pid,
                self.port,
            )
            self._process.terminate()
            try:
                return_code = await asyncio.wait_for(
                    asyncio.to_thread(self._process.wait),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(
                    "UES server (pid %d) did not stop within %.1fs, "
                    "sending SIGKILL",
                    pid,
                    timeout,
                )
                self._process.kill()
                return_code = await asyncio.to_thread(self._process.wait)
        else:
            return_code = self._process.returncode

        logger.info(
            "UES server (pid %d) exited with code %s",
            pid,
            return_code,
        )

        # Cancel the drain task.
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        self._process = None
        return return_code

    async def check_health(self) -> bool:
        """Check if the UES server is responding to health checks.

        Returns:
            ``True`` if the server responds with HTTP 200 to ``/health``,
            ``False`` otherwise (connection error, timeout, non-200).
        """
        if not self.is_running:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=5.0,
                )
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_process(self) -> subprocess.Popen[str]:
        """Spawn the UES server subprocess.

        Constructs the command and environment, then starts the process
        with piped stdout/stderr.

        Returns:
            The started ``Popen`` instance.

        Raises:
            UESServerError: If the process cannot be started.
        """
        cmd = [
            sys.executable, "-m", "ues.cli", "server",
            "--host", self.host,
            "--port", str(self.port),
        ]

        env = {**os.environ, "UES_ADMIN_KEY": self.admin_api_key}

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line-buffered
                env=env,
            )
        except OSError as exc:
            raise UESServerError(
                f"Failed to start UES server: {exc}"
            ) from exc

        # Catch immediate startup failure (e.g. bad executable).
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            # Still running — expected.
            pass
        else:
            stdout_tail = ""
            if process.stdout:
                stdout_tail = process.stdout.read()
            raise UESServerError(
                f"UES server exited immediately with code "
                f"{process.returncode}:\n{stdout_tail}"
            )

        return process

    async def _wait_for_ready(self, timeout: float) -> None:
        """Poll ``/health`` until the server responds or timeout.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            TimeoutError: If the server doesn't become ready in time.
            UESServerError: If the server process exits while waiting.
        """
        health_url = f"{self.base_url}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                # Check that the process hasn't died.
                if not self.is_running:
                    raise UESServerError(
                        f"UES server exited with code "
                        f"{self._process.returncode if self._process else '?'}"
                        f" while waiting for readiness"
                    )

                try:
                    response = await client.get(
                        health_url, timeout=2.0,
                    )
                    if response.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(
            f"UES server on port {self.port} not ready after {timeout}s"
        )

    async def _drain_output(self) -> None:
        """Continuously read lines from the process stdout pipe.

        Lines are stored in a bounded deque and logged at DEBUG level.
        This task runs until the process exits or it is cancelled.
        """
        process = self._process
        if process is None or process.stdout is None:
            return

        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(
                    None, process.stdout.readline,
                )
                if not line:
                    # EOF — process closed stdout.
                    break
                line = line.rstrip("\n")
                self._output_lines.append(line)
                logger.debug("[UES:%d] %s", self.port, line)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Error draining UES server output on port %d",
                self.port,
            )
