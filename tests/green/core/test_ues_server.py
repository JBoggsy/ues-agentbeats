"""Tests for src.green.core.ues_server module.

This module provides comprehensive tests for the ``UESServerManager`` class,
which manages starting, monitoring, and stopping a UES server subprocess.

Tests cover:
    - Construction and property defaults
    - Admin API key generation (random and pre-set)
    - Process spawning and environment variable injection
    - Health-check polling (``_wait_for_ready``)
    - Graceful and forced shutdown
    - Output draining and buffering
    - Error handling (startup failure, premature exit)
"""

from __future__ import annotations

import asyncio
import subprocess
from collections import deque
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest

from src.green.core.ues_server import (
    UESServerError,
    UESServerManager,
    _ADMIN_KEY_BYTES,
    _DEFAULT_HOST,
    _HEALTH_TIMEOUT,
    _LOG_BUFFER_MAXLEN,
    _SHUTDOWN_TIMEOUT,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager() -> UESServerManager:
    """Create a UESServerManager with default settings."""
    return UESServerManager(port=8100)


@pytest.fixture
def manager_with_key() -> UESServerManager:
    """Create a UESServerManager with a pre-set admin key."""
    return UESServerManager(
        port=8200,
        admin_api_key="a" * 64,
    )


# =============================================================================
# Construction Tests
# =============================================================================


class TestConstruction:
    """Tests for UESServerManager initialization."""

    def test_default_host(self, manager: UESServerManager) -> None:
        """Manager uses default host 127.0.0.1."""
        assert manager.host == _DEFAULT_HOST

    def test_custom_host(self) -> None:
        """Manager accepts a custom host."""
        mgr = UESServerManager(port=8100, host="0.0.0.0")
        assert mgr.host == "0.0.0.0"

    def test_port_stored(self, manager: UESServerManager) -> None:
        """Manager stores the assigned port."""
        assert manager.port == 8100

    def test_base_url(self, manager: UESServerManager) -> None:
        """Base URL is constructed from host and port."""
        assert manager.base_url == "http://127.0.0.1:8100"

    def test_random_admin_key_generated(
        self, manager: UESServerManager,
    ) -> None:
        """A random admin key is generated when none is provided."""
        # token_hex(32) produces 64 hex chars
        assert len(manager.admin_api_key) == _ADMIN_KEY_BYTES * 2
        assert all(c in "0123456789abcdef" for c in manager.admin_api_key)

    def test_preset_admin_key_used(
        self, manager_with_key: UESServerManager,
    ) -> None:
        """A pre-set admin key is used verbatim."""
        assert manager_with_key.admin_api_key == "a" * 64

    def test_unique_keys_per_instance(self) -> None:
        """Each manager instance generates a unique random key."""
        mgr1 = UESServerManager(port=8100)
        mgr2 = UESServerManager(port=8101)
        assert mgr1.admin_api_key != mgr2.admin_api_key

    def test_not_running_initially(
        self, manager: UESServerManager,
    ) -> None:
        """Manager is not running before start() is called."""
        assert manager.is_running is False

    def test_output_lines_empty_initially(
        self, manager: UESServerManager,
    ) -> None:
        """Output lines buffer is empty before start."""
        assert manager.output_lines == []

    def test_custom_log_buffer_size(self) -> None:
        """Custom log buffer size is respected."""
        mgr = UESServerManager(port=8100, log_buffer_size=50)
        assert mgr._output_lines.maxlen == 50

    def test_default_log_buffer_size(
        self, manager: UESServerManager,
    ) -> None:
        """Default log buffer size matches the module constant."""
        assert manager._output_lines.maxlen == _LOG_BUFFER_MAXLEN


# =============================================================================
# Property Tests
# =============================================================================


class TestIsRunning:
    """Tests for the is_running property."""

    def test_false_when_no_process(
        self, manager: UESServerManager,
    ) -> None:
        """Returns False when no process exists."""
        assert manager._process is None
        assert manager.is_running is False

    def test_true_when_process_alive(
        self, manager: UESServerManager,
    ) -> None:
        """Returns True when process poll() returns None (running)."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc
        assert manager.is_running is True

    def test_false_when_process_exited(
        self, manager: UESServerManager,
    ) -> None:
        """Returns False when process poll() returns an exit code."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        manager._process = mock_proc
        assert manager.is_running is False


class TestOutputLines:
    """Tests for the output_lines property."""

    def test_returns_copy(self, manager: UESServerManager) -> None:
        """Returns a list copy, not a deque reference."""
        manager._output_lines.append("line1")
        lines = manager.output_lines
        assert isinstance(lines, list)
        assert lines == ["line1"]
        # Mutating the returned list doesn't affect the buffer.
        lines.append("line2")
        assert len(manager._output_lines) == 1


# =============================================================================
# _spawn_process Tests
# =============================================================================


class TestSpawnProcess:
    """Tests for the _spawn_process method."""

    @patch("src.green.core.ues_server.subprocess.Popen")
    def test_command_includes_port_and_host(
        self,
        mock_popen: MagicMock,
        manager: UESServerManager,
    ) -> None:
        """Subprocess command includes --host and --port flags."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(
            cmd="ues", timeout=0.1,
        )
        mock_popen.return_value = mock_proc

        manager._spawn_process()

        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "8100" in cmd

    @patch("src.green.core.ues_server.subprocess.Popen")
    def test_admin_key_in_environment(
        self,
        mock_popen: MagicMock,
        manager: UESServerManager,
    ) -> None:
        """UES_ADMIN_KEY is set in the subprocess environment."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(
            cmd="ues", timeout=0.1,
        )
        mock_popen.return_value = mock_proc

        manager._spawn_process()

        _, kwargs = mock_popen.call_args
        env = kwargs["env"]
        assert env["UES_ADMIN_KEY"] == manager.admin_api_key

    @patch("src.green.core.ues_server.subprocess.Popen")
    def test_stdout_piped(
        self,
        mock_popen: MagicMock,
        manager: UESServerManager,
    ) -> None:
        """Subprocess stdout is piped for draining."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(
            cmd="ues", timeout=0.1,
        )
        mock_popen.return_value = mock_proc

        manager._spawn_process()

        _, kwargs = mock_popen.call_args
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True

    @patch("src.green.core.ues_server.subprocess.Popen")
    def test_os_error_wraps_in_ues_server_error(
        self,
        mock_popen: MagicMock,
        manager: UESServerManager,
    ) -> None:
        """OSError during Popen is wrapped in UESServerError."""
        mock_popen.side_effect = OSError("no such file")

        with pytest.raises(UESServerError, match="Failed to start"):
            manager._spawn_process()

    @patch("src.green.core.ues_server.subprocess.Popen")
    def test_immediate_exit_raises(
        self,
        mock_popen: MagicMock,
        manager: UESServerManager,
    ) -> None:
        """Immediate process exit raises UESServerError."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        # wait(timeout=0.1) returns immediately (process already exited)
        mock_proc.wait.return_value = 1
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "Error: port in use"
        mock_popen.return_value = mock_proc

        with pytest.raises(UESServerError, match="exited immediately"):
            manager._spawn_process()


# =============================================================================
# _wait_for_ready Tests
# =============================================================================


class TestWaitForReady:
    """Tests for the _wait_for_ready method."""

    @pytest.mark.asyncio
    async def test_returns_on_200(
        self, manager: UESServerManager,
    ) -> None:
        """Returns successfully when /health responds with 200."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await manager._wait_for_ready(timeout=5.0)

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(
        self, manager: UESServerManager,
    ) -> None:
        """Retries when connection fails, then succeeds."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        ok_response = MagicMock()
        ok_response.status_code = 200

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [
                httpx.ConnectError("refused"),
                httpx.ConnectError("refused"),
                ok_response,
            ]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await manager._wait_for_ready(timeout=5.0)

            assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_raises(
        self, manager: UESServerManager,
    ) -> None:
        """Raises TimeoutError if server never becomes ready."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(TimeoutError, match="not ready"):
                await manager._wait_for_ready(timeout=0.3)

    @pytest.mark.asyncio
    async def test_process_exit_while_waiting(
        self, manager: UESServerManager,
    ) -> None:
        """Raises UESServerError if process exits during wait."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        # Process has exited (poll returns exit code).
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        manager._process = mock_proc

        with pytest.raises(UESServerError, match="exited with code"):
            await manager._wait_for_ready(timeout=5.0)


# =============================================================================
# start() Tests
# =============================================================================


class TestStart:
    """Tests for the start() method."""

    @pytest.mark.asyncio
    async def test_start_spawns_and_waits(
        self, manager: UESServerManager,
    ) -> None:
        """start() calls _spawn_process and _wait_for_ready."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""

        with (
            patch.object(
                manager, "_spawn_process", return_value=mock_proc,
            ) as mock_spawn,
            patch.object(
                manager, "_wait_for_ready", new_callable=AsyncMock,
            ) as mock_wait,
        ):
            await manager.start()

            mock_spawn.assert_called_once()
            mock_wait.assert_called_once_with(timeout=_HEALTH_TIMEOUT)

    @pytest.mark.asyncio
    async def test_start_creates_drain_task(
        self, manager: UESServerManager,
    ) -> None:
        """start() creates a background drain task."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.return_value = ""

        with (
            patch.object(
                manager, "_spawn_process", return_value=mock_proc,
            ),
            patch.object(
                manager, "_wait_for_ready", new_callable=AsyncMock,
            ),
        ):
            await manager.start()

            assert manager._drain_task is not None
            assert not manager._drain_task.done()

            # Cleanup
            manager._drain_task.cancel()
            try:
                await manager._drain_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_start_skips_if_already_running(
        self, manager: UESServerManager,
    ) -> None:
        """start() logs a warning and returns if already running."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        manager._process = mock_proc

        with patch.object(manager, "_spawn_process") as mock_spawn:
            await manager.start()
            mock_spawn.assert_not_called()


# =============================================================================
# stop() Tests
# =============================================================================


class TestStop:
    """Tests for the stop() method."""

    @pytest.mark.asyncio
    async def test_stop_no_process(
        self, manager: UESServerManager,
    ) -> None:
        """stop() returns None when no process is running."""
        result = await manager.stop()
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_terminates_process(
        self, manager: UESServerManager,
    ) -> None:
        """stop() sends SIGTERM and waits for the process."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        manager._process = mock_proc

        result = await manager.stop()

        mock_proc.terminate.assert_called_once()
        assert result == 0
        assert manager._process is None

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(
        self, manager: UESServerManager,
    ) -> None:
        """stop() escalates to SIGKILL if SIGTERM times out."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        # First wait (terminate) raises TimeoutError, second (kill) returns
        mock_proc.wait.side_effect = [
            TimeoutError(),  # asyncio.wait_for timeout
            -9,
        ]
        manager._process = mock_proc

        with patch("src.green.core.ues_server.asyncio.wait_for") as mock_wf:
            mock_wf.side_effect = TimeoutError()
            with patch(
                "src.green.core.ues_server.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=-9,
            ):
                result = await manager.stop(timeout=0.1)

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert result == -9

    @pytest.mark.asyncio
    async def test_stop_already_exited(
        self, manager: UESServerManager,
    ) -> None:
        """stop() handles already-exited process gracefully."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        manager._process = mock_proc

        result = await manager.stop()

        mock_proc.terminate.assert_not_called()
        assert result == 0
        assert manager._process is None

    @pytest.mark.asyncio
    async def test_stop_cancels_drain_task(
        self, manager: UESServerManager,
    ) -> None:
        """stop() cancels the background drain task."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        manager._process = mock_proc

        # Create a real asyncio task that blocks, so we can verify cancel.
        async def blocking_coro() -> None:
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                raise

        drain_task = asyncio.create_task(blocking_coro())
        manager._drain_task = drain_task

        await manager.stop()

        assert drain_task.cancelled()
        assert manager._drain_task is None


# =============================================================================
# check_health Tests
# =============================================================================


class TestCheckHealth:
    """Tests for the check_health method."""

    @pytest.mark.asyncio
    async def test_true_on_200(
        self, manager: UESServerManager,
    ) -> None:
        """Returns True when /health responds 200."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            assert await manager.check_health() is True

    @pytest.mark.asyncio
    async def test_false_on_connect_error(
        self, manager: UESServerManager,
    ) -> None:
        """Returns False when connection fails."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            assert await manager.check_health() is False

    @pytest.mark.asyncio
    async def test_false_when_not_running(
        self, manager: UESServerManager,
    ) -> None:
        """Returns False when no process is running."""
        assert await manager.check_health() is False

    @pytest.mark.asyncio
    async def test_false_on_non_200(
        self, manager: UESServerManager,
    ) -> None:
        """Returns False when /health responds with non-200 status."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        manager._process = mock_proc

        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("src.green.core.ues_server.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            assert await manager.check_health() is False


# =============================================================================
# _drain_output Tests
# =============================================================================


class TestDrainOutput:
    """Tests for the _drain_output background task."""

    @pytest.mark.asyncio
    async def test_stores_lines_in_buffer(
        self, manager: UESServerManager,
    ) -> None:
        """Drain task reads lines and stores them in the output buffer."""
        lines = ["Starting UES...\n", "Server ready\n", ""]

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = MagicMock(side_effect=lines)
        manager._process = mock_proc

        # Run _drain_output with run_in_executor doing synchronous calls.
        with patch(
            "asyncio.get_event_loop",
        ) as mock_loop_fn:
            mock_loop = MagicMock()

            async def fake_executor(executor, fn, *args):
                return fn(*args)

            mock_loop.run_in_executor = fake_executor
            mock_loop_fn.return_value = mock_loop

            await manager._drain_output()

        assert "Starting UES..." in manager.output_lines
        assert "Server ready" in manager.output_lines

    @pytest.mark.asyncio
    async def test_no_process_returns_immediately(
        self, manager: UESServerManager,
    ) -> None:
        """Drain returns immediately if no process exists."""
        manager._process = None
        await manager._drain_output()
        assert manager.output_lines == []

    @pytest.mark.asyncio
    async def test_no_stdout_returns_immediately(
        self, manager: UESServerManager,
    ) -> None:
        """Drain returns immediately if process has no stdout."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.stdout = None
        manager._process = mock_proc
        await manager._drain_output()
        assert manager.output_lines == []

    @pytest.mark.asyncio
    async def test_respects_buffer_size_limit(self) -> None:
        """Buffer is bounded by log_buffer_size."""
        manager = UESServerManager(port=8100, log_buffer_size=3)

        # Simulate 5 lines being drained.
        lines = [f"line{i}\n" for i in range(5)] + [""]

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = MagicMock(side_effect=lines)
        manager._process = mock_proc

        with patch("asyncio.get_event_loop") as mock_loop_fn:
            mock_loop = MagicMock()

            async def fake_executor(executor, fn, *args):
                return fn(*args)

            mock_loop.run_in_executor = fake_executor
            mock_loop_fn.return_value = mock_loop

            await manager._drain_output()

        # Only the last 3 lines should be in the buffer.
        assert len(manager.output_lines) == 3
        assert manager.output_lines == ["line2", "line3", "line4"]
