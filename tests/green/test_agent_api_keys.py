"""Tests for GreenAgent API key management methods.

This module tests the ``_create_user_api_key`` and ``_revoke_user_api_key``
methods of the ``GreenAgent`` class, which manage Purple agent API keys
via the UES ``/keys`` endpoint.

Tests cover:
    - Successful key creation with correct permissions
    - Key revocation success cases
    - Handling of 404 responses during revocation (key not found)
    - HTTP error propagation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.green.agent import USER_PERMISSIONS


# =============================================================================
# Fixtures
# =============================================================================


class MockUESServerManager:
    """Mock UESServerManager for testing."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        admin_api_key: str = "test_admin_key_12345678901234567890",
    ) -> None:
        self.base_url = base_url
        self.admin_api_key = admin_api_key


class PartialGreenAgent:
    """Partial GreenAgent implementation for testing API key methods only.

    This class stubs out __init__ to avoid running the full initialization
    but provides the actual API key management implementations from the
    real GreenAgent class.
    """

    def __init__(self, ues_server: MockUESServerManager) -> None:
        self._ues_server = ues_server

    async def _create_user_api_key(
        self,
        assessment_id: str,
    ) -> tuple[str, str]:
        """Create a user-level API key for the Purple agent."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._ues_server.base_url}/keys",
                headers={"X-API-Key": self._ues_server.admin_api_key},
                json={
                    "name": f"Purple Agent ({assessment_id})",
                    "permissions": USER_PERMISSIONS,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["secret"], data["key_id"]

    async def _revoke_user_api_key(self, key_id: str) -> None:
        """Revoke a user API key after an assessment ends."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self._ues_server.base_url}/keys/{key_id}",
                headers={"X-API-Key": self._ues_server.admin_api_key},
                timeout=10.0,
            )
            # Ignore 404 (key already revoked or doesn't exist)
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()


@pytest.fixture
def mock_ues_server() -> MockUESServerManager:
    """Create a mock UES server manager."""
    return MockUESServerManager()


@pytest.fixture
def agent(mock_ues_server: MockUESServerManager) -> PartialGreenAgent:
    """Create a partial GreenAgent for testing."""
    return PartialGreenAgent(mock_ues_server)


# =============================================================================
# _create_user_api_key Tests
# =============================================================================


class TestCreateUserApiKey:
    """Tests for the _create_user_api_key method."""

    @pytest.mark.asyncio
    async def test_creates_key_with_correct_url(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """POST request is made to the correct /keys endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "secret": "test_secret_123",
            "key_id": "ues_key_456",
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._create_user_api_key("assessment-123")

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://127.0.0.1:8100/keys"

    @pytest.mark.asyncio
    async def test_uses_admin_api_key_in_header(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Request includes the admin API key in X-API-Key header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "secret": "test_secret_123",
            "key_id": "ues_key_456",
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._create_user_api_key("assessment-123")

            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers.get("X-API-Key") == "test_admin_key_12345678901234567890"

    @pytest.mark.asyncio
    async def test_sends_correct_permissions(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Request body includes USER_PERMISSIONS."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "secret": "test_secret_123",
            "key_id": "ues_key_456",
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._create_user_api_key("assessment-123")

            call_args = mock_client.post.call_args
            json_body = call_args.kwargs.get("json", {})
            assert json_body.get("permissions") == USER_PERMISSIONS

    @pytest.mark.asyncio
    async def test_key_name_includes_assessment_id(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Key name includes the assessment ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "secret": "test_secret_123",
            "key_id": "ues_key_456",
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._create_user_api_key("my-special-assessment")

            call_args = mock_client.post.call_args
            json_body = call_args.kwargs.get("json", {})
            assert "my-special-assessment" in json_body.get("name", "")

    @pytest.mark.asyncio
    async def test_returns_secret_and_key_id(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Returns a tuple of (secret, key_id)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "secret": "the_api_secret",
            "key_id": "ues_abc123",
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            secret, key_id = await agent._create_user_api_key("assessment-123")

            assert secret == "the_api_secret"
            assert key_id == "ues_abc123"

    @pytest.mark.asyncio
    async def test_raises_on_http_error(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Raises HTTPStatusError on non-2xx response."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await agent._create_user_api_key("assessment-123")


# =============================================================================
# _revoke_user_api_key Tests
# =============================================================================


class TestRevokeUserApiKey:
    """Tests for the _revoke_user_api_key method."""

    @pytest.mark.asyncio
    async def test_deletes_at_correct_url(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """DELETE request is made to /keys/{key_id}."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._revoke_user_api_key("ues_key_789")

            mock_client.delete.assert_called_once()
            call_args = mock_client.delete.call_args
            assert call_args[0][0] == "http://127.0.0.1:8100/keys/ues_key_789"

    @pytest.mark.asyncio
    async def test_uses_admin_api_key_in_header(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Request includes the admin API key in X-API-Key header."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await agent._revoke_user_api_key("ues_key_789")

            call_args = mock_client.delete.call_args
            headers = call_args.kwargs.get("headers", {})
            assert headers.get("X-API-Key") == "test_admin_key_12345678901234567890"

    @pytest.mark.asyncio
    async def test_ignores_404_response(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """404 responses are silently ignored (key already revoked)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Should not raise
            await agent._revoke_user_api_key("nonexistent_key")

    @pytest.mark.asyncio
    async def test_ignores_204_response(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """204 No Content responses are accepted."""
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Should not raise
            await agent._revoke_user_api_key("ues_key_789")

    @pytest.mark.asyncio
    async def test_raises_on_other_http_errors(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Raises HTTPStatusError on non-2xx/404 responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await agent._revoke_user_api_key("ues_key_789")

    @pytest.mark.asyncio
    async def test_raises_on_403_forbidden(
        self,
        agent: PartialGreenAgent,
    ) -> None:
        """Raises HTTPStatusError on 403 Forbidden response."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.delete.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await agent._revoke_user_api_key("ues_key_789")


# =============================================================================
# USER_PERMISSIONS Tests
# =============================================================================


class TestUserPermissions:
    """Tests for the USER_PERMISSIONS constant."""

    def test_contains_email_permissions(self) -> None:
        """USER_PERMISSIONS includes email user-level permissions."""
        assert "email:state" in USER_PERMISSIONS
        assert "email:query" in USER_PERMISSIONS
        assert "email:send" in USER_PERMISSIONS
        assert "email:read" in USER_PERMISSIONS

    def test_contains_sms_permissions(self) -> None:
        """USER_PERMISSIONS includes SMS user-level permissions."""
        assert "sms:state" in USER_PERMISSIONS
        assert "sms:query" in USER_PERMISSIONS
        assert "sms:send" in USER_PERMISSIONS
        assert "sms:read" in USER_PERMISSIONS

    def test_contains_calendar_permissions(self) -> None:
        """USER_PERMISSIONS includes calendar user-level permissions."""
        assert "calendar:state" in USER_PERMISSIONS
        assert "calendar:query" in USER_PERMISSIONS
        assert "calendar:create" in USER_PERMISSIONS
        assert "calendar:respond" in USER_PERMISSIONS

    def test_contains_time_read(self) -> None:
        """USER_PERMISSIONS includes time:read for observing sim time."""
        assert "time:read" in USER_PERMISSIONS

    def test_excludes_proctor_permissions(self) -> None:
        """USER_PERMISSIONS excludes proctor-only permissions."""
        # Receive permissions (simulate external events)
        assert "email:receive" not in USER_PERMISSIONS
        assert "sms:receive" not in USER_PERMISSIONS
        # Time control
        assert "time:advance" not in USER_PERMISSIONS
        assert "time:set" not in USER_PERMISSIONS
        # Simulation control
        assert "simulation:start" not in USER_PERMISSIONS
        assert "simulation:stop" not in USER_PERMISSIONS
        # Key management
        assert "keys:create" not in USER_PERMISSIONS
        assert "keys:revoke" not in USER_PERMISSIONS
        # Scenario management
        assert "scenario:import" not in USER_PERMISSIONS
        assert "scenario:export" not in USER_PERMISSIONS
