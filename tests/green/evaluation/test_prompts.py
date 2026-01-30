"""Tests for the evaluation prompts module.

Tests cover:
- Prompt template constants
- Context section building functions
- Action log formatting
- State comparison formatting
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.green.evaluation.prompts import (
    LLM_EVALUATION_SYSTEM_PROMPT,
    LLM_EVALUATION_USER_TEMPLATE,
    build_action_log_section,
    build_evaluation_context_section,
    build_state_comparison_section,
)


# =============================================================================
# Tests for Prompt Constants
# =============================================================================


class TestPromptConstants:
    """Tests for prompt template constants."""

    def test_system_prompt_is_string(self) -> None:
        """System prompt should be a non-empty string."""
        assert isinstance(LLM_EVALUATION_SYSTEM_PROMPT, str)
        assert len(LLM_EVALUATION_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_key_instructions(self) -> None:
        """System prompt should contain key evaluation instructions."""
        assert "impartial" in LLM_EVALUATION_SYSTEM_PROMPT.lower()
        assert "objective" in LLM_EVALUATION_SYSTEM_PROMPT.lower()
        assert "score" in LLM_EVALUATION_SYSTEM_PROMPT.lower()
        assert "json" in LLM_EVALUATION_SYSTEM_PROMPT.lower()

    def test_user_template_is_string(self) -> None:
        """User template should be a non-empty string."""
        assert isinstance(LLM_EVALUATION_USER_TEMPLATE, str)
        assert len(LLM_EVALUATION_USER_TEMPLATE) > 100

    def test_user_template_has_required_placeholders(self) -> None:
        """User template should have all required format placeholders."""
        required_placeholders = [
            "{criterion_id}",
            "{criterion_name}",
            "{dimension}",
            "{max_score}",
            "{criterion_description}",
            "{evaluation_prompt}",
            "{context_section}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in LLM_EVALUATION_USER_TEMPLATE, f"Missing {placeholder}"


# =============================================================================
# Tests for build_action_log_section
# =============================================================================


class TestBuildActionLogSection:
    """Tests for build_action_log_section function."""

    def test_empty_action_log(self) -> None:
        """Empty action log should produce appropriate message."""
        result = build_action_log_section([])
        assert "no actions" in result.lower()
        assert "Action Log" in result

    def test_single_action(self) -> None:
        """Single action should be formatted correctly."""
        action = {
            "turn": 1,
            "timestamp": datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
            "action": "email.send",
            "success": True,
            "parameters": {"to": ["alice@example.com"]},
        }
        result = build_action_log_section([action])

        assert "Action Log" in result
        assert "1 actions" in result
        assert "email.send" in result
        assert "Turn 1" in result
        assert "✓" in result
        assert "alice@example.com" in result

    def test_multiple_actions(self) -> None:
        """Multiple actions should all be formatted."""
        actions = [
            {
                "turn": 1,
                "action": "email.query",
                "success": True,
                "parameters": {},
            },
            {
                "turn": 1,
                "action": "email.send",
                "success": True,
                "parameters": {"to": ["bob@example.com"]},
            },
            {
                "turn": 2,
                "action": "calendar.create",
                "success": False,
                "error_message": "Calendar not available",
                "parameters": {"title": "Meeting"},
            },
        ]
        result = build_action_log_section(actions)

        assert "3 actions" in result
        assert "email.query" in result
        assert "email.send" in result
        assert "calendar.create" in result
        assert "✗" in result  # Failed action
        assert "Calendar not available" in result

    def test_failed_action_shows_error(self) -> None:
        """Failed action should show error message."""
        action = {
            "turn": 1,
            "action": "email.delete",
            "success": False,
            "error_message": "Permission denied",
            "parameters": {"email_id": "123"},
        }
        result = build_action_log_section([action])

        assert "✗" in result
        assert "False" in result or "success" in result.lower()
        assert "Permission denied" in result

    def test_long_parameter_values_truncated(self) -> None:
        """Long parameter values should be truncated."""
        long_value = "x" * 300
        action = {
            "turn": 1,
            "action": "test.action",
            "success": True,
            "parameters": {"long_param": long_value},
        }
        result = build_action_log_section([action])

        # Should be truncated with ellipsis
        assert "..." in result
        # Should not contain the full 300 characters
        assert len(result) < 500

    def test_datetime_timestamp_formatted(self) -> None:
        """Datetime timestamps should be formatted as ISO string."""
        timestamp = datetime(2026, 1, 29, 14, 30, 45, tzinfo=timezone.utc)
        action = {
            "turn": 1,
            "timestamp": timestamp,
            "action": "test.action",
            "success": True,
            "parameters": {},
        }
        result = build_action_log_section([action])

        assert "2026-01-29" in result
        assert "14:30" in result


# =============================================================================
# Tests for build_state_comparison_section
# =============================================================================


class TestBuildStateComparisonSection:
    """Tests for build_state_comparison_section function."""

    def test_empty_states(self) -> None:
        """Empty states should produce appropriate message."""
        result = build_state_comparison_section({}, {})
        assert "State Summary" in result
        assert "no state" in result.lower() or len(result) < 100

    def test_email_state_comparison(self) -> None:
        """Email state changes should be formatted."""
        initial = {
            "email": {
                "folders": {
                    "inbox": {"email_count": 10, "unread_count": 5},
                    "sent": {"email_count": 3, "unread_count": 0},
                }
            }
        }
        final = {
            "email": {
                "folders": {
                    "inbox": {"email_count": 12, "unread_count": 2},
                    "sent": {"email_count": 5, "unread_count": 0},
                }
            }
        }
        result = build_state_comparison_section(initial, final)

        assert "Email" in result
        assert "13" in result or "→" in result  # Initial total
        assert "17" in result or "→" in result  # Final total

    def test_calendar_state_comparison(self) -> None:
        """Calendar state changes should be formatted."""
        initial = {"calendar": {"event_count": 5}}
        final = {"calendar": {"event_count": 7}}
        result = build_state_comparison_section(initial, final)

        assert "Calendar" in result
        assert "5" in result
        assert "7" in result

    def test_sms_state_comparison(self) -> None:
        """SMS state changes should be formatted."""
        initial = {"sms": {"total_messages": 10}}
        final = {"sms": {"total_messages": 15}}
        result = build_state_comparison_section(initial, final)

        assert "SMS" in result
        assert "10" in result
        assert "15" in result

    def test_chat_state_comparison(self) -> None:
        """Chat state changes should be formatted."""
        initial = {"chat": {"total_message_count": 1}}
        final = {"chat": {"total_message_count": 5}}
        result = build_state_comparison_section(initial, final)

        assert "Chat" in result
        assert "1" in result
        assert "5" in result

    def test_multiple_modalities(self) -> None:
        """Multiple modalities should all be included."""
        initial = {
            "email": {"total_emails": 10},
            "calendar": {"event_count": 5},
            "sms": {"total_messages": 20},
        }
        final = {
            "email": {"total_emails": 15},
            "calendar": {"event_count": 6},
            "sms": {"total_messages": 25},
        }
        result = build_state_comparison_section(initial, final)

        assert "Email" in result
        assert "Calendar" in result
        assert "SMS" in result


# =============================================================================
# Tests for build_evaluation_context_section
# =============================================================================


class TestBuildEvaluationContextSection:
    """Tests for build_evaluation_context_section function."""

    def test_includes_action_log(self) -> None:
        """Context section should include action log."""
        actions = [
            {"turn": 1, "action": "email.send", "success": True, "parameters": {}}
        ]
        result = build_evaluation_context_section(
            action_log=actions,
            initial_state={},
            final_state={},
            user_prompt="Test prompt",
        )

        assert "Action Log" in result
        assert "email.send" in result

    def test_includes_state_comparison(self) -> None:
        """Context section should include state comparison by default."""
        result = build_evaluation_context_section(
            action_log=[],
            initial_state={"email": {"total_emails": 10}},
            final_state={"email": {"total_emails": 15}},
            user_prompt="Test prompt",
        )

        assert "State Summary" in result
        assert "Email" in result

    def test_exclude_states(self) -> None:
        """Context section can exclude state comparison."""
        result = build_evaluation_context_section(
            action_log=[],
            initial_state={"email": {"total_emails": 10}},
            final_state={"email": {"total_emails": 15}},
            user_prompt="Test prompt",
            include_states=False,
        )

        assert "Action Log" in result
        assert "State Summary" not in result

    def test_sections_separated(self) -> None:
        """Sections should be separated by dividers."""
        result = build_evaluation_context_section(
            action_log=[
                {"turn": 1, "action": "test", "success": True, "parameters": {}}
            ],
            initial_state={"email": {"total_emails": 10}},
            final_state={"email": {"total_emails": 10}},
            user_prompt="Test prompt",
        )

        assert "---" in result
