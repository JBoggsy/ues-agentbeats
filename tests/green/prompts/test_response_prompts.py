"""Tests for response generation prompt templates.

Tests cover:
- Prompt template string formatting
- Helper function outputs
- Edge cases (empty values, special characters)
"""

from __future__ import annotations

import pytest

from src.green.prompts.response_prompts import (
    CALENDAR_RSVP_SYSTEM_PROMPT,
    CALENDAR_RSVP_USER_PROMPT,
    GENERATE_RESPONSE_SYSTEM_PROMPT,
    GENERATE_RESPONSE_USER_PROMPT,
    SHOULD_RESPOND_SYSTEM_PROMPT,
    SHOULD_RESPOND_USER_PROMPT,
    SUMMARIZE_THREAD_PROMPT,
    build_config_section,
    build_relationships_section,
    build_special_instructions_section,
    build_thread_summary_section,
    format_participant_list,
)


# =============================================================================
# Should-Respond Prompt Tests
# =============================================================================


class TestShouldRespondPrompts:
    """Tests for should-respond prompt templates."""

    def test_system_prompt_email(self) -> None:
        """Test system prompt formatting for email modality."""
        prompt = SHOULD_RESPOND_SYSTEM_PROMPT.format(modality="email")

        assert "email" in prompt
        assert "should_respond" in prompt
        assert "reasoning" in prompt
        assert "JSON" in prompt

    def test_system_prompt_sms(self) -> None:
        """Test system prompt formatting for SMS modality."""
        prompt = SHOULD_RESPOND_SYSTEM_PROMPT.format(modality="SMS")

        assert "SMS" in prompt

    def test_user_prompt_formatting(self) -> None:
        """Test user prompt with all fields populated."""
        prompt = SHOULD_RESPOND_USER_PROMPT.format(
            character_name="Alice Chen",
            character_personality="Professional but friendly",
            special_instructions_section="- Special Instructions: Always respond formally",
            config_section="- Additional details:\n  - department: Engineering",
            modality="email",
            thread_summary_section="Summary: Previous discussion about the project.",
            formatted_thread_history="[10:00] Bob: What's the status?",
            sender_name="Bob Smith",
            formatted_latest_message="Subject: Update needed\n\nHi Alice, can you send an update?",
        )

        assert "Alice Chen" in prompt
        assert "Professional but friendly" in prompt
        assert "Always respond formally" in prompt
        assert "Engineering" in prompt
        assert "email" in prompt
        assert "Bob Smith" in prompt
        assert "Update needed" in prompt

    def test_user_prompt_minimal(self) -> None:
        """Test user prompt with minimal fields (empty optional sections)."""
        prompt = SHOULD_RESPOND_USER_PROMPT.format(
            character_name="Bob Smith",
            character_personality="Casual and brief",
            special_instructions_section="",
            config_section="",
            modality="SMS",
            thread_summary_section="",
            formatted_thread_history="",
            sender_name="Alice",
            formatted_latest_message="Hey, are you coming?",
        )

        assert "Bob Smith" in prompt
        assert "Casual and brief" in prompt
        assert "SMS" in prompt
        assert "Hey, are you coming?" in prompt


# =============================================================================
# Generate Response Prompt Tests
# =============================================================================


class TestGenerateResponsePrompts:
    """Tests for response generation prompt templates."""

    def test_system_prompt_formatting(self) -> None:
        """Test system prompt with character context."""
        prompt = GENERATE_RESPONSE_SYSTEM_PROMPT.format(
            character_name="Alice Chen",
            character_personality="Professional but friendly. Prefers concise messages.",
            special_instructions_section="- Special Instructions: Use formal greetings",
            config_section="- Additional details:\n  - role: Manager",
            modality="email",
        )

        assert "Alice Chen" in prompt
        assert "Professional but friendly" in prompt
        assert "formal greetings" in prompt
        assert "Manager" in prompt
        assert "email" in prompt
        assert "Respond with just the message content" in prompt

    def test_user_prompt_formatting(self) -> None:
        """Test user prompt with conversation context."""
        prompt = GENERATE_RESPONSE_USER_PROMPT.format(
            modality="email",
            participant_names="Bob Smith, Carol Davis",
            thread_summary_section="",
            formatted_thread_history="[09:00] Bob: Can we reschedule?\n[09:15] Carol: +1",
            sender_name="Bob Smith",
            formatted_latest_message="What time works for you?",
            character_name="Alice Chen",
        )

        assert "email" in prompt
        assert "Bob Smith, Carol Davis" in prompt
        assert "Can we reschedule?" in prompt
        assert "What time works for you?" in prompt
        assert "Alice Chen" in prompt


# =============================================================================
# Calendar RSVP Prompt Tests
# =============================================================================


class TestCalendarRSVPPrompts:
    """Tests for calendar RSVP prompt templates."""

    def test_system_prompt_formatting(self) -> None:
        """Test system prompt for calendar RSVP."""
        prompt = CALENDAR_RSVP_SYSTEM_PROMPT.format(
            character_name="Alice Chen",
            character_personality="Always busy but tries to attend important meetings",
            special_instructions_section="- Special Instructions: Prioritize 1:1 meetings",
            config_section="",
        )

        assert "Alice Chen" in prompt
        assert "always attend important meetings" in prompt.lower() or "busy" in prompt.lower()
        assert "status" in prompt
        assert "accepted" in prompt or "declined" in prompt or "tentative" in prompt

    def test_user_prompt_formatting(self) -> None:
        """Test user prompt with event details."""
        prompt = CALENDAR_RSVP_USER_PROMPT.format(
            event_title="Weekly Team Sync",
            organizer="Bob Smith (bob@example.com)",
            start_time="2026-01-29 14:00",
            end_time="2026-01-29 15:00",
            location="Conference Room A",
            description="Weekly sync to discuss project progress.",
            attendee_list="alice@example.com, carol@example.com, dave@example.com",
            character_name="Alice Chen",
        )

        assert "Weekly Team Sync" in prompt
        assert "Bob Smith" in prompt
        assert "Conference Room A" in prompt
        assert "Weekly sync" in prompt
        assert "alice@example.com" in prompt
        assert "Alice Chen" in prompt


# =============================================================================
# Summarize Thread Prompt Tests
# =============================================================================


class TestSummarizeThreadPrompt:
    """Tests for thread summarization prompt template."""

    def test_prompt_formatting(self) -> None:
        """Test summarization prompt with conversation history."""
        prompt = SUMMARIZE_THREAD_PROMPT.format(
            modality="email",
            formatted_messages=(
                "[Monday 09:00] Alice: We need to finalize the budget.\n"
                "[Monday 10:30] Bob: I can get the numbers by Wednesday.\n"
                "[Tuesday 14:00] Alice: Great, please send them when ready."
            ),
        )

        assert "email" in prompt
        assert "finalize the budget" in prompt
        assert "Summary" in prompt
        assert "2-3 sentences" in prompt


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestBuildSpecialInstructionsSection:
    """Tests for build_special_instructions_section helper."""

    def test_with_instructions(self) -> None:
        """Test with special instructions provided."""
        result = build_special_instructions_section("Always respond within 1 hour")

        assert "Special Instructions:" in result
        assert "Always respond within 1 hour" in result

    def test_with_none(self) -> None:
        """Test with None (no special instructions)."""
        result = build_special_instructions_section(None)

        assert result == ""

    def test_with_empty_string(self) -> None:
        """Test with empty string."""
        result = build_special_instructions_section("")

        assert result == ""


class TestBuildConfigSection:
    """Tests for build_config_section helper."""

    def test_with_simple_config(self) -> None:
        """Test with simple key-value config."""
        config = {
            "department": "Engineering",
            "role": "Senior Developer",
        }
        result = build_config_section(config)

        assert "Additional details:" in result
        assert "department: Engineering" in result
        assert "role: Senior Developer" in result

    def test_with_nested_config(self) -> None:
        """Test with nested dictionary in config."""
        config = {
            "pricing": {
                "base_rate": 100,
                "discount": "10%",
            },
        }
        result = build_config_section(config)

        assert "pricing:" in result
        assert "base_rate: 100" in result
        assert "discount: 10%" in result

    def test_with_list_config(self) -> None:
        """Test with list values in config."""
        config = {
            "skills": ["Python", "JavaScript", "SQL"],
        }
        result = build_config_section(config)

        assert "skills: Python, JavaScript, SQL" in result

    def test_with_none(self) -> None:
        """Test with None (no config)."""
        result = build_config_section(None)

        assert result == ""

    def test_with_empty_dict(self) -> None:
        """Test with empty dict."""
        result = build_config_section({})

        assert result == ""


class TestBuildThreadSummarySection:
    """Tests for build_thread_summary_section helper."""

    def test_with_summary(self) -> None:
        """Test with a summary provided."""
        summary = "Previous discussion covered project timeline and budget."
        result = build_thread_summary_section(summary)

        assert "Summary of earlier messages:" in result
        assert "project timeline and budget" in result

    def test_with_none(self) -> None:
        """Test with None (no summary needed)."""
        result = build_thread_summary_section(None)

        assert result == ""


class TestBuildRelationshipsSection:
    """Tests for build_relationships_section helper."""

    def test_with_matching_participants(self) -> None:
        """Test with relationships matching participants."""
        relationships = {
            "Bob Smith": "direct report",
            "Carol Davis": "team member",
            "Dave Wilson": "manager",
        }
        participants = ["Bob Smith", "Carol Davis"]
        result = build_relationships_section("Alice", relationships, participants)

        assert "Alice's relationships:" in result
        assert "Bob Smith: direct report" in result
        assert "Carol Davis: team member" in result
        assert "Dave Wilson" not in result  # Not a participant

    def test_with_partial_name_match(self) -> None:
        """Test with partial name matching."""
        relationships = {
            "Bob": "colleague",
        }
        participants = ["Bob Smith (bob@example.com)"]
        result = build_relationships_section("Alice", relationships, participants)

        assert "Bob: colleague" in result

    def test_with_no_matching_participants(self) -> None:
        """Test when no relationships match participants."""
        relationships = {
            "Frank": "friend",
        }
        participants = ["Bob Smith"]
        result = build_relationships_section("Alice", relationships, participants)

        assert result == ""

    def test_with_empty_relationships(self) -> None:
        """Test with empty relationships dict."""
        result = build_relationships_section("Alice", {}, ["Bob"])

        assert result == ""


class TestFormatParticipantList:
    """Tests for format_participant_list helper."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        result = format_participant_list([])

        assert result == "others"

    def test_single_participant(self) -> None:
        """Test with one participant."""
        result = format_participant_list(["Alice"])

        assert result == "Alice"

    def test_two_participants(self) -> None:
        """Test with two participants."""
        result = format_participant_list(["Alice", "Bob"])

        assert result == "Alice and Bob"

    def test_three_participants(self) -> None:
        """Test with three participants."""
        result = format_participant_list(["Alice", "Bob", "Carol"])

        assert result == "Alice, Bob, and Carol"

    def test_four_participants(self) -> None:
        """Test with four participants."""
        result = format_participant_list(["Alice", "Bob", "Carol", "Dave"])

        assert result == "Alice, Bob, Carol, and Dave"

    def test_many_participants(self) -> None:
        """Test with more than four participants."""
        result = format_participant_list(
            ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
        )

        assert result == "Alice, Bob, and 4 others"
