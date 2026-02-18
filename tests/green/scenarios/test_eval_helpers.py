"""Tests for email_triage_basic scenario _eval_helpers module.

Tests the shared helper functions used by evaluators: extraction,
assignment, and LLM helper functions.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.agentbeats.results import ActionLogEntry


# =============================================================================
# Ensure scenario directory is importable
# =============================================================================

_SCENARIO_DIR = str(
    (
        __import__("pathlib").Path(__file__).parent.parent.parent.parent
        / "scenarios"
        / "email_triage_basic"
    ).resolve()
)
if _SCENARIO_DIR not in sys.path:
    sys.path.insert(0, _SCENARIO_DIR)

import _eval_helpers as helpers  # noqa: E402
import ground_truth as gt  # noqa: E402


# =============================================================================
# Constants
# =============================================================================

SCENARIO_START = datetime(2026, 1, 28, 6, 0, 0, tzinfo=timezone.utc)


def _make_action(
    turn: int,
    action: str,
    timestamp: datetime | None = None,
    parameters: dict[str, Any] | None = None,
) -> ActionLogEntry:
    """Create an ActionLogEntry for testing."""
    if timestamp is None:
        timestamp = SCENARIO_START + timedelta(hours=turn - 1)
    return ActionLogEntry(
        turn=turn,
        timestamp=timestamp,
        action=action,
        parameters=parameters or {},
        success=True,
    )


def _make_chat(
    turn: int,
    content: str,
    timestamp: datetime | None = None,
    role: str = "assistant",
) -> ActionLogEntry:
    """Create a chat.send_message entry."""
    if timestamp is None:
        timestamp = SCENARIO_START + timedelta(hours=turn - 1)
    return ActionLogEntry(
        turn=turn,
        timestamp=timestamp,
        action="chat.send_message",
        parameters={"role": role, "content": content},
        success=True,
    )


# =============================================================================
# Tests: extract_agent_chat_messages
# =============================================================================


class TestExtractAgentChatMessages:
    """Tests for extract_agent_chat_messages."""

    def test_extracts_assistant_messages(self):
        """Extracts only assistant chat messages."""
        log = [
            _make_chat(1, "Hello from assistant", role="assistant"),
            _make_chat(1, "User says hi", role="user"),
            _make_action(1, "email.get_state"),
            _make_chat(2, "Summary 2", role="assistant"),
        ]
        result = helpers.extract_agent_chat_messages(log)

        assert len(result) == 2
        assert result[0].content == "Hello from assistant"
        assert result[1].content == "Summary 2"

    def test_ignores_non_chat_actions(self):
        """Non-chat actions are ignored."""
        log = [
            _make_action(1, "email.get_state"),
            _make_action(1, "email.send"),
        ]
        result = helpers.extract_agent_chat_messages(log)

        assert len(result) == 0

    def test_ignores_user_messages(self):
        """User messages are ignored."""
        log = [_make_chat(1, "User message", role="user")]
        result = helpers.extract_agent_chat_messages(log)

        assert len(result) == 0

    def test_sorted_by_timestamp(self):
        """Results are sorted by timestamp."""
        ts1 = SCENARIO_START + timedelta(hours=2)
        ts2 = SCENARIO_START + timedelta(hours=1)
        log = [
            _make_chat(2, "Second", timestamp=ts1),
            _make_chat(1, "First", timestamp=ts2),
        ]
        result = helpers.extract_agent_chat_messages(log)

        assert result[0].content == "First"
        assert result[1].content == "Second"

    def test_incremental_indices(self):
        """Messages get 0-based sequential indices."""
        log = [
            _make_chat(1, "A"),
            _make_chat(2, "B"),
            _make_chat(3, "C"),
        ]
        result = helpers.extract_agent_chat_messages(log)

        assert [m.index for m in result] == [0, 1, 2]

    def test_multimodal_content(self):
        """Handles list-type multimodal content."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "..."},
            {"type": "text", "text": "World"},
        ]
        log = [ActionLogEntry(
            turn=1,
            timestamp=SCENARIO_START,
            action="chat.send_message",
            parameters={"role": "assistant", "content": content},
            success=True,
        )]
        result = helpers.extract_agent_chat_messages(log)

        assert len(result) == 1
        assert result[0].content == "Hello\nWorld"

    def test_empty_log(self):
        """Empty action log returns empty list."""
        assert helpers.extract_agent_chat_messages([]) == []

    def test_turn_recorded(self):
        """Turn number is captured correctly."""
        log = [_make_chat(7, "Summary")]
        result = helpers.extract_agent_chat_messages(log)

        assert result[0].turn == 7


# =============================================================================
# Tests: get_email_availability_time
# =============================================================================


class TestGetEmailAvailabilityTime:
    """Tests for get_email_availability_time."""

    def test_hour_1_at_scenario_start(self):
        """Pre-existing emails (hour 1) available at scenario start."""
        ec = gt.EMAIL_CLASSIFICATIONS["email_001"]
        assert ec.hour == 1
        result = helpers.get_email_availability_time(ec)
        assert result == SCENARIO_START

    def test_hour_2_at_07_00(self):
        """Hour 2 emails available at 07:00."""
        # Find an hour 2 email
        hour_2 = [
            ec for ec in gt.EMAIL_CLASSIFICATIONS.values()
            if ec.hour == 2
        ]
        assert len(hour_2) > 0
        result = helpers.get_email_availability_time(hour_2[0])
        assert result == SCENARIO_START + timedelta(hours=1)

    def test_hour_12_at_17_00(self):
        """Hour 12 emails available at 17:00."""
        hour_12 = [
            ec for ec in gt.EMAIL_CLASSIFICATIONS.values()
            if ec.hour == 12
        ]
        assert len(hour_12) > 0
        result = helpers.get_email_availability_time(hour_12[0])
        assert result == SCENARIO_START + timedelta(hours=11)


# =============================================================================
# Tests: assign_emails_to_summaries
# =============================================================================


class TestAssignEmailsToSummaries:
    """Tests for assign_emails_to_summaries."""

    def test_single_summary_covers_all(self):
        """A summary after all emails covers everything."""
        ts = SCENARIO_START + timedelta(hours=12)
        summaries = [helpers.AgentSummary(
            index=0, timestamp=ts, content="All done", turn=1,
        )]
        result = helpers.assign_emails_to_summaries(summaries)

        # Should have one assignment covering all 49 emails
        assert len(result) == 1
        total_emails = (
            len(result[0].substantive_emails)
            + len(result[0].noise_emails)
        )
        assert total_emails == 49

    def test_no_summaries_all_uncovered(self):
        """No summaries means all emails are in uncovered bucket."""
        result = helpers.assign_emails_to_summaries([])

        assert len(result) == 1  # One uncovered assignment
        assert result[0].summary is None
        total = (
            len(result[0].substantive_emails) + len(result[0].noise_emails)
        )
        assert total == 49

    def test_two_summaries_split_emails(self):
        """Two summaries split emails by timing."""
        mid = SCENARIO_START + timedelta(hours=6)
        end = SCENARIO_START + timedelta(hours=12)
        summaries = [
            helpers.AgentSummary(
                index=0, timestamp=mid, content="First half", turn=1,
            ),
            helpers.AgentSummary(
                index=1, timestamp=end, content="Second half", turn=2,
            ),
        ]
        result = helpers.assign_emails_to_summaries(summaries)

        # Should have 2 assignments, no uncovered
        assert len(result) == 2
        assert result[0].summary is not None
        assert result[1].summary is not None

        total = sum(
            len(a.substantive_emails) + len(a.noise_emails)
            for a in result
        )
        assert total == 49

    def test_emails_classified_correctly(self):
        """Noise emails go into noise_emails, substantive into substantive."""
        ts = SCENARIO_START + timedelta(hours=12)
        summaries = [helpers.AgentSummary(
            index=0, timestamp=ts, content="All", turn=1,
        )]
        result = helpers.assign_emails_to_summaries(summaries)

        noise_count = len(result[0].noise_emails)
        sub_count = len(result[0].substantive_emails)
        assert noise_count == 20
        assert sub_count == 29

    def test_early_summary_leaves_uncovered(self):
        """A summary before all emails creates uncovered bucket."""
        # Summary only at hour 1 — later emails are uncovered
        ts = SCENARIO_START + timedelta(minutes=30)
        summaries = [helpers.AgentSummary(
            index=0, timestamp=ts, content="Early", turn=1,
        )]
        result = helpers.assign_emails_to_summaries(summaries)

        # Should have 2 entries: one summary + one uncovered
        assert len(result) == 2
        assert result[0].summary is not None
        assert result[1].summary is None


# =============================================================================
# Tests: LLM JSON call helper
# =============================================================================


class TestLlmJsonCall:
    """Tests for _llm_json_call."""

    @pytest.mark.asyncio
    async def test_parses_plain_json(self):
        """Parses plain JSON response."""
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = '[{"key": "value"}]'
        mock_llm.ainvoke = AsyncMock(return_value=response)

        result = await helpers._llm_json_call(mock_llm, "system", "user")
        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        """Strips ```json``` fences from response."""
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = '```json\n[{"key": "value"}]\n```'
        mock_llm.ainvoke = AsyncMock(return_value=response)

        result = await helpers._llm_json_call(mock_llm, "system", "user")
        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        """Non-JSON response raises ValueError."""
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "This is not JSON"
        mock_llm.ainvoke = AsyncMock(return_value=response)

        with pytest.raises(ValueError, match="not valid JSON"):
            await helpers._llm_json_call(mock_llm, "system", "user")


# =============================================================================
# Tests: LLM noise check
# =============================================================================


class TestLlmCheckNoiseMentions:
    """Tests for llm_check_noise_mentions."""

    @pytest.mark.asyncio
    async def test_empty_noise_returns_empty(self):
        """No noise emails = no results."""
        mock_llm = MagicMock()
        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        result = await helpers.llm_check_noise_mentions(
            mock_llm, summary, [],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_mention_results(self):
        """Parses LLM response into EmailMentionResult objects."""
        json_resp = json.dumps([
            {"email_id": "email_005", "mentioned": True,
             "explanation": "Mentioned spam"},
            {"email_id": "email_010", "mentioned": False,
             "explanation": "Not found"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        noise = [
            gt.EMAIL_CLASSIFICATIONS["email_005"],
            gt.EMAIL_CLASSIFICATIONS["email_010"],
        ]
        result = await helpers.llm_check_noise_mentions(
            mock_llm, summary, noise,
        )

        assert len(result) == 2
        assert result[0].mentioned is True
        assert result[1].mentioned is False

    @pytest.mark.asyncio
    async def test_llm_failure_assumes_excluded(self):
        """If LLM fails, assume all noise emails are excluded."""
        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "INVALID JSON"
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        # Use first two noise emails
        noise_ids = sorted(gt.NOISE_EMAIL_IDS)[:2]
        noise = [gt.EMAIL_CLASSIFICATIONS[eid] for eid in noise_ids]

        result = await helpers.llm_check_noise_mentions(
            mock_llm, summary, noise,
        )

        assert len(result) == 2
        assert all(not r.mentioned for r in result)


# =============================================================================
# Tests: LLM coverage check
# =============================================================================


class TestLlmCheckSummaryCoverage:
    """Tests for llm_check_summary_coverage."""

    @pytest.mark.asyncio
    async def test_empty_emails_returns_empty(self):
        """No substantive emails = no results."""
        mock_llm = MagicMock()
        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        result = await helpers.llm_check_summary_coverage(
            mock_llm, summary, [],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_coverage_results(self):
        """Parses LLM response into CoverageResult objects."""
        sub_ids = sorted(gt.SUBSTANTIVE_EMAIL_IDS)[:2]
        json_resp = json.dumps([
            {"email_id": sub_ids[0], "covered": True, "accurate": True,
             "explanation": "Good"},
            {"email_id": sub_ids[1], "covered": True, "accurate": False,
             "explanation": "Inaccurate"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        emails = [gt.EMAIL_CLASSIFICATIONS[eid] for eid in sub_ids]
        result = await helpers.llm_check_summary_coverage(
            mock_llm, summary, emails,
        )

        assert len(result) == 2
        assert result[0].covered is True
        assert result[0].accurate is True
        assert result[1].accurate is False


# =============================================================================
# Tests: LLM urgency extraction
# =============================================================================


class TestLlmExtractUrgency:
    """Tests for llm_extract_urgency."""

    @pytest.mark.asyncio
    async def test_empty_emails_returns_empty(self):
        """No emails = no results."""
        mock_llm = MagicMock()
        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        result = await helpers.llm_extract_urgency(
            mock_llm, summary, [],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_correct_urgency_detected(self):
        """Correctly classified urgency is marked correct."""
        # Find an email with known urgency
        sub_ids = sorted(gt.SUBSTANTIVE_EMAIL_IDS)
        test_email = gt.EMAIL_CLASSIFICATIONS[sub_ids[0]]
        gt_urgency = test_email.urgency.value if test_email.urgency else "low"

        json_resp = json.dumps([
            {"email_id": sub_ids[0], "agent_urgency": gt_urgency,
             "explanation": "Correct"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        result = await helpers.llm_extract_urgency(
            mock_llm, summary, [test_email],
        )

        assert len(result) == 1
        assert result[0].correct is True

    @pytest.mark.asyncio
    async def test_not_found_urgency_incorrect(self):
        """Email not found in summary is marked incorrect."""
        sub_ids = sorted(gt.SUBSTANTIVE_EMAIL_IDS)

        json_resp = json.dumps([
            {"email_id": sub_ids[0], "agent_urgency": "not_found",
             "explanation": "Not mentioned"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        email = gt.EMAIL_CLASSIFICATIONS[sub_ids[0]]
        result = await helpers.llm_extract_urgency(
            mock_llm, summary, [email],
        )

        assert result[0].correct is False
        assert result[0].agent_urgency is None


# =============================================================================
# Tests: LLM thread awareness
# =============================================================================


class TestLlmCheckThreadAwareness:
    """Tests for llm_check_thread_awareness."""

    @pytest.mark.asyncio
    async def test_empty_emails_returns_empty(self):
        """No thread emails = no results."""
        mock_llm = MagicMock()
        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        result = await helpers.llm_check_thread_awareness(
            mock_llm, summary, [],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_aware_result_parsed(self):
        """Thread-aware emails are parsed correctly."""
        # Use a non-initial thread email
        thread_eids = sorted(gt.NON_INITIAL_THREAD_EMAIL_IDS)
        test_eid = thread_eids[0]

        json_resp = json.dumps([
            {"email_id": test_eid, "thread_aware": True,
             "explanation": "References prior email"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        summary = helpers.AgentSummary(
            index=0, timestamp=SCENARIO_START, content="Summary", turn=1,
        )
        email = gt.EMAIL_CLASSIFICATIONS[test_eid]
        result = await helpers.llm_check_thread_awareness(
            mock_llm, summary, [email],
        )

        assert len(result) == 1
        assert result[0].thread_aware is True


# =============================================================================
# Tests: LLM summary classification
# =============================================================================


class TestLlmClassifySummaries:
    """Tests for llm_classify_summaries."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self):
        """No messages = no results."""
        mock_llm = MagicMock()
        result = await helpers.llm_classify_summaries(mock_llm, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_classifies_messages(self):
        """Classifies messages as summaries or not."""
        messages = [
            helpers.AgentSummary(
                index=0,
                timestamp=SCENARIO_START,
                content="Here is your email triage...",
                turn=1,
            ),
            helpers.AgentSummary(
                index=1,
                timestamp=SCENARIO_START + timedelta(minutes=5),
                content="Hello! How can I help?",
                turn=1,
            ),
        ]

        json_resp = json.dumps([
            {"index": 0, "is_summary": True,
             "explanation": "Contains email triage"},
            {"index": 1, "is_summary": False,
             "explanation": "Greeting only"},
        ])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = json_resp
        mock_llm.ainvoke = AsyncMock(return_value=response)

        result = await helpers.llm_classify_summaries(mock_llm, messages)

        assert len(result) == 2
        assert result[0].is_summary is True
        assert result[1].is_summary is False

    @pytest.mark.asyncio
    async def test_llm_failure_assumes_all_summaries(self):
        """If LLM fails, all messages are treated as summaries."""
        messages = [
            helpers.AgentSummary(
                index=0, timestamp=SCENARIO_START, content="Msg", turn=1,
            ),
        ]

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "NOT JSON"
        mock_llm.ainvoke = AsyncMock(return_value=response)

        result = await helpers.llm_classify_summaries(mock_llm, messages)

        assert len(result) == 1
        assert result[0].is_summary is True


# =============================================================================
# Tests: Data structures
# =============================================================================


class TestDataStructures:
    """Tests for helper data structures."""

    def test_agent_summary_fields(self):
        """AgentSummary has expected fields."""
        s = helpers.AgentSummary(
            index=0,
            timestamp=SCENARIO_START,
            content="Test",
            turn=1,
        )
        assert s.index == 0
        assert s.content == "Test"
        assert s.turn == 1

    def test_summary_assignment_defaults(self):
        """SummaryAssignment has correct defaults."""
        a = helpers.SummaryAssignment(summary=None)
        assert a.substantive_emails == []
        assert a.noise_emails == []

    def test_scenario_constants(self):
        """SCENARIO_START and SCENARIO_END are correct."""
        assert helpers.SCENARIO_START == SCENARIO_START
        assert helpers.SCENARIO_END == SCENARIO_START + timedelta(hours=12)
