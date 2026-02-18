"""Shared evaluation helpers for the email_triage_basic scenario.

This module provides helper functions used by multiple programmatic evaluators
in evaluators.py. The underscore prefix ensures ScenarioLoader's
_extract_evaluators() skips this file (it ignores names starting with _).

Key Helpers:
    extract_agent_summaries: Pull agent chat messages from the action log.
    assign_emails_to_summaries: Map each email to its covering summary.
    llm_check_noise_mentions: LLM batch check for noise email mentions.
    llm_check_summary_coverage: LLM batch check for substantive email coverage.
    llm_extract_urgency: LLM batch extraction of urgency labels.
    llm_check_thread_awareness: LLM batch check for thread context awareness.
    llm_classify_summaries: LLM classification of chat messages as summaries.

Design:
    All LLM helpers use per-evaluator batching — one LLM call per summary
    per evaluator. This keeps prompts focused and evaluators independent
    while being much more efficient than one call per email.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.agentbeats.results import ActionLogEntry

import ground_truth as gt  # noqa: E402 — sibling import via sys.path


logger = logging.getLogger(__name__)

# Scenario timing constants
SCENARIO_START = datetime(2026, 1, 28, 6, 0, 0, tzinfo=timezone.utc)
SCENARIO_END = datetime(2026, 1, 28, 18, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AgentSummary:
    """A single agent summary extracted from the action log.

    Attributes:
        index: Zero-based index of this summary in chronological order.
        timestamp: When the summary was sent.
        content: The text content of the summary message.
        turn: The turn number during which it was sent.
    """

    index: int
    timestamp: datetime
    content: str
    turn: int


@dataclass
class SummaryAssignment:
    """Mapping of emails to the summary responsible for covering them.

    Attributes:
        summary: The agent summary, or None if no summary covers this window.
        substantive_emails: Substantive emails assigned to this summary.
        noise_emails: Noise emails assigned to this summary.
    """

    summary: AgentSummary | None
    substantive_emails: list[gt.EmailClassification] = field(
        default_factory=list,
    )
    noise_emails: list[gt.EmailClassification] = field(
        default_factory=list,
    )


@dataclass
class EmailMentionResult:
    """Result of checking whether an email was mentioned in a summary.

    Attributes:
        email_id: The email's message ID.
        mentioned: Whether the email was mentioned.
        explanation: Brief explanation from the LLM.
    """

    email_id: str
    mentioned: bool
    explanation: str = ""


@dataclass
class CoverageResult:
    """Result of checking coverage and accuracy for a substantive email.

    Attributes:
        email_id: The email's message ID.
        covered: Whether the email was mentioned at all.
        accurate: Whether the summary was accurate and complete.
        explanation: Brief explanation from the LLM.
    """

    email_id: str
    covered: bool
    accurate: bool
    explanation: str = ""


@dataclass
class UrgencyResult:
    """Result of extracting the agent's urgency label for an email.

    Attributes:
        email_id: The email's message ID.
        agent_urgency: The urgency label the agent assigned (or None).
        ground_truth_urgency: The canonical urgency label.
        correct: Whether the agent's label matches ground truth.
        explanation: Brief explanation from the LLM.
    """

    email_id: str
    agent_urgency: str | None
    ground_truth_urgency: str
    correct: bool
    explanation: str = ""


@dataclass
class ThreadAwarenessResult:
    """Result of checking thread context awareness for an email.

    Attributes:
        email_id: The email's message ID.
        thread_aware: Whether the summary showed thread awareness.
        explanation: Brief explanation from the LLM.
    """

    email_id: str
    thread_aware: bool
    explanation: str = ""


@dataclass
class SummaryClassification:
    """Classification of an agent chat message as summary or not.

    Attributes:
        index: Index of the message in the agent messages list.
        is_summary: Whether this message is a triage summary.
        timestamp: When the message was sent.
        explanation: Brief explanation of classification.
    """

    index: int
    is_summary: bool
    timestamp: datetime
    explanation: str = ""


# =============================================================================
# Core Extraction Functions
# =============================================================================


def extract_agent_chat_messages(
    action_log: list[ActionLogEntry],
) -> list[AgentSummary]:
    """Extract agent chat messages from the action log.

    Filters for ``chat.send_message`` actions with role ``"assistant"``
    and returns them as AgentSummary objects sorted by timestamp.

    Args:
        action_log: The full action log from the assessment.

    Returns:
        List of AgentSummary objects in chronological order.
    """
    messages: list[AgentSummary] = []
    idx = 0

    for entry in action_log:
        if entry.action != "chat.send_message":
            continue
        role = entry.parameters.get("role", "")
        if role != "assistant":
            continue

        content = entry.parameters.get("content", "")
        if isinstance(content, list):
            # Multimodal content — extract text parts
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        messages.append(AgentSummary(
            index=idx,
            timestamp=entry.timestamp,
            content=str(content),
            turn=entry.turn,
        ))
        idx += 1

    # Sort by timestamp (should already be ordered, but ensure)
    messages.sort(key=lambda m: m.timestamp)
    return messages


def get_email_availability_time(
    classification: gt.EmailClassification,
) -> datetime:
    """Determine when an email becomes available to the agent.

    Pre-existing emails (hour 1) are available at scenario start.
    Arriving emails are available at the start of their hour window.

    Args:
        classification: The email's ground truth classification.

    Returns:
        The datetime when this email is first available.
    """
    if classification.hour == 1:
        return SCENARIO_START
    # Hour N starts at 06:00 + (N-1) hours
    return SCENARIO_START + timedelta(hours=classification.hour - 1)


def assign_emails_to_summaries(
    summaries: list[AgentSummary],
) -> list[SummaryAssignment]:
    """Assign each email to the first summary after its availability time.

    Walks through summaries chronologically. For each summary, collects
    all emails that became available between the previous summary (or
    scenario start) and this summary's timestamp.

    Emails that arrive after the last summary are collected into a final
    ``SummaryAssignment`` with ``summary=None`` (uncovered emails).

    Args:
        summaries: Agent summaries sorted by timestamp.

    Returns:
        List of SummaryAssignment objects, one per summary plus an optional
        final entry for uncovered emails.
    """
    assignments: list[SummaryAssignment] = []

    # Get all emails sorted by availability time
    all_emails = sorted(
        gt.EMAIL_CLASSIFICATIONS.values(),
        key=lambda c: (get_email_availability_time(c), c.email_number),
    )

    # Track which emails have been assigned
    email_index = 0
    prev_cutoff = SCENARIO_START

    for summary in summaries:
        assignment = SummaryAssignment(summary=summary)
        # Collect emails available between prev_cutoff and this summary
        while email_index < len(all_emails):
            email = all_emails[email_index]
            avail_time = get_email_availability_time(email)
            if avail_time <= summary.timestamp:
                if email.category.is_noise:
                    assignment.noise_emails.append(email)
                else:
                    assignment.substantive_emails.append(email)
                email_index += 1
            else:
                break
        assignments.append(assignment)
        prev_cutoff = summary.timestamp

    # Any remaining emails are uncovered
    if email_index < len(all_emails):
        uncovered = SummaryAssignment(summary=None)
        while email_index < len(all_emails):
            email = all_emails[email_index]
            if email.category.is_noise:
                uncovered.noise_emails.append(email)
            else:
                uncovered.substantive_emails.append(email)
            email_index += 1
        assignments.append(uncovered)

    return assignments


# =============================================================================
# LLM Helpers
# =============================================================================

_SYSTEM_PROMPT = (
    "You are an evaluation assistant. You analyze AI agent outputs to "
    "determine whether specific emails were mentioned, correctly classified, "
    "or properly contextualized. Respond ONLY with valid JSON."
)


async def _llm_json_call(
    llm: BaseChatModel,
    system: str,
    user: str,
) -> Any:
    """Make an LLM call expecting a JSON response.

    Args:
        llm: The LangChain LLM to use.
        system: System message content.
        user: User message content.

    Returns:
        Parsed JSON response.

    Raises:
        ValueError: If the LLM response is not valid JSON.
    """
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    response = await llm.ainvoke(messages)
    text = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s", text[:200])
        raise ValueError(f"LLM response was not valid JSON: {e}") from e


def _format_email_list(
    emails: list[gt.EmailClassification],
    include_hints: bool = False,
) -> str:
    """Format a list of emails for inclusion in an LLM prompt.

    Args:
        emails: Email classifications to format.
        include_hints: Whether to include summary_hint (for accuracy checks).

    Returns:
        Formatted string with one email per block.
    """
    parts: list[str] = []
    for email in emails:
        entry = (
            f"- email_id: {email.email_id}\n"
            f"  sender: {email.sender_name}\n"
            f"  subject: {email.subject}"
        )
        if include_hints and email.summary_hint:
            entry += f"\n  expected_content: {email.summary_hint}"
        parts.append(entry)
    return "\n".join(parts)


# =============================================================================
# Noise Exclusion LLM Helper
# =============================================================================


async def llm_check_noise_mentions(
    llm: BaseChatModel,
    summary: AgentSummary,
    noise_emails: list[gt.EmailClassification],
) -> list[EmailMentionResult]:
    """Check whether noise emails are mentioned in a summary.

    Makes a single LLM call per summary to check all noise emails
    assigned to that summary window.

    Args:
        llm: The LLM to use.
        summary: The agent's summary text.
        noise_emails: Noise emails to check for mentions.

    Returns:
        List of EmailMentionResult, one per noise email.
    """
    if not noise_emails:
        return []

    email_list = _format_email_list(noise_emails)
    prompt = (
        "Below is an AI assistant's email triage summary, followed by a "
        "list of noise emails (spam, newsletters, notifications) that "
        "should NOT appear in the summary.\n\n"
        "For each noise email, determine whether the summary mentions, "
        "references, or alludes to it in any way. A mention includes "
        "referencing the sender name, subject, or content of the email.\n\n"
        f"=== SUMMARY ===\n{summary.content}\n\n"
        f"=== NOISE EMAILS ===\n{email_list}\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '  "email_id": string,\n'
        '  "mentioned": boolean,\n'
        '  "explanation": string (brief reason)\n\n'
        "Example: "
        '[{"email_id": "email_005", "mentioned": false, '
        '"explanation": "Not referenced in summary"}]'
    )

    try:
        data = await _llm_json_call(llm, _SYSTEM_PROMPT, prompt)
        results: list[EmailMentionResult] = []
        for item in data:
            results.append(EmailMentionResult(
                email_id=item["email_id"],
                mentioned=bool(item.get("mentioned", False)),
                explanation=item.get("explanation", ""),
            ))
        return results
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("LLM noise check failed: %s — assuming all excluded", e)
        return [
            EmailMentionResult(email_id=em.email_id, mentioned=False,
                               explanation="LLM check failed, default excluded")
            for em in noise_emails
        ]


# =============================================================================
# Summary Coverage & Accuracy LLM Helper
# =============================================================================


async def llm_check_summary_coverage(
    llm: BaseChatModel,
    summary: AgentSummary,
    substantive_emails: list[gt.EmailClassification],
) -> list[CoverageResult]:
    """Check coverage and accuracy for substantive emails in a summary.

    Makes a single LLM call per summary to evaluate all substantive emails
    assigned to that summary window.

    Args:
        llm: The LLM to use.
        summary: The agent's summary text.
        substantive_emails: Substantive emails expected in this summary.

    Returns:
        List of CoverageResult, one per substantive email.
    """
    if not substantive_emails:
        return []

    email_list = _format_email_list(substantive_emails, include_hints=True)
    prompt = (
        "Below is an AI assistant's email triage summary, followed by a "
        "list of important emails that SHOULD appear in the summary.\n\n"
        "For each email, evaluate two things:\n"
        "1. COVERAGE: Is this email mentioned at all in the summary? "
        "(sender name, subject, or topic referenced)\n"
        "2. ACCURACY: If mentioned, is the agent's summary of this email "
        "reasonably accurate and complete? Compare to the expected_content.\n\n"
        f"=== SUMMARY ===\n{summary.content}\n\n"
        f"=== EXPECTED EMAILS ===\n{email_list}\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '  "email_id": string,\n'
        '  "covered": boolean (is the email mentioned?),\n'
        '  "accurate": boolean (if covered, is the summary accurate?),\n'
        '  "explanation": string (brief reason)\n\n'
        "If an email is not covered, set accurate to false.\n\n"
        "Example: "
        '[{"email_id": "email_003", "covered": true, "accurate": true, '
        '"explanation": "Production alert correctly summarized"}]'
    )

    try:
        data = await _llm_json_call(llm, _SYSTEM_PROMPT, prompt)
        results: list[CoverageResult] = []
        for item in data:
            results.append(CoverageResult(
                email_id=item["email_id"],
                covered=bool(item.get("covered", False)),
                accurate=bool(item.get("accurate", False)),
                explanation=item.get("explanation", ""),
            ))
        return results
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(
            "LLM coverage check failed: %s — assuming none covered", e,
        )
        return [
            CoverageResult(email_id=em.email_id, covered=False, accurate=False,
                           explanation="LLM check failed, default not covered")
            for em in substantive_emails
        ]


# =============================================================================
# Urgency Extraction LLM Helper
# =============================================================================


async def llm_extract_urgency(
    llm: BaseChatModel,
    summary: AgentSummary,
    substantive_emails: list[gt.EmailClassification],
) -> list[UrgencyResult]:
    """Extract the agent's urgency classification for substantive emails.

    Makes a single LLM call per summary to extract urgency labels
    for all substantive emails in that window.

    Args:
        llm: The LLM to use.
        summary: The agent's summary text.
        substantive_emails: Substantive emails to extract urgency for.

    Returns:
        List of UrgencyResult, one per substantive email.
    """
    if not substantive_emails:
        return []

    email_list = _format_email_list(substantive_emails)
    prompt = (
        "Below is an AI assistant's email triage summary, followed by a "
        "list of emails. The agent was asked to sort emails by urgency.\n\n"
        "For each email, determine what urgency level the agent assigned "
        "to it. Look for explicit urgency labels (high, medium, low, "
        "urgent, important, etc.) or infer from ordering and language.\n\n"
        "Map the agent's classification to one of: "
        '"high", "medium", "low", or "not_found" if the email is not '
        "mentioned in the summary.\n\n"
        f"=== SUMMARY ===\n{summary.content}\n\n"
        f"=== EMAILS ===\n{email_list}\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '  "email_id": string,\n'
        '  "agent_urgency": "high" | "medium" | "low" | "not_found",\n'
        '  "explanation": string (brief reason)\n\n'
        "Example: "
        '[{"email_id": "email_003", "agent_urgency": "high", '
        '"explanation": "Labeled as urgent/critical by agent"}]'
    )

    try:
        data = await _llm_json_call(llm, _SYSTEM_PROMPT, prompt)
        results: list[UrgencyResult] = []
        for item in data:
            eid = item["email_id"]
            agent_urg = item.get("agent_urgency", "not_found")
            ec = gt.EMAIL_CLASSIFICATIONS.get(eid)
            gt_urg = ec.urgency.value if ec and ec.urgency else "unknown"
            correct = (
                agent_urg == gt_urg
                if agent_urg != "not_found"
                else False
            )
            results.append(UrgencyResult(
                email_id=eid,
                agent_urgency=agent_urg if agent_urg != "not_found" else None,
                ground_truth_urgency=gt_urg,
                correct=correct,
                explanation=item.get("explanation", ""),
            ))
        return results
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(
            "LLM urgency extraction failed: %s — assuming all incorrect", e,
        )
        return [
            UrgencyResult(
                email_id=em.email_id,
                agent_urgency=None,
                ground_truth_urgency=em.urgency.value if em.urgency else "unknown",
                correct=False,
                explanation="LLM check failed, default incorrect",
            )
            for em in substantive_emails
        ]


# =============================================================================
# Thread Awareness LLM Helper
# =============================================================================


def _get_prior_thread_emails(
    email: gt.EmailClassification,
) -> list[gt.EmailClassification]:
    """Get prior emails in the same thread, for context.

    Args:
        email: The current thread email.

    Returns:
        List of prior EmailClassification objects (earlier in the thread).
    """
    if not email.thread_id or email.thread_id not in gt.THREAD_MEMBERSHIP:
        return []

    thread = gt.THREAD_MEMBERSHIP[email.thread_id]
    prior: list[gt.EmailClassification] = []
    for eid in thread.email_ids:
        if eid == email.email_id:
            break
        ec = gt.EMAIL_CLASSIFICATIONS.get(eid)
        if ec:
            prior.append(ec)
    return prior


async def llm_check_thread_awareness(
    llm: BaseChatModel,
    summary: AgentSummary,
    thread_emails: list[gt.EmailClassification],
) -> list[ThreadAwarenessResult]:
    """Check whether the summary shows thread context awareness.

    For each non-initial thread email, determines whether the agent's
    summary incorporates information from prior emails in the thread.

    Args:
        llm: The LLM to use.
        summary: The agent's summary text.
        thread_emails: Thread emails to check awareness for.

    Returns:
        List of ThreadAwarenessResult, one per email.
    """
    if not thread_emails:
        return []

    # Build context for each email including its thread history
    email_contexts: list[str] = []
    for email in thread_emails:
        prior = _get_prior_thread_emails(email)
        thread_info = gt.THREAD_MEMBERSHIP.get(email.thread_id or "")

        prior_text = ""
        if prior:
            prior_subjects = [
                f"    - #{p.email_number}: {p.sender_name}: {p.subject}"
                for p in prior
            ]
            prior_text = (
                "\n  prior_emails_in_thread:\n" + "\n".join(prior_subjects)
            )

        thread_desc = ""
        if thread_info:
            thread_desc = f"\n  thread_arc: {thread_info.arc_description}"

        # Check for cross-thread connection
        cross_thread_note = ""
        if email.email_id in gt.CROSS_THREAD_CONNECTIONS:
            target_tid = gt.CROSS_THREAD_CONNECTIONS[email.email_id]
            target_thread = gt.THREAD_MEMBERSHIP.get(target_tid)
            if target_thread:
                cross_thread_note = (
                    f"\n  cross_thread_note: This email is contextually "
                    f"related to the '{target_thread.name}' thread. "
                    f"A good summary should connect this email to that "
                    f"ongoing situation."
                )

        email_contexts.append(
            f"- email_id: {email.email_id}\n"
            f"  sender: {email.sender_name}\n"
            f"  subject: {email.subject}"
            f"{thread_desc}{prior_text}{cross_thread_note}"
        )

    emails_text = "\n".join(email_contexts)

    prompt = (
        "Below is an AI assistant's email triage summary, followed by "
        "emails that belong to multi-email threads.\n\n"
        "For each email, determine whether the summary demonstrates "
        "THREAD AWARENESS — i.e., the agent's summary for this email "
        "incorporates or references information from prior emails in "
        "the same thread, showing understanding of how the conversation "
        "has progressed.\n\n"
        "Thread awareness means the summary goes beyond describing the "
        "single email in isolation. Examples:\n"
        '- "Priya confirms the root cause of this morning\'s incident" '
        "(references prior investigation)\n"
        '- "Karen\'s third follow-up, tone increasingly frustrated" '
        "(references escalation pattern)\n"
        '- "David flags the same Acme situation internally" '
        "(connects to related thread)\n\n"
        f"=== SUMMARY ===\n{summary.content}\n\n"
        f"=== THREAD EMAILS ===\n{emails_text}\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '  "email_id": string,\n'
        '  "thread_aware": boolean,\n'
        '  "explanation": string (brief reason)\n\n'
        "Example: "
        '[{"email_id": "email_013", "thread_aware": true, '
        '"explanation": "Summary references the overnight alert and '
        'investigation progress"}]'
    )

    try:
        data = await _llm_json_call(llm, _SYSTEM_PROMPT, prompt)
        results: list[ThreadAwarenessResult] = []
        for item in data:
            results.append(ThreadAwarenessResult(
                email_id=item["email_id"],
                thread_aware=bool(item.get("thread_aware", False)),
                explanation=item.get("explanation", ""),
            ))
        return results
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(
            "LLM thread awareness check failed: %s — assuming unaware", e,
        )
        return [
            ThreadAwarenessResult(
                email_id=em.email_id, thread_aware=False,
                explanation="LLM check failed, default unaware",
            )
            for em in thread_emails
        ]


# =============================================================================
# Summary Classification LLM Helper
# =============================================================================


async def llm_classify_summaries(
    llm: BaseChatModel,
    messages: list[AgentSummary],
) -> list[SummaryClassification]:
    """Classify agent chat messages as triage summaries or not.

    Some agent messages might be conversational (greetings, questions)
    rather than triage summaries. This function uses LLM to distinguish.

    Args:
        llm: The LLM to use.
        messages: All agent chat messages to classify.

    Returns:
        List of SummaryClassification, one per message.
    """
    if not messages:
        return []

    messages_text = "\n\n".join(
        f"--- Message {m.index} (timestamp: {m.timestamp.isoformat()}) ---\n"
        f"{m.content[:500]}{'...' if len(m.content) > 500 else ''}"
        for m in messages
    )

    prompt = (
        "Below are chat messages sent by an AI assistant that was asked "
        "to triage an email inbox. Classify each message as either a "
        '"triage_summary" (an email triage report with email summaries) '
        'or "other" (a greeting, question, acknowledgment, or other '
        "non-summary message).\n\n"
        "A triage summary typically contains:\n"
        "- References to specific emails, senders, or subjects\n"
        "- Urgency labels or priority ordering\n"
        "- Brief descriptions of email content\n"
        '- Or a "quiet hour" message noting no important emails\n\n'
        f"=== MESSAGES ===\n{messages_text}\n\n"
        "Respond with a JSON array. Each element must have:\n"
        '  "index": integer (message index),\n'
        '  "is_summary": boolean,\n'
        '  "explanation": string (brief reason)\n\n'
        "Example: "
        '[{"index": 0, "is_summary": true, '
        '"explanation": "Contains email triage with urgency labels"}]'
    )

    try:
        data = await _llm_json_call(llm, _SYSTEM_PROMPT, prompt)
        results: list[SummaryClassification] = []
        msg_by_index = {m.index: m for m in messages}
        for item in data:
            idx = item["index"]
            msg = msg_by_index.get(idx)
            results.append(SummaryClassification(
                index=idx,
                is_summary=bool(item.get("is_summary", False)),
                timestamp=msg.timestamp if msg else SCENARIO_START,
                explanation=item.get("explanation", ""),
            ))
        return results
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(
            "LLM summary classification failed: %s — assuming all summaries",
            e,
        )
        return [
            SummaryClassification(
                index=m.index, is_summary=True, timestamp=m.timestamp,
                explanation="LLM check failed, default classified as summary",
            )
            for m in messages
        ]
