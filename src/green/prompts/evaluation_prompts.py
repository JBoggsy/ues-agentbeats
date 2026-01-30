"""Prompt templates for LLM-based criterion evaluation.

This module provides prompt templates used by the CriteriaJudge for
LLM-based evaluation of assessment criteria. It follows the same
patterns as response_prompts.py.

The evaluation flow:
1. Build context from action log, initial/final states, etc.
2. Format the scenario-defined evaluation_prompt with context
3. Send to LLM with structured output parsing
4. Parse response into EvalResult

Constants:
    LLM_EVALUATION_SYSTEM_PROMPT: System prompt for the evaluation LLM.
    LLM_EVALUATION_USER_TEMPLATE: User prompt template with placeholders.

Functions:
    build_evaluation_context_section: Format context for evaluation prompts.
    build_action_log_section: Format action log for evaluation prompts.
    build_state_comparison_section: Format initial/final state comparison.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


# =============================================================================
# System Prompt
# =============================================================================

LLM_EVALUATION_SYSTEM_PROMPT = """You are an impartial evaluator assessing an AI assistant's performance on a specific criterion.

Your role is to:
1. Carefully review the assistant's actions during the assessment
2. Compare the actions against the criterion's requirements
3. Provide a fair and objective score with clear reasoning

Evaluation Guidelines:
- Be objective and consistent in your scoring
- Consider both what the assistant did well and what it could improve
- Provide specific examples from the action log to support your score
- Use the full scoring range when appropriate (don't default to middle scores)
- Focus ONLY on the specific criterion being evaluated

Scoring:
- You will assign a score from 0 to the specified max_score
- 0 = Complete failure to meet the criterion
- max_score = Perfect performance on the criterion
- Partial scores for partial success

Output your evaluation in the requested JSON format."""


# =============================================================================
# User Prompt Template
# =============================================================================

LLM_EVALUATION_USER_TEMPLATE = """## Criterion Being Evaluated

**ID**: {criterion_id}
**Name**: {criterion_name}
**Dimension**: {dimension}
**Max Score**: {max_score}

**Description**:
{criterion_description}

---

## User Task (Given to Assistant)

{user_prompt}

---

## Evaluation Instructions

{evaluation_prompt}

---

{context_section}

---

## Your Task

Based on the criterion description and evaluation instructions above, evaluate the assistant's performance.

Respond with a JSON object containing:
- "score": A number from 0 to {max_score}
- "explanation": A clear explanation of why you assigned this score (2-4 sentences)
- "strengths": A list of specific things the assistant did well (can be empty)
- "weaknesses": A list of specific areas for improvement (can be empty)

Important: The score must be a number between 0 and {max_score}. Use the full range appropriately."""


# =============================================================================
# Context Building Functions
# =============================================================================


def build_evaluation_context_section(
    action_log: list[dict[str, Any]],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    user_prompt: str,
    include_states: bool = True,
) -> str:
    """Build the context section for evaluation prompts.

    Combines action log and state information into a formatted string
    for inclusion in the evaluation prompt.

    Args:
        action_log: List of actions taken by the assistant.
        initial_state: Modality states at assessment start.
        final_state: Modality states at assessment end.
        user_prompt: The task description given to the assistant.
        include_states: Whether to include state comparison (default True).

    Returns:
        Formatted context section string.
    """
    sections = []

    # Action log section
    sections.append(build_action_log_section(action_log))

    # State comparison section (if requested)
    if include_states:
        sections.append(build_state_comparison_section(initial_state, final_state))

    return "\n\n---\n\n".join(sections)


def build_action_log_section(action_log: list[dict[str, Any]]) -> str:
    """Format the action log for evaluation prompts.

    Args:
        action_log: List of actions taken by the assistant.

    Returns:
        Formatted action log section string.
    """
    if not action_log:
        return "## Action Log\n\nNo actions were taken during the assessment."

    lines = ["## Action Log", "", f"The assistant took {len(action_log)} actions:", ""]

    for i, action in enumerate(action_log, 1):
        turn = action.get("turn", "?")
        timestamp = action.get("timestamp", "unknown")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        action_type = action.get("action", "unknown")
        success = action.get("success", True)
        params = action.get("parameters", {})

        status = "✓" if success else "✗"
        lines.append(f"### Action {i} (Turn {turn}) {status}")
        lines.append(f"- **Type**: {action_type}")
        lines.append(f"- **Time**: {timestamp}")
        lines.append(f"- **Success**: {success}")

        if params:
            lines.append("- **Parameters**:")
            for key, value in params.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 200:
                    str_value = str_value[:200] + "..."
                lines.append(f"  - {key}: {str_value}")

        if not success and action.get("error_message"):
            lines.append(f"- **Error**: {action.get('error_message')}")

        lines.append("")

    return "\n".join(lines)


def build_state_comparison_section(
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
) -> str:
    """Format initial and final state comparison for evaluation.

    Args:
        initial_state: Modality states at assessment start.
        final_state: Modality states at assessment end.

    Returns:
        Formatted state comparison section string.
    """
    lines = ["## State Summary", ""]

    # Email state comparison
    if "email" in initial_state or "email" in final_state:
        lines.append("### Email")
        initial_email = initial_state.get("email", {})
        final_email = final_state.get("email", {})
        lines.append(_format_modality_state_change("email", initial_email, final_email))
        lines.append("")

    # Calendar state comparison
    if "calendar" in initial_state or "calendar" in final_state:
        lines.append("### Calendar")
        initial_cal = initial_state.get("calendar", {})
        final_cal = final_state.get("calendar", {})
        lines.append(_format_modality_state_change("calendar", initial_cal, final_cal))
        lines.append("")

    # SMS state comparison
    if "sms" in initial_state or "sms" in final_state:
        lines.append("### SMS")
        initial_sms = initial_state.get("sms", {})
        final_sms = final_state.get("sms", {})
        lines.append(_format_modality_state_change("sms", initial_sms, final_sms))
        lines.append("")

    # Chat state comparison
    if "chat" in initial_state or "chat" in final_state:
        lines.append("### Chat")
        initial_chat = initial_state.get("chat", {})
        final_chat = final_state.get("chat", {})
        lines.append(_format_modality_state_change("chat", initial_chat, final_chat))
        lines.append("")

    if len(lines) == 2:  # Only header
        return "## State Summary\n\nNo state information available."

    return "\n".join(lines)


def _format_modality_state_change(
    modality: str,
    initial: dict[str, Any],
    final: dict[str, Any],
) -> str:
    """Format a single modality's state change.

    Args:
        modality: The modality name.
        initial: Initial state dict.
        final: Final state dict.

    Returns:
        Formatted state change string.
    """
    lines = []

    # Extract relevant metrics based on modality
    if modality == "email":
        initial_count = _count_emails(initial)
        final_count = _count_emails(final)
        initial_unread = _count_unread_emails(initial)
        final_unread = _count_unread_emails(final)
        lines.append(f"- Total emails: {initial_count} → {final_count}")
        lines.append(f"- Unread: {initial_unread} → {final_unread}")

    elif modality == "calendar":
        initial_events = initial.get("event_count", 0)
        final_events = final.get("event_count", 0)
        lines.append(f"- Events: {initial_events} → {final_events}")

    elif modality == "sms":
        initial_msgs = initial.get("total_messages", 0)
        final_msgs = final.get("total_messages", 0)
        lines.append(f"- Messages: {initial_msgs} → {final_msgs}")

    elif modality == "chat":
        initial_msgs = initial.get("total_message_count", 0)
        final_msgs = final.get("total_message_count", 0)
        lines.append(f"- Messages: {initial_msgs} → {final_msgs}")

    return "\n".join(lines) if lines else "No data available"


def _count_emails(state: dict[str, Any]) -> int:
    """Count total emails from email state."""
    if "emails" in state:
        return len(state["emails"])
    if "folders" in state:
        return sum(f.get("email_count", 0) for f in state["folders"].values())
    return state.get("total_emails", 0)


def _count_unread_emails(state: dict[str, Any]) -> int:
    """Count unread emails from email state."""
    if "folders" in state:
        return sum(f.get("unread_count", 0) for f in state["folders"].values())
    return state.get("unread", 0)


# =============================================================================
# Response Model for LLM Output
# =============================================================================

# Note: The Pydantic model for LLM output parsing is defined in judge_models.py
# to avoid circular imports and keep this module focused on prompts.
