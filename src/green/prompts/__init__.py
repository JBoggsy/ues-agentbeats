"""Green agent prompts package.

This package contains prompt templates for LLM-based operations
in the Green agent, including response generation, evaluation,
and decision-making.
"""

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

__all__ = [
    "SHOULD_RESPOND_SYSTEM_PROMPT",
    "SHOULD_RESPOND_USER_PROMPT",
    "GENERATE_RESPONSE_SYSTEM_PROMPT",
    "GENERATE_RESPONSE_USER_PROMPT",
    "CALENDAR_RSVP_SYSTEM_PROMPT",
    "CALENDAR_RSVP_USER_PROMPT",
    "SUMMARIZE_THREAD_PROMPT",
    "build_special_instructions_section",
    "build_config_section",
    "build_thread_summary_section",
    "build_relationships_section",
    "format_participant_list",
]
