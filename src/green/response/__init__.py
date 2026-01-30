"""Response generation for the Green agent (Phase 3.6).

This package contains the response generation system that creates in-character
responses from simulated contacts during assessments. It transforms UES from
a static simulation into a dynamic, interactive environment.

Modules:
    generator: Main ResponseGenerator class
    models: Data models for scheduled responses and LLM outputs
    prompts: LLM prompt templates for response decisions and generation
"""

from src.green.response.generator import (
    ResponseGenerationError,
    ResponseGenerator,
    ResponseGeneratorError,
)
from src.green.response.models import (
    CalendarEventContext,
    CalendarRSVPResult,
    MessageContext,
    ScheduledResponse,
    ShouldRespondResult,
    ThreadContext,
)
from src.green.response.prompts import (
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
    # Generator (Phase 3.6)
    "ResponseGenerator",
    "ResponseGeneratorError",
    "ResponseGenerationError",
    # Models
    "ScheduledResponse",
    "ShouldRespondResult",
    "CalendarRSVPResult",
    "ThreadContext",
    "MessageContext",
    "CalendarEventContext",
    # Prompts
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
