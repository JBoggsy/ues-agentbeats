# Response Generation Prompts

This module contains LLM prompt templates for the response generation system.

## Overview

The prompt templates guide LLMs in:

1. **Deciding if characters should respond** to messages
2. **Generating in-character response content**
3. **Making calendar RSVP decisions**
4. **Summarizing long conversation threads**

## Module Structure

```
src/green/prompts/
├── __init__.py
└── response_prompts.py    # All prompt templates and helpers
```

## Prompt Templates

### Should-Respond Decision

Determines if a character should respond to a message based on:
- Character personality and special instructions
- Message content and context
- Thread history

**Templates:**
- `SHOULD_RESPOND_SYSTEM_PROMPT` - System context for the decision
- `SHOULD_RESPOND_USER_PROMPT` - User prompt with message details

**Output:** `ShouldRespondResult` (Pydantic model with `should_respond: bool` and `reasoning: str`)

### Response Generation

Generates in-character response content matching:
- Character personality and communication style
- Appropriate modality conventions (email vs SMS)
- Thread context and conversation flow

**Templates:**
- `GENERATE_RESPONSE_SYSTEM_PROMPT` - System context for generation
- `GENERATE_RESPONSE_USER_PROMPT` - User prompt with full context

**Output:** Plain text response content

### Calendar RSVP

Decides how a character should respond to calendar invitations:
- Accept, decline, or tentative
- Optional comment explaining decision

**Templates:**
- `CALENDAR_RSVP_SYSTEM_PROMPT` - System context for RSVP
- `CALENDAR_RSVP_USER_PROMPT` - User prompt with event details

**Output:** `CalendarRSVPResult` (Pydantic model with `status`, `reasoning`, and optional `comment`)

### Thread Summarization

Summarizes long conversation threads to fit within context limits.

**Template:** `SUMMARIZE_THREAD_PROMPT`

**Output:** Concise summary of earlier messages

## Helper Functions

### `build_special_instructions_section(instructions: str | None) -> str`
Formats character special instructions for prompts.

### `build_config_section(config: dict | None) -> str`
Formats character configuration for prompts.

### `build_thread_summary_section(summary: str | None) -> str`
Formats thread summary for prompts.

### `build_relationships_section(relationships: dict[str, str]) -> str`
Formats character relationships for prompts.

### `format_participant_list(participants: list[str]) -> str`
Formats a list of conversation participants.

## Usage Example

```python
from src.green.prompts.response_prompts import (
    SHOULD_RESPOND_SYSTEM_PROMPT,
    SHOULD_RESPOND_USER_PROMPT,
    build_special_instructions_section,
    build_config_section,
)

# Build prompts
system = SHOULD_RESPOND_SYSTEM_PROMPT.format(modality="email")
user = SHOULD_RESPOND_USER_PROMPT.format(
    character_name="Alice Chen",
    character_personality="Friendly and helpful software engineer",
    special_instructions_section=build_special_instructions_section(
        "Responds promptly to urgent matters"
    ),
    config_section=build_config_section(None),
    modality="email",
    thread_summary_section="",
    formatted_thread_history="(No previous messages)",
    sender_name="Bob Smith",
    formatted_latest_message="Can you help me with the API?",
)

# Call LLM with structured output
result = await llm.with_structured_output(ShouldRespondResult).ainvoke([
    SystemMessage(content=system),
    HumanMessage(content=user),
])
```

## Design Principles

1. **Explicit context**: All relevant information included in prompts
2. **Structured output**: Use Pydantic models for reliable parsing
3. **Character consistency**: Personality influences all decisions
4. **Modality awareness**: Different conventions for email vs SMS
5. **Graceful fallbacks**: Handle edge cases and errors
