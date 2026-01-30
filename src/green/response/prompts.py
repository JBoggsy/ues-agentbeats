"""Prompt templates for response generation.

This module contains all prompt templates used by the ResponseGenerator for
LLM-based decision-making and content generation. Templates are defined as
string constants that can be formatted with context-specific values.

Template Categories:
    - Should-Respond: Decide if a character should reply to a message
    - Response Generation: Generate in-character response content
    - Calendar RSVP: Decide how a character responds to calendar invitations
    - Summarization: Summarize long thread histories

Template Design Principles:
    1. Clear role definition: System prompts define the LLM's task
    2. Comprehensive context: User prompts include all relevant information
    3. Structured output: Request JSON where parsing is needed
    4. Modality-aware: Templates adapt to email/SMS/calendar contexts
"""

# =============================================================================
# Should-Respond Prompts
# =============================================================================

SHOULD_RESPOND_SYSTEM_PROMPT = """You are evaluating whether a simulated character would realistically respond to a message they received.

Consider:
- The character's personality and special instructions
- The conversation context and history
- Whether the message asks a question, makes a request, or clearly expects a reply
- Common communication patterns (e.g., "Thanks!" or "Got it" messages don't usually warrant a response)
- The communication medium ({modality}) and its conventions

Respond with JSON only: {{"should_respond": true/false, "reasoning": "brief explanation"}}"""

SHOULD_RESPOND_USER_PROMPT = """Character Profile:
- Name: {character_name}
- Personality: {character_personality}
{special_instructions_section}
{config_section}

This is a conversation via {modality}.

{thread_summary_section}
Conversation History (most recent last):
{formatted_thread_history}

Latest Message (sent by {sender_name}):
{formatted_latest_message}

Based on this character's personality and the conversation context, would {character_name} send a response to this {modality} message?"""


# =============================================================================
# Response Generation Prompts
# =============================================================================

GENERATE_RESPONSE_SYSTEM_PROMPT = """You are roleplaying as {character_name}.

Character Profile:
- Name: {character_name}
- Personality: {character_personality}
{special_instructions_section}
{config_section}

Write a realistic {modality} response that:
- Matches your character's communication style and personality
- Suits the character's relationship(s) with the conversation participants and message sender
- Is appropriate for the conversation context
- Continues the conversation naturally
- Uses appropriate tone and length for a {modality} message

Respond with just the message content. Do not include headers, subject lines, signatures, or metadata."""

GENERATE_RESPONSE_USER_PROMPT = """This is a {modality} conversation with {participant_names}.

{thread_summary_section}
Conversation History:
{formatted_thread_history}

{sender_name} just sent this message:
{formatted_latest_message}

Write your response as {character_name}."""


# =============================================================================
# Calendar RSVP Prompts
# =============================================================================

CALENDAR_RSVP_SYSTEM_PROMPT = """You are deciding how a simulated character would respond to a calendar invitation.

Character Profile:
- Name: {character_name}
- Personality: {character_personality}
{special_instructions_section}
{config_section}

Consider:
- The character's personality and typical availability
- Their relationship with the event organizer
- The nature of the event
- Any scheduling hints in their special instructions or config

Respond with JSON: {{"status": "accepted"|"declined"|"tentative", "comment": "optional brief comment", "reasoning": "explanation"}}"""

CALENDAR_RSVP_USER_PROMPT = """Calendar Invitation:
- Title: {event_title}
- Organizer: {organizer}
- Start: {start_time}
- End: {end_time}
- Location: {location}
- Description: {description}
- Attendees: {attendee_list}

How would {character_name} respond to this invitation?"""


# =============================================================================
# Thread Summarization Prompts
# =============================================================================

SUMMARIZE_THREAD_PROMPT = """Summarize the following {modality} conversation history in 2-3 sentences.
Focus on key topics discussed, decisions made, and any outstanding questions or requests.

Conversation:
{formatted_messages}

Summary:"""


# =============================================================================
# Helper Functions for Building Prompts
# =============================================================================


def build_special_instructions_section(special_instructions: str | None) -> str:
    """Build the special instructions section for a prompt.

    Args:
        special_instructions: Character's special instructions, or None.

    Returns:
        Formatted section string, or empty string if no instructions.
    """
    if special_instructions:
        return f"- Special Instructions: {special_instructions}"
    return ""


def build_config_section(config: dict | None) -> str:
    """Build the config section for a prompt.

    Args:
        config: Character's config dict, or None.

    Returns:
        Formatted section string showing relevant config, or empty string.
    """
    if not config:
        return ""

    # Format config as readable key-value pairs
    lines = ["- Additional details:"]
    for key, value in config.items():
        # Format the value based on type
        if isinstance(value, dict):
            lines.append(f"  - {key}:")
            for k, v in value.items():
                lines.append(f"    - {k}: {v}")
        elif isinstance(value, list):
            lines.append(f"  - {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def build_thread_summary_section(summary: str | None) -> str:
    """Build the thread summary section for a prompt.

    Args:
        summary: Summary of earlier messages, or None if no truncation.

    Returns:
        Formatted section string, or empty string if no summary.
    """
    if summary:
        return f"Summary of earlier messages:\n{summary}\n"
    return ""


def build_relationships_section(
    character_name: str,
    relationships: dict[str, str],
    participant_names: list[str],
) -> str:
    """Build the relationships section for a prompt.

    Includes only relationships with participants in the current conversation.

    Args:
        character_name: Name of the character.
        relationships: Dict mapping names to relationship descriptions.
        participant_names: Names of participants in the conversation.

    Returns:
        Formatted section string showing relevant relationships.
    """
    if not relationships:
        return ""

    # Filter to only include relationships with current participants
    relevant = {}
    for name, relationship in relationships.items():
        # Check if any participant matches (partial match for robustness)
        for participant in participant_names:
            if name.lower() in participant.lower() or participant.lower() in name.lower():
                relevant[name] = relationship
                break

    if not relevant:
        return ""

    lines = [f"- {character_name}'s relationships:"]
    for name, relationship in relevant.items():
        lines.append(f"  - {name}: {relationship}")

    return "\n".join(lines)


def format_participant_list(participants: list[str]) -> str:
    """Format a list of participant names for natural language.

    Args:
        participants: List of participant names or addresses.

    Returns:
        Formatted string like "Alice, Bob, and Carol" or "the group".
    """
    if not participants:
        return "others"
    elif len(participants) == 1:
        return participants[0]
    elif len(participants) == 2:
        return f"{participants[0]} and {participants[1]}"
    elif len(participants) <= 4:
        return ", ".join(participants[:-1]) + f", and {participants[-1]}"
    else:
        return f"{participants[0]}, {participants[1]}, and {len(participants) - 2} others"
