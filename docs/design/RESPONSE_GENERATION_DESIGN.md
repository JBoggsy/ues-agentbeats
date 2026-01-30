# Response Generation Design Document

This document provides the complete implementation design for Phase 3.6: Response Generation. It consolidates information from `IMPLEMENTATION_PLAN.md`, `ASSESSMENT_FLOW.md`, and `RESPONSE_GENERATION_DESIGN_NOTES.md` into an actionable implementation specification.

**Date**: January 29, 2026  
**Status**: ✅ IMPLEMENTED

**Implementation Summary**:
- `src/green/response/models.py` - Data models (ScheduledResponse, ShouldRespondResult, CalendarRSVPResult, ThreadContext, MessageContext, CalendarEventContext)
- `src/green/response/prompts.py` - LLM prompt templates
- `src/green/response/generator.py` - Main ResponseGenerator class (~1200 lines)
- 123 unit tests + 11 integration tests (Ollama and OpenAI)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Models](#3-data-models)
4. [ResponseGenerator Class](#4-responsegenerator-class)
5. [LLM Integration](#5-llm-integration)
6. [Thread History Handling](#6-thread-history-handling)
7. [Response Scheduling](#7-response-scheduling)
8. [Error Handling](#8-error-handling)
9. [Configuration](#9-configuration)
10. [Implementation Plan](#10-implementation-plan)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Overview

### Purpose

The `ResponseGenerator` component creates in-character responses from simulated contacts (characters) during assessments. It transforms the UES environment from a static simulation into a dynamic, interactive world where characters respond realistically to the Purple agent's actions.

### Responsibilities

1. **Process new messages** from `NewMessageCollector` (emails, SMS, calendar events)
2. **Determine if characters should respond** using LLM-based evaluation
3. **Generate in-character response content** matching personality and context
4. **Calculate response timing** based on character configuration
5. **Return scheduled responses** for the `GreenAgent` to inject into UES

### Why Response Generation Matters

Without response generation, assessments would be limited to:
- Pre-scripted email/SMS sequences
- Static calendar states
- No ability to test negotiation, follow-up, or conversation tracking

With response generation, scenarios can test:
- **Multi-turn conversations**: Agent tracks ongoing email threads
- **Negotiation**: Vendor pricing discussions, scheduling conflicts
- **Uncertainty handling**: Different characters respond differently
- **Time management**: Responses arrive with realistic delays

---

## 2. Architecture

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GreenAgent                                      │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │ NewMessageCollector│    │ ResponseGenerator │    │   UES Client     │      │
│  │                  │    │                  │    │   (Proctor)      │      │
│  │  • Collects new  │───▶│  • Processes msgs │───▶│  • email.receive │      │
│  │    emails/SMS/cal│    │  • Checks respond │    │  • sms.receive   │      │
│  │  • Tracks seen   │    │  • Generates text │    │  • calendar.     │      │
│  │    messages      │    │  • Calc timing    │    │    respond_to_   │      │
│  └──────────────────┘    └────────┬─────────┘    │    event         │      │
│                                   │               └──────────────────┘      │
│                                   │                                          │
│                    ┌──────────────┴──────────────┐                          │
│                    ▼                              ▼                          │
│            ┌──────────────┐              ┌──────────────┐                   │
│            │ Should-Respond│              │ Response Gen │                   │
│            │     LLM      │              │     LLM      │                   │
│            └──────────────┘              └──────────────┘                   │
│                                                  │                          │
│                                   ┌──────────────┴──────────────┐          │
│                                   ▼                              ▼          │
│                           ┌──────────────┐              ┌──────────────┐   │
│                           │ Summarization│              │ Calendar RSVP│   │
│                           │     LLM      │              │     LLM      │   │
│                           └──────────────┘              └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Per Turn)

```
1. Purple completes turn → Green receives turn_complete
                               │
2. Green advances simulation time by time_step
                               │
3. Green queries UES via NewMessageCollector:
   └── collect() returns NewMessages(emails, sms_messages, calendar_events)
                               │
4. ResponseGenerator.process_new_messages(new_messages):
   │
   ├── For each email:
   │   ├── Find recipient characters (excluding sender)
   │   ├── Pre-LLM heuristic checks (skip non-responders)
   │   ├── Get thread history
   │   ├── LLM: should_respond? → If no, skip
   │   ├── LLM: generate response content
   │   ├── Calculate scheduled_time
   │   └── Create ScheduledResponse
   │
   ├── For each SMS: (same flow)
   │
   └── For each calendar event:
       ├── Find attendee characters (with needs_action status)
       ├── Pre-LLM heuristic checks
       ├── LLM: decide RSVP (accept/decline/tentative)
       ├── Calculate scheduled_time
       └── Create ScheduledResponse
                               │
5. Returns list[ScheduledResponse]
                               │
6. GreenAgent schedules responses in UES:
   ├── email.receive() for email responses
   ├── sms.receive() for SMS responses
   └── calendar.respond_to_event() for RSVPs
```

---

## 3. Data Models

### 3.1 ScheduledResponse

The output model holding all information needed to schedule a character's response in UES.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class ScheduledResponse:
    """A response to be scheduled in UES.
    
    Contains all information needed to call the appropriate UES client method
    (email.receive, sms.receive, or calendar.respond_to_event) to inject
    the character's response into the simulation.
    
    Attributes:
        character_id: ID of the responding character (key in scenario.characters).
        modality: Which communication channel this response uses.
        content: Message body text for email/SMS, or RSVP comment for calendar.
        scheduled_time: When to inject the response into UES.
        
        # Email-specific (required when modality="email")
        email_subject: Subject line for the email.
        email_to_addresses: Recipients (typically includes the original sender).
        email_cc_addresses: CC recipients (optional).
        email_thread_id: Thread ID for proper threading.
        email_in_reply_to: Message ID of the email being replied to.
        email_references: Full reference chain for threading.
        
        # SMS-specific (required when modality="sms")
        sms_to_numbers: Recipients (typically includes the original sender).
        sms_thread_id: Thread ID for conversation threading.
        
        # Calendar-specific (required when modality="calendar")
        calendar_event_id: Event ID to respond to.
        calendar_id: Calendar containing the event (default: "primary").
        rsvp_status: The RSVP decision.
    """
    
    # Common fields
    character_id: str
    modality: Literal["email", "sms", "calendar"]
    content: str
    scheduled_time: datetime
    
    # Email-specific
    email_subject: str | None = None
    email_to_addresses: list[str] | None = None
    email_cc_addresses: list[str] | None = None
    email_thread_id: str | None = None
    email_in_reply_to: str | None = None
    email_references: list[str] | None = None
    
    # SMS-specific
    sms_to_numbers: list[str] | None = None
    sms_thread_id: str | None = None
    
    # Calendar-specific
    calendar_event_id: str | None = None
    calendar_id: str = "primary"
    rsvp_status: Literal["accepted", "declined", "tentative"] | None = None
```

### 3.2 LLM Response Models

Structured output models for LLM calls:

```python
from pydantic import BaseModel

class ShouldRespondResult(BaseModel):
    """Result from should_respond LLM check."""
    should_respond: bool
    reasoning: str

class CalendarRSVPResult(BaseModel):
    """Result from calendar RSVP LLM decision."""
    status: Literal["accepted", "declined", "tentative"]
    comment: str | None = None
    reasoning: str
```

### 3.3 Internal Context Models

```python
@dataclass
class ThreadContext:
    """Prepared thread history for LLM prompts."""
    summary: str  # Summary of older messages (empty if none)
    recent_messages: list[str]  # Formatted recent messages
    formatted_history: str  # Combined for prompt injection

@dataclass
class MessageContext:
    """Context for a single message being processed."""
    modality: Literal["email", "sms", "calendar"]
    sender_name: str
    sender_address: str  # Email or phone number
    content: str
    subject: str | None  # Email only
    thread_id: str | None
    message_id: str | None
    timestamp: datetime
    all_recipients: set[str]  # All to/cc addresses or phone numbers
```

---

## 4. ResponseGenerator Class

### 4.1 Class Definition

```python
from langchain_core.language_models.chat_models import BaseChatModel
from ues.client import AsyncUESClient
from src.green.scenarios.schema import ScenarioConfig, CharacterProfile
from src.green.core.message_collector import NewMessages

class ResponseGenerator:
    """Generates character responses to new messages during assessment.
    
    Created per-assessment since it depends on the scenario's character
    profiles. The GreenAgent handles actually scheduling the responses
    in UES after receiving them from this generator.
    
    Processes new messages from all modalities uniformly. Each message
    is checked to see if any recipient character should respond.
    
    Attributes:
        _client: UES client for fetching thread history.
        _scenario: Scenario configuration with character profiles.
        _response_llm: LLM for should-respond checks and response generation.
        _summarization_llm: LLM for summarizing long thread histories.
        _user_character: The CharacterProfile representing the user.
        _random: Random instance for reproducible timing (if seeded).
    """
    
    def __init__(
        self,
        client: AsyncUESClient,
        scenario: ScenarioConfig,
        response_llm: BaseChatModel,
        summarization_llm: BaseChatModel,
        seed: int | None = None,
    ) -> None:
        """Initialize the response generator.
        
        Args:
            client: UES client for fetching thread history.
            scenario: Scenario configuration with character profiles.
            response_llm: LLM for should-respond checks and response generation.
            summarization_llm: LLM for summarizing long thread histories.
            seed: Random seed for reproducible response timing.
        """
        self._client = client
        self._scenario = scenario
        self._response_llm = response_llm
        self._summarization_llm = summarization_llm
        self._user_character = scenario.get_user_character_profile()
        self._random = random.Random(seed)
```

### 4.2 Public Interface

```python
class ResponseGenerator:
    # ... (init from above)
    
    async def process_new_messages(
        self,
        new_messages: NewMessages,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None = None,
    ) -> list[ScheduledResponse]:
        """Generate character responses to new messages.
        
        Processes all new messages collected by NewMessageCollector.
        For each message, identifies recipient characters and determines
        if they should respond.
        
        Args:
            new_messages: Container with new emails, SMS, and calendar events.
            current_time: Current simulation time (for scheduling responses).
            task_updater: Optional emitter for logging warnings/errors.
            
        Returns:
            List of responses to schedule in UES.
        """
```

### 4.3 Message Processing Methods

```python
class ResponseGenerator:
    # ... (public interface from above)
    
    async def _process_email(
        self,
        email: Email,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a single email and generate any character responses.
        
        Flow:
        1. Extract all recipients (to + cc addresses)
        2. For each recipient that is a scenario character (not sender, not user):
           a. Run pre-LLM heuristic checks
           b. Fetch thread history
           c. Check if should respond (LLM)
           d. Generate response content (LLM)
           e. Calculate response timing
           f. Build ScheduledResponse
        """
    
    async def _process_sms(
        self,
        sms: SMSMessage,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a single SMS message and generate any character responses.
        
        Same flow as _process_email but for SMS modality.
        """
    
    async def _process_calendar_event(
        self,
        event: CalendarEvent,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a calendar event and generate RSVP responses from attendees.
        
        Flow:
        1. For each attendee with response_status="needsAction":
           a. Check if attendee is a scenario character (not user)
           b. Run pre-LLM heuristic checks
           c. Decide RSVP status (LLM)
           d. Calculate response timing
           e. Build ScheduledResponse
        """
```

### 4.4 Character Lookup Methods

```python
class ResponseGenerator:
    # ... (message processing from above)
    
    def _find_character_by_email(self, email: str) -> CharacterProfile | None:
        """Find a character by email address.
        
        Uses scenario.get_character_by_email() which searches all characters.
        Returns None if no character has this email.
        """
        return self._scenario.get_character_by_email(email)
    
    def _find_character_by_phone(self, phone: str) -> CharacterProfile | None:
        """Find a character by phone number.
        
        Uses scenario.get_character_by_phone() which searches all characters.
        Returns None if no character has this phone number.
        """
        return self._scenario.get_character_by_phone(phone)
    
    def _is_user_character(self, character: CharacterProfile, character_id: str) -> bool:
        """Check if a character is the user being assisted."""
        return character_id == self._scenario.user_character
    
    def _is_sender(
        self,
        character: CharacterProfile,
        sender_address: str,
        modality: Literal["email", "sms"],
    ) -> bool:
        """Check if a character is the sender of a message."""
        if modality == "email":
            return character.email == sender_address
        else:  # sms
            return character.phone == sender_address
```

### 4.5 Pre-LLM Heuristic Checks

```python
class ResponseGenerator:
    # ... (character lookup from above)
    
    def _should_skip_character(self, character: CharacterProfile) -> bool:
        """Pre-LLM check to skip obvious non-responders.
        
        Checks:
        1. special_instructions contains "no response", "automated", etc.
        2. Response timing indicates never responds (base_delay >= 24h, variance = 0)
        
        Returns:
            True if character should be skipped (won't respond), False otherwise.
        """
        # Check special instructions for non-response keywords
        if character.special_instructions:
            lower = character.special_instructions.lower()
            non_response_keywords = ["no response", "automated", "do not respond", "never responds"]
            if any(kw in lower for kw in non_response_keywords):
                return True
        
        # Check for "never responds" timing pattern
        timing = character.response_timing
        base_hours = timing.base_delay_timedelta.total_seconds() / 3600
        variance_seconds = timing.variance_timedelta.total_seconds()
        if base_hours >= 24 and variance_seconds == 0:
            return True
        
        return False
```

---

## 5. LLM Integration

### 5.1 Should-Respond Check

**Purpose**: Determine if a character would realistically respond to a message.

**System Prompt**:
```
You are evaluating whether a simulated character would realistically respond to a message they received.

Consider:
- The character's personality and special instructions
- The conversation context and history
- Whether the message asks a question, makes a request, or clearly expects a reply
- Common communication patterns (e.g., "Thanks!" or "Got it" messages don't usually warrant a response)
- The communication medium ({modality}) and its conventions

Respond with JSON only: {"should_respond": true/false, "reasoning": "brief explanation"}
```

**User Prompt**:
```
Character Profile:
- Name: {character_name}
- Personality: {character_personality}
- Special Instructions: {special_instructions_section}
- Additional details:
{config_section}

This is a conversation via {modality}.

{thread_summary_section}
Conversation History (most recent last):
{formatted_thread_history}

Latest Message (sent by {sender_name}):
{formatted_latest_message}

Based on this character's personality and the conversation context, would {character_name} send a response to this {modality} message?
```

**Implementation**:
```python
async def _should_respond(
    self,
    character: CharacterProfile,
    message_context: MessageContext,
    thread_context: ThreadContext,
) -> bool:
    """Use LLM to decide if character should respond.
    
    Returns:
        True if character should respond, False otherwise.
    """
    system_prompt = self._build_should_respond_system_prompt(message_context.modality)
    user_prompt = self._build_should_respond_user_prompt(
        character, message_context, thread_context
    )
    
    # Use structured output
    result = await self._response_llm.with_structured_output(
        ShouldRespondResult
    ).ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    return result.should_respond
```

### 5.2 Response Generation

**Purpose**: Generate in-character response content.

**System Prompt**:
```
You are roleplaying as {character_name}.

Character Profile:
- Name: {character_name}
- Personality: {character_personality}
- Special Instructions: {special_instructions_section}
- Additional details:
{config_section}

Write a realistic {modality} response that:
- Matches your character's communication style and personality
- Suits the character's relationship(s) with the conversation participants and message sender
- Is appropriate for the conversation context
- Continues the conversation naturally
- Uses appropriate tone and length for a {modality} message

Respond with just the message content. Do not include headers, subject lines, signatures, or metadata.
```

**User Prompt**:
```
This is a {modality} conversation with {participant_names}.

{thread_summary_section}
Conversation History:
{formatted_thread_history}

{sender_name} just sent this message:
{formatted_latest_message}

Write your response as {character_name}.
```

**Implementation**:
```python
async def _generate_response(
    self,
    character: CharacterProfile,
    message_context: MessageContext,
    thread_context: ThreadContext,
) -> str:
    """Use LLM to generate in-character response content.
    
    Returns:
        The response text (message body only, no headers/signatures).
    """
    system_prompt = self._build_generate_response_system_prompt(
        character, message_context.modality
    )
    user_prompt = self._build_generate_response_user_prompt(
        character, message_context, thread_context
    )
    
    result = await self._response_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    return result.content.strip()
```

### 5.3 Calendar RSVP Decision

**Purpose**: Decide how a character would respond to a calendar invitation.

**System Prompt**:
```
You are deciding how a simulated character would respond to a calendar invitation.

Character Profile:
- Name: {character_name}
- Personality: {character_personality}
- Special Instructions: {special_instructions_section}
- Additional details:
{config_section}

Consider:
- The character's personality and typical availability
- Their relationship with the event organizer
- The nature of the event
- Any scheduling hints in their special instructions or config

Respond with JSON: {"status": "accepted"|"declined"|"tentative", "comment": "optional brief comment", "reasoning": "explanation"}
```

**User Prompt**:
```
Calendar Invitation:
- Title: {event_title}
- Organizer: {organizer}
- Start: {start_time}
- End: {end_time}
- Location: {location}
- Description: {description}
- Attendees: {attendee_list}

How would {character_name} respond to this invitation?
```

**Implementation**:
```python
async def _decide_calendar_rsvp(
    self,
    character: CharacterProfile,
    event: CalendarEvent,
) -> CalendarRSVPResult:
    """Use LLM to decide calendar RSVP status.
    
    Returns:
        CalendarRSVPResult with status and optional comment.
    """
    system_prompt = self._build_calendar_rsvp_system_prompt(character)
    user_prompt = self._build_calendar_rsvp_user_prompt(event, character)
    
    result = await self._response_llm.with_structured_output(
        CalendarRSVPResult
    ).ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    return result
```

---

## 6. Thread History Handling

### 6.1 Retrieval

```python
async def _get_email_thread_history(
    self,
    thread_id: str,
    task_updater: TaskUpdateEmitter | None,
) -> list[Email]:
    """Retrieve full email thread history.
    
    On failure, logs warning via task_updater and returns empty list.
    """
    try:
        result = await self._client.email.query(
            thread_id=thread_id,
            sort_order="asc",
        )
        return result.emails
    except Exception as e:
        if task_updater:
            await task_updater.emit_error(
                error_type="thread_retrieval_warning",
                message=f"Failed to retrieve email thread {thread_id}: {e}",
                recoverable=True,
            )
        return []

async def _get_sms_thread_history(
    self,
    thread_id: str,
    task_updater: TaskUpdateEmitter | None,
) -> list[SMSMessage]:
    """Retrieve full SMS thread history.
    
    On failure, logs warning via task_updater and returns empty list.
    """
    try:
        result = await self._client.sms.query(
            thread_id=thread_id,
            sort_order="asc",
        )
        return result.messages
    except Exception as e:
        if task_updater:
            await task_updater.emit_error(
                error_type="thread_retrieval_warning",
                message=f"Failed to retrieve SMS thread {thread_id}: {e}",
                recoverable=True,
            )
        return []
```

### 6.2 Context Preparation

```python
MAX_THREAD_MESSAGES = 10

async def _prepare_thread_context(
    self,
    history: list[Email] | list[SMSMessage],
    modality: Literal["email", "sms"],
) -> ThreadContext:
    """Prepare thread history for LLM prompts.
    
    If history exceeds MAX_THREAD_MESSAGES, summarizes older messages
    using the summarization LLM.
    """
    if not history:
        return ThreadContext(summary="", recent_messages=[], formatted_history="(No prior messages)")
    
    if len(history) <= MAX_THREAD_MESSAGES:
        summary = ""
        recent = history
    else:
        # Summarize older messages
        older_messages = history[:-MAX_THREAD_MESSAGES]
        summary = await self._summarize_messages(older_messages, modality)
        recent = history[-MAX_THREAD_MESSAGES:]
    
    # Format recent messages
    formatted_messages = [
        self._format_message_for_prompt(msg, modality)
        for msg in recent
    ]
    
    # Build combined history
    if summary:
        formatted_history = f"Summary of earlier messages:\n{summary}\n\nRecent messages:\n" + "\n\n".join(formatted_messages)
    else:
        formatted_history = "\n\n".join(formatted_messages)
    
    return ThreadContext(
        summary=summary,
        recent_messages=formatted_messages,
        formatted_history=formatted_history,
    )
```

### 6.3 Message Formatting

```python
def _format_message_for_prompt(
    self,
    message: Email | SMSMessage,
    modality: Literal["email", "sms"],
) -> str:
    """Format a single message for inclusion in a prompt."""
    if modality == "email":
        timestamp = message.sent_at.strftime("%Y-%m-%d %H:%M")
        recipients = ", ".join(message.to_addresses)
        return (
            f"[{timestamp}] From: {message.from_address} → {recipients}\n"
            f"Subject: {message.subject}\n"
            f"{message.body_text}"
        )
    else:  # SMS
        timestamp = message.sent_at.strftime("%Y-%m-%d %H:%M")
        recipients = ", ".join(message.to_numbers)
        return (
            f"[{timestamp}] From: {message.from_number} → {recipients}\n"
            f"{message.body}"
        )
```

### 6.4 Summarization

```python
async def _summarize_messages(
    self,
    messages: list[Email] | list[SMSMessage],
    modality: Literal["email", "sms"],
) -> str:
    """Summarize a list of messages using the summarization LLM.
    
    Used when thread history exceeds MAX_THREAD_MESSAGES to preserve
    context while controlling prompt size.
    """
    formatted = "\n\n".join(
        self._format_message_for_prompt(msg, modality)
        for msg in messages
    )
    
    prompt = f"""Summarize the following {modality} conversation history in 2-3 sentences. 
Focus on key topics discussed, decisions made, and any outstanding questions or requests.

Conversation:
{formatted}

Summary:"""
    
    result = await self._summarization_llm.ainvoke([HumanMessage(content=prompt)])
    return result.content.strip()
```

---

## 7. Response Scheduling

### 7.1 Timing Calculation

```python
def _calculate_response_time(
    self,
    timing: ResponseTiming,
    reference_time: datetime,
) -> datetime:
    """Calculate when a response should be scheduled.
    
    Uses the character's ResponseTiming config with random variance
    for realistic human behavior patterns.
    
    Args:
        timing: Character's response timing configuration.
        reference_time: When the triggering message was sent/received.
        
    Returns:
        Datetime when the response should appear in UES.
    """
    base_seconds = timing.base_delay_timedelta.total_seconds()
    variance_seconds = timing.variance_timedelta.total_seconds()
    
    min_delay = max(0, base_seconds - variance_seconds)
    max_delay = base_seconds + variance_seconds
    delay_seconds = self._random.uniform(min_delay, max_delay)
    
    return reference_time + timedelta(seconds=delay_seconds)
```

### 7.2 Email Subject Derivation

```python
def _derive_email_subject(self, original_subject: str) -> str:
    """Derive reply subject line following email conventions.
    
    Adds "Re: " prefix if not already present.
    """
    if original_subject.lower().startswith("re:"):
        return original_subject
    return f"Re: {original_subject}"
```

### 7.3 Building ScheduledResponse Objects

```python
def _build_email_response(
    self,
    character: CharacterProfile,
    character_id: str,
    email: Email,
    content: str,
    scheduled_time: datetime,
) -> ScheduledResponse:
    """Build a ScheduledResponse for an email reply."""
    # Reply goes to sender, and CC goes to other recipients minus the responder
    to_addresses = [email.from_address]
    cc_addresses = [
        addr for addr in email.to_addresses + email.cc_addresses
        if addr != character.email and addr != email.from_address
    ]
    
    # Build references chain for proper threading
    references = list(email.references) if email.references else []
    if email.message_id and email.message_id not in references:
        references.append(email.message_id)
    
    return ScheduledResponse(
        character_id=character_id,
        modality="email",
        content=content,
        scheduled_time=scheduled_time,
        email_subject=self._derive_email_subject(email.subject),
        email_to_addresses=to_addresses,
        email_cc_addresses=cc_addresses if cc_addresses else None,
        email_thread_id=email.thread_id,
        email_in_reply_to=email.message_id,
        email_references=references if references else None,
    )

def _build_sms_response(
    self,
    character: CharacterProfile,
    character_id: str,
    sms: SMSMessage,
    content: str,
    scheduled_time: datetime,
) -> ScheduledResponse:
    """Build a ScheduledResponse for an SMS reply."""
    # Reply goes to sender and other recipients (group chat)
    to_numbers = [sms.from_number] + [
        num for num in sms.to_numbers
        if num != character.phone and num != sms.from_number
    ]
    
    return ScheduledResponse(
        character_id=character_id,
        modality="sms",
        content=content,
        scheduled_time=scheduled_time,
        sms_to_numbers=to_numbers,
        sms_thread_id=sms.thread_id,
    )

def _build_calendar_response(
    self,
    character: CharacterProfile,
    character_id: str,
    event: CalendarEvent,
    rsvp_result: CalendarRSVPResult,
    scheduled_time: datetime,
) -> ScheduledResponse:
    """Build a ScheduledResponse for a calendar RSVP."""
    return ScheduledResponse(
        character_id=character_id,
        modality="calendar",
        content=rsvp_result.comment or "",
        scheduled_time=scheduled_time,
        calendar_event_id=event.event_id,
        calendar_id=event.calendar_id,
        rsvp_status=rsvp_result.status,
    )
```

---

## 8. Error Handling

### 8.1 Strategy

**Principle**: Degrade gracefully and continue processing. Log warnings via task updates for observability.

| Error Type | Handling |
|------------|----------|
| Thread retrieval fails | Log warning, proceed with empty history |
| LLM should_respond fails | Log warning, skip this character (don't respond) |
| LLM generate_response fails | Log warning, skip this response |
| LLM calendar_rsvp fails | Log warning, skip this RSVP |
| Character lookup fails | Skip silently (expected for non-character addresses) |

### 8.2 Warning Emission

```python
async def _emit_warning(
    self,
    task_updater: TaskUpdateEmitter | None,
    warning_type: str,
    message: str,
    details: dict | None = None,
) -> None:
    """Emit a warning via task updater if available.
    
    Warnings don't stop processing but are logged for debugging.
    """
    if task_updater:
        await task_updater.emit_error(
            error_type=warning_type,
            message=message,
            recoverable=True,
            details=details,
        )
```

### 8.3 Exception Classes

```python
class ResponseGeneratorError(Exception):
    """Base exception for response generator errors."""
    pass

class ResponseGenerationError(ResponseGeneratorError):
    """Error during response content generation."""
    def __init__(self, character_id: str, modality: str, cause: Exception):
        self.character_id = character_id
        self.modality = modality
        self.cause = cause
        super().__init__(
            f"Failed to generate {modality} response for {character_id}: {cause}"
        )
```

---

## 9. Configuration

### 9.1 GreenAgentConfig Updates

Add the following fields to `GreenAgentConfig` in `src/common/agentbeats/config.py`:

```python
class GreenAgentConfig(AgentBeatsConfig):
    """Configuration specific to Green agents."""
    
    # ... existing fields ...
    
    # Response generation LLM (used for should-respond and generate-response)
    response_generation_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for response generation and should-respond checks",
    )
    
    # Summarization LLM (used for summarizing long thread histories)
    summarization_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for summarizing thread histories (can be cheaper/faster)",
    )
```

### 9.2 LLM Instantiation in GreenAgent

```python
class GreenAgent:
    def __init__(self, config: GreenAgentConfig, ...):
        # ... other initialization ...
        
        # Create LLMs for response generation
        self._response_llm = LLMFactory.create(
            model=config.response_generation_model,
            temperature=0.7,
            seed=config.seed,
        )
        self._summarization_llm = LLMFactory.create(
            model=config.summarization_model,
            temperature=0.3,  # Lower temperature for factual summarization
            seed=config.seed,
        )
```

---

## 10. Implementation Plan

### 10.1 File Structure

```
src/green/
├── response_generator.py    # Main ResponseGenerator class
├── response_models.py       # ScheduledResponse and LLM response models
└── prompts/
    └── response_prompts.py  # Prompt templates as constants
```

### 10.2 Implementation Tasks

#### Task 1: Response Models (`response_models.py`)
**Estimated time**: 1 hour

1. Create `ScheduledResponse` dataclass
2. Create `ShouldRespondResult` Pydantic model
3. Create `CalendarRSVPResult` Pydantic model
4. Create `ThreadContext` dataclass
5. Create `MessageContext` dataclass
6. Write unit tests for all models

#### Task 2: Prompt Templates (`prompts/response_prompts.py`)
**Estimated time**: 30 minutes

1. Define `SHOULD_RESPOND_SYSTEM_PROMPT` template
2. Define `SHOULD_RESPOND_USER_PROMPT` template
3. Define `GENERATE_RESPONSE_SYSTEM_PROMPT` template
4. Define `GENERATE_RESPONSE_USER_PROMPT` template
5. Define `CALENDAR_RSVP_SYSTEM_PROMPT` template
6. Define `CALENDAR_RSVP_USER_PROMPT` template
7. Define `SUMMARIZE_THREAD_PROMPT` template

#### Task 3: ResponseGenerator Core (`response_generator.py`)
**Estimated time**: 3 hours

1. Implement `__init__` with dependency injection
2. Implement `process_new_messages` main entry point
3. Implement `_process_email` method
4. Implement `_process_sms` method
5. Implement `_process_calendar_event` method
6. Implement character lookup methods
7. Implement `_should_skip_character` heuristic checks
8. Write integration tests with mocked LLMs

#### Task 4: Thread History Handling
**Estimated time**: 1.5 hours

1. Implement `_get_email_thread_history` with error handling
2. Implement `_get_sms_thread_history` with error handling
3. Implement `_prepare_thread_context` with summarization
4. Implement `_format_message_for_prompt` for email/SMS
5. Implement `_summarize_messages` using summarization LLM
6. Write tests for thread history handling

#### Task 5: LLM Integration
**Estimated time**: 2 hours

1. Implement `_should_respond` with structured output
2. Implement `_generate_response` 
3. Implement `_decide_calendar_rsvp` with structured output
4. Implement prompt building methods
5. Write tests with mocked LLM responses

#### Task 6: Response Building
**Estimated time**: 1 hour

1. Implement `_calculate_response_time`
2. Implement `_derive_email_subject`
3. Implement `_build_email_response`
4. Implement `_build_sms_response`
5. Implement `_build_calendar_response`
6. Write tests for response building

#### Task 7: Configuration Updates
**Estimated time**: 30 minutes

1. Add `response_generation_model` to `GreenAgentConfig`
2. Add `summarization_model` to `GreenAgentConfig`
3. Update config tests
4. Update config documentation

#### Task 8: GreenAgent Integration (Scheduling)
**Estimated time**: 1.5 hours

1. Add response scheduling methods to `GreenAgent`:
   - `_schedule_email_response(response: ScheduledResponse)`
   - `_schedule_sms_response(response: ScheduledResponse)`
   - `_schedule_calendar_rsvp(response: ScheduledResponse)`
2. Integrate `ResponseGenerator` into turn loop
3. Write integration tests

### 10.3 Dependencies

**Task Dependencies**:
```
Task 1 (Models) ──────────┐
                          ├──▶ Task 3 (Core) ──▶ Task 8 (Integration)
Task 2 (Prompts) ─────────┤         │
                          │         ▼
Task 7 (Config) ──────────┘   Task 4 (Thread)
                              Task 5 (LLM)
                              Task 6 (Building)
```

**External Dependencies**:
- UES v0.2.1+ required for `calendar.respond_to_event()` method

### 10.4 Total Estimated Time

| Task | Time |
|------|------|
| Task 1: Response Models | 1h |
| Task 2: Prompt Templates | 0.5h |
| Task 3: ResponseGenerator Core | 3h |
| Task 4: Thread History | 1.5h |
| Task 5: LLM Integration | 2h |
| Task 6: Response Building | 1h |
| Task 7: Configuration | 0.5h |
| Task 8: GreenAgent Integration | 1.5h |
| **Total** | **11h** |

---

## 11. Testing Strategy

### 11.1 Unit Tests

**Models** (`tests/green/test_response_models.py`):
- `ScheduledResponse` field validation
- `ShouldRespondResult` JSON parsing
- `CalendarRSVPResult` validation
- `ThreadContext` formatting

**Prompts** (`tests/green/test_response_prompts.py`):
- Template string formatting
- Variable substitution
- Edge cases (empty values, special characters)

**Response Building** (`tests/green/test_response_generator.py`):
- `_calculate_response_time` with various timings
- `_derive_email_subject` with/without existing "Re:"
- `_build_email_response` field mapping
- `_build_sms_response` recipient handling
- `_build_calendar_response` RSVP mapping

**Heuristics** (`tests/green/test_response_generator.py`):
- `_should_skip_character` with various special_instructions
- `_should_skip_character` with various timing configs

### 11.2 Integration Tests

**Thread History** (with mocked UES client):
- Successful thread retrieval
- Thread retrieval failure (graceful degradation)
- Thread summarization for long histories
- Empty thread handling

**LLM Integration** (with mocked LLMs):
- Should-respond returns True/False appropriately
- Response generation produces valid content
- Calendar RSVP returns valid status
- LLM error handling (graceful degradation)

**Full Pipeline** (with mocked UES and LLMs):
- Process email → generates response
- Process SMS → generates response
- Process calendar event → generates RSVP
- Multiple recipients → multiple responses
- User character excluded from responses
- Sender excluded from responses

### 11.3 Test Fixtures

```python
@pytest.fixture
def sample_scenario() -> ScenarioConfig:
    """Create a sample scenario with test characters."""
    
@pytest.fixture
def sample_email() -> Email:
    """Create a sample email for testing."""

@pytest.fixture
def sample_sms() -> SMSMessage:
    """Create a sample SMS for testing."""

@pytest.fixture
def sample_calendar_event() -> CalendarEvent:
    """Create a sample calendar event for testing."""

@pytest.fixture
def mock_response_llm():
    """Create a mock LLM for response generation."""

@pytest.fixture
def mock_summarization_llm():
    """Create a mock LLM for summarization."""
```

---

## Appendix A: UES Client Reference

### Calendar RSVP Method

```python
async def respond_to_event(
    self,
    event_id: str,
    attendee_email: str,
    response: AttendeeResponse,  # Literal["accepted", "declined", "tentative", "needsAction"]
    comment: str | None = None,
) -> ModalityActionResponse:
    """Respond to a calendar event invitation (RSVP).
    
    Args:
        event_id: Event ID to respond to.
        attendee_email: Email of the attendee responding.
        response: Response status.
        comment: Optional comment with the response.
    
    Returns:
        Action response with execution status.
    """
```

### Email Receive Method

```python
async def receive(
    self,
    from_address: str,
    to_addresses: list[str],
    subject: str,
    body_text: str,
    cc_addresses: list[str] | None = None,
    thread_id: str | None = None,
    in_reply_to: str | None = None,
    references: list[str] | None = None,
    sent_at: datetime | None = None,
    # ... other optional fields
) -> ModalityActionResponse:
    """Simulate receiving an email (for Green agent use)."""
```

### SMS Receive Method

```python
async def receive(
    self,
    from_number: str,
    to_numbers: list[str],
    body: str,
    thread_id: str | None = None,
    sent_at: datetime | None = None,
    # ... other optional fields
) -> ModalityActionResponse:
    """Simulate receiving an SMS (for Green agent use)."""
```

---

## Appendix B: Character Profile Reference

From `src/green/scenarios/schema.py`:

```python
class CharacterProfile(BaseModel):
    """Profile for a simulated character."""
    
    name: str                           # Display name
    relationships: dict[str, str]       # {character_name: relationship_description}
    personality: str                    # Personality and communication style
    email: str | None                   # Email address (optional)
    phone: str | None                   # Phone number (optional)
    response_timing: ResponseTiming     # Timing config
    special_instructions: str | None    # Additional instructions
    config: dict[str, Any] | None       # Character-specific data

class ResponseTiming(BaseModel):
    """Timing configuration for responses."""
    
    base_delay: str     # ISO 8601 duration (e.g., "PT30M")
    variance: str       # ISO 8601 duration (e.g., "PT10M")
    
    @property
    def base_delay_timedelta(self) -> timedelta: ...
    
    @property
    def variance_timedelta(self) -> timedelta: ...
```
