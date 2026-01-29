# Response Generation Design Notes

This document summarizes design discussions for the Green Agent's response generation system (Phase 3.5 of the implementation plan). While the implementation direction may change, these design decisions and learnings remain valuable.

**Date**: January 29, 2026

---

## Overview

The `ResponseGenerator` component is responsible for generating in-character responses from simulated contacts (characters) when the Purple agent sends messages. It determines:
1. Whether a character should respond to a given action
2. What the response content should be
3. When the response should be delivered

---

## Key Design Decisions

### 1. Pre-LLM Heuristic Checks

**Decision**: Perform simple programmatic checks *before* calling `_should_respond` LLM to short-circuit obvious non-response cases.

**Rationale**: Reduces LLM inference time and cost for clear-cut cases.

**Checks implemented**:
| Check | Result | Rationale |
|-------|--------|-----------|
| Action not in `{email.send, email.reply, sms.send, calendar.create}` | Skip | Not a communication action |
| Character not in recipient list | Skip | They weren't contacted |
| `special_instructions` contains "no response", "automated", etc. | Skip | Explicit non-responder |
| `base_delay` ≥ 24h AND `variance` = 0 | Skip | "Never responds" timing pattern |

**Not checked programmatically**: Message content semantics (e.g., "Thanks!" messages). Left to LLM to handle nuanced cases.

### 2. Thread History Handling

**Decision**: Retrieve full thread history, but limit to last 10 messages with LLM-generated summary of older messages.

**Rationale**: 
- Thread context is essential for realistic response decisions
- Unbounded history creates prompt size issues
- Summarization preserves important context while controlling tokens

**Implementation approach**:
```python
MAX_THREAD_MESSAGES = 10

if len(thread_history) <= MAX_THREAD_MESSAGES:
    summary = ""
    recent = thread_history
else:
    summary = await _summarize_messages(thread_history[:-10])
    recent = thread_history[-10:]
```

### 3. Thread History Presentation Format

**Decision**: Place all thread history in the user prompt as formatted text (Option A), not as LLM message history.

**Rationale**:
- Multi-party threads (group emails, group SMS) don't map to two-role LLM conversation format
- Explicit timestamps and sender info provides clearer context
- Consistent format works for both `should_respond` and `generate_response`
- Simpler implementation

**Format**:
```
[2026-01-28 09:15] From: alice.chen@company.com → user@company.com
Subject: Urgent: Q4 Report Review
Need your feedback on the attached report by EOD.

[2026-01-28 09:45] From: user@company.com → alice.chen@company.com
Subject: Re: Urgent: Q4 Report Review
I'll review it right away and get back to you within the hour.
```

### 4. Response Timing

**Decision**: Programmatic random sampling using `ResponseTiming` config, not LLM-decided.

**Rationale**:
- Simpler and faster (no extra LLM call)
- More reproducible given same random seed
- Response delay reflects character availability patterns, not message content
- `ResponseTiming` model already designed for this

**Implementation**:
```python
def _calculate_response_time(timing: ResponseTiming, current_time: datetime) -> datetime:
    base_seconds = timing.base_delay_timedelta.total_seconds()
    variance_seconds = timing.variance_timedelta.total_seconds()
    
    min_delay = max(0, base_seconds - variance_seconds)
    max_delay = base_seconds + variance_seconds
    delay_seconds = random.uniform(min_delay, max_delay)
    
    return current_time + timedelta(seconds=delay_seconds)
```

### 5. Email Subject Lines

**Decision**: Programmatically derive subject lines following real-world conventions.

**Implementation**:
```python
def _derive_email_subject(original_subject: str, is_reply: bool = True) -> str:
    if not is_reply:
        return original_subject
    if original_subject.lower().startswith("re:"):
        return original_subject
    return f"Re: {original_subject}"
```

### 6. User Name in Prompts

**Decision**: Add `user_character` field to `ScenarioConfig` that references a character entry representing the user.

**Rationale**: The user being assisted has a name, email, phone that characters interact with. This is scenario-specific configuration.

**Implementation**: Added to scenario schema with helper method `get_user_character_profile()`.

### 7. Character Config in Prompts

**Decision**: Include `CharacterProfile.config` in system prompts as raw JSON.

**Rationale**: Config structure is unpredictable (vendor pricing, service lists, etc.). Raw JSON lets the LLM interpret it contextually.

**Format in prompt**:
```
Character Configuration:
```json
{"company": "Acme Corporation", "services": ["software licenses", "cloud hosting"]}
```
```

### 8. Modality Awareness

**Decision**: Explicitly mention the communication modality (email/SMS/calendar) in both system and user prompts.

**Rationale**: Email vs SMS have different conventions for length, formality, response expectations.

---

## Prompt Templates

### Should-Respond Check

**System Prompt**:
```
You are evaluating whether a simulated character would realistically respond to a message they received.

Consider:
- The character's personality and their relationship to the message sender
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
- Relationship to {user_name}: {character_role}
- Personality: {character_personality}
{special_instructions_section}
{config_section}

This is a conversation via {modality}.

{thread_summary_section}
Conversation History (most recent last):
{formatted_thread_history}

Latest Message (sent by {user_name}):
{formatted_latest_message}

Based on this character's personality and the conversation context, would {character_name} send a response to this {modality} message?
```

### Response Generation

**System Prompt**:
```
You are roleplaying as {character_name}, {user_name}'s {character_role}.

Your personality: {character_personality}
{special_instructions_section}
{config_section}

Write a realistic {modality} response that:
- Matches your character's communication style and personality
- Is appropriate for the conversation context
- Continues the conversation naturally
- Uses appropriate tone and length for a {modality} message

Respond with just the message content. Do not include headers, subject lines, signatures, or metadata.
```

**User Prompt**:
```
This is a {modality} conversation with {user_name}.

{thread_summary_section}
Conversation History:
{formatted_thread_history}

{user_name} just sent you this message:
{formatted_latest_message}

Write your response as {character_name}.
```

---

## Actions That Trigger Response Checking

```python
RESPONSE_TRIGGERING_ACTIONS = {
    "email.send",
    "email.reply",
    "sms.send",
    "calendar.create",  # For meeting invites with attendees
}
```

For each action:
1. Extract potential recipients based on action type and parameters
2. Match recipients against scenario characters
3. For each matched character, run through response generation flow

---

## UES Data Structures Reference

### Email Thread Retrieval
```python
result = await client.email.query(thread_id=thread_id, sort_order="asc")
# result.emails: list of Email objects
```

**Email fields**: `message_id`, `thread_id`, `from_address`, `to_addresses`, `cc_addresses`, `subject`, `body_text`, `sent_at`, `received_at`, `in_reply_to`

### SMS Conversation Retrieval
```python
result = await client.sms.query(thread_id=thread_id, sort_order="asc")
# result.messages: list of SMSMessage objects
```

**SMSMessage fields**: `message_id`, `thread_id`, `from_number`, `to_numbers`, `body`, `direction`, `sent_at`

### Calendar Event (for invites)
```python
event = await client.calendar.get(event_id)
```

**CalendarEvent attendee fields**: `email`, `display_name`, `optional`, `response_status`

**Note**: UES RSVP endpoints not yet implemented. Calendar responses would need to update attendee status directly via `calendar.update()`.

---

## Output Model

```python
@dataclass
class ScheduledResponse:
    """A response to be scheduled in UES."""
    character: CharacterProfile
    modality: Literal["email", "sms", "calendar"]
    content: str
    scheduled_time: datetime
    
    # Email-specific
    subject: str | None = None
    thread_id: str | None = None
    in_reply_to: str | None = None
    
    # Calendar-specific (RSVP)
    event_id: str | None = None
    rsvp_status: Literal["accepted", "declined", "tentative"] | None = None
```

---

## Open Questions (Unresolved)

1. **LLM instance sharing**: Should `ResponseGenerator` create its own LLMs or receive them from `GreenAgent`? Implementation plan suggests `GreenAgent` owns LLMs.

2. **Summarization LLM**: Should summarization use a separate cheaper/faster model, or share with response checking?

3. **Calendar RSVP mechanics**: Without UES RSVP endpoints, options are:
   - Update event's attendee list directly
   - Store RSVPs separately for evaluators
   - Skip until UES supports it

4. **Error handling for thread retrieval**: Decided to degrade gracefully (proceed with empty history), but exact logging/warning behavior TBD.

---

## Processing Flow

```
For each action in turn_complete.actions:
    │
    ├─► _get_potential_recipients(action, scenario)
    │   └─► Returns [(character, modality), ...]
    │
    └─► For each (character, modality):
        │
        ├─► _should_skip_character(character)  [programmatic]
        │   └─► If True: skip, continue
        │
        ├─► _get_thread_history(action, modality)
        │
        ├─► _prepare_thread_context(history, modality)
        │   └─► Returns (summary, recent_messages)
        │
        ├─► _should_respond(...)  [LLM call]
        │   └─► If False: skip, continue
        │
        ├─► _generate_response(...)  [LLM call]
        │
        ├─► _calculate_response_time(timing, current_time)
        │
        └─► Create ScheduledResponse and add to results
```

---

## Related Files

- `src/green/scenarios/schema.py` - `CharacterProfile`, `ResponseTiming`, `ScenarioConfig`
- `src/common/agentbeats/messages.py` - `ActionLogEntry`
- `src/green/llm_config.py` - LLM factory
- `scenarios/email_triage_basic/scenario.json` - Example scenario with characters
