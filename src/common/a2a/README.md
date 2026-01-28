# A2A Helper Module

This module provides simplified wrappers and helper functions for working with the [A2A (Agent2Agent) Protocol](https://a2a-protocol.org/). It builds on the official [a2a-sdk](https://github.com/a2aproject/a2a-python) to provide convenient, type-safe utilities for building both Green and Purple agents.

## Overview

The A2A protocol enables interoperability between AI agents through a standardized communication format. This module provides:

| Module | Purpose |
|--------|---------|
| `agent_card.py` | Fluent builder for creating AgentCard objects |
| `server.py` | Simplified A2A server setup with sensible defaults |
| `client.py` | Client wrapper for communicating with A2A agents |
| `messages.py` | Message creation and parsing utilities |
| `tasks.py` | Task state management helpers |
| `artifacts.py` | Artifact creation and extraction utilities |

## Installation

The module requires the following dependencies (already configured in `pyproject.toml`):

```toml
[project.dependencies]
a2a-sdk = { version = ">=0.3.22", extras = ["http-server"] }
httpx = ">=0.28.0"
pydantic = ">=2.0"
uvicorn = ">=0.30"
```

Install with:

```bash
uv sync
```

## Quick Start

### Creating an Agent Card

```python
from src.common.a2a import AgentCardBuilder, create_skill

# Create skills for your agent
email_skill = create_skill(
    id="email_triage",
    name="Email Triage",
    description="Triages and prioritizes incoming emails",
    tags=["email", "productivity"],
    examples=["Triage my unread emails", "Find urgent messages"],
)

calendar_skill = create_skill(
    id="calendar_management",
    name="Calendar Management",
    description="Manages calendar events and scheduling",
    tags=["calendar", "scheduling"],
)

# Build the agent card
card = (
    AgentCardBuilder()
    .with_name("Personal Assistant")
    .with_description("AI assistant for productivity tasks")
    .with_url("http://localhost:8000")
    .with_version("1.0.0")
    .with_skill(email_skill)
    .with_skill(calendar_skill)
    .with_capabilities(streaming=True, push_notifications=False)
    .with_provider("My Organization", "https://example.org")
    .build()
)
```

### Setting Up a Server

```python
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue

from src.common.a2a import A2AServer, AgentCardBuilder, create_skill

class MyExecutor(AgentExecutor):
    """Custom executor that handles incoming requests."""
    
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # Handle the request
        # Use event_queue to emit status updates and artifacts
        pass
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # Handle cancellation
        pass

# Create the agent card
card = (
    AgentCardBuilder()
    .with_name("My Agent")
    .with_description("An example A2A agent")
    .with_url("http://localhost:8000")
    .build()
)

# Create and run the server
server = A2AServer(
    agent_card=card,
    executor=MyExecutor(),
    host="0.0.0.0",
    port=8000,
)

# Run blocking
server.run()

# Or get the ASGI app for testing/mounting
app = server.get_asgi_app()
```

### Communicating with Agents (Client)

```python
from src.common.a2a import A2AClientWrapper, A2AClientError

async def communicate_with_agent():
    async with A2AClientWrapper("http://localhost:9000") as client:
        # Send a text message
        task = await client.send_text("Help me with my calendar")
        print(f"Task ID: {task.id}")
        print(f"Task State: {task.status.state}")
        
        # Send structured data
        task = await client.send_data({
            "action": "schedule_meeting",
            "participants": ["alice@example.com", "bob@example.com"],
            "duration_minutes": 30,
        })
        
        # Get task status
        task = await client.get_task(task.id)
        
        # Cancel a task
        await client.cancel_task(task.id)
```

### Working with Messages

```python
from a2a.types import Role
from src.common.a2a import (
    text_part,
    data_part,
    file_part,
    create_text_message,
    create_data_message,
    create_multipart_message,
    get_text_content,
    get_data_content,
    new_agent_text_message,
)

# Create simple messages
user_msg = create_text_message("Hello!", role=Role.user)
agent_msg = new_agent_text_message("How can I help you?")

# Create a message with data
data_msg = create_data_message(
    {"status": "success", "items": [1, 2, 3]},
    role=Role.agent,
)

# Create a multipart message
multipart = create_multipart_message([
    text_part("Here is the document:"),
    file_part(b"file content", "report.txt", "text/plain"),
    data_part({"pages": 10}),
])

# Extract content from messages
text = get_text_content(user_msg)  # "Hello!"
data = get_data_content(data_msg)  # {"status": "success", "items": [1, 2, 3]}
```

### Managing Task State

```python
from a2a.types import TaskState
from src.common.a2a import (
    is_terminal_state,
    is_completed,
    is_failed,
    is_working,
    get_task_state,
    create_status_update,
    create_completed_status,
    create_failed_status,
    create_working_status,
)

# Check task state
if is_terminal_state(task.status.state):
    if is_completed(task.status.state):
        print("Task completed successfully!")
    elif is_failed(task.status.state):
        print("Task failed")

# Create status updates for event queue
working = create_working_status("Processing your request...")
completed = create_completed_status("Done!")
failed = create_failed_status("Something went wrong", error_code="E001")

# Custom status update
update = create_status_update(
    state=TaskState.working,
    message="Step 2 of 5...",
    metadata={"progress": 40},
)
```

### Creating and Parsing Artifacts

```python
from src.common.a2a import (
    create_artifact,
    create_json_artifact,
    create_text_artifact,
    extract_json_from_artifact,
    extract_text_from_artifact,
    artifact_to_json_string,
    artifact_from_json_string,
)

# Create artifacts
json_artifact = create_json_artifact(
    {"score": 95, "passed": True},
    name="evaluation_result",
    description="Assessment evaluation results",
)

text_artifact = create_text_artifact(
    "Task completed successfully.",
    name="completion_message",
)

# Flexible artifact creation
mixed_artifact = create_artifact(
    text="Summary of results",
    data={"detailed": "data"},
    name="report",
)

# Extract content
data = extract_json_from_artifact(json_artifact)  # {"score": 95, "passed": True}
text = extract_text_from_artifact(text_artifact)  # "Task completed successfully."

# Serialize for storage/transmission
json_str = artifact_to_json_string(json_artifact)
restored = artifact_from_json_string(json_str)
```

## API Reference

### agent_card.py

| Function/Class | Description |
|----------------|-------------|
| `create_skill(id, name, description, ...)` | Create an AgentSkill with sensible defaults |
| `AgentCardBuilder` | Fluent builder for AgentCard objects |

**AgentCardBuilder Methods:**
- `with_name(name)` - Set agent name (required)
- `with_description(description)` - Set description (required)
- `with_url(url)` - Set service URL (required)
- `with_version(version)` - Set version string
- `with_skill(skill)` / `with_skills(skills)` - Add skills
- `with_capabilities(streaming, push_notifications, state_transition_history)` - Set capabilities
- `with_default_input_modes(modes)` / `with_default_output_modes(modes)` - Set media types
- `with_provider(organization, url)` - Set provider info
- `with_documentation_url(url)` / `with_icon_url(url)` - Set optional URLs
- `with_supports_authenticated_extended_card(value)` - Set extended card support
- `copy()` - Create an independent copy of the builder
- `build()` - Build the AgentCard (validates required fields)

### server.py

| Class | Description |
|-------|-------------|
| `A2AServer` | Simplified wrapper for A2A HTTP server setup |

**A2AServer Methods:**
- `__init__(agent_card, executor, host, port, task_store, extended_agent_card)` - Initialize server
- `run(log_level)` - Run server (blocking)
- `start()` - Start server asynchronously
- `get_asgi_app()` - Get underlying ASGI application for testing

### client.py

| Class/Exception | Description |
|-----------------|-------------|
| `A2AClientWrapper` | Async context manager for A2A client communication |
| `A2AClientError` | Exception for client errors (with optional error code) |

**A2AClientWrapper Methods:**
- `__init__(agent_url, httpx_client, timeout)` - Initialize client
- `send_text(text, task_id, context_id)` - Send text message
- `send_data(data, task_id, context_id)` - Send structured data
- `send_message(message, task_id, context_id)` - Send custom message
- `send_streaming_message(message, task_id, context_id)` - Send with streaming response
- `get_task(task_id)` - Get task by ID
- `cancel_task(task_id)` - Cancel a task

### messages.py

| Function | Description |
|----------|-------------|
| `text_part(text, metadata)` | Create a TextPart |
| `data_part(data, metadata)` | Create a DataPart |
| `file_part(content, filename, media_type, metadata)` | Create a FilePart with inline bytes |
| `create_text_message(text, role, ...)` | Create a text message |
| `create_data_message(data, role, ...)` | Create a data message |
| `create_multipart_message(parts, role, ...)` | Create a multipart message |
| `get_text_content(message)` | Extract concatenated text from message |
| `get_data_content(message)` | Extract first data part from message |
| `get_all_text_parts(message)` | Get all text parts as list |
| `get_all_data_parts(message)` | Get all data parts as list |
| `new_agent_text_message(text, task_id)` | Convenience for agent text messages |
| `new_user_text_message(text, task_id)` | Convenience for user text messages |

### tasks.py

| Function | Description |
|----------|-------------|
| `is_terminal_state(state)` | Check if state is terminal (completed/failed/canceled/rejected) |
| `is_completed(state)` / `is_failed(state)` / etc. | Check specific states |
| `get_task_state(task)` | Extract state from task object |
| `create_status_update(state, message, final, metadata)` | Create TaskStatusUpdateEvent |
| `create_completed_status(message)` | Create completed status update |
| `create_failed_status(message, error_code)` | Create failed status update |
| `create_working_status(message)` | Create working status update |
| `create_input_required_status(message)` | Create input-required status update |
| `create_artifact_update(artifact, append, last_chunk, metadata)` | Create TaskArtifactUpdateEvent |

### artifacts.py

| Function | Description |
|----------|-------------|
| `create_artifact(artifact_id, name, description, parts, text, data, metadata)` | Create artifact flexibly |
| `create_json_artifact(data, name, description, ...)` | Create artifact with JSON data |
| `create_text_artifact(text, name, description, ...)` | Create artifact with text |
| `extract_json_from_artifact(artifact)` | Extract first DataPart's data |
| `extract_text_from_artifact(artifact)` | Extract first TextPart's text |
| `extract_all_json_from_artifact(artifact)` | Extract all DataParts' data |
| `extract_all_text_from_artifact(artifact)` | Extract all TextParts' text |
| `artifact_to_json_string(artifact, indent)` | Serialize artifact to JSON |
| `artifact_from_json_string(json_str)` | Deserialize artifact from JSON |

## Important Notes

### A2A SDK Field Naming Convention

The A2A SDK uses **camelCase for constructor parameters** but **snake_case for attribute access**:

```python
# Creating objects - use camelCase
message = Message(
    messageId="msg-123",
    taskId="task-456",
    ...
)

# Reading attributes - use snake_case
print(message.message_id)  # "msg-123"
print(message.task_id)     # "task-456"
```

This module's helper functions accept **snake_case parameters** for consistency with Python conventions and handle the conversion internally.

### Part Wrapping

The A2A SDK wraps `TextPart`, `DataPart`, and `FilePart` objects in a `Part` container:

```python
# When iterating parts, you may need to unwrap:
for part in message.parts:
    if hasattr(part, "root"):
        inner = part.root
    else:
        inner = part
    
    if isinstance(inner, TextPart):
        print(inner.text)
```

The helper functions in this module (like `get_text_content`, `extract_json_from_artifact`) handle this unwrapping automatically.

## Testing

Run the module tests with:

```bash
uv run pytest tests/common/a2a/ -v
```

The module has comprehensive test coverage with 107 tests covering all functionality.

## Related Resources

- [A2A Protocol Specification](https://a2a-protocol.org/latest/)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python)
- [AgentBeats Competition Documentation](https://docs.agentbeats.org/)
