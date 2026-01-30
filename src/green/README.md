# Green Agent Module

This module implements the Green Agent (Evaluator) for the AgentBeats competition. The Green Agent orchestrates assessments of AI personal assistants using the User Environment Simulator (UES).

## Overview

The Green Agent:

1. **Initializes** a UES environment with scenario configuration
2. **Communicates** with Purple agents via the A2A protocol
3. **Generates responses** from simulated characters during assessments
4. **Evaluates** Purple agent performance against scenario criteria
5. **Reports results** as A2A artifacts

## Module Structure

```
src/green/
├── __init__.py
├── scenarios/              # Scenario schema and loader (✅ complete)
│   ├── schema.py          # Pydantic models for scenarios
│   ├── loader.py          # Scenario discovery and loading
│   └── README.md          # Scenario module documentation
├── prompts/                # LLM prompt templates (✅ complete)
│   ├── __init__.py
│   └── response_prompts.py # Templates for response generation
├── assessment/             # Assessment orchestration (🚧 in progress)
│   └── ...
├── action_log.py           # Action log builder (✅ complete)
├── llm_config.py           # LLM factory for multiple providers (✅ complete)
├── message_collector.py    # New message collector (✅ complete)
├── response_models.py      # Response data models (✅ complete)
└── response_generator.py   # Character response generation (✅ complete)
```

## Key Components

### Response Generation

The response generation system creates in-character responses from simulated contacts during assessments. It transforms UES from a static simulation into a dynamic, interactive environment.

**Files:**
- `response_generator.py` - Main `ResponseGenerator` class
- `response_models.py` - Data models (`ScheduledResponse`, `ShouldRespondResult`, etc.)
- `prompts/response_prompts.py` - LLM prompt templates

**Usage:**
```python
from src.green.response_generator import ResponseGenerator
from src.green.llm_config import LLMFactory

# Create LLMs
response_llm = LLMFactory.create("gpt-4o-mini")
summarization_llm = LLMFactory.create("gpt-4o-mini")

# Create generator
generator = ResponseGenerator(
    client=ues_client,
    scenario_config=scenario,
    response_llm=response_llm,
    summarization_llm=summarization_llm,
)

# Process new messages and get scheduled responses
responses = await generator.process_new_messages(
    new_messages=new_messages,
    current_time=current_sim_time,
)
```

### LLM Configuration

The `LLMFactory` creates LangChain chat model instances for multiple providers:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`, etc.
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, etc.
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`, etc.
- **Ollama**: `ollama/llama3.2`, `ollama/gemma3:12b`, etc.

```python
from src.green.llm_config import LLMFactory

# Create different LLM instances
openai_llm = LLMFactory.create("gpt-4o-mini", temperature=0.7)
ollama_llm = LLMFactory.create("ollama/gemma3:12b")
anthropic_llm = LLMFactory.create("claude-3-sonnet-20240229")
```

### Scenario Management

See [scenarios/README.md](scenarios/README.md) for detailed documentation on scenario configuration.

### Message Collection

The `NewMessageCollector` tracks and collects new messages from UES modalities:

```python
from src.green.message_collector import NewMessageCollector

collector = NewMessageCollector(ues_client)
await collector.initialize()  # Record initial state

# After Purple agent acts...
new_messages = await collector.collect()
# Returns NewMessages(emails=[...], sms_messages=[...], calendar_events=[...])
```

### Action Log

The `ActionLogBuilder` creates action log entries from UES event history:

```python
from src.green.action_log import ActionLogBuilder

builder = ActionLogBuilder()
entries = builder.build_from_events(
    events=ues_events,
    turn_number=current_turn,
    agent_id=purple_agent_id,
)
```

## Testing

```bash
# Run all Green agent tests
uv run pytest tests/green/ -v

# Run integration tests with real LLMs
uv run pytest tests/green/test_response_generator_integration.py -m ollama -v
uv run pytest tests/green/test_response_generator_integration.py -m openai -v

# Skip integration tests
uv run pytest tests/green/ -m "not integration" -v
```

## Configuration

Environment variables (set in `.env` file):

```bash
OPENAI_API_KEY=sk-...      # Required for OpenAI models
ANTHROPIC_API_KEY=sk-ant-... # Required for Anthropic models
GOOGLE_API_KEY=...         # Required for Google models
```

Ollama models require a running Ollama server at `localhost:11434`.

## Design Documents

- [Response Generation Design](../../docs/design/RESPONSE_GENERATION_DESIGN.md)
- [Assessment Flow](../../docs/ASSESSMENT_FLOW.md)
- [Implementation Plan](../../docs/IMPLEMENTATION_PLAN.md)
