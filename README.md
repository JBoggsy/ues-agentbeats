# UES AgentBeats

A submission to the **AgentX AgentBeats Competition** building on the [User Environment Simulator (UES)](https://github.com/JBoggsy/ues) project.

## Overview

This project implements:

- **Green Agent** (Evaluator): Orchestrates assessments of AI personal assistants using UES as the testing environment. Manages scenarios, generates character responses, and evaluates agent performance.
- **Purple Agent Template**: A lightweight template to help participants build A2A-compliant personal assistant agents.

### How It Works

1. Green agent initializes a UES environment with a predefined scenario
2. Purple agent receives task context via the [A2A protocol](https://a2a-protocol.org)
3. Purple agent interacts with UES (email, calendar, SMS, chat) to complete tasks
4. Green agent generates character responses and advances simulation time
5. Assessment results are scored across multiple dimensions (accuracy, efficiency, safety, etc.)

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Start Green agent (example)
uv run python -m src.green --host 0.0.0.0 --port 8000

# Start Purple agent (example)
uv run python -m src.purple --host 0.0.0.0 --port 8001
```

## Project Structure

```
src/
├── common/
│   ├── a2a/          # A2A protocol helpers (complete)
│   └── agentbeats/   # AgentBeats-specific utilities (complete)
├── green/            # Green agent implementation
│   ├── scenarios/    # Scenario schema and loader (complete)
│   ├── prompts/      # LLM prompt templates (complete)
│   ├── action_log.py # Action log builder (complete)
│   ├── llm_config.py # LLM factory (complete)
│   ├── message_collector.py  # New message collector (complete)
│   ├── response_models.py    # Response data models (complete)
│   └── response_generator.py # Character response generation (complete)
└── purple/           # Purple agent template
scenarios/            # Assessment scenarios
tests/                # Test suite (1,424 tests)
docs/                 # Documentation
```

## Documentation

**Design Documents:**
- [Competition Description](docs/AGENTBEATS_COMPETITION_DESCRIPTION.md)
- [Assessment Flow](docs/ASSESSMENT_FLOW.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Response Generation Design](docs/design/RESPONSE_GENERATION_DESIGN.md)

**Module READMEs:**
- [Green Agent Module](src/green/README.md)
- [Scenario Management](src/green/scenarios/README.md)
- [Response Prompts](src/green/prompts/README.md)
- [A2A Helpers](src/common/a2a/README.md)
- [AgentBeats Helpers](src/common/agentbeats/README.md)

## Dependencies

- **UES**: Installed as a local editable dependency from `../ues/`. Ensure the UES repository is cloned adjacent to this project.
- **pytest-dotenv**: Automatically loads `.env` during test runs.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for OpenAI-based LLMs
OPENAI_API_KEY=sk-your-key-here

# Optional for other providers
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# GOOGLE_API_KEY=your-key-here
```

---

## TODO

### Phase 2: Common AgentBeats Helper Code ✅ COMPLETE
- [x] Implement `src/common/agentbeats/messages.py` - Assessment message types
- [x] Implement `src/common/agentbeats/results.py` - Assessment results models
- [x] Implement `src/common/agentbeats/updates.py` - Task update helpers
- [x] Implement `src/common/agentbeats/config.py` - Configuration models
- [x] Write tests for serialization/validation (237 tests passing)

### Phase 3: Green Agent Implementation ✅ COMPLETE
- [x] Implement scenario schema and loader (`src/green/scenarios/`)
- [x] Implement LLM configuration factory (`src/green/llm_config.py`)
- [x] Implement action log builder (`src/green/action_log.py`)
- [x] Implement new message collector (`src/green/message_collector.py`)
- [x] Implement response generator (`src/green/response_generator.py`)
- [x] Implement response data models (`src/green/response_models.py`)
- [x] Implement LLM prompt templates (`src/green/prompts/`)
- [x] Integrate LangChain for LLM-based response generation
- [x] Write integration tests with Ollama and OpenAI
- [x] Implement assessment orchestrator (`src/green/agent.py`)
- [x] Implement criteria judge (`src/green/evaluation/`)
- [x] Implement GreenAgent with full turn loop, UES management, Purple communication
- [x] Full test suite: 1,424 tests passing (0 failures, 0 skipped)

### Phase 4: Purple Agent Template
- [ ] Implement base agent class (`src/purple/base_agent.py`)
- [ ] Implement assessment handler (`src/purple/handler.py`)
- [ ] Implement Purple executor (`src/purple/executor.py`)
- [ ] Create UES client wrapper (`src/purple/ues_wrapper.py`)
- [ ] Create simple example agent

### Phase 5: Submission Requirements
- [ ] Dockerize Green agent (`Dockerfile.green`)
- [ ] Dockerize Purple agent (`Dockerfile.purple`)
- [ ] Write scenario documentation (`docs/SCENARIOS.md`)
- [ ] Write evaluation criteria docs (`docs/EVALUATION_CRITERIA.md`)
- [ ] Write Purple agent guide (`docs/PURPLE_AGENT_GUIDE.md`)
- [ ] Create demo video
- [ ] Register on AgentBeats

---

## License

MIT
