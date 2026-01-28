# UES AgentBeats

A submission to the **AgentX AgentBeats Competition** building on the [User Environment Simulator (UES)](https://github.com/your-org/ues) project.

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
└── purple/           # Purple agent template
scenarios/            # Assessment scenarios
tests/                # Test suite
docs/                 # Documentation
```

## Documentation

- [Competition Description](docs/AGENTBEATS_COMPETITION_DESCRIPTION.md)
- [Assessment Flow](docs/ASSESSMENT_FLOW.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)

---

## TODO

### Phase 2: Common AgentBeats Helper Code ✅ COMPLETE
- [x] Implement `src/common/agentbeats/messages.py` - Assessment message types
- [x] Implement `src/common/agentbeats/results.py` - Assessment results models
- [x] Implement `src/common/agentbeats/updates.py` - Task update helpers
- [x] Implement `src/common/agentbeats/config.py` - Configuration models
- [x] Write tests for serialization/validation (237 tests passing)

### Phase 3: Green Agent Implementation
- [ ] Implement scenario schema and loader (`src/green/scenarios/`)
- [ ] Implement assessment orchestrator (`src/green/assessment/orchestrator.py`)
- [ ] Implement turn handler (`src/green/assessment/turn_handler.py`)
- [ ] Implement response generator (`src/green/response_generator/`)
- [ ] Implement criteria judge (`src/green/evaluation/`)
- [ ] Integrate LangChain for LLM-based response generation
- [ ] Integrate LangChain for LLM-based criteria judging
- [ ] Create first complete scenario

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
