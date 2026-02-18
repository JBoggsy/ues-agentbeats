# UES AgentBeats Competition - AI Agent Instructions

## Project Overview

This codebase is a submission to the **AgentX AgentBeats Competition** building on the **User Environment Simulator (UES)** project.

- **Green Agent** (evaluator): Orchestrates assessments, provides the UES environment, evaluates AI personal assistant agents
- **Purple Agents** (participants): AI assistants that interact with UES to demonstrate their capabilities

### Assessment Flow
1. Green agent receives `participants` and `config` via A2A
2. Green agent initializes UES environment with scenario configuration
3. Purple agent receives task context via A2A
4. Purple agent interacts with UES API (email, calendar, SMS, etc.)
5. Green agent observes actions and evaluates performance
6. Results returned as A2A artifacts

### Project Structure
```
src/
├─ green/                    # Green agent (evaluator)
├─ purple/                   # Purple agents (participants)
└─ common/                   # Shared utilities
scenarios/                   # Assessment scenarios
tests/
Dockerfile.green
Dockerfile.purple
```

Add files and subdirectories as needed.

## Development Environment

**IMPORTANT**: Always use `uv run python ...` or `uv run <command>`. Never use plain `python ...` commands.

```bash
uv sync                              # Install dependencies
uv run python main.py                # Run main entry point
uv run pytest                        # Run all tests
```

### Docker

Both agents must be Dockerized. Images must build and run without manual intervention.

- Define an `ENTRYPOINT` that starts the agent server
- Accept arguments: `--host`, `--port`, `--card-url`
- Build for `linux/amd64` architecture
- Test Docker builds regularly, not just at submission time
- Avoid patterns that complicate containerization

## Design Constraints

### Reproducibility
- All assessments must start from the same initial state
- Use `task_id` to namespace local resources
- Avoid wall-clock time dependencies—use UES simulator time
- Environment configurations must be savable/loadable as JSON

### A2A Protocol
All agents communicate via the **A2A (Agent2Agent) Protocol**:
- Implement proper message handling (tasks, updates, artifacts)
- Emit traces via `task update` messages for debugging
- Handle timeouts and error conditions gracefully

### UES Integration
- **Always use the UES Python client library**—never call the REST API directly
- Import from `ues.client` (`UESClient` for sync, `AsyncUESClient` for async)
- Refer to `~/Coding/personal/ues/docs/client/CLIENT_QUICK_REFERENCE.md`

## Code Style

### Formatting & Structure
- **100 characters per line** maximum
- Keep imports at the **top of the file**, grouped: stdlib, third-party, local
- Prioritize **readability over cleverness**
- Break large functions into smaller helpers

### Type Hints & Docstrings
- **Google-style docstrings** for all functions, classes, and modules
- Type hints on all function parameters and return values

### Datetime
- Always use **timezone-aware** datetime objects
- Use UES simulator time, not `datetime.now()`, for simulation logic

### Error Handling
- Let errors surface naturally during development
- Catch exceptions only when there's a specific recovery strategy

### Testing
- All code must be thoroughly tested before considering it complete
- Write tests alongside implementation, not as an afterthought

### Commit Messages
- Use clear, descriptive verbs (Add, Fix, Update, Refactor, Remove)
- Always add "AI generated commit message" at the end of the body

## AI Behavioral Instructions

### Communication Style
- Ask clarifying questions when intent is unclear
- Provide honest feedback—don't just agree with everything
- Point out potential issues, bugs, or improvements proactively
- Consider the broader project context, but deliver focused code

### Coding Practices
- Read and understand relevant code before calling UES or A2A SDK functions
- Refer to existing code, UES docs, or A2A specs—never hallucinate APIs

### Documentation Maintenance
- Update relevant docs and plans after implementing features
- When completing a task in a plan/todo list, compress that section to include only essential completion details

## References

| Resource | Location |
|---|---|
| Competition Rules | `docs/AGENTBEATS_COMPETITION_DESCRIPTION.md` |
| UES Source Code | `~/Coding/personal/ues/` |
| UES Docs | `~/Coding/personal/ues/docs/` |
| A2A Protocol | https://a2a-protocol.org/latest/ |
| A2A Python SDK | https://github.com/a2aproject/a2a-python |
| AgentBeats Docs | https://docs.agentbeats.org/ |
| Green Agent Template | https://github.com/RDI-Foundation/green-agent-template |
| Purple Agent Template | https://github.com/RDI-Foundation/agent-template |