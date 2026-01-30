# UES AgentBeats Competition - AI Agent Instructions

## Project Overview

This codebase is a submission to the **AgentX AgentBeats Competition** building on the **User Environment Simulator (UES)** project.

- **Competition Goal**: Build a Green Agent (evaluator) and baseline Purple Agents using UES as the testing environment
- **Green Agent**: Orchestrates assessments, provides the UES environment, and evaluates AI personal assistant agents
- **Purple Agents**: AI assistants that interact with UES to demonstrate their capabilities

**Essential Reading**:
- `docs/AGENTBEATS_COMPETITION_DESCRIPTION.md` - Competition rules, A2A protocol, submission requirements
- UES source: `~/Coding/personal/ues` - The User Environment Simulator codebase

## Development Environment

### Common Commands
```bash
uv sync                              # Install dependencies
uv run python main.py                # Run main entry point
uv run pytest                        # Run all tests
```

**IMPORTANT**: Always use `uv run python ...` or `uv run <command>`. Never use plain `python ...` commands.

## Architecture Overview

### Agent Communication
All agents communicate via the **A2A (Agent2Agent) Protocol**:
- Green agents receive participant endpoints and orchestrate assessments
- Purple agents receive tasks and interact with UES via its REST API
- Assessment results are returned as A2A artifacts (JSON)

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

This structure will evolve as the project develops. Add files and subdirectories as needed.

### Assessment Flow
1. Green agent receives `participants` and `config` via A2A
2. Green agent initializes UES environment with scenario configuration
3. Purple agent receives task context via A2A
4. Purple agent interacts with UES API (email, calendar, SMS, etc.)
5. Green agent observes actions and evaluates performance
6. Results returned as A2A artifacts

## Design Patterns to Follow

### Agent Responsibilities
- **Green Agent** (lightweight orchestrator): Environment setup, task distribution, scoring, result aggregation
- **Purple Agent** (workhorse): Complex reasoning, multi-step planning, UES API interactions

### Reproducibility Requirements
- All assessments must start from the same initial state
- Use `task_id` to namespace local resources
- Avoid wall-clock time dependencies—use UES simulator time
- Environment configurations must be savable/loadable as JSON

### A2A Protocol Compliance
- Implement proper A2A message handling (tasks, updates, artifacts)
- Emit traces via `task update` messages for debugging
- Handle timeouts and error conditions gracefully

### UES Integration
- **Always use the UES Python client library** for all UES interactions—never call the REST API directly
- Import from `ues.client` (see UES docs at `~/Coding/personal/ues/docs/client/`)
- The client provides both sync (`UESClient`) and async (`AsyncUESClient`) interfaces
- Refer to UES `docs/client/CLIENT_QUICK_REFERENCE.md` for available methods and patterns

## Code Style Guidelines

### Documentation
- Use **Google-style docstrings** for all functions, classes, and modules
- Always include type hints on function parameters and return values

### Timezone Handling
- Always use **timezone-aware** datetime objects
- Use UES simulator time, not `datetime.now()` for simulation logic

### Error Handling
- Let errors surface naturally during development
- Catch exceptions only when there's a specific recovery strategy

### Imports
- Keep all imports at the **top of the file**
- Group imports: standard library, third-party, local modules

### Code Clarity
- Prioritize **readability over cleverness**
- Break large functions into smaller helper functions
- Aim for **100 characters per line** maximum

### Commit Messages
- Use clear, descriptive messages (Add, Fix, Update, Refactor, Remove)
- Always add "AI generated commit message" at the end of the body

### Testing & Documentation Requirements
- **All code must be thoroughly tested** before considering implementation complete
- **All code must be documented** with Google-style docstrings
- Write tests alongside implementation, not as an afterthought
- Update relevant documentation when implementing features

## Dockerization

The competition requires Docker images for reproducible assessments:
- Both Green and Purple agents must be Dockerized
- Images must build and run without manual intervention
- Keep Dockerization in mind during development—avoid patterns that complicate containerization
- Test Docker builds regularly, not just at submission time

**Docker requirements:**
- Define an `ENTRYPOINT` that starts the agent server
- Accept arguments: `--host`, `--port`, `--card-url`
- Build for `linux/amd64` architecture

## Coding Instructions
- When calling UES methods or A2A SDK functions, always read and understand the relevant code first
- Avoid hallucinations—refer to existing code, UES docs, or A2A protocol specs when unsure
- Update relevant documentation after implementing features

## User Interaction Instructions
- Always be curious: ask clarifying questions if user intent is unclear
- Be a critical thinker: don't assume the user is always right, consider the user's suggestions/designs/approach carefully and identify weaknesses
- Don't be a yes-man: provide honest feedback and alternative suggestions, don't just agree with everything the user says
- If you see something, say something: point out potential issues, bugs, or improvements you notice in the user's code or design
- Think globally, act locally: consider the broader context of the project when making decisions, but focus on delivering high-quality code for the specific task at hand

## Key External Resources

- **A2A Protocol**: https://a2a-protocol.org/latest/
- **A2A Python SDK**: https://github.com/a2aproject/a2a-python
- **AgentBeats Docs**: https://docs.agentbeats.org/
- **Green Agent Template**: https://github.com/RDI-Foundation/green-agent-template
- **Purple Agent Template**: https://github.com/RDI-Foundation/agent-template
- **UES Source Code**: `~/Coding/personal/ues/`
- **UES Docs**: `~/Coding/personal/ues/docs/`