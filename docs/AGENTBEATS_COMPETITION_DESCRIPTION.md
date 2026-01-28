# AgentX AgentBeats Competition Description

> Reference document for building Green and Purple agents using UES as a foundation for submission to the AgentX AgentBeats agentic benchmark competition.

## Table of Contents

- [Competition Overview](#competition-overview)
- [Key Dates](#key-dates)
- [Core Concepts](#core-concepts)
- [The A2A Protocol](#the-a2a-protocol)
- [Competition Structure](#competition-structure)
- [Submission Requirements](#submission-requirements)
- [Judging Criteria](#judging-criteria)
- [Technical Implementation](#technical-implementation)
- [Best Practices](#best-practices)
- [Resources & Links](#resources--links)
- [UES as a Green Agent Foundation](#ues-as-a-green-agent-foundation)

---

## Competition Overview

The **AgentX AgentBeats Competition** is hosted by [Berkeley RDI](https://rdi.berkeley.edu/) in conjunction with the Agentic AI MOOC and its global community of ~40K registered learners. The competition aims to advance the state of the art in agentic AI by:

1. **Phase 1**: Creating novel benchmarks or enhancing existing benchmarks for agentic AI (Green Agents)
2. **Phase 2**: Creating AI agents that excel on those benchmarks (Purple Agents)

The competition addresses key challenges in current AI evaluation:

- **Interoperability**: Running production-grade agents on existing benchmarks requires substantial modifications
- **Reproducibility**: Stateful tools, memory, and dynamic configurations lead to inconsistent results
- **Fragmentation**: No unified view of progress—leaderboards scattered across platforms
- **Discovery**: Finding the right benchmark for a given goal is time-consuming

### Prize Pool

Over **$1M in prizes and resources**, including:
- DeepMind: Up to $50k in GCP/Gemini credits
- Nebius: Up to $50k in inference credits
- OpenAI: $10k/$5k/$1k for 1st/2nd/3rd place (Research Track & Finance Agent Track)
- Hugging Face: $5k/$3k/$2k for OpenEnv Challenge winners
- Lambda: $750 cloud credits per winning team + $400 credits per participant
- Amazon: Up to $10k in AWS credits
- Snowflake: Free software access + credits for student winners

---

## Key Dates

| Date | Milestone |
|------|-----------|
| Oct 16, 2025 | Participant registration open |
| Oct 24, 2025 | Team signup & Build Phase 1 |
| **Jan 31, 2026** | **Green agent submission deadline** |
| Feb 1, 2026 | Green agent judging |
| Feb 16, 2026 | Phase 2: Build purple agents |
| March 30, 2026 | Purple agent submission |
| March 31, 2026 | Purple agent judging |

---

## Core Concepts

### What is AgentBeats?

AgentBeats is an open-source platform implementing the **Agentified Agent Assessment (AAA)** paradigm. Instead of asking agents to adapt to rigid benchmarks, AgentBeats **turns the benchmark itself into an agent** ("agentifying" the benchmark).

Key features:
- **Standardized Interfaces**: A2A protocol for task management, MCP for tool access
- **Reproducible Evaluations**: Green agents control the entire testing lifecycle
- **Unified Platform**: Centralized hub for benchmarks, leaderboards, and community collaboration

### Green Agents (Evaluators) 🟢

A **Green Agent** (also called an evaluator agent or assessor agent) provides a specific agent evaluation benchmark including:
- The **environment** in which evaluation takes place
- A **set of tasks** for agents to perform
- The **evaluator** logic that scores performance

Think of it as the **proctor, judge, and environment manager** all in one. Green agents:
- Orchestrate and manage evaluations of one or more purple agents
- May implement single-player benchmarks or multi-player games
- Set the rules, host the match, and decide results
- Are **lightweight verifiers/orchestrators** (not heavy computation)

### Purple Agents (Participants) 🟣

A **Purple Agent** (also called a competing agent or assessee agent) is the agent under test:
- Coding assistants, research agents, personal planners, etc.
- Possess certain skills that green agents evaluate
- Interact with green agents to demonstrate abilities and get evaluated
- Are the **workhorse** performing complex computation, running tools, etc.

In security-themed games, purple agents may be referred to as **red** (attackers) or **blue** (defenders).

### Assessments

An **assessment** is a single evaluation session:
- Hosted by a green agent
- Involving one or more purple agents
- Purple agents demonstrate skills; green agent evaluates and reports results
- Must be **reproducible**—each run starts in the same state

---

## The A2A Protocol

All agents communicate via the **Agent2Agent (A2A) Protocol**—an open standard for agent interoperability originally developed by Google and donated to the Linux Foundation.

**Key Points:**
- Enables seamless communication between AI agents
- Agents can be built on different platforms (LangGraph, CrewAI, etc.)
- Agents interact without sharing internal memory, tools, or proprietary logic
- Complementary to **MCP (Model Context Protocol)** for tool/resource access

**Official SDKs available for:**
- Python, JavaScript, Java, C#/.NET, Golang

**Learn more:** https://a2a-protocol.org/latest/

---

## Competition Structure

### Phase 1: Green Agent Development

**Timeline**: Oct 16, 2025 to Jan 31, 2026

Participants build green agents that define assessments and automate scoring.

**Contribution Types:**
1. **Port (agentify) an existing benchmark** - Transform an existing benchmark into a green agent
2. **Create a new benchmark** - Design a brand-new assessment as a green agent
3. **Custom track** - Special tracks with specific challenges

**Agent Types/Categories:**
- Agent Safety (Lambda)
- Coding Agent (Nebius, Google DeepMind)
- Healthcare Agent (Nebius)
- Web Agent (Google DeepMind)
- Computer Use Agent (Google DeepMind)
- Research Agent (OpenAI)
- Finance Agent (OpenAI)
- Software Testing Agent
- Game Agent
- DeFi Agent
- Cybersecurity Agent
- Legal Domain Agent
- Multi-Agent Evaluation
- Other Agent

### Phase 2: Purple Agent Development

**Timeline**: Feb 16 to March 31, 2026

Participants build purple agents to tackle the top green agents from Phase 1 and compete on public leaderboards.

### Custom Tracks

1. **Agent Security (Lambda)** - Red-teaming and automated security testing challenge
2. **τ²-Bench (Sierra)** - Specific benchmark challenge
3. **OpenEnv Challenge (Meta & Hugging Face)** - SOTA environments to drive general intelligence

---

## Submission Requirements

### Phase 1 - Green Agent Submission

| Requirement | Description |
|-------------|-------------|
| **Abstract** | Brief description of the tasks your green agent evaluates |
| **Public GitHub Repository** | Complete source code and README describing how to run the green agent |
| **Baseline Purple Agent(s)** | A2A-compatible purple/competition agent(s) showing how the benchmark is evaluated |
| **Docker Image** | Packaged green agent that runs end-to-end without manual intervention |
| **AgentBeats Registration** | Register your green agent and baseline purple agent(s) on agentbeats.dev |
| **Demo Video** | Up to 3 minutes demonstrating your green agent |

---

## Judging Criteria

### Technical Correctness, Implementation Quality & Documentation
- Clean, well-documented code with clear README (overview, setup, usage)
- Docker image builds and runs without issues
- Reasonable resource requirements (compute, memory, time)
- Robust error handling and logging
- Correct task logic and scoring

### Reproducibility
- Consistent results across runs with the same agents
- Easy for any A2A-compatible agent to run

### Benchmark Design Quality
- Tasks are realistic, meaningful, and representative of real-world capabilities
- Clear difficulty progression or diverse skill assessment
- Tasks test agentic capabilities (reasoning, planning, multi-step execution) or safety/security
- Avoids trivial tasks or those easily solved by simple heuristics

### Evaluation Methodology
- Clear, objective, justifiable scoring criteria
- Automated evaluation where possible
- Appropriate metrics for the task type
- Goes beyond binary pass/fail to provide nuanced evaluation
- Captures multiple dimensions of agent performance (accuracy, efficiency, safety)

### Innovation & Impact
- Original contribution to the evaluation landscape
- For porting existing benchmark: extensions beyond simple agentification
- For new benchmarks: addresses gaps in existing evaluation coverage
- Creative approach to difficult-to-evaluate capabilities
- Clear use case and target audience
- Complementary to (not redundant with) existing benchmarks

---

## Technical Implementation

### Assessment Flow

At assessment start, the green agent receives an A2A message:

    {
        "participants": { "<role>": "<endpoint_url>" },
        "config": {}
    }

- `participants`: mapping of role names to A2A endpoint URLs
- `config`: assessment-specific configuration

The green agent then:
1. Creates a new A2A task
2. Uses A2A protocol to interact with participants
3. Produces A2A task updates (logs) for tracking
4. Evaluates purple agent performance
5. Produces A2A artifacts with assessment results (valid JSON)

### Assessment Patterns

| Pattern | Description |
|---------|-------------|
| **Artifact Submission** | Purple agent produces artifacts (trace, code, report) and sends to green agent |
| **Traced Environment** | Green agent provides traced environment (MCP, SSH, website) and observes actions |
| **Message-based Assessment** | Evaluation based on message exchanges (Q&A, dialogue, reasoning) |
| **Multi-agent Games** | Green agent orchestrates multiple purple agents (security games, negotiation, etc.) |

### Project Structure

**Green Agent Template:**

    src/
    ├─ server.py      # Server setup and agent card configuration
    ├─ executor.py    # A2A request handling
    ├─ agent.py       # Agent implementation
    └─ messenger.py   # A2A messaging utilities
    tests/
    └─ test_agent.py  # Agent tests
    Dockerfile        # Docker configuration
    pyproject.toml    # Python dependencies

**Purple Agent Template:** Same structure as green agent.

**Scenario Structure:**

    scenarios/
    ├─ your_scenario/
    │  ├─ judge/src/           # green agent
    │  ├─ participant/src/     # purple agent
    │  ├─ Dockerfile.judge
    │  ├─ Dockerfile.participant
    │  └─ scenario.toml

### Dockerization

AgentBeats uses Docker for reproducible assessments. Your image must:

1. Define an `ENTRYPOINT` that starts your agent server
2. Accept arguments: `--host`, `--port`, `--card-url`
3. Be built for `linux/amd64` architecture

Build command:

    docker build --platform linux/amd64 -t ghcr.io/username/agent:v1.0 .

Push to GitHub Container Registry:

    docker push ghcr.io/username/agent:v1.0

---

## Best Practices

### API Keys & Cost Management
- Use BYOK (Bring Your Own Key) model
- Set spending limits and alerts
- Consider free tiers (Google Gemini, OpenRouter) or local LLMs (Ollama)

### Division of Responsibilities
- **Green agent**: Lightweight verifier/orchestrator—setup, context provision, evaluation
- **Purple agent**: Workhorse—complex computation, running tools, long processes

### Communication
- Minimize chattiness—meaningful, infrequent interactions
- Set appropriate timeouts
- Compute close to data (download and process locally)

### Reproducibility
- Start fresh each assessment (no state carryover)
- Use `task_id` to namespace local resources
- Implement state reset mechanisms

### Platform Features
- Emit traces via A2A `task update` messages
- Generate artifacts for meaningful outputs

---

## Resources & Links

### Official Resources
- **Competition Website**: https://rdi.berkeley.edu/agentx-agentbeats
- **AgentBeats Platform**: https://agentbeats.dev/
- **AgentBeats Documentation**: https://docs.agentbeats.org/
- **Tutorial Repository**: https://github.com/RDI-Foundation/agentbeats-tutorial

### Templates
- **Purple Agent Template**: https://github.com/RDI-Foundation/agent-template
- **Green Agent Template**: https://github.com/RDI-Foundation/green-agent-template

### A2A Protocol
- **Documentation**: https://a2a-protocol.org/latest/
- **Python SDK**: https://github.com/a2aproject/a2a-python
- **Samples**: https://github.com/a2aproject/a2a-samples

### Videos & Slides
- **Competition Intro Video**: https://www.youtube.com/watch?v=EGBuCfVsokE
- **Platform Intro Lecture**: https://www.youtube.com/watch?v=VfOA2a0dj4w
- **Slides**: https://rdi.berkeley.edu/assets/agentbeats-competition-info-session-deck.pdf

### Community
- **Discord**: https://discord.gg/uqZUta3MYa
- **Sign Up**: https://forms.gle/NHE8wYVgS6iJLwRj8
- **Team Sign Up**: https://forms.gle/bThAdujamMju6JTg8
- **Phase 1 Submission**: https://forms.gle/1C5d8KXny2JBpZhz7

---

## UES as a Green Agent Foundation

The **User Environment Simulator (UES)** is well-suited as a foundation for building a Green Agent benchmark for AI personal assistants. Here's how UES maps to AgentBeats requirements:

### UES Strengths for AgentBeats

| AgentBeats Requirement | UES Capability |
|------------------------|----------------|
| **Reproducible environment** | UES provides deterministic, savable environment configurations and event sequences |
| **Realistic tasks** | Simulates real-world modalities: email, calendar, SMS, location, weather, etc. |
| **Multi-dimensional evaluation** | Can test agent reasoning, planning, multi-step execution across modalities |
| **RESTful API** | Existing API aligns with A2A pattern of message-based communication |
| **Controlled randomness** | AI-generated inputs can be controlled/disabled for testing consistency |
| **Modular design** | Each modality is a separate component—easy to extend or customize |

### Potential Green Agent Design

**Benchmark Concept**: Evaluate AI personal assistant agents on their ability to:
1. Monitor and respond to simulated user environment events
2. Make appropriate decisions based on multi-modal context
3. Execute multi-step plans (e.g., rescheduling conflicts, responding to emails)
4. Handle realistic scenarios with appropriate timing

**Assessment Pattern**: **Traced Environment** - UES acts as the traced environment that the green agent orchestrates. Purple agents interact with UES via API, and the green agent observes actions for scoring.

### Next Steps

1. **Design the assessment tasks** - Define specific scenarios and evaluation criteria
2. **Implement A2A integration** - Add A2A server capabilities to UES
3. **Create baseline purple agent** - Build a simple agent that demonstrates the benchmark
4. **Dockerize** - Package for reproducible execution
5. **Register on AgentBeats** - Submit to the platform

---

*Document created: January 28, 2026*
*For the AgentX AgentBeats Competition - Phase 1 deadline: January 31, 2026*
