# TODO

Extraneous tasks not directly tied to implementation work.

---

## Bugs

(none currently)

---

## Recently Completed

- ✅ Email Triage Basic scenario fully implemented (scenario.json, initial_state.json, ground_truth.py, _eval_helpers.py, evaluators.py)
- ✅ All 8 programmatic evaluators tested (38 tests) and all helpers tested (35 tests)
- ✅ Infrastructure updates: LLM field on AgentBeatsEvalContext, CriteriaJudge LLM injection, ScenarioLoader sibling imports
- ✅ Full test suite: 1,542+ tests passing
- ✅ Green Agent Executor (`src/green/executor.py`) — 41 tests, all steps complete

---

## Remaining Work

### High Priority
- [ ] Green Agent Server (`src/green/server.py`) — A2A server setup, agent card, request routing using `GreenAgentExecutor`
- [ ] End-to-end integration test: load email_triage_basic → run against a mock Purple agent → verify scoring
- [ ] UES import validation: import `initial_state.json` into a live UES instance and verify all 42 events fire correctly

### Medium Priority
- [ ] Purple Agent Template (Phase 4) — base agent class, assessment handler, executor, example agent
- [ ] Dockerize Green and Purple agents (Phase 5)

### Low Priority
- [ ] Additional scenarios beyond email_triage_basic
- [ ] Performance benchmarking tools
- [ ] Demo video for submission