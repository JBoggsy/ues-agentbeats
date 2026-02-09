# TODO

Extraneous tasks not directly tied to implementation work.

---

## Scenario Updates

- [ ] **Update `scenarios/email_triage_basic/initial_state.json` to UES export format**
  - Current file uses a simplified custom format
  - Must match UES `/scenario/import/full` endpoint requirements
  - Required structure: `metadata`, `environment` (with `time_state` and `modality_states`), `events`
  - Recommendation: Export a properly configured UES scenario using `GET /scenario/export/full` to get the correct format, then modify as needed
  - Reference: UES `docs/client/CLIENT_QUICK_REFERENCE.md`
