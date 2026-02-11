# TODO

Extraneous tasks not directly tied to implementation work.

---

## Bugs

- [x] **Fix `ActionLogBuilder._convert_event_to_entry()` reading wrong key from event data**
  - Code was reading `data.get("action")` instead of `data.get("operation")`
  - Fixed to use `"operation"` key (UES standard across all modalities)
  - Parameter filtering now excludes `"operation"` instead of `"action"`
  - Updated `make_ues_event` test helper in both test files
  - Added `TestOperationKeyHandling` test class (7 new tests per file)

---

## Scenario Updates

- [ ] **Update `scenarios/email_triage_basic/initial_state.json` to UES export format**
  - Current file uses a simplified custom format
  - Must match UES `/scenario/import/full` endpoint requirements
  - Required structure: `metadata`, `environment` (with `time_state` and `modality_states`), `events`
  - Recommendation: Export a properly configured UES scenario using `GET /scenario/export/full` to get the correct format, then modify as needed
  - Reference: UES `docs/client/CLIENT_QUICK_REFERENCE.md`
