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

- [x] **Update `scenarios/email_triage_basic/initial_state.json` to UES export format**
  - Converted from simplified custom format to full UES export format
  - Now matches UES `/scenario/import/full` endpoint requirements
  - Added required sections: `metadata`, `environment` (with `time_state` and `modality_states`), `events`
  - Validated with Pydantic schema successfully
  - Updated documentation in `src/green/scenarios/README.md` to describe the UES export format
