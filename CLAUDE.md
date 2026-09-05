# SIH_WINNER — working agreement

Build target and full spec: `satquery_ai_full_build_report_1.md`.

## Session memory lives in Obsidian

The vault folder `C:\Users\adithya\Documents\Obsidian Vault\SIH\` is this
project's memory, because context is lost to compaction and usage limits.

- `Project Brain.md` — the curated state. Auto-injected into context at the
  start of every session by the `SessionStart` hook.
- `Logs/YYYY-MM-DD.md` — raw turn-by-turn record, appended automatically by the
  `Stop` hook. Read-only; never hand-edit.

**Before finishing any task, update `Project Brain.md`:** move the task's row in
the Task ledger, add a one-line entry to the Decisions log saying what was
chosen and why, and record any new constraint or blocker. Anything not written
there does not survive to the next session.

## Standing checks

Every task is done only when the checks under "Standing checks" in
`Project Brain.md` pass. Add project-specific checks to that list as they come
up, rather than to this file.
