# s13: Personal Time Management

`s01 > s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12 > [ s13 ]`

> *"Turn tasks into daily execution and reflection"* -- sync todos, log progress, and generate end-of-day summaries from one persistent state store.

## Why this step exists

By `s12`, the agent can plan, delegate, and isolate execution. But personal productivity needs a different loop:
- carry todo state across days,
- capture progress in small increments during the day,
- generate an end-of-day report without rebuilding context.

`s13` adds a lightweight personal workflow layer on top of the same tool loop.

## What `s13` adds

- Persistent state under `.ptm/`
  - `todos.json`
  - `progress/YYYY-MM-DD.jsonl`
  - `summaries/YYYY-MM-DD.md`
- Tools for:
  - syncing todo lists from text/JSON/markdown checklist input,
  - appending progress logs,
  - generating and reading daily summaries.
- Same agent-loop pattern with tool calls and tool results.

## Run

```bash
python agents/s13_personal_time_management.py
```

## Typical flow

1. Sync todos from your current task list.
2. Log progress updates as you complete work.
3. Ask for the daily summary at the end of the day.

This keeps planning, execution, and reflection in one persistent local system.
