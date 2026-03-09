#!/usr/bin/env python3
"""
s13_personal_time_management.py - Personal Time Management Agent

Persistent daily workflow:
1) sync todos
2) log progress throughout the day
3) generate a daily summary report

Data layout:
  .ptm/
    todos.json
    progress/YYYY-MM-DD.jsonl
    summaries/YYYY-MM-DD.md
"""

import ast
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from anthropic import APIConnectionError, Anthropic, AnthropicError, BadRequestError
from dotenv import load_dotenv

# Keep shell-exported vars authoritative; only fill missing values from .env.
load_dotenv(override=False)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"""You are a personal time-management coding agent at {WORKDIR}.
Use PTM tools to:
1) ingest/sync todos,
2) log progress updates during the day,
3) produce daily summaries.

Prefer PTM tools over prose when state should persist.
"""


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class PersonalTimeManager:
    def __init__(self, root: Path):
        self.root = root / ".ptm"
        self.progress_dir = self.root / "progress"
        self.summary_dir = self.root / "summaries"
        self.todos_path = self.root / "todos.json"

        self.root.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        if not self.todos_path.exists():
            self.todos_path.write_text("[]")

    def _load_todos(self) -> list:
        try:
            data = json.loads(self.todos_path.read_text() or "[]")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_todos(self, todos: list):
        self.todos_path.write_text(json.dumps(todos, indent=2))

    def _next_id(self, todos: list) -> int:
        ids = [int(t.get("id", 0)) for t in todos if str(t.get("id", "")).isdigit()]
        return max(ids, default=0) + 1

    def _normalize_status(self, status: str) -> str:
        s = (status or "pending").strip().lower()
        valid = {"pending", "in_progress", "completed", "blocked", "cancelled"}
        return s if s in valid else "pending"

    def _parse_todo_source(self, raw: str) -> list[dict]:
        text = (raw or "").strip()
        if not text:
            return []

        # JSON first
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    parsed = parsed.get("todos") or parsed.get("items") or []
                if isinstance(parsed, list):
                    return [item if isinstance(item, dict) else {"title": str(item)} for item in parsed]
            except Exception:
                pass

        # Markdown checklist: - [ ] task / - [x] task
        md = []
        for line in text.splitlines():
            m = re.match(r"^\s*[-*]\s*\[( |x|X)\]\s+(.+?)\s*$", line)
            if m:
                md.append({
                    "title": m.group(2).strip(),
                    "status": "completed" if m.group(1).lower() == "x" else "pending",
                })
        if md:
            return md

        # Plain lines fallback
        lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
        return [{"title": ln} for ln in lines]

    def sync_todos(self, source_path: str = "", content: str = "", mode: str = "merge") -> str:
        source_text = content or ""
        if source_path:
            source_text = safe_path(source_path).read_text()

        incoming = self._parse_todo_source(source_text)
        if not incoming:
            return "No todos parsed from input."

        existing = self._load_todos()
        now = time.time()
        next_id = self._next_id(existing)

        by_title = {str(t.get("title", "")).strip().lower(): t for t in existing}
        out = [] if mode == "replace" else list(existing)
        created, updated = 0, 0

        for item in incoming:
            title = str(item.get("title") or item.get("subject") or item.get("content") or "").strip()
            if not title:
                continue
            key = title.lower()
            status = self._normalize_status(str(item.get("status", "pending")))
            estimate = int(item.get("estimate_minutes", 0) or 0)
            due = str(item.get("due", "") or "")
            priority = str(item.get("priority", "") or "")
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                tags = [str(tags)]

            if mode != "replace" and key in by_title:
                t = by_title[key]
                t["status"] = status if t.get("status") != "completed" else "completed"
                if estimate:
                    t["estimate_minutes"] = estimate
                if due:
                    t["due"] = due
                if priority:
                    t["priority"] = priority
                if tags:
                    t["tags"] = tags
                t["updated_at"] = now
                updated += 1
            else:
                todo = {
                    "id": next_id,
                    "title": title,
                    "status": status,
                    "estimate_minutes": estimate,
                    "due": due,
                    "priority": priority,
                    "tags": tags,
                    "notes": str(item.get("notes", "") or ""),
                    "created_at": now,
                    "updated_at": now,
                }
                next_id += 1
                out.append(todo)
                by_title[key] = todo
                created += 1

        self._save_todos(out)
        return f"Todos synced. created={created}, updated={updated}, total={len(out)}"

    def list_todos(self, status: str = "") -> str:
        todos = self._load_todos()
        want = self._normalize_status(status) if status else ""
        if want:
            todos = [t for t in todos if t.get("status") == want]
        if not todos:
            return "No todos."
        lines = []
        marks = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "blocked": "[!]",
            "cancelled": "[-]",
        }
        for t in todos:
            m = marks.get(t.get("status", "pending"), "[?]")
            lines.append(f"{m} #{t['id']}: {t['title']}")
        return "\n".join(lines)

    def _progress_path(self, day: str) -> Path:
        return self.progress_dir / f"{day}.jsonl"

    def _append_progress(self, day: str, event: dict):
        path = self._progress_path(day)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _read_progress(self, day: str) -> list:
        path = self._progress_path(day)
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    def log_progress(self, task_id: int, minutes: int = 0, note: str = "", status: str = "") -> str:
        todos = self._load_todos()
        target = None
        for t in todos:
            if int(t.get("id", -1)) == int(task_id):
                target = t
                break
        if not target:
            return f"Error: Task #{task_id} not found"

        before = target.get("status", "pending")
        after = before
        if status:
            after = self._normalize_status(status)
            target["status"] = after
        elif before == "pending" and minutes > 0:
            after = "in_progress"
            target["status"] = after

        target["updated_at"] = time.time()
        self._save_todos(todos)

        event = {
            "ts": time.time(),
            "task_id": int(task_id),
            "title": target.get("title", ""),
            "minutes": max(0, int(minutes or 0)),
            "note": str(note or ""),
            "status_before": before,
            "status_after": after,
        }
        self._append_progress(today_key(), event)
        return f"Progress logged for #{task_id}: +{event['minutes']}m, {before} -> {after}"

    def today_status(self, day: str = "") -> str:
        date_key = day or today_key()
        todos = self._load_todos()
        events = self._read_progress(date_key)

        minutes = sum(int(e.get("minutes", 0) or 0) for e in events)
        touched = sorted({int(e.get("task_id")) for e in events if str(e.get("task_id", "")).isdigit()})
        completed_today = sorted({
            int(e.get("task_id")) for e in events
            if str(e.get("task_id", "")).isdigit() and e.get("status_after") == "completed"
        })

        counts = {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0, "cancelled": 0}
        for t in todos:
            s = t.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1

        return (
            f"Date: {date_key}\n"
            f"Events: {len(events)}\n"
            f"Focus minutes: {minutes}\n"
            f"Touched tasks: {touched or '[]'}\n"
            f"Completed today: {completed_today or '[]'}\n"
            f"Board: pending={counts['pending']}, in_progress={counts['in_progress']}, "
            f"completed={counts['completed']}, blocked={counts['blocked']}, cancelled={counts['cancelled']}"
        )

    def daily_summary(self, day: str = "") -> str:
        date_key = day or today_key()
        todos = self._load_todos()
        events = self._read_progress(date_key)
        by_id = {int(t["id"]): t for t in todos if str(t.get("id", "")).isdigit()}

        total_minutes = sum(int(e.get("minutes", 0) or 0) for e in events)
        touched_ids = sorted({int(e.get("task_id")) for e in events if str(e.get("task_id", "")).isdigit()})
        completed_ids = sorted({
            int(e.get("task_id")) for e in events
            if str(e.get("task_id", "")).isdigit() and e.get("status_after") == "completed"
        })

        in_progress = [t for t in todos if t.get("status") == "in_progress"]
        blocked = [t for t in todos if t.get("status") == "blocked"]
        carry_over = [t for t in todos if t.get("status") in ("pending", "in_progress", "blocked")]

        notes = [str(e.get("note", "")).strip() for e in events if str(e.get("note", "")).strip()]
        top_notes = notes[-10:]

        lines = [
            f"# Daily Summary - {date_key}",
            "",
            "## Metrics",
            f"- Progress events: {len(events)}",
            f"- Focus minutes: {total_minutes}",
            f"- Tasks touched: {len(touched_ids)} ({', '.join(f'#{i}' for i in touched_ids) if touched_ids else 'none'})",
            f"- Completed today: {len(completed_ids)} ({', '.join(f'#{i}' for i in completed_ids) if completed_ids else 'none'})",
            "",
            "## Completed Today",
        ]

        if completed_ids:
            for tid in completed_ids:
                t = by_id.get(tid, {})
                lines.append(f"- #{tid} {t.get('title', '(unknown)')}")
        else:
            lines.append("- None")

        lines += ["", "## In Progress"]
        if in_progress:
            for t in in_progress:
                lines.append(f"- #{t['id']} {t['title']}")
        else:
            lines.append("- None")

        lines += ["", "## Blocked"]
        if blocked:
            for t in blocked:
                lines.append(f"- #{t['id']} {t['title']}")
        else:
            lines.append("- None")

        lines += ["", "## Notes"]
        if top_notes:
            for n in top_notes:
                lines.append(f"- {n}")
        else:
            lines.append("- None")

        lines += ["", "## Carry Over (Tomorrow)"]
        if carry_over:
            for t in carry_over:
                lines.append(f"- #{t['id']} [{t['status']}] {t['title']}")
        else:
            lines.append("- None")

        report = "\n".join(lines) + "\n"
        out_path = self.summary_dir / f"{date_key}.md"
        out_path.write_text(report)
        return f"Daily summary written: {out_path}"


PTM = PersonalTimeManager(WORKDIR)


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "sync_todos": lambda **kw: PTM.sync_todos(kw.get("source_path", ""), kw.get("content", ""), kw.get("mode", "merge")),
    "list_todos": lambda **kw: PTM.list_todos(kw.get("status", "")),
    "log_progress": lambda **kw: PTM.log_progress(kw["task_id"], kw.get("minutes", 0), kw.get("note", ""), kw.get("status", "")),
    "today_status": lambda **kw: PTM.today_status(kw.get("date", "")),
    "daily_summary": lambda **kw: PTM.daily_summary(kw.get("date", "")),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "sync_todos",
        "description": "Sync todo list from file or inline content. Supports JSON, markdown checklist, or plain lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["merge", "replace"]},
            },
        },
    },
    {
        "name": "list_todos",
        "description": "List todos, optionally filtered by status.",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        },
    },
    {
        "name": "log_progress",
        "description": "Log progress event for a task and optionally set status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "minutes": {"type": "integer"},
                "note": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"]},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "today_status",
        "description": "Show today's progress metrics and current board status.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
        },
    },
    {
        "name": "daily_summary",
        "description": "Generate and write daily markdown summary report.",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
        },
    },
]


def _coerce_param(name: str, raw: str):
    value = raw.strip()
    if name in ("task_id", "minutes", "limit") and value.isdigit():
        return int(value)
    return value


def extract_text_tool_calls(content_blocks):
    """
    Fallback parser for models that emit pseudo tool calls as plain text.
    """
    texts = []
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
            texts.append(block.text)
    joined = "\n".join(texts)
    calls = []
    call_pattern = re.compile(
        r"<function=([a-zA-Z0-9_]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        re.IGNORECASE | re.DOTALL,
    )
    param_pattern = re.compile(
        r"<parameter=([a-zA-Z0-9_]+)>\s*(.*?)\s*</parameter>",
        re.IGNORECASE | re.DOTALL,
    )
    for name, body in call_pattern.findall(joined):
        args = {}
        for key, value in param_pattern.findall(body):
            args[key] = _coerce_param(key, value)
        calls.append((name, args))
    return calls


def call_messages_api(messages: list):
    try:
        return client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
    except BadRequestError as e:
        msg = str(e)
        print(f"\033[31mBad request to Anthropic-compatible API:\033[0m {msg}")
        if "model" in msg.lower() and "not found" in msg.lower():
            base_url = os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            print(f"Configured MODEL_ID={MODEL}")
            print(f"Configured ANTHROPIC_BASE_URL={base_url}")
            print("For Ollama, set MODEL_ID to an installed local tag, e.g. qwen3-coder:30b")
        return None
    except APIConnectionError as e:
        print(f"\033[31mNetwork/API connection error:\033[0m {e}")
        return None
    except AnthropicError as e:
        print(f"\033[31mAnthropic API error:\033[0m {e}")
        return None


def agent_loop(messages: list):
    while True:
        response = call_messages_api(messages)
        if response is None:
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                    print(f"> {block.name}: {str(output)[:200]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            messages.append({"role": "user", "content": results})
            continue

        text_calls = extract_text_tool_calls(response.content)
        if not text_calls:
            return

        feedback_lines = []
        for tool_name, args in text_calls:
            handler = TOOL_HANDLERS.get(tool_name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_name}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {tool_name}: {str(output)[:200]}")
            args_preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
            feedback_lines.append(f"{tool_name}({args_preview}) -> {output}")

        messages.append({
            "role": "user",
            "content": (
                "Tool execution results:\n\n"
                + "\n\n".join(feedback_lines)
                + "\n\nIf finished, provide a concise final answer. "
                "If more actions are needed, emit another function call."
            ),
        })


if __name__ == "__main__":
    print(f"PTM data dir: {WORKDIR / '.ptm'}")
    history = []
    while True:
        try:
            query = input("\033[36ms13 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/today":
            print(PTM.today_status())
            print()
            continue
        if query.strip() == "/summary":
            print(PTM.daily_summary())
            print()
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
