#!/usr/bin/env python3
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
"""

import os
import ast
import json
import re
import subprocess
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
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
        return f"Wrote {len(content)} bytes"
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
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


def _coerce_param(name: str, raw: str):
    value = raw.strip()
    if name == "limit" and value.isdigit():
        return int(value)
    if name == "items":
        # Accept either JSON or Python-literal list/dict text.
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        raise ValueError("todo.items must be a JSON/Python list")
    return value


def extract_text_tool_calls(content_blocks):
    """
    Fallback parser for models that emit pseudo tool calls as plain text, e.g.:
    <function=read_file><parameter=path>requirements.txt</parameter></function></tool_call>
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
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
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


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # Nag reminder is injected below, alongside tool results.
        response = call_messages_api(messages)
        if response is None:
            return
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "tool_use":
            results = []
            used_todo = False
            for block in response.content:
                if block.type == "tool_use":
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                    print(f"> {block.name}: {str(output)[:200]}")
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                    if block.name == "todo":
                        used_todo = True
            rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
            if rounds_since_todo >= 3:
                results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})
            messages.append({"role": "user", "content": results})
            continue

        # Fallback for models that return text-style function calls.
        text_calls = extract_text_tool_calls(response.content)
        if not text_calls:
            return

        feedback_lines = []
        used_todo = False
        for name, args in text_calls:
            handler = TOOL_HANDLERS.get(name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {name}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {name}: {str(output)[:200]}")
            args_preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
            feedback_lines.append(f"{name}({args_preview}) -> {output}")
            if name == "todo":
                used_todo = True

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        reminder = "\n\n<reminder>Update your todos.</reminder>" if rounds_since_todo >= 3 else ""
        messages.append({
            "role": "user",
            "content": (
                "Tool execution results:\n\n"
                + "\n\n".join(feedback_lines)
                + reminder
                + "\n\nIf finished, provide a concise final answer. "
                "If more actions are needed, emit another function call."
            ),
        })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
