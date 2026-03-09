#!/usr/bin/env python3
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import re
import subprocess
import time
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = call_messages_api(
        system="You summarize coding-agent conversations for continuity.",
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        tools=None,
        max_tokens=2000,
    )
    summary = (
        "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        if response is not None
        else "Summary unavailable (API error). Transcript was saved for recovery."
    )
    # Replace all messages with compressed summary
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]


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
    "compact":    lambda **kw: "Manual compression requested.",
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
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


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
            raw = value.strip()
            args[key] = int(raw) if key == "limit" and raw.isdigit() else raw
        calls.append((name, args))
    return calls


def call_messages_api(system: str, messages: list, tools=None, max_tokens: int = 8000):
    kwargs = {
        "model": MODEL,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return client.messages.create(**kwargs)
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


def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    if name == "compact":
        return "Compressing...", True
    handler = TOOL_HANDLERS.get(name)
    try:
        output = handler(**args) if handler else f"Unknown tool: {name}"
    except Exception as e:
        output = f"Error: {e}"
    return str(output), False


def agent_loop(messages: list):
    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        response = call_messages_api(SYSTEM, messages, TOOLS, 8000)
        if response is None:
            return
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "tool_use":
            results = []
            manual_compact = False
            for block in response.content:
                if block.type == "tool_use":
                    output, should_compact = execute_tool(block.name, block.input)
                    manual_compact = manual_compact or should_compact
                    print(f"> {block.name}: {output[:200]}")
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})
            # Layer 3: manual compact triggered by the compact tool
            if manual_compact:
                print("[manual compact]")
                messages[:] = auto_compact(messages)
            continue

        # Fallback for models that return text-style function calls.
        text_calls = extract_text_tool_calls(response.content)
        if not text_calls:
            return

        feedback_lines = []
        manual_compact = False
        for name, args in text_calls:
            output, should_compact = execute_tool(name, args)
            manual_compact = manual_compact or should_compact
            print(f"> {name}: {output[:200]}")
            args_preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
            feedback_lines.append(f"{name}({args_preview}) -> {output}")

        messages.append({
            "role": "user",
            "content": (
                "Tool execution results:\n\n"
                + "\n\n".join(feedback_lines)
                + "\n\nIf finished, provide a concise final answer. "
                "If more actions are needed, emit another function call."
            ),
        })
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
