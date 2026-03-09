#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import re
import subprocess

from anthropic import APIConnectionError, Anthropic, AnthropicError, BadRequestError
from dotenv import load_dotenv

# Keep shell-exported vars authoritative; only fill missing values from .env.
load_dotenv(override=False)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def extract_text_tool_commands(content_blocks) -> list[str]:
    """
    Fallback parser for models that emit tool calls as plain text, e.g.:
    <function=bash><parameter=command>...</parameter></function></tool_call>
    """
    texts = []
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
            texts.append(block.text)
    joined = "\n".join(texts)
    pattern = re.compile(
        r"<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
        re.IGNORECASE | re.DOTALL,
    )
    return [cmd.strip() for cmd in pattern.findall(joined) if cmd.strip()]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def call_messages_api(messages: list):
    try:
        return client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
    except BadRequestError as e:
        msg = str(e)
        if "credit balance is too low" in msg.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            key_hint = (
                f"{api_key[:12]}...{api_key[-6:]}" if len(api_key) >= 20 else "(unset/short)"
            )
            base_url = os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            print("\033[31mError: Anthropic API credits are exhausted.\033[0m")
            print("Top up billing credits or switch to a different provider/base URL.")
            print(f"Using key: {key_hint}")
            print(f"Using base URL: {base_url}")
            print("If you exported a new key, note that .env may still point to an older account.")
            return None
        print(f"\033[31mBad request to Anthropic API:\033[0m {msg}")
        return None
    except APIConnectionError as e:
        print(f"\033[31mNetwork/API connection error:\033[0m {e}")
        return None
    except AnthropicError as e:
        print(f"\033[31mAnthropic API error:\033[0m {e}")
        return None


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = call_messages_api(messages)
        if response is None:
            return
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # Native Anthropic/Ollama tool-calling path
        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\033[33m$ {block.input['command']}\033[0m")
                    output = run_bash(block.input["command"])
                    print(output[:200])
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "user", "content": results})
            continue

        # Fallback for models that emit textual pseudo tool calls
        text_commands = extract_text_tool_commands(response.content)
        if not text_commands:
            return

        feedback_lines = []
        for command in text_commands:
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            feedback_lines.append(f"$ {command}\n{output}")

        messages.append({
            "role": "user",
            "content": (
                "Tool execution results:\n\n"
                + "\n\n".join(feedback_lines)
                + "\n\nIf finished, give a concise final answer. "
                "If more commands are needed, emit another bash tool call."
            ),
        })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
