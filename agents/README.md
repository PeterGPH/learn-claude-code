# Agents

Reference Python implementations for the progressive agent sessions:
- `s01` to `s13`
- `s_full.py` (capstone composition)

## Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `MODEL_ID` in `.env`.

## Provider Setup

### Anthropic (hosted)

```bash
ANTHROPIC_API_KEY=your_real_key
MODEL_ID=claude-sonnet-4-6
# ANTHROPIC_BASE_URL is optional for default Anthropic endpoint
```

### Ollama (local, example: qwen3-coder:30b)

```bash
ollama run qwen3-coder:30b
```

```bash
ANTHROPIC_BASE_URL=http://localhost:11434/v1
ANTHROPIC_API_KEY=ollama
MODEL_ID=qwen3-coder:30b
```

These scripts include fallback parsing for text-form tool calls (for models that do not emit native `tool_use` blocks).

## Run

```bash
python agents/s01_agent_loop.py
python agents/s12_worktree_task_isolation.py
python agents/s13_personal_time_management.py
python agents/s_full.py
```

## CI Sanity Check

```bash
python -m py_compile agents/*.py
```
