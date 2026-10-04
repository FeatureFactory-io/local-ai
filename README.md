# Local AI

A dispatcher that lets Cursor, Claude Desktop, or ChatGPT hand one task at a time to a local Qwen model served by Ollama. The frontier model never gets a shell. It calls four MCP tools. The dispatcher pins the task, runs one worker, and writes progress on the pin.

The contract is [docs/SAO.md](docs/SAO.md). This file is how to start that server and attach it.

## Before the first call

1. Start Ollama in another terminal: `ollama serve`
2. Pull the tags you will map below. `ollama list` shows what is already on the machine.
3. From this repo, with the project virtualenv:

```bash
uv sync
```

Run one dispatcher process. A second process on the same `data/local-ai.sqlite` is a second writer. Use either the stdio server (Cursor or Claude) or the HTTP server (ChatGPT), not both at once.

## Tools the frontier model calls

Call them in this order.

1. `capacity` — no arguments. Returns free memory, whether Ollama is up, and `recommended` (`qwen-8b`, `qwen-14b`, `qwen-24b`, or `none`).
2. `submit` — pin a task. The response `status` is `queued` even when the worker starts immediately. A following `read` shows `running`.
3. `read` — pass `task_id` from `submit`. Repeat it to see `phase`, `steps`, and `reasoning`.
4. `reprioritize` — change `priority` of a pin that is still `queued`.

`submit` fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `goal` | yes | What the worker must achieve |
| `done_when` | yes | How you will know it finished |
| `priority` | no | Integer 1 (first) through 100 (last). Default 50 |
| `model` | no | `qwen-8b`, `qwen-14b`, or `qwen-24b`. Default is `recommended` from `capacity` |
| `inputs` | no | File paths copied into the job folder when `access.mode` is `job` |
| `access` | no | `{ "mode": "job" }` (default), `{ "mode": "paths", "paths": ["/absolute/dir"] }`, or `{ "mode": "home" }` |

A rejected call is still a normal MCP result. Check `status`. `rejected` with `reason: ollama_down` means Ollama was not up. `reason: model_unavailable` means `capacity` recommended `none` and `submit` did not name a model.

## Environment

| Variable | This repo | Purpose |
|----------|-----------|---------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama |
| `LOCAL_AI_DB` | `data/local-ai.sqlite` | Queue and pins |
| `LOCAL_AI_JOBS` | `data/jobs` | One folder per task |
| `LOCAL_AI_MODEL_8B` | tag from `ollama list` | Ollama name for `qwen-8b` |
| `LOCAL_AI_MODEL_14B` | tag from `ollama list` | Ollama name for `qwen-14b` |
| `LOCAL_AI_MODEL_24B` | tag from `ollama list` | Ollama name for `qwen-24b` |
| `LOCAL_AI_HTTP_PORT` | unset for stdio | Set only for the ChatGPT HTTP transport |

The logical names `qwen-8b`, `qwen-14b`, and `qwen-24b` are what `submit` stores. The `LOCAL_AI_MODEL_*` variables are the tags Ollama actually runs. The values in [.cursor/mcp.json](.cursor/mcp.json) are `qwen2.5:7b`, `qwen2.5:14b`, and `qwen2.5:32b`. Change them to the tags `ollama list` prints before you submit.

Logs go to `logs/app.log` and to stderr. Stdout stays empty so the MCP handshake is not corrupted. Job files and shell transcripts are under `data/jobs/<task_id>/`.

## Cursor

This checkout is already wired. [.cursor/mcp.json](.cursor/mcp.json) starts [scripts/run-mcp.sh](scripts/run-mcp.sh), which `cd`s to the repo and runs `.venv/bin/python -m local_ai` on stdio.

1. `ollama serve`
2. Open this folder in Cursor.
3. Settings → MCP. Enable **local-ai**. If it was already listed, reload the window so Cursor reads the new file.
4. In a chat that has tools enabled, ask it to call `capacity`, then `submit`, then `read`.

Paste this as the first try:

```text
Call the local-ai MCP tool capacity. If ollama is up, submit a task whose goal is "Write hello.txt in the job folder containing the single word hello." and whose done_when is "hello.txt exists and its contents are hello". Use the recommended model, priority 40, and the default job access. Then call read with the returned task_id until status is completed or failed, and tell me the pin status, the error, and the job folder path.
```

## Claude Desktop

Claude speaks the same stdio server. Add this block to `~/Library/Application Support/Claude/claude_desktop_config.json` under `mcpServers`, then quit and reopen Claude. Do not also leave the Cursor server running against the same database.

```json
"local-ai": {
  "command": "/Users/denispetelin/local-ai/scripts/run-mcp.sh",
  "args": [],
  "env": {
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "LOCAL_AI_DB": "/Users/denispetelin/local-ai/data/local-ai.sqlite",
    "LOCAL_AI_JOBS": "/Users/denispetelin/local-ai/data/jobs",
    "LOCAL_AI_MODEL_8B": "qwen2.5:7b",
    "LOCAL_AI_MODEL_14B": "qwen2.5:14b",
    "LOCAL_AI_MODEL_24B": "qwen2.5:32b"
  }
}
```

## ChatGPT

Stop the Cursor stdio server first. In this repo:

```bash
LOCAL_AI_HTTP_PORT=8765 \
OLLAMA_HOST=http://127.0.0.1:11434 \
LOCAL_AI_DB=/Users/denispetelin/local-ai/data/local-ai.sqlite \
LOCAL_AI_JOBS=/Users/denispetelin/local-ai/data/jobs \
LOCAL_AI_MODEL_8B=qwen2.5:7b \
LOCAL_AI_MODEL_14B=qwen2.5:14b \
LOCAL_AI_MODEL_24B=qwen2.5:32b \
.venv/bin/python -m local_ai
```

Point ChatGPT at `http://127.0.0.1:8765/mcp`. The process binds `127.0.0.1` only.

## Tests

```bash
uv run pytest tests/
```

The pytest gate uses `ScriptedLLM`. It does not call Ollama.
