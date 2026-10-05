# Local AI

A dispatcher that lets Cursor, Claude Desktop, or ChatGPT hand one task at a time to a local Qwen model served by Ollama. The frontier model never gets a shell. It calls the dispatcher MCP tools. The dispatcher pins the task, runs one worker, and writes progress on the pin.

The contract is [docs/SAO.md](docs/SAO.md). This file is how to start that server and attach it. A JCL process exposes the same tools over HTTP so a frontier model can curl another machine. An optional catalog lists those machines.

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
5. `list` — compact rows. Optional `status`: `queued`, `running`, `completed`, `failed`, or `all`.
6. `cancel` — delete one pin that is still `queued`, plus its job folder.
7. `purge` — delete `completed` and `failed` pins and their job folders. A `running` pin stays.

`submit` fields:

| Field | Required | Meaning |
|-------|----------|---------|
| `goal` | yes | What the worker must achieve |
| `done_when` | yes | How you will know it finished |
| `priority` | no | Integer 1 (first) through 100 (last). Default 50 |
| `model` | no | `qwen-8b`, `qwen-14b`, or `qwen-24b`. Default is `recommended` from `capacity` |
| `inputs` | no | File paths copied into the job folder when `access.mode` is `job` |
| `access` | no | `{ "mode": "job" }` (default), `{ "mode": "paths", "paths": ["/absolute/dir"] }`, or `{ "mode": "home" }` |
| `plan` | no | Ordered step sentences. When present, the worker skips planning and acts on these steps |

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
| `LOCAL_AI_BOARD_PORT` | `8766` | Read-only blackboard page. Binds `127.0.0.1` only |
| `LOCAL_AI_JCL_TOKEN_<priority>` | required to serve JCL | Bearer secret. The suffix is the best priority that token may set |
| `LOCAL_AI_JCL_HOST` | `127.0.0.1` | Set to `0.0.0.0` when another machine will curl this host |
| `LOCAL_AI_JCL_PORT` | `8770` | JCL API port |
| `LOCAL_AI_CATALOG_URL` | unset | Catalog base URL. When set, `LOCAL_AI_ADVERTISE_URL` is required |
| `LOCAL_AI_ADVERTISE_URL` | unset | URL other machines use to reach this JCL API |
| `LOCAL_AI_HOST_NAME` | machine hostname | Name stored on the catalog |
| `LOCAL_AI_CATALOG_REFRESH_S` | `60` | Seconds between catalog registration posts |
| `LOCAL_AI_CATALOG_TOKEN` | required to serve the catalog | Bearer secret for `GET /hosts` |
| `LOCAL_AI_CATALOG_HOST` | `127.0.0.1` | Set to `0.0.0.0` on the catalog machine when other hosts must reach it |
| `LOCAL_AI_CATALOG_PORT` | `8771` | Catalog port |
| `LOCAL_AI_CATALOG_DB` | `data/catalog.sqlite` | Catalog phone book. Not the dispatcher database |
| `LOCAL_AI_CATALOG_STALE_S` | `180` | Age after which a listed host is marked stale |

The logical names `qwen-8b`, `qwen-14b`, and `qwen-24b` are what `submit` stores. The `LOCAL_AI_MODEL_*` variables are the tags Ollama actually runs. This machine maps `qwen-8b` to `qwen3.5:9b`, `qwen-14b` to `qwen3:14b`, and `qwen-24b` to `qwen3.5:27b`. Qwen3.5 has no 14B tag. Change the variables if `ollama list` changes.

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

### Job control

The same stdio server exposes `list`, `cancel`, and `purge`. Paste this when a pin is still `queued` and finished pins should go:

```text
Call the local-ai MCP tool list with status "queued". From jobs, take one row and note task_id and position. Call cancel with that task_id. Then call purge with no task_id and no status, so both completed and failed pins are removed. Report list status and count, the task_id and position you cancelled, cancel status and job_dir_removed, and purge status and count.
```

`list` takes optional `status` and returns `status`, `count`, and `jobs`. A queued row sets `position` to an integer. `cancel` takes `task_id` and returns `status`, `task_id`, and `job_dir_removed`. `purge` takes optional `task_id` and optional `status` (`completed` or `failed`). Omit both to delete every finished pin. It returns `status`, `count`, and `purged` (each entry has `task_id` and `job_dir_removed`).

Cancel of a running `task_id` returns `status: rejected` and `reason: not_queued`. Purge of a running `task_id` returns `status: rejected` and `reason: not_terminal`.

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
    "LOCAL_AI_MODEL_8B": "qwen3.5:9b",
    "LOCAL_AI_MODEL_14B": "qwen3:14b",
    "LOCAL_AI_MODEL_24B": "qwen3.5:27b"
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
LOCAL_AI_MODEL_8B=qwen3.5:9b \
LOCAL_AI_MODEL_14B=qwen3:14b \
LOCAL_AI_MODEL_24B=qwen3.5:27b \
.venv/bin/python -m local_ai
```

Point ChatGPT at `http://127.0.0.1:8765/mcp`. The process binds `127.0.0.1` only.

## JCL API

`python -m local_ai.jcl` runs the same dispatcher and the same worker as the MCP server, then serves the seven tools as POST routes. Do not run it against a database that another dispatcher process already has open.

The process reads tokens once at startup. The variable suffix is a priority from 1 (first) through 100 (last). That number is the best priority the token may set. Submit and reprioritize may use it or any worse priority, which is a larger integer. If submit omits `priority`, the pin uses the token's number. Any accepted token may call capacity, read, list, cancel, and purge.

The process refuses to listen when no token is set. The default bind is `127.0.0.1:8770`. Set `LOCAL_AI_JCL_HOST=0.0.0.0` when another machine on the LAN will curl this one. `inputs` are paths on this host.

From this repo:

```bash
LOCAL_AI_JCL_TOKEN_10=replace-with-a-long-secret \
LOCAL_AI_JCL_TOKEN_50=replace-with-another-secret \
LOCAL_AI_JCL_HOST=0.0.0.0 \
LOCAL_AI_JCL_PORT=8770 \
OLLAMA_HOST=http://127.0.0.1:11434 \
LOCAL_AI_DB=/Users/denispetelin/local-ai/data/local-ai.sqlite \
LOCAL_AI_JOBS=/Users/denispetelin/local-ai/data/jobs \
LOCAL_AI_MODEL_8B=qwen3.5:9b \
LOCAL_AI_MODEL_14B=qwen3:14b \
LOCAL_AI_MODEL_24B=qwen3.5:27b \
.venv/bin/python -m local_ai.jcl
```

Submit a task with the lane-50 token. The response `status` is `queued` even when the worker starts immediately. Repeat `read` until it is `completed` or `failed`.

```bash
curl -s -X POST http://127.0.0.1:8770/jcl/submit \
  -H "Authorization: Bearer replace-with-another-secret" \
  -H "Content-Type: application/json" \
  -d '{"goal":"Write hello.txt in the job folder containing the single word hello.","done_when":"hello.txt exists and its contents are hello"}'
```

| Route | Body |
|-------|------|
| `POST /jcl/capacity` | empty object |
| `POST /jcl/submit` | `goal`, `done_when`, optional `priority`, `model`, `inputs`, `access`, `plan` |
| `POST /jcl/read` | `task_id` |
| `POST /jcl/reprioritize` | `task_id`, `priority` |
| `POST /jcl/list` | optional `status` |
| `POST /jcl/cancel` | `task_id` |
| `POST /jcl/purge` | optional `task_id`, optional `status` |

Every route takes `Authorization: Bearer <token>`. A missing or unknown token is HTTP 401 `reason: unauthorized`. A priority better than the token allows is HTTP 403 `reason: priority_forbidden`. A body that is not a JSON object is HTTP 400 `reason: bad_json`. Dispatcher results, including `rejected`, stay HTTP 200 and use the same JSON as the MCP tools. The log records the route, the decision, and the priority. It does not record the token.

## Catalog

A catalog host, which can be a small machine such as a Pi, lists JCL hosts and the tokens, capacity, and models they last reported. It does not run Ollama or a worker. Its database is `data/catalog.sqlite`, not the dispatcher database.

On the catalog machine:

```bash
LOCAL_AI_CATALOG_TOKEN=replace-with-a-catalog-secret \
LOCAL_AI_CATALOG_HOST=0.0.0.0 \
LOCAL_AI_CATALOG_PORT=8771 \
.venv/bin/python -m local_ai.catalog
```

The default bind is `127.0.0.1:8771`. The process refuses to listen when `LOCAL_AI_CATALOG_TOKEN` is unset. `POST /register` does not check a token. `GET /hosts` requires the catalog token. That listing includes every host token, so the catalog token is the key to those secrets.

On each JCL machine, add the catalog URL and the URL other machines should curl. `LOCAL_AI_ADVERTISE_URL` is required when `LOCAL_AI_CATALOG_URL` is set. Without the advertise URL the JCL process does not open its database. The host posts at startup and again every `LOCAL_AI_CATALOG_REFRESH_S` seconds (default 60). A failed post is logged and the JCL API keeps serving.

```bash
LOCAL_AI_CATALOG_URL=http://192.168.1.40:8771 \
LOCAL_AI_ADVERTISE_URL=http://192.168.1.20:8770 \
LOCAL_AI_HOST_NAME=studio-mac \
```

Read the phone book:

```bash
curl -s http://192.168.1.40:8771/hosts \
  -H "Authorization: Bearer replace-with-a-catalog-secret"
```

Each host has `name`, `advertise_url`, `tokens`, `capacity`, `models`, `last_seen`, and `stale`. A row older than `LOCAL_AI_CATALOG_STALE_S` seconds (default 180) has `stale: true` and stays in the list. Use the advertise URL and one of that host's tokens to POST `/jcl/submit` on that machine.

## Blackboard page

A read-only page renders the same pins as `list` and `read`. It does not cancel, purge, submit, or reprioritize, and it does not migrate the database. Queued and running jobs are expanded and show the plan, the steps, and the reasoning.

From this repo:

```bash
.venv/bin/python -m local_ai.board
```

Open `http://127.0.0.1:8766/`. The process binds `127.0.0.1` only. `LOCAL_AI_BOARD_PORT` defaults to `8766` so it does not take the MCP HTTP port. `LOCAL_AI_DB` is the same SQLite file the dispatcher uses. The query `status` accepts `queued`, `running`, `completed`, `failed`, or `all` (the default). Any other value is rejected with HTTP 400 and `reason: bad_status`.

```text
http://127.0.0.1:8766/?status=queued
http://127.0.0.1:8766/?status=running
http://127.0.0.1:8766/?status=completed
http://127.0.0.1:8766/?status=failed
http://127.0.0.1:8766/?status=all
```

`http://127.0.0.1:8766/` and `?status=all` are the same filter.

## Tests

```bash
uv run pytest tests/
```

The pytest gate uses `ScriptedLLM`. It does not call Ollama.
