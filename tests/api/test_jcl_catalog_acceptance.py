"""Acceptance: spin the catalog and the JCL process, then collect a result over HTTP.

Ollama is a local HTTP stand-in, the same shape the stdio acceptance tests use.
The catalog process, the JCL process, registration, discovery, submit, read, the
worker, the shell, SQLite, and LiteLLM are the real ones.
"""

import json
import os
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = _ROOT / ".venv" / "bin" / "python"
_MODEL_TAG = "qwen3:14b"
_CATALOG_TOKEN = "catalog-acceptance-secret"
_HOST_NAME = "acceptance-host"
_GOAL = "Write hello.txt in the job folder containing the single word hello."
_DONE_WHEN = "hello.txt exists and its contents are hello"
_HELLO = "hello\n"
_READY_TIMEOUT_S = 15.0
_POLL_TIMEOUT_S = 20.0


class _OllamaHandler(BaseHTTPRequestHandler):
    """Answers capacity GETs and the chat posts the worker sends."""

    replies: list[dict] = []

    def do_GET(self) -> None:
        """Return tags or the loaded model. Anything else is 404."""
        if self.path.split("?", 1)[0] == "/api/tags":
            self._send({"models": [{"name": _MODEL_TAG}]})
            return
        if self.path.split("?", 1)[0] == "/api/ps":
            self._send({"models": [{"name": _MODEL_TAG}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        """Return the next scripted chat message."""
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path.split("?", 1)[0] != "/api/chat":
            self.send_error(404)
            return
        if not type(self).replies:
            self.send_error(500, "no chat reply left")
            return
        self._send(type(self).replies.pop(0))

    def log_message(self, format: str, *args: object) -> None:
        """Stay off the test stderr."""
        return

    def _send(self, body: dict) -> None:
        """Write one JSON object."""
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_frontier_discovers_the_host_and_collects_hello(tmp_path: Path) -> None:
    """Start both processes, read the catalog, submit on the advertised URL, and read the file."""
    _OllamaHandler.replies = _chat_script(_write_command())
    ollama_host, ollama_port, ollama = _start_ollama()
    catalog_port = _free_port()
    jcl_port = _free_port()
    catalog_url = f"http://127.0.0.1:{catalog_port}"
    advertise_url = f"http://127.0.0.1:{jcl_port}"
    catalog_log = tmp_path / "catalog.log"
    jcl_log = tmp_path / "jcl.log"
    catalog = _spawn("local_ai.catalog", _catalog_env(tmp_path, catalog_port), tmp_path, catalog_log)
    jcl = None
    try:
        _wait_for_catalog(catalog_url, catalog, catalog_log)
        jcl = _spawn(
            "local_ai.jcl",
            _jcl_env(tmp_path, jcl_port, catalog_url, advertise_url, ollama_host, ollama_port),
            tmp_path,
            jcl_log,
        )
        record = _wait_for_host(catalog_url, advertise_url, catalog, catalog_log, jcl, jcl_log)
        token = _token_from(record)
        accepted = _post(f"{record['advertise_url']}/jcl/submit", token, _submit_body())
        pin = _poll_read(str(record["advertise_url"]), token, str(accepted["task_id"]), jcl, jcl_log)
    finally:
        _stop(jcl)
        _stop(catalog)
        ollama.shutdown()

    job = tmp_path / "jobs" / str(accepted["task_id"])
    assert record["name"] == _HOST_NAME
    assert record["advertise_url"] == advertise_url
    assert record["capacity"]["ollama"] == "up"
    assert token
    assert accepted["status"] == "queued"
    assert accepted["priority"] == 50
    assert pin["status"] == "completed", pin.get("error")
    assert pin["goal"] == _GOAL
    assert (job / "hello.txt").read_text(encoding="utf-8") == _HELLO
    assert (job / "shell.log").is_file()


def _start_ollama() -> tuple[str, int, ThreadingHTTPServer]:
    """Bind a stand-in Ollama on a free port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return str(host), int(port), server


def _free_port() -> int:
    """A loopback port the next process can bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _catalog_env(tmp_path: Path, port: int) -> dict[str, str]:
    """Environment for the catalog process."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "LOCAL_AI_CATALOG_TOKEN": _CATALOG_TOKEN,
        "LOCAL_AI_CATALOG_HOST": "127.0.0.1",
        "LOCAL_AI_CATALOG_PORT": str(port),
        "LOCAL_AI_CATALOG_DB": str(tmp_path / "catalog.sqlite"),
    }


def _jcl_env(
    tmp_path: Path,
    port: int,
    catalog_url: str,
    advertise_url: str,
    ollama_host: str,
    ollama_port: int,
) -> dict[str, str]:
    """Environment for the JCL process. The lane-50 token is what the catalog will publish."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
        "OLLAMA_HOST": f"http://{ollama_host}:{ollama_port}",
        "LOCAL_AI_DB": str(tmp_path / "local-ai.sqlite"),
        "LOCAL_AI_JOBS": str(tmp_path / "jobs"),
        "LOCAL_AI_MODEL_8B": _MODEL_TAG,
        "LOCAL_AI_MODEL_14B": _MODEL_TAG,
        "LOCAL_AI_MODEL_24B": _MODEL_TAG,
        "LOCAL_AI_JCL_TOKEN_50": "lane-50-from-the-catalog",
        "LOCAL_AI_JCL_HOST": "127.0.0.1",
        "LOCAL_AI_JCL_PORT": str(port),
        "LOCAL_AI_CATALOG_URL": catalog_url,
        "LOCAL_AI_ADVERTISE_URL": advertise_url,
        "LOCAL_AI_HOST_NAME": _HOST_NAME,
        "LOCAL_AI_CATALOG_REFRESH_S": "1",
    }


def _spawn(module: str, env: dict[str, str], cwd: Path, log_path: Path) -> subprocess.Popen[str]:
    """Start one real process. Its output goes to a log file so a full pipe cannot stall it."""
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(_PYTHON), "-m", module],
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.log_file = log_file  # type: ignore[attr-defined]
    return process


def _wait_for_catalog(catalog_url: str, process: subprocess.Popen[str], log_path: Path) -> None:
    """Block until GET /hosts answers. A missing token is still proof the process is listening."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        _require_alive(process, log_path, "catalog")
        status, _body = _request("GET", f"{catalog_url}/hosts")
        if status in {200, 401}:
            return
        time.sleep(0.05)
    raise AssertionError(f"catalog did not listen. log={_tail(log_path)}")


def _wait_for_host(
    catalog_url: str,
    advertise_url: str,
    catalog: subprocess.Popen[str],
    catalog_log: Path,
    jcl: subprocess.Popen[str],
    jcl_log: Path,
) -> dict:
    """Poll the catalog until this host's row is present. The row is what the client may use."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    last: object = None
    while time.monotonic() < deadline:
        _require_alive(catalog, catalog_log, "catalog")
        _require_alive(jcl, jcl_log, "jcl")
        status, body = _request("GET", f"{catalog_url}/hosts", token=_CATALOG_TOKEN)
        last = body
        if status == 200:
            for host in body.get("hosts", []):
                if host.get("advertise_url") == advertise_url:
                    return host
        time.sleep(0.05)
    raise AssertionError(
        f"host {advertise_url} was not listed. last={last!r} "
        f"catalog={_tail(catalog_log)} jcl={_tail(jcl_log)}"
    )


def _token_from(record: dict) -> str:
    """The bearer secret the catalog published. The client does not read it from the environment."""
    tokens = record.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise AssertionError(f"catalog row has no tokens: {record!r}")
    secret = tokens[0].get("token")
    if not isinstance(secret, str) or not secret:
        raise AssertionError(f"catalog token was {secret!r}")
    return secret


def _submit_body() -> dict:
    """The task a frontier model sends after it has learned the token."""
    return {"goal": _GOAL, "done_when": _DONE_WHEN}


def _post(url: str, token: str, body: dict) -> dict:
    """POST JSON and require HTTP 200."""
    status, payload = _request("POST", url, body, token)
    if status != 200:
        raise AssertionError(f"{url} returned {status}: {payload!r}")
    return payload


def _poll_read(
    advertise_url: str,
    token: str,
    task_id: str,
    process: subprocess.Popen[str],
    log_path: Path,
) -> dict:
    """Read the pin until it is completed or failed."""
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    last: dict = {}
    while time.monotonic() < deadline:
        _require_alive(process, log_path, "jcl")
        last = _post(f"{advertise_url}/jcl/read", token, {"task_id": task_id})
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"pin stayed {last.get('status')}: {last.get('error')} log={_tail(log_path)}")


def _request(method: str, url: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    """One real HTTP call. HTTP errors return their status and JSON body."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, _json_body(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json_body(exc.read())
    except urllib.error.URLError:
        return 0, {}


def _json_body(raw: bytes) -> dict:
    """Decode a JSON object. A non-object becomes an empty dict."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _chat_script(command: str) -> list[dict]:
    """Plan, write the file, then say the step is done."""
    return [
        _chat("1. Write hello.txt", thinking="The word hello belongs in hello.txt."),
        _chat("", thinking="Writing hello.txt.", command=command),
        _chat("Wrote hello.txt", thinking="hello.txt contains hello."),
    ]


def _chat(content: str, thinking: str, command: str | None = None) -> dict:
    """One Ollama /api/chat body."""
    message: dict = {"role": "assistant", "content": content, "thinking": thinking}
    if command is not None:
        message["tool_calls"] = [
            {"function": {"name": "shell", "arguments": {"command": command}}}
        ]
    return {
        "model": _MODEL_TAG,
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 8,
    }


def _write_command() -> str:
    """Shell command that writes hello.txt in the job directory."""
    script = f"open('hello.txt','w').write({_HELLO!r})"
    return "python3 -c " + shlex.quote(script)


def _require_alive(process: subprocess.Popen[str], log_path: Path, name: str) -> None:
    """Fail when the process has already exited."""
    code = process.poll()
    if code is not None:
        raise AssertionError(f"{name} exited {code}. log={_tail(log_path)}")


def _tail(log_path: Path) -> str:
    """The end of a process log, for a failed wait."""
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-2000:]


def _stop(process: subprocess.Popen[str] | None) -> None:
    """Stop one process. A process that was never started is left alone."""
    if process is None:
        return
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    log_file = getattr(process, "log_file", None)
    if log_file is not None:
        log_file.close()
