"""Acceptance: a real MCP client talks to python -m local_ai over stdio.

Ollama is a local HTTP stand-in. The worker, the shell, SQLite, and LiteLLM are the real ones.
"""

import json
import os
import select
import shlex
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = _ROOT / ".venv" / "bin" / "python"
_READ_TIMEOUT_S = 30.0
_POLL_TIMEOUT_S = 20.0
_MODEL_TAG = "qwen3:14b"
_GOAL = "Write a small poem about the beauty of Python and save it as poem.md."
_DONE_WHEN = "poem.md exists in the job folder and has 8 lines about Python."
_POEM = "\n".join(
    (
        "Python bends like a river of names,",
        "each indent a path the reader can trust,",
        "functions small enough to hold in one hand,",
        "errors that speak before the damage is done,",
        "lists and dicts, the pockets of a coat,",
        "a REPL that answers while you are still thinking,",
        "libraries waiting, patient as dawn,",
        "beauty is the code you can prove.",
    )
) + "\n"


class _OllamaHandler(BaseHTTPRequestHandler):
    """Answers the two capacity GETs and the chat posts the worker sends."""

    replies: list[dict] = []
    chat_posts: int = 0

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
        type(self).chat_posts += 1
        if not type(self).replies:
            self.send_error(500, "no chat reply left")
            return
        self._send(type(self).replies.pop(0))

    def log_message(self, format: str, *args: object) -> None:
        """Stay off stderr. The test reads the dispatcher streams itself."""
        return

    def _send(self, body: dict) -> None:
        """Write one JSON object."""
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_stdio_client_finishes_a_poem_and_keeps_stdout_json(tmp_path: Path) -> None:
    """A client calls capacity, submit, and read. The pin completes and stdout stays JSON."""
    host, port, server = _start_ollama(_chat_script(_write_command()))
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        capacity = _tool(process, 2, "capacity", {}, stdout)
        accepted = _tool(process, 3, "submit", _submit_arguments(), stdout)
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    poem = tmp_path / "jobs" / str(accepted["task_id"]) / "poem.md"
    assert capacity["status"] == "ok"
    assert capacity["ollama"] == "up"
    assert accepted["status"] == "queued"
    assert accepted["model"] == "qwen-14b"
    assert pin["status"] == "completed", pin.get("error")
    assert pin["phase"] == "completed"
    assert pin["error"] == ""
    assert pin["goal"] == _GOAL
    assert any("poem" in note["text"].lower() for note in pin["reasoning"])
    assert poem.read_text(encoding="utf-8") == _POEM
    assert (poem.parent / "shell.log").is_file()
    _assert_stdout_is_only_jsonrpc(stdout)


_CODE_GOAL = "Write add.py with a function add(left, right) that returns left + right, then run it."
_CODE_DONE = "python3 add.py prints 5."
_ADD_PY = "def add(left: int, right: int) -> int:\n    return left + right\n\n\nif __name__ == '__main__':\n    print(add(2, 3))\n"


def test_stdio_client_generates_code_that_runs(tmp_path: Path) -> None:
    """Submit a code task. The generated add.py runs and prints 5."""
    host, port, server = _start_ollama(_code_script())
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        accepted = _tool(process, 2, "submit", _code_submit(), stdout)
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    job = tmp_path / "jobs" / str(accepted["task_id"])
    source = (job / "add.py").read_text(encoding="utf-8")
    assert accepted["status"] == "queued"
    assert pin["status"] == "completed", pin.get("error")
    assert pin["goal"] == _CODE_GOAL
    assert "def add" in source
    assert "return left + right" in source
    ran = subprocess.run(
        [_PYTHON, str(job / "add.py")],
        cwd=job,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0
    assert ran.stdout.strip() == "5"
    assert "5" in (job / "shell.log").read_text(encoding="utf-8")
    _assert_stdout_is_only_jsonrpc(stdout)


_SUMMARY_GOAL = "Summarize ch-01.md and ch-02.md into summary.md, one thesis line per chapter."
_SUMMARY_DONE = "summary.md names ch-01 and ch-02 and repeats each chapter's first line."
_CHAPTERS = {
    "ch-01.md": "The river is the argument.\nThe rest of the chapter wanders.\n",
    "ch-02.md": "The indent is the proof.\nExamples follow the claim.\n",
}
_SUMMARIZE = """
from pathlib import Path
lines = []
for path in sorted(Path('.').glob('ch-*.md')):
    thesis = path.read_text(encoding='utf-8').splitlines()[0]
    lines.append(path.name + ': ' + thesis)
Path('summary.md').write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
""".strip()


def test_stdio_client_summarizes_copied_chapters(tmp_path: Path) -> None:
    """Submit two chapter files. The worker writes a summary that quotes each thesis."""
    sources = _write_chapters(tmp_path)
    host, port, server = _start_ollama(_summary_script())
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        accepted = _tool(process, 2, "submit", _summary_submit(sources), stdout)
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    job = tmp_path / "jobs" / str(accepted["task_id"])
    summary = (job / "summary.md").read_text(encoding="utf-8")
    assert accepted["status"] == "queued"
    assert pin["status"] == "completed", pin.get("error")
    assert pin["goal"] == _SUMMARY_GOAL
    assert (job / "ch-01.md").is_file()
    assert (job / "ch-02.md").is_file()
    assert "ch-01.md: The river is the argument." in summary
    assert "ch-02.md: The indent is the proof." in summary
    assert any("summary" in note["text"].lower() for note in pin["reasoning"])
    _assert_stdout_is_only_jsonrpc(stdout)


_MARKER_GOAL = "Write marker.txt in the job folder."
_MARKER_DONE = "marker.txt exists"


def test_stdio_client_rejects_a_shell_planning_reply(tmp_path: Path) -> None:
    """A shell call is not a plan. The numbered step runs later and writes marker.txt."""
    early = "python3 -c " + shlex.quote("open('early.txt','w').write('early\\n')")
    write_marker = "python3 -c " + shlex.quote("open('marker.txt','w').write('marker\\n')")
    host, port, server = _start_ollama(
        [
            _chat("", thinking="I will just run it.", command=early),
            _chat("1. Write marker.txt", thinking="One step is enough."),
            _chat("Wrote marker.txt", thinking="Writing marker.txt.", command=write_marker),
        ]
    )
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        accepted = _tool(
            process,
            2,
            "submit",
            {
                "goal": _MARKER_GOAL,
                "done_when": _MARKER_DONE,
                "priority": 40,
                "model": "qwen-14b",
                "access": {"mode": "job"},
            },
            stdout,
        )
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    job = tmp_path / "jobs" / str(accepted["task_id"])
    assert accepted["status"] == "queued"
    assert pin["status"] == "completed", pin.get("error")
    assert pin["steps"][0]["action"] == "Write marker.txt"
    assert not (job / "early.txt").exists()
    assert (job / "marker.txt").read_text(encoding="utf-8") == "marker\n"
    _assert_stdout_is_only_jsonrpc(stdout)


def test_stdio_client_fails_when_planning_never_numbers_steps(tmp_path: Path) -> None:
    """Three planning replies with no numbered step fail the pin and run no command."""
    command = "python3 -c " + shlex.quote("open('should-not-exist.txt','w').write('no\\n')")
    host, port, server = _start_ollama(
        [_chat("", thinking="Running it.", command=command) for _index in range(3)]
    )
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        accepted = _tool(
            process,
            2,
            "submit",
            {
                "goal": _MARKER_GOAL,
                "done_when": _MARKER_DONE,
                "priority": 40,
                "model": "qwen-14b",
                "access": {"mode": "job"},
            },
            stdout,
        )
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    job = tmp_path / "jobs" / str(accepted["task_id"])
    assert pin["status"] == "failed"
    assert pin["error"] == "empty_plan"
    assert pin["steps"] == []
    assert not (job / "should-not-exist.txt").exists()
    _assert_stdout_is_only_jsonrpc(stdout)


_CAP_GOAL = "Write marker.txt in the job folder."
_CAP_DONE = "marker.txt exists"
_FIRST_FAIL = "false"
_SECOND_FAIL = "python3 -c \"import sys; sys.stderr.write('step-two\\n'); raise SystemExit(2)\""
_THIRD_FAIL = "python3 -c \"import sys; sys.stderr.write('step-three\\n'); raise SystemExit(3)\""


def test_stdio_client_stops_after_three_failed_steps(tmp_path: Path) -> None:
    """Three different failing commands fail the pin and do not ask for a fourth reply."""
    assert len({_FIRST_FAIL, _SECOND_FAIL, _THIRD_FAIL}) == 3
    host, port, server = _start_ollama(_three_failures())
    process = _start_dispatcher(tmp_path, host, port)
    stdout: list[str] = []
    try:
        _send(process, _initialize())
        _response(process, 1, stdout)
        _send(process, _initialized())
        accepted = _tool(process, 2, "submit", _cap_submit(), stdout)
        pin = _poll_until_terminal(process, str(accepted["task_id"]), stdout)
    finally:
        _stop(process)
        server.shutdown()

    failed = [step for step in pin["steps"] if step["status"] == "failed"]
    assert accepted["status"] == "queued"
    assert pin["status"] == "failed"
    assert pin["error"] == "failed_step_cap"
    assert len(failed) == 3
    assert len(pin["steps"]) == 3
    assert _OllamaHandler.chat_posts == 4
    assert any("failed_step_cap" in note["text"] and "3" in note["text"] for note in pin["reasoning"])
    _assert_stdout_is_only_jsonrpc(stdout)


def _three_failures() -> list[dict]:
    """Plan with one failing command, then two different failing replacements."""
    return [
        _chat("1. Run the check", thinking="One numbered step."),
        _chat("", thinking="Trying the first command.", command=_FIRST_FAIL),
        _chat("Run a different check", thinking="The first command failed.", command=_SECOND_FAIL),
        _chat("Run one more check", thinking="The second command failed.", command=_THIRD_FAIL),
    ]


def _cap_submit() -> dict:
    """The submit a person would send for the failed-step cap."""
    return {
        "goal": _CAP_GOAL,
        "done_when": _CAP_DONE,
        "priority": 40,
        "model": "qwen-14b",
        "access": {"mode": "job"},
    }


def _write_chapters(tmp_path: Path) -> list[str]:
    """Write the two source chapters and return their paths."""
    folder = tmp_path / "chapters"
    folder.mkdir()
    paths: list[str] = []
    for name, text in _CHAPTERS.items():
        path = folder / name
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
    return paths


def _summary_script() -> list[dict]:
    """Plan, then write summary.md from the copied chapter files."""
    command = "python3 -c " + shlex.quote(_SUMMARIZE)
    return [
        _chat("1. Write summary.md from ch-01.md and ch-02.md", thinking="One thesis line per chapter."),
        _chat("Wrote summary.md", thinking="Writing summary.md from the chapter files.", command=command),
    ]


def _summary_submit(sources: list[str]) -> dict:
    """The submit a person would send to summarize the two chapters."""
    return {
        "goal": _SUMMARY_GOAL,
        "done_when": _SUMMARY_DONE,
        "priority": 40,
        "model": "qwen-14b",
        "inputs": sources,
        "access": {"mode": "job"},
    }


def _code_script() -> list[dict]:
    """Plan, write add.py, then run it."""
    write = "python3 -c " + shlex.quote(f"open('add.py','w').write({_ADD_PY!r})")
    return [
        _chat("1. Write add.py\n2. Run python3 add.py", thinking="add returns the sum of its arguments."),
        _chat("Wrote add.py", thinking="Writing add.py.", command=write),
        _chat("Printed 5", thinking="Running add.py.", command="python3 add.py"),
    ]


def _code_submit() -> dict:
    """The submit a person would send to generate add.py."""
    return {
        "goal": _CODE_GOAL,
        "done_when": _CODE_DONE,
        "priority": 40,
        "model": "qwen-14b",
        "access": {"mode": "job"},
    }


def _start_ollama(replies: list[dict]) -> tuple[str, int, ThreadingHTTPServer]:
    """Bind a stand-in Ollama on an ephemeral port."""
    _OllamaHandler.replies = replies
    _OllamaHandler.chat_posts = 0
    socket_server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    threading.Thread(target=socket_server.serve_forever, daemon=True).start()
    host, port = socket_server.server_address[:2]
    return str(host), int(port), socket_server


def _start_dispatcher(tmp_path: Path, host: str, port: int) -> subprocess.Popen[str]:
    """Spawn python -m local_ai on stdio, aimed at the stand-in Ollama."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OLLAMA_HOST": f"http://{host}:{port}",
        "LOCAL_AI_DB": str(tmp_path / "local-ai.sqlite"),
        "LOCAL_AI_JOBS": str(tmp_path / "jobs"),
        "LOCAL_AI_MODEL_8B": _MODEL_TAG,
        "LOCAL_AI_MODEL_14B": _MODEL_TAG,
        "LOCAL_AI_MODEL_24B": _MODEL_TAG,
    }
    return subprocess.Popen(
        [str(_PYTHON), "-m", "local_ai"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        text=True,
    )


def _chat_script(command: str) -> list[dict]:
    """Plan, then write the file, then say the step is done."""
    return [
        _chat("1. Write poem.md", thinking="A short poem belongs in poem.md."),
        _chat("", thinking="Writing poem.md.", command=command),
        _chat("Wrote poem.md", thinking="poem.md is in the job folder."),
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
    """Shell command that writes the poem into the job directory."""
    script = f"open('poem.md','w').write({_POEM!r})"
    return "python3 -c " + shlex.quote(script)


def _submit_arguments() -> dict:
    """The submit a person would send for this poem."""
    return {
        "goal": _GOAL,
        "done_when": _DONE_WHEN,
        "priority": 40,
        "model": "qwen-14b",
        "access": {"mode": "job"},
    }


def _initialize() -> dict:
    """MCP initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "acceptance", "version": "0"},
        },
    }


def _initialized() -> dict:
    """Notification that finishes the handshake."""
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _tool(
    process: subprocess.Popen[str],
    request_id: int,
    name: str,
    arguments: dict,
    stdout: list[str],
) -> dict:
    """Call one MCP tool and return its JSON body."""
    _send(
        process,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    message = _response(process, request_id, stdout)
    if "error" in message:
        raise AssertionError(message["error"])
    return _tool_body(message["result"])


def _poll_until_terminal(
    process: subprocess.Popen[str],
    task_id: str,
    stdout: list[str],
) -> dict:
    """Read the pin until it is completed or failed."""
    import time

    deadline = time.monotonic() + _POLL_TIMEOUT_S
    request_id = 4
    last: dict = {}
    while time.monotonic() < deadline:
        last = _tool(process, request_id, "read", {"task_id": task_id}, stdout)
        request_id += 1
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"pin stayed {last.get('status')}: {last.get('error')}")


def _tool_body(result: dict) -> dict:
    """Unwrap a tools/call result into the dict the tool returned."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return json.loads(content[0]["text"])
    raise AssertionError(f"tool result was {result!r}")


def _send(process: subprocess.Popen[str], payload: dict) -> None:
    """Write one JSON-RPC message."""
    if process.stdin is None:
        raise RuntimeError("stdin was not piped")
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def _response(process: subprocess.Popen[str], request_id: int, stdout: list[str]) -> dict:
    """Read stdout until the response with this id. Every line must be JSON."""
    while True:
        line = _read_line(process)
        stdout.append(line)
        message = _jsonrpc_line(line)
        if message.get("id") == request_id:
            return message


def _read_line(process: subprocess.Popen[str]) -> str:
    """Read one stdout line or fail when the process stays silent."""
    if process.stdout is None:
        raise RuntimeError("stdout was not piped")
    ready, _, _ = select.select([process.stdout], [], [], _READ_TIMEOUT_S)
    if not ready:
        raise AssertionError(_silence_error(process))
    return process.stdout.readline()


def _jsonrpc_line(line: str) -> dict:
    """Parse one stdout line. A log line is a failure."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"stdout was not JSON: {line!r}") from exc
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise AssertionError(f"stdout was not a JSON-RPC message: {line!r}")
    return message


def _assert_stdout_is_only_jsonrpc(stdout: list[str]) -> None:
    """Every captured line is JSON-RPC. LiteLLM text must not appear."""
    for line in stdout:
        _jsonrpc_line(line)
        assert "LiteLLM" not in line


def _silence_error(process: subprocess.Popen[str]) -> str:
    """Explain a hung read using whatever stderr has arrived."""
    stderr = ""
    if process.stderr is not None:
        ready, _, _ = select.select([process.stderr], [], [], 0)
        if ready:
            stderr = process.stderr.read()
    return f"no stdout line. stderr={stderr!r}"


def _stop(process: subprocess.Popen[str]) -> None:
    """Stop the dispatcher."""
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
