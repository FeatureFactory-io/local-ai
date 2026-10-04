"""Stdio startup must keep log lines off stdout."""

import json
import os
import select
import subprocess
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = _ROOT / ".venv" / "bin" / "python"
_READ_TIMEOUT_S = 30.0
_STOP_TIMEOUT_S = 5.0
_LOG_MARK = "task_id="


def test_stdio_initialize_writes_no_log_line(tmp_path: Path) -> None:
    """The initialize handshake is JSON on stdout. Logs stay on stderr and logs/app.log."""
    process = _start(tmp_path)
    try:
        _send(process, _initialize())
        first = _read_line(process, _READ_TIMEOUT_S)
        _send(process, _initialized())
        _stop(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=_STOP_TIMEOUT_S)

    stdout = first + _read_remaining(process.stdout)
    stderr = _read_remaining(process.stderr)
    log_text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")

    _assert_stdout_is_only_jsonrpc(stdout)
    assert _LOG_MARK in stderr
    assert _LOG_MARK in log_text
    payload = json.loads(first)
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 1
    assert "result" in payload


def _start(tmp_path: Path) -> subprocess.Popen[str]:
    """
    Spawn the dispatcher on stdio with a temp database and job root.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: process as subprocess.Popen[str]. Example: a running python -m local_ai
    """
    env = os.environ.copy()
    env["LOCAL_AI_DB"] = str(tmp_path / "local-ai.sqlite")
    env["LOCAL_AI_JOBS"] = str(tmp_path / "jobs")
    env.pop("LOCAL_AI_HTTP_PORT", None)
    return subprocess.Popen(
        [str(_PYTHON), "-m", "local_ai"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        text=True,
    )


def _initialize() -> dict:
    """
    Build the MCP initialize request.

    :return: JSON-RPC request as dict. Example: {"jsonrpc": "2.0", "method": "initialize"}
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "stdio-hygiene", "version": "0"},
        },
    }


def _initialized() -> dict:
    """
    Build the notification that finishes the handshake.

    :return: JSON-RPC notification as dict. Example: {"jsonrpc": "2.0", "method": "notifications/initialized"}
    """
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _send(process: subprocess.Popen[str], payload: dict) -> None:
    """
    Write one JSON-RPC message and flush.

    :param process: dispatcher process as subprocess.Popen[str]. Example: the process from _start
    :param payload: message as dict. Example: {"jsonrpc": "2.0", "method": "initialize"}
    :return: None. Example: None
    """
    if process.stdin is None:
        raise RuntimeError("stdin was not piped")
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def _read_line(process: subprocess.Popen[str], timeout: float) -> str:
    """
    Read one stdout line, or fail when the process stays silent.

    :param process: dispatcher process as subprocess.Popen[str]. Example: the process from _start
    :param timeout: seconds to wait as float. Example: 30.0
    :return: one line as str. Example: '{"jsonrpc":"2.0","id":1,"result":{}}\\n'
    :raises AssertionError: when stdout stays empty. Example: a hung process
    """
    if process.stdout is None:
        raise RuntimeError("stdout was not piped")
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise AssertionError("initialize produced no stdout line")
    return process.stdout.readline()


def _stop(process: subprocess.Popen[str]) -> None:
    """
    Ask the process to exit and wait.

    :param process: dispatcher process as subprocess.Popen[str]. Example: the process from _start
    :return: None. Example: None
    """
    process.terminate()
    try:
        process.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_STOP_TIMEOUT_S)


def _read_remaining(stream: object) -> str:
    """
    Read whatever is already buffered. A missing stream is empty.

    :param stream: pipe as object. Example: process.stderr
    :return: text as str. Example: "local_ai | task_id= | compose | recovered=0"
    """
    if stream is None or not hasattr(stream, "read"):
        return ""
    return stream.read()


def _assert_stdout_is_only_jsonrpc(stdout: str) -> None:
    """
    Every stdout line is a JSON-RPC message and none is a log line.

    :param stdout: captured stdout as str. Example: '{"jsonrpc":"2.0","id":1}\\n'
    :return: None. Example: None
    :raises AssertionError: when a line is a log or not JSON-RPC. Example: "INFO task_id="
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines, "stdout was empty"
    for line in lines:
        assert _LOG_MARK not in line
        payload = json.loads(line)
        assert payload.get("jsonrpc") == "2.0"
