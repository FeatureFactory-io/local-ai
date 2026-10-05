"""Acceptance: a real stdio MCP client calls list, cancel, and purge.

The client is a subprocess, the same shape as tests/api/test_acceptance.py.
Finished pins are rows in the spawned server's SQLite file. A live pin is
submitted through the client. The stand-in Ollama answers capacity and then
holds the chat call so the worker stays in planning and the queue stays put.
"""

import json
import os
import select
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import AccessGrant, ReasoningNote, TaskPin
from local_ai.db.database import Database


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = _ROOT / ".venv" / "bin" / "python"
_READ_TIMEOUT_S = 30.0
_POLL_TIMEOUT_S = 20.0
_HOLD_S = 120.0
_MODEL_TAG = "qwen3:14b"
_OLDER_ID = "22222222-3333-4444-8555-666666666666"
_NEWER_ID = "33333333-4444-4555-8666-777777777777"
_MISSING_ID = "99999999-9999-4999-8999-999999999999"
_NOTE = "Thesis is on the first line."
_OLDER_GOAL = "Summarize the older chapter into summary.md."
_NEWER_GOAL = "Summarize the failed chapter into summary.md."
_HOLD_GOAL = "Hold the worker until the operator is done."
_BEHIND_GOAL = "Combine the later chapters into book.md."
_AHEAD_GOAL = "Combine the urgent chapters into book.md."
_HOLD_PRIORITY = 40
_BEHIND_PRIORITY = 30
_AHEAD_PRIORITY = 10
_STEP_ACTION = "List chapter files"
_ROW_KEYS = frozenset(
    {
        "task_id",
        "status",
        "priority",
        "model",
        "phase",
        "goal",
        "error",
        "created_at",
        "started_at",
        "finished_at",
        "position",
    }
)


@dataclass
class _Board:
    """Pins a held dispatcher is showing to the client."""

    client: "_Client"
    root: Path
    running_id: str
    ahead_id: str
    behind_id: str


class _HoldHandler(BaseHTTPRequestHandler):
    """Capacity answers immediately. Chat waits so the worker stays running."""

    release: threading.Event = threading.Event()

    def do_GET(self) -> None:
        """Return tags or the loaded model. Anything else is 404."""
        if self.path.split("?", 1)[0] in {"/api/tags", "/api/ps"}:
            self._send({"models": [{"name": _MODEL_TAG}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        """Hold /api/chat until the test releases it."""
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.split("?", 1)[0] != "/api/chat":
            self.send_error(404)
            return
        type(self).release.wait(timeout=_HOLD_S)
        self.send_error(500, "released")

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


class _Client:
    """One stdio dispatcher and the JSON-RPC calls made to it."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        ollama: ThreadingHTTPServer | None,
        release: threading.Event | None,
    ) -> None:
        """
        Store the process. The first request id is the initialize call.

        :param process: dispatcher as subprocess.Popen[str]. Example: python -m local_ai
        :param ollama: stand-in as ThreadingHTTPServer | None. Example: None
        :param release: chat hold as threading.Event | None. Example: None
        """
        self.process = process
        self.stdout: list[str] = []
        self._ollama = ollama
        self._release = release
        self._next_id = 1

    @classmethod
    def start(cls, tmp_path: Path, *, hold_chat: bool) -> "_Client":
        """
        Spawn python -m local_ai. hold_chat binds a stand-in Ollama first.

        :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
        :param hold_chat: whether to hold the worker as bool. Example: True
        :return: client as _Client. Example: a client whose process is running
        """
        env = _dispatcher_env(tmp_path)
        ollama: ThreadingHTTPServer | None = None
        release: threading.Event | None = None
        if hold_chat:
            host, port, ollama, release = _start_hold()
            env["OLLAMA_HOST"] = f"http://{host}:{port}"
            env["LOCAL_AI_MODEL_8B"] = _MODEL_TAG
            env["LOCAL_AI_MODEL_14B"] = _MODEL_TAG
            env["LOCAL_AI_MODEL_24B"] = _MODEL_TAG
        process = _spawn(tmp_path, env)
        return cls(process, ollama, release)

    def handshake(self) -> None:
        """
        Finish the MCP initialize exchange.

        :return: None. Example: None
        """
        self._send(_initialize())
        self._response(1)
        self._send(_initialized())
        self._next_id = 2

    def submit(self, priority: int, goal: str) -> str:
        """
        Submit one job-mode pin and return its id.

        :param priority: queue rank as int. Example: 30
        :param goal: worker goal as str. Example: "Combine the later chapters into book.md."
        :return: task id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :raises AssertionError: when submit does not queue the pin. Example: ollama_down
        """
        body = self.tool("submit", _submit_arguments(priority, goal))
        if body.get("status") != "queued":
            raise AssertionError(body)
        return str(body["task_id"])

    def submit_until_running(self, priority: int, goal: str) -> str:
        """
        Submit a pin and wait until the worker has published planning.

        :param priority: queue rank as int. Example: 40
        :param goal: worker goal as str. Example: "Hold the worker until the operator is done."
        :return: task id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        """
        task_id = self.submit(priority, goal)
        self._poll_until_planning(task_id)
        return task_id

    def tool(self, name: str, arguments: dict) -> dict:
        """
        Call one MCP tool and return its JSON body.

        :param name: tool name as str. Example: "list"
        :param arguments: tool arguments as dict. Example: {"status": "queued"}
        :return: tool body as dict. Example: {"status": "ok", "count": 0, "jobs": []}
        :raises AssertionError: when the response is a JSON-RPC error. Example: unknown tool
        """
        request_id = self._next_id
        self._next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        message = self._response(request_id)
        if "error" in message:
            raise AssertionError(message["error"])
        return _tool_body(message["result"])

    def close(self) -> None:
        """
        Release the chat hold, stop the dispatcher, and stop the stand-in.

        :return: None. Example: None
        """
        if self._release is not None:
            self._release.set()
        _stop(self.process)
        if self._ollama is not None:
            self._ollama.shutdown()
            self._ollama.server_close()

    def _poll_until_planning(self, task_id: str) -> dict:
        """
        Read until the pin is running and the worker has entered planning.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: pin body as dict. Example: {"status": "running", "phase": "planning"}
        :raises AssertionError: when the pin finishes or never reaches planning. Example: failed
        """
        deadline = time.monotonic() + _POLL_TIMEOUT_S
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.tool("read", {"task_id": task_id})
            if last.get("status") == "running" and last.get("phase") == "planning":
                return last
            if last.get("status") in {"completed", "failed", "rejected"}:
                raise AssertionError(_left_queue(last, self.process))
            time.sleep(0.05)
        raise AssertionError(_left_queue(last, self.process))

    def _send(self, payload: dict) -> None:
        """
        Write one JSON-RPC message.

        :param payload: message as dict. Example: {"jsonrpc": "2.0", "method": "initialize"}
        :return: None. Example: None
        :raises RuntimeError: when stdin was not piped. Example: a process without a pipe
        """
        if self.process.stdin is None:
            raise RuntimeError("stdin was not piped")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def _response(self, request_id: int) -> dict:
        """
        Read stdout until the response with this id.

        :param request_id: JSON-RPC id as int. Example: 2
        :return: message as dict. Example: {"jsonrpc": "2.0", "id": 2, "result": {}}
        """
        while True:
            line = _read_line(self.process)
            self.stdout.append(line)
            message = _jsonrpc_line(line)
            if message.get("id") == request_id:
                return message


def test_stdio_list_orders_running_queue_then_newest_finished(tmp_path: Path) -> None:
    """Omit status and pass all. Running, then the queue, then newest finished."""
    with _live_board(tmp_path) as board:
        read = board.client.tool("read", {"task_id": _OLDER_ID})
        omitted = board.client.tool("list", {})
        listed = board.client.tool("list", {"status": "all"})

        _assert_read_is_full(read)
        _assert_board_order(omitted, board)
        assert listed == omitted


def test_stdio_list_filtered_to_queued_returns_only_those_rows(tmp_path: Path) -> None:
    """status queued returns the queue in position order and nothing else."""
    with _live_board(tmp_path) as board:
        body = board.client.tool("list", {"status": "queued"})

        assert body["status"] == "ok"
        assert body["count"] == 2
        assert [row["task_id"] for row in body["jobs"]] == [board.ahead_id, board.behind_id]
        assert [row["position"] for row in body["jobs"]] == [1, 2]
        assert [row["priority"] for row in body["jobs"]] == [_AHEAD_PRIORITY, _BEHIND_PRIORITY]
        assert all(set(row) == _ROW_KEYS for row in body["jobs"])


def test_stdio_cancel_queued_pin_removes_the_pin_steps_and_job_directory(tmp_path: Path) -> None:
    """Cancel deletes a queued pin, its plan step, and its job directory."""
    with _live_board(tmp_path) as board:
        _insert_step(tmp_path, board.behind_id)
        job = _job(tmp_path, board.behind_id)
        (job / "notes.md").write_text("keep", encoding="utf-8")
        seen = board.client.tool("read", {"task_id": board.behind_id})
        body = board.client.tool("cancel", {"task_id": board.behind_id})
        missing = board.client.tool("read", {"task_id": board.behind_id})

        assert seen["status"] == "queued"
        assert seen["steps"][0]["action"] == _STEP_ACTION
        assert body == {
            "status": "cancelled",
            "task_id": board.behind_id,
            "job_dir_removed": True,
        }
        assert missing["status"] == "rejected"
        assert missing["reason"] == "not_found"
        assert _step_count(tmp_path, board.behind_id) == 0
        assert not job.exists()
        assert _task_exists(tmp_path, board.running_id)
        assert _task_exists(tmp_path, board.ahead_id)


def test_stdio_cancel_running_pin_is_rejected_and_keeps_the_folder(tmp_path: Path) -> None:
    """A running pin is not_queued. The pin and its job folder stay."""
    with _live_board(tmp_path) as board:
        job = _job(tmp_path, board.running_id)
        (job / "live.txt").write_text("live", encoding="utf-8")
        body = board.client.tool("cancel", {"task_id": board.running_id})
        pin = board.client.tool("read", {"task_id": board.running_id})

        assert body == {
            "status": "rejected",
            "reason": "not_queued",
            "task_id": board.running_id,
        }
        assert pin["status"] == "running"
        assert (job / "live.txt").read_text(encoding="utf-8") == "live"


def test_stdio_cancel_missing_id_is_not_found(tmp_path: Path) -> None:
    """An unknown id is not_found and does not create a pin."""
    with _stdio(tmp_path) as client:
        body = client.tool("cancel", {"task_id": _MISSING_ID})
        listed = client.tool("list", {})

        assert body == {
            "status": "rejected",
            "reason": "not_found",
            "task_id": _MISSING_ID,
        }
        assert listed == {"status": "ok", "count": 0, "jobs": []}


def test_stdio_purge_completed_pin_removes_its_folder_and_leaves_the_queue(tmp_path: Path) -> None:
    """Purge one completed pin. The queued neighbor and the failed pin stay."""
    with _live_board(tmp_path) as board:
        body = board.client.tool("purge", {"task_id": _OLDER_ID, "status": "completed"})
        listed = board.client.tool("list", {})

        assert body["status"] == "ok"
        assert body["count"] == 1
        assert body["purged"] == [
            {"task_id": _OLDER_ID, "prior_status": "completed", "job_dir_removed": True}
        ]
        assert not _job(tmp_path, _OLDER_ID).exists()
        assert not _task_exists(tmp_path, _OLDER_ID)
        assert _task_exists(tmp_path, _NEWER_ID)
        assert {row["task_id"] for row in listed["jobs"]} >= {board.ahead_id, board.behind_id}
        assert _OLDER_ID not in {row["task_id"] for row in listed["jobs"]}


def test_stdio_purge_without_task_id_removes_finished_pins_only(tmp_path: Path) -> None:
    """An open purge deletes completed and failed and leaves running and queued."""
    with _live_board(tmp_path) as board:
        body = board.client.tool("purge", {})
        listed = board.client.tool("list", {})
        left = {row["task_id"] for row in listed["jobs"]}

        assert body["status"] == "ok"
        assert body["count"] == 2
        assert {row["task_id"]: row["prior_status"] for row in body["purged"]} == {
            _OLDER_ID: "completed",
            _NEWER_ID: "failed",
        }
        assert all(row["job_dir_removed"] for row in body["purged"])
        assert not _job(tmp_path, _OLDER_ID).exists()
        assert not _job(tmp_path, _NEWER_ID).exists()
        assert left == {board.running_id, board.ahead_id, board.behind_id}
        assert {row["status"] for row in listed["jobs"]} == {"running", "queued"}


def test_stdio_purge_running_pin_is_not_terminal(tmp_path: Path) -> None:
    """A running task_id is not_terminal. The pin and its folder stay."""
    with _live_board(tmp_path) as board:
        job = _job(tmp_path, board.running_id)
        (job / "live.txt").write_text("live", encoding="utf-8")
        body = board.client.tool("purge", {"task_id": board.running_id})
        pin = board.client.tool("read", {"task_id": board.running_id})

        assert body == {
            "status": "rejected",
            "reason": "not_terminal",
            "task_id": board.running_id,
        }
        assert pin["status"] == "running"
        assert (job / "live.txt").read_text(encoding="utf-8") == "live"
        assert _task_exists(tmp_path, board.running_id)


def test_stdio_purge_completed_pin_as_failed_is_not_status(tmp_path: Path) -> None:
    """A completed pin stays when purge asked for failed."""
    _seed_pins(tmp_path, (_older_pin(),))
    marker = _write_marker(tmp_path, _OLDER_ID)
    with _stdio(tmp_path) as client:
        body = client.tool("purge", {"task_id": _OLDER_ID, "status": "failed"})
        pin = client.tool("read", {"task_id": _OLDER_ID})

        assert body == {
            "status": "rejected",
            "reason": "not_status",
            "task_id": _OLDER_ID,
            "filter": "failed",
        }
        assert pin["status"] == "completed"
        assert marker.read_text(encoding="utf-8") == "keep"


def test_stdio_list_and_purge_reject_a_bad_status(tmp_path: Path) -> None:
    """done and all are rejected. The completed pin and its folder stay."""
    _seed_pins(tmp_path, (_older_pin(),))
    marker = _write_marker(tmp_path, _OLDER_ID)
    with _stdio(tmp_path) as client:
        listed = client.tool("list", {"status": "done"})
        purged_all = client.tool("purge", {"status": "all"})
        purged_done = client.tool("purge", {"status": "done"})
        still = client.tool("list", {})

        assert listed == {"status": "rejected", "reason": "bad_status", "filter": "done"}
        assert purged_all == {"status": "rejected", "reason": "bad_status", "filter": "all"}
        assert purged_done == {"status": "rejected", "reason": "bad_status", "filter": "done"}
        assert [row["task_id"] for row in still["jobs"]] == [_OLDER_ID]
        assert marker.read_text(encoding="utf-8") == "keep"
        assert _task_exists(tmp_path, _OLDER_ID)


def test_stdio_cancel_unlinks_a_job_symlink_and_leaves_the_target(tmp_path: Path) -> None:
    """A job path that is a symlink is unlinked. The outside target stays."""
    with _live_board(tmp_path) as board:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep", encoding="utf-8")
        job = _job(tmp_path, board.ahead_id)
        assert job.is_dir() and not job.is_symlink()
        assert not outside.resolve().is_relative_to((tmp_path / "jobs").resolve())
        _insert_step(tmp_path, board.ahead_id)
        _replace_job_with_symlink(job, outside)
        body = board.client.tool("cancel", {"task_id": board.ahead_id})

        assert body["status"] == "cancelled"
        assert body["job_dir_removed"] is True
        assert not job.exists()
        assert not job.is_symlink()
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert _step_count(tmp_path, board.ahead_id) == 0
        assert not _task_exists(tmp_path, board.ahead_id)
        assert _task_exists(tmp_path, board.running_id)


@contextmanager
def _live_board(tmp_path: Path) -> Iterator[_Board]:
    """
    Seed two finished pins, then hold one running pin in front of two queued pins.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: board as Iterator[_Board]. Example: a board with one running pin
    """
    _seed_finished(tmp_path)
    client = _Client.start(tmp_path, hold_chat=True)
    try:
        client.handshake()
        running_id = client.submit_until_running(_HOLD_PRIORITY, _HOLD_GOAL)
        behind_id = client.submit(_BEHIND_PRIORITY, _BEHIND_GOAL)
        ahead_id = client.submit(_AHEAD_PRIORITY, _AHEAD_GOAL)
        board = _Board(client, tmp_path, running_id, ahead_id, behind_id)
        yield board
        _assert_stdout_is_only_jsonrpc(client.stdout)
    finally:
        client.close()


@contextmanager
def _stdio(tmp_path: Path) -> Iterator[_Client]:
    """
    Start a dispatcher with no stand-in model. The slot is never kicked.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: client as Iterator[_Client]. Example: a client past initialize
    """
    client = _Client.start(tmp_path, hold_chat=False)
    try:
        client.handshake()
        yield client
        _assert_stdout_is_only_jsonrpc(client.stdout)
    finally:
        client.close()


def _assert_read_is_full(pin: dict) -> None:
    """
    The full pin still carries the fields list leaves off.

    :param pin: read body as dict. Example: {"status": "completed", "reasoning": []}
    :return: None. Example: None
    """
    assert pin["status"] == "completed"
    assert pin["reasoning"][0]["text"] == _NOTE
    assert pin["artifacts"] == ["summary.md"]
    assert pin["access"] == {"mode": "paths", "paths": ["/tmp/books"]}
    assert pin["steps"][0]["action"] == _STEP_ACTION


def _assert_board_order(body: dict, board: _Board) -> None:
    """
    Running first, queued by priority, finished newest finished_at first.

    :param body: list body as dict. Example: {"status": "ok", "count": 5, "jobs": []}
    :param board: live ids as _Board. Example: the board from _live_board
    :return: None. Example: None
    """
    jobs = body["jobs"]
    assert body["status"] == "ok"
    assert body["count"] == 5
    assert [row["task_id"] for row in jobs] == [
        board.running_id,
        board.ahead_id,
        board.behind_id,
        _NEWER_ID,
        _OLDER_ID,
    ]
    assert [row["status"] for row in jobs] == [
        "running",
        "queued",
        "queued",
        "failed",
        "completed",
    ]
    assert [row["position"] for row in jobs] == [None, 1, 2, None, None]
    assert [row["priority"] for row in jobs if row["status"] == "queued"] == [
        _AHEAD_PRIORITY,
        _BEHIND_PRIORITY,
    ]
    assert [row["finished_at"] for row in jobs if row["status"] in {"completed", "failed"}] == [
        "2026-10-04T18:56:00Z",
        "2026-10-04T18:50:00Z",
    ]
    assert all(set(row) == _ROW_KEYS for row in jobs)


def _seed_finished(tmp_path: Path) -> None:
    """
    Store the completed pin and the newer failed pin, plus the completed pin's step.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: None. Example: None
    """
    _seed_pins(tmp_path, (_older_pin(), _newer_pin()))
    _insert_step(tmp_path, _OLDER_ID)
    _write_marker(tmp_path, _OLDER_ID)
    _write_marker(tmp_path, _NEWER_ID)


def _seed_pins(tmp_path: Path, pins: tuple[TaskPin, ...]) -> None:
    """
    Migrate the server's SQLite file and insert pins through Blackboard.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param pins: rows as tuple[TaskPin, ...]. Example: ()
    :return: None. Example: None
    """
    database = Database(_db_path(tmp_path))
    connection = database.connect()
    try:
        database.migrate()
        board = Blackboard(database)
        for pin in pins:
            board.insert_pin(connection, pin)
    finally:
        connection.close()


def _older_pin() -> TaskPin:
    """
    Build the completed pin that finished before the failed one.

    :return: pin as TaskPin. Example: a pin with status "completed"
    """
    return _finished_pin(
        _OLDER_ID,
        "completed",
        "2026-10-04T18:48:00Z",
        "2026-10-04T18:50:00Z",
        _OLDER_GOAL,
        "",
    )


def _newer_pin() -> TaskPin:
    """
    Build the failed pin that finished last, even though it was created earlier.

    :return: pin as TaskPin. Example: a pin with status "failed"
    """
    return _finished_pin(
        _NEWER_ID,
        "failed",
        "2026-10-04T18:41:00Z",
        "2026-10-04T18:56:00Z",
        _NEWER_GOAL,
        "missing chapter",
    )


def _finished_pin(
    task_id: str,
    status: str,
    created_at: str,
    finished_at: str,
    goal: str,
    error: str,
) -> TaskPin:
    """
    Build one finished pin that still has reasoning, an artifact, and a grant.

    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :param status: completed or failed as str. Example: "completed"
    :param created_at: enqueue time as str. Example: "2026-10-04T18:48:00Z"
    :param finished_at: stop time as str. Example: "2026-10-04T18:50:00Z"
    :param goal: worker goal as str. Example: "Summarize the older chapter into summary.md."
    :param error: failure summary as str. Example: ""
    :return: pin as TaskPin. Example: a pin with status "completed"
    """
    return TaskPin(
        task_id=task_id,
        priority=40,
        status=status,
        model="qwen-14b",
        goal=goal,
        done_when="summary.md exists",
        access=AccessGrant("paths", ("/tmp/books",)),
        phase=status,
        current_plan="1. Read the chapter (completed).",
        reasoning=(ReasoningNote("2026-10-04T18:50:01Z", 1, _NOTE),),
        artifacts=("summary.md",),
        error=error,
        created_at=created_at,
        started_at="2026-10-04T18:40:02Z",
        finished_at=finished_at,
        final_status=status,
    )


def _insert_step(tmp_path: Path, task_id: str) -> None:
    """
    Insert one pending step into the server's SQLite file.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :return: None. Example: None
    """
    connection = sqlite3.connect(_db_path(tmp_path), timeout=5.0)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO plan_steps (task_id, step_order, action, status) VALUES (?, 1, ?, 'pending')",
            (task_id, _STEP_ACTION),
        )
        connection.commit()
    finally:
        connection.close()


def _step_count(tmp_path: Path, task_id: str) -> int:
    """
    Count plan steps for one pin in the server's SQLite file.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :return: count as int. Example: 0
    """
    connection = sqlite3.connect(_db_path(tmp_path), timeout=5.0)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM plan_steps WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _task_exists(tmp_path: Path, task_id: str) -> bool:
    """
    True when the pin row is still in the server's SQLite file.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :return: whether the row exists as bool. Example: True
    """
    connection = sqlite3.connect(_db_path(tmp_path), timeout=5.0)
    try:
        row = connection.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone()
    finally:
        connection.close()
    return row is not None


def _write_marker(tmp_path: Path, task_id: str) -> Path:
    """
    Write summary.md in the job directory and return that file.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :return: marker file as Path. Example: Path("/tmp/pytest/test0/jobs/id/summary.md")
    """
    job = _job(tmp_path, task_id)
    job.mkdir(parents=True, exist_ok=True)
    marker = job / "summary.md"
    marker.write_text("keep", encoding="utf-8")
    return marker


def _replace_job_with_symlink(job: Path, target: Path) -> None:
    """
    Replace a real job directory with a symlink to a directory outside the job root.

    :param job: job path as Path. Example: Path("/tmp/pytest/test0/jobs/task")
    :param target: directory outside the job root as Path. Example: Path("/tmp/pytest/test0/outside")
    :return: None. Example: None
    """
    if job.is_symlink():
        job.unlink()
    elif job.exists():
        shutil.rmtree(job)
    job.symlink_to(target, target_is_directory=True)


def _job(tmp_path: Path, task_id: str) -> Path:
    """
    Return the job directory the dispatcher uses for one pin.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :return: job directory as Path. Example: Path("/tmp/pytest/test0/jobs/task")
    """
    return tmp_path / "jobs" / task_id


def _db_path(tmp_path: Path) -> Path:
    """
    Return the SQLite file the spawned server opens.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: database path as Path. Example: Path("/tmp/pytest/test0/local-ai.sqlite")
    """
    return tmp_path / "local-ai.sqlite"


def _start_hold() -> tuple[str, int, ThreadingHTTPServer, threading.Event]:
    """
    Bind a stand-in Ollama that holds chat posts.

    :return: host, port, server, and release as tuple. Example: ("127.0.0.1", 9, server, event)
    """
    release = threading.Event()
    _HoldHandler.release = release
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HoldHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return str(host), int(port), server, release


def _dispatcher_env(tmp_path: Path) -> dict[str, str]:
    """
    Environment for one spawned dispatcher. HTTP mode stays off.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: environment as dict[str, str]. Example: {"LOCAL_AI_DB": "/tmp/local-ai.sqlite"}
    """
    return {
        "PATH": os.environ.get("PATH", ""),
        "LOCAL_AI_DB": str(_db_path(tmp_path)),
        "LOCAL_AI_JOBS": str(tmp_path / "jobs"),
    }


def _spawn(tmp_path: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    """
    Start python -m local_ai on stdio.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param env: process environment as dict[str, str]. Example: {"LOCAL_AI_DB": "/tmp/db"}
    :return: process as subprocess.Popen[str]. Example: a running dispatcher
    """
    return subprocess.Popen(
        [str(_PYTHON), "-m", "local_ai"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        text=True,
    )


def _submit_arguments(priority: int, goal: str) -> dict:
    """
    The submit a person would send for one operator pin.

    :param priority: queue rank as int. Example: 40
    :param goal: worker goal as str. Example: "Hold the worker until the operator is done."
    :return: arguments as dict. Example: {"goal": "Hold the worker until the operator is done.", "priority": 40}
    """
    return {
        "goal": goal,
        "done_when": "summary.md exists",
        "priority": priority,
        "model": "qwen-14b",
        "access": {"mode": "job"},
    }


def _initialize() -> dict:
    """
    MCP initialize request.

    :return: JSON-RPC request as dict. Example: {"jsonrpc": "2.0", "method": "initialize"}
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "job-control-acceptance", "version": "0"},
        },
    }


def _initialized() -> dict:
    """
    Notification that finishes the handshake.

    :return: JSON-RPC notification as dict. Example: {"jsonrpc": "2.0", "method": "notifications/initialized"}
    """
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _tool_body(result: dict) -> dict:
    """
    Unwrap a tools/call result into the dict the tool returned.

    :param result: tools/call result as dict. Example: {"structuredContent": {"status": "ok"}}
    :return: tool body as dict. Example: {"status": "ok"}
    :raises AssertionError: when the result has no JSON body. Example: {}
    """
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return json.loads(content[0]["text"])
    raise AssertionError(f"tool result was {result!r}")


def _read_line(process: subprocess.Popen[str]) -> str:
    """
    Read one stdout line or fail when the process stays silent.

    :param process: dispatcher as subprocess.Popen[str]. Example: the process from _spawn
    :return: one line as str. Example: '{"jsonrpc":"2.0","id":1}\\n'
    :raises AssertionError: when stdout stays empty. Example: a hung process
    :raises RuntimeError: when stdout was not piped. Example: a process without a pipe
    """
    if process.stdout is None:
        raise RuntimeError("stdout was not piped")
    ready, _, _ = select.select([process.stdout], [], [], _READ_TIMEOUT_S)
    if not ready:
        raise AssertionError(_silence_error(process))
    line = process.stdout.readline()
    if line == "":
        raise AssertionError(_silence_error(process))
    return line


def _jsonrpc_line(line: str) -> dict:
    """
    Parse one stdout line. A log line is a failure.

    :param line: stdout line as str. Example: '{"jsonrpc":"2.0","id":1}\\n'
    :return: message as dict. Example: {"jsonrpc": "2.0", "id": 1}
    :raises AssertionError: when the line is not JSON-RPC. Example: "INFO task_id="
    """
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"stdout was not JSON: {line!r}") from exc
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise AssertionError(f"stdout was not a JSON-RPC message: {line!r}")
    return message


def _assert_stdout_is_only_jsonrpc(stdout: list[str]) -> None:
    """
    Every captured line is JSON-RPC.

    :param stdout: captured lines as list[str]. Example: ['{"jsonrpc":"2.0"}\\n']
    :return: None. Example: None
    """
    assert stdout, "stdout was empty"
    for line in stdout:
        _jsonrpc_line(line)


def _left_queue(last: dict, process: subprocess.Popen[str]) -> str:
    """
    Explain a pin that did not stay in planning.

    :param last: last read body as dict. Example: {"status": "failed", "error": "empty_plan"}
    :param process: dispatcher as subprocess.Popen[str]. Example: the process from _spawn
    :return: message as str. Example: "pin left the queue as failed"
    """
    return f"pin left the queue as {last.get('status')}: {last.get('error')} stderr={_stderr_tail(process)!r}"


def _silence_error(process: subprocess.Popen[str]) -> str:
    """
    Explain a hung read using whatever stderr has arrived.

    :param process: dispatcher as subprocess.Popen[str]. Example: the process from _spawn
    :return: message as str. Example: "no stdout line. stderr=''"
    """
    return f"no stdout line. stderr={_stderr_tail(process)!r}"


def _stderr_tail(process: subprocess.Popen[str]) -> str:
    """
    Read stderr that is already buffered. A live process is not drained to EOF.

    :param process: dispatcher as subprocess.Popen[str]. Example: the process from _spawn
    :return: tail as str. Example: ""
    """
    stream = process.stderr
    if stream is None:
        return ""
    os.set_blocking(stream.fileno(), False)
    parts: list[bytes] = []
    while True:
        try:
            part = os.read(stream.fileno(), 4096)
        except BlockingIOError:
            break
        if not part:
            break
        parts.append(part)
    return b"".join(parts).decode("utf-8", errors="replace")[-2000:]


def _stop(process: subprocess.Popen[str]) -> None:
    """
    Stop the dispatcher.

    :param process: dispatcher as subprocess.Popen[str]. Example: the process from _spawn
    :return: None. Example: None
    """
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
