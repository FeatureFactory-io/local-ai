"""Slot seam between DispatcherService and Worker on a real sqlite file."""

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import Completion, SubmitRequest, ToolCall
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.llm.port import ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker
from tests.support.log_story import assert_log_story


_STARTED_AT = "2026-10-04T18:40:02Z"
_CREATED_AT = "2026-10-04T18:40:00Z"
_FINISHED_AT = "2026-10-04T18:55:00Z"
_JOIN_TIMEOUT_S = 5
_PAUSE_S = "0.6"
_COMPLETE_TURN = (
    Completion(content="1. Finish the brief"),
    Completion(final_answer="The brief is done"),
)
_PAUSE_TURN = (
    Completion(content="1. Pause in the job folder"),
    Completion(
        tool_calls=(ToolCall("shell", f"sleep {_PAUSE_S}"),),
        final_answer="Paused",
    ),
)


def test_kick_with_an_empty_queue_returns_without_a_thread(tmp_path: Path) -> None:
    """An empty queue leaves the slot idle."""
    _service, connection, slot = _open(tmp_path)

    slot.kick()

    assert slot._thread is None
    assert _task_count(connection) == 0


def test_kick_after_submit_claims_the_pin_on_a_daemon_thread(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Submit's queued snapshot stays queued. The following read shows running."""
    service, connection, slot = _open(tmp_path, script=_PAUSE_TURN, slot_clock=lambda: _STARTED_AT)
    caplog.set_level(logging.INFO)
    snapshot = _submit(service, connection, "Write a brief.")

    slot.kick()

    assert snapshot["status"] == "queued"
    body = service.read_pin(str(snapshot["task_id"]), connection)
    thread = slot._thread
    assert body["status"] == "running"
    assert thread is not None
    assert thread.daemon is True
    assert_log_story(
        caplog.records,
        "dispatcher.slot",
        "kick",
        [f"task_id={snapshot['task_id']}"],
    )
    assert thread.join(timeout=_JOIN_TIMEOUT_S) is None
    assert not thread.is_alive()


def test_kick_with_a_live_thread_leaves_both_pins_queued(tmp_path: Path) -> None:
    """A live slot thread returns before claim, so the queued pins stay queued."""
    service, connection, slot = _open(tmp_path)
    first = _submit(service, connection, "Write the first brief.")
    second = _submit(service, connection, "Write the second brief.")
    release = threading.Event()
    thread = threading.Thread(target=release.wait, kwargs={"timeout": 5}, daemon=True)
    thread.start()
    slot._thread = thread

    slot.kick()

    assert service.read_pin(str(first["task_id"]), connection)["status"] == "queued"
    assert service.read_pin(str(second["task_id"]), connection)["status"] == "queued"
    release.set()
    thread.join(timeout=2)


def test_run_finishes_a_completed_pin_and_kicks_the_next(tmp_path: Path) -> None:
    """A completed run_task writes the clock time, then the next pin runs."""
    service, connection, slot = _open(
        tmp_path,
        script=_COMPLETE_TURN + _COMPLETE_TURN,
        slot_clock=lambda: _FINISHED_AT,
    )
    first = _submit(service, connection, "Write the first brief.")
    second = _submit(service, connection, "Write the second brief.")

    slot.kick()
    _join_slot(slot)

    first_body = service.read_pin(str(first["task_id"]), connection)
    second_body = service.read_pin(str(second["task_id"]), connection)
    assert first_body["status"] == "completed"
    assert first_body["error"] == ""
    assert first_body["artifacts"] == []
    assert second_body["status"] == "completed"
    assert second_body["error"] == ""
    assert _finished_at(connection, str(first["task_id"])) == _FINISHED_AT
    assert _finished_at(connection, str(second["task_id"])) == _FINISHED_AT
    assert slot._thread is None


def test_run_with_a_worker_exception_finishes_failed(tmp_path: Path) -> None:
    """An exception from the worker still finishes the pin with that text."""
    service, connection, slot = _open(tmp_path, script=(), slot_clock=lambda: _FINISHED_AT)
    snapshot = _submit(service, connection, "Write a brief.")

    slot.kick()
    _join_slot(slot)

    body = service.read_pin(str(snapshot["task_id"]), connection)
    assert body["status"] == "failed"
    assert body["error"] == "script has no turn left at index=0 model_name=qwen-14b"
    assert body["artifacts"] == []
    assert _finished_at(connection, str(snapshot["task_id"])) == _FINISHED_AT
    assert slot._thread is None


def _open(
    tmp_path: Path,
    script: tuple[Completion, ...] = (),
    slot_clock: Callable[[], str] | None = None,
    service_clock: Callable[[], str] | None = None,
) -> tuple[DispatcherService, sqlite3.Connection, Slot]:
    """
    Open a migrated database, a real worker, and a slot.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param script: model turns as tuple[Completion, ...]. Example: ()
    :param slot_clock: dispatch clock as Callable[[], str] | None. Example: lambda: "2026-10-04T18:40:02Z"
    :param service_clock: submit clock as Callable[[], str] | None. Example: lambda: "2026-10-04T18:40:00Z"
    :return: service, connection, and slot. Example: (DispatcherService(...), connection, Slot(...))
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    board = Blackboard(database)
    plan = PlanStore(database)
    service = DispatcherService(
        database,
        board,
        plan,
        CapacityReader(),
        tmp_path / "jobs",
        memory=lambda: (64.0, 40.2),
        ollama=_ollama_up,
        pressure=lambda: "normal",
        ollama_host="http://127.0.0.1:11434",
        clock=service_clock if service_clock is not None else (lambda: _CREATED_AT),
    )
    worker = Worker(
        ScriptedLLM(script),
        ShellTool(GrantChecker()),
        PromptAssembler(),
        plan,
        board,
        connection,
    )
    slot = Slot(
        service,
        worker,
        connection,
        slot_clock if slot_clock is not None else (lambda: _STARTED_AT),
    )
    return service, connection, slot


def _submit(service: DispatcherService, connection: sqlite3.Connection, goal: str) -> dict:
    """
    Queue one pin. The caller kicks after this snapshot.

    :param service: dispatcher as DispatcherService. Example: the service from _open
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param goal: worker goal as str. Example: "Write a brief."
    :return: queued snapshot as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    """
    return service.submit(
        SubmitRequest(goal, "brief.md exists", priority=40, model="qwen-14b"),
        connection,
    )


def _ollama_up(url: str) -> tuple[int, dict]:
    """
    Report Ollama up with one loaded model.

    :param url: request URL as str. Example: "http://127.0.0.1:11434/api/tags"
    :return: status and body as tuple[int, dict]. Example: (200, {"models": []})
    :raises AssertionError: when the path is not tags or ps. Example: "/api/version"
    """
    if url.endswith("/api/tags"):
        return 200, {"models": []}
    if url.endswith("/api/ps"):
        return 200, {"models": [{"name": "qwen2.5:14b"}]}
    raise AssertionError(url)


def _join_slot(slot: Slot, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """
    Wait until the slot thread and any pin it kicks have stopped.

    :param slot: slot under test as Slot. Example: the slot from _open
    :param timeout: seconds to wait as float. Example: 5
    :return: None. Example: None
    :raises AssertionError: when a worker is still alive at the deadline. Example: a stuck sleep
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = slot._thread
        if thread is None:
            return
        remaining = deadline - time.monotonic()
        thread.join(timeout=remaining)
        if thread.is_alive():
            break
        if slot._thread is None or slot._thread is thread:
            return
    raise AssertionError("slot thread did not finish")


def _finished_at(connection: sqlite3.Connection, task_id: str) -> str | None:
    """
    Read the pin's finished_at column.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: timestamp or None as str | None. Example: "2026-10-04T18:55:00Z"
    """
    row = connection.execute("SELECT finished_at FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    value = row["finished_at"]
    if value is None:
        return None
    return str(value)


def _task_count(connection: sqlite3.Connection) -> int:
    """
    Count tasks rows.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: row count as int. Example: 0
    """
    row = connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
    assert row is not None
    return int(row["n"])
