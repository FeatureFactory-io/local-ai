"""Plan H sequences across the dispatcher, the slot, and the shell."""

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

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


_CREATED_AT = "2026-10-04T18:40:00Z"
_STARTED_AT = "2026-10-04T18:40:02Z"
_JOIN_TIMEOUT_S = 5.0
_HOLD_COMMAND = "python3 -c \"open('awake.txt','w').write('1'); import time; time.sleep(2)\""
_HOLD_TURN = (
    Completion(content="1. Pause in the job folder"),
    Completion(tool_calls=(ToolCall("shell", _HOLD_COMMAND),), final_answer="Paused"),
)


def test_second_submit_stays_queued_while_the_first_run_is_in_flight(tmp_path: Path) -> None:
    """A second submit during Worker.run_task stays queued and leaves the running pin unchanged."""
    service, connection, slot = _open(tmp_path, script=_HOLD_TURN)
    first = _submit(service, connection, "Summarize the book into 2-3 pages.", priority=40)
    slot.kick()
    _wait_for_hold(tmp_path, str(first["task_id"]))
    running_before = _task_row(connection, str(first["task_id"]))

    second = _submit(service, connection, "Combine chapter markdown into book.md.", priority=30)
    slot.kick()

    assert second["status"] == "queued"
    assert service.read_pin(str(second["task_id"]), connection)["status"] == "queued"
    assert running_before["status"] == "running"
    assert _task_row(connection, str(first["task_id"])) == running_before
    _join_slot(slot)


def _open(
    tmp_path: Path,
    script: tuple[Completion, ...],
    service_clock: Callable[[], str] | None = None,
) -> tuple[DispatcherService, sqlite3.Connection, Slot]:
    """
    Open a real sqlite file, a real shell, and a scripted worker behind the slot.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param script: model turns as tuple[Completion, ...]. Example: ()
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
    slot = Slot(service, worker, connection, lambda: _STARTED_AT)
    return service, connection, slot


def _submit(
    service: DispatcherService,
    connection: sqlite3.Connection,
    goal: str,
    priority: int,
) -> dict:
    """
    Queue one pin. The caller kicks after this snapshot.

    :param service: dispatcher as DispatcherService. Example: the service from _open
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
    :param priority: queue rank as int. Example: 40
    :return: queued snapshot as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    """
    return service.submit(
        SubmitRequest(goal, "brief.md exists", priority=priority, model="qwen-14b"),
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


def _wait_for_hold(tmp_path: Path, task_id: str, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """
    Wait until the in-flight shell has created awake.txt and is still sleeping.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: running pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param timeout: seconds to wait as float. Example: 5.0
    :return: None. Example: None
    :raises AssertionError: when the marker does not appear. Example: a denied shell
    """
    marker = tmp_path / "jobs" / task_id / "awake.txt"
    deadline = time.monotonic() + timeout
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError(f"hold marker missing path={marker}")
        time.sleep(0.01)


def _join_slot(slot: Slot, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """
    Wait until the slot thread and any pin it kicks have stopped.

    :param slot: slot under test as Slot. Example: the slot from _open
    :param timeout: seconds to wait as float. Example: 5.0
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


def _task_row(connection: sqlite3.Connection, task_id: str) -> dict[str, object]:
    """
    Read one tasks row as a plain dict so a later read can be compared.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: column values as dict[str, object]. Example: {"status": "running", "priority": 40}
    """
    row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    return {key: row[key] for key in row.keys()}
