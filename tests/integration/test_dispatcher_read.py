"""Dispatcher read, reprioritize, recovery, and finish on a real sqlite file."""

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import AccessGrant, InvalidPriority, ReasoningNote, StepDraft, TaskPin
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.plan.steps import PlanStore
from tests.support.log_story import assert_log_story


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_OTHER_ID = "11111111-2222-4333-8444-555555555555"
_GOAL = "Summarize the book into 2-3 pages, one thesis per chapter."
_PLAN = (
    "1. List chapter files (completed). "
    "2. Write a thesis per chapter (running). "
    "3. Assemble summary.md (pending)."
)
_NOTE = (
    "Reading ch-04.md. The claim is in the last section. "
    "I will append one paragraph to theses.md."
)
_DRAFT_PATH = "/Users/denispetelin/books/draft"
_CONTRACTS = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"


def test_read_pin_with_missing_id_returns_not_found(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing pin is rejected and does not invent a body."""
    service, connection, _blackboard, _plan = _open(tmp_path)
    caplog.set_level(logging.INFO)

    body = service.read_pin(_TASK_ID, connection)

    assert body == {"status": "rejected", "reason": "not_found", "task_id": _TASK_ID}
    assert _task_count(connection) == 0
    assert_log_story(
        caplog.records,
        "dispatcher.read",
        "entry",
        [f"task_id={_TASK_ID}", "status=rejected"],
    )


def test_read_pin_with_running_pin_matches_the_contract(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A live pin returns the section 18 running shape, including steps and reasoning."""
    service, connection, blackboard, plan = _open(tmp_path)
    _seed_running_contract(connection, blackboard, plan)
    caplog.set_level(logging.INFO)

    body = service.read_pin(_TASK_ID, connection)

    assert body == _contract("read_running.json")
    assert_log_story(
        caplog.records,
        "dispatcher.read",
        "entry",
        [f"task_id={_TASK_ID}", "status=running"],
    )


def test_read_pin_with_no_steps_returns_empty_steps_and_job_access(tmp_path: Path) -> None:
    """A pin with no plan still returns the pin keys and an empty step list."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(status="queued", phase="queued", access=AccessGrant("job")))

    body = service.read_pin(_TASK_ID, connection)

    assert body["status"] == "queued"
    assert body["access"] == {"mode": "job"}
    assert body["steps"] == []
    assert body["reasoning"] == []
    assert body["artifacts"] == []
    assert body["error"] == ""
    assert body["current_plan"] == ""


def test_read_pin_with_artifacts_returns_the_paths(tmp_path: Path) -> None:
    """Artifact paths on the pin are returned as a list of strings."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(
        connection,
        _pin(
            status="failed",
            phase="failed",
            artifacts=("/tmp/summary.md",),
            error="three identical shell failures",
        ),
    )

    body = service.read_pin(_TASK_ID, connection)

    assert body["artifacts"] == ["/tmp/summary.md"]
    assert body["error"] == "three identical shell failures"
    assert body["status"] == "failed"


def test_read_pin_with_blank_task_id_raises(tmp_path: Path) -> None:
    """A blank id is rejected before a pin body is built."""
    service, connection, _blackboard, _plan = _open(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        service.read_pin("  ", connection)


def test_reprioritize_with_queued_pin_returns_the_new_position(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A queued pin stores the new priority and reports its place."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(status="queued", phase="queued", priority=40))
    caplog.set_level(logging.INFO)

    body = service.reprioritize(_TASK_ID, 5, connection)

    assert body == _contract("reprioritize_queued.json")
    assert _priority(connection, _TASK_ID) == 5
    assert_log_story(
        caplog.records,
        "dispatcher.reprioritize",
        "queued",
        [f"task_id={_TASK_ID}", "priority=5", "position=1"],
    )


def test_reprioritize_with_a_pin_ahead_keeps_the_later_place(tmp_path: Path) -> None:
    """A lower number jumps the queue. An equal priority stays behind the earlier pin."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(
        connection,
        _pin(
            status="queued",
            phase="queued",
            priority=10,
            task_id=_OTHER_ID,
            created_at="2026-10-04T18:39:00Z",
            goal="Other.",
        ),
    )
    blackboard.insert_pin(
        connection,
        _pin(status="queued", phase="queued", priority=80, created_at="2026-10-04T18:41:00Z"),
    )

    behind = service.reprioritize(_TASK_ID, 20, connection)
    ahead = service.reprioritize(_TASK_ID, 5, connection)
    tied = service.reprioritize(_TASK_ID, 10, connection)

    assert behind == {"status": "queued", "task_id": _TASK_ID, "priority": 20, "position": 2}
    assert ahead == {"status": "queued", "task_id": _TASK_ID, "priority": 5, "position": 1}
    assert tied == {"status": "queued", "task_id": _TASK_ID, "priority": 10, "position": 2}
    assert _priority(connection, _OTHER_ID) == 10
    assert _priority(connection, _TASK_ID) == 10


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
def test_reprioritize_with_unqueued_pin_does_not_write(
    tmp_path: Path,
    status: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Running, completed, and failed pins stay at their old priority."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(status=status, phase=status, priority=40))
    caplog.set_level(logging.INFO)

    body = service.reprioritize(_TASK_ID, 5, connection)

    assert body == _contract("reprioritize_rejected.json")
    assert _priority(connection, _TASK_ID) == 40
    assert _status(connection, _TASK_ID) == status
    assert_log_story(
        caplog.records,
        "dispatcher.reprioritize",
        "not_queued",
        [f"task_id={_TASK_ID}", f"status={status}"],
    )


def test_reprioritize_with_missing_pin_returns_not_found(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing pin is rejected and nothing is inserted."""
    service, connection, _blackboard, _plan = _open(tmp_path)
    caplog.set_level(logging.INFO)

    body = service.reprioritize(_TASK_ID, 5, connection)

    assert body == {"status": "rejected", "reason": "not_found", "task_id": _TASK_ID}
    assert _task_count(connection) == 0
    assert_log_story(
        caplog.records,
        "dispatcher.reprioritize",
        "not_found",
        [f"task_id={_TASK_ID}"],
    )


@pytest.mark.parametrize("priority", [0, 101])
def test_reprioritize_with_priority_outside_range_does_not_write(
    tmp_path: Path,
    priority: int,
) -> None:
    """An illegal priority raises and leaves the queued pin unchanged."""
    service, connection, blackboard, _plan = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(status="queued", phase="queued", priority=40))

    with pytest.raises(InvalidPriority):
        service.reprioritize(_TASK_ID, priority, connection)

    assert _priority(connection, _TASK_ID) == 40


def _open(
    tmp_path: Path,
) -> tuple[DispatcherService, sqlite3.Connection, Blackboard, PlanStore]:
    """
    Open a migrated sqlite file and the dispatcher around it.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: service, connection, blackboard, and plan. Example: (DispatcherService(...), connection, Blackboard(...), PlanStore(...))
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    blackboard = Blackboard(database)
    plan = PlanStore(database)
    service = DispatcherService(
        database,
        blackboard,
        plan,
        CapacityReader(),
        tmp_path / "jobs",
    )
    return service, connection, blackboard, plan


def _seed_running_contract(
    connection: sqlite3.Connection,
    blackboard: Blackboard,
    plan: PlanStore,
) -> None:
    """
    Store the section 18 running pin and its three steps.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param blackboard: pin store as Blackboard. Example: Blackboard(database)
    :param plan: step store as PlanStore. Example: PlanStore(database)
    :return: None. Example: None
    """
    blackboard.insert_pin(
        connection,
        _pin(
            status="running",
            phase="acting",
            access=AccessGrant("paths", (_DRAFT_PATH,)),
            current_plan=_PLAN,
            reasoning=(ReasoningNote("2026-10-04T18:40:01Z", 2, _NOTE),),
            started_at="2026-10-04T18:40:02Z",
        ),
    )
    plan.insert_steps(
        connection,
        _TASK_ID,
        (
            StepDraft(1, "List chapter files"),
            StepDraft(2, "Write a thesis per chapter"),
            StepDraft(3, "Assemble summary.md"),
        ),
    )
    plan.mark_running(connection, _TASK_ID, 1)
    plan.complete_step(connection, _TASK_ID, 1, "Found ch-01.md through ch-12.md", "")
    plan.mark_running(connection, _TASK_ID, 2)


def _pin(
    *,
    status: str,
    phase: str,
    access: AccessGrant | None = None,
    current_plan: str = "",
    reasoning: tuple[ReasoningNote, ...] = (),
    artifacts: tuple[str, ...] = (),
    error: str = "",
    started_at: str | None = None,
    priority: int = 40,
    task_id: str = _TASK_ID,
    created_at: str = "2026-10-04T18:40:00Z",
    goal: str = _GOAL,
    model: str = "qwen-14b",
) -> TaskPin:
    """
    Build one pin row for the read tests.

    :param status: queued, running, completed, or failed as str. Example: "running"
    :param phase: phase word as str. Example: "acting"
    :param access: grant as AccessGrant | None. Example: AccessGrant("job")
    :param current_plan: readable plan as str. Example: ""
    :param reasoning: notes as tuple[ReasoningNote, ...]. Example: ()
    :param artifacts: output paths as tuple[str, ...]. Example: ()
    :param error: failure summary as str. Example: ""
    :param started_at: dispatch time as str | None. Example: None
    :param priority: queue rank as int. Example: 40
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param created_at: enqueue time as str. Example: "2026-10-04T18:40:00Z"
    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages, one thesis per chapter."
    :param model: chosen size as str. Example: "qwen-14b"
    :return: pin as TaskPin. Example: a pin with status "running"
    """
    return TaskPin(
        task_id=task_id,
        priority=priority,
        status=status,
        model=model,
        goal=goal,
        done_when="summary.md exists",
        access=AccessGrant("job") if access is None else access,
        phase=phase,
        current_plan=current_plan,
        reasoning=reasoning,
        artifacts=artifacts,
        error=error,
        created_at=created_at,
        started_at=started_at,
    )


def _contract(name: str) -> dict:
    """
    Load one section 18 JSON example.

    :param name: file name as str. Example: "read_running.json"
    :return: parsed object as dict. Example: {"status": "running"}
    """
    return json.loads((_CONTRACTS / name).read_text(encoding="utf-8"))


def _priority(connection: sqlite3.Connection, task_id: str) -> int:
    """
    Read one pin's priority.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: priority as int. Example: 40
    """
    return int(_task_column(connection, task_id, "priority"))


def _status(connection: sqlite3.Connection, task_id: str) -> str:
    """
    Read one pin's status.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: status as str. Example: "queued"
    """
    return str(_task_column(connection, task_id, "status"))


def _task_column(connection: sqlite3.Connection, task_id: str, column: str) -> object:
    """
    Read one tasks column.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param column: column name as str. Example: "priority"
    :return: column value as object. Example: 40
    """
    row = connection.execute(f"SELECT {column} FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    return row[column]


def _task_count(connection: sqlite3.Connection) -> int:
    """
    Count tasks rows.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: row count as int. Example: 0
    """
    row = connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
    assert row is not None
    return int(row["n"])
