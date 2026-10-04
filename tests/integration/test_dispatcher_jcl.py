"""Dispatcher list, cancel, and purge on a real sqlite file and job root."""

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import AccessGrant, TaskPin
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.plan.steps import PlanStore
from tests.support.log_story import assert_log_story


_RUNNING_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_QUEUED_ID = "11111111-2222-4333-8444-555555555555"
_DONE_ID = "22222222-3333-4444-8555-666666666666"
_FAILED_ID = "33333333-4444-4555-8666-777777777777"
_CONTRACTS = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
_SUMMARY_GOAL = "Summarize the book into 2-3 pages, one thesis per chapter."
_BOOK_GOAL = "Combine chapter markdown into book.md."


def test_list_jobs_with_running_and_queued_matches_the_contract(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """list returns the section 18 shape: running first, then the queue, no reasoning."""
    service, connection, blackboard = _open(tmp_path)
    _seed_running_and_queued(connection, blackboard)
    caplog.set_level(logging.INFO)

    body = service.list_jobs(None, connection)

    assert body == _contract("list_all.json")
    assert service.list_jobs("all", connection) == body
    assert_log_story(caplog.records, "dispatcher.list", "ok", ["filter=all", "count=2"])


def test_list_jobs_with_unknown_status_rejects_and_leaves_pins(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown filter is rejected and does not drop pins."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(_QUEUED_ID, status="queued", phase="queued"))
    caplog.set_level(logging.INFO)

    body = service.list_jobs("done", connection)

    assert body == _contract("list_rejected.json")
    assert blackboard.get_pin(connection, _QUEUED_ID) is not None
    assert_log_story(
        caplog.records,
        "dispatcher.list",
        "rejected",
        ["reason=bad_status", "filter=done"],
    )


def test_cancel_queued_pin_removes_the_row_steps_and_job_directory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """cancel deletes a queued pin and its folder. The running pin stays."""
    service, connection, blackboard = _open(tmp_path)
    _seed_running_and_queued(connection, blackboard)
    _insert_step(connection, _QUEUED_ID)
    job = _job(tmp_path, _QUEUED_ID)
    job.mkdir(parents=True)
    (job / "notes.md").write_text("keep", encoding="utf-8")
    caplog.set_level(logging.INFO)

    body = service.cancel(_QUEUED_ID, connection)

    assert body == _contract("cancel_cancelled.json")
    assert blackboard.get_pin(connection, _QUEUED_ID) is None
    assert _step_count(connection, _QUEUED_ID) == 0
    assert not job.exists()
    assert blackboard.get_pin(connection, _RUNNING_ID) is not None
    assert_log_story(
        caplog.records,
        "dispatcher.cancel",
        "cancelled",
        [f"task_id={_QUEUED_ID}", "job_dir_removed=True"],
    )


def test_cancel_running_pin_leaves_the_row_and_the_folder(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A running pin is rejected and its job folder stays."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _running_pin())
    job = _job(tmp_path, _RUNNING_ID)
    job.mkdir(parents=True)
    (job / "summary.md").write_text("live", encoding="utf-8")
    caplog.set_level(logging.INFO)

    body = service.cancel(_RUNNING_ID, connection)

    assert body == _contract("cancel_rejected.json")
    assert blackboard.get_pin(connection, _RUNNING_ID) is not None
    assert (job / "summary.md").read_text(encoding="utf-8") == "live"
    assert_log_story(
        caplog.records,
        "dispatcher.cancel",
        "not_queued",
        [f"task_id={_RUNNING_ID}", "status=running"],
    )


def test_cancel_unlinks_a_symlink_and_leaves_its_target(tmp_path: Path) -> None:
    """A job path that is a symlink is removed. The directory it points at stays."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(_QUEUED_ID, status="queued", phase="queued", priority=30))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    link = _job(tmp_path, _QUEUED_ID)
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    body = service.cancel(_QUEUED_ID, connection)

    assert body["status"] == "cancelled"
    assert body["job_dir_removed"] is True
    assert not link.exists()
    assert not link.is_symlink()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_purge_completed_pin_matches_the_contract(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """purge of one completed pin removes that pin and its folder."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _finished_pin(_DONE_ID, "completed"))
    blackboard.insert_pin(connection, _pin(_QUEUED_ID, status="queued", phase="queued"))
    job = _job(tmp_path, _DONE_ID)
    job.mkdir(parents=True)
    (job / "summary.md").write_text("done", encoding="utf-8")
    caplog.set_level(logging.INFO)

    body = service.purge(_DONE_ID, "completed", connection)

    assert body == _contract("purge_ok.json")
    assert blackboard.get_pin(connection, _DONE_ID) is None
    assert not job.exists()
    assert blackboard.get_pin(connection, _QUEUED_ID) is not None
    assert_log_story(caplog.records, "dispatcher.purge", "ok", ["count=1"])


def test_purge_running_pin_returns_not_terminal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A running pin is rejected and its folder stays."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _running_pin())
    job = _job(tmp_path, _RUNNING_ID)
    job.mkdir(parents=True)
    caplog.set_level(logging.INFO)

    body = service.purge(_RUNNING_ID, None, connection)

    assert body == _contract("purge_not_terminal.json")
    assert blackboard.get_pin(connection, _RUNNING_ID) is not None
    assert job.is_dir()
    assert_log_story(
        caplog.records,
        "dispatcher.purge",
        "not_terminal",
        [f"task_id={_RUNNING_ID}"],
    )


def test_purge_with_the_wrong_finished_status_leaves_the_pin(tmp_path: Path) -> None:
    """A completed pin is not deleted when the caller asked for failed."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _finished_pin(_DONE_ID, "completed"))

    body = service.purge(_DONE_ID, "failed", connection)

    assert body == _contract("purge_not_status.json")
    assert blackboard.get_pin(connection, _DONE_ID) is not None


def test_purge_without_a_task_id_removes_finished_pins_and_leaves_live_ones(tmp_path: Path) -> None:
    """An open purge deletes completed and failed pins and leaves queued and running."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _finished_pin(_DONE_ID, "completed"))
    blackboard.insert_pin(connection, _finished_pin(_FAILED_ID, "failed", finished_at="2026-10-04T18:56:00Z"))
    blackboard.insert_pin(connection, _running_pin())
    blackboard.insert_pin(connection, _pin(_QUEUED_ID, status="queued", phase="queued"))
    _job(tmp_path, _DONE_ID).mkdir(parents=True)
    _job(tmp_path, _FAILED_ID).mkdir(parents=True)

    body = service.purge(None, None, connection)

    assert body["status"] == "ok"
    assert body["count"] == 2
    assert {row["task_id"] for row in body["purged"]} == {_DONE_ID, _FAILED_ID}
    assert blackboard.get_pin(connection, _RUNNING_ID) is not None
    assert blackboard.get_pin(connection, _QUEUED_ID) is not None
    assert not _job(tmp_path, _DONE_ID).exists()
    assert not _job(tmp_path, _FAILED_ID).exists()


def test_purge_with_status_all_deletes_nothing(tmp_path: Path) -> None:
    """all is a list filter, not a purge filter."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _finished_pin(_DONE_ID, "completed"))

    body = service.purge(None, "all", connection)

    assert body == {"status": "rejected", "reason": "bad_status", "filter": "all"}
    assert blackboard.get_pin(connection, _DONE_ID) is not None


def test_purge_with_no_finished_pins_returns_an_empty_list(tmp_path: Path) -> None:
    """An empty match is success with count 0."""
    service, connection, blackboard = _open(tmp_path)
    blackboard.insert_pin(connection, _pin(_QUEUED_ID, status="queued", phase="queued"))

    body = service.purge(None, None, connection)

    assert body == {"status": "ok", "count": 0, "purged": []}
    assert blackboard.get_pin(connection, _QUEUED_ID) is not None


def _open(tmp_path: Path) -> tuple[DispatcherService, sqlite3.Connection, Blackboard]:
    """
    Open a migrated sqlite file and the dispatcher around it.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: service, connection, and blackboard. Example: (DispatcherService(...), connection, Blackboard(...))
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    blackboard = Blackboard(database)
    service = DispatcherService(
        database,
        blackboard,
        PlanStore(database),
        CapacityReader(),
        tmp_path / "jobs",
    )
    return service, connection, blackboard


def _seed_running_and_queued(connection: sqlite3.Connection, blackboard: Blackboard) -> None:
    """
    Store the section 18 list example: one running pin and one queued pin.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param blackboard: pin store as Blackboard. Example: Blackboard(database)
    :return: None. Example: None
    """
    blackboard.insert_pin(connection, _running_pin())
    blackboard.insert_pin(
        connection,
        _pin(
            _QUEUED_ID,
            status="queued",
            phase="queued",
            priority=30,
            model="qwen-24b",
            goal=_BOOK_GOAL,
            created_at="2026-10-04T18:41:00Z",
        ),
    )


def _running_pin() -> TaskPin:
    """
    Build the running pin from the list contract.

    :return: pin as TaskPin. Example: a pin with status "running"
    """
    return _pin(
        _RUNNING_ID,
        status="running",
        phase="acting",
        priority=40,
        model="qwen-14b",
        goal=_SUMMARY_GOAL,
        created_at="2026-10-04T18:40:00Z",
        started_at="2026-10-04T18:40:02Z",
    )


def _finished_pin(task_id: str, status: str, finished_at: str = "2026-10-04T18:55:00Z") -> TaskPin:
    """
    Build a completed or failed pin.

    :param task_id: pin id as str. Example: "22222222-3333-4444-8555-666666666666"
    :param status: completed or failed as str. Example: "completed"
    :param finished_at: stop time as str. Example: "2026-10-04T18:55:00Z"
    :return: pin as TaskPin. Example: a pin with status "completed"
    """
    return _pin(
        task_id,
        status=status,
        phase=status,
        finished_at=finished_at,
        goal=_SUMMARY_GOAL,
    )


def _pin(
    task_id: str,
    *,
    status: str,
    phase: str,
    priority: int = 40,
    model: str = "qwen-14b",
    goal: str = _SUMMARY_GOAL,
    created_at: str = "2026-10-04T18:40:00Z",
    started_at: str | None = None,
    finished_at: str | None = None,
) -> TaskPin:
    """
    Build one pin row.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param status: pin status as str. Example: "queued"
    :param phase: phase word as str. Example: "queued"
    :param priority: queue rank as int. Example: 40
    :param model: chosen size as str. Example: "qwen-14b"
    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
    :param created_at: enqueue time as str. Example: "2026-10-04T18:40:00Z"
    :param started_at: dispatch time as str | None. Example: None
    :param finished_at: stop time as str | None. Example: None
    :return: pin as TaskPin. Example: a pin with status "queued"
    """
    return TaskPin(
        task_id=task_id,
        priority=priority,
        status=status,
        model=model,
        goal=goal,
        done_when="summary.md exists",
        access=AccessGrant("job"),
        phase=phase,
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
    )


def _insert_step(connection: sqlite3.Connection, task_id: str) -> None:
    """
    Insert one pending step under a pin.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "11111111-2222-4333-8444-555555555555"
    :return: None. Example: None
    """
    connection.execute(
        "INSERT INTO plan_steps (task_id, step_order, action, status) VALUES (?, 1, ?, 'pending')",
        (task_id, "List chapter files"),
    )
    connection.commit()


def _step_count(connection: sqlite3.Connection, task_id: str) -> int:
    """
    Count plan steps for one pin.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "11111111-2222-4333-8444-555555555555"
    :return: count as int. Example: 0
    """
    row = connection.execute("SELECT COUNT(*) FROM plan_steps WHERE task_id = ?", (task_id,)).fetchone()
    assert row is not None
    return int(row[0])


def _job(tmp_path: Path, task_id: str) -> Path:
    """
    Return the job directory the dispatcher would use.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :param task_id: pin id as str. Example: "11111111-2222-4333-8444-555555555555"
    :return: job directory as Path. Example: Path("/tmp/pytest/jobs/11111111-2222-4333-8444-555555555555")
    """
    return tmp_path / "jobs" / task_id


def _contract(name: str) -> dict:
    """
    Load one section 18 example.

    :param name: fixture file as str. Example: "list_all.json"
    :return: example as dict. Example: {"status": "ok", "count": 0, "jobs": []}
    """
    return json.loads((_CONTRACTS / name).read_text(encoding="utf-8"))
