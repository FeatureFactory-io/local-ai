"""Blackboard pin rows on a real sqlite file."""

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import AccessGrant, ReasoningNote, TaskPin
from local_ai.db.database import Database


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_CREATED_AT = "2026-10-04T18:40:00Z"


def test_insert_pin_stores_access_json_and_empty_reasoning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A queued pin is one tasks row: access is JSON and reasoning is []."""
    board, connection = _open_board(tmp_path)
    pin = _pin(access=AccessGrant("paths", ("/Users/denispetelin/books/draft",)))
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    board.insert_pin(connection, pin)

    row = _task_row(connection, _TASK_ID)
    assert row["priority"] == 40
    assert row["status"] == "queued"
    assert row["model"] == "qwen-14b"
    assert row["goal"] == "Summarize the book into 2-3 pages."
    assert row["done_when"] == "summary.md exists"
    assert row["phase"] == "queued"
    assert row["current_plan"] == ""
    assert row["reasoning"] == "[]"
    assert row["error"] == ""
    assert row["created_at"] == _CREATED_AT
    assert row["started_at"] is None
    assert row["finished_at"] is None
    assert row["final_status"] == ""
    assert json.loads(row["access_json"]) == {
        "mode": "paths",
        "paths": ["/Users/denispetelin/books/draft"],
    }
    assert json.loads(row["artifacts_json"]) == []
    assert any(
        record.levelno == logging.INFO
        and "blackboard.insert" in record.message
        and f"task_id={_TASK_ID}" in record.message
        and "priority=40" in record.message
        and "model=qwen-14b" in record.message
        for record in caplog.records
    )


def test_insert_pin_is_visible_on_a_second_connection(tmp_path: Path) -> None:
    """The insert is committed, so another connection sees the row."""
    board, connection = _open_board(tmp_path)
    database_path = tmp_path / "local-ai.sqlite"

    board.insert_pin(connection, _pin())

    fresh = sqlite3.connect(database_path)
    fresh.row_factory = sqlite3.Row
    try:
        row = fresh.execute("SELECT id, access_json FROM tasks WHERE id = ?", (_TASK_ID,)).fetchone()
    finally:
        fresh.close()
    assert row["id"] == _TASK_ID
    assert json.loads(row["access_json"]) == {"mode": "job"}


def test_insert_pin_with_a_non_pin_raises(tmp_path: Path) -> None:
    """A mapping is not a pin and is rejected before a write."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(TypeError, match="TaskPin"):
        board.insert_pin(connection, {"task_id": _TASK_ID})  # type: ignore[arg-type]

    assert connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0


def test_insert_pin_with_a_blank_task_id_raises(tmp_path: Path) -> None:
    """An empty id cannot be the tasks primary key."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        board.insert_pin(connection, _pin(task_id="  "))


def test_get_pin_returns_the_stored_pin(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Access JSON and reasoning notes come back as the same pin."""
    board, connection = _open_board(tmp_path)
    pin = _pin(
        access=AccessGrant("paths", ("/Users/denispetelin/books/draft",)),
        current_plan="1. List chapter files (running).",
        reasoning=(ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md."),),
        artifacts=("/tmp/summary.md",),
    )
    board.insert_pin(connection, pin)
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    loaded = board.get_pin(connection, _TASK_ID)

    assert loaded == pin
    assert any(
        record.levelno == logging.INFO
        and f"task_id={_TASK_ID}" in record.message
        and "status=queued" in record.message
        for record in caplog.records
    )


def test_get_pin_with_unknown_id_returns_none(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A missing id is None. The caller decides not_found."""
    board, connection = _open_board(tmp_path)
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    assert board.get_pin(connection, _TASK_ID) is None
    assert any(
        record.levelno == logging.INFO
        and f"task_id={_TASK_ID}" in record.message
        and "status=missing" in record.message
        for record in caplog.records
    )


def test_get_pin_with_malformed_reasoning_raises(tmp_path: Path) -> None:
    """Stored reasoning that is not a JSON array fails instead of inventing notes."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    connection.execute("UPDATE tasks SET reasoning = ? WHERE id = ?", ("not-json", _TASK_ID))
    connection.commit()

    with pytest.raises(ValueError, match="reasoning"):
        board.get_pin(connection, _TASK_ID)


def test_append_reasoning_adds_notes_newest_last(tmp_path: Path) -> None:
    """Each note is stored as JSON, newest last."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    first = ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    second = ReasoningNote("2026-10-04T18:40:02Z", 2, "The claim is in the last section.")

    board.append_reasoning(connection, _TASK_ID, first)
    board.append_reasoning(connection, _TASK_ID, second)

    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.reasoning == (first, second)


def test_append_reasoning_with_empty_text_keeps_previous(tmp_path: Path) -> None:
    """None and blank text do not replace the stored array."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    kept = ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    board.append_reasoning(connection, _TASK_ID, kept)
    previous = _task_row(connection, _TASK_ID)["reasoning"]

    board.append_reasoning(connection, _TASK_ID, None)
    board.append_reasoning(connection, _TASK_ID, ReasoningNote("2026-10-04T18:40:02Z", 2, "  "))

    assert _task_row(connection, _TASK_ID)["reasoning"] == previous


def test_append_reasoning_with_unknown_id_raises(tmp_path: Path) -> None:
    """A missing pin is not created by an append."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        board.append_reasoning(
            connection,
            _TASK_ID,
            ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md."),
        )
    assert connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0


def test_claim_moves_a_queued_pin_to_running(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The first claim sets running and started_at. Phase stays as it was."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    assert board.claim(connection, _TASK_ID, "2026-10-04T18:40:02Z") is True

    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.status == "running"
    assert loaded.started_at == "2026-10-04T18:40:02Z"
    assert loaded.phase == "queued"
    assert any(
        record.levelno == logging.INFO
        and f"task_id={_TASK_ID}" in record.message
        and "claimed=True" in record.message
        for record in caplog.records
    )


def test_claim_a_second_time_returns_false(tmp_path: Path) -> None:
    """A pin that is already running is not claimed again."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    assert board.claim(connection, _TASK_ID, "2026-10-04T18:40:02Z") is True

    assert board.claim(connection, _TASK_ID, "2026-10-04T18:41:00Z") is False

    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.status == "running"
    assert loaded.started_at == "2026-10-04T18:40:02Z"


def test_claim_with_unknown_id_returns_false(tmp_path: Path) -> None:
    """A missing pin is not created by a claim."""
    board, connection = _open_board(tmp_path)

    assert board.claim(connection, _TASK_ID, "2026-10-04T18:40:02Z") is False


def test_claim_with_a_blank_started_at_raises(tmp_path: Path) -> None:
    """Dispatch time is required before the conditional update."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())

    with pytest.raises(ValueError, match="started_at"):
        board.claim(connection, _TASK_ID, " ")
    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.status == "queued"


def test_update_priority_changes_priority_only(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Priority changes. Status is left to the caller."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin(status="running", priority=40))
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    board.update_priority(connection, _TASK_ID, 10)

    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.priority == 10
    assert loaded.status == "running"
    assert any(
        record.levelno == logging.INFO
        and f"task_id={_TASK_ID}" in record.message
        and "priority=10" in record.message
        for record in caplog.records
    )


def test_update_priority_with_unknown_id_raises(tmp_path: Path) -> None:
    """A missing pin is not created by a priority change."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        board.update_priority(connection, _TASK_ID, 10)


def test_update_priority_with_a_bool_raises(tmp_path: Path) -> None:
    """True is not a queue priority."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())

    with pytest.raises(TypeError, match="priority"):
        board.update_priority(connection, _TASK_ID, True)  # type: ignore[arg-type]
    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.priority == 40


def test_set_phase_updates_phase_and_plan(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Phase and the readable plan change together. Status stays put."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    board.set_phase(connection, _TASK_ID, "acting", "1. List chapter files (running).")

    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.phase == "acting"
    assert loaded.current_plan == "1. List chapter files (running)."
    assert loaded.status == "queued"
    assert any(
        record.levelno == logging.INFO
        and f"task_id={_TASK_ID}" in record.message
        and "phase=acting" in record.message
        for record in caplog.records
    )


def test_set_phase_with_unknown_id_raises(tmp_path: Path) -> None:
    """A missing pin is not created by a phase update."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        board.set_phase(connection, _TASK_ID, "acting", "1. List chapter files (running).")


def test_set_phase_with_a_blank_phase_raises(tmp_path: Path) -> None:
    """An empty phase word is rejected before the update."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())

    with pytest.raises(ValueError, match="phase"):
        board.set_phase(connection, _TASK_ID, "  ", "1. List chapter files (running).")
    loaded = board.get_pin(connection, _TASK_ID)
    assert loaded is not None
    assert loaded.phase == "queued"


def test_list_queued_orders_priority_then_created_at(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Queued pins sort by priority, then created_at. Other statuses stay out."""
    board, connection = _open_board(tmp_path)
    later_low = _pin(task_id="task-later-80", priority=80, created_at="2026-10-04T18:41:00Z")
    first = _pin(task_id="task-first-10", priority=10, created_at="2026-10-04T18:50:00Z")
    earlier_low = _pin(task_id="task-early-80", priority=80, created_at="2026-10-04T18:40:00Z")
    running = _pin(task_id="task-running", priority=1, status="running")
    completed = _pin(task_id="task-done", priority=1, status="completed")
    for pin in (later_low, first, earlier_low, running, completed):
        board.insert_pin(connection, pin)
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    queued = board.list_queued(connection)

    assert [pin.task_id for pin in queued] == ["task-first-10", "task-early-80", "task-later-80"]
    assert all(pin.status == "queued" for pin in queued)
    assert any(
        record.levelno == logging.INFO and "blackboard.list_queued" in record.message and "count=3" in record.message
        for record in caplog.records
    )


def test_list_queued_with_no_queued_pins_returns_empty(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty queue is an empty tuple and logs a zero count."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin(status="running"))
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    assert board.list_queued(connection) == ()
    assert any(
        record.levelno == logging.INFO and "count=0" in record.message
        for record in caplog.records
    )


def test_list_queued_with_a_non_connection_raises(tmp_path: Path) -> None:
    """The caller must pass the Database connection."""
    board, _connection = _open_board(tmp_path)

    with pytest.raises(TypeError, match="Connection"):
        board.list_queued(None)  # type: ignore[arg-type]


def test_get_pin_with_a_blank_task_id_raises(tmp_path: Path) -> None:
    """A blank id is rejected before the select."""
    board, connection = _open_board(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        board.get_pin(connection, " ")


def test_insert_pin_with_a_non_connection_raises(tmp_path: Path) -> None:
    """The caller must pass the Database connection."""
    board, _connection = _open_board(tmp_path)

    with pytest.raises(TypeError, match="Connection"):
        board.insert_pin(None, _pin())  # type: ignore[arg-type]


def _open_board(tmp_path: Path) -> tuple[Blackboard, sqlite3.Connection]:
    """Open a migrated sqlite file under tmp_path. Example: (Blackboard(...), connection)."""
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    return Blackboard(database), connection


def _pin(**overrides: object) -> TaskPin:
    """Build one queued pin. Example: _pin(priority=10)."""
    values: dict[str, object] = {
        "task_id": _TASK_ID,
        "priority": 40,
        "status": "queued",
        "model": "qwen-14b",
        "goal": "Summarize the book into 2-3 pages.",
        "done_when": "summary.md exists",
        "access": AccessGrant("job"),
        "phase": "queued",
        "created_at": _CREATED_AT,
    }
    values.update(overrides)
    return TaskPin(**values)  # type: ignore[arg-type]


def _task_row(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """Return one tasks row. Example: the row for 7c1e0a2e-...."""
    row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    return row
