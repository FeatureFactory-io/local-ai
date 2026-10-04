"""Blackboard pin rows on a real sqlite file."""

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import AccessGrant, TaskPin
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
