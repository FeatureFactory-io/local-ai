"""A reasoning note that cannot be parsed leaves the previous pin text."""

import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import AccessGrant, ReasoningNote, TaskPin
from local_ai.db.database import Database
from tests.support.log_story import assert_log_story


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


def test_append_reasoning_unparsed_note_keeps_previous_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A note whose step is not an int is discarded. The stored text stays."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    board.append_reasoning(
        connection,
        _TASK_ID,
        ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md."),
    )
    previous = _reasoning_text(connection)
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    board.append_reasoning(
        connection,
        _TASK_ID,
        ReasoningNote("2026-10-04T18:40:02Z", "2", "step is not an int"),  # type: ignore[arg-type]
    )

    assert _reasoning_text(connection) == previous
    assert_log_story(caplog.records, "blackboard.append", "entry", [f"task_id={_TASK_ID}"])


def test_append_reasoning_malformed_column_keeps_previous_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stored chunk that is not a note array is left byte for byte."""
    board, connection = _open_board(tmp_path)
    board.insert_pin(connection, _pin())
    malformed = '[{"at": "2026-10-04T18:40:01Z", "step": 2}]'
    connection.execute("UPDATE tasks SET reasoning = ? WHERE id = ?", (malformed, _TASK_ID))
    connection.commit()
    caplog.set_level(logging.INFO, logger="local_ai.blackboard.pins")

    board.append_reasoning(
        connection,
        _TASK_ID,
        ReasoningNote("2026-10-04T18:40:02Z", 2, "Reading ch-04.md."),
    )

    assert _reasoning_text(connection) == malformed
    assert_log_story(caplog.records, "blackboard.append", "entry", [f"task_id={_TASK_ID}"])


def _open_board(tmp_path: Path) -> tuple[Blackboard, sqlite3.Connection]:
    """Open a migrated sqlite file under tmp_path. Example: (Blackboard(...), connection)."""
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    return Blackboard(database), connection


def _pin() -> TaskPin:
    """Build one queued pin. Example: status queued."""
    return TaskPin(
        task_id=_TASK_ID,
        priority=40,
        status="queued",
        model="qwen-14b",
        goal="Summarize the book into 2-3 pages.",
        done_when="summary.md exists",
        access=AccessGrant("job"),
        phase="queued",
        created_at="2026-10-04T18:40:00Z",
    )


def _reasoning_text(connection: sqlite3.Connection) -> str:
    """Return the raw reasoning column. Example: '[]'."""
    row = connection.execute("SELECT reasoning FROM tasks WHERE id = ?", (_TASK_ID,)).fetchone()
    assert row is not None
    return str(row["reasoning"])
