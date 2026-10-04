"""PlanStore writes step rows in a real sqlite file."""

import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.contracts import StepDraft
from local_ai.db.database import Database
from local_ai.plan.steps import PlanStore


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_OTHER_TASK_ID = "aaaaaaaa-bbbb-4ccc-9ddd-eeeeeeeeeeee"


def test_insert_steps_writes_pending_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Each draft becomes a pending row. The log records task_id and the count."""
    store, connection = _open_store(tmp_path)
    drafts = (
        StepDraft(2, "Write a thesis per chapter"),
        StepDraft(1, "List chapter files"),
    )

    with caplog.at_level(logging.INFO, logger="local_ai.plan.steps"):
        store.insert_steps(connection, _TASK_ID, drafts)

    rows = _step_rows(connection, _TASK_ID)
    assert [(row["step_order"], row["action"], row["status"]) for row in rows] == [
        (1, "List chapter files", "pending"),
        (2, "Write a thesis per chapter", "pending"),
    ]
    assert all(row["result_json"] == "" and row["error"] == "" and row["llm_synthesis"] == "" for row in rows)
    assert _TASK_ID in caplog.text
    assert "count=2" in caplog.text


def test_insert_steps_rejects_duplicate_orders_and_writes_nothing(tmp_path: Path) -> None:
    """Orders must be unique for the task. A repeat leaves the table unchanged."""
    store, connection = _open_store(tmp_path)
    drafts = (
        StepDraft(1, "List chapter files"),
        StepDraft(1, "List chapter files again"),
    )

    with pytest.raises(ValueError, match="unique"):
        store.insert_steps(connection, _TASK_ID, drafts)

    assert _step_rows(connection, _TASK_ID) == []


def test_insert_steps_rejects_an_order_already_stored(tmp_path: Path) -> None:
    """A second insert cannot reuse an order that this task already has."""
    store, connection = _open_store(tmp_path)
    store.insert_steps(connection, _TASK_ID, (StepDraft(1, "List chapter files"),))

    with pytest.raises(ValueError, match="already exists"):
        store.insert_steps(connection, _TASK_ID, (StepDraft(1, "Write a thesis per chapter"),))

    assert len(_step_rows(connection, _TASK_ID)) == 1


def test_insert_steps_allows_the_same_order_on_another_task(tmp_path: Path) -> None:
    """Uniqueness is per task. Another pin may also have order 1."""
    store, connection = _open_store(tmp_path)
    _insert_task(connection, _OTHER_TASK_ID)
    store.insert_steps(connection, _TASK_ID, (StepDraft(1, "List chapter files"),))

    store.insert_steps(connection, _OTHER_TASK_ID, (StepDraft(1, "List chapter files"),))

    assert len(_step_rows(connection, _TASK_ID)) == 1
    assert len(_step_rows(connection, _OTHER_TASK_ID)) == 1


def test_insert_steps_empty_tuple_writes_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """No drafts means no rows and a count of zero."""
    store, connection = _open_store(tmp_path)

    with caplog.at_level(logging.INFO, logger="local_ai.plan.steps"):
        store.insert_steps(connection, _TASK_ID, ())

    assert _step_rows(connection, _TASK_ID) == []
    assert "count=0" in caplog.text


def test_insert_steps_rejects_bad_inputs(tmp_path: Path) -> None:
    """Fail fast on a non-tuple, a non-draft, a bad order, or an empty action."""
    store, connection = _open_store(tmp_path)

    with pytest.raises(TypeError, match="tuple"):
        store.insert_steps(connection, _TASK_ID, [StepDraft(1, "List chapter files")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StepDraft"):
        store.insert_steps(connection, _TASK_ID, ("List chapter files",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="order"):
        store.insert_steps(connection, _TASK_ID, (StepDraft(0, "List chapter files"),))
    with pytest.raises(ValueError, match="action"):
        store.insert_steps(connection, _TASK_ID, (StepDraft(1, "  "),))
    with pytest.raises(ValueError, match="task_id"):
        store.insert_steps(connection, "", (StepDraft(1, "List chapter files"),))
    assert _step_rows(connection, _TASK_ID) == []


def test_insert_steps_requires_the_parent_pin(tmp_path: Path) -> None:
    """plan_steps.task_id references tasks. A missing pin fails the insert."""
    store, connection = _open_store(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_steps(connection, "missing-task", (StepDraft(1, "List chapter files"),))


def _open_store(tmp_path: Path) -> tuple[PlanStore, sqlite3.Connection]:
    """Migrate a sqlite file under tmp_path and insert one pin.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: store and connection as tuple[PlanStore, sqlite3.Connection]. Example: (PlanStore(...), connection)
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    _insert_task(connection, _TASK_ID)
    return PlanStore(database), connection


def _insert_task(connection: sqlite3.Connection, task_id: str) -> None:
    """Insert one tasks row so plan_steps can reference it.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    """
    connection.execute(
        """
        INSERT INTO tasks (
            id, priority, status, model, goal, done_when, access_json, phase, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            40,
            "running",
            "qwen-14b",
            "Summarize the book into 2-3 pages.",
            "summary.md exists",
            '{"mode": "job"}',
            "acting",
            "2026-10-04T18:40:00Z",
        ),
    )


def _step_rows(connection: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    """Read step rows in order.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: rows as list[sqlite3.Row]. Example: []
    """
    return list(
        connection.execute(
            """
            SELECT step_order, action, status, result_json, error, llm_synthesis
            FROM plan_steps
            WHERE task_id = ?
            ORDER BY step_order
            """,
            (task_id,),
        ).fetchall()
    )
