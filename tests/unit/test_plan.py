"""PlanStore writes step rows in a real sqlite file."""

import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.contracts import InvalidStepTransition, PlanStep, StepDraft
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


def test_list_steps_returns_rows_ordered_by_step_order(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """list_steps returns PlanStep rows for one pin, lowest order first."""
    store, connection = _open_store(tmp_path)
    store.insert_steps(
        connection,
        _TASK_ID,
        (
            StepDraft(3, "Assemble summary.md"),
            StepDraft(1, "List chapter files"),
            StepDraft(2, "Write a thesis per chapter"),
        ),
    )
    connection.execute(
        """
        UPDATE plan_steps
        SET result_json = ?, error = ?, llm_synthesis = ?
        WHERE task_id = ? AND step_order = ?
        """,
        ('{"success": true}', "", "Found ch-01.md through ch-12.md", _TASK_ID, 1),
    )

    with caplog.at_level(logging.INFO, logger="local_ai.plan.steps"):
        steps = store.list_steps(connection, _TASK_ID)

    assert steps == (
        PlanStep(
            _TASK_ID,
            1,
            "List chapter files",
            "pending",
            '{"success": true}',
            "",
            "Found ch-01.md through ch-12.md",
        ),
        PlanStep(_TASK_ID, 2, "Write a thesis per chapter", "pending"),
        PlanStep(_TASK_ID, 3, "Assemble summary.md", "pending"),
    )
    assert _TASK_ID in caplog.text
    assert "count=3" in caplog.text


def test_list_steps_returns_empty_for_a_pin_with_no_steps(tmp_path: Path) -> None:
    """A pin with no plan returns an empty tuple."""
    store, connection = _open_store(tmp_path)

    assert store.list_steps(connection, _TASK_ID) == ()


def test_list_steps_ignores_another_pins_steps(tmp_path: Path) -> None:
    """Rows for a different task_id stay out of the result."""
    store, connection = _open_store(tmp_path)
    _insert_task(connection, _OTHER_TASK_ID)
    store.insert_steps(connection, _TASK_ID, (StepDraft(1, "List chapter files"),))
    store.insert_steps(connection, _OTHER_TASK_ID, (StepDraft(1, "Export pdf"),))

    steps = store.list_steps(connection, _TASK_ID)

    assert steps == (PlanStep(_TASK_ID, 1, "List chapter files", "pending"),)


@pytest.mark.parametrize("status", ["pending", "waiting_retry"])
def test_mark_running_moves_an_open_step_to_running(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    status: str,
) -> None:
    """pending and waiting_retry become running. The log records task_id and order."""
    store, connection = _open_store(tmp_path)
    store.insert_steps(connection, _TASK_ID, (StepDraft(2, "Write a thesis per chapter"),))
    _set_status(connection, _TASK_ID, 2, status)

    with caplog.at_level(logging.INFO, logger="local_ai.plan.steps"):
        store.mark_running(connection, _TASK_ID, 2)

    assert _step_rows(connection, _TASK_ID)[0]["status"] == "running"
    assert _TASK_ID in caplog.text
    assert "order=2" in caplog.text


@pytest.mark.parametrize("status", ["completed", "failed", "running"])
def test_mark_running_rejects_a_closed_or_running_step(tmp_path: Path, status: str) -> None:
    """completed, failed, and running stay put and raise InvalidStepTransition."""
    store, connection = _open_store(tmp_path)
    store.insert_steps(connection, _TASK_ID, (StepDraft(1, "List chapter files"),))
    _set_status(connection, _TASK_ID, 1, status)

    with pytest.raises(InvalidStepTransition):
        store.mark_running(connection, _TASK_ID, 1)

    assert _step_rows(connection, _TASK_ID)[0]["status"] == status


def test_mark_running_rejects_a_missing_step(tmp_path: Path) -> None:
    """A missing order is not a transition."""
    store, connection = _open_store(tmp_path)
    store.insert_steps(connection, _TASK_ID, (StepDraft(1, "List chapter files"),))

    with pytest.raises(InvalidStepTransition, match="order=9"):
        store.mark_running(connection, _TASK_ID, 9)

    assert _step_rows(connection, _TASK_ID)[0]["status"] == "pending"


def test_list_steps_rejects_a_blank_task_id(tmp_path: Path) -> None:
    """A blank pin id is not a query."""
    store, connection = _open_store(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        store.list_steps(connection, "  ")


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


def _set_status(connection: sqlite3.Connection, task_id: str, order: int, status: str) -> None:
    """Force one step status so a transition test can start from it.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: step order as int. Example: 1
    :param status: step status as str. Example: "waiting_retry"
    :return: None. Example: None
    """
    connection.execute(
        "UPDATE plan_steps SET status = ? WHERE task_id = ? AND step_order = ?",
        (status, task_id, order),
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
