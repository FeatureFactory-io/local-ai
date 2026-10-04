"""Pin rows. Worker appends reasoning. DispatcherService reads and finishes pins."""

import json
import logging
import sqlite3

from local_ai.contracts import REASONING_CAP, AccessGrant, JobSummary, ReasoningNote, TaskPin
from local_ai.db.database import Database


logger = logging.getLogger(__name__)

_QUEUED = "queued"
_RUNNING = "running"
_TERMINAL_STATUSES = ("completed", "failed")

_SET_FINAL_SQL = """
UPDATE tasks
SET status = ?, phase = ?, final_status = ?, error = ?, artifacts_json = ?, finished_at = ?
WHERE id = ?
"""
_TRUNCATION_TEXT = "earlier notes were truncated"
_TRUNCATION_AT = "2026-10-04T00:00:00Z"

_LIST_QUEUED_SQL = """
SELECT * FROM tasks
WHERE status = ?
ORDER BY priority ASC, created_at ASC
"""

_SUMMARY_COLUMNS = (
    "id, priority, status, model, phase, goal, error, created_at, started_at, finished_at"
)

_LIST_SUMMARY_SQL = f"""
SELECT {_SUMMARY_COLUMNS}
FROM tasks
WHERE (? IS NULL OR status = ?)
"""

_INSERT_PIN_SQL = """
INSERT INTO tasks (
    id, priority, status, model, goal, done_when, access_json, phase,
    current_plan, reasoning, artifacts_json, error, created_at, started_at,
    finished_at, final_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Blackboard:
    """
    The pin. Durability is the tasks row.

    Called by DispatcherService and Worker. Calls Database only.
    """

    def __init__(self, database: Database) -> None:
        """
        Store the shared database.

        :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
        """
        self._database = database

    def insert_pin(
        self,
        connection: sqlite3.Connection,
        pin: TaskPin,
    ) -> None:
        """
        Insert one queued pin.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param pin: new pin as TaskPin. Example: a pin with status "queued" and phase "queued"
        :return: None. Example: None
        :raises TypeError: when connection or pin has the wrong type. Example: None
        :raises ValueError: when task_id is blank. Example: "  "
        """
        _require_connection(connection)
        _require_pin(pin)
        connection.execute(_INSERT_PIN_SQL, _insert_parameters(pin))
        connection.commit()
        logger.info(
            "blackboard.insert | task_id=%s | priority=%s | model=%s",
            pin.task_id,
            pin.priority,
            pin.model,
        )

    def get_pin(self, connection: sqlite3.Connection, task_id: str) -> TaskPin | None:
        """
        Load one pin.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: pin or None as TaskPin | None. Example: None
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id is blank or stored JSON cannot be parsed. Example: "not-json"
        """
        _require_connection(connection)
        _require_task_id(task_id)
        row = _select_task(connection, task_id)
        if row is None:
            logger.info("blackboard.get | task_id=%s | status=missing", task_id)
            return None
        pin = _pin_from_row(row)
        logger.info("blackboard.get | task_id=%s | status=%s", pin.task_id, pin.status)
        return pin

    def append_reasoning(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        note: ReasoningNote | None,
    ) -> None:
        """
        Append one note. A bad note leaves the previous text in place.

        Notes past 32_000 characters drop the oldest and keep a truncation marker.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param note: note to append as ReasoningNote | None. Example: ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
        :return: None. Example: None
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id is blank or the pin is missing. Example: "missing-id"
        """
        _require_connection(connection)
        _require_task_id(task_id)
        logger.info("blackboard.append | entry | task_id=%s", task_id)
        _append_parsed_note(connection, task_id, _incoming_note(note))

    def set_phase(self, connection: sqlite3.Connection, task_id: str, phase: str, current_plan: str) -> None:
        """
        Update the live phase and the readable plan.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param phase: phase word as str. Example: "acting"
        :param current_plan: readable plan as str. Example: "1. List chapter files (running)."
        :return: None. Example: None
        :raises TypeError: when connection or current_plan has the wrong type. Example: None
        :raises ValueError: when task_id or phase is blank, or the pin is missing. Example: "  "
        """
        _require_connection(connection)
        _require_task_id(task_id)
        _require_text(phase, "phase", "acting")
        _require_plan(current_plan)
        _update_one(
            connection,
            "UPDATE tasks SET phase = ?, current_plan = ? WHERE id = ?",
            (phase, current_plan, task_id),
            task_id,
        )
        logger.info("blackboard.set_phase | task_id=%s | phase=%s", task_id, phase)

    def set_final(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        status: str,
        error: str,
        artifacts: tuple[str, ...],
        finished_at: str,
    ) -> None:
        """
        Write the terminal pin state when the slot is released.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param status: completed or failed as str. Example: "completed"
        :param error: failure summary as str. Example: ""
        :param artifacts: output paths as tuple[str, ...]. Example: ("/tmp/summary.md",)
        :param finished_at: stop time as str. Example: "2026-10-04T18:55:00Z"
        :return: None. Example: None
        :raises TypeError: when connection, error, or artifacts has the wrong type. Example: "/tmp/summary.md"
        :raises ValueError: when status is not completed or failed, or the pin is missing. Example: "running"
        """
        _require_connection(connection)
        _require_task_id(task_id)
        _require_terminal(status)
        _finish_row(connection, task_id, status, error, artifacts, finished_at)
        logger.info("blackboard.set_final | task_id=%s | status=%s", task_id, status)

    def update_priority(self, connection: sqlite3.Connection, task_id: str, priority: int) -> None:
        """
        Change priority of a queued pin. The caller has already checked the status.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 10
        :return: None. Example: None
        :raises TypeError: when connection is not a Connection or priority is not an int. Example: True
        :raises ValueError: when task_id is blank or the pin is missing. Example: "missing-id"
        """
        _require_connection(connection)
        _require_task_id(task_id)
        _require_priority(priority)
        _update_one(
            connection,
            "UPDATE tasks SET priority = ? WHERE id = ?",
            (priority, task_id),
            task_id,
        )
        logger.info("blackboard.update_priority | task_id=%s | priority=%s", task_id, priority)

    def claim(self, connection: sqlite3.Connection, task_id: str, started_at: str) -> bool:
        """
        Move one queued pin to running. A second claim returns False.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param started_at: dispatch time as str. Example: "2026-10-04T18:40:02Z"
        :return: whether this call won the row as bool. Example: True
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id or started_at is blank. Example: " "
        """
        _require_connection(connection)
        _require_task_id(task_id)
        _require_text(started_at, "started_at", "2026-10-04T18:40:02Z")
        won = _claim_queued(connection, task_id, started_at)
        logger.info("blackboard.claim | task_id=%s | claimed=%s", task_id, won)
        return won

    def requeue_running(self, connection: sqlite3.Connection) -> tuple[str, ...]:
        """
        Return every running pin to queued and clear started_at.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: requeued ids as tuple[str, ...]. Example: ("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f",)
        :raises TypeError: when connection is not a Connection. Example: None
        """
        _require_connection(connection)
        task_ids = _running_ids(connection)
        if task_ids:
            _requeue_rows(connection)
        _log_requeued(task_ids)
        return task_ids

    def list_queued(self, connection: sqlite3.Connection) -> tuple[TaskPin, ...]:
        """
        Queued pins in queue order.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: pins as tuple[TaskPin, ...]. Example: ()
        :raises TypeError: when connection is not a Connection. Example: None
        """
        _require_connection(connection)
        rows = connection.execute(_LIST_QUEUED_SQL, (_QUEUED,)).fetchall()
        pins = tuple(_pin_from_row(row) for row in rows)
        logger.info("blackboard.list_queued | count=%s", len(pins))
        return pins

    def list_summaries(
        self,
        connection: sqlite3.Connection,
        status: str | None,
    ) -> tuple[JobSummary, ...]:
        """
        Compact rows for one status, or every pin when status is omitted.

        Running rows come first, then queued rows in queue order, then finished rows
        newest finished_at first. Reasoning and steps are not loaded.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param status: queued, running, completed, failed, or None as str | None. Example: "queued"
        :return: rows as tuple[JobSummary, ...]. Example: ()
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when status is not a pin status. Example: "done"
        """
        _require_connection(connection)
        _require_summary_status(status)
        rows = connection.execute(_LIST_SUMMARY_SQL, (status, status)).fetchall()
        summaries = _ordered_summaries(tuple(_summary_from_row(row) for row in rows))
        logger.info("blackboard.list_summaries | status=%s count=%s", status or "all", len(summaries))
        return summaries

    def delete_queued(self, connection: sqlite3.Connection, task_id: str) -> bool:
        """
        Delete one queued pin and its plan steps. A pin that is not queued stays.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: whether the queued row was deleted as bool. Example: True
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id is blank or not one path segment. Example: "../other"
        """
        _require_connection(connection)
        _require_task_id(task_id)
        removed = _delete_queued_row(connection, task_id)
        logger.info("blackboard.delete_queued | task_id=%s | removed=%s", task_id, removed)
        return removed

    def delete_terminal(
        self,
        connection: sqlite3.Connection,
        task_id: str | None,
        statuses: tuple[str, ...],
    ) -> tuple[JobSummary, ...]:
        """
        Delete completed or failed pins and their plan steps. Queued and running stay.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: one pin id, or None for every match, as str | None. Example: None
        :param statuses: completed, failed, or both as tuple[str, ...]. Example: ("completed", "failed")
        :return: deleted rows as tuple[JobSummary, ...]. Example: ()
        :raises TypeError: when connection or statuses has the wrong type. Example: ["completed"]
        :raises ValueError: when a status is not terminal, or task_id is not one path segment. Example: "running"
        """
        _require_connection(connection)
        _require_terminal_statuses(statuses)
        if task_id is not None:
            _require_task_id(task_id)
        removed = _delete_terminal_rows(connection, task_id, statuses)
        logger.info(
            "blackboard.delete_terminal | task_id=%s | count=%s",
            task_id or "*",
            len(removed),
        )
        return removed


def _require_summary_status(status: str | None) -> None:
    """
    Accept a pin status or None for every pin.

    :param status: filter as str | None. Example: "queued"
    :return: None. Example: None
    :raises ValueError: when status is not a pin status. Example: "done"
    """
    if status is None:
        return
    if status not in ("queued", "running", *_TERMINAL_STATUSES):
        raise ValueError("status must be queued, running, completed, or failed. Example: 'queued'")


def _require_terminal_statuses(statuses: tuple[str, ...]) -> None:
    """
    Accept a non-empty tuple of completed and failed.

    :param statuses: terminal statuses as tuple[str, ...]. Example: ("completed", "failed")
    :return: None. Example: None
    :raises TypeError: when statuses is not a tuple of strings. Example: ["completed"]
    :raises ValueError: when a status is not terminal or the tuple is empty. Example: ("running",)
    """
    if isinstance(statuses, str) or not isinstance(statuses, tuple) or not statuses:
        raise TypeError("statuses must be a non-empty tuple. Example: ('completed', 'failed')")
    if any(not isinstance(status, str) or status not in _TERMINAL_STATUSES for status in statuses):
        raise ValueError("statuses must be completed or failed. Example: ('completed',)")


def _ordered_summaries(rows: tuple[JobSummary, ...]) -> tuple[JobSummary, ...]:
    """
    Place running first, then the queue, then finished pins newest first.

    :param rows: unsorted rows as tuple[JobSummary, ...]. Example: ()
    :return: ordered rows as tuple[JobSummary, ...]. Example: ()
    """
    running = sorted(
        (row for row in rows if row.status == "running"),
        key=lambda row: (row.started_at or "", row.task_id),
    )
    queued = sorted(
        (row for row in rows if row.status == "queued"),
        key=lambda row: (row.priority, row.created_at, row.task_id),
    )
    terminal = [row for row in rows if row.status in _TERMINAL_STATUSES]
    terminal.sort(key=lambda row: row.task_id)
    terminal.sort(key=lambda row: row.finished_at or row.created_at, reverse=True)
    return tuple(running) + tuple(queued) + tuple(terminal)


def _summary_from_row(row: sqlite3.Row) -> JobSummary:
    """
    Map the compact columns onto a summary. NULL times stay None.

    :param row: tasks row as sqlite3.Row. Example: the row for id 7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f
    :return: summary as JobSummary. Example: a summary with status "queued"
    """
    return JobSummary(
        task_id=str(row["id"]),
        priority=int(row["priority"]),
        status=str(row["status"]),
        model=str(row["model"]),
        phase=str(row["phase"]),
        goal=str(row["goal"]),
        error=str(row["error"]),
        created_at=str(row["created_at"]),
        started_at=_optional_text(row["started_at"]),
        finished_at=_optional_text(row["finished_at"]),
    )


def _delete_queued_row(connection: sqlite3.Connection, task_id: str) -> bool:
    """
    Delete steps and the pin only while it is still queued, then commit.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: whether one queued row was deleted as bool. Example: True
    """
    try:
        _delete_steps_if_status(connection, (task_id,), (_QUEUED,))
        cursor = connection.execute(
            "DELETE FROM tasks WHERE id = ? AND status = ?",
            (task_id, _QUEUED),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise


def _delete_terminal_rows(
    connection: sqlite3.Connection,
    task_id: str | None,
    statuses: tuple[str, ...],
) -> tuple[JobSummary, ...]:
    """
    Delete the matching finished pins. A status change rolls the delete back.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: one pin id, or None for every match, as str | None. Example: None
    :param statuses: terminal statuses as tuple[str, ...]. Example: ("completed",)
    :return: deleted rows as tuple[JobSummary, ...]. Example: ()
    """
    rows = _select_terminal(connection, task_id, statuses)
    if not rows:
        return ()
    ids = tuple(row.task_id for row in rows)
    try:
        _delete_steps_if_status(connection, ids, statuses)
        cursor = connection.execute(_delete_tasks_sql(len(ids), len(statuses)), (*ids, *statuses))
        if cursor.rowcount != len(ids):
            connection.rollback()
            return ()
        connection.commit()
        return _ordered_summaries(rows)
    except Exception:
        connection.rollback()
        raise


def _select_terminal(
    connection: sqlite3.Connection,
    task_id: str | None,
    statuses: tuple[str, ...],
) -> tuple[JobSummary, ...]:
    """
    Load finished summaries that match the filter.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: one pin id, or None for every match, as str | None. Example: None
    :param statuses: terminal statuses as tuple[str, ...]. Example: ("failed",)
    :return: rows as tuple[JobSummary, ...]. Example: ()
    """
    status_clause = _placeholders(len(statuses))
    if task_id is None:
        sql = f"SELECT {_SUMMARY_COLUMNS} FROM tasks WHERE status IN ({status_clause})"
        parameters: tuple[object, ...] = statuses
    else:
        sql = f"SELECT {_SUMMARY_COLUMNS} FROM tasks WHERE id = ? AND status IN ({status_clause})"
        parameters = (task_id, *statuses)
    found = connection.execute(sql, parameters).fetchall()
    return tuple(_summary_from_row(row) for row in found)


def _delete_steps_if_status(
    connection: sqlite3.Connection,
    task_ids: tuple[str, ...],
    statuses: tuple[str, ...],
) -> None:
    """
    Delete plan steps whose parent pin still has one of the given statuses.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_ids: pin ids as tuple[str, ...]. Example: ("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f",)
    :param statuses: required parent statuses as tuple[str, ...]. Example: ("queued",)
    :return: None. Example: None
    """
    id_clause = _placeholders(len(task_ids))
    status_clause = _placeholders(len(statuses))
    connection.execute(
        f"""
        DELETE FROM plan_steps
        WHERE task_id IN (
            SELECT id FROM tasks
            WHERE id IN ({id_clause}) AND status IN ({status_clause})
        )
        """,
        (*task_ids, *statuses),
    )


def _delete_tasks_sql(id_count: int, status_count: int) -> str:
    """
    SQL that deletes tasks still in the bound statuses.

    :param id_count: number of id placeholders as int. Example: 1
    :param status_count: number of status placeholders as int. Example: 2
    :return: SQL as str. Example: "DELETE FROM tasks WHERE id IN (?) AND status IN (?, ?)"
    """
    return (
        "DELETE FROM tasks "
        f"WHERE id IN ({_placeholders(id_count)}) AND status IN ({_placeholders(status_count)})"
    )


def _placeholders(count: int) -> str:
    """
    Build a bound-parameter list. The count is not user text.

    :param count: placeholder count as int. Example: 2
    :return: comma-separated question marks as str. Example: "?, ?"
    :raises ValueError: when count is below 1. Example: 0
    """
    if count < 1:
        raise ValueError("count must be at least 1. Example: 1")
    return ", ".join("?" for _ in range(count))


def _require_connection(connection: sqlite3.Connection) -> None:
    """
    Reject a value that is not an open sqlite connection.

    :param connection: candidate connection as sqlite3.Connection. Example: the Database connection
    :return: None. Example: None
    :raises TypeError: when connection is not a Connection. Example: None
    """
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be sqlite3.Connection. Example: the Database connection")


def _require_pin(pin: TaskPin) -> None:
    """
    Reject a pin that cannot be stored as a tasks row.

    :param pin: candidate pin as TaskPin. Example: a pin with status "queued"
    :return: None. Example: None
    :raises TypeError: when pin or its access is the wrong type. Example: {}
    :raises ValueError: when task_id is blank. Example: "  "
    """
    if not isinstance(pin, TaskPin):
        raise TypeError("pin must be TaskPin. Example: a pin with status 'queued'")
    _require_task_id(pin.task_id)
    if not isinstance(pin.access, AccessGrant):
        raise TypeError("pin.access must be AccessGrant. Example: AccessGrant('job')")


def _require_terminal(status: object) -> None:
    """
    Accept only completed or failed.

    :param status: candidate status as object. Example: "completed"
    :return: None. Example: None
    :raises ValueError: when status is not terminal. Example: "running"
    """
    if status not in _TERMINAL_STATUSES:
        raise ValueError("status must be completed or failed. Example: 'completed'")


def _finish_row(
    connection: sqlite3.Connection,
    task_id: str,
    status: str,
    error: str,
    artifacts: tuple[str, ...],
    finished_at: str,
) -> None:
    """
    Write the terminal columns. Phase uses the same word as status.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param status: completed or failed as str. Example: "completed"
    :param error: failure summary as str. Example: ""
    :param artifacts: output paths as tuple[str, ...]. Example: ("/tmp/summary.md",)
    :param finished_at: stop time as str. Example: "2026-10-04T18:55:00Z"
    :return: None. Example: None
    :raises TypeError: when error or artifacts has the wrong type. Example: "/tmp/summary.md"
    :raises ValueError: when finished_at is blank or the pin is missing. Example: " "
    """
    if not isinstance(error, str):
        raise TypeError("error must be str. Example: ''")
    _require_text(finished_at, "finished_at", "2026-10-04T18:55:00Z")
    _update_one(
        connection,
        _SET_FINAL_SQL,
        (status, status, status, error, _dump_artifacts(artifacts), finished_at, task_id),
        task_id,
    )


def _running_ids(connection: sqlite3.Connection) -> tuple[str, ...]:
    """
    Ids of running pins in created_at order.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: ids as tuple[str, ...]. Example: ("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f",)
    """
    rows = connection.execute(
        "SELECT id FROM tasks WHERE status = ? ORDER BY created_at ASC",
        (_RUNNING,),
    ).fetchall()
    return tuple(str(row["id"]) for row in rows)


def _requeue_rows(connection: sqlite3.Connection) -> None:
    """
    Set running rows back to queued and clear started_at. Plan steps are not updated.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: None. Example: None
    """
    connection.execute(
        "UPDATE tasks SET status = ?, phase = ?, started_at = NULL WHERE status = ?",
        (_QUEUED, _QUEUED, _RUNNING),
    )
    connection.commit()


def _log_requeued(task_ids: tuple[str, ...]) -> None:
    """
    Log each requeued id, or a zero count when none were running.

    :param task_ids: requeued ids as tuple[str, ...]. Example: ()
    :return: None. Example: None
    """
    if not task_ids:
        logger.info("blackboard.requeue_running | count=0")
        return
    for task_id in task_ids:
        logger.info("blackboard.requeue_running | task_id=%s", task_id)


def _claim_queued(connection: sqlite3.Connection, task_id: str, started_at: str) -> bool:
    """
    Set running only while the row is still queued.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param started_at: dispatch time as str. Example: "2026-10-04T18:40:02Z"
    :return: whether one row changed as bool. Example: True
    """
    updated = connection.execute(
        "UPDATE tasks SET status = ?, started_at = ? WHERE id = ? AND status = ?",
        (_RUNNING, started_at, task_id, _QUEUED),
    )
    if updated.rowcount != 1:
        return False
    connection.commit()
    return True


def _require_priority(priority: object) -> None:
    """
    Reject a priority that is not an int. Bool is rejected because it is a subclass of int.

    :param priority: candidate priority as object. Example: 10
    :return: None. Example: None
    :raises TypeError: when priority is not an int. Example: True
    """
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise TypeError("priority must be int. Example: 10")


def _require_text(value: object, name: str, example: str) -> None:
    """
    Reject a blank or non-string value.

    :param value: candidate text as object. Example: "acting"
    :param name: argument name as str. Example: "phase"
    :param example: example used in the error as str. Example: "acting"
    :return: None. Example: None
    :raises ValueError: when value is blank or not a string. Example: "  "
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string. Example: '{example}'")


def _require_plan(current_plan: str) -> None:
    """
    Reject a plan that is not text. An empty plan is allowed.

    :param current_plan: readable plan as str. Example: "1. List chapter files (running)."
    :return: None. Example: None
    :raises TypeError: when current_plan is not a string. Example: None
    """
    if not isinstance(current_plan, str):
        raise TypeError("current_plan must be str. Example: '1. List chapter files (running).'")


def _update_one(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...],
    task_id: str,
) -> None:
    """
    Update one pin row and commit. Zero rows means the pin is missing.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param sql: update statement as str. Example: "UPDATE tasks SET phase = ? WHERE id = ?"
    :param parameters: bound values as tuple. Example: ("acting", "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    :raises ValueError: when no row changed. Example: "missing-id"
    """
    updated = connection.execute(sql, parameters)
    if updated.rowcount != 1:
        raise ValueError(
            f"pin not found task_id={task_id}. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'"
        )
    connection.commit()


def _require_task_id(task_id: str) -> None:
    """
    Reject a blank pin id.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    :raises ValueError: when task_id is blank. Example: "  "
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(
            "task_id must be a non-empty string. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'"
        )


def _insert_parameters(pin: TaskPin) -> tuple[object, ...]:
    """
    Map a pin onto the tasks insert columns.

    :param pin: new pin as TaskPin. Example: a pin with status "queued"
    :return: bound values as tuple. Example: ("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", 40, "queued")
    """
    return (
        pin.task_id,
        pin.priority,
        pin.status,
        pin.model,
        pin.goal,
        pin.done_when,
        _dump_access(pin.access),
        pin.phase,
        pin.current_plan,
        _dump_notes(pin.reasoning),
        _dump_artifacts(pin.artifacts),
        pin.error,
        pin.created_at,
        pin.started_at,
        pin.finished_at,
        pin.final_status,
    )


def _dump_access(access: AccessGrant) -> str:
    """
    Encode a grant as JSON. Paths are omitted when the grant has none.

    :param access: grant as AccessGrant. Example: AccessGrant("job")
    :return: JSON object as str. Example: '{"mode": "job"}'
    """
    payload: dict[str, object] = {"mode": access.mode}
    if access.paths:
        payload["paths"] = list(access.paths)
    return json.dumps(payload)


def _dump_notes(notes: tuple[ReasoningNote, ...]) -> str:
    """
    Encode reasoning notes as a JSON array.

    :param notes: notes as tuple[ReasoningNote, ...]. Example: ()
    :return: JSON array as str. Example: "[]"
    :raises TypeError: when notes is not a tuple of ReasoningNote. Example: []
    """
    if not isinstance(notes, tuple):
        raise TypeError("reasoning must be a tuple of ReasoningNote. Example: ()")
    payload = [_note_payload(note) for note in notes]
    return json.dumps(payload)


def _note_payload(note: ReasoningNote) -> dict[str, object]:
    """
    Encode one note as at, step, and text.

    :param note: one note as ReasoningNote. Example: ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    :return: note object as dict. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}
    :raises TypeError: when note is not a ReasoningNote. Example: "Reading ch-04.md."
    """
    if not isinstance(note, ReasoningNote):
        raise TypeError(
            "reasoning entry must be ReasoningNote. "
            "Example: ReasoningNote('2026-10-04T18:40:01Z', 2, 'Reading ch-04.md.')"
        )
    return {"at": note.at, "step": note.step, "text": note.text}


def _dump_artifacts(artifacts: tuple[str, ...]) -> str:
    """
    Encode artifact paths as a JSON array.

    :param artifacts: paths as tuple[str, ...]. Example: ("/tmp/summary.md",)
    :return: JSON array as str. Example: '["/tmp/summary.md"]'
    :raises TypeError: when artifacts is not a tuple of strings. Example: "/tmp/summary.md"
    """
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, tuple):
        raise TypeError("artifacts must be a tuple of strings. Example: ('/tmp/summary.md',)")
    if any(not isinstance(path, str) for path in artifacts):
        raise TypeError("artifacts must be a tuple of strings. Example: ('/tmp/summary.md',)")
    return json.dumps(list(artifacts))


def _select_task(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    """
    Load one tasks row by id.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: row or None as sqlite3.Row | None. Example: None
    """
    return connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _pin_from_row(row: sqlite3.Row) -> TaskPin:
    """
    Map a tasks row onto a pin. Access and reasoning are JSON on the row.

    :param row: tasks row as sqlite3.Row. Example: the row for id 7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f
    :return: pin as TaskPin. Example: a pin with status "queued"
    :raises ValueError: when access, reasoning, or artifacts JSON cannot be parsed. Example: "not-json"
    """
    return TaskPin(
        task_id=str(row["id"]),
        priority=int(row["priority"]),
        status=str(row["status"]),
        model=str(row["model"]),
        goal=str(row["goal"]),
        done_when=str(row["done_when"]),
        access=_load_access(str(row["access_json"])),
        phase=str(row["phase"]),
        current_plan=str(row["current_plan"]),
        reasoning=_load_notes(str(row["reasoning"])),
        artifacts=_load_artifacts(str(row["artifacts_json"])),
        error=str(row["error"]),
        created_at=str(row["created_at"]),
        started_at=_optional_text(row["started_at"]),
        finished_at=_optional_text(row["finished_at"]),
        final_status=str(row["final_status"]),
    )


def _optional_text(value: object) -> str | None:
    """
    Keep NULL as None and other values as text.

    :param value: column value as object. Example: None
    :return: text or None as str | None. Example: None
    """
    if value is None:
        return None
    return str(value)


def _load_access(text: str) -> AccessGrant:
    """
    Parse access_json into a grant.

    :param text: JSON object as str. Example: '{"mode": "job"}'
    :return: grant as AccessGrant. Example: AccessGrant("job")
    :raises ValueError: when text is not a grant object. Example: "not-json"
    """
    payload = _load_object(text, "access_json")
    mode = payload.get("mode")
    paths = payload.get("paths", [])
    if not isinstance(mode, str) or not _string_list(paths):
        raise ValueError('access_json must be a grant object. Example: {"mode": "job"}')
    return AccessGrant(mode, tuple(paths))


def _load_notes(text: str) -> tuple[ReasoningNote, ...]:
    """
    Parse the reasoning column into notes, newest last.

    :param text: JSON array as str. Example: "[]"
    :return: notes as tuple[ReasoningNote, ...]. Example: ()
    :raises ValueError: when text is not an array of notes. Example: "not-json"
    """
    return tuple(_note_from_payload(item) for item in _load_array(text, "reasoning"))


def _note_from_payload(item: object) -> ReasoningNote:
    """
    Parse one stored note object.

    :param item: JSON value as object. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}
    :return: note as ReasoningNote. Example: ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    :raises ValueError: when item is not a note object. Example: "Reading ch-04.md."
    """
    if not isinstance(item, dict):
        raise ValueError(
            'reasoning note must be an object. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}'
        )
    at = item.get("at")
    step = item.get("step")
    text = item.get("text")
    if not isinstance(at, str) or isinstance(step, bool) or not isinstance(step, int) or not isinstance(text, str):
        raise ValueError(
            'reasoning note must have at, step, and text. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}'
        )
    return ReasoningNote(at, step, text)


def _load_artifacts(text: str) -> tuple[str, ...]:
    """
    Parse artifacts_json into paths.

    :param text: JSON array as str. Example: "[]"
    :return: paths as tuple[str, ...]. Example: ()
    :raises ValueError: when text is not an array of strings. Example: "not-json"
    """
    payload = _load_array(text, "artifacts_json")
    if not _string_list(payload):
        raise ValueError('artifacts_json must be an array of strings. Example: ["/tmp/summary.md"]')
    return tuple(payload)


def _load_object(text: str, column: str) -> dict[str, object]:
    """
    Parse a JSON object column.

    :param text: JSON text as str. Example: '{"mode": "job"}'
    :param column: column name as str. Example: "access_json"
    :return: object as dict. Example: {"mode": "job"}
    :raises ValueError: when text is not a JSON object. Example: "not-json"
    """
    payload = _load_json(text, column)
    if not isinstance(payload, dict):
        raise ValueError(f"{column} must be a JSON object. Example: " + '{"mode": "job"}')
    return payload


def _load_array(text: str, column: str) -> list[object]:
    """
    Parse a JSON array column.

    :param text: JSON text as str. Example: "[]"
    :param column: column name as str. Example: "reasoning"
    :return: array as list. Example: []
    :raises ValueError: when text is not a JSON array. Example: "not-json"
    """
    payload = _load_json(text, column)
    if not isinstance(payload, list):
        raise ValueError(f"{column} must be a JSON array. Example: []")
    return payload


def _load_json(text: str, column: str) -> object:
    """
    Parse JSON or raise a column-specific error.

    :param text: JSON text as str. Example: "[]"
    :param column: column name as str. Example: "reasoning"
    :return: parsed value as object. Example: []
    :raises ValueError: when text is not JSON. Example: "not-json"
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{column} is not JSON. Example: []") from exc


def _string_list(value: object) -> bool:
    """
    Tell whether a value is a list of strings.

    :param value: candidate as object. Example: ["/tmp/summary.md"]
    :return: True when every entry is a string. Example: True
    """
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _append_parsed_note(
    connection: sqlite3.Connection,
    task_id: str,
    parsed: dict[str, object] | None,
) -> None:
    """
    Store one parsed note. A column that cannot be parsed is left unchanged.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param parsed: note object or None as dict | None. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}
    :return: None. Example: None
    :raises ValueError: when the pin does not exist. Example: "missing-id"
    """
    if parsed is None:
        return
    previous = _reasoning_text(connection, task_id)
    if previous is None:
        raise ValueError(
            "pin not found task_id="
            f"{task_id}. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'"
        )
    try:
        notes = _stored_note_payloads(previous)
    except ValueError:
        logger.info("blackboard.append kept previous task_id=%s", task_id)
        return
    notes.append(parsed)
    _write_reasoning(connection, task_id, _capped_reasoning(notes, task_id))


def _capped_reasoning(notes: list[dict[str, object]], task_id: str) -> str:
    """
    Drop oldest notes once the JSON would pass the board cap, and keep one marker.

    :param notes: note objects newest last as list. Example: [{"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}]
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: reasoning JSON as str. Example: "[]"
    """
    encoded = json.dumps(notes)
    if len(encoded) <= REASONING_CAP:
        return encoded
    logger.info("blackboard.append truncated task_id=%s", task_id)
    kept = [note for note in notes if note.get("text") != _TRUNCATION_TEXT]
    while kept and len(_dump_with_marker(kept)) > REASONING_CAP:
        kept.pop(0)
    return _dump_with_marker(kept)


def _dump_with_marker(notes: list[dict[str, object]]) -> str:
    """
    Put the truncation marker ahead of the notes that still fit.

    :param notes: kept notes as list. Example: [{"at": "2026-10-04T18:41:00Z", "step": 2, "text": "newest"}]
    :return: reasoning JSON as str. Example: '[{"at": "2026-10-04T00:00:00Z", "step": 0, "text": "earlier notes were truncated"}]'
    """
    marker: dict[str, object] = {"at": _TRUNCATION_AT, "step": 0, "text": _TRUNCATION_TEXT}
    return json.dumps([marker, *notes])


def _incoming_note(note: ReasoningNote | None) -> dict[str, object] | None:
    """
    Parse one note into at, step, and text. A bad note is None.

    :param note: note to append as ReasoningNote | None. Example: ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    :return: note object or None as dict | None. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}
    """
    if not isinstance(note, ReasoningNote):
        return None
    if not isinstance(note.at, str) or not note.at.strip():
        return None
    if isinstance(note.step, bool) or not isinstance(note.step, int):
        return None
    if not isinstance(note.text, str) or not note.text.strip():
        return None
    return {"at": note.at, "step": note.step, "text": note.text}


def _stored_note_payloads(text: str) -> list[dict[str, object]]:
    """
    Parse stored reasoning into note objects.

    :param text: reasoning column as str. Example: "[]"
    :return: note objects as list. Example: []
    :raises ValueError: when text is not an array of notes. Example: "not-json"
    """
    return [_note_payload(note) for note in _load_notes(text)]


def _reasoning_text(connection: sqlite3.Connection, task_id: str) -> str | None:
    """
    Read the reasoning column, or None when the pin is missing.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: reasoning text or None as str | None. Example: "[]"
    """
    row = connection.execute("SELECT reasoning FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return None
    return str(row["reasoning"])


def _write_reasoning(connection: sqlite3.Connection, task_id: str, text: str) -> None:
    """
    Replace the reasoning column and commit.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param text: reasoning JSON as str. Example: "[]"
    :return: None. Example: None
    """
    connection.execute("UPDATE tasks SET reasoning = ? WHERE id = ?", (text, task_id))
    connection.commit()

