"""Pin rows. Worker appends reasoning. DispatcherService reads and finishes pins."""

import json
import logging
import sqlite3

from local_ai.contracts import AccessGrant, ReasoningNote, TaskPin
from local_ai.db.database import Database


logger = logging.getLogger(__name__)

_QUEUED = "queued"

_LIST_QUEUED_SQL = """
SELECT * FROM tasks
WHERE status = ?
ORDER BY priority ASC, created_at ASC
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
        """
        # Update phase and current_plan on that row.
        # Log task_id and phase at INFO.
        raise NotImplementedError

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
        """
        # Set status, phase, final_status, error, artifacts_json, and finished_at together.
        # Log task_id and status at INFO.
        raise NotImplementedError

    def update_priority(self, connection: sqlite3.Connection, task_id: str, priority: int) -> None:
        """
        Change priority of a queued pin. The caller has already checked the status.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 10
        :return: None. Example: None
        """
        # Update priority only. Do not change status.
        # Log task_id and priority at INFO.
        raise NotImplementedError

    def claim(self, connection: sqlite3.Connection, task_id: str, started_at: str) -> bool:
        """
        Move one queued pin to running. A second claim returns False.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param started_at: dispatch time as str. Example: "2026-10-04T18:40:02Z"
        :return: whether this call won the row as bool. Example: True
        """
        # UPDATE tasks SET status='running', started_at=? WHERE id=? AND status='queued'.
        # Return True only when the update changed one row.
        # Log the task_id and the boolean at INFO.
        raise NotImplementedError

    def requeue_running(self, connection: sqlite3.Connection) -> tuple[str, ...]:
        """
        Return every running pin to queued and clear started_at.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: requeued ids as tuple[str, ...]. Example: ("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f",)
        """
        # Select ids where status is running. Set those rows to queued, phase queued, started_at NULL.
        # Do not touch plan_steps. Completed steps stay completed.
        # Log each task_id at INFO.
        raise NotImplementedError

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
    _write_reasoning(connection, task_id, json.dumps(notes))


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

