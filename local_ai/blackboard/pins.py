"""Pin rows. Worker appends reasoning. DispatcherService reads and finishes pins."""

import json
import logging
import sqlite3

from local_ai.contracts import AccessGrant, ReasoningNote, TaskPin
from local_ai.db.database import Database


logger = logging.getLogger(__name__)

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
        """
        # Select the tasks row. Parse access_json and reasoning JSON into the dataclass.
        # A missing id returns None. The caller decides not_found.
        # Log the task_id and status at INFO.
        raise NotImplementedError

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
        """
        # If note is None or its text is empty, return without writing.
        # Load the current reasoning JSON. If the new note cannot be stored, keep the previous text.
        # If the stored text would pass REASONING_CAP, drop the oldest notes and keep a truncation marker.
        # Log blackboard.append with task_id at INFO. Do not log the note body.
        raise NotImplementedError

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
        """
        # SELECT status='queued' ORDER BY priority ASC, created_at ASC.
        # Log the count at INFO.
        raise NotImplementedError


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
