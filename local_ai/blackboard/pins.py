"""Pin rows. Worker appends reasoning. DispatcherService reads and finishes pins."""

import sqlite3

from local_ai.contracts import ReasoningNote, TaskPin
from local_ai.db.database import Database


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
        """
        # Insert the tasks row. Store access as JSON and reasoning as "[]".
        # Log blackboard.insert with task_id, priority, and model at INFO.
        raise NotImplementedError

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
