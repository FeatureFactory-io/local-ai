"""Durable steps. Worker advances them. DispatcherService reads them for the pin response."""

import sqlite3

from local_ai.contracts import InvalidStepTransition, PlanStep, StepDraft
from local_ai.db.database import Database


class PlanStore:
    """
    Step state machine.

    Called by Worker and DispatcherService.read_pin. Calls Database only.
    Illegal jumps raise InvalidStepTransition.
    """

    illegal_transition = InvalidStepTransition

    def __init__(self, database: Database) -> None:
        """
        Store the shared database.

        :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
        """
        self._database = database

    def insert_steps(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        steps: tuple[StepDraft, ...],
    ) -> None:
        """
        Insert pending steps for a new plan.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param steps: proposed steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :return: None. Example: None
        """
        # Insert each draft as status pending. Orders must be unique for the task.
        # Log task_id and the count at INFO.
        raise NotImplementedError

    def list_steps(self, connection: sqlite3.Connection, task_id: str) -> tuple[PlanStep, ...]:
        """
        Load steps in order.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: steps as tuple[PlanStep, ...]. Example: ()
        """
        # SELECT by task_id ORDER BY step_order.
        raise NotImplementedError

    def mark_running(self, connection: sqlite3.Connection, task_id: str, order: int) -> None:
        """
        Move a pending or waiting_retry step to running.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: None. Example: None
        :raises InvalidStepTransition: when the step is completed. Example: order 1 already completed
        """
        # Allowed sources: pending, waiting_retry.
        # completed, failed, and running raise InvalidStepTransition.
        # Log task_id and order at INFO.
        raise NotImplementedError

    def complete_step(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        order: int,
        synthesis: str,
        result_json: str,
    ) -> None:
        """
        Mark a running step completed and store the handoff.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 1
        :param synthesis: next-step handoff as str. Example: "Found ch-01.md through ch-12.md"
        :param result_json: tool envelope as str. Example: "{\\"success\\": true}"
        :return: None. Example: None
        :raises InvalidStepTransition: when the step is not running. Example: a pending step
        """
        # Set status completed, llm_synthesis, and result_json.
        # Log task_id and order at INFO.
        raise NotImplementedError

    def fail_step(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        order: int,
        error: str,
    ) -> None:
        """
        Mark a running step failed. A completed step is left completed.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param error: step error as str. Example: "weasyprint: CSS path missing"
        :return: None. Example: None
        :raises InvalidStepTransition: when the step is pending. Example: order 3 still pending
        """
        # If the row is already completed, return without writing.
        # If the row is running, set status failed and store error.
        # Log task_id, order, and that the step failed at INFO. Do not log the full stderr.
        raise NotImplementedError

    def insert_after(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        after_order: int,
        action: str,
    ) -> int:
        """
        Insert a pending step after a failed one and shift later pending orders.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param after_order: failed step order as int. Example: 2
        :param action: new step sentence as str. Example: "Fix the CSS path and run weasyprint again"
        :return: new order as int. Example: 3
        """
        # Shift pending and waiting_retry rows with order greater than after_order.
        # Leave the failed row failed.
        # Insert the new row as pending at after_order + 1.
        # Log task_id and the new order at INFO.
        raise NotImplementedError

    def cancel_pending(self, connection: sqlite3.Connection, task_id: str) -> int:
        """
        Cancel every still-pending step. Completed and failed rows stay.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: number cancelled as int. Example: 1
        """
        # Set pending and waiting_retry rows to failed with error "cancelled".
        # Log task_id and the count at INFO.
        raise NotImplementedError

    def render_plan(self, steps: tuple[PlanStep, ...]) -> str:
        """
        Render steps as the pin's current_plan text.

        :param steps: ordered steps as tuple[PlanStep, ...]. Example: (PlanStep("id", 1, "List chapter files", "completed"),)
        :return: one block as str. Example: "1. List chapter files (completed)."
        """
        # One line per step: "{order}. {action} ({status})."
        # Pure. No database call.
        raise NotImplementedError
