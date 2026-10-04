"""Durable steps. Worker advances them. DispatcherService reads them for the pin response."""

import logging
import sqlite3

from local_ai.contracts import InvalidStepTransition, PlanStep, StepDraft
from local_ai.db.database import Database

logger = logging.getLogger(__name__)

_STATUS_PENDING = "pending"
_STATUS_RUNNING = "running"
_STATUS_WAITING_RETRY = "waiting_retry"
_MARK_RUNNING_SOURCES = frozenset({_STATUS_PENDING, _STATUS_WAITING_RETRY})
_MIN_STEP_ORDER = 1


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
        :raises TypeError: when steps is not a tuple of StepDraft. Example: ["List chapter files"]
        :raises ValueError: when orders repeat or an action is empty. Example: (StepDraft(1, "List"), StepDraft(1, "Again"))
        :raises sqlite3.IntegrityError: when the parent pin is missing. Example: task_id "missing-task"
        """
        self._validate_insert_steps(connection, task_id, steps)
        self._write_pending_steps(connection, task_id, steps)
        logger.info("PlanStore.insert_steps task_id=%s count=%s", task_id, len(steps))

    def list_steps(self, connection: sqlite3.Connection, task_id: str) -> tuple[PlanStep, ...]:
        """
        Load steps in order.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: steps as tuple[PlanStep, ...]. Example: ()
        :raises TypeError: when connection is not sqlite3.Connection. Example: None
        :raises ValueError: when task_id is blank. Example: ""
        """
        self._require_connection(connection)
        self._require_task_id(task_id)
        steps = tuple(self._plan_step_from_row(row) for row in self._select_steps(connection, task_id))
        logger.info("PlanStore.list_steps task_id=%s count=%s", task_id, len(steps))
        return steps

    def mark_running(self, connection: sqlite3.Connection, task_id: str, order: int) -> None:
        """
        Move a pending or waiting_retry step to running.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: None. Example: None
        :raises InvalidStepTransition: when the step is completed, failed, running, or missing. Example: order 1 already completed
        :raises ValueError: when order is below 1. Example: 0
        """
        self._require_step_address(connection, task_id, order)
        self._require_mark_running_source(connection, task_id, order)
        self._set_status(connection, task_id, order, _STATUS_RUNNING)
        logger.info("PlanStore.mark_running task_id=%s order=%s", task_id, order)

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
        :raises TypeError: when steps is not a tuple of PlanStep. Example: ["List chapter files"]
        """
        self._validate_plan_steps(steps)
        return "\n".join(self._render_step_line(step) for step in steps)

    def _validate_plan_steps(self, steps: tuple[PlanStep, ...]) -> None:
        """
        Reject anything other than a tuple of PlanStep.

        :param steps: ordered steps as tuple[PlanStep, ...]. Example: ()
        :return: None. Example: None
        :raises TypeError: when steps is not a tuple of PlanStep. Example: ["List chapter files"]
        """
        if not isinstance(steps, tuple):
            raise TypeError(f"steps must be a tuple, got {type(steps).__name__}")
        for step in steps:
            if not isinstance(step, PlanStep):
                raise TypeError(f"steps must contain PlanStep, got {type(step).__name__}")

    def _validate_insert_steps(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        steps: tuple[StepDraft, ...],
    ) -> None:
        """
        Reject drafts the plan_steps table cannot store.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param steps: proposed steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :return: None. Example: None
        :raises TypeError: when steps is not a tuple of StepDraft. Example: ["List chapter files"]
        :raises ValueError: when orders repeat or an action is empty. Example: (StepDraft(1, "List"), StepDraft(1, "Again"))
        """
        self._require_connection(connection)
        self._require_task_id(task_id)
        self._require_step_drafts(steps)
        self._require_unique_new_orders(connection, task_id, steps)

    def _require_connection(self, connection: sqlite3.Connection) -> None:
        """
        Reject an object that is not an open sqlite connection.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: None. Example: None
        :raises TypeError: when connection is not sqlite3.Connection. Example: None
        """
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError(f"connection must be sqlite3.Connection, got {type(connection).__name__}")

    def _require_task_id(self, task_id: str) -> None:
        """
        Reject a blank or non-string pin id.

        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: None. Example: None
        :raises ValueError: when task_id is blank. Example: ""
        :raises TypeError: when task_id is not a str. Example: None
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be str, got {type(task_id).__name__}")
        if not task_id.strip():
            raise ValueError(f"task_id must be a non-empty str, got {task_id!r}")

    def _require_step_drafts(self, steps: tuple[StepDraft, ...]) -> None:
        """
        Reject anything other than a tuple of StepDraft with a usable order and action.

        :param steps: proposed steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :return: None. Example: None
        :raises TypeError: when steps is not a tuple of StepDraft. Example: ["List chapter files"]
        :raises ValueError: when an order is below 1 or an action is empty. Example: StepDraft(0, "List")
        """
        if not isinstance(steps, tuple):
            raise TypeError(f"steps must be a tuple, got {type(steps).__name__}")
        for step in steps:
            self._require_step_draft(step)

    def _require_step_draft(self, step: StepDraft) -> None:
        """
        Reject one draft the table cannot store.

        :param step: one draft as StepDraft. Example: StepDraft(1, "List chapter files")
        :return: None. Example: None
        :raises TypeError: when step is not a StepDraft. Example: "List chapter files"
        :raises ValueError: when order is below 1 or action is empty. Example: StepDraft(0, "List")
        """
        if not isinstance(step, StepDraft):
            raise TypeError(f"steps must contain StepDraft, got {type(step).__name__}")
        if isinstance(step.order, bool) or not isinstance(step.order, int) or step.order < _MIN_STEP_ORDER:
            raise ValueError(f"step order must be an int >= {_MIN_STEP_ORDER}, got {step.order!r}")
        if not isinstance(step.action, str) or not step.action.strip():
            raise ValueError(f"step action must be a non-empty str, got {step.action!r}")

    def _require_unique_new_orders(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        steps: tuple[StepDraft, ...],
    ) -> None:
        """
        Reject orders that repeat in the batch or already exist for the task.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param steps: proposed steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :return: None. Example: None
        :raises ValueError: when an order is repeated. Example: (StepDraft(1, "List"), StepDraft(1, "Again"))
        """
        orders = [step.order for step in steps]
        if len(orders) != len(set(orders)):
            raise ValueError(f"step orders must be unique, got {orders}")
        self._require_orders_absent(connection, task_id, orders)

    def _require_orders_absent(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        orders: list[int],
    ) -> None:
        """
        Reject an order this task already stored.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param orders: candidate orders as list[int]. Example: [1, 2]
        :return: None. Example: None
        :raises ValueError: when an order is already stored. Example: 1
        """
        if not orders:
            return
        placeholders = ", ".join("?" for _ in orders)
        row = connection.execute(
            f"SELECT step_order FROM plan_steps WHERE task_id = ? AND step_order IN ({placeholders})",
            (task_id, *orders),
        ).fetchone()
        if row is not None:
            raise ValueError(f"step order {int(row['step_order'])} already exists for task_id={task_id}")

    def _require_step_address(self, connection: sqlite3.Connection, task_id: str, order: int) -> None:
        """
        Reject a connection, pin, or order the state machine cannot address.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: None. Example: None
        :raises TypeError: when connection is not sqlite3.Connection. Example: None
        :raises ValueError: when task_id is blank or order is below 1. Example: 0
        """
        self._require_connection(connection)
        self._require_task_id(task_id)
        self._require_order(order)

    def _require_order(self, order: int) -> None:
        """
        Reject an order that is not an int of at least 1.

        :param order: step order as int. Example: 2
        :return: None. Example: None
        :raises ValueError: when order is below 1. Example: 0
        :raises TypeError: when order is not an int. Example: "2"
        """
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError(f"order must be int, got {type(order).__name__}")
        if order < _MIN_STEP_ORDER:
            raise ValueError(f"order must be an int >= {_MIN_STEP_ORDER}, got {order}")

    def _require_mark_running_source(self, connection: sqlite3.Connection, task_id: str, order: int) -> None:
        """
        Allow only pending and waiting_retry to become running.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: None. Example: None
        :raises InvalidStepTransition: when the step is completed, failed, running, or missing. Example: order 1 already completed
        """
        status = self._require_step_status(connection, task_id, order)
        if status not in _MARK_RUNNING_SOURCES:
            logger.info(
                "PlanStore.mark_running rejected task_id=%s order=%s status=%s",
                task_id,
                order,
                status,
            )
            raise InvalidStepTransition(
                f"cannot mark_running from status={status} task_id={task_id} order={order}"
            )

    def _require_step_status(self, connection: sqlite3.Connection, task_id: str, order: int) -> str:
        """
        Return the current status of one step.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :return: status as str. Example: "pending"
        :raises InvalidStepTransition: when that order is missing. Example: order 9
        """
        row = connection.execute(
            "SELECT status FROM plan_steps WHERE task_id = ? AND step_order = ?",
            (task_id, order),
        ).fetchone()
        if row is None:
            raise InvalidStepTransition(f"no step task_id={task_id} order={order}")
        return str(row["status"])

    def _set_status(self, connection: sqlite3.Connection, task_id: str, order: int, status: str) -> None:
        """
        Write one step's status.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param order: step order as int. Example: 2
        :param status: next status as str. Example: "running"
        :return: None. Example: None
        """
        connection.execute(
            "UPDATE plan_steps SET status = ? WHERE task_id = ? AND step_order = ?",
            (status, task_id, order),
        )

    def _select_steps(self, connection: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
        """
        Load one pin's steps from lowest order to highest.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: rows as list[sqlite3.Row]. Example: []
        """
        return list(
            connection.execute(
                """
                SELECT task_id, step_order, action, status, result_json, error, llm_synthesis
                FROM plan_steps
                WHERE task_id = ?
                ORDER BY step_order
                """,
                (task_id,),
            ).fetchall()
        )

    def _plan_step_from_row(self, row: sqlite3.Row) -> PlanStep:
        """
        Map one plan_steps row onto PlanStep.

        :param row: sqlite row as sqlite3.Row. Example: a row with step_order 1
        :return: step as PlanStep. Example: PlanStep("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", 1, "List chapter files", "pending")
        """
        return PlanStep(
            task_id=str(row["task_id"]),
            order=int(row["step_order"]),
            action=str(row["action"]),
            status=str(row["status"]),
            result_json=str(row["result_json"]),
            error=str(row["error"]),
            llm_synthesis=str(row["llm_synthesis"]),
        )

    def _write_pending_steps(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        steps: tuple[StepDraft, ...],
    ) -> None:
        """
        Insert each draft as status pending.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param steps: proposed steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
        :return: None. Example: None
        :raises sqlite3.IntegrityError: when the parent pin is missing. Example: task_id "missing-task"
        """
        connection.executemany(
            """
            INSERT INTO plan_steps (task_id, step_order, action, status)
            VALUES (?, ?, ?, ?)
            """,
            [(task_id, step.order, step.action, _STATUS_PENDING) for step in steps],
        )

    def _render_step_line(self, step: PlanStep) -> str:
        """
        Format one step as a single line.

        :param step: one step as PlanStep. Example: PlanStep("id", 1, "List chapter files", "completed")
        :return: one line as str. Example: "1. List chapter files (completed)."
        """
        return f"{step.order}. {step.action} ({step.status})."
