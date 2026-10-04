"""Single worker slot. Starts Worker.run_task and then DispatcherService.finish."""

import logging
import sqlite3
import threading

from local_ai.contracts import WorkerRunner
from local_ai.dispatcher.service import DispatcherService


logger = logging.getLogger(__name__)

_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_FAILED_RETURN = "run_task returned failed"
_NO_ARTIFACTS: tuple[str, ...] = ()


class Slot:
    """
    One running worker. A second kick leaves the next pin queued.

    Called by DispatcherService's MCP wrapper after submit returns its snapshot,
    and again from finish so the following pin can start.
    """

    def __init__(
        self,
        service: DispatcherService,
        runner: WorkerRunner,
        connection: sqlite3.Connection,
        clock,
    ) -> None:
        """
        Store the service and the runner.

        :param service: pin service as DispatcherService. Example: the composed service
        :param runner: worker as WorkerRunner. Example: Worker(...)
        :param connection: shared connection as sqlite3.Connection. Example: the Database connection
        :param clock: returns a UTC timestamp as a zero-arg callable. Example: lambda: "2026-10-04T18:40:02Z"
        """
        self._service = service
        self._runner = runner
        self._connection = connection
        self._clock = clock
        self._thread: threading.Thread | None = None

    def kick(self) -> None:
        """
        Start the head pin when the slot is free.

        A live worker thread returns before claim, so a second pin stays queued.
        An empty queue returns without starting a thread.

        :return: None. Example: None
        """
        logger.info("Starting kick")
        if self._slot_busy():
            logger.info("dispatcher.slot | busy")
            return
        pin = self._service.claim_next(self._connection, self._timestamp())
        if pin is None:
            logger.info("dispatcher.slot | empty")
            return
        self._start_worker(pin.task_id)

    def _slot_busy(self) -> bool:
        """
        True when the worker thread is still running.

        :return: whether the slot is taken as bool. Example: False
        """
        return self._thread is not None and self._thread.is_alive()

    def _timestamp(self) -> str:
        """
        Read one UTC timestamp from the clock.

        :return: timestamp as str. Example: "2026-10-04T18:40:02Z"
        :raises TypeError: when the clock does not return a string. Example: None
        """
        stamp = self._clock()
        if not isinstance(stamp, str) or not stamp.strip():
            raise TypeError("clock must return a non-empty string. Example: '2026-10-04T18:40:02Z'")
        return stamp

    def _start_worker(self, task_id: str) -> None:
        """
        Start a daemon thread that calls _run for the claimed pin.

        :param task_id: claimed pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: None. Example: None
        """
        logger.info("dispatcher.slot | kick | task_id=%s", task_id)
        worker = threading.Thread(target=self._run, args=(task_id,), daemon=True, name=f"slot-{task_id}")
        self._thread = worker
        worker.start()

    def _run(self, task_id: str) -> None:
        """
        Run the worker and finish the pin.

        :param task_id: claimed pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: None. Example: None
        """
        logger.info("Starting _run task_id=%s", task_id)
        try:
            status, error = self._worker_outcome(task_id)
            self._finish_pin(task_id, status, error)
        finally:
            self._release_slot()

    def _worker_outcome(self, task_id: str) -> tuple[str, str]:
        """
        Run the task. An exception becomes status failed and the exception text.

        :param task_id: claimed pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: status and error as tuple[str, str]. Example: ("completed", "")
        """
        try:
            status = self._runner.run_task(task_id)
        except Exception as exc:
            logger.info("dispatcher.slot | worker_raised | task_id=%s", task_id)
            return _STATUS_FAILED, _exception_text(exc)
        logger.info("dispatcher.slot | worker_returned | task_id=%s status=%s", task_id, status)
        return _status_and_error(status, _failure_reason(self._runner))

    def _finish_pin(self, task_id: str, status: str, error: str) -> None:
        """
        Write the terminal pin using the slot clock. Artifacts stay empty until the worker declares them.

        :param task_id: claimed pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param status: completed or failed as str. Example: "completed"
        :param error: failure summary as str. Example: ""
        :return: None. Example: None
        """
        self._service.finish(task_id, status, error, _NO_ARTIFACTS, self._timestamp(), self._connection)

    def _release_slot(self) -> None:
        """
        Drop the thread marker and start the next queued pin when one exists.

        :return: None. Example: None
        """
        self._thread = None
        self.kick()


def _status_and_error(status: str, reason: str = "") -> tuple[str, str]:
    """
    Accept completed or failed. Anything else is stored as failed.

    A failed return stores the worker's failure_reason when it set one.

    :param status: worker result as str. Example: "failed"
    :param reason: worker failure_reason as str. Example: "empty_plan"
    :return: status and error as tuple[str, str]. Example: ("failed", "empty_plan")
    """
    if status == _STATUS_COMPLETED:
        return _STATUS_COMPLETED, ""
    if status == _STATUS_FAILED:
        text = reason.strip() or _FAILED_RETURN
        return _STATUS_FAILED, text
    return _STATUS_FAILED, f"unexpected worker status {status!r}"


def _failure_reason(runner: object) -> str:
    """
    Read failure_reason when the runner recorded one.

    :param runner: worker as object. Example: Worker(...)
    :return: reason as str. Example: "empty_plan"
    """
    reason = getattr(runner, "failure_reason", "")
    if not isinstance(reason, str):
        return ""
    return reason


def _exception_text(exc: Exception) -> str:
    """
    Use the exception message. A blank message keeps the class name.

    :param exc: worker failure as Exception. Example: ScriptExhausted("script has no turn left")
    :return: error text as str. Example: "script has no turn left at index=0 model_name=qwen-14b"
    """
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__
