"""Single worker slot. Starts Worker.run_task and then DispatcherService.finish."""

import logging
import sqlite3
import threading

from local_ai.contracts import WorkerRunner
from local_ai.dispatcher.service import DispatcherService


logger = logging.getLogger(__name__)


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
        # WorkerRunner.run_task. It returns "completed" or "failed".
        # DispatcherService.finish with that status.
        # Clear the thread marker and call kick so the next queued pin can start.
        # An exception from the worker still calls finish with status failed and the exception text.
        # Log dispatcher.finish is inside DispatcherService.finish.
        raise NotImplementedError
