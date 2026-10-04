"""Single worker slot. Starts Worker.run_task and then DispatcherService.finish."""

import sqlite3
import threading

from local_ai.contracts import WorkerRunner
from local_ai.dispatcher.service import DispatcherService


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

        :return: None. Example: None
        """
        # If self._thread is alive, return. The head pin stays queued.
        # DispatcherService.claim_next. None means the queue is empty: return.
        # Start a daemon thread that calls _run so submit's queued snapshot is not blocked
        # on the worker. Log the task_id at INFO.
        raise NotImplementedError

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
