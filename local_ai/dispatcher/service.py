"""Queue, pin updates, and the single-owner claim. Does not import the worker package."""

import logging
import shutil
import sqlite3
from pathlib import Path

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import SubmitRequest, TaskPin
from local_ai.db.database import Database
from local_ai.plan.steps import PlanStore


logger = logging.getLogger(__name__)

_GRANT_MODES = ("job", "paths", "home")
_JOB_MODE = "job"


class DispatcherService:
    """
    Pins tasks and answers read and reprioritize.

    Collaborators: Database, Blackboard, PlanStore, CapacityReader.
    The slot calls claim_next and finish. This class does not start the worker.
    """

    def __init__(
        self,
        database: Database,
        blackboard: Blackboard,
        plan: PlanStore,
        capacity: CapacityReader,
        jobs_root: Path,
    ) -> None:
        """
        Store collaborators.

        :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
        :param blackboard: pin store as Blackboard. Example: Blackboard(database)
        :param plan: step store as PlanStore. Example: PlanStore(database)
        :param capacity: memory and Ollama reader as CapacityReader. Example: CapacityReader()
        :param jobs_root: job directory root as Path. Example: Path("data/jobs")
        """
        self._database = database
        self._blackboard = blackboard
        self._plan = plan
        self._capacity = capacity
        self._jobs_root = jobs_root

    def submit(self, request: SubmitRequest, connection: sqlite3.Connection) -> dict:
        """
        Pin a task and return the queued snapshot. The slot kicks after this returns the dict.

        :param request: submit arguments as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: queued or rejected body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "position": 1}
        """
        # Call CapacityReader.read. ollama down returns status rejected, reason ollama_down, task_id null.
        # validate_priority, default_grant. A missing model uses recommended. recommended "none" rejects.
        # validate_model. copy_job_inputs. Blackboard.insert_pin with status queued.
        # queue_position. Log dispatcher.submit with task_id, priority, model, and grant mode.
        # Return the queued snapshot. Do not call the worker here.
        raise NotImplementedError

    def copy_job_inputs(self, task_id: str, inputs: tuple[str, ...], grant_mode: str) -> Path:
        """
        Create the job folder and copy declared files when the grant is job.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param inputs: source files as tuple[str, ...]. Example: ()
        :param grant_mode: job, paths, or home as str. Example: "job"
        :return: job directory as Path. Example: Path("data/jobs/7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
        :raises ValueError: when task_id is blank or grant_mode is unknown. Example: "root"
        :raises TypeError: when job-mode inputs is not a tuple of strings. Example: ["/tmp/notes.md"]
        """
        logger.info("Starting copy_job_inputs task_id=%s grant_mode=%s", task_id, grant_mode)
        _require_grant_mode(grant_mode)
        directory = _create_job_directory(self._jobs_root, task_id)
        if grant_mode == _JOB_MODE:
            _copy_job_files(inputs, directory)
        else:
            logger.info("dispatcher.copy_job_inputs | skip_copy | grant=%s", grant_mode)
        logger.info("dispatcher.copy_job_inputs | directory=%s grant=%s", directory, grant_mode)
        return directory

    def read_pin(self, task_id: str, connection: sqlite3.Connection) -> dict:
        """
        Return one pin, including live reasoning and steps.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: pin body as dict. Example: {"status": "running", "phase": "acting", "steps": []}
        """
        # Blackboard.get_pin. None returns status rejected, reason not_found.
        # PlanStore.list_steps. Include goal, access, current_plan, reasoning, artifacts, and error.
        # Log task_id and status at INFO.
        raise NotImplementedError

    def reprioritize(self, task_id: str, priority: int, connection: sqlite3.Connection) -> dict:
        """
        Change priority of a queued pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 5
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: queued or rejected body as dict. Example: {"status": "queued", "priority": 5, "position": 1}
        """
        # validate_priority. Blackboard.get_pin.
        # status running, completed, or failed returns rejected, reason not_queued, and does not write.
        # status queued calls Blackboard.update_priority and queue_position.
        # Log the branch at INFO.
        raise NotImplementedError

    def claim_next(self, connection: sqlite3.Connection, started_at: str) -> TaskPin | None:
        """
        Claim the head queued pin. A lost race returns None.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param started_at: dispatch time as str. Example: "2026-10-04T18:40:02Z"
        :return: claimed pin or None as TaskPin | None. Example: None
        """
        # Blackboard.list_queued. The first row is the head.
        # Blackboard.claim. If it returns False, another caller took it: return None.
        # Log the task_id at INFO when a pin is claimed.
        raise NotImplementedError

    def recover_running(self, connection: sqlite3.Connection) -> tuple[str, ...]:
        """
        On process start, put running pins back on the queue.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: requeued ids as tuple[str, ...]. Example: ()
        """
        # Blackboard.requeue_running. Completed plan steps stay completed.
        # Log each task_id at INFO.
        raise NotImplementedError

    def finish(
        self,
        task_id: str,
        status: str,
        error: str,
        artifacts: tuple[str, ...],
        finished_at: str,
        connection: sqlite3.Connection,
    ) -> None:
        """
        Write the terminal status and release the pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param status: completed or failed as str. Example: "failed"
        :param error: failure summary as str. Example: "three identical shell failures"
        :param artifacts: output paths as tuple[str, ...]. Example: ()
        :param finished_at: stop time as str. Example: "2026-10-04T18:55:00Z"
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: None. Example: None
        """
        # Blackboard.set_final. PlanStore.cancel_pending when status is failed.
        # Log dispatcher.finish with task_id, status, and the artifact count at INFO.
        raise NotImplementedError


def _require_grant_mode(grant_mode: str) -> None:
    """
    Accept job, paths, or home.

    :param grant_mode: grant mode as str. Example: "job"
    :return: None. Example: None
    :raises ValueError: when grant_mode is unknown. Example: "root"
    """
    if grant_mode not in _GRANT_MODES:
        raise ValueError(
            f"grant_mode must be job, paths, or home. Example: 'job'; got {grant_mode!r}"
        )


def _create_job_directory(jobs_root: Path, task_id: str) -> Path:
    """
    Make jobs_root/task_id. The id must be one path segment.

    :param jobs_root: job directory root as Path. Example: Path("data/jobs")
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: job directory as Path. Example: Path("data/jobs/7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
    :raises TypeError: when jobs_root is not a Path. Example: "data/jobs"
    :raises ValueError: when task_id is blank or contains a separator. Example: "  "
    """
    checked = _require_task_id(task_id)
    if not isinstance(jobs_root, Path):
        raise TypeError("jobs_root must be Path. Example: Path('data/jobs')")
    directory = jobs_root / checked
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _require_task_id(task_id: str) -> str:
    """
    Reject a blank id or one that could leave the job root.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: the same id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :raises ValueError: when task_id is blank or not a single segment. Example: "../other"
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(
            "task_id must be a non-empty string. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'"
        )
    if "/" in task_id or "\\" in task_id or task_id in {".", ".."}:
        raise ValueError(
            "task_id must be a single path segment. Example: '7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f'"
        )
    return task_id


def _copy_job_files(inputs: tuple[str, ...], directory: Path) -> None:
    """
    Copy existing files into the job directory. Directories are left in place.

    :param inputs: source paths as tuple[str, ...]. Example: ("/tmp/notes.md",)
    :param directory: job directory as Path. Example: Path("data/jobs/7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
    :return: None. Example: None
    :raises TypeError: when inputs is not a tuple. Example: ["/tmp/notes.md"]
    :raises ValueError: when an entry is not a non-empty string. Example: ""
    """
    if not isinstance(inputs, tuple):
        raise TypeError("inputs must be a tuple of paths. Example: ()")
    for raw in inputs:
        _copy_one_file(raw, directory)


def _copy_one_file(raw: object, directory: Path) -> None:
    """
    Copy one existing file. A missing path or a directory is skipped.

    :param raw: source path as object. Example: "/tmp/notes.md"
    :param directory: job directory as Path. Example: Path("data/jobs/7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
    :return: None. Example: None
    :raises ValueError: when raw is not a non-empty string. Example: ""
    """
    if not isinstance(raw, str) or raw == "":
        raise ValueError("input path must be a non-empty string. Example: '/tmp/notes.md'")
    source = Path(raw)
    if not source.is_file():
        logger.info("dispatcher.copy_job_inputs | skip | reason=not_a_file name=%s", source.name)
        return
    shutil.copy2(source, directory / source.name)
    logger.info("dispatcher.copy_job_inputs | copied | name=%s", source.name)
