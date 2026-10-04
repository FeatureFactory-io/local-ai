"""Queue, pin updates, and the single-owner claim. Does not import the worker package."""

import logging
import shutil
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import (
    OLLAMA_DOWN,
    RECOMMENDATION_NONE,
    CapacityReader,
    HttpReader,
    MemoryReader,
    PressureReader,
)
from local_ai.contracts import (
    AccessGrant,
    InvalidModel,
    PlanStep,
    ReasoningNote,
    SubmitRequest,
    TaskPin,
    default_grant,
    validate_model,
    validate_priority,
)
from local_ai.db.database import Database
from local_ai.dispatcher.queue import queue_position
from local_ai.plan.steps import PlanStore


logger = logging.getLogger(__name__)

_GRANT_MODES = ("job", "paths", "home")
_JOB_MODE = "job"
_STATUS_QUEUED = "queued"
_STATUS_RUNNING = "running"
_STATUS_FAILED = "failed"
_PHASE_QUEUED = "queued"
_REASON_OLLAMA_DOWN = "ollama_down"
_REASON_MODEL_UNAVAILABLE = "model_unavailable"
_STATUS_REJECTED = "rejected"
_REASON_NOT_FOUND = "not_found"
_REASON_NOT_QUEUED = "not_queued"


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
        memory: MemoryReader | None = None,
        ollama: HttpReader | None = None,
        pressure: PressureReader | None = None,
        ollama_host: str | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """
        Store collaborators.

        :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
        :param blackboard: pin store as Blackboard. Example: Blackboard(database)
        :param plan: step store as PlanStore. Example: PlanStore(database)
        :param capacity: memory and Ollama reader as CapacityReader. Example: CapacityReader()
        :param jobs_root: job directory root as Path. Example: Path("data/jobs")
        :param memory: returns (total_gb, free_gb) as MemoryReader | None. Example: lambda: (64.0, 40.2)
        :param ollama: HTTP reader as HttpReader | None. Example: lambda path: (200, {"models": []})
        :param pressure: pressure reader as PressureReader | None. Example: lambda: "normal"
        :param ollama_host: base URL as str | None. Example: "http://127.0.0.1:11434"
        :param clock: UTC timestamp source as Callable[[], str] | None. Example: lambda: "2026-10-04T18:40:00Z"
        """
        self._database = database
        self._blackboard = blackboard
        self._plan = plan
        self._capacity = capacity
        self._jobs_root = jobs_root
        self._memory = memory
        self._ollama = ollama
        self._pressure = pressure
        self._ollama_host = ollama_host
        self._clock = _utc_timestamp if clock is None else clock

    def submit(self, request: SubmitRequest, connection: sqlite3.Connection) -> dict:
        """
        Pin a task and return the queued snapshot. The slot kicks after this returns the dict.

        Ollama down returns status rejected, reason ollama_down, and task_id null.
        Recommended none with no model returns status rejected and reason model_unavailable.
        Neither rejection inserts a pin. This method does not start the worker.

        :param request: submit arguments as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: queued or rejected body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "position": 1}
        :raises TypeError: when request is not a SubmitRequest. Example: {}
        :raises InvalidPriority: when priority is outside 1..100. Example: 0
        :raises InvalidModel: when the model name is unknown. Example: "qwen-70b"
        :raises InvalidAccess: when the grant mode is unknown. Example: {"mode": "root"}
        :raises RuntimeError: when the capacity ports were not stored. Example: memory=None
        """
        _require_submit_request(request)
        reading = _capacity_reading(
            self._capacity, self._memory, self._ollama, self._pressure, self._ollama_host
        )
        rejected = _capacity_rejection(reading, request.model)
        if rejected is not None:
            logger.info("dispatcher.submit | rejected | reason=%s", rejected["reason"])
            return rejected
        return self._enqueue(request, connection, reading)

    def _enqueue(self, request: SubmitRequest, connection: sqlite3.Connection, reading: dict) -> dict:
        """
        Validate, copy inputs, insert the queued pin, and return its snapshot.

        :param request: submit arguments as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param reading: capacity body as dict. Example: {"ollama": "up", "recommended": "qwen-14b"}
        :return: queued body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "position": 1}
        :raises InvalidPriority: when priority is outside 1..100. Example: 0
        :raises InvalidModel: when the model name is unknown. Example: "qwen-70b"
        :raises InvalidAccess: when the grant mode is unknown. Example: {"mode": "root"}
        :raises TypeError: when connection is not a Connection. Example: None
        """
        priority = validate_priority(request.priority)
        grant = default_grant(request.access)
        model = validate_model(_chosen_model(request.model, reading))
        _require_connection(connection)
        task_id = str(uuid.uuid4())
        pin = _queued_pin(task_id, request, priority, model, grant, _timestamp(self._clock))
        self.copy_job_inputs(task_id, request.inputs, grant.mode)
        self._blackboard.insert_pin(connection, pin)
        position = queue_position(pin, self._blackboard.list_queued(connection))
        _log_submit(task_id, priority, model, grant.mode)
        return _queued_body(pin, position)

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

        A missing id returns status rejected and reason not_found. Steps are loaded only when the pin exists.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: pin body or rejection as dict. Example: {"status": "running", "phase": "acting", "steps": []}
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id is blank. Example: "  "
        """
        pin = self._blackboard.get_pin(connection, task_id)
        steps: tuple[PlanStep, ...] = ()
        if pin is not None:
            steps = self._plan.list_steps(connection, task_id)
        body = _read_body(task_id, pin, steps)
        _log_read(task_id, str(body["status"]))
        return body

    def reprioritize(self, task_id: str, priority: int, connection: sqlite3.Connection) -> dict:
        """
        Change priority of a queued pin.

        Running, completed, and failed pins return status rejected and reason not_queued. The row is not written.
        A missing pin returns reason not_found.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 5
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: queued or rejected body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "priority": 5, "position": 1}
        :raises InvalidPriority: when priority is outside 1..100. Example: 0
        :raises TypeError: when connection is not a Connection. Example: None
        :raises ValueError: when task_id is blank. Example: "  "
        """
        checked = validate_priority(priority)
        pin = self._blackboard.get_pin(connection, task_id)
        if pin is None or pin.status != _STATUS_QUEUED:
            return _reject_reprioritize(pin, task_id)
        return _apply_queued_priority(self._blackboard, connection, pin, checked)

    def claim_next(self, connection: sqlite3.Connection, started_at: str) -> TaskPin | None:
        """
        Claim the head queued pin. A lost race returns None.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :param started_at: dispatch time as str. Example: "2026-10-04T18:40:02Z"
        :return: claimed pin or None as TaskPin | None. Example: None
        :raises ValueError: when started_at is blank. Example: " "
        :raises TypeError: when connection is not a Connection. Example: None
        """
        logger.info("Starting claim_next")
        head = _head_pin(self._blackboard.list_queued(connection))
        if head is None:
            logger.info("dispatcher.claim_next | empty")
            return None
        if not self._blackboard.claim(connection, head.task_id, started_at):
            logger.info("dispatcher.claim_next | lost | task_id=%s", head.task_id)
            return None
        logger.info("dispatcher.claim_next | claimed | task_id=%s", head.task_id)
        return replace(head, status=_STATUS_RUNNING, started_at=started_at)

    def recover_running(self, connection: sqlite3.Connection) -> tuple[str, ...]:
        """
        On process start, put running pins back on the queue.

        Completed plan steps stay completed. This method does not update plan steps.

        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: requeued ids as tuple[str, ...]. Example: ()
        :raises TypeError: when connection is not a Connection. Example: None
        """
        task_ids = self._blackboard.requeue_running(connection)
        _log_recovered(task_ids)
        return task_ids

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

        A failed status cancels pending steps. A completed status leaves them.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param status: completed or failed as str. Example: "failed"
        :param error: failure summary as str. Example: "three identical shell failures"
        :param artifacts: output paths as tuple[str, ...]. Example: ()
        :param finished_at: stop time as str. Example: "2026-10-04T18:55:00Z"
        :param connection: open connection as sqlite3.Connection. Example: the Database connection
        :return: None. Example: None
        :raises TypeError: when connection, error, or artifacts has the wrong type. Example: "/tmp/summary.md"
        :raises ValueError: when status is not completed or failed, or the pin is missing. Example: "running"
        """
        self._blackboard.set_final(connection, task_id, status, error, artifacts, finished_at)
        _cancel_pending_on_failure(self._plan, connection, task_id, status)
        _log_finish(task_id, status, len(artifacts))


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


def _utc_timestamp() -> str:
    """
    UTC time with microseconds so two submits in one second stay ordered.

    :return: timestamp as str. Example: "2026-10-04T18:40:00.000000Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _head_pin(queued: tuple[TaskPin, ...]) -> TaskPin | None:
    """
    Return the first queued pin. list_queued already applied queue order.

    :param queued: queued pins as tuple[TaskPin, ...]. Example: ()
    :return: head pin or None as TaskPin | None. Example: None
    """
    if len(queued) == 0:
        return None
    return queued[0]


def _require_submit_request(request: SubmitRequest) -> None:
    """
    Reject a value that is not a submit request.

    :param request: submit arguments as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
    :return: None. Example: None
    :raises TypeError: when request is not a SubmitRequest. Example: {}
    """
    if not isinstance(request, SubmitRequest):
        raise TypeError(
            "request must be SubmitRequest. Example: SubmitRequest('Write a brief.', 'brief.md exists')"
        )


def _require_connection(connection: sqlite3.Connection) -> None:
    """
    Reject a value that is not an open sqlite connection.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: None. Example: None
    :raises TypeError: when connection is not a Connection. Example: None
    """
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be sqlite3.Connection. Example: the Database connection")


def _capacity_reading(
    capacity: CapacityReader,
    memory: MemoryReader | None,
    ollama: HttpReader | None,
    pressure: PressureReader | None,
    ollama_host: str | None,
) -> dict:
    """
    Call CapacityReader.read with the stored ports.

    :param capacity: memory and Ollama reader as CapacityReader. Example: CapacityReader()
    :param memory: returns (total_gb, free_gb) as MemoryReader | None. Example: lambda: (64.0, 40.2)
    :param ollama: HTTP reader as HttpReader | None. Example: lambda path: (200, {"models": []})
    :param pressure: pressure reader as PressureReader | None. Example: lambda: "normal"
    :param ollama_host: base URL as str | None. Example: "http://127.0.0.1:11434"
    :return: capacity body as dict. Example: {"ollama": "up", "recommended": "qwen-14b"}
    :raises RuntimeError: when a port was not stored. Example: memory=None
    :raises TypeError: when read does not return a dict. Example: None
    """
    if memory is None or ollama is None or pressure is None or ollama_host is None:
        raise RuntimeError(
            "submit needs memory, ollama, pressure, and ollama_host. Example: CapacityReader.read ports"
        )
    reading = capacity.read(memory, ollama, pressure, ollama_host)
    if not isinstance(reading, dict):
        raise TypeError(
            "CapacityReader.read must return a dict. Example: {'ollama': 'up', 'recommended': 'qwen-14b'}"
        )
    return reading


def _capacity_rejection(reading: dict, model: str | None) -> dict | None:
    """
    Reject a down daemon, or a none recommendation when the caller omitted the model.

    :param reading: capacity body as dict. Example: {"ollama": "down", "recommended": "none"}
    :param model: requested model as str | None. Example: None
    :return: rejected body or None as dict | None. Example: {"status": "rejected", "reason": "ollama_down", "task_id": None}
    """
    if reading.get("ollama") == OLLAMA_DOWN:
        logger.info("dispatcher.submit | branch=ollama_down")
        return _rejected(_REASON_OLLAMA_DOWN)
    if reading.get("recommended") == RECOMMENDATION_NONE and model is None:
        logger.info("dispatcher.submit | branch=model_unavailable")
        return _rejected(_REASON_MODEL_UNAVAILABLE)
    return None


def _rejected(reason: str) -> dict:
    """
    Build a rejection that has no pin.

    :param reason: rejection reason as str. Example: "ollama_down"
    :return: rejected body as dict. Example: {"status": "rejected", "reason": "ollama_down", "task_id": None}
    """
    return {"status": "rejected", "reason": reason, "task_id": None}


def _chosen_model(requested: str | None, reading: dict) -> str:
    """
    Use the caller's model, or the capacity recommendation when it was omitted.

    :param requested: requested model as str | None. Example: None
    :param reading: capacity body as dict. Example: {"recommended": "qwen-14b"}
    :return: model name as str. Example: "qwen-14b"
    :raises InvalidModel: when the recommendation is missing. Example: {}
    """
    if requested is not None:
        return requested
    recommended = reading.get("recommended")
    if not isinstance(recommended, str):
        raise InvalidModel("recommended model is missing. Example: 'qwen-14b'")
    logger.info("dispatcher.submit | model omitted; using recommended=%s", recommended)
    return recommended


def _timestamp(clock: Callable[[], str]) -> str:
    """
    Read one non-empty timestamp from the clock.

    :param clock: UTC timestamp source as Callable[[], str]. Example: lambda: "2026-10-04T18:40:00Z"
    :return: timestamp as str. Example: "2026-10-04T18:40:00Z"
    :raises TypeError: when clock is not callable. Example: None
    :raises ValueError: when the clock returns a blank value. Example: "  "
    """
    if not callable(clock):
        raise TypeError("clock must be callable. Example: lambda: '2026-10-04T18:40:00Z'")
    stamp = clock()
    if not isinstance(stamp, str) or not stamp.strip():
        raise ValueError("clock must return a non-empty timestamp. Example: '2026-10-04T18:40:00Z'")
    return stamp


def _queued_pin(
    task_id: str,
    request: SubmitRequest,
    priority: int,
    model: str,
    grant: AccessGrant,
    created_at: str,
) -> TaskPin:
    """
    Build the pin inserted with status queued.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param request: submit arguments as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
    :param priority: queue rank as int. Example: 40
    :param model: chosen size as str. Example: "qwen-14b"
    :param grant: grant as AccessGrant. Example: AccessGrant(mode="job")
    :param created_at: enqueue time as str. Example: "2026-10-04T18:40:00Z"
    :return: pin as TaskPin. Example: a pin with status "queued"
    """
    return TaskPin(
        task_id=task_id,
        priority=priority,
        status=_STATUS_QUEUED,
        model=model,
        goal=request.goal,
        done_when=request.done_when,
        access=grant,
        phase=_PHASE_QUEUED,
        created_at=created_at,
    )


def _queued_body(pin: TaskPin, position: int) -> dict:
    """
    Build the snapshot returned before the slot claims the pin.

    :param pin: queued pin as TaskPin. Example: a pin with status "queued"
    :param position: one-based place as int. Example: 1
    :return: queued body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "position": 1}
    """
    return {
        "status": _STATUS_QUEUED,
        "task_id": pin.task_id,
        "priority": pin.priority,
        "model": pin.model,
        "position": position,
        "access": _access_body(pin.access),
    }


def _access_body(grant: AccessGrant) -> dict:
    """
    Match the stored grant JSON. Empty path lists are omitted.

    :param grant: grant as AccessGrant. Example: AccessGrant(mode="job")
    :return: access object as dict. Example: {"mode": "job"}
    """
    body: dict[str, object] = {"mode": grant.mode}
    if grant.paths:
        body["paths"] = list(grant.paths)
    return body


def _read_body(task_id: str, pin: TaskPin | None, steps: tuple[PlanStep, ...]) -> dict:
    """
    Build the read response. A missing pin is a not_found rejection.

    :param task_id: requested pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param pin: loaded pin as TaskPin | None. Example: None
    :param steps: ordered steps as tuple[PlanStep, ...]. Example: ()
    :return: pin body or rejection as dict. Example: {"status": "rejected", "reason": "not_found", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    """
    if pin is None:
        return _rejected_task(task_id, _REASON_NOT_FOUND)
    return _present_pin(pin, steps)


def _rejected_task(task_id: str, reason: str) -> dict:
    """
    Build a rejection that names the pin.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param reason: rejection reason as str. Example: "not_found"
    :return: rejected body as dict. Example: {"status": "rejected", "reason": "not_found", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    """
    return {"status": _STATUS_REJECTED, "reason": reason, "task_id": task_id}


def _present_pin(pin: TaskPin, steps: tuple[PlanStep, ...]) -> dict:
    """
    Map a pin and its steps onto the read contract keys.

    :param pin: loaded pin as TaskPin. Example: a pin with status "running"
    :param steps: ordered steps as tuple[PlanStep, ...]. Example: ()
    :return: pin body as dict. Example: {"status": "running", "phase": "acting", "steps": []}
    """
    return {
        "status": pin.status,
        "task_id": pin.task_id,
        "priority": pin.priority,
        "model": pin.model,
        "phase": pin.phase,
        "goal": pin.goal,
        "access": _access_body(pin.access),
        "current_plan": pin.current_plan,
        "steps": [_step_body(step) for step in steps],
        "reasoning": [_note_body(note) for note in pin.reasoning],
        "artifacts": list(pin.artifacts),
        "error": pin.error,
    }


def _step_body(step: PlanStep) -> dict:
    """
    Map one step onto the read contract keys.

    :param step: one step as PlanStep. Example: PlanStep("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", 1, "List chapter files", "completed")
    :return: step object as dict. Example: {"order": 1, "action": "List chapter files", "status": "completed", "llm_synthesis": "", "error": ""}
    """
    return {
        "order": step.order,
        "action": step.action,
        "status": step.status,
        "llm_synthesis": step.llm_synthesis,
        "error": step.error,
    }


def _note_body(note: ReasoningNote) -> dict:
    """
    Map one note onto at, step, and text.

    :param note: one note as ReasoningNote. Example: ReasoningNote("2026-10-04T18:40:01Z", 2, "Reading ch-04.md.")
    :return: note object as dict. Example: {"at": "2026-10-04T18:40:01Z", "step": 2, "text": "Reading ch-04.md."}
    """
    return {"at": note.at, "step": note.step, "text": note.text}


def _reject_reprioritize(pin: TaskPin | None, task_id: str) -> dict:
    """
    Reject a pin that is missing or not queued. Neither branch writes.

    :param pin: loaded pin as TaskPin | None. Example: None
    :param task_id: requested pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: rejected body as dict. Example: {"status": "rejected", "reason": "not_queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    """
    if pin is None:
        logger.info("dispatcher.reprioritize | not_found | task_id=%s", task_id)
        return _rejected_task(task_id, _REASON_NOT_FOUND)
    logger.info(
        "dispatcher.reprioritize | not_queued | task_id=%s status=%s",
        task_id,
        pin.status,
    )
    return _rejected_task(task_id, _REASON_NOT_QUEUED)


def _apply_queued_priority(
    blackboard: Blackboard,
    connection: sqlite3.Connection,
    pin: TaskPin,
    priority: int,
) -> dict:
    """
    Store a new priority and report the pin's place among queued pins.

    :param blackboard: pin store as Blackboard. Example: Blackboard(database)
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param pin: queued pin as TaskPin. Example: a pin with status "queued"
    :param priority: new priority as int. Example: 5
    :return: queued body as dict. Example: {"status": "queued", "task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "priority": 5, "position": 1}
    """
    blackboard.update_priority(connection, pin.task_id, priority)
    position = queue_position(replace(pin, priority=priority), blackboard.list_queued(connection))
    logger.info(
        "dispatcher.reprioritize | queued | task_id=%s priority=%s position=%s",
        pin.task_id,
        priority,
        position,
    )
    return {
        "status": _STATUS_QUEUED,
        "task_id": pin.task_id,
        "priority": priority,
        "position": position,
    }


def _cancel_pending_on_failure(
    plan: PlanStore,
    connection: sqlite3.Connection,
    task_id: str,
    status: str,
) -> None:
    """
    Cancel open steps only when the pin failed, then commit that update.

    :param plan: step store as PlanStore. Example: PlanStore(database)
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param status: completed or failed as str. Example: "failed"
    :return: None. Example: None
    """
    if status != _STATUS_FAILED:
        return
    plan.cancel_pending(connection, task_id)
    connection.commit()


def _log_finish(task_id: str, status: str, artifact_count: int) -> None:
    """
    Log the terminal finish on one INFO line.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param status: completed or failed as str. Example: "completed"
    :param artifact_count: number of output paths as int. Example: 2
    :return: None. Example: None
    """
    logger.info(
        "dispatcher.finish | exit | task_id=%s status=%s artifacts=%s",
        task_id,
        status,
        artifact_count,
    )


def _log_recovered(task_ids: tuple[str, ...]) -> None:
    """
    Log each requeued id. An empty result logs that nothing was running.

    :param task_ids: requeued ids as tuple[str, ...]. Example: ()
    :return: None. Example: None
    """
    if len(task_ids) == 0:
        logger.info("dispatcher.recover_running | empty")
        return
    for task_id in task_ids:
        logger.info("dispatcher.recover_running | requeued | task_id=%s", task_id)


def _log_read(task_id: str, status: str) -> None:
    """
    Log the read outcome on one INFO line.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param status: returned status as str. Example: "running"
    :return: None. Example: None
    """
    logger.info("dispatcher.read | entry | task_id=%s status=%s", task_id, status)


def _log_submit(task_id: str, priority: int, model: str, grant_mode: str) -> None:
    """
    Log the queued submit on one INFO line.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param priority: queue rank as int. Example: 40
    :param model: chosen size as str. Example: "qwen-14b"
    :param grant_mode: grant mode as str. Example: "paths"
    :return: None. Example: None
    """
    logger.info(
        "dispatcher.submit | entry | task_id=%s priority=%s model=%s grant=%s",
        task_id,
        priority,
        model,
        grant_mode,
    )
