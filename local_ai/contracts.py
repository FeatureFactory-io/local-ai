"""Shared types and validators. Collaborators import these and do not redefine them."""

import logging
from dataclasses import dataclass
from typing import Protocol


logger = logging.getLogger(__name__)


PRIORITY_MIN = 1
PRIORITY_MAX = 100
PRIORITY_DEFAULT = 50
REASONING_CAP = 32_000
MAX_IDENTICAL_FAILURES = 3
MAX_ROUNDS_PER_STEP = 12
SCHEMA_VERSION = 1

MODEL_NAMES = ("qwen-8b", "qwen-14b", "qwen-24b")
TASK_STATUSES = ("queued", "running", "completed", "failed")
PHASES = ("queued", "planning", "acting", "recovering", "completed", "failed")
STEP_STATUSES = ("pending", "running", "completed", "failed", "waiting_retry")


class InvalidPriority(ValueError):
    """Raised when priority is outside 1..100."""


class InvalidModel(ValueError):
    """Raised when the model name is not one of the three Qwen sizes."""


class InvalidAccess(ValueError):
    """Raised when the grant object is not job, paths, or home."""


class AccessDenied(Exception):
    """Raised before a shell process starts when a path leaves the grant."""


class InvalidStepTransition(Exception):
    """Raised when a plan step jumps to a status the machine does not allow."""


class ScriptExhausted(Exception):
    """Raised when ScriptedLLM is asked for a completion past the end of its script."""


@dataclass(frozen=True)
class AccessGrant:
    """
    Folders a job's shell may touch.

    :param mode: grant mode as str. Example: "paths"
    :param paths: allowed directories as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
    """

    mode: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolEnvelope:
    """
    Result of one shell call. Failures stay in this object.

    :param success: whether the command ran and exited 0 as bool. Example: False
    :param result: stdout as str or None. Example: "ch-01.md"
    :param error: failure code or stderr as str. Example: "access_denied"
    """

    success: bool
    result: str | None
    error: str


@dataclass(frozen=True)
class ReasoningNote:
    """
    One blackboard note.

    :param at: UTC timestamp as str. Example: "2026-10-04T18:40:01Z"
    :param step: plan step order as int. Example: 2
    :param text: note text as str. Example: "Reading ch-04.md."
    """

    at: str
    step: int
    text: str


@dataclass(frozen=True)
class ToolCall:
    """
    One model request to run the shell.

    :param name: tool name as str. Example: "shell"
    :param command: shell command as str. Example: "ls *.md"
    """

    name: str
    command: str


@dataclass(frozen=True)
class Completion:
    """
    One model turn.

    :param content: assistant text as str. Example: "List the chapter files."
    :param thinking: reasoning stream as str. Example: "The claim is in the last section."
    :param tool_calls: requested tools as tuple[ToolCall, ...]. Example: (ToolCall("shell", "ls *.md"),)
    :param final_answer: text that completes the step as str. Example: "Found ch-01.md through ch-12.md"
    """

    content: str = ""
    thinking: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: str = ""


@dataclass(frozen=True)
class StepDraft:
    """
    A step the model proposed and the plan store has not saved yet.

    :param order: execution order as int. Example: 1
    :param action: one sentence as str. Example: "List chapter files"
    """

    order: int
    action: str


@dataclass(frozen=True)
class TaskPin:
    """
    One blackboard row.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param priority: queue rank as int. Example: 40
    :param status: queued, running, completed, or failed as str. Example: "running"
    :param model: chosen size as str. Example: "qwen-14b"
    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
    :param done_when: completion condition as str. Example: "summary.md exists"
    :param access: grant as AccessGrant. Example: AccessGrant("job")
    :param phase: latest phase as str. Example: "acting"
    :param current_plan: readable plan as str. Example: "1. List chapter files (completed)."
    :param reasoning: notes newest last as tuple[ReasoningNote, ...]. Example: ()
    :param artifacts: output paths as tuple[str, ...]. Example: ("/tmp/summary.md",)
    :param error: failure summary as str. Example: ""
    :param created_at: enqueue time as str. Example: "2026-10-04T18:40:00Z"
    :param started_at: dispatch time as str or None. Example: None
    :param finished_at: stop time as str or None. Example: None
    :param final_status: completed or failed as str. Example: ""
    """

    task_id: str
    priority: int
    status: str
    model: str
    goal: str
    done_when: str
    access: AccessGrant
    phase: str
    current_plan: str = ""
    reasoning: tuple[ReasoningNote, ...] = ()
    artifacts: tuple[str, ...] = ()
    error: str = ""
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    final_status: str = ""


@dataclass(frozen=True)
class PlanStep:
    """
    One durable step.

    :param task_id: parent pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param order: execution order as int. Example: 2
    :param action: one sentence as str. Example: "Write a thesis per chapter"
    :param status: step status as str. Example: "running"
    :param result_json: tool envelope as str. Example: ""
    :param error: step error as str. Example: ""
    :param llm_synthesis: handoff to the next step as str. Example: "Found ch-01.md through ch-12.md"
    """

    task_id: str
    order: int
    action: str
    status: str
    result_json: str = ""
    error: str = ""
    llm_synthesis: str = ""


@dataclass(frozen=True)
class SubmitRequest:
    """
    Arguments of one submit call.

    :param goal: what the worker must achieve as str. Example: "Turn the notes into a one-page brief."
    :param done_when: observable completion as str. Example: "brief.md exists in the job folder."
    :param priority: 1..100 or None as int | None. Example: 40
    :param model: qwen size or None as str | None. Example: "qwen-14b"
    :param inputs: file paths to copy as tuple[str, ...]. Example: ()
    :param access: grant object or None as dict | None. Example: {"mode": "job"}
    """

    goal: str
    done_when: str
    priority: int | None = None
    model: str | None = None
    inputs: tuple[str, ...] = ()
    access: dict | None = None


class WorkerRunner(Protocol):
    """The slot calls this. DispatcherService does not import the worker package."""

    def run_task(self, task_id: str) -> str:
        """
        Run one pin to a terminal status.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: final status as str. Example: "completed"
        :raises ScriptExhausted: when a scripted model has no turn left
        """
        ...


def validate_priority(value: int | None) -> int:
    """
    Accept a queue priority or apply the default.

    :param value: requested priority as int | None. Example: 40
    :return: priority as int. Example: 40
    :raises InvalidPriority: when value is outside 1..100. Example: 0
    """
    logger.info("Starting validate_priority value=%s", value)
    if value is None:
        logger.info("Priority omitted; using default=%s", PRIORITY_DEFAULT)
        return PRIORITY_DEFAULT
    _reject_priority_unless_in_range(value)
    logger.info("Accepted priority value=%s", value)
    return value


def _reject_priority_unless_in_range(value: object) -> None:
    """
    Raise when priority is not an int inside 1..100.

    :param value: candidate priority as object. Example: 0
    :return: None. Example: None
    :raises InvalidPriority: when value is outside 1..100. Example: 101
    """
    # bool is an int subclass; accepting it would turn True into priority 1.
    if type(value) is not int:
        logger.info("Rejecting priority branch=type value_type=%s", type(value).__name__)
        raise InvalidPriority(f"priority must be an int, got {type(value).__name__}")
    if value < PRIORITY_MIN or value > PRIORITY_MAX:
        logger.info("Rejecting priority branch=range value=%s", value)
        raise InvalidPriority(f"priority {value} is outside {PRIORITY_MIN}..{PRIORITY_MAX}")


def validate_model(name: str) -> str:
    """
    Accept one of the three logical model names.

    :param name: requested model as str. Example: "qwen-14b"
    :return: the same name as str. Example: "qwen-14b"
    :raises InvalidModel: when name is not in MODEL_NAMES. Example: "qwen-70b"
    """
    logger.info("Starting validate_model name=%s", name)
    if type(name) is not str or name not in MODEL_NAMES:
        logger.info("Rejecting model branch=unknown name=%s", name)
        raise InvalidModel(f"model {name!r} is not one of {MODEL_NAMES}")
    logger.info("Accepted model name=%s", name)
    return name


def default_grant(access: dict | None) -> AccessGrant:
    """
    Turn an omitted grant into a job grant.

    :param access: raw grant object as dict | None. Example: {"mode": "paths", "paths": ["/Users/denispetelin/books/draft"]}
    :return: grant as AccessGrant. Example: AccessGrant(mode="job")
    :raises InvalidAccess: when mode is unknown or paths mode has no paths. Example: {"mode": "root"}
    """
    # None becomes AccessGrant(mode="job").
    # mode "home" ignores paths. mode "paths" requires a non-empty paths list.
    # Reject other modes with InvalidAccess.
    # Log the mode at INFO.
    raise NotImplementedError

