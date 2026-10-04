"""FastMCP tools. Each tool calls DispatcherService or CapacityReader and returns the dict."""

import logging
import os
import sqlite3
from collections.abc import Mapping
from typing import TypeVar

from local_ai.capacity.reader import (
    CapacityReader,
    HttpReader,
    MemoryReader,
    PressureReader,
)
from local_ai.contracts import SubmitRequest
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.worker.loop import Worker


logger = logging.getLogger(__name__)

_HTTP_PORT_ENV = "LOCAL_AI_HTTP_PORT"
_STATUS_QUEUED = "queued"
_ReaderT = TypeVar("_ReaderT")

# Worker is constructed in main and passed to Slot. Service does not import it.
COMPOSER_WORKER = Worker


class LocalAiServer:
    """
    MCP surface. Cursor and Claude use stdio. ChatGPT uses loopback streamable HTTP.

    Tools call the service. They do not embed queue policy.
    """

    def __init__(
        self,
        service: DispatcherService,
        slot: Slot,
        capacity: CapacityReader,
        connection: sqlite3.Connection,
        *,
        memory: MemoryReader | None = None,
        ollama: HttpReader | None = None,
        pressure: PressureReader | None = None,
        ollama_host: str | None = None,
    ) -> None:
        """
        Store the wired objects. FastMCP registration happens in build.

        :param service: pin service as DispatcherService. Example: the composed service
        :param slot: worker slot as Slot. Example: Slot(service, worker, connection, clock)
        :param capacity: capacity reader as CapacityReader. Example: CapacityReader()
        :param connection: shared connection as sqlite3.Connection. Example: the Database connection
        :param memory: returns (total_gb, free_gb) as MemoryReader | None. Example: lambda: (64.0, 40.2)
        :param ollama: HTTP reader as HttpReader | None. Example: lambda path: (200, {"models": []})
        :param pressure: pressure reader as PressureReader | None. Example: lambda: "normal"
        :param ollama_host: base URL as str | None. Example: "http://127.0.0.1:11434"
        """
        self._service = service
        self._slot = slot
        self._capacity = capacity
        self._connection = connection
        self._memory = memory
        self._ollama = ollama
        self._pressure = pressure
        self._ollama_host = ollama_host

    def build(self):
        """
        Register the four tools on a FastMCP server.

        :return: FastMCP server. Example: a server named "local-ai"
        """
        # Import FastMCP here only if module-level import is already done in this module.
        # Register capacity, submit, read, and reprioritize.
        # Each tool returns the dict from the matching method. A rejected dict is not an exception.
        raise NotImplementedError

    def capacity(self) -> dict:
        """
        Current machine load and a model-size recommendation.

        :return: capacity body as dict. Example: {"status": "ok", "ollama": "up", "recommended": "qwen-14b"}
        :raises RuntimeError: when a reader or host was not stored. Example: memory=None
        """
        logger.info("capacity | entry")
        body = self._capacity.read(
            _require_reader(self._memory, "memory", "lambda: (64.0, 40.2)"),
            _require_reader(self._ollama, "ollama", "lambda path: (200, {'models': []})"),
            _require_reader(self._pressure, "pressure", "lambda: 'normal'"),
            _require_ollama_host(self._ollama_host),
        )
        logger.info(
            "capacity | exit | ollama=%s recommended=%s",
            body.get("ollama"),
            body.get("recommended"),
        )
        return body

    def submit(
        self,
        goal: str,
        done_when: str,
        priority: int | None = None,
        model: str | None = None,
        inputs: list[str] | None = None,
        access: dict | None = None,
    ) -> dict:
        """
        Pin a task and enqueue it.

        :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
        :param done_when: completion condition as str. Example: "summary.md exists"
        :param priority: 1..100 or omitted as int | None. Example: 40
        :param model: qwen size or omitted as str | None. Example: "qwen-14b"
        :param inputs: files to copy as list[str] | None. Example: []
        :param access: grant object or omitted as dict | None. Example: {"mode": "job"}
        :return: queued or rejected body as dict. Example: {"status": "queued", "position": 1}
        :raises ValueError: when goal or done_when is blank. Example: ""
        :raises TypeError: when inputs is not a list of strings. Example: "notes.md"
        """
        logger.info("submit | entry")
        snapshot = self._service.submit(
            _submit_request(goal, done_when, priority, model, inputs, access),
            self._connection,
        )
        if snapshot.get("status") == _STATUS_QUEUED:
            self._slot.kick()
        logger.info("submit | exit | status=%s", snapshot.get("status"))
        return snapshot

    def read(self, task_id: str) -> dict:
        """
        Return one pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: pin body as dict. Example: {"status": "running", "phase": "acting"}
        :raises ValueError: when task_id is blank. Example: "  "
        """
        logger.info("read | entry")
        _require_text(task_id, "task_id", "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
        body = self._service.read_pin(task_id, self._connection)
        logger.info("read | exit | status=%s", body.get("status"))
        return body

    def reprioritize(self, task_id: str, priority: int) -> dict:
        """
        Change priority of a queued pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 5
        :return: queued or rejected body as dict. Example: {"status": "rejected", "reason": "not_queued"}
        """
        # DispatcherService.reprioritize. Do not call the slot.
        raise NotImplementedError


def main() -> None:
    """
    Start the dispatcher. Stdio unless LOCAL_AI_HTTP_PORT is set.

    :return: None. Example: None
    """
    # configure_logging. Open Database, migrate, DispatcherService.recover_running.
    # Compose Blackboard, PlanStore, CapacityReader, GrantChecker, ShellTool,
    # PromptAssembler, LiteLLMAdapter, Worker, Slot, and LocalAiServer.
    # LocalAiServer.build.
    # If LOCAL_AI_HTTP_PORT is unset, FastMCP run transport stdio.
    # If it is set, FastMCP run transport streamable-http host 127.0.0.1 and that port.
    # Logs stay on stderr and logs/app.log.
    raise NotImplementedError


def http_port_from_env(env: dict[str, str] | None = None) -> int | None:
    """
    Read the loopback port. Unset means stdio.

    :param env: environment as dict[str, str] | None. Example: {"LOCAL_AI_HTTP_PORT": "8765"}
    :return: port or None as int | None. Example: 8765
    :raises ValueError: when the variable is not an integer. Example: "abc"
    :raises TypeError: when env is not a mapping. Example: []
    """
    source = _port_environment(env)
    raw = source.get(_HTTP_PORT_ENV)
    if raw is None or raw == "":
        logger.info("http_port_from_env | unset")
        return None
    return _integer_port(raw)


def _port_environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """
    Use the passed mapping, or the process environment when env is omitted.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_HTTP_PORT": "8765"}
    :return: mapping to read as Mapping[str, str]. Example: {"LOCAL_AI_HTTP_PORT": "8765"}
    :raises TypeError: when env is not a mapping. Example: []
    """
    if env is None:
        return os.environ
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_HTTP_PORT': '8765'}")
    return env


def _integer_port(raw: object) -> int:
    """
    Parse a port string. This does not open a socket.

    :param raw: environment value as object. Example: "8765"
    :return: port as int. Example: 8765
    :raises ValueError: when raw is not an integer string. Example: "abc"
    """
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise ValueError("LOCAL_AI_HTTP_PORT must be an integer. Example: 8765")
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError("LOCAL_AI_HTTP_PORT must be an integer. Example: 8765") from exc
    logger.info("http_port_from_env | port=%s", port)
    return port


def _submit_request(
    goal: str,
    done_when: str,
    priority: int | None,
    model: str | None,
    inputs: list[str] | None,
    access: dict | None,
) -> SubmitRequest:
    """
    Build the request the dispatcher accepts.

    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
    :param done_when: completion condition as str. Example: "summary.md exists"
    :param priority: 1..100 or omitted as int | None. Example: 40
    :param model: qwen size or omitted as str | None. Example: "qwen-14b"
    :param inputs: files to copy as list[str] | None. Example: []
    :param access: grant object or omitted as dict | None. Example: {"mode": "job"}
    :return: request as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
    :raises ValueError: when goal or done_when is blank. Example: ""
    :raises TypeError: when inputs is not a list of strings. Example: "notes.md"
    """
    _require_text(goal, "goal", "Summarize the book into 2-3 pages.")
    _require_text(done_when, "done_when", "summary.md exists")
    return SubmitRequest(goal, done_when, priority, model, _input_paths(inputs), access)


def _require_text(value: object, name: str, example: str) -> None:
    """
    Reject a blank string before the dispatcher sees it.

    :param value: candidate text as object. Example: "Write a brief."
    :param name: field name as str. Example: "goal"
    :param example: usage hint as str. Example: "Summarize the book into 2-3 pages."
    :return: None. Example: None
    :raises ValueError: when value is blank or not a string. Example: ""
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty str. Example: {example!r}")


def _input_paths(inputs: list[str] | None) -> tuple[str, ...]:
    """
    Copy the input list into the tuple SubmitRequest stores.

    :param inputs: files to copy as list[str] | None. Example: []
    :return: paths as tuple[str, ...]. Example: ()
    :raises TypeError: when inputs is not a list of strings. Example: "notes.md"
    """
    if inputs is None:
        return ()
    if isinstance(inputs, str) or not isinstance(inputs, (list, tuple)):
        raise TypeError("inputs must be a list of paths. Example: []")
    paths = tuple(inputs)
    if any(not isinstance(path, str) for path in paths):
        raise TypeError("inputs must be a list of paths. Example: []")
    return paths


def _require_reader(reader: _ReaderT | None, name: str, example: str) -> _ReaderT:
    """
    Reject a missing probe before CapacityReader.read runs.

    :param reader: injected probe as _ReaderT | None. Example: lambda: (64.0, 40.2)
    :param name: probe name as str. Example: "memory"
    :param example: usage hint as str. Example: "lambda: (64.0, 40.2)"
    :return: the probe as _ReaderT. Example: lambda: (64.0, 40.2)
    :raises RuntimeError: when the probe is missing. Example: memory=None
    """
    if not callable(reader):
        raise RuntimeError(f"{name} reader is required. Example: {example}")
    return reader


def _require_ollama_host(host: str | None) -> str:
    """
    Reject a blank Ollama base URL.

    :param host: base URL as str | None. Example: "http://127.0.0.1:11434"
    :return: the same URL as str. Example: "http://127.0.0.1:11434"
    :raises RuntimeError: when host is missing or blank. Example: None
    """
    if not isinstance(host, str) or not host.strip():
        raise RuntimeError("ollama_host is required. Example: 'http://127.0.0.1:11434'")
    return host
