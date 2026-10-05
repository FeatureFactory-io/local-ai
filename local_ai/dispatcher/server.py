"""FastMCP tools. Each tool calls DispatcherService or CapacityReader and returns the dict."""

import json
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import litellm
import psutil
from mcp.server.fastmcp import FastMCP

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import (
    CapacityReader,
    HttpReader,
    MemoryReader,
    PressureReader,
)
from local_ai.contracts import SubmitRequest
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.llm.port import LiteLLMAdapter
from local_ai.logging_setup import configure_logging
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker


logger = logging.getLogger(__name__)

_HTTP_PORT_ENV = "LOCAL_AI_HTTP_PORT"
_ENV_DB = "LOCAL_AI_DB"
_ENV_JOBS = "LOCAL_AI_JOBS"
_ENV_OLLAMA = "OLLAMA_HOST"
_STATUS_QUEUED = "queued"
_SERVER_NAME = "local-ai"
_LOOPBACK_HOST = "127.0.0.1"
_DEFAULT_DB = Path("data/local-ai.sqlite")
_DEFAULT_JOBS = Path("data/jobs")
_DEFAULT_LOG = Path("logs/app.log")
_DEFAULT_OLLAMA = "http://127.0.0.1:11434"
_BYTES_PER_GIGABYTE = 1024**3
_PRESSURE_CRITICAL = 0.10
_PRESSURE_WARN = 0.20
_OLLAMA_TIMEOUT_S = 2.0
_TRANSPORT_STDIO = "stdio"
_TRANSPORT_HTTP = "streamable-http"
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
        _store_probes(self, memory, ollama, pressure, ollama_host)

    def build(self) -> FastMCP:
        """
        Register the dispatcher tools on a FastMCP server.

        :return: FastMCP server. Example: a server named "local-ai"
        """
        logger.info("build | entry | name=%s", _SERVER_NAME)
        mcp = FastMCP(_SERVER_NAME, host=_LOOPBACK_HOST)
        _register_tools(mcp, self)
        logger.info("build | exit | tools=capacity,submit,read,reprioritize,list,cancel,purge")
        return mcp

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
        plan: list[str] | None = None,
    ) -> dict:
        """
        Pin a task and enqueue it.

        :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
        :param done_when: completion condition as str. Example: "summary.md exists"
        :param priority: 1..100 or omitted as int | None. Example: 40
        :param model: qwen size or omitted as str | None. Example: "qwen-14b"
        :param inputs: files to copy as list[str] | None. Example: []
        :param access: grant object or omitted as dict | None. Example: {"mode": "job"}
        :param plan: ordered step sentences, or omitted to let the worker plan. Example: ["Write marker.txt"]
        :return: queued or rejected body as dict. Example: {"status": "queued", "position": 1}
        :raises ValueError: when goal, done_when, or a plan step is blank. Example: ""
        :raises TypeError: when inputs or plan is not a list of strings. Example: "notes.md"
        """
        logger.info("submit | entry")
        request = _submit_request(goal, done_when, priority, model, inputs, access, plan)
        snapshot = self._service.submit(request, self._connection)
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
        :raises ValueError: when task_id is blank. Example: "  "
        :raises InvalidPriority: when priority is outside 1..100. Example: 0
        """
        logger.info("reprioritize | entry")
        _require_text(task_id, "task_id", "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
        body = self._service.reprioritize(task_id, priority, self._connection)
        logger.info("reprioritize | exit | status=%s", body.get("status"))
        return body

    def list_jobs(self, status: str | None = None) -> dict:
        """
        Compact rows for the queue, the running pin, and finished pins.

        :param status: queued, running, completed, failed, all, or omitted as str | None. Example: "queued"
        :return: list body as dict. Example: {"status": "ok", "count": 0, "jobs": []}
        :raises TypeError: when status is not a string or None. Example: 1
        """
        logger.info("list | entry")
        body = self._service.list_jobs(status, self._connection)
        logger.info("list | exit | status=%s", body.get("status"))
        return body

    def cancel(self, task_id: str) -> dict:
        """
        Delete one queued pin and its job folder.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: cancelled or rejected body as dict. Example: {"status": "cancelled", "job_dir_removed": true}
        :raises ValueError: when task_id is blank. Example: "  "
        """
        logger.info("cancel | entry")
        _require_text(task_id, "task_id", "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
        body = self._service.cancel(task_id, self._connection)
        logger.info("cancel | exit | status=%s", body.get("status"))
        return body

    def purge(self, task_id: str | None = None, status: str | None = None) -> dict:
        """
        Delete finished pins and their job folders.

        :param task_id: one pin id, or omitted for every match, as str | None. Example: None
        :param status: completed, failed, or omitted as str | None. Example: "completed"
        :return: purge body as dict. Example: {"status": "ok", "count": 0, "purged": []}
        :raises ValueError: when task_id is blank. Example: "  "
        :raises TypeError: when status is not a string or None. Example: 1
        """
        logger.info("purge | entry")
        if task_id is not None:
            _require_text(task_id, "task_id", "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")
        body = self._service.purge(task_id, status, self._connection)
        logger.info("purge | exit | status=%s", body.get("status"))
        return body


def main() -> None:
    """
    Start the dispatcher. Stdio unless LOCAL_AI_HTTP_PORT is set.

    :return: None. Example: None
    """
    logger.info("main | entry")
    _server, mcp = compose()
    _run_server(mcp, http_port_from_env())


def compose(env: Mapping[str, str] | None = None) -> tuple[LocalAiServer, FastMCP]:
    """
    Wire the process and recover running pins. Does not call mcp.run.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_DB": "data/local-ai.sqlite"}
    :return: server and FastMCP as tuple[LocalAiServer, FastMCP]. Example: (LocalAiServer(...), FastMCP("local-ai"))
    :raises TypeError: when env is not a mapping. Example: []
    """
    settings = _process_settings(env)
    configure_logging(settings.log_path)
    database, connection = _open_database(settings.db_path)
    server = _wire(database, connection, settings)
    recovered = server._service.recover_running(connection)
    mcp = server.build()
    logger.info("compose | recovered=%s", len(recovered))
    return server, mcp


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


def _store_probes(
    server: "LocalAiServer",
    memory: MemoryReader | None,
    ollama: HttpReader | None,
    pressure: PressureReader | None,
    ollama_host: str | None,
) -> None:
    """
    Keep the capacity probes on the server.

    :param server: tool owner as LocalAiServer. Example: the composed server
    :param memory: returns (total_gb, free_gb) as MemoryReader | None. Example: lambda: (64.0, 40.2)
    :param ollama: HTTP reader as HttpReader | None. Example: lambda path: (200, {"models": []})
    :param pressure: pressure reader as PressureReader | None. Example: lambda: "normal"
    :param ollama_host: base URL as str | None. Example: "http://127.0.0.1:11434"
    :return: None. Example: None
    """
    server._memory = memory
    server._ollama = ollama
    server._pressure = pressure
    server._ollama_host = ollama_host


def _register_tools(mcp: FastMCP, server: "LocalAiServer") -> None:
    """
    Register the dispatcher tools. Each returns its dict, including a rejection.

    :param mcp: FastMCP server as FastMCP. Example: a server named "local-ai"
    :param server: tool owner as LocalAiServer. Example: the composed server
    :return: None. Example: None
    """
    mcp.add_tool(server.capacity)
    mcp.add_tool(server.submit)
    mcp.add_tool(server.read)
    mcp.add_tool(server.reprioritize)
    mcp.add_tool(server.list_jobs, name="list")
    mcp.add_tool(server.cancel)
    mcp.add_tool(server.purge)


def _submit_request(
    goal: str,
    done_when: str,
    priority: int | None,
    model: str | None,
    inputs: list[str] | None,
    access: dict | None,
    plan: list[str] | None,
) -> SubmitRequest:
    """
    Build the request the dispatcher accepts.

    :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
    :param done_when: completion condition as str. Example: "summary.md exists"
    :param priority: 1..100 or omitted as int | None. Example: 40
    :param model: qwen size or omitted as str | None. Example: "qwen-14b"
    :param inputs: files to copy as list[str] | None. Example: []
    :param access: grant object or omitted as dict | None. Example: {"mode": "job"}
    :param plan: ordered step sentences or omitted as list[str] | None. Example: ["Write marker.txt"]
    :return: request as SubmitRequest. Example: SubmitRequest("Write a brief.", "brief.md exists")
    :raises ValueError: when goal, done_when, or a plan step is blank. Example: ""
    :raises TypeError: when inputs or plan is not a list of strings. Example: "notes.md"
    """
    _require_text(goal, "goal", "Summarize the book into 2-3 pages.")
    _require_text(done_when, "done_when", "summary.md exists")
    return SubmitRequest(
        goal,
        done_when,
        priority,
        model,
        _input_paths(inputs),
        access,
        _plan_steps(plan),
    )


def _plan_steps(plan: list[str] | None) -> tuple[str, ...] | None:
    """
    Accept omitted plan, or a list of non-blank step sentences.

    :param plan: ordered steps or omitted as list[str] | None. Example: ["Write marker.txt"]
    :return: steps or None as tuple[str, ...] | None. Example: ("Write marker.txt",)
    :raises TypeError: when plan is not a list of strings. Example: "Write marker.txt"
    :raises ValueError: when plan is empty or a step is blank. Example: []
    """
    if plan is None:
        return None
    if not isinstance(plan, list):
        raise TypeError("plan must be a list of strings. Example: ['Write marker.txt']")
    if not plan:
        raise ValueError("plan must contain at least one step. Example: ['Write marker.txt']")
    actions: list[str] = []
    for action in plan:
        if not isinstance(action, str) or not action.strip():
            raise ValueError("plan step must be a non-empty string. Example: 'Write marker.txt'")
        actions.append(action.strip())
    return tuple(actions)


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


@dataclass(frozen=True)
class _ProcessSettings:
    """Paths and the environment the composition root reads once."""

    db_path: Path
    jobs_root: Path
    ollama_host: str
    log_path: Path
    env: Mapping[str, str]


def _run_server(mcp: FastMCP, port: int | None) -> None:
    """
    Start stdio, or streamable HTTP on the loopback port.

    FastMCP.run takes the transport name. Host and port live on the server settings.

    :param mcp: FastMCP server as FastMCP. Example: a server named "local-ai"
    :param port: loopback port or None as int | None. Example: 8765
    :return: None. Example: None
    """
    if port is None:
        logger.info("main | transport=%s", _TRANSPORT_STDIO)
        mcp.run(transport=_TRANSPORT_STDIO)
        return
    _run_loopback(mcp, port)


def _run_loopback(mcp: FastMCP, port: int) -> None:
    """
    Bind streamable HTTP to 127.0.0.1. A non-loopback host is replaced.

    :param mcp: FastMCP server as FastMCP. Example: a server named "local-ai"
    :param port: loopback port as int. Example: 8765
    :return: None. Example: None
    """
    mcp.settings.host = _LOOPBACK_HOST
    mcp.settings.port = port
    logger.info("main | transport=%s host=%s port=%s", _TRANSPORT_HTTP, _LOOPBACK_HOST, port)
    mcp.run(transport=_TRANSPORT_HTTP)


def _process_settings(env: Mapping[str, str] | None) -> _ProcessSettings:
    """
    Resolve database, jobs, Ollama, and the log path.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_DB": "data/local-ai.sqlite"}
    :return: settings as _ProcessSettings. Example: _ProcessSettings(Path("data/local-ai.sqlite"), ...)
    :raises TypeError: when env is not a mapping. Example: []
    """
    source = _port_environment(env)
    return _ProcessSettings(
        db_path=_path_setting(source, _ENV_DB, _DEFAULT_DB),
        jobs_root=_path_setting(source, _ENV_JOBS, _DEFAULT_JOBS),
        ollama_host=_text_setting(source, _ENV_OLLAMA, _DEFAULT_OLLAMA),
        log_path=_DEFAULT_LOG,
        env=source,
    )


def _path_setting(env: Mapping[str, str], key: str, default: Path) -> Path:
    """
    Read a path variable. Missing or blank keeps the default.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_DB": "data/local-ai.sqlite"}
    :param key: variable name as str. Example: "LOCAL_AI_DB"
    :param default: fallback path as Path. Example: Path("data/local-ai.sqlite")
    :return: path as Path. Example: Path("data/local-ai.sqlite")
    """
    raw = env.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return default
    return Path(raw)


def _text_setting(env: Mapping[str, str], key: str, default: str) -> str:
    """
    Read a text variable. Missing or blank keeps the default.

    :param env: environment as Mapping[str, str]. Example: {"OLLAMA_HOST": "http://127.0.0.1:11434"}
    :param key: variable name as str. Example: "OLLAMA_HOST"
    :param default: fallback text as str. Example: "http://127.0.0.1:11434"
    :return: text as str. Example: "http://127.0.0.1:11434"
    """
    raw = env.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return default
    return raw


def _open_database(path: Path) -> tuple[Database, sqlite3.Connection]:
    """
    Connect and migrate.

    :param path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :return: database and connection as tuple. Example: (Database(...), connection)
    """
    database = Database(path)
    connection = database.connect()
    database.migrate()
    logger.info("compose | migrated path=%s", path)
    return database, connection


def _wire(
    database: Database,
    connection: sqlite3.Connection,
    settings: _ProcessSettings,
) -> LocalAiServer:
    """
    Construct the objects section 19 names, in that order.

    :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param settings: process settings as _ProcessSettings. Example: the settings from _process_settings
    :return: server as LocalAiServer. Example: LocalAiServer(...)
    """
    board = Blackboard(database)
    plan = PlanStore(database)
    capacity = CapacityReader()
    shell = ShellTool(GrantChecker())
    prompt = PromptAssembler()
    llm = LiteLLMAdapter(litellm.completion, settings.env, settings.ollama_host)
    worker = COMPOSER_WORKER(llm, shell, prompt, plan, board, connection)
    service = _dispatcher(database, board, plan, capacity, settings)
    slot = Slot(service, worker, connection, _utc_now)
    return _local_server(service, slot, capacity, connection, settings)


def _dispatcher(
    database: Database,
    board: Blackboard,
    plan: PlanStore,
    capacity: CapacityReader,
    settings: _ProcessSettings,
) -> DispatcherService:
    """
    Build the dispatcher with the process capacity probes.

    :param database: process database as Database. Example: Database(Path("data/local-ai.sqlite"))
    :param board: pin store as Blackboard. Example: Blackboard(database)
    :param plan: step store as PlanStore. Example: PlanStore(database)
    :param capacity: capacity reader as CapacityReader. Example: CapacityReader()
    :param settings: process settings as _ProcessSettings. Example: the settings from _process_settings
    :return: service as DispatcherService. Example: DispatcherService(...)
    """
    return DispatcherService(
        database,
        board,
        plan,
        capacity,
        settings.jobs_root,
        memory=_read_memory,
        ollama=_read_ollama,
        pressure=_read_pressure,
        ollama_host=settings.ollama_host,
    )


def _local_server(
    service: DispatcherService,
    slot: Slot,
    capacity: CapacityReader,
    connection: sqlite3.Connection,
    settings: _ProcessSettings,
) -> LocalAiServer:
    """
    Store the same probes the dispatcher uses, so capacity does not open its own client.

    :param service: pin service as DispatcherService. Example: the composed service
    :param slot: worker slot as Slot. Example: Slot(service, worker, connection, clock)
    :param capacity: capacity reader as CapacityReader. Example: CapacityReader()
    :param connection: shared connection as sqlite3.Connection. Example: the Database connection
    :param settings: process settings as _ProcessSettings. Example: the settings from _process_settings
    :return: server as LocalAiServer. Example: LocalAiServer(...)
    """
    return LocalAiServer(
        service,
        slot,
        capacity,
        connection,
        memory=_read_memory,
        ollama=_read_ollama,
        pressure=_read_pressure,
        ollama_host=settings.ollama_host,
    )


def _utc_now() -> str:
    """
    UTC timestamp for the slot clock.

    :return: timestamp as str. Example: "2026-10-04T18:40:02.000000Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_memory() -> tuple[float, float]:
    """
    Read total and available memory in gigabytes.

    :return: total and free gigabytes as tuple[float, float]. Example: (64.0, 40.2)
    """
    memory = psutil.virtual_memory()
    total = memory.total / _BYTES_PER_GIGABYTE
    free = memory.available / _BYTES_PER_GIGABYTE
    logger.info("memory | total_gb=%.2f free_gb=%.2f", total, free)
    return total, free


def _read_pressure() -> str:
    """
    Map free memory over total onto normal, warn, or critical.

    :return: pressure word as str. Example: "normal"
    """
    total, free = _read_memory()
    if total <= 0:
        logger.warning("pressure | total_gb=0 word=unknown")
        return "unknown"
    word = _pressure_word(free / total)
    logger.info("pressure | word=%s", word)
    return word


def _pressure_word(ratio: float) -> str:
    """
    Pick a pressure word from the free-memory ratio.

    :param ratio: free divided by total as float. Example: 0.5
    :return: pressure word as str. Example: "normal"
    """
    if ratio < _PRESSURE_CRITICAL:
        return "critical"
    if ratio < _PRESSURE_WARN:
        return "warn"
    return "normal"


def _read_ollama(url: str) -> tuple[int, dict]:
    """
    GET one Ollama URL. A connection error propagates so CapacityReader marks Ollama down.

    :param url: full URL as str. Example: "http://127.0.0.1:11434/api/tags"
    :return: status and JSON object as tuple[int, dict]. Example: (200, {"models": []})
    :raises urllib.error.URLError: when the host does not answer. Example: connection refused
    """
    logger.info("ollama | GET")
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_OLLAMA_TIMEOUT_S) as response:
            return _json_body(response.status, response.read())
    except urllib.error.HTTPError as exc:
        logger.info("ollama | status=%s", exc.code)
        return exc.code, {}


def _json_body(status: int, raw: bytes) -> tuple[int, dict]:
    """
    Decode an Ollama JSON object. A non-object becomes an empty dict.

    :param status: HTTP status as int. Example: 200
    :param raw: response bytes as bytes. Example: b'{"models": []}'
    :return: status and object as tuple[int, dict]. Example: (200, {"models": []})
    """
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("ollama | body was not json status=%s", status)
        return status, {}
    if not isinstance(body, dict):
        return status, {}
    return status, body
