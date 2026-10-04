"""MCP tool surface. Readers are injected. Ollama is not contacted."""

import json
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import anyio
import pytest
from mcp.server.fastmcp import FastMCP

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import AccessGrant, Completion, TaskPin, ToolCall
from local_ai.db.database import Database
from local_ai.dispatcher.server import LocalAiServer, compose, http_port_from_env, main
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.llm.port import LiteLLMAdapter, ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
_HOST = "http://127.0.0.1:11434"
_TAGS = f"{_HOST}/api/tags"
_PS = f"{_HOST}/api/ps"
_JOIN_TIMEOUT_S = 8.0
_HOLD_COMMAND = "python3 -c \"open('awake.txt','w').write('1'); import time; time.sleep(4)\""
_HOLD_SCRIPT = (
    Completion(content="1. Pause in the job folder"),
    Completion(tool_calls=(ToolCall("shell", _HOLD_COMMAND),), final_answer="Paused"),
)

Http = Callable[[str], tuple[int, dict]]
Memory = Callable[[], tuple[float, float]]
Pressure = Callable[[], str]
_PIN = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Put the root logger back after compose attaches process handlers."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in saved_handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(saved_level)


def test_http_port_from_env_missing_returns_none() -> None:
    """An environment without LOCAL_AI_HTTP_PORT stays on stdio."""
    assert http_port_from_env({}) is None


def test_http_port_from_env_empty_returns_none() -> None:
    """An empty LOCAL_AI_HTTP_PORT stays on stdio."""
    assert http_port_from_env({"LOCAL_AI_HTTP_PORT": ""}) is None


def test_http_port_from_env_integer_string_returns_that_int() -> None:
    """A decimal port string becomes that integer."""
    assert http_port_from_env({"LOCAL_AI_HTTP_PORT": "8765"}) == 8765


def test_http_port_from_env_abc_raises_value_error() -> None:
    """A non-integer port is refused before any socket is opened."""
    with pytest.raises(ValueError, match="LOCAL_AI_HTTP_PORT"):
        http_port_from_env({"LOCAL_AI_HTTP_PORT": "abc"})


def test_http_port_from_env_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting env reads LOCAL_AI_HTTP_PORT from the process."""
    monkeypatch.setenv("LOCAL_AI_HTTP_PORT", "9001")
    assert http_port_from_env() == 9001


def test_capacity_returns_the_injected_reader_dict(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """capacity calls CapacityReader.read and returns that body. Keys match capacity_up.json."""
    fixture = _contract("capacity_up.json")
    ollama, calls = _scripted_ollama(tags_status=200, loaded_model=fixture["loaded_model"])
    memory: Memory = lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"])
    pressure: Pressure = lambda: fixture["memory_pressure"]
    reader = _CountingReader()
    server = _server(tmp_path, reader=reader, memory=memory, ollama=ollama, pressure=pressure)
    caplog.set_level(logging.INFO)

    body = server.capacity()

    assert reader.calls == 1
    assert calls == [_TAGS, _PS]
    assert body == CapacityReader().read(memory, ollama, pressure, _HOST)
    assert set(body) == set(fixture)
    assert any(record.message.startswith("capacity |") for record in caplog.records)


def test_capacity_when_tags_fail_returns_the_down_body(tmp_path: Path) -> None:
    """A failed tags read is the down body. Keys match capacity_down.json. No live host is used."""
    fixture = _contract("capacity_down.json")
    ollama, calls = _scripted_ollama(tags_status=503, loaded_model=None)
    memory: Memory = lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"])
    pressure: Pressure = lambda: fixture["memory_pressure"]
    server = _server(tmp_path, memory=memory, ollama=ollama, pressure=pressure)

    body = server.capacity()

    assert set(body) == set(fixture)
    assert body["ollama"] == "down"
    assert body["recommended"] == "none"
    assert body["loaded_model"] is None
    assert body["models"] == []
    assert calls == [_TAGS]


def test_submit_returns_the_queued_snapshot_after_kick_claims(tmp_path: Path) -> None:
    """The tool returns status queued. kick has already claimed the pin, so a later read is running."""
    server = _server(tmp_path, script=_HOLD_SCRIPT)
    held: str | None = None
    try:
        body = server.submit("Write a brief.", "brief.md exists", priority=40, model="qwen-14b")
        held = str(body["task_id"])
        _wait_until_holding(tmp_path, held)
        live = server._service.read_pin(held, server._connection)
    finally:
        _release_and_join(tmp_path, server, held)

    assert set(body) == set(_contract("submit_queued.json"))
    assert body["status"] == "queued"
    assert body["priority"] == 40
    assert body["model"] == "qwen-14b"
    assert body["position"] == 1
    assert live["status"] == "running"


def test_submit_when_ollama_is_down_returns_rejected_without_kick(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A down daemon returns the rejected dict and does not call Slot.kick."""
    ollama, _calls = _scripted_ollama(tags_status=503, loaded_model=None)
    server = _server(tmp_path, ollama=ollama)
    caplog.set_level(logging.INFO)

    body = server.submit("Write a brief.", "brief.md exists", priority=40, model="qwen-14b")

    assert set(body) == set(_contract("submit_rejected.json"))
    assert body == {"status": "rejected", "reason": "ollama_down", "task_id": None}
    assert not any(record.message == "Starting kick" for record in caplog.records)


def test_read_returns_the_pin_body(tmp_path: Path) -> None:
    """read returns the read_pin body. Keys match the running example."""
    server = _server(tmp_path, script=_HOLD_SCRIPT)
    held: str | None = None
    try:
        queued = server.submit("Write a brief.", "brief.md exists", priority=40, model="qwen-14b")
        held = str(queued["task_id"])
        _wait_until_holding(tmp_path, held)
        body = server.read(held)
    finally:
        _release_and_join(tmp_path, server, held)

    assert set(body) == set(_contract("read_running.json"))
    assert body["status"] == "running"
    assert body["task_id"] == queued["task_id"]
    assert body["goal"] == "Write a brief."
    assert body["model"] == "qwen-14b"


def test_read_missing_pin_returns_not_found(tmp_path: Path) -> None:
    """A missing id is a rejected dict, not an exception."""
    server = _server(tmp_path)

    body = server.read("7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f")

    assert body["status"] == "rejected"
    assert body["reason"] == "not_found"
    assert body["task_id"] == "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


def test_reprioritize_queued_pin_returns_the_new_position_without_kicking(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A queued pin changes priority. The slot is not called, so it stays queued."""
    server = _server(tmp_path, script=_HOLD_SCRIPT)
    caplog.set_level(logging.INFO)
    held: str | None = None
    try:
        first = server.submit("Write the first brief.", "brief.md exists", priority=40, model="qwen-14b")
        held = str(first["task_id"])
        _wait_until_holding(tmp_path, held)
        second = server.submit("Write the second brief.", "brief.md exists", priority=80, model="qwen-14b")
        kicks_before = _kick_count(caplog)
        body = server.reprioritize(str(second["task_id"]), 5)
        queued = server.read(str(second["task_id"]))
        running = server.read(held)
        kicks_after = _kick_count(caplog)
    finally:
        _release_and_join(tmp_path, server, held)

    assert second["status"] == "queued"
    assert set(body) == set(_contract("reprioritize_queued.json"))
    assert body["status"] == "queued"
    assert body["priority"] == 5
    assert body["task_id"] == second["task_id"]
    assert body["position"] == 1
    assert kicks_after == kicks_before
    assert queued["status"] == "queued"
    assert queued["priority"] == 5
    assert running["status"] == "running"


def test_reprioritize_running_pin_returns_not_queued(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A running pin is rejected and its priority stays put. The slot is not called."""
    server = _server(tmp_path, script=_HOLD_SCRIPT)
    caplog.set_level(logging.INFO)
    held: str | None = None
    try:
        first = server.submit("Write a brief.", "brief.md exists", priority=40, model="qwen-14b")
        held = str(first["task_id"])
        _wait_until_holding(tmp_path, held)
        kicks_before = _kick_count(caplog)
        body = server.reprioritize(held, 5)
        live = server.read(held)
        kicks_after = _kick_count(caplog)
    finally:
        _release_and_join(tmp_path, server, held)

    assert set(body) == set(_contract("reprioritize_rejected.json"))
    assert body == {
        "status": "rejected",
        "reason": "not_queued",
        "task_id": first["task_id"],
    }
    assert kicks_after == kicks_before
    assert live["status"] == "running"
    assert live["priority"] == 40


def test_build_registers_dispatcher_tools_and_returns_rejected_dict(tmp_path: Path) -> None:
    """build names the server local-ai. A rejected submit is a dict, not an exception."""
    ollama, _calls = _scripted_ollama(tags_status=503, loaded_model=None)
    server = _server(tmp_path, ollama=ollama)
    mcp = server.build()

    assert isinstance(mcp, FastMCP)
    assert mcp.name == "local-ai"
    assert _tool_names(mcp) == [
        "cancel",
        "capacity",
        "list",
        "purge",
        "read",
        "reprioritize",
        "submit",
    ]
    rejected = _call_tool(
        mcp,
        "submit",
        {"goal": "Write a brief.", "done_when": "brief.md exists", "priority": 40, "model": "qwen-14b"},
    )
    capacity = _call_tool(mcp, "capacity", {})

    assert set(rejected) == set(_contract("submit_rejected.json"))
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "ollama_down"
    assert rejected["task_id"] is None
    assert set(capacity) == set(_contract("capacity_down.json"))


def test_list_cancel_and_purge_on_an_empty_board(tmp_path: Path) -> None:
    """The operator tools return contract dicts when no pin exists."""
    server = _server(tmp_path)
    mcp = server.build()

    listed = _call_tool(mcp, "list", {})
    cancelled = _call_tool(mcp, "cancel", {"task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"})
    purged = _call_tool(mcp, "purge", {})

    assert listed == {"status": "ok", "count": 0, "jobs": []}
    assert cancelled == _contract("cancel_rejected.json") | {"reason": "not_found"}
    assert purged == {"status": "ok", "count": 0, "purged": []}


def test_build_in_process_calls_match_contract_keys(tmp_path: Path) -> None:
    """In-process tool calls return the section 18 keys for up, queued, running, and reprioritize."""
    server = _server(tmp_path, script=_HOLD_SCRIPT)
    mcp = server.build()
    held: str | None = None
    try:
        queued = _call_tool(
            mcp,
            "submit",
            {"goal": "Write a brief.", "done_when": "brief.md exists", "priority": 40, "model": "qwen-14b"},
        )
        held = str(queued["task_id"])
        _wait_until_holding(tmp_path, held)
        read_body = _call_tool(mcp, "read", {"task_id": held})
        second = _call_tool(
            mcp,
            "submit",
            {"goal": "Write the second brief.", "done_when": "brief.md exists", "priority": 80, "model": "qwen-14b"},
        )
        changed = _call_tool(mcp, "reprioritize", {"task_id": second["task_id"], "priority": 5})
        rejected = _call_tool(mcp, "reprioritize", {"task_id": held, "priority": 5})
        capacity = _call_tool(mcp, "capacity", {})
    finally:
        _release_and_join(tmp_path, server, held)

    assert set(queued) == set(_contract("submit_queued.json"))
    assert queued["status"] == "queued"
    assert set(read_body) == set(_contract("read_running.json"))
    assert read_body["status"] == "running"
    assert set(changed) == set(_contract("reprioritize_queued.json"))
    assert changed["status"] == "queued"
    assert set(rejected) == set(_contract("reprioritize_rejected.json"))
    assert rejected["status"] == "rejected"
    assert set(capacity) == set(_contract("capacity_up.json"))


def test_compose_requeues_a_running_pin_without_calling_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup opens the database, requeues a running pin, and returns before mcp.run."""
    db_path = tmp_path / "local-ai.sqlite"
    _insert_running(db_path, _PIN)
    monkeypatch.chdir(tmp_path)
    server, mcp = compose(
        {"LOCAL_AI_DB": str(db_path), "LOCAL_AI_JOBS": str(tmp_path / "jobs")},
    )

    body = server.read(_PIN)
    worker = server._slot._runner

    assert body["status"] == "queued"
    assert mcp.name == "local-ai"
    assert _tool_names(mcp) == [
        "cancel",
        "capacity",
        "list",
        "purge",
        "read",
        "reprioritize",
        "submit",
    ]
    assert isinstance(server._service, DispatcherService)
    assert isinstance(server._capacity, CapacityReader)
    assert isinstance(server._slot, Slot)
    assert isinstance(worker, Worker)
    assert isinstance(worker._llm, LiteLLMAdapter)
    assert isinstance(worker._shell, ShellTool)
    assert isinstance(worker._shell._grant_checker, GrantChecker)
    assert isinstance(worker._prompt, PromptAssembler)
    assert isinstance(server._service._blackboard, Blackboard)
    assert isinstance(server._service._plan, PlanStore)
    assert (tmp_path / "logs" / "app.log").is_file()


def test_main_uses_stdio_when_the_port_is_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset port calls mcp.run with the stdio transport and does not bind."""
    _point_process_at(monkeypatch, tmp_path)
    monkeypatch.delenv("LOCAL_AI_HTTP_PORT", raising=False)
    seen = _capture_run(monkeypatch)

    main()

    assert seen["transport"] == "stdio"
    assert seen["called"] is True


def test_main_uses_loopback_http_when_the_port_is_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set port runs streamable HTTP on 127.0.0.1 even if the server host was changed."""
    _point_process_at(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_AI_HTTP_PORT", "8765")
    seen = _capture_run(monkeypatch, spoil_host="0.0.0.0")

    main()

    assert seen["transport"] == "streamable-http"
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8765


class _CountingReader(CapacityReader):
    """Real CapacityReader that counts read calls."""

    def __init__(self) -> None:
        """Start at zero reads."""
        self.calls = 0

    def read(
        self,
        memory: Memory,
        ollama: Http,
        pressure: Pressure,
        ollama_host: str,
    ) -> dict:
        """
        Count one read, then build the real body.

        :param memory: gigabyte probe as Memory. Example: lambda: (64.0, 40.2)
        :param ollama: HTTP reader as Http. Example: lambda path: (200, {"models": []})
        :param pressure: pressure word as Pressure. Example: lambda: "normal"
        :param ollama_host: base URL as str. Example: "http://127.0.0.1:11434"
        :return: capacity body as dict. Example: {"status": "ok", "ollama": "up"}
        """
        self.calls += 1
        return super().read(memory, ollama, pressure, ollama_host)


def _server(
    tmp_path: Path,
    *,
    reader: CapacityReader | None = None,
    memory: Memory | None = None,
    ollama: Http | None = None,
    pressure: Pressure | None = None,
    script: tuple[Completion, ...] = (),
) -> LocalAiServer:
    """
    Wire a server on a temp database. Readers are the only stand-ins.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param reader: capacity reader as CapacityReader | None. Example: CapacityReader()
    :param memory: gigabyte probe as Memory | None. Example: lambda: (64.0, 40.2)
    :param ollama: HTTP reader as Http | None. Example: lambda path: (200, {"models": []})
    :param pressure: pressure word as Pressure | None. Example: lambda: "normal"
    :param script: model turns as tuple[Completion, ...]. Example: ()
    :return: server as LocalAiServer. Example: LocalAiServer(...)
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    board = Blackboard(database)
    plan = PlanStore(database)
    capacity = reader if reader is not None else CapacityReader()
    probe = memory if memory is not None else (lambda: (64.0, 40.2))
    http = ollama if ollama is not None else _scripted_ollama(200, "qwen2.5:14b")[0]
    words = pressure if pressure is not None else (lambda: "normal")
    service = DispatcherService(
        database,
        board,
        plan,
        capacity,
        tmp_path / "jobs",
        memory=probe,
        ollama=http,
        pressure=words,
        ollama_host=_HOST,
    )
    worker = Worker(ScriptedLLM(script), ShellTool(GrantChecker()), PromptAssembler(), plan, board, connection)
    slot = Slot(service, worker, connection, lambda: "2026-10-04T18:40:02Z")
    return LocalAiServer(
        service,
        slot,
        capacity,
        connection,
        memory=probe,
        ollama=http,
        pressure=words,
        ollama_host=_HOST,
    )


def _scripted_ollama(tags_status: int, loaded_model: str | None) -> tuple[Http, list[str]]:
    """
    Answer tags and ps without opening a socket. Record each URL.

    :param tags_status: /api/tags status as int. Example: 200
    :param loaded_model: /api/ps name as str | None. Example: "qwen2.5:14b"
    :return: reader and the URLs it saw. Example: (reader, [])
    """
    calls: list[str] = []

    def ollama(url: str) -> tuple[int, dict]:
        calls.append(url)
        if url == _TAGS:
            return tags_status, {"models": []}
        if url == _PS:
            models = [] if loaded_model is None else [{"name": loaded_model}]
            return 200, {"models": models}
        raise AssertionError(url)

    return ollama, calls


def _tool_names(mcp: FastMCP) -> list[str]:
    """
    List registered tool names.

    :param mcp: server as FastMCP. Example: a server named "local-ai"
    :return: sorted names as list[str]. Example: ["capacity", "read", "reprioritize", "submit"]
    """
    tools = anyio.run(mcp.list_tools)
    return sorted(tool.name for tool in tools)


def _call_tool(mcp: FastMCP, name: str, arguments: dict) -> dict:
    """
    Call one registered tool in-process and return its dict.

    :param mcp: server as FastMCP. Example: a server named "local-ai"
    :param name: tool name as str. Example: "submit"
    :param arguments: tool arguments as dict. Example: {"goal": "Write a brief.", "done_when": "brief.md exists"}
    :return: tool body as dict. Example: {"status": "rejected", "reason": "ollama_down", "task_id": None}
    """
    result = anyio.run(mcp.call_tool, name, arguments)
    return _dict_from_tool(result)


def _dict_from_tool(result: object) -> dict:
    """
    Unwrap a FastMCP tool result into the dict the method returned.

    :param result: call_tool result as object. Example: a list of text content
    :return: body as dict. Example: {"status": "ok"}
    :raises AssertionError: when the result is not a dict payload. Example: None
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[1]
    if isinstance(result, list) and result and hasattr(result[0], "text"):
        return json.loads(result[0].text)
    raise AssertionError(f"tool result was {type(result).__name__}")


def _insert_running(db_path: Path, task_id: str) -> None:
    """
    Leave one running pin in a migrated database.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    """
    database = Database(db_path)
    connection = database.connect()
    database.migrate()
    Blackboard(database).insert_pin(connection, _queued_pin(task_id))
    assert Blackboard(database).claim(connection, task_id, "2026-10-04T18:40:02Z") is True
    connection.close()


def _queued_pin(task_id: str) -> TaskPin:
    """
    Build a queued pin the blackboard will accept.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: pin as TaskPin. Example: a pin with status "queued"
    """
    return TaskPin(
        task_id=task_id,
        priority=40,
        status="queued",
        model="qwen-14b",
        goal="Write a brief.",
        done_when="brief.md exists",
        access=AccessGrant("job"),
        phase="queued",
        created_at="2026-10-04T18:40:00Z",
    )


def _point_process_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Point the process database and job root at a temp directory.

    :param monkeypatch: env patcher as pytest.MonkeyPatch. Example: the test's monkeypatch
    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: None. Example: None
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCAL_AI_DB", str(tmp_path / "local-ai.sqlite"))
    monkeypatch.setenv("LOCAL_AI_JOBS", str(tmp_path / "jobs"))


def _capture_run(monkeypatch: pytest.MonkeyPatch, spoil_host: str | None = None) -> dict:
    """
    Replace mcp.run so main records the transport instead of binding.

    :param monkeypatch: env patcher as pytest.MonkeyPatch. Example: the test's monkeypatch
    :param spoil_host: host to set before run as str | None. Example: "0.0.0.0"
    :return: captured call as dict. Example: {"called": False}
    """
    seen: dict[str, object] = {"called": False}
    original = compose

    def wrapped(env: dict[str, str] | None = None) -> tuple[LocalAiServer, FastMCP]:
        server, mcp = original(env)
        if spoil_host is not None:
            mcp.settings.host = spoil_host

        def run(transport: str = "stdio", mount_path: str | None = None, **kwargs: object) -> None:
            seen["called"] = True
            seen["transport"] = transport
            seen["host"] = kwargs.get("host", mcp.settings.host)
            seen["port"] = kwargs.get("port", mcp.settings.port)
            seen["mount_path"] = mount_path

        mcp.run = run  # type: ignore[method-assign]
        return server, mcp

    monkeypatch.setattr("local_ai.dispatcher.server.compose", wrapped)
    return seen


def _kick_count(caplog: pytest.LogCaptureFixture) -> int:
    """
    Count Slot.kick entry lines.

    :param caplog: captured logs as pytest.LogCaptureFixture. Example: the test's caplog
    :return: entry count as int. Example: 1
    """
    return sum(1 for record in caplog.records if record.message == "Starting kick")


def _wait_until_holding(tmp_path: Path, task_id: str, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """
    Wait until the worker is blocked in the shell, so the test thread is not in sqlite with it.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: running pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param timeout: seconds to wait as float. Example: 5.0
    :return: None. Example: None
    :raises AssertionError: when awake.txt does not appear. Example: a denied shell
    """
    marker = tmp_path / "jobs" / task_id / "awake.txt"
    deadline = time.monotonic() + timeout
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError(f"hold marker missing path={marker}")
        time.sleep(0.01)


def _release_and_join(tmp_path: Path, server: LocalAiServer, task_id: str | None) -> None:
    """
    Let the holding shell exit, then wait until the slot thread stops.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param server: server under test as LocalAiServer. Example: the server from _server
    :param task_id: holding pin as str | None. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    """
    if task_id:
        release = tmp_path / "jobs" / task_id / "release"
        release.parent.mkdir(parents=True, exist_ok=True)
        release.write_text("1", encoding="utf-8")
    _join_slot(server)


def _join_slot(server: LocalAiServer, timeout: float = _JOIN_TIMEOUT_S) -> None:
    """
    Wait until the slot thread and any pin it starts next have stopped.

    :param server: server under test as LocalAiServer. Example: the server from _server
    :param timeout: seconds to wait as float. Example: 5.0
    :return: None. Example: None
    :raises AssertionError: when a worker is still alive at the deadline. Example: a stuck sleep
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = server._slot._thread
        if thread is None:
            return
        remaining = deadline - time.monotonic()
        thread.join(timeout=remaining)
        if thread.is_alive():
            break
        if server._slot._thread is None or server._slot._thread is thread:
            return
    raise AssertionError("slot thread did not finish")


def _contract(name: str) -> dict:
    """
    Load one section 18 example.

    :param name: file name as str. Example: "capacity_up.json"
    :return: example object as dict. Example: {"status": "ok"}
    """
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
