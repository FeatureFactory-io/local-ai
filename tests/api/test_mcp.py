"""MCP tool surface. Readers are injected. Ollama is not contacted."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.db.database import Database
from local_ai.dispatcher.server import LocalAiServer, http_port_from_env
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.llm.port import ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
_HOST = "http://127.0.0.1:11434"
_TAGS = f"{_HOST}/api/tags"
_PS = f"{_HOST}/api/ps"

Http = Callable[[str], tuple[int, dict]]
Memory = Callable[[], tuple[float, float]]
Pressure = Callable[[], str]


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
) -> LocalAiServer:
    """
    Wire a server on a temp database. Readers are the only stand-ins.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param reader: capacity reader as CapacityReader | None. Example: CapacityReader()
    :param memory: gigabyte probe as Memory | None. Example: lambda: (64.0, 40.2)
    :param ollama: HTTP reader as Http | None. Example: lambda path: (200, {"models": []})
    :param pressure: pressure word as Pressure | None. Example: lambda: "normal"
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
    worker = Worker(ScriptedLLM(()), ShellTool(GrantChecker()), PromptAssembler(), plan, board, connection)
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


def _contract(name: str) -> dict:
    """
    Load one section 18 example.

    :param name: file name as str. Example: "capacity_up.json"
    :return: example object as dict. Example: {"status": "ok"}
    """
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
