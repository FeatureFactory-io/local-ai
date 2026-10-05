"""Token-gated JCL routes on a real sqlite file and a ScriptedLLM worker."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import Completion, ToolCall
from local_ai.db.database import Database
from local_ai.dispatcher.server import LocalAiServer
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.jcl.app import create_app
from local_ai.jcl.tokens import load_token_table
from local_ai.llm.port import ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker
from tests.support.log_story import assert_log_story


_HOST = "http://127.0.0.1:11434"
_TAGS = f"{_HOST}/api/tags"
_PS = f"{_HOST}/api/ps"
_LANE_10 = "lane-10-secret"
_LANE_50 = "lane-50-secret"
_JOIN_TIMEOUT_S = 8.0
_HOLD_COMMAND = (
    "python3 -c \"import pathlib,time; pathlib.Path('awake.txt').write_text('1'); "
    "p=pathlib.Path('release.txt'); "
    "[time.sleep(0.05) for _ in range(300) if not p.exists()]\""
)
_HOLD_SCRIPT = (
    Completion(content="1. Pause in the job folder"),
    Completion(tool_calls=(ToolCall("shell", _HOLD_COMMAND),), final_answer="Paused"),
)
_DONE_SCRIPT = (
    Completion(content="1. Finish the brief"),
    Completion(
        tool_calls=(ToolCall("shell", "python3 -c \"open('brief.md','w').write('brief\\n')\""),),
        final_answer="The brief is done",
    ),
)
_GOAL = "Write a brief."
_DONE = "brief.md exists"
_TOKENS = load_token_table(
    {"LOCAL_AI_JCL_TOKEN_10": _LANE_10, "LOCAL_AI_JCL_TOKEN_50": _LANE_50}
)

Http = Callable[[str], tuple[int, dict]]


def test_missing_token_is_unauthorized(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A call with no Authorization header is rejected and does not pin a task."""
    client, _server = _open(tmp_path)
    caplog.set_level(logging.INFO)

    response = client.post("/jcl/capacity")

    assert response.status_code == 401
    assert response.get_json() == {"status": "rejected", "reason": "unauthorized"}
    assert_log_story(caplog.records, "jcl.auth", "rejected", ["reason=unauthorized", "route=/jcl/capacity"])


def test_unknown_token_is_unauthorized_and_is_not_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown bearer secret is the same rejection. The secret stays out of the log."""
    client, _server = _open(tmp_path)
    caplog.set_level(logging.INFO)
    secret = "not-a-configured-token"

    response = _post(client, "/jcl/list", secret, {})

    assert response.status_code == 401
    assert response.get_json()["reason"] == "unauthorized"
    assert secret not in caplog.text
    assert _LANE_50 not in caplog.text


def test_better_priority_is_forbidden_and_creates_no_pin(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lane-50 token cannot submit priority 10, and no pin is written."""
    client, _server = _open(tmp_path)
    caplog.set_level(logging.INFO)

    response = _post(client, "/jcl/submit", _LANE_50, _task(priority=10))
    listed = _post(client, "/jcl/list", _LANE_50, {})

    assert response.status_code == 403
    assert response.get_json() == {"status": "rejected", "reason": "priority_forbidden"}
    assert listed.status_code == 200
    assert listed.get_json()["count"] == 0
    assert_log_story(caplog.records, "jcl.priority", "rejected", ["reason=priority_forbidden", "ceiling=50"])
    assert _LANE_50 not in caplog.text


def test_job_control_routes_follow_the_dispatcher(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Submit, read, list, reprioritize, and cancel use the same dispatcher results."""
    client, server = _open(tmp_path, script=_HOLD_SCRIPT)
    caplog.set_level(logging.INFO)

    capacity = _post(client, "/jcl/capacity", _LANE_50, {})
    first = _post(client, "/jcl/submit", _LANE_50, _task())
    first_id = first.get_json()["task_id"]
    running = _post(client, "/jcl/read", _LANE_10, {"task_id": first_id})
    second = _post(client, "/jcl/submit", _LANE_50, _task(priority=80, goal="Index the archive."))
    second_id = second.get_json()["task_id"]
    queued = _post(client, "/jcl/list", _LANE_50, {"status": "queued"})
    forbidden = _post(
        client,
        "/jcl/reprioritize",
        _LANE_50,
        {"task_id": second_id, "priority": 10},
    )
    moved = _post(
        client,
        "/jcl/reprioritize",
        _LANE_50,
        {"task_id": second_id, "priority": 90},
    )
    cancelled = _post(client, "/jcl/cancel", _LANE_10, {"task_id": second_id})

    assert capacity.status_code == 200
    assert capacity.get_json()["status"] == "ok"
    assert capacity.get_json()["ollama"] == "up"
    assert first.status_code == 200
    assert first.get_json()["status"] == "queued"
    assert first.get_json()["priority"] == 50
    assert running.status_code == 200
    assert running.get_json()["status"] == "running"
    assert running.get_json()["priority"] == 50
    assert second.get_json()["status"] == "queued"
    assert second.get_json()["priority"] == 80
    assert queued.get_json()["count"] == 1
    assert queued.get_json()["jobs"][0]["task_id"] == second_id
    assert forbidden.status_code == 403
    assert moved.status_code == 200
    assert moved.get_json()["priority"] == 90
    assert moved.get_json()["status"] == "queued"
    assert cancelled.status_code == 200
    assert cancelled.get_json()["status"] == "cancelled"
    assert cancelled.get_json()["job_dir_removed"] is True
    assert_log_story(caplog.records, "jcl.submit", "exit", ["priority=50", "status=queued"])
    assert _LANE_10 not in caplog.text
    assert _LANE_50 not in caplog.text
    _release(tmp_path, first_id)
    _join(server)


def test_purge_removes_a_completed_pin(tmp_path: Path) -> None:
    """A finished pin is deleted through the purge route."""
    client, server = _open(tmp_path, script=_DONE_SCRIPT)

    submitted = _post(client, "/jcl/submit", _LANE_50, _task())
    task_id = submitted.get_json()["task_id"]
    _join(server)
    finished = _post(client, "/jcl/read", _LANE_50, {"task_id": task_id})
    purged = _post(client, "/jcl/purge", _LANE_50, {"task_id": task_id, "status": "completed"})

    assert finished.get_json()["status"] == "completed"
    assert purged.status_code == 200
    assert purged.get_json()["status"] == "ok"
    assert purged.get_json()["count"] == 1
    assert purged.get_json()["purged"][0]["task_id"] == task_id
    assert purged.get_json()["purged"][0]["job_dir_removed"] is True


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    """A body that is not a JSON object is HTTP 400 and does not reach the dispatcher."""
    client, _server = _open(tmp_path)

    response = client.post(
        "/jcl/submit",
        data=b"not-json",
        content_type="application/json",
        headers={"Authorization": f"Bearer {_LANE_50}"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"status": "rejected", "reason": "bad_json"}


def test_create_app_rejects_a_non_server() -> None:
    """The app is wired to a LocalAiServer, not a stand-in."""
    with pytest.raises(TypeError):
        create_app(None, _TOKENS)  # type: ignore[arg-type]


def _open(
    tmp_path: Path,
    script: tuple[Completion, ...] = (),
) -> tuple[FlaskClient, LocalAiServer]:
    """
    Build a JCL app on a temp database. Ollama is the scripted reader.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param script: model turns as tuple[Completion, ...]. Example: ()
    :return: client and server as tuple[FlaskClient, LocalAiServer]. Example: (client, server)
    """
    server = _server(tmp_path, script)
    return create_app(server, _TOKENS).test_client(), server


def _server(tmp_path: Path, script: tuple[Completion, ...]) -> LocalAiServer:
    """
    Wire the dispatcher the MCP tests wire, with a ScriptedLLM worker.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param script: model turns as tuple[Completion, ...]. Example: ()
    :return: server as LocalAiServer. Example: LocalAiServer(...)
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    board = Blackboard(database)
    plan = PlanStore(database)
    capacity = CapacityReader()
    http = _scripted_ollama()
    probe = lambda: (64.0, 40.2)
    words = lambda: "normal"
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


def _scripted_ollama() -> Http:
    """
    Answer tags and ps without opening a socket.

    :return: reader as Http. Example: lambda path: (200, {"models": []})
    """

    def ollama(url: str) -> tuple[int, dict]:
        if url == _TAGS:
            return 200, {"models": []}
        if url == _PS:
            return 200, {"models": [{"name": "qwen2.5:14b"}]}
        raise AssertionError(url)

    return ollama


def _post(client: FlaskClient, path: str, token: str, payload: dict) -> object:
    """
    POST one JCL route with a bearer token.

    :param client: flask client as FlaskClient. Example: app.test_client()
    :param path: route as str. Example: "/jcl/submit"
    :param token: bearer secret as str. Example: "lane-50-secret"
    :param payload: JSON body as dict. Example: {"goal": "Write a brief.", "done_when": "brief.md exists"}
    :return: response as object. Example: a response with status_code 200
    """
    return client.post(path, json=payload, headers={"Authorization": f"Bearer {token}"})


def _task(priority: int | None = None, goal: str = _GOAL) -> dict:
    """
    One submit body. Omitted priority lets the token choose the lane.

    :param priority: requested priority as int | None. Example: 80
    :param goal: worker goal as str. Example: "Write a brief."
    :return: body as dict. Example: {"goal": "Write a brief.", "done_when": "brief.md exists"}
    """
    body = {"goal": goal, "done_when": _DONE}
    if priority is not None:
        body["priority"] = priority
    return body


def _release(tmp_path: Path, task_id: str) -> None:
    """
    Let the holding worker exit. The shell waits until release.txt exists.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param task_id: running pin as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: None. Example: None
    """
    job = tmp_path / "jobs" / task_id
    job.mkdir(parents=True, exist_ok=True)
    (job / "release.txt").write_text("go", encoding="utf-8")


def _join(server: LocalAiServer) -> None:
    """
    Wait until the slot thread finishes. A thread that already exited is done.

    :param server: dispatcher surface as LocalAiServer. Example: the test server
    :return: None. Example: None
    """
    thread = server._slot._thread
    if thread is None:
        return
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not thread.is_alive()
