"""Worker loop against a real sqlite file, a real shell, and ScriptedLLM."""

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import (
    MAX_ROUNDS_PER_STEP,
    AccessGrant,
    Completion,
    StepDraft,
    TaskPin,
    ToolCall,
    ToolEnvelope,
)
from local_ai.db.database import Database
from local_ai.llm.port import ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker
from tests.support.agent_story import assert_agent_story
from tests.support.log_story import assert_log_story


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm_scripts"
_HAPPY_BEATS = (
    {"workflow": "planning"},
    {"tool": "shell"},
    {"effect": "step_completed"},
)
_FAIL_COMMAND = "python3 -c \"import sys; sys.stderr.write('boom'); raise SystemExit(4)\""


def test_parse_plan_with_numbered_content_returns_drafts(tmp_path: Path) -> None:
    """Numbered lines in content become step drafts in that order."""
    worker, _connection, _job_dir = _open_worker(tmp_path, ())

    drafts = worker.parse_plan(Completion(content="1. List chapter files\n2. Write the brief"))

    assert drafts == (
        StepDraft(1, "List chapter files"),
        StepDraft(2, "Write the brief"),
    )


def test_parse_plan_with_numbered_final_answer_prefers_that_text(tmp_path: Path) -> None:
    """A numbered final answer is the plan. Content is the fallback."""
    worker, _connection, _job_dir = _open_worker(tmp_path, ())

    drafts = worker.parse_plan(
        Completion(content="1. Ignore this", final_answer="1. List chapter files")
    )

    assert drafts == (StepDraft(1, "List chapter files"),)


def test_parse_plan_with_empty_completion_raises(tmp_path: Path) -> None:
    """A completion with no numbered line cannot start a plan."""
    worker, _connection, _job_dir = _open_worker(tmp_path, ())

    with pytest.raises(ValueError, match="no step"):
        worker.parse_plan(Completion())


def test_run_round_appends_note_runs_shell_and_logs_acting(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A note is stored, the shell runs, and the acting line names the shell."""
    command = "python3 -c \"open('marker.txt','w').write('marker')\""
    turn = Completion(thinking="Writing the marker.", tool_calls=(ToolCall("shell", command),))
    worker, connection, job_dir = _open_worker(tmp_path, (turn,))
    caplog.set_level(logging.INFO)

    returned = worker.run_round(_TASK_ID, 1, "Foundation\nShell\nTask", "qwen-14b")

    assert returned == turn
    assert worker._last_envelope == ToolEnvelope(success=True, result="", error="")
    assert (job_dir / "marker.txt").read_text(encoding="utf-8") == "marker"
    assert json.loads(_reasoning(connection))[-1]["text"] == "Writing the marker."
    assert_log_story(
        caplog.records,
        "worker.step",
        "acting",
        [f"task_id={_TASK_ID}", "step=1", "tool=shell"],
    )


def test_run_round_with_empty_note_does_not_append(tmp_path: Path) -> None:
    """An empty note leaves the reasoning column untouched."""
    turn = Completion(tool_calls=(ToolCall("shell", "printf marker"),))
    worker, connection, job_dir = _open_worker(tmp_path, (turn,))

    worker.run_round(_TASK_ID, 1, "Foundation\nShell\nTask", "qwen-14b")

    assert _reasoning(connection) == "[]"
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == "marker"


def test_run_round_with_failing_shell_keeps_the_envelope(tmp_path: Path) -> None:
    """A non-zero exit stays in the envelope and does not raise."""
    command = "python3 -c \"import sys; sys.stderr.write('boom'); raise SystemExit(4)\""
    turn = Completion(thinking="Running the check.", tool_calls=(ToolCall("shell", command),))
    worker, _connection, job_dir = _open_worker(tmp_path, (turn,))

    worker.run_round(_TASK_ID, 2, "Foundation\nShell\nTask", "qwen-14b")

    assert worker._last_envelope == ToolEnvelope(success=False, result="", error="boom")
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == "boom"


def test_run_step_with_final_answer_completes(tmp_path: Path) -> None:
    """A final answer marks the running step completed and stores the handoff."""
    turn = Completion(final_answer="Found ch-01.md")
    worker, connection, _job_dir = _open_worker(tmp_path, (turn,))
    _insert_pending(worker, connection, (StepDraft(1, "List chapter files"),))

    outcome = worker.run_step(_TASK_ID, 1, "qwen-14b")

    assert outcome == "completed"
    step = _steps(worker, connection)[0]
    assert step.status == "completed"
    assert step.llm_synthesis == "Found ch-01.md"
    assert _phase(worker, connection) == "acting"


def test_run_step_with_failed_shell_fails_the_step(tmp_path: Path) -> None:
    """A non-zero shell marks the step failed and keeps the stderr."""
    command = "python3 -c \"import sys; sys.stderr.write('boom'); raise SystemExit(4)\""
    turn = Completion(thinking="Running the check.", tool_calls=(ToolCall("shell", command),))
    worker, connection, _job_dir = _open_worker(tmp_path, (turn,))
    _insert_pending(worker, connection, (StepDraft(1, "Run the check"),))

    outcome = worker.run_step(_TASK_ID, 1, "qwen-14b")

    assert outcome == "failed"
    step = _steps(worker, connection)[0]
    assert step.status == "failed"
    assert step.error == "boom"
    assert worker._failed_command == command


def test_run_step_at_round_cap_fails_the_step(tmp_path: Path) -> None:
    """Twelve shell rounds without a final answer fail the step."""
    turns = tuple(
        Completion(tool_calls=(ToolCall("shell", "printf x"),))
        for _index in range(MAX_ROUNDS_PER_STEP)
    )
    worker, connection, _job_dir = _open_worker(tmp_path, turns)
    _insert_pending(worker, connection, (StepDraft(1, "Keep looking"),))

    outcome = worker.run_step(_TASK_ID, 1, "qwen-14b")

    assert outcome == "failed"
    step = _steps(worker, connection)[0]
    assert step.status == "failed"
    assert step.error == "max rounds without a final answer"


def test_force_final_with_answer_completes_the_step(tmp_path: Path) -> None:
    """A non-empty final answer completes the running step."""
    worker, connection, _job_dir = _open_worker(tmp_path, (Completion(final_answer="Found ch-01.md"),))
    _insert_pending(worker, connection, (StepDraft(1, "List chapter files"),))
    worker._plan.mark_running(connection, _TASK_ID, 1)

    outcome = worker.force_final(_TASK_ID, 1, "qwen-14b")

    assert outcome == "completed"
    assert _steps(worker, connection)[0].llm_synthesis == "Found ch-01.md"


def test_force_final_with_empty_answer_returns_failed(tmp_path: Path) -> None:
    """An empty answer returns failed and leaves the step running."""
    worker, connection, _job_dir = _open_worker(tmp_path, (Completion(),))
    _insert_pending(worker, connection, (StepDraft(1, "List chapter files"),))
    worker._plan.mark_running(connection, _TASK_ID, 1)

    outcome = worker.force_final(_TASK_ID, 1, "qwen-14b")

    assert outcome == "failed"
    assert _steps(worker, connection)[0].status == "running"


def test_run_step_without_tool_or_answer_calls_force_final_once(tmp_path: Path) -> None:
    """An empty round asks once. A second empty answer fails the step's round."""
    worker, connection, _job_dir = _open_worker(
        tmp_path,
        (Completion(), Completion(final_answer="Found ch-01.md")),
    )
    _insert_pending(worker, connection, (StepDraft(1, "List chapter files"),))

    outcome = worker.run_step(_TASK_ID, 1, "qwen-14b")

    assert outcome == "completed"
    assert _steps(worker, connection)[0].status == "completed"


def test_run_step_with_empty_force_final_returns_failed(tmp_path: Path) -> None:
    """Two empty turns ask once and return failed instead of looping."""
    worker, connection, _job_dir = _open_worker(tmp_path, (Completion(), Completion()))
    _insert_pending(worker, connection, (StepDraft(1, "List chapter files"),))

    outcome = worker.run_step(_TASK_ID, 1, "qwen-14b")

    assert outcome == "failed"
    assert _steps(worker, connection)[0].status == "running"


def test_note_identical_failure_third_same_command_returns_true(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The third identical command stops the pin. The command text stays out of the log."""
    worker, _connection, _job_dir = _open_worker(tmp_path, ())
    command = "weasyprint book.html book.pdf"
    caplog.set_level(logging.INFO)

    first = worker.note_identical_failure(_TASK_ID, 2, command)
    second = worker.note_identical_failure(_TASK_ID, 2, command)
    third = worker.note_identical_failure(_TASK_ID, 2, command)

    assert (first, second, third) == (False, False, True)
    assert_log_story(
        caplog.records,
        "worker.failure",
        "identical",
        [f"task_id={_TASK_ID}", "step=2", "count=3"],
    )
    assert command not in caplog.text


def test_note_identical_failure_different_command_resets_count(tmp_path: Path) -> None:
    """A different command starts the count again at one."""
    worker, _connection, _job_dir = _open_worker(tmp_path, ())

    assert worker.note_identical_failure(_TASK_ID, 2, "printf a") is False
    assert worker.note_identical_failure(_TASK_ID, 2, "printf a") is False
    assert worker.note_identical_failure(_TASK_ID, 2, "printf b") is False
    assert worker.note_identical_failure(_TASK_ID, 2, "printf b") is False
    assert worker.note_identical_failure(_TASK_ID, 2, "printf b") is True


def test_adapt_after_failure_inserts_replacement_and_keeps_failed_row(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A replacement step is inserted. The failed row stays failed."""
    worker, connection, _job_dir = _open_worker(
        tmp_path,
        (Completion(final_answer="Write the marker with python3"),),
    )
    _insert_pending(
        worker,
        connection,
        (StepDraft(1, "Run the check"), StepDraft(2, "Record the result")),
    )
    worker._plan.mark_running(connection, _TASK_ID, 1)
    worker._plan.fail_step(connection, _TASK_ID, 1, "boom")
    caplog.set_level(logging.INFO)

    worker.adapt_after_failure(_TASK_ID, 1, "qwen-14b", "boom")

    steps = _steps(worker, connection)
    assert [(step.order, step.action, step.status) for step in steps] == [
        (1, "Run the check", "failed"),
        (2, "Write the marker with python3", "pending"),
        (3, "Record the result", "pending"),
    ]
    pin = worker._blackboard.get_pin(connection, _TASK_ID)
    assert pin is not None
    assert "Write the marker with python3 (pending)." in pin.current_plan
    assert_log_story(caplog.records, "blackboard.set_phase", "phase=recovering", [f"task_id={_TASK_ID}"])
    assert_log_story(caplog.records, "worker.adapt", "inserted", [f"task_id={_TASK_ID}", "step=2"])


def test_worker_module_does_not_import_dispatcher() -> None:
    """The worker returns a status word. It does not import the dispatcher."""
    text = (Path(__file__).resolve().parents[2] / "local_ai" / "worker" / "loop.py").read_text(
        encoding="utf-8"
    )
    assert "from local_ai.dispatcher" not in text
    assert "import local_ai.dispatcher" not in text


def test_run_task_happy_path_records_agent_story(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Planning, the shell, and a completed step happen in that order."""
    worker, connection, job_dir = _open_worker(tmp_path, _script("prf-hybrid"))
    caplog.set_level(logging.INFO)

    outcome = worker.run_task(_TASK_ID)

    assert outcome == "completed"
    assert [(step.order, step.status) for step in _steps(worker, connection)] == [
        (1, "completed"),
        (2, "completed"),
    ]
    assert (job_dir / "marker.txt").read_text(encoding="utf-8") == "marker"
    assert_agent_story(worker.trace, _HAPPY_BEATS)
    assert_log_story(
        caplog.records,
        "worker.step",
        "acting",
        [f"task_id={_TASK_ID}", "step=1", "tool=shell"],
    )


def test_run_task_with_empty_plan_returns_failed(tmp_path: Path) -> None:
    """A planning turn with no numbered step and no shell call fails before any step runs."""
    worker, connection, _job_dir = _open_worker(tmp_path, (Completion(content="I cannot plan"),))

    outcome = worker.run_task(_TASK_ID)

    assert outcome == "failed"
    assert worker.failure_reason == "empty_plan"
    assert _steps(worker, connection) == ()


def test_run_task_with_shell_tool_and_no_numbered_plan_runs_that_command(tmp_path: Path) -> None:
    """A planning turn that is only a shell call writes the file and completes."""
    command = "python3 -c \"open('poem.md','w').write('python\\n')\""
    worker, connection, job_dir = _open_worker(
        tmp_path,
        (Completion(thinking="Writing poem.md.", tool_calls=(ToolCall("shell", command),)),),
    )

    outcome = worker.run_task(_TASK_ID)

    assert outcome == "completed"
    assert (job_dir / "poem.md").read_text(encoding="utf-8") == "python\n"
    assert [(step.order, step.status) for step in _steps(worker, connection)] == [(1, "completed")]


def test_run_task_stops_on_third_identical_failure(tmp_path: Path) -> None:
    """The third identical command returns failed and does not insert another step."""
    script = (
        Completion(content="1. Run the check"),
        Completion(thinking="Running the check.", tool_calls=(ToolCall("shell", _FAIL_COMMAND),)),
        Completion(final_answer="Retry the check"),
        Completion(thinking="Running the check.", tool_calls=(ToolCall("shell", _FAIL_COMMAND),)),
        Completion(final_answer="Retry the check again"),
        Completion(thinking="Running the check.", tool_calls=(ToolCall("shell", _FAIL_COMMAND),)),
    )
    worker, connection, _job_dir = _open_worker(tmp_path, script)

    outcome = worker.run_task(_TASK_ID)

    assert outcome == "failed"
    assert [step.status for step in _steps(worker, connection)] == ["failed", "failed", "failed"]


def test_run_task_recovery_keeps_failed_step_and_completes_the_replacement(tmp_path: Path) -> None:
    """A non-zero shell stays failed. The inserted step writes the marker and completes."""
    worker, connection, job_dir = _open_worker(tmp_path, _script("prf-recovery"))

    outcome = worker.run_task(_TASK_ID)

    assert outcome == "completed"
    steps = _steps(worker, connection)
    assert [(step.order, step.action, step.status) for step in steps] == [
        (1, "Run the check", "failed"),
        (2, "Write the marker file", "completed"),
    ]
    assert steps[0].error == "boom"
    assert (job_dir / "marker.txt").read_text(encoding="utf-8") == "marker"


def _open_worker(
    tmp_path: Path,
    script: tuple[Completion, ...],
) -> tuple[Worker, sqlite3.Connection, Path]:
    """
    Open a migrated database, a job directory, and a worker.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param script: model turns as tuple[Completion, ...]. Example: ()
    :return: worker, connection, and job directory. Example: (Worker(...), connection, Path("/tmp/jobs/id"))
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    job_dir = tmp_path / "jobs" / _TASK_ID
    job_dir.mkdir(parents=True)
    board = Blackboard(database)
    board.insert_pin(connection, _pin())
    worker = Worker(
        ScriptedLLM(script),
        ShellTool(GrantChecker()),
        PromptAssembler(),
        PlanStore(database),
        board,
        connection,
    )
    return worker, connection, job_dir


def _script(name: str) -> tuple[Completion, ...]:
    """
    Load one scripted-LLM fixture.

    :param name: fixture directory as str. Example: "prf-hybrid"
    :return: turns as tuple[Completion, ...]. Example: (Completion(content="1. Write the marker file"),)
    """
    payload = json.loads((_FIXTURES / name / "turns.json").read_text(encoding="utf-8"))
    return tuple(_completion(turn) for turn in payload["turns"])


def _completion(turn: dict) -> Completion:
    """
    Build one completion from a fixture turn.

    :param turn: fixture object as dict. Example: {"content": "1. Write the marker file"}
    :return: completion as Completion. Example: Completion(content="1. Write the marker file")
    """
    calls = tuple(
        ToolCall(str(item["name"]), str(item["command"]))
        for item in turn.get("tool_calls", [])
    )
    return Completion(
        content=str(turn.get("content", "")),
        thinking=str(turn.get("thinking", "")),
        tool_calls=calls,
        final_answer=str(turn.get("final_answer", "")),
    )


def _insert_pending(worker: Worker, connection: sqlite3.Connection, drafts: tuple[StepDraft, ...]) -> None:
    """
    Store pending steps for the open pin.

    :param worker: worker under test as Worker. Example: the worker from _open_worker
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :param drafts: steps as tuple[StepDraft, ...]. Example: (StepDraft(1, "List chapter files"),)
    :return: None. Example: None
    """
    worker._plan.insert_steps(connection, _TASK_ID, drafts)


def _steps(worker: Worker, connection: sqlite3.Connection):
    """
    Load the pin's steps in order.

    :param worker: worker under test as Worker. Example: the worker from _open_worker
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: steps as tuple. Example: ()
    """
    return worker._plan.list_steps(connection, _TASK_ID)


def _phase(worker: Worker, connection: sqlite3.Connection) -> str:
    """
    Return the pin phase.

    :param worker: worker under test as Worker. Example: the worker from _open_worker
    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: phase as str. Example: "acting"
    """
    pin = worker._blackboard.get_pin(connection, _TASK_ID)
    assert pin is not None
    return pin.phase


def _reasoning(connection: sqlite3.Connection) -> str:
    """
    Return the stored reasoning column.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: reasoning JSON as str. Example: "[]"
    """
    row = connection.execute("SELECT reasoning FROM tasks WHERE id = ?", (_TASK_ID,)).fetchone()
    assert row is not None
    return str(row["reasoning"])


def _pin() -> TaskPin:
    """
    Build one running pin with a job grant.

    :return: pin as TaskPin. Example: status running
    """
    return TaskPin(
        task_id=_TASK_ID,
        priority=40,
        status="running",
        model="qwen-14b",
        goal="Write a marker file in the job folder.",
        done_when="marker.txt exists",
        access=AccessGrant("job"),
        phase="queued",
        created_at="2026-10-04T18:40:00Z",
    )
