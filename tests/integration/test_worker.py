"""Worker loop against a real sqlite file, a real shell, and ScriptedLLM."""

import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.contracts import AccessGrant, Completion, StepDraft, TaskPin
from local_ai.db.database import Database
from local_ai.llm.port import ScriptedLLM
from local_ai.plan.steps import PlanStore
from local_ai.prompt.assemble import PromptAssembler
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool
from local_ai.worker.loop import Worker


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


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
