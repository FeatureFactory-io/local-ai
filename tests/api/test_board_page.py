"""Read-only blackboard page on a real sqlite file and the Flask test client."""

import logging
import sqlite3
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.board.app import bind_settings, create_app
from local_ai.contracts import AccessGrant, ReasoningNote, StepDraft, TaskPin
from local_ai.db.database import Database
from local_ai.plan.steps import PlanStore


_RUNNING_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
_QUEUED_ID = "11111111-2222-4333-8444-555555555555"
_DONE_ID = "22222222-3333-4444-8555-666666666666"
_SECRET_PATH = "/tmp/secret-token-do-not-render"
_PLAN = "1. List chapter files (running)."
_NOTE = "Reading ch-04.md."
_STEP = "List chapter files"
_QUEUED_GOAL = "Write <script>alert(1)</script> the glossary."


def test_empty_board_says_there_are_no_jobs(tmp_path: Path) -> None:
    """A migrated database with no pins renders an empty board."""
    client = create_app(_empty_db(tmp_path)).test_client()

    response = client.get("/")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "No jobs on the blackboard." in text
    assert 'data-testid="job"' not in text
    assert client.post("/").status_code == 405


def test_missing_database_is_empty_and_is_not_created(tmp_path: Path) -> None:
    """A missing sqlite file stays missing. The page does not migrate."""
    missing = tmp_path / "local-ai.sqlite"
    client = create_app(missing).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "No jobs on the blackboard." in response.get_data(as_text=True)
    assert not missing.exists()


def test_filter_queued_hides_completed_and_completed_hides_queued(tmp_path: Path) -> None:
    """queued shows the waiting pin and its position. completed shows the finished pin."""
    database = _seed_queued_and_completed(tmp_path)
    client = create_app(database).test_client()
    before = _task_ids(database)

    everything = client.get("/")
    explicit_all = client.get("/?status=all")
    queued = client.get("/?status=queued")
    completed = client.get("/?status=completed")

    assert everything.status_code == 200
    all_text = everything.get_data(as_text=True)
    assert explicit_all.get_data(as_text=True) == all_text
    assert _QUEUED_ID in all_text
    assert _DONE_ID in all_text
    assert "0 running, 1 queued" in all_text

    assert queued.status_code == 200
    queued_text = queued.get_data(as_text=True)
    assert _QUEUED_ID in queued_text
    assert _DONE_ID not in queued_text
    assert 'data-testid="job-position">1<' in queued_text
    assert "<script>" not in queued_text
    assert "&lt;script&gt;" in queued_text
    assert completed.status_code == 200
    completed_text = completed.get_data(as_text=True)
    assert _DONE_ID in completed_text
    assert _QUEUED_ID not in completed_text
    assert "chapter missing" in completed_text
    assert 'data-testid="job-position"' not in completed_text
    assert "<details open" not in completed_text
    assert _task_ids(database) == before


def test_running_pin_shows_plan_and_reasoning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A running pin shows current_plan, the step, and the reasoning note."""
    database = _seed_running(tmp_path)
    caplog.set_level(logging.INFO)
    client = create_app(database).test_client()

    response = client.get("/")

    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert _PLAN in text
    assert _NOTE in text
    assert _STEP in text
    assert "acting" in text
    assert 'data-testid="job-model">qwen-14b<' in text
    assert "2026-10-04T18:40:02Z" in text
    assert "<details open" in text
    assert _SECRET_PATH not in text
    assert any("board.render | ok" in record.message for record in caplog.records)
    assert all(_NOTE not in record.message for record in caplog.records)
    assert all(_SECRET_PATH not in record.message for record in caplog.records)


def test_unknown_filter_is_rejected_and_pins_stay(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown status is HTTP 400 with bad_status. Pins are not listed or changed."""
    database = _seed_queued_and_completed(tmp_path)
    before = _task_ids(database)
    caplog.set_level(logging.INFO)
    client = create_app(database).test_client()

    response = client.get("/?status=done")

    assert response.status_code == 400
    text = response.get_data(as_text=True)
    assert "bad_status" in text
    assert _QUEUED_ID not in text
    assert _DONE_ID not in text
    assert 'data-testid="job"' not in text
    assert _task_ids(database) == before
    assert any("reason=bad_status" in record.message for record in caplog.records)
    assert all("Write <script>" not in record.message for record in caplog.records)


def test_bind_settings_stays_on_loopback() -> None:
    """The page always binds 127.0.0.1. The port defaults to 8766."""
    assert bind_settings({}) == ("127.0.0.1", 8766)
    assert bind_settings({"LOCAL_AI_BOARD_PORT": "8770", "HOST": "0.0.0.0"}) == ("127.0.0.1", 8770)
    with pytest.raises(ValueError):
        bind_settings({"LOCAL_AI_BOARD_PORT": "0"})


def _empty_db(tmp_path: Path) -> Path:
    """
    Migrate an empty sqlite file and close it.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: sqlite path as Path. Example: Path("/tmp/pytest/local-ai.sqlite")
    """
    path = tmp_path / "local-ai.sqlite"
    database = Database(path)
    connection = database.connect()
    database.migrate()
    _checkpoint_and_close(connection)
    return path


def _seed_running(tmp_path: Path) -> Path:
    """
    Store one running pin with a plan, a step, and a reasoning note.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: sqlite path as Path. Example: Path("/tmp/pytest/local-ai.sqlite")
    """
    path = tmp_path / "local-ai.sqlite"
    database = Database(path)
    connection = database.connect()
    database.migrate()
    Blackboard(database).insert_pin(connection, _running_pin())
    PlanStore(database).insert_steps(connection, _RUNNING_ID, (StepDraft(1, _STEP),))
    connection.execute(
        "UPDATE plan_steps SET status = ? WHERE task_id = ?",
        ("running", _RUNNING_ID),
    )
    connection.commit()
    _checkpoint_and_close(connection)
    return path


def _seed_queued_and_completed(tmp_path: Path) -> Path:
    """
    Store one queued pin and one completed pin.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: sqlite path as Path. Example: Path("/tmp/pytest/local-ai.sqlite")
    """
    path = tmp_path / "local-ai.sqlite"
    database = Database(path)
    connection = database.connect()
    database.migrate()
    board = Blackboard(database)
    board.insert_pin(connection, _queued_pin())
    board.insert_pin(connection, _completed_pin())
    _checkpoint_and_close(connection)
    return path


def _running_pin() -> TaskPin:
    """
    Build the running pin. The access path must not appear on the page.

    :return: pin as TaskPin. Example: a pin with status "running"
    """
    return TaskPin(
        task_id=_RUNNING_ID,
        priority=40,
        status="running",
        model="qwen-14b",
        goal="Summarize the book into 2-3 pages.",
        done_when="summary.md exists",
        access=AccessGrant("paths", (_SECRET_PATH,)),
        phase="acting",
        current_plan=_PLAN,
        reasoning=(ReasoningNote("2026-10-04T18:40:01Z", 1, _NOTE),),
        artifacts=(_SECRET_PATH,),
        created_at="2026-10-04T18:40:00Z",
        started_at="2026-10-04T18:40:02Z",
    )


def _queued_pin() -> TaskPin:
    """
    Build the queued pin. Its goal includes markup that must be escaped.

    :return: pin as TaskPin. Example: a pin with status "queued"
    """
    return TaskPin(
        task_id=_QUEUED_ID,
        priority=10,
        status="queued",
        model="qwen-8b",
        goal=_QUEUED_GOAL,
        done_when="glossary.md exists",
        access=AccessGrant("job"),
        phase="queued",
        created_at="2026-10-04T18:41:00Z",
    )


def _completed_pin() -> TaskPin:
    """
    Build the completed pin with an error string the page should show.

    :return: pin as TaskPin. Example: a pin with status "completed"
    """
    return TaskPin(
        task_id=_DONE_ID,
        priority=80,
        status="completed",
        model="qwen-24b",
        goal="Combine chapter markdown into book.md.",
        done_when="book.md exists",
        access=AccessGrant("job"),
        phase="completed",
        error="chapter missing",
        created_at="2026-10-04T19:00:00Z",
        started_at="2026-10-04T19:01:00Z",
        finished_at="2026-10-04T19:10:00Z",
        final_status="completed",
    )


def _checkpoint_and_close(connection: sqlite3.Connection) -> None:
    """
    Flush WAL into the database file and close the writer.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: None. Example: None
    """
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()


def _task_ids(path: Path) -> list[str]:
    """
    Read pin ids without writing.

    :param path: sqlite file as Path. Example: Path("/tmp/pytest/local-ai.sqlite")
    :return: ids as list[str]. Example: ["11111111-2222-4333-8444-555555555555"]
    """
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT id FROM tasks ORDER BY id").fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]
