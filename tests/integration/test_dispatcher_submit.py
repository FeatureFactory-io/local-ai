"""Dispatcher submit sequence on a real sqlite file and a real job directory."""

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.contracts import InvalidModel, InvalidPriority, SubmitRequest
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.plan.steps import PlanStore
from tests.support.log_story import assert_log_story


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


def test_copy_job_inputs_with_job_mode_copies_existing_files_only(tmp_path: Path) -> None:
    """Mode job copies files that exist and leaves directories and missing paths behind."""
    service = _service(tmp_path)
    source = tmp_path / "notes.md"
    source.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.md"
    tree = tmp_path / "draft"
    tree.mkdir()
    (tree / "ch.md").write_text("chapter", encoding="utf-8")

    job_dir = service.copy_job_inputs(
        _TASK_ID,
        (str(source), str(missing), str(tree)),
        "job",
    )

    assert job_dir == tmp_path / "jobs" / _TASK_ID
    assert job_dir.is_dir()
    assert (job_dir / "notes.md").read_text(encoding="utf-8") == "hello"
    assert source.read_text(encoding="utf-8") == "hello"
    assert not (job_dir / "missing.md").exists()
    assert not (job_dir / "draft").exists()
    assert not (job_dir / "ch.md").exists()


def test_copy_job_inputs_with_paths_mode_creates_the_directory_and_does_not_copy(
    tmp_path: Path,
) -> None:
    """Mode paths still makes the job folder and does not copy the granted tree."""
    service = _service(tmp_path)
    source = tmp_path / "notes.md"
    source.write_text("hello", encoding="utf-8")

    job_dir = service.copy_job_inputs(_TASK_ID, (str(source),), "paths")

    assert job_dir.is_dir()
    assert list(job_dir.iterdir()) == []
    assert source.read_text(encoding="utf-8") == "hello"


def test_copy_job_inputs_with_home_mode_creates_the_directory_and_does_not_copy(
    tmp_path: Path,
) -> None:
    """Mode home still makes the job folder and does not copy the home tree."""
    service = _service(tmp_path)
    source = tmp_path / "notes.md"
    source.write_text("hello", encoding="utf-8")

    job_dir = service.copy_job_inputs(_TASK_ID, (str(source),), "home")

    assert job_dir.is_dir()
    assert list(job_dir.iterdir()) == []


def test_copy_job_inputs_logs_the_directory_at_info(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The job directory path is on one INFO line."""
    service = _service(tmp_path)
    caplog.set_level(logging.INFO)

    job_dir = service.copy_job_inputs(_TASK_ID, (), "job")

    assert any(
        record.levelno == logging.INFO and str(job_dir) in record.getMessage()
        for record in caplog.records
    )


def test_copy_job_inputs_with_blank_task_id_raises(tmp_path: Path) -> None:
    """A blank task id is rejected before a directory is created."""
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="task_id"):
        service.copy_job_inputs("  ", (), "job")

    assert not (tmp_path / "jobs").exists()


def test_copy_job_inputs_with_unknown_grant_mode_raises(tmp_path: Path) -> None:
    """An unknown grant mode does not create a job directory."""
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="grant_mode"):
        service.copy_job_inputs(_TASK_ID, (), "root")

    assert not (tmp_path / "jobs" / _TASK_ID).exists()


def test_submit_with_ollama_down_rejects_and_does_not_insert(tmp_path: Path) -> None:
    """A down daemon returns ollama_down and leaves the tasks table empty."""
    service, connection = _dispatcher(tmp_path, tags_status=503)
    request = SubmitRequest(
        "Write a brief.",
        "brief.md exists",
        priority=40,
        model="qwen-14b",
    )

    body = service.submit(request, connection)

    assert body == {"status": "rejected", "reason": "ollama_down", "task_id": None}
    assert _task_count(connection) == 0
    assert not (tmp_path / "jobs").exists()


def test_submit_with_recommended_none_and_no_model_rejects(tmp_path: Path) -> None:
    """Below 8 GB with no model returns model_unavailable and does not insert."""
    service, connection = _dispatcher(tmp_path, free_gb=7.0)

    body = service.submit(SubmitRequest("Write a brief.", "brief.md exists", priority=40), connection)

    assert body == {"status": "rejected", "reason": "model_unavailable", "task_id": None}
    assert _task_count(connection) == 0
    assert not (tmp_path / "jobs").exists()


def test_submit_with_named_model_when_recommended_is_none_queues(tmp_path: Path) -> None:
    """A caller-named model is stored when the recommendation is none."""
    service, connection = _dispatcher(tmp_path, free_gb=7.0, clock=lambda: "2026-10-04T18:40:00Z")

    body = service.submit(
        SubmitRequest("Write a brief.", "brief.md exists", priority=40, model="qwen-8b"),
        connection,
    )

    assert body["status"] == "queued"
    assert body["model"] == "qwen-8b"
    assert body["position"] == 1
    assert _task_count(connection) == 1


def test_submit_queues_the_pin_and_logs_the_entry(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An up daemon inserts a queued pin and logs dispatcher.submit at INFO."""
    service, connection = _dispatcher(tmp_path, clock=lambda: "2026-10-04T18:40:00Z")
    source = tmp_path / "notes.md"
    source.write_text("hello", encoding="utf-8")
    caplog.set_level(logging.INFO)
    request = SubmitRequest(
        "Write a brief.",
        "brief.md exists",
        priority=40,
        model="qwen-14b",
        inputs=(str(source),),
    )

    body = service.submit(request, connection)

    assert body["status"] == "queued"
    assert body["priority"] == 40
    assert body["model"] == "qwen-14b"
    assert body["position"] == 1
    assert body["access"] == {"mode": "job"}
    task_id = body["task_id"]
    assert isinstance(task_id, str) and task_id != ""
    row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["status"] == "queued"
    assert row["phase"] == "queued"
    assert row["priority"] == 40
    assert row["model"] == "qwen-14b"
    assert row["goal"] == "Write a brief."
    assert row["done_when"] == "brief.md exists"
    assert row["created_at"] == "2026-10-04T18:40:00Z"
    assert (tmp_path / "jobs" / task_id / "notes.md").read_text(encoding="utf-8") == "hello"
    assert_log_story(
        caplog.records,
        "dispatcher.submit",
        "entry",
        [f"task_id={task_id}", "priority=40", "model=qwen-14b", "grant=job"],
    )


def test_submit_with_omitted_model_and_priority_uses_the_defaults(tmp_path: Path) -> None:
    """Omitted model uses recommended. Omitted priority is 50. Omitted access is job."""
    service, connection = _dispatcher(tmp_path, free_gb=20.0)

    body = service.submit(SubmitRequest("Write a brief.", "brief.md exists"), connection)

    assert body["status"] == "queued"
    assert body["model"] == "qwen-24b"
    assert body["priority"] == 50
    assert body["access"] == {"mode": "job"}
    assert body["position"] == 1


def test_submit_with_paths_grant_creates_the_job_directory_and_does_not_copy(
    tmp_path: Path,
) -> None:
    """Mode paths is stored on the pin and the input file stays outside the job folder."""
    service, connection = _dispatcher(tmp_path)
    books = tmp_path / "books"
    books.mkdir()
    chapter = books / "ch-01.md"
    chapter.write_text("chapter", encoding="utf-8")

    body = service.submit(
        SubmitRequest(
            "Summarize the book.",
            "summary.md exists",
            priority=40,
            model="qwen-14b",
            inputs=(str(chapter),),
            access={"mode": "paths", "paths": [str(books)]},
        ),
        connection,
    )

    job_dir = tmp_path / "jobs" / body["task_id"]
    assert body["access"] == {"mode": "paths", "paths": [str(books)]}
    assert body["position"] == 1
    assert job_dir.is_dir()
    assert list(job_dir.iterdir()) == []
    assert chapter.read_text(encoding="utf-8") == "chapter"


def test_submit_second_pin_reports_its_queue_position(tmp_path: Path) -> None:
    """A later, lower-priority pin is position 2 behind the pin already queued."""
    stamps = iter(["2026-10-04T18:40:00Z", "2026-10-04T18:41:00Z"])
    service, connection = _dispatcher(tmp_path, clock=lambda: next(stamps))

    first = service.submit(
        SubmitRequest("First.", "one.md exists", priority=40, model="qwen-14b"),
        connection,
    )
    second = service.submit(
        SubmitRequest("Second.", "two.md exists", priority=80, model="qwen-8b"),
        connection,
    )

    assert first["position"] == 1
    assert second["position"] == 2
    assert second["priority"] == 80
    assert _task_count(connection) == 2


def test_submit_with_priority_outside_range_does_not_insert(tmp_path: Path) -> None:
    """An illegal priority raises and leaves the queue empty."""
    service, connection = _dispatcher(tmp_path)

    with pytest.raises(InvalidPriority):
        service.submit(
            SubmitRequest("Write a brief.", "brief.md exists", priority=0, model="qwen-14b"),
            connection,
        )

    assert _task_count(connection) == 0
    assert not (tmp_path / "jobs").exists()


def test_submit_with_unknown_model_does_not_insert(tmp_path: Path) -> None:
    """An unknown model raises and leaves the queue empty."""
    service, connection = _dispatcher(tmp_path)

    with pytest.raises(InvalidModel):
        service.submit(
            SubmitRequest("Write a brief.", "brief.md exists", priority=40, model="qwen-70b"),
            connection,
        )

    assert _task_count(connection) == 0


def _service(tmp_path: Path) -> DispatcherService:
    """
    Build a service whose job root is under tmp_path.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: service as DispatcherService. Example: DispatcherService(...)
    """
    database = Database(tmp_path / "local-ai.sqlite")
    database.connect()
    database.migrate()
    jobs_root = tmp_path / "jobs"
    return DispatcherService(
        database,
        Blackboard(database),
        PlanStore(database),
        CapacityReader(),
        jobs_root,
    )


def _dispatcher(
    tmp_path: Path,
    *,
    free_gb: float = 14.0,
    tags_status: int = 200,
    clock: Callable[[], str] | None = None,
) -> tuple[DispatcherService, sqlite3.Connection]:
    """
    Open a migrated database and a service with an injected Ollama reader.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :param free_gb: free gigabytes as float. Example: 14.0
    :param tags_status: /api/tags status as int. Example: 200
    :param clock: timestamp source as Callable[[], str] | None. Example: lambda: "2026-10-04T18:40:00Z"
    :return: service and connection as tuple. Example: (DispatcherService(...), connection)
    """
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    database.migrate()
    service = DispatcherService(
        database,
        Blackboard(database),
        PlanStore(database),
        CapacityReader(),
        tmp_path / "jobs",
        memory=lambda: (64.0, free_gb),
        ollama=_ollama(tags_status),
        pressure=lambda: "normal",
        ollama_host="http://127.0.0.1:11434",
        clock=clock,
    )
    return service, connection


def _ollama(tags_status: int) -> Callable[[str], tuple[int, dict]]:
    """
    Script tags and ps. A non-200 tags status is Ollama down.

    :param tags_status: /api/tags status as int. Example: 503
    :return: HTTP reader as Callable. Example: lambda path: (200, {"models": []})
    """

    def read(url: str) -> tuple[int, dict]:
        if url.endswith("/api/tags"):
            return tags_status, {"models": []}
        if url.endswith("/api/ps"):
            return 200, {"models": [{"name": "qwen2.5:14b"}]}
        raise AssertionError(url)

    return read


def _task_count(connection: sqlite3.Connection) -> int:
    """
    Count tasks rows.

    :param connection: open connection as sqlite3.Connection. Example: the Database connection
    :return: row count as int. Example: 0
    """
    row = connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
    assert row is not None
    return int(row["n"])
