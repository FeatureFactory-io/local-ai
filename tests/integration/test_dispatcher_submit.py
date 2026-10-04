"""DispatcherService copies job inputs into a real directory under tmp_path."""

import logging
from pathlib import Path

import pytest

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.plan.steps import PlanStore


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
