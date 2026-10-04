"""Leaf behavior for configure_logging. Stdout stays free of log handlers."""

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_ai.logging_setup import configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """Put the root logger back so later tests keep their own handlers."""
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


def test_second_call_renames_previous_log_aside(tmp_path: Path) -> None:
    """A restart keeps the old lines in the aside file and starts an empty log."""
    log_path = tmp_path / "logs" / "app.log"
    configure_logging(log_path)
    logging.getLogger("local_ai.probe").info("old-line-marker")
    configure_logging(log_path)
    logging.getLogger("local_ai.probe").info("new-line-marker")
    _flush_root()
    current = log_path.read_text(encoding="utf-8")
    aside = (tmp_path / "logs" / "app.log.1").read_text(encoding="utf-8")
    assert "old-line-marker" in aside
    assert "old-line-marker" not in current
    assert "new-line-marker" in current


def test_record_can_carry_task_id(tmp_path: Path) -> None:
    """A call may pass task_id, and the line also names the logger."""
    log_path = tmp_path / "app.log"
    configure_logging(log_path)
    logging.getLogger("local_ai.probe").info("carrying", extra={"task_id": "7c1e0a2e"})
    _flush_root()
    text = log_path.read_text(encoding="utf-8")
    assert "task_id=7c1e0a2e" in text
    assert "local_ai.probe" in text
    assert "carrying" in text


def test_record_without_task_id_formats_empty(tmp_path: Path) -> None:
    """A call with no task_id still formats. The field is an empty string."""
    log_path = tmp_path / "app.log"
    configure_logging(log_path)
    logging.getLogger("local_ai.probe").info("no-task")
    _flush_root()
    text = log_path.read_text(encoding="utf-8")
    assert "no-task" in text
    assert "task_id=" in text
    assert "task_id=7" not in text


def test_stdout_is_not_a_handler(tmp_path: Path) -> None:
    """Log records go to stderr and the file. stdout has no handler."""
    configure_logging(tmp_path / "app.log")
    streams = [getattr(handler, "stream", None) for handler in logging.getLogger().handlers]
    assert sys.stdout not in streams
    assert sys.stderr in streams


def _flush_root() -> None:
    """Flush root handlers so the assertions read a complete file."""
    for handler in logging.getLogger().handlers:
        handler.flush()
