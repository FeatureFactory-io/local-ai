"""Leaf behavior for queue_position. Only queued pins take a place."""

import logging

import pytest

from local_ai.contracts import AccessGrant, TaskPin
from local_ai.dispatcher.queue import queue_position


def test_priority_1_is_before_priority_100(caplog: pytest.LogCaptureFixture) -> None:
    """Priority 1 is first even when it was created later than priority 100."""
    caplog.set_level(logging.INFO, logger="local_ai.dispatcher.queue")
    first = _pin("first", 1, created_at="2026-10-04T19:00:00Z")
    last = _pin("last", 100, created_at="2026-10-04T18:00:00Z")
    queued = (last, first)
    assert queue_position(first, queued) == 1
    assert queue_position(last, queued) == 2
    assert "first" in _info_text(caplog)
    assert "last" in _info_text(caplog)


def test_equal_priority_keeps_created_at_order() -> None:
    """The earlier created_at stays ahead when priority matches."""
    earlier = _pin("earlier", 40, created_at="2026-10-04T18:00:00Z")
    middle = _pin("middle", 40, created_at="2026-10-04T19:00:00Z")
    later = _pin("later", 40, created_at="2026-10-04T20:00:00Z")
    queued = (later, earlier, middle)
    assert queue_position(earlier, queued) == 1
    assert queue_position(middle, queued) == 2
    assert queue_position(later, queued) == 3


def test_position_is_one_based_among_queued_pins_only() -> None:
    """Running, failed, and completed pins do not take a queue place."""
    running = _pin("running", 1, status="running", created_at="2026-10-04T17:00:00Z")
    failed = _pin("failed", 1, status="failed", created_at="2026-10-04T17:00:00Z")
    completed = _pin("completed", 1, status="completed", created_at="2026-10-04T17:00:00Z")
    ahead = _pin("ahead", 10, created_at="2026-10-04T18:00:00Z")
    target = _pin("target", 80, created_at="2026-10-04T19:00:00Z")
    queued = (running, failed, completed, target, ahead)
    assert queue_position(ahead, queued) == 1
    assert queue_position(target, queued) == 2


def _pin(
    task_id: str,
    priority: int,
    status: str = "queued",
    created_at: str = "2026-10-04T18:40:00Z",
) -> TaskPin:
    """Build a pin for queue order. Example: task_id 'first', priority 1."""
    return TaskPin(
        task_id=task_id,
        priority=priority,
        status=status,
        model="qwen-14b",
        goal="Summarize the book into 2-3 pages.",
        done_when="summary.md exists",
        access=AccessGrant(mode="job"),
        phase="queued",
        created_at=created_at,
    )


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    """Join INFO messages. Example: 'Queue position task_id=first position=1'."""
    return " ".join(record.message for record in caplog.records if record.levelno == logging.INFO)
