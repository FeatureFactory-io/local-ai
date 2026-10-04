"""Contract for assert_log_story. One record must hold the whole beat."""

import logging

import pytest

from tests.support.log_story import assert_log_story


def _record(message: str, level: int = logging.INFO) -> logging.LogRecord:
    """Build one captured record. Example message: 'dispatcher.submit | queued'."""
    return logging.LogRecord(
        name="local_ai.test",
        level=level,
        pathname="service.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_assert_log_story_with_matching_info_record_passes() -> None:
    """A later INFO line can carry where, beat, and the required fragment."""
    records = [
        _record("dispatcher.submit | entry"),
        _record("dispatcher.submit | queued | task_id=7c1e"),
    ]
    assert_log_story(records, "dispatcher.submit", "queued", ["task_id=7c1e"])


def test_assert_log_story_with_mapping_record_passes() -> None:
    """Callers may pass mappings with message and level instead of LogRecord."""
    records = [{"message": "worker.step | acting | tool=shell", "level": "INFO"}]
    assert_log_story(records, "worker.step", "acting", ["tool=shell"])


def test_assert_log_story_with_empty_must_include_passes() -> None:
    """Where and beat are enough when the story names no extra fragment."""
    records = [_record("dispatcher.finish | completed")]
    assert_log_story(records, "dispatcher.finish", "completed", [])


def test_assert_log_story_with_missing_beat_raises() -> None:
    """The failure names the beat that never shared a line with where."""
    records = [_record("dispatcher.submit | entry | task_id=7c1e")]
    with pytest.raises(AssertionError, match="queued"):
        assert_log_story(records, "dispatcher.submit", "queued", ["task_id=7c1e"])


def test_assert_log_story_with_fragment_on_another_record_raises() -> None:
    """Fragments split across lines do not count as one beat."""
    records = [
        _record("dispatcher.submit | queued"),
        _record("task_id=7c1e"),
    ]
    with pytest.raises(AssertionError, match="task_id=7c1e"):
        assert_log_story(records, "dispatcher.submit", "queued", ["task_id=7c1e"])


def test_assert_log_story_with_debug_record_when_info_required_raises() -> None:
    """A DEBUG line does not satisfy a story declared at INFO."""
    records = [_record("worker.step | acting | tool=shell", logging.DEBUG)]
    with pytest.raises(AssertionError, match="INFO"):
        assert_log_story(records, "worker.step", "acting", ["tool=shell"])


def test_assert_log_story_with_warning_level_passes() -> None:
    """A reject path can require WARNING instead of the INFO default."""
    records = [_record("dispatcher.submit | rejected | reason=ollama_down", logging.WARNING)]
    assert_log_story(
        records,
        "dispatcher.submit",
        "rejected",
        ["reason=ollama_down"],
        level="WARNING",
    )


def test_assert_log_story_with_blank_where_raises() -> None:
    """A blank location would match every line."""
    with pytest.raises(ValueError, match="where"):
        assert_log_story([], "  ", "queued", [])


def test_assert_log_story_with_unknown_level_raises() -> None:
    """Only the standard level names are accepted."""
    with pytest.raises(ValueError, match="level"):
        assert_log_story([], "dispatcher.submit", "queued", [], level="verbose")


def test_assert_log_story_with_string_must_include_raises() -> None:
    """A string must not be walked character by character."""
    with pytest.raises(TypeError, match="must_include"):
        assert_log_story([_record("dispatcher.submit | queued")], "dispatcher.submit", "queued", "task")  # type: ignore[arg-type]
