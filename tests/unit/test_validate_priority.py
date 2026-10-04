"""Leaf behavior for validate_priority. Other contract stubs stay uncalled."""

import logging

import pytest

from local_ai.contracts import InvalidPriority, validate_priority


def test_validate_priority_none_returns_50(caplog: pytest.LogCaptureFixture) -> None:
    """Omitted priority is the middle of the queue, 50."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    assert validate_priority(None) == 50
    assert "50" in _info_text(caplog)


def test_validate_priority_40_stays_40(caplog: pytest.LogCaptureFixture) -> None:
    """An in-range priority is returned unchanged."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    assert validate_priority(40) == 40
    assert "40" in _info_text(caplog)


@pytest.mark.parametrize("value", [1, 100])
def test_validate_priority_accepts_inclusive_ends(value: int) -> None:
    """1 is first and 100 is last. Both are legal."""
    assert validate_priority(value) == value


@pytest.mark.parametrize("value", [0, 101, -1])
def test_validate_priority_outside_range_raises(value: int, caplog: pytest.LogCaptureFixture) -> None:
    """0, 101, and negatives are outside 1..100."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    with pytest.raises(InvalidPriority):
        validate_priority(value)
    assert str(value) in _info_text(caplog)


def test_validate_priority_bool_raises() -> None:
    """True is an int subclass and must not become priority 1."""
    with pytest.raises(InvalidPriority):
        validate_priority(True)  # type: ignore[arg-type]


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    """Join INFO messages. Example: 'Accepted priority value=40'."""
    return " ".join(record.message for record in caplog.records if record.levelno == logging.INFO)
