"""Leaf behavior for CapacityReader.recommend and read_pressure."""

import logging
import math

import pytest

from local_ai.capacity.reader import CapacityReader


def _just_under(edge: float) -> float:
    """Float immediately below an inclusive threshold. Example: 19.999999999999996."""
    return math.nextafter(edge, 0.0)


@pytest.mark.parametrize(
    ("free_gb", "expected"),
    [
        (20.0, "qwen-24b"),
        (_just_under(20.0), "qwen-14b"),
        (12.0, "qwen-14b"),
        (_just_under(12.0), "qwen-8b"),
        (8.0, "qwen-8b"),
        (_just_under(8.0), "none"),
        (0.0, "none"),
        (64.0, "qwen-24b"),
    ],
)
def test_recommend_maps_free_gb_at_and_just_under_each_edge(free_gb: float, expected: str) -> None:
    """Inclusive edges pick that band. The next float down picks the smaller band."""
    assert CapacityReader().recommend(free_gb) == expected


@pytest.mark.parametrize("free_gb", [-0.1, -1.0, -20.0])
def test_recommend_with_negative_free_gb_raises_value_error(free_gb: float) -> None:
    """A negative reading is not a size band."""
    with pytest.raises(ValueError):
        CapacityReader().recommend(free_gb)


@pytest.mark.parametrize("free_gb", [math.nan, math.inf, -math.inf, True, False, "20"])
def test_recommend_with_non_finite_free_gb_raises_value_error(free_gb: object) -> None:
    """NaN, infinity, bool, and non-numbers are rejected before a band is chosen."""
    with pytest.raises(ValueError):
        CapacityReader().recommend(free_gb)  # type: ignore[arg-type]


def test_recommend_logs_chosen_band_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The band decision is an INFO line so a later submit can be traced."""
    caplog.set_level(logging.INFO, logger="local_ai.capacity.reader")
    CapacityReader().recommend(20.0)
    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert any("qwen-24b" in message for message in info_messages)


def test_read_still_raises_not_implemented() -> None:
    """Plan A fills recommend and read_pressure. read stays a stub until Plan B."""
    with pytest.raises(NotImplementedError):
        CapacityReader().read(
            memory=lambda: (64.0, 40.2),
            ollama=lambda path: (200, {"models": []}),
            pressure=lambda: "normal",
            ollama_host="http://127.0.0.1:11434",
        )
