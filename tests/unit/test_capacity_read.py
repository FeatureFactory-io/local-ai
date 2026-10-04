"""CapacityReader.read builds the capacity body from memory and injected HTTP."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from local_ai.capacity.reader import CapacityReader
from tests.support.log_story import assert_log_story

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"
HOST = "http://127.0.0.1:11434"
TAGS_URL = f"{HOST}/api/tags"
PS_URL = f"{HOST}/api/ps"
LOGICAL_MODELS = ["qwen-8b", "qwen-14b", "qwen-24b"]

Http = Callable[[str], tuple[int, dict]]


def _fixture(name: str) -> dict:
    """Load one section 18 capacity example. Example: capacity_up.json."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _up_reader(loaded_name: str | None = "qwen2.5:14b") -> tuple[Http, list[str]]:
    """
    Tags 200, then ps with one loaded name. Records each URL.

    :param loaded_name: name to put on /api/ps, or None for an empty list. Example: "qwen2.5:14b"
    :return: reader and the URLs it saw. Example: (reader, ["http://127.0.0.1:11434/api/tags"])
    """
    calls: list[str] = []

    def ollama(url: str) -> tuple[int, dict]:
        calls.append(url)
        if url == PS_URL:
            models = [] if loaded_name is None else [{"name": loaded_name}]
            return 200, {"models": models}
        if url == TAGS_URL:
            return 200, {"models": []}
        raise AssertionError(url)

    return ollama, calls


def test_read_with_tags_200_returns_up_body_and_fixture_fields() -> None:
    """A 200 from /api/tags is up. recommend(40.2) is qwen-24b. Keys match capacity_up.json."""
    fixture = _fixture("capacity_up.json")
    ollama, calls = _up_reader(fixture["loaded_model"])
    body = CapacityReader().read(
        memory=lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"]),
        ollama=ollama,
        pressure=lambda: fixture["memory_pressure"],
        ollama_host=HOST,
    )
    assert set(body) == set(fixture)
    assert body["status"] == fixture["status"]
    assert body["memory_total_gb"] == fixture["memory_total_gb"]
    assert body["memory_free_gb"] == fixture["memory_free_gb"]
    assert body["memory_pressure"] == fixture["memory_pressure"]
    assert body["ollama"] == "up"
    assert body["loaded_model"] == fixture["loaded_model"]
    assert body["models"] == fixture["models"]
    assert body["recommended"] == "qwen-24b"
    assert calls == [TAGS_URL, PS_URL]


@pytest.mark.parametrize(
    ("free_gb", "expected"),
    [(20.0, "qwen-24b"), (14.0, "qwen-14b"), (12.0, "qwen-14b"), (8.0, "qwen-8b"), (7.9, "none")],
)
def test_read_with_ollama_up_recommends_from_free_gb(free_gb: float, expected: str) -> None:
    """The up path calls recommend. Below 8 GB the size is none and the three names stay listed."""
    body = CapacityReader().read(
        memory=lambda: (64.0, free_gb),
        ollama=lambda path: (200, {"models": []}),
        pressure=lambda: "normal",
        ollama_host=HOST,
    )
    assert body["ollama"] == "up"
    assert body["recommended"] == expected
    assert body["loaded_model"] is None
    assert body["models"] == LOGICAL_MODELS
    assert body["memory_free_gb"] == free_gb


def test_read_with_tags_non_200_matches_capacity_down_fixture() -> None:
    """A non-200 from /api/tags is the capacity_down.json body and does not call /api/ps."""
    fixture = _fixture("capacity_down.json")
    calls: list[str] = []

    def ollama(url: str) -> tuple[int, dict]:
        calls.append(url)
        return 503, {}

    body = CapacityReader().read(
        memory=lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"]),
        ollama=ollama,
        pressure=lambda: fixture["memory_pressure"],
        ollama_host=HOST,
    )
    assert body == fixture
    assert calls == [TAGS_URL]


def test_read_with_tags_exception_matches_capacity_down_fixture() -> None:
    """A connection error is ollama down. No live Ollama is contacted."""
    fixture = _fixture("capacity_down.json")
    calls: list[str] = []

    def ollama(url: str) -> tuple[int, dict]:
        calls.append(url)
        raise ConnectionError("connection refused")

    body = CapacityReader().read(
        memory=lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"]),
        ollama=ollama,
        pressure=lambda: fixture["memory_pressure"],
        ollama_host=HOST,
    )
    assert body == fixture
    assert calls == [TAGS_URL]


def test_read_with_ps_error_stays_up_and_loaded_model_is_null() -> None:
    """Tags succeeded, so Ollama is up. A failed /api/ps only clears loaded_model."""

    def ollama(url: str) -> tuple[int, dict]:
        if url == TAGS_URL:
            return 200, {"models": []}
        raise TimeoutError("ps timed out")

    body = CapacityReader().read(
        memory=lambda: (64.0, 40.2),
        ollama=ollama,
        pressure=lambda: "warn",
        ollama_host=HOST,
    )
    assert body["ollama"] == "up"
    assert body["loaded_model"] is None
    assert body["recommended"] == "qwen-24b"
    assert body["memory_pressure"] == "warn"
    assert body["models"] == LOGICAL_MODELS


def test_read_with_ps_non_200_stays_up_and_loaded_model_is_null() -> None:
    """A non-200 from /api/ps does not flip the tags result to down."""

    def ollama(url: str) -> tuple[int, dict]:
        if url == TAGS_URL:
            return 200, {}
        return 500, {}

    body = CapacityReader().read(
        memory=lambda: (64.0, 12.0),
        ollama=ollama,
        pressure=lambda: "critical",
        ollama_host=HOST,
    )
    assert body["ollama"] == "up"
    assert body["loaded_model"] is None
    assert body["recommended"] == "qwen-14b"
    assert body["memory_pressure"] == "critical"


def test_read_with_model_key_uses_that_name_when_name_is_missing() -> None:
    """Ollama sometimes sends model instead of name. The first loaded row wins."""

    def ollama(url: str) -> tuple[int, dict]:
        if url == TAGS_URL:
            return 200, {}
        return 200, {"models": [{"model": "qwen2.5:14b"}, {"name": "other:7b"}]}

    body = CapacityReader().read(
        memory=lambda: (64.0, 14.0),
        ollama=ollama,
        pressure=lambda: "normal",
        ollama_host=HOST,
    )
    assert body["loaded_model"] == "qwen2.5:14b"
    assert body["recommended"] == "qwen-14b"


def test_read_with_trailing_slash_host_requests_one_slash() -> None:
    """A trailing slash on the host must not produce /api/api or a double slash."""
    calls: list[str] = []

    def ollama(url: str) -> tuple[int, dict]:
        calls.append(url)
        if url == PS_URL:
            return 200, {"models": [{"name": "qwen2.5:14b"}]}
        return 200, {}

    CapacityReader().read(
        memory=lambda: (64.0, 40.2),
        ollama=ollama,
        pressure=lambda: "normal",
        ollama_host=f"{HOST}/",
    )
    assert calls == [TAGS_URL, PS_URL]


def test_read_logs_up_state_and_recommended_size_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The up decision names ollama and the recommended size on one INFO line."""
    caplog.set_level(logging.INFO, logger="local_ai.capacity.reader")
    ollama, _calls = _up_reader("qwen2.5:14b")
    CapacityReader().read(
        memory=lambda: (64.0, 40.2),
        ollama=ollama,
        pressure=lambda: "normal",
        ollama_host=HOST,
    )
    assert_log_story(
        caplog.records,
        where="capacity.read",
        beat="up",
        must_include=["recommended=qwen-24b"],
    )


def test_read_logs_down_state_and_recommended_none_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The down decision names ollama down and recommended none on one INFO line."""
    caplog.set_level(logging.INFO, logger="local_ai.capacity.reader")

    def ollama(_path: str) -> tuple[int, dict]:
        raise ConnectionError("down")

    CapacityReader().read(
        memory=lambda: (64.0, 40.2),
        ollama=ollama,
        pressure=lambda: "normal",
        ollama_host=HOST,
    )
    assert_log_story(
        caplog.records,
        where="capacity.read",
        beat="down",
        must_include=["recommended=none"],
    )


def test_read_with_pressure_probe_raising_reports_unknown() -> None:
    """read calls read_pressure, so a failed probe is unknown and the body is still returned."""

    def boom() -> str:
        raise OSError("memory_pressure down")

    body = CapacityReader().read(
        memory=lambda: (64.0, 40.2),
        ollama=lambda path: (503, {}),
        pressure=boom,
        ollama_host=HOST,
    )
    assert body["memory_pressure"] == "unknown"
    assert body["ollama"] == "down"
    assert body["recommended"] == "none"


def test_read_with_malformed_tags_response_returns_down() -> None:
    """A reader that does not return (status, body) is not a 200, so Ollama is down."""
    fixture = _fixture("capacity_down.json")
    body = CapacityReader().read(
        memory=lambda: (fixture["memory_total_gb"], fixture["memory_free_gb"]),
        ollama=lambda path: "nope",  # type: ignore[return-value, misc]
        pressure=lambda: fixture["memory_pressure"],
        ollama_host=HOST,
    )
    assert body == fixture


def test_read_with_non_callable_memory_raises_type_error() -> None:
    """A memory port that cannot be called is rejected before HTTP."""
    with pytest.raises(TypeError):
        CapacityReader().read(
            memory=None,  # type: ignore[arg-type]
            ollama=lambda path: (200, {}),
            pressure=lambda: "normal",
            ollama_host=HOST,
        )


def test_read_with_memory_pair_of_wrong_shape_raises_value_error() -> None:
    """memory must return (total_gb, free_gb)."""
    with pytest.raises(ValueError):
        CapacityReader().read(
            memory=lambda: (64.0,),  # type: ignore[return-value]
            ollama=lambda path: (200, {}),
            pressure=lambda: "normal",
            ollama_host=HOST,
        )


def test_read_with_negative_free_gb_raises_value_error() -> None:
    """A negative free reading is not a capacity body."""
    with pytest.raises(ValueError):
        CapacityReader().read(
            memory=lambda: (64.0, -1.0),
            ollama=lambda path: (200, {}),
            pressure=lambda: "normal",
            ollama_host=HOST,
        )


def test_read_with_blank_ollama_host_raises_value_error() -> None:
    """The tags URL needs a host."""
    with pytest.raises(ValueError):
        CapacityReader().read(
            memory=lambda: (64.0, 40.2),
            ollama=lambda path: (200, {}),
            pressure=lambda: "normal",
            ollama_host="  ",
        )


def test_read_with_non_callable_ollama_raises_type_error() -> None:
    """HTTP is an injected reader. A missing reader is a type error."""
    with pytest.raises(TypeError):
        CapacityReader().read(
            memory=lambda: (64.0, 40.2),
            ollama=None,  # type: ignore[arg-type]
            pressure=lambda: "normal",
            ollama_host=HOST,
        )


def test_read_when_memory_reader_raises_propagates() -> None:
    """A broken memory probe is not reported as ollama down."""

    def boom() -> tuple[float, float]:
        raise RuntimeError("memory failed")

    with pytest.raises(RuntimeError, match="memory failed"):
        CapacityReader().read(
            memory=boom,
            ollama=lambda path: (200, {}),
            pressure=lambda: "normal",
            ollama_host=HOST,
        )
