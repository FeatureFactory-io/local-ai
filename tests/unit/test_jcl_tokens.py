"""Startup parsing of LOCAL_AI_JCL_TOKEN_<priority>."""

import logging

import pytest

from local_ai.contracts import InvalidPriority
from local_ai.jcl.tokens import (
    PriorityForbidden,
    TokenTableError,
    load_token_table,
    priority_within_ceiling,
)
from tests.support.log_story import assert_log_story


_LANE_10 = "lane-10-secret"
_LANE_50 = "lane-50-secret"
_HAPPY_ENV = {
    "LOCAL_AI_JCL_TOKEN_10": _LANE_10,
    "LOCAL_AI_JCL_TOKEN_50": _LANE_50,
    "OLLAMA_HOST": "http://127.0.0.1:11434",
}


def test_load_token_table_maps_each_secret_to_its_priority(caplog: pytest.LogCaptureFixture) -> None:
    """Two token variables become two ceilings. Unrelated variables are ignored."""
    caplog.set_level(logging.INFO)

    table = load_token_table(_HAPPY_ENV)

    assert table.ceiling(_LANE_10) == 10
    assert table.ceiling(_LANE_50) == 50
    assert table.ceiling("missing-secret") is None
    assert table.registration_rows() == (
        {"priority": 10, "token": _LANE_10},
        {"priority": 50, "token": _LANE_50},
    )
    assert_log_story(caplog.records, "load_token_table", "exit", ["priorities=10,50"])
    assert _LANE_10 not in caplog.text
    assert _LANE_50 not in caplog.text


@pytest.mark.parametrize("name", ["LOCAL_AI_JCL_TOKEN_0", "LOCAL_AI_JCL_TOKEN_101"])
def test_load_token_table_rejects_a_priority_outside_1_to_100(name: str) -> None:
    """0 and 101 are not queue priorities."""
    with pytest.raises(InvalidPriority):
        load_token_table({name: _LANE_50})


def test_load_token_table_rejects_an_empty_secret() -> None:
    """A blank value is not a bearer token."""
    with pytest.raises(TokenTableError, match="non-empty"):
        load_token_table({"LOCAL_AI_JCL_TOKEN_50": "   "})


def test_load_token_table_rejects_a_secret_used_twice() -> None:
    """One secret cannot name two ceilings."""
    env = {
        "LOCAL_AI_JCL_TOKEN_10": _LANE_50,
        "LOCAL_AI_JCL_TOKEN_50": _LANE_50,
    }
    with pytest.raises(TokenTableError, match="share one secret"):
        load_token_table(env)


def test_load_token_table_rejects_an_environment_with_no_tokens() -> None:
    """Startup refuses to bind when no JCL token is set."""
    with pytest.raises(TokenTableError, match="no JCL tokens"):
        load_token_table({"OLLAMA_HOST": "http://127.0.0.1:11434"})


def test_load_token_table_rejects_a_non_numeric_suffix() -> None:
    """The suffix is the priority integer, not a label."""
    with pytest.raises(TokenTableError, match="priority"):
        load_token_table({"LOCAL_AI_JCL_TOKEN_high": _LANE_50})


def test_load_token_table_rejects_a_non_mapping() -> None:
    """The environment argument is a mapping."""
    with pytest.raises(TypeError):
        load_token_table(["LOCAL_AI_JCL_TOKEN_50"])  # type: ignore[arg-type]


def test_priority_within_ceiling_uses_the_ceiling_when_omitted() -> None:
    """An omitted submit priority takes the token's lane."""
    assert priority_within_ceiling(50, None) == 50


def test_priority_within_ceiling_accepts_a_worse_priority() -> None:
    """80 is later than 50, so a lane-50 token may set it."""
    assert priority_within_ceiling(50, 80) == 80


def test_priority_within_ceiling_rejects_a_better_priority() -> None:
    """10 runs before 50. A lane-50 token may not set it."""
    with pytest.raises(PriorityForbidden):
        priority_within_ceiling(50, 10)


def test_priority_within_ceiling_rejects_a_non_int() -> None:
    """A JSON string is not a priority."""
    with pytest.raises(TypeError):
        priority_within_ceiling(50, "50")  # type: ignore[arg-type]
