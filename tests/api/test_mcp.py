"""MCP tool surface. Readers are injected. Ollama is not contacted."""

import pytest

from local_ai.dispatcher.server import http_port_from_env


def test_http_port_from_env_missing_returns_none() -> None:
    """An environment without LOCAL_AI_HTTP_PORT stays on stdio."""
    assert http_port_from_env({}) is None


def test_http_port_from_env_empty_returns_none() -> None:
    """An empty LOCAL_AI_HTTP_PORT stays on stdio."""
    assert http_port_from_env({"LOCAL_AI_HTTP_PORT": ""}) is None


def test_http_port_from_env_integer_string_returns_that_int() -> None:
    """A decimal port string becomes that integer."""
    assert http_port_from_env({"LOCAL_AI_HTTP_PORT": "8765"}) == 8765


def test_http_port_from_env_abc_raises_value_error() -> None:
    """A non-integer port is refused before any socket is opened."""
    with pytest.raises(ValueError, match="LOCAL_AI_HTTP_PORT"):
        http_port_from_env({"LOCAL_AI_HTTP_PORT": "abc"})


def test_http_port_from_env_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting env reads LOCAL_AI_HTTP_PORT from the process."""
    monkeypatch.setenv("LOCAL_AI_HTTP_PORT", "9001")
    assert http_port_from_env() == 9001
