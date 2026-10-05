"""Catalog registration from a JCL host. The poster talks to a real catalog app."""

import logging
import threading
from pathlib import Path

import pytest

from local_ai.catalog.app import create_app
from local_ai.jcl.process import prepare
from local_ai.jcl.register import (
    CONFIG_REGISTER,
    RegisterConfigError,
    RegisterSettings,
    register_once,
    start_refresh,
)
from local_ai.jcl.tokens import load_token_table
from tests.api.test_jcl import _server
from tests.support.log_story import assert_log_story


_LANE_10 = "lane-10-secret"
_LANE_50 = "lane-50-secret"
_CATALOG = "catalog-secret"
_ADVERTISE = "http://192.168.1.20:8770"
_CATALOG_URL = "http://192.168.1.40:8771"
_SEEN = "2026-10-05T14:00:00Z"


def test_prepare_refuses_a_catalog_url_without_an_advertise_url(tmp_path: Path) -> None:
    """A catalog target with no advertise URL does not open the database."""
    database = tmp_path / "local-ai.sqlite"
    env = _env(tmp_path)
    env["LOCAL_AI_CATALOG_URL"] = _CATALOG_URL

    with pytest.raises(RegisterConfigError, match="LOCAL_AI_ADVERTISE_URL"):
        prepare(env)

    assert not database.exists()


def test_prepare_stores_register_settings_without_posting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both URLs are kept on the app. prepare does not open a socket to the catalog."""
    monkeypatch.chdir(tmp_path)
    env = _env(tmp_path)
    env["LOCAL_AI_CATALOG_URL"] = _CATALOG_URL + "/"
    env["LOCAL_AI_ADVERTISE_URL"] = _ADVERTISE
    env["LOCAL_AI_HOST_NAME"] = "studio-mac"

    application, _settings = prepare(env)
    catalog = application.config[CONFIG_REGISTER]

    assert catalog == RegisterSettings(_CATALOG_URL, _ADVERTISE, "studio-mac", 60)


def test_refresh_posts_capacity_and_tokens_to_the_catalog(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The refresh thread posts capacity() and the token table, then stops."""
    caplog.set_level(logging.INFO)
    server = _server(tmp_path, ())
    tokens = load_token_table({"LOCAL_AI_JCL_TOKEN_10": _LANE_10, "LOCAL_AI_JCL_TOKEN_50": _LANE_50})
    catalog = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=lambda: _SEEN).test_client()
    settings = RegisterSettings(_CATALOG_URL, _ADVERTISE, "studio-mac", 60)
    stop = threading.Event()

    def poster(url: str, body: dict) -> None:
        assert url == _CATALOG_URL + "/register"
        response = catalog.post("/register", json=body)
        assert response.status_code == 200
        stop.set()

    thread = start_refresh(server, tokens, settings, poster, stop)

    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()
    listed = catalog.get("/hosts", headers={"Authorization": f"Bearer {_CATALOG}"})
    host = listed.get_json()["hosts"][0]
    assert host["name"] == "studio-mac"
    assert host["advertise_url"] == _ADVERTISE
    assert host["capacity"]["ollama"] == "up"
    assert "qwen-14b" in host["models"]
    assert {"priority": 10, "token": _LANE_10} in host["tokens"]
    assert {"priority": 50, "token": _LANE_50} in host["tokens"]
    assert_log_story(caplog.records, "jcl.register", "ok", [f"advertise_url={_ADVERTISE}"])
    assert _LANE_10 not in caplog.text
    assert _LANE_50 not in caplog.text


def test_register_once_logs_a_failure_and_leaves_the_secret_out(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A catalog that cannot be reached does not raise and does not log the token."""
    caplog.set_level(logging.INFO)
    server = _server(tmp_path, ())
    tokens = load_token_table({"LOCAL_AI_JCL_TOKEN_50": _LANE_50})
    settings = RegisterSettings(_CATALOG_URL, _ADVERTISE, "studio-mac", 60)

    def poster(url: str, body: dict) -> None:
        raise RuntimeError("catalog down")

    register_once(server, tokens, settings, poster)

    assert_log_story(
        caplog.records,
        "jcl.register",
        "failed",
        [f"advertise_url={_ADVERTISE}", "error=RuntimeError"],
        level="ERROR",
    )
    assert _LANE_50 not in caplog.text


def test_start_refresh_does_nothing_without_a_catalog_url(tmp_path: Path) -> None:
    """No catalog URL means no thread."""
    server = _server(tmp_path, ())
    tokens = load_token_table({"LOCAL_AI_JCL_TOKEN_50": _LANE_50})
    settings = RegisterSettings(None, "", "studio-mac", 60)

    assert start_refresh(server, tokens, settings, _unused_poster, None) is None


def _env(tmp_path: Path) -> dict[str, str]:
    """
    A process environment that keeps the database inside the test directory.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :return: environment as dict[str, str]. Example: {"LOCAL_AI_JCL_TOKEN_50": "lane-50-secret"}
    """
    return {
        "LOCAL_AI_JCL_TOKEN_50": _LANE_50,
        "LOCAL_AI_DB": str(tmp_path / "local-ai.sqlite"),
        "LOCAL_AI_JOBS": str(tmp_path / "jobs"),
    }


def _unused_poster(url: str, body: dict) -> None:
    """
    A poster that fails the test if it is called.

    :param url: register URL as str. Example: "http://192.168.1.40:8771/register"
    :param body: register body as dict. Example: {}
    :return: None. Example: None
    """
    raise AssertionError(url)
