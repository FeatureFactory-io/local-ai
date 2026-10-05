"""Catalog register and host list on a real sqlite file."""

import logging
from pathlib import Path

import pytest

from local_ai.catalog.app import create_app
from local_ai.catalog.process import CatalogConfigError, CatalogSettings, prepare
from tests.support.log_story import assert_log_story


_CATALOG = "catalog-secret"
_LANE_10 = "lane-10-secret"
_LANE_50 = "lane-50-secret"
_URL = "http://192.168.1.20:8770"
_SEEN = "2026-10-05T14:00:00Z"
_EXACT = "2026-10-05T14:03:00Z"
_STALE = "2026-10-05T14:03:01Z"
_LATER = "2026-10-05T14:02:00Z"


class _Clock:
    """Mutable UTC stamp so a test can age a row without sleeping."""

    def __init__(self, stamp: str) -> None:
        """
        Store the first stamp.

        :param stamp: UTC stamp as str. Example: "2026-10-05T14:00:00Z"
        """
        self.stamp = stamp

    def __call__(self) -> str:
        """
        Return the current stamp.

        :return: stamp as str. Example: "2026-10-05T14:00:00Z"
        """
        return self.stamp


def test_register_without_a_token_and_list_with_the_catalog_token(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Register is open. GET /hosts returns the tokens and leaves them out of the log."""
    clock = _Clock(_SEEN)
    client = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=clock).test_client()
    caplog.set_level(logging.INFO)

    registered = client.post("/register", json=_body(), headers={"Authorization": "Bearer wrong-token"})
    missing = client.get("/hosts")
    listed = client.get("/hosts", headers={"Authorization": f"Bearer {_CATALOG}"})

    assert registered.status_code == 200
    assert registered.get_json()["status"] == "ok"
    assert missing.status_code == 401
    assert missing.get_json() == {"status": "rejected", "reason": "unauthorized"}
    assert listed.status_code == 200
    body = listed.get_json()
    assert body["count"] == 1
    host = body["hosts"][0]
    assert host["name"] == "studio-mac"
    assert host["advertise_url"] == _URL
    assert host["tokens"] == [
        {"priority": 10, "token": _LANE_10},
        {"priority": 50, "token": _LANE_50},
    ]
    assert host["models"] == ["qwen-8b", "qwen-14b"]
    assert host["capacity"]["recommended"] == "qwen-14b"
    assert host["last_seen"] == _SEEN
    assert host["stale"] is False
    assert_log_story(caplog.records, "catalog.register", "ok", [f"advertise_url={_URL}", "priorities=10,50"])
    assert_log_story(caplog.records, "catalog.hosts", "rejected", ["reason=unauthorized"])
    assert _LANE_10 not in caplog.text
    assert _LANE_50 not in caplog.text
    assert _CATALOG not in caplog.text
    assert "wrong-token" not in caplog.text


def test_second_register_updates_capacity_and_last_seen(tmp_path: Path) -> None:
    """The same advertise URL is one row. The later capacity and stamp replace the first."""
    clock = _Clock(_SEEN)
    client = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=clock).test_client()
    client.post("/register", json=_body())
    clock.stamp = _LATER

    updated = client.post("/register", json=_body(recommended="qwen-8b"))
    listed = client.get("/hosts", headers=_auth())

    assert updated.status_code == 200
    body = listed.get_json()
    assert body["count"] == 1
    assert body["hosts"][0]["capacity"]["recommended"] == "qwen-8b"
    assert body["hosts"][0]["last_seen"] == _LATER
    assert body["hosts"][0]["stale"] is False


def test_a_row_older_than_the_limit_is_stale_and_stays_listed(tmp_path: Path) -> None:
    """180 seconds is still fresh. One second past that is stale, and the row remains."""
    clock = _Clock(_SEEN)
    client = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=clock).test_client()
    client.post("/register", json=_body())

    clock.stamp = _EXACT
    fresh = client.get("/hosts", headers=_auth()).get_json()["hosts"][0]["stale"]
    clock.stamp = _STALE
    stale = client.get("/hosts", headers=_auth()).get_json()

    assert fresh is False
    assert stale["count"] == 1
    assert stale["hosts"][0]["stale"] is True
    assert stale["hosts"][0]["advertise_url"] == _URL


def test_unknown_catalog_token_is_unauthorized_and_is_not_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wrong catalog secret is rejected and does not appear in the log."""
    client = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=_Clock(_SEEN)).test_client()
    caplog.set_level(logging.INFO)
    secret = "not-the-catalog-token"

    response = client.get("/hosts", headers={"Authorization": f"Bearer {secret}"})

    assert response.status_code == 401
    assert secret not in caplog.text
    assert _CATALOG not in caplog.text


def test_malformed_register_does_not_store_a_host(tmp_path: Path) -> None:
    """A body that is not a registration is HTTP 400 and leaves the phone book empty."""
    client = create_app(tmp_path / "catalog.sqlite", _CATALOG, clock=_Clock(_SEEN)).test_client()

    rejected = client.post(
        "/register",
        data=b"not-json",
        content_type="application/json",
    )
    priority = client.post("/register", json=_body(priority=0))
    listed = client.get("/hosts", headers=_auth())

    assert rejected.status_code == 400
    assert rejected.get_json()["reason"] == "bad_json"
    assert priority.status_code == 400
    assert priority.get_json()["reason"] == "bad_request"
    assert listed.get_json()["count"] == 0


def test_prepare_refuses_to_open_the_database_without_a_token(tmp_path: Path) -> None:
    """No catalog token means no bind and no sqlite file."""
    database = tmp_path / "catalog.sqlite"

    with pytest.raises(CatalogConfigError, match="LOCAL_AI_CATALOG_TOKEN"):
        prepare({"LOCAL_AI_CATALOG_DB": str(database)})

    assert not database.exists()


def test_prepare_defaults_to_loopback_8771(tmp_path: Path) -> None:
    """A catalog token and no host variable bind 127.0.0.1:8771."""
    database = tmp_path / "catalog.sqlite"
    application, settings = prepare(
        {
            "LOCAL_AI_CATALOG_TOKEN": _CATALOG,
            "LOCAL_AI_CATALOG_DB": str(database),
        }
    )

    assert settings == CatalogSettings("127.0.0.1", 8771, database, 180)
    assert database.exists()
    denied = application.test_client().get("/hosts")
    assert denied.status_code == 401


def _body(recommended: str = "qwen-14b", priority: int = 10) -> dict:
    """
    One registration. The second token keeps priority 50.

    :param recommended: capacity recommendation as str. Example: "qwen-14b"
    :param priority: first token priority as int. Example: 10
    :return: JSON body as dict. Example: {"name": "studio-mac", "advertise_url": "http://192.168.1.20:8770"}
    """
    return {
        "name": "studio-mac",
        "advertise_url": _URL,
        "tokens": [
            {"priority": 50, "token": _LANE_50},
            {"priority": priority, "token": _LANE_10},
        ],
        "capacity": {"status": "ok", "recommended": recommended, "ollama": "up"},
        "models": ["qwen-8b", "qwen-14b"],
    }


def _auth() -> dict[str, str]:
    """
    The catalog bearer header.

    :return: header as dict[str, str]. Example: {"Authorization": "Bearer catalog-secret"}
    """
    return {"Authorization": f"Bearer {_CATALOG}"}
