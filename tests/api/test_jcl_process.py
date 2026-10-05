"""JCL process bind settings. The socket stays closed until a token table exists."""

from pathlib import Path

import pytest

from local_ai.jcl.process import BindSettings, bind_settings, prepare
from local_ai.jcl.tokens import TokenTableError


_SECRET = "lane-50-secret"


def test_prepare_defaults_to_loopback_8770(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token and no host variable bind 127.0.0.1:8770 and expose submit."""
    monkeypatch.chdir(tmp_path)
    application, settings = prepare(_env(tmp_path))

    assert settings == BindSettings("127.0.0.1", 8770)
    assert "/jcl/submit" in {rule.rule for rule in application.url_map.iter_rules()}
    denied = application.test_client().post(
        "/jcl/submit",
        json={"goal": "Write a brief.", "done_when": "brief.md exists"},
    )
    assert denied.status_code == 401


def test_prepare_reads_a_lan_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LOCAL_AI_JCL_HOST may be 0.0.0.0. The token is what makes that bind legal."""
    monkeypatch.chdir(tmp_path)
    _application, settings = prepare(_env(tmp_path, host="0.0.0.0", port="9001"))

    assert settings == BindSettings("0.0.0.0", 9001)


def test_prepare_refuses_to_open_the_database_without_a_token(tmp_path: Path) -> None:
    """No token means no bind and no sqlite file."""
    database = tmp_path / "local-ai.sqlite"

    with pytest.raises(TokenTableError, match="no JCL tokens"):
        prepare({"LOCAL_AI_DB": str(database), "LOCAL_AI_JOBS": str(tmp_path / "jobs")})

    assert not database.exists()


def test_prepare_rejects_a_bad_port_before_opening_the_database(tmp_path: Path) -> None:
    """Port 0 is rejected before compose migrates the database."""
    database = tmp_path / "local-ai.sqlite"

    with pytest.raises(ValueError, match="LOCAL_AI_JCL_PORT"):
        prepare(_env(tmp_path, port="0"))

    assert not database.exists()


def test_bind_settings_rejects_a_blank_host() -> None:
    """A set-but-blank host is a configuration error, not a silent loopback."""
    with pytest.raises(ValueError, match="LOCAL_AI_JCL_HOST"):
        bind_settings({"LOCAL_AI_JCL_HOST": "   "})


def test_bind_settings_rejects_a_non_mapping() -> None:
    """The environment argument is a mapping."""
    with pytest.raises(TypeError):
        bind_settings(["127.0.0.1"])  # type: ignore[arg-type]


def _env(tmp_path: Path, host: str | None = None, port: str | None = None) -> dict[str, str]:
    """
    A process environment that keeps the database inside the test directory.

    :param tmp_path: pytest directory as Path. Example: Path("/tmp/pytest/test0")
    :param host: bind host as str | None. Example: "0.0.0.0"
    :param port: bind port as str | None. Example: "9001"
    :return: environment as dict[str, str]. Example: {"LOCAL_AI_JCL_TOKEN_50": "lane-50-secret"}
    """
    env = {
        "LOCAL_AI_JCL_TOKEN_50": _SECRET,
        "LOCAL_AI_DB": str(tmp_path / "local-ai.sqlite"),
        "LOCAL_AI_JOBS": str(tmp_path / "jobs"),
    }
    if host is not None:
        env["LOCAL_AI_JCL_HOST"] = host
    if port is not None:
        env["LOCAL_AI_JCL_PORT"] = port
    return env
