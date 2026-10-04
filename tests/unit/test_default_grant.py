"""Leaf behavior for default_grant. Other contract stubs stay uncalled."""

import logging

import pytest

from local_ai.contracts import AccessGrant, InvalidAccess, default_grant


def test_default_grant_none_is_job(caplog: pytest.LogCaptureFixture) -> None:
    """Omitted access is a job grant with no path list."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    assert default_grant(None) == AccessGrant(mode="job")
    assert "job" in _info_text(caplog)


def test_default_grant_paths_keeps_the_path_list(caplog: pytest.LogCaptureFixture) -> None:
    """paths mode stores the directories the shell may touch."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    grant = default_grant({"mode": "paths", "paths": ["/Users/denispetelin/books/draft"]})
    assert grant == AccessGrant(mode="paths", paths=("/Users/denispetelin/books/draft",))
    assert "paths" in _info_text(caplog)


@pytest.mark.parametrize(
    "access",
    [
        {"mode": "paths"},
        {"mode": "paths", "paths": []},
        {"mode": "paths", "paths": ()},
    ],
)
def test_default_grant_paths_requires_paths(access: dict) -> None:
    """paths mode with no directories is not a grant."""
    with pytest.raises(InvalidAccess):
        default_grant(access)


def test_default_grant_unknown_mode_raises(caplog: pytest.LogCaptureFixture) -> None:
    """root is not job, paths, or home."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    with pytest.raises(InvalidAccess):
        default_grant({"mode": "root"})
    assert "root" in _info_text(caplog)


def test_default_grant_home_ignores_paths() -> None:
    """home is the user home directory. A paths list on that mode is not stored."""
    assert default_grant({"mode": "home", "paths": ["/tmp/ignored"]}) == AccessGrant(mode="home")


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    """Join INFO messages. Example: 'Accepted grant mode=job'."""
    return " ".join(record.message for record in caplog.records if record.levelno == logging.INFO)
