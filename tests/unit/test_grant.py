"""GrantChecker resolves roots and rejects paths that leave them."""

import logging
from pathlib import Path

import pytest

from local_ai.contracts import AccessDenied, AccessGrant
from local_ai.tools.grant import GrantChecker


def test_resolve_job_allows_only_job_dir(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mode job keeps the job directory and drops caller paths."""
    job_dir = tmp_path / "data" / "jobs" / "7c1e0a2e"
    job_dir.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    caplog.set_level(logging.INFO)

    grant = GrantChecker().resolve(
        "job",
        ("/Users/denispetelin/books/draft",),
        job_dir,
        home,
    )

    assert grant == AccessGrant(mode="job", paths=(str(job_dir),))
    assert any(
        record.levelno == logging.INFO and "job" in record.message and str(job_dir) in record.message
        for record in caplog.records
    )


def test_resolve_paths_allows_given_paths_plus_job_dir(tmp_path: Path) -> None:
    """Mode paths keeps the caller directories and adds the job directory."""
    job_dir = tmp_path / "data" / "jobs" / "7c1e0a2e"
    job_dir.mkdir(parents=True)
    books = "/Users/denispetelin/books/draft"

    grant = GrantChecker().resolve("paths", (books,), job_dir, tmp_path)

    assert grant == AccessGrant(mode="paths", paths=(books, str(job_dir)))


def test_resolve_home_allows_home(tmp_path: Path) -> None:
    """Mode home keeps the home directory and ignores caller paths."""
    job_dir = tmp_path / "data" / "jobs" / "7c1e0a2e"
    job_dir.mkdir(parents=True)
    home = Path("/Users/denispetelin")

    grant = GrantChecker().resolve("home", ("/tmp/elsewhere",), job_dir, home)

    assert grant == AccessGrant(mode="home", paths=(str(home),))


@pytest.mark.parametrize("mode", ["root", "", "Job"])
def test_resolve_unknown_mode_raises_access_denied(mode: str, tmp_path: Path) -> None:
    """An unknown mode is rejected before a grant is returned."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(AccessDenied):
        GrantChecker().resolve(mode, (), job_dir, tmp_path)


def test_resolve_rejects_a_non_string_mode(tmp_path: Path) -> None:
    """A missing mode fails before any root is chosen."""
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(TypeError):
        GrantChecker().resolve(None, (), job_dir, tmp_path)  # type: ignore[arg-type]
