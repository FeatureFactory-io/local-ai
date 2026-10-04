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


def test_check_paths_raises_when_dotdot_leaves_the_job(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A path-like argument that resolves outside the job is denied before any process."""
    job_dir = tmp_path / "job"
    books = tmp_path / "books"
    job_dir.mkdir()
    books.mkdir()
    secret = books / "secret.md"
    secret.write_text("hidden")
    grant = GrantChecker().resolve("job", (str(books),), job_dir, tmp_path)
    caplog.set_level(logging.WARNING)

    with pytest.raises(AccessDenied):
        GrantChecker().check_paths(grant, job_dir, ("cat", "../books/secret.md"))

    assert secret.read_text() == "hidden"
    assert any(
        record.levelno == logging.WARNING and "secret.md" in record.message
        for record in caplog.records
    )


def test_check_paths_allows_a_file_inside_the_paths_grant(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mode paths allows the caller directory and the job directory."""
    job_dir = tmp_path / "job"
    books = tmp_path / "books" / "draft"
    job_dir.mkdir()
    books.mkdir(parents=True)
    chapter = books / "ch-01.md"
    chapter.write_text("chapter")
    grant = GrantChecker().resolve("paths", (str(books),), job_dir, tmp_path / "home")
    caplog.set_level(logging.INFO)

    GrantChecker().check_paths(grant, job_dir, ("cat", str(chapter)))

    assert any(
        record.levelno == logging.INFO
        and "allowed cwd=" in record.message
        and str(job_dir.resolve()) in record.message
        for record in caplog.records
    )


def test_check_paths_allows_a_path_under_home(tmp_path: Path) -> None:
    """Mode home allows a file under home that sits outside the job directory."""
    home = tmp_path / "home"
    job_dir = home / "jobs" / "7c1e0a2e"
    job_dir.mkdir(parents=True)
    note = home / "notes.md"
    note.write_text("ok")
    grant = GrantChecker().resolve("home", (), job_dir, home)

    GrantChecker().check_paths(grant, job_dir, ("cat", str(note)))


def test_check_paths_raises_when_cwd_escapes_with_dotdot(tmp_path: Path) -> None:
    """A working directory that resolves through .. is denied before argv runs."""
    job_dir = tmp_path / "job"
    nested = job_dir / "nested"
    nested.mkdir(parents=True)
    grant = GrantChecker().resolve("job", (), job_dir, tmp_path)

    with pytest.raises(AccessDenied):
        GrantChecker().check_paths(grant, nested / ".." / "..", ("ls",))


def test_check_paths_allows_dotdot_that_stays_inside_the_job(tmp_path: Path) -> None:
    """.. is resolved, so a climb that lands back inside the job is allowed."""
    job_dir = tmp_path / "job"
    nested = job_dir / "nested"
    nested.mkdir(parents=True)
    grant = GrantChecker().resolve("job", (), job_dir, tmp_path)

    GrantChecker().check_paths(grant, nested / "..", ("ls",))


def test_check_paths_rejects_a_sibling_prefix_of_the_job(tmp_path: Path) -> None:
    """A directory that only shares the job name as a prefix is outside the grant."""
    job_dir = tmp_path / "job"
    sibling = tmp_path / "job-extra"
    job_dir.mkdir()
    sibling.mkdir()
    leaked = sibling / "x.txt"
    leaked.write_text("nope")
    grant = GrantChecker().resolve("job", (), job_dir, tmp_path)

    with pytest.raises(AccessDenied):
        GrantChecker().check_paths(grant, job_dir, ("cat", str(leaked)))
