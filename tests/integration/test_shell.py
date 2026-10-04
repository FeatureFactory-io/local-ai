"""ShellTool runs a real command inside a grant and keeps the transcript in the job."""

import logging
import shlex
from pathlib import Path

import pytest

from local_ai.contracts import AccessGrant, ToolEnvelope
from local_ai.tools.grant import GrantChecker
from local_ai.tools.shell import ShellTool


def _job_shell(tmp_path: Path) -> tuple[ShellTool, AccessGrant, Path]:
    """
    Build a job-mode shell rooted at a temp job directory.

    :param tmp_path: pytest temp directory as Path. Example: Path("/tmp/pytest")
    :return: tool, grant, and job directory as tuple. Example: (ShellTool(GrantChecker()), AccessGrant(mode="job"), Path("/tmp/job"))
    """
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    checker = GrantChecker()
    grant = checker.resolve("job", (), job_dir, tmp_path / "home")
    return ShellTool(checker), grant, job_dir


def test_run_denies_a_path_outside_the_job_and_creates_nothing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A path that leaves the job returns access_denied and does not start."""
    tool, grant, job_dir = _job_shell(tmp_path)
    escaped = tmp_path / "escaped.txt"
    caplog.set_level(logging.INFO)

    envelope = tool.run(grant, "touch ../escaped.txt", job_dir)

    assert envelope == ToolEnvelope(success=False, result=None, error="access_denied")
    assert not escaped.exists()
    assert list(job_dir.iterdir()) == []
    assert any(
        record.levelno == logging.INFO and record.message == "ShellTool.run access_denied"
        for record in caplog.records
    )
    assert not any("exit_code=" in record.message for record in caplog.records)


def test_run_returns_stdout_appends_shell_log_and_logs_only_the_exit_code(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exit 0 returns stdout, appends both streams, and does not log the transcript."""
    tool, grant, job_dir = _job_shell(tmp_path)
    caplog.set_level(logging.INFO)
    command = "sh -c 'printf %s zz; printf %s marker; printf %s ee >&2; printf %s rr >&2'"

    envelope = tool.run(grant, command, job_dir)
    appended = tool.run(grant, "printf %s tail", job_dir)

    assert envelope == ToolEnvelope(success=True, result="zzmarker", error="")
    assert appended == ToolEnvelope(success=True, result="tail", error="")
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == "zzmarkereerrtail"
    assert any(
        record.levelno == logging.INFO and record.message == "ShellTool.run exit_code=0"
        for record in caplog.records
    )
    assert not any("zzmarker" in record.message or "eerr" in record.message for record in caplog.records)


def test_run_returns_stderr_when_the_command_exits_nonzero(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-zero exit stays in the envelope and still appends the transcript."""
    tool, grant, job_dir = _job_shell(tmp_path)
    caplog.set_level(logging.INFO)
    command = "sh -c 'printf %s aa >&2; printf %s bb >&2; exit 4'"

    envelope = tool.run(grant, command, job_dir)

    assert envelope == ToolEnvelope(success=False, result="", error="aabb")
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == "aabb"
    assert any(
        record.levelno == logging.INFO and record.message == "ShellTool.run exit_code=4"
        for record in caplog.records
    )
    assert not any("aabb" in record.message for record in caplog.records)


def test_run_reads_a_file_the_paths_grant_allows(tmp_path: Path) -> None:
    """Mode paths runs a command whose path argument stays inside the grant."""
    job_dir = tmp_path / "job"
    books = tmp_path / "books"
    job_dir.mkdir()
    books.mkdir()
    chapter = books / "ch-01.md"
    chapter.write_text("chapter", encoding="utf-8")
    checker = GrantChecker()
    grant = checker.resolve("paths", (str(books),), job_dir, tmp_path / "home")

    envelope = ShellTool(checker).run(grant, f"cat {shlex.quote(str(chapter))}", job_dir)

    assert envelope == ToolEnvelope(success=True, result="chapter", error="")
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == "chapter"


def test_run_creates_a_file_in_the_job_directory(tmp_path: Path) -> None:
    """An allowed command uses the job directory as its working directory."""
    tool, grant, job_dir = _job_shell(tmp_path)

    envelope = tool.run(grant, "touch kept.txt", job_dir)

    assert envelope.success is True
    assert (job_dir / "kept.txt").is_file()
    assert not (tmp_path / "kept.txt").exists()


def test_run_empty_command_does_not_start(tmp_path: Path) -> None:
    """A blank command returns an envelope and does not create shell.log."""
    tool, grant, job_dir = _job_shell(tmp_path)

    envelope = tool.run(grant, "   ", job_dir)

    assert envelope == ToolEnvelope(success=False, result=None, error="empty command")
    assert list(job_dir.iterdir()) == []


def test_run_unparsable_command_does_not_start(tmp_path: Path) -> None:
    """An unmatched quote returns an envelope and does not start a process."""
    tool, grant, job_dir = _job_shell(tmp_path)

    envelope = tool.run(grant, 'printf "', job_dir)

    assert envelope == ToolEnvelope(success=False, result=None, error="unparsable command")
    assert list(job_dir.iterdir()) == []


def test_run_missing_executable_returns_stderr_without_raising(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing program becomes an envelope with exit code 127."""
    tool, grant, job_dir = _job_shell(tmp_path)
    caplog.set_level(logging.INFO)

    envelope = tool.run(grant, "no-such-binary-xyz-shelltool", job_dir)

    assert envelope.success is False
    assert envelope.result == ""
    assert "no-such-binary-xyz-shelltool" in envelope.error
    assert (job_dir / "shell.log").read_text(encoding="utf-8") == envelope.error
    assert any(
        record.levelno == logging.INFO and record.message == "ShellTool.run exit_code=127"
        for record in caplog.records
    )


def test_run_rejects_a_non_string_command(tmp_path: Path) -> None:
    """A command that is not text fails before a process starts."""
    tool, grant, job_dir = _job_shell(tmp_path)

    with pytest.raises(TypeError):
        tool.run(grant, None, job_dir)  # type: ignore[arg-type]

    assert list(job_dir.iterdir()) == []


def test_run_rejects_a_missing_job_directory(tmp_path: Path) -> None:
    """A job directory that does not exist fails before a process starts."""
    missing = tmp_path / "missing"
    checker = GrantChecker()
    grant = checker.resolve("job", (), missing, tmp_path / "home")

    with pytest.raises(ValueError):
        ShellTool(checker).run(grant, "printf %s hi", missing)

    assert not missing.exists()
