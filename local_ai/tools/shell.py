"""Shell tool. The worker's only tool. Calls GrantChecker, then the operating system."""

import logging
import shlex
import subprocess
from pathlib import Path

from local_ai.contracts import AccessDenied, AccessGrant, ToolEnvelope
from local_ai.tools.grant import GrantChecker


logger = logging.getLogger(__name__)

_SHELL_LOG_NAME = "shell.log"
_ACCESS_DENIED = "access_denied"
_EMPTY_COMMAND = "empty command"
_UNPARSABLE_COMMAND = "unparsable command"
_COMMAND_NOT_FOUND = 127


class ShellTool:
    """
    Runs one command inside a grant.

    Called by Worker.run_round. Does not import the dispatcher or the LLM.
    """

    def __init__(self, grant_checker: GrantChecker) -> None:
        """
        Store the checker. Do not start a process here.

        :param grant_checker: path enforcer as GrantChecker. Example: GrantChecker()
        """
        self._grant_checker = grant_checker

    def run(self, grant: AccessGrant, command: str, job_dir: Path) -> ToolEnvelope:
        """
        Run a command or return access_denied without running it.

        :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job")
        :param command: shell command as str. Example: "python3 -c \\"print(1)\\""
        :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
        :return: envelope as ToolEnvelope. Example: ToolEnvelope(success=False, result=None, error="access_denied")
        """
        _require_run_args(grant, command, job_dir)
        argv = _parse_argv(command)
        if isinstance(argv, ToolEnvelope):
            return argv
        logger.info(f"Starting ShellTool.run argc={len(argv)} cwd={job_dir}")
        denied = self._denied_envelope(grant, job_dir, argv)
        if denied is not None:
            return denied
        return _finish(_start_process(argv, job_dir), job_dir)

    def _denied_envelope(
        self,
        grant: AccessGrant,
        job_dir: Path,
        argv: tuple[str, ...],
    ) -> ToolEnvelope | None:
        """
        Return access_denied when the checker rejects the command.

        :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job")
        :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
        :param argv: command tokens as tuple[str, ...]. Example: ("touch", "../escaped.txt")
        :return: envelope or None as ToolEnvelope | None. Example: ToolEnvelope(success=False, result=None, error="access_denied")
        """
        try:
            self._grant_checker.check_paths(grant, job_dir, argv)
        except AccessDenied:
            logger.info("ShellTool.run access_denied")
            return _failed(_ACCESS_DENIED)
        return None


def _require_run_args(grant: AccessGrant, command: str, job_dir: Path) -> None:
    """
    Reject a grant, command, or job directory the shell cannot run.

    :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job")
    :param command: shell command as str. Example: "printf %s hello"
    :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
    :return: None. Example: None
    :raises TypeError: when an argument has the wrong type. Example: command=None
    :raises ValueError: when job_dir is not an existing directory. Example: Path("/missing")
    """
    if not isinstance(grant, AccessGrant):
        raise TypeError(f"grant must be AccessGrant, got {type(grant).__name__}")
    if not isinstance(command, str):
        raise TypeError(f"command must be str, got {type(command).__name__}")
    if not isinstance(job_dir, Path):
        raise TypeError(f"job_dir must be Path, got {type(job_dir).__name__}")
    if not job_dir.is_dir():
        raise ValueError(f"job_dir must be an existing directory, got {job_dir}")


def _parse_argv(command: str) -> tuple[str, ...] | ToolEnvelope:
    """
    Split a command into tokens without starting a process.

    :param command: shell command as str. Example: "printf %s hello"
    :return: tokens or a failure envelope as tuple[str, ...] | ToolEnvelope. Example: ("printf", "%s", "hello")
    """
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        logger.info("ShellTool.run unparsable_command")
        return _failed(_UNPARSABLE_COMMAND)
    if not argv:
        logger.info("ShellTool.run empty_command")
        return _failed(_EMPTY_COMMAND)
    return argv


def _start_process(argv: tuple[str, ...], job_dir: Path) -> subprocess.CompletedProcess[str]:
    """
    Run argv in the job directory and capture both streams.

    A missing program becomes exit 127 so the worker receives an envelope.

    :param argv: command tokens as tuple[str, ...]. Example: ("printf", "%s", "hello")
    :param job_dir: working directory as Path. Example: Path("data/jobs/7c1e0a2e")
    :return: completed process as subprocess.CompletedProcess[str]. Example: returncode 0 and stdout "hello"
    """
    try:
        return subprocess.run(
            argv,
            cwd=job_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, _COMMAND_NOT_FOUND, "", str(exc))


def _finish(completed: subprocess.CompletedProcess[str], job_dir: Path) -> ToolEnvelope:
    """
    Store the transcript, log the exit code, and build the envelope.

    :param completed: finished process as subprocess.CompletedProcess[str]. Example: returncode 0 and stdout "hello"
    :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
    :return: envelope as ToolEnvelope. Example: ToolEnvelope(success=True, result="hello", error="")
    """
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    _append_transcript(job_dir, stdout, stderr)
    logger.info(f"ShellTool.run exit_code={completed.returncode}")
    return _envelope_for(completed.returncode, stdout, stderr)


def _append_transcript(job_dir: Path, stdout: str, stderr: str) -> None:
    """
    Append captured streams to the job transcript.

    :param job_dir: job folder as Path. Example: Path("data/jobs/7c1e0a2e")
    :param stdout: captured stdout as str. Example: "hello"
    :param stderr: captured stderr as str. Example: "boom"
    :return: None. Example: None
    """
    with (job_dir / _SHELL_LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(stdout)
        handle.write(stderr)


def _envelope_for(returncode: int, stdout: str, stderr: str) -> ToolEnvelope:
    """
    Map an exit code to an envelope. Zero keeps stdout; any other code keeps stderr.

    :param returncode: process exit code as int. Example: 0
    :param stdout: captured stdout as str. Example: "hello"
    :param stderr: captured stderr as str. Example: "boom"
    :return: envelope as ToolEnvelope. Example: ToolEnvelope(success=True, result="hello", error="")
    """
    if returncode == 0:
        return ToolEnvelope(success=True, result=stdout, error="")
    return ToolEnvelope(success=False, result=stdout, error=stderr)


def _failed(error: str) -> ToolEnvelope:
    """
    Build a failure envelope for a command that did not produce output.

    :param error: failure code as str. Example: "access_denied"
    :return: envelope as ToolEnvelope. Example: ToolEnvelope(success=False, result=None, error="access_denied")
    """
    return ToolEnvelope(success=False, result=None, error=error)
