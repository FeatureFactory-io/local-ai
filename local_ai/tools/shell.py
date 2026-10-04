"""Shell tool. The worker's only tool. Calls GrantChecker, then the operating system."""

from pathlib import Path

from local_ai.contracts import AccessGrant, ToolEnvelope
from local_ai.tools.grant import GrantChecker


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
        # Split the command into tokens.
        # Call GrantChecker.check_paths. On AccessDenied, return the envelope and do not exec.
        # On success, run the command with cwd=job_dir and append stdout/stderr to job_dir/shell.log.
        # Non-zero exit returns success False and stderr in error. Do not raise.
        # Log the exit code at INFO. Do not log the full transcript.
        raise NotImplementedError
