"""Access grants. ShellTool calls GrantChecker and does not interpret modes itself."""

from pathlib import Path

from local_ai.contracts import AccessDenied, AccessGrant


class GrantChecker:
    """Resolves job, paths, and home grants and rejects paths that leave them."""

    def resolve(
        self,
        mode: str,
        paths: tuple[str, ...],
        job_dir: Path,
        home: Path,
    ) -> AccessGrant:
        """
        Build the grant the shell will enforce.

        :param mode: job, paths, or home as str. Example: "paths"
        :param paths: caller directories as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
        :param job_dir: this task's folder as Path. Example: Path("data/jobs/7c1e0a2e")
        :param home: user home as Path. Example: Path("/Users/denispetelin")
        :return: grant as AccessGrant. Example: AccessGrant(mode="paths", paths=("/Users/denispetelin/books/draft",))
        :raises AccessDenied: when mode is unknown. Example: "root"
        """
        # mode "job": allowed root is job_dir only.
        # mode "paths": allowed roots are the given paths plus job_dir.
        # mode "home": allowed root is home.
        # Log the mode and the roots at INFO. Do not log file contents.
        raise NotImplementedError

    def check_paths(self, grant: AccessGrant, cwd: Path, argv: tuple[str, ...]) -> None:
        """
        Reject a command whose cwd or path-like argument leaves the grant.

        :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job")
        :param cwd: working directory as Path. Example: Path("data/jobs/7c1e0a2e")
        :param argv: command tokens as tuple[str, ...]. Example: ("cat", "/etc/passwd")
        :return: None. Example: None
        :raises AccessDenied: when cwd or a resolved path argument is outside the grant. Example: Path("/etc/passwd")
        """
        # Resolve cwd and every token that looks like a path, including ".." that escapes.
        # Compare each resolved path to the grant roots.
        # Raise AccessDenied before any process is started.
        # Log the denied path at WARNING. Log an allowed cwd at INFO.
        raise NotImplementedError
