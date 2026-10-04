"""Access grants. ShellTool calls GrantChecker and does not interpret modes itself."""

import logging
from pathlib import Path

from local_ai.contracts import AccessDenied, AccessGrant

logger = logging.getLogger(__name__)

_MODE_JOB = "job"
_MODE_PATHS = "paths"
_MODE_HOME = "home"
_KNOWN_MODES = frozenset({_MODE_JOB, _MODE_PATHS, _MODE_HOME})


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
        :return: grant as AccessGrant. Example: AccessGrant(mode="paths", paths=("/Users/denispetelin/books/draft", "data/jobs/7c1e0a2e"))
        :raises AccessDenied: when mode is unknown. Example: "root"
        :raises TypeError: when mode, paths, job_dir, or home has the wrong type. Example: mode=None
        """
        self._validate_resolve_inputs(mode, paths, job_dir, home)
        roots = self._roots_for(mode, paths, job_dir, home)
        logger.info(f"GrantChecker.resolve mode={mode} roots={roots}")
        return AccessGrant(mode=mode, paths=roots)

    def _validate_resolve_inputs(
        self,
        mode: str,
        paths: tuple[str, ...],
        job_dir: Path,
        home: Path,
    ) -> None:
        """
        Reject a mode or argument the shell cannot enforce.

        :param mode: job, paths, or home as str. Example: "job"
        :param paths: caller directories as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
        :param job_dir: this task's folder as Path. Example: Path("data/jobs/7c1e0a2e")
        :param home: user home as Path. Example: Path("/Users/denispetelin")
        :return: None. Example: None
        :raises AccessDenied: when mode is unknown. Example: "root"
        :raises TypeError: when an argument has the wrong type. Example: mode=None
        """
        if not isinstance(mode, str):
            raise TypeError(f"mode must be str, got {type(mode).__name__}")
        if not isinstance(paths, tuple) or any(not isinstance(item, str) for item in paths):
            raise TypeError("paths must be tuple[str, ...]")
        if not isinstance(job_dir, Path) or not isinstance(home, Path):
            raise TypeError("job_dir and home must be Path")
        if mode not in _KNOWN_MODES:
            logger.warning(f"GrantChecker.resolve denied unknown mode={mode!r}")
            raise AccessDenied(f"unknown grant mode: {mode!r}")

    def _roots_for(
        self,
        mode: str,
        paths: tuple[str, ...],
        job_dir: Path,
        home: Path,
    ) -> tuple[str, ...]:
        """
        Pick the directories this mode allows.

        :param mode: known mode as str. Example: "paths"
        :param paths: caller directories as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft",)
        :param job_dir: this task's folder as Path. Example: Path("data/jobs/7c1e0a2e")
        :param home: user home as Path. Example: Path("/Users/denispetelin")
        :return: allowed roots as tuple[str, ...]. Example: ("/Users/denispetelin/books/draft", "data/jobs/7c1e0a2e")
        """
        if mode == _MODE_JOB:
            return (str(job_dir),)
        if mode == _MODE_PATHS:
            return tuple(paths) + (str(job_dir),)
        return (str(home),)

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
