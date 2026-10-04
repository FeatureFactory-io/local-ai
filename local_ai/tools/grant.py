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
        :raises TypeError: when grant, cwd, or argv has the wrong type. Example: cwd="/tmp"
        """
        self._validate_check_inputs(grant, cwd, argv)
        roots = _resolved_roots(grant)
        resolved_cwd = cwd.resolve()
        self._require_inside(resolved_cwd, roots)
        self._require_argv_inside(resolved_cwd, argv, roots)
        logger.info(f"GrantChecker.check_paths allowed cwd={resolved_cwd}")

    def _validate_check_inputs(
        self,
        grant: AccessGrant,
        cwd: Path,
        argv: tuple[str, ...],
    ) -> None:
        """
        Reject arguments that are not a grant, a directory, and command tokens.

        :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job")
        :param cwd: working directory as Path. Example: Path("data/jobs/7c1e0a2e")
        :param argv: command tokens as tuple[str, ...]. Example: ("cat", "/etc/passwd")
        :return: None. Example: None
        :raises TypeError: when an argument has the wrong type. Example: cwd="/tmp"
        """
        if not isinstance(grant, AccessGrant):
            raise TypeError(f"grant must be AccessGrant, got {type(grant).__name__}")
        if not isinstance(cwd, Path):
            raise TypeError(f"cwd must be Path, got {type(cwd).__name__}")
        if not isinstance(argv, tuple) or any(not isinstance(token, str) for token in argv):
            raise TypeError("argv must be tuple[str, ...]")

    def _require_argv_inside(
        self,
        cwd: Path,
        argv: tuple[str, ...],
        roots: tuple[Path, ...],
    ) -> None:
        """
        Reject every path-like token that resolves outside the grant.

        :param cwd: resolved working directory as Path. Example: Path("/data/jobs/7c1e0a2e")
        :param argv: command tokens as tuple[str, ...]. Example: ("cat", "../secret.md")
        :param roots: allowed directories as tuple[Path, ...]. Example: (Path("/data/jobs/7c1e0a2e"),)
        :return: None. Example: None
        :raises AccessDenied: when a token resolves outside the roots. Example: Path("/etc/passwd")
        """
        for token in argv:
            if _is_path_token(token):
                self._require_inside(_resolve_token(cwd, token), roots)

    def _require_inside(self, path: Path, roots: tuple[Path, ...]) -> None:
        """
        Raise when path is not the root and not inside one.

        :param path: resolved path as Path. Example: Path("/etc/passwd")
        :param roots: allowed directories as tuple[Path, ...]. Example: (Path("/data/jobs/7c1e0a2e"),)
        :return: None. Example: None
        :raises AccessDenied: when path leaves every root. Example: Path("/etc/passwd")
        """
        if any(path == root or path.is_relative_to(root) for root in roots):
            return
        logger.warning(f"GrantChecker.check_paths denied path={path}")
        raise AccessDenied(path)


def _resolved_roots(grant: AccessGrant) -> tuple[Path, ...]:
    """
    Resolve each stored root so .. and symlinks compare as real directories.

    :param grant: resolved grant as AccessGrant. Example: AccessGrant(mode="job", paths=("data/jobs/7c1e0a2e",))
    :return: absolute roots as tuple[Path, ...]. Example: (Path("/repo/data/jobs/7c1e0a2e"),)
    """
    return tuple(Path(root).resolve() for root in grant.paths)


def _is_path_token(token: str) -> bool:
    """
    Decide whether the shell would treat this token as a filesystem path.

    :param token: one argv token as str. Example: "../books/secret.md"
    :return: True when the token is path-like as bool. Example: True
    """
    if token in {".", ".."} or token.startswith(("/", "./", "../", "~")):
        return True
    return "/" in token or "\\" in token


def _resolve_token(cwd: Path, token: str) -> Path:
    """
    Resolve a path token against cwd, collapsing .. before the grant compare.

    :param cwd: resolved working directory as Path. Example: Path("/data/jobs/7c1e0a2e")
    :param token: path-like argv token as str. Example: "../books/secret.md"
    :return: absolute path as Path. Example: Path("/data/books/secret.md")
    """
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (cwd / candidate).resolve()
