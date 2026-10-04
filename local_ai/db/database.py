"""SQLite connection and schema_version 1. Blackboard and PlanStore share this object."""

import logging
import sqlite3
import threading
from pathlib import Path


logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    goal TEXT NOT NULL,
    done_when TEXT NOT NULL,
    access_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    current_plan TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '[]',
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    final_status TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS plan_steps (
    id INTEGER PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(id),
    step_order INTEGER NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    llm_synthesis TEXT NOT NULL DEFAULT '',
    UNIQUE (task_id, step_order)
);
"""


class Database:
    """
    The only writer of data/local-ai.sqlite.

    Blackboard and PlanStore call this class. They do not open their own connections.
    """

    def __init__(self, path: Path) -> None:
        """
        Hold the database path. The connection opens in connect.

        :param path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
        """
        # Store path. Do not open the file until connect.
        self._path = path
        self._connection: sqlite3.Connection | None = None
        self._lock: threading.Lock | None = None

    def connect(self) -> sqlite3.Connection:
        """
        Open the single connection.

        :return: connection as sqlite3.Connection. Example: a connection with row_factory Row
        :raises TypeError: when path is not a Path. Example: "data/local-ai.sqlite"
        """
        self._validate_path()
        if self._connection is not None:
            logger.info("Database.connect reused path=%s", self._path)
            return self._connection
        self._prepare_parent_directory()
        connection = self._open_connection()
        self._connection = connection
        self._lock = threading.Lock()
        logger.info("Database.connect path=%s", self._path)
        return connection

    def _validate_path(self) -> None:
        """
        Reject a path that is not a Path.

        :return: None. Example: None
        :raises TypeError: when path is not a Path. Example: "data/local-ai.sqlite"
        """
        if not isinstance(self._path, Path):
            raise TypeError(f"path must be Path, got {type(self._path).__name__}")

    def _prepare_parent_directory(self) -> None:
        """
        Create the parent directory for the sqlite file.

        :return: None. Example: None
        """
        parent = self._path.parent
        logger.info("Database.connect creating parent=%s", parent)
        parent.mkdir(parents=True, exist_ok=True)

    def _open_connection(self) -> sqlite3.Connection:
        """
        Open sqlite for shared threads, with Row rows, foreign keys, and WAL.

        :return: connection as sqlite3.Connection. Example: a connection with row_factory Row
        """
        # The slot thread and the MCP thread share this connection.
        connection = sqlite3.connect(self._path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        journal_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
        journal_mode = journal_row[0] if journal_row is not None else ""
        logger.info("Database.connect journal_mode=%s", journal_mode)
        return connection

    def migrate(self) -> int:
        """
        Apply schema version 1. A second call returns the same version.

        :return: schema version as int. Example: 1
        """
        # Take the connection lock.
        # If schema_meta already has a row, return that version and do not rewrite tables.
        # Otherwise execute SCHEMA_SQL and insert schema_version 1.
        # Commit. Log the version at INFO.
        raise NotImplementedError

    @property
    def path(self) -> Path:
        """
        Path passed to the constructor.

        :return: sqlite path as Path. Example: Path("data/local-ai.sqlite")
        """
        return self._path
