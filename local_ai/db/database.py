"""SQLite connection and schema_version 1. Blackboard and PlanStore share this object."""

import sqlite3
from pathlib import Path


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

    def connect(self) -> sqlite3.Connection:
        """
        Open the single connection.

        :return: connection as sqlite3.Connection. Example: a connection with row_factory Row
        """
        # Create the parent directory.
        # sqlite3.connect(check_same_thread=False) because the slot thread and the MCP thread share it.
        # Set row_factory to sqlite3.Row.
        # Execute PRAGMA foreign_keys=ON and PRAGMA journal_mode=WAL.
        # Store the connection and a threading.Lock that every later call must take.
        # Log the path at INFO.
        raise NotImplementedError

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
