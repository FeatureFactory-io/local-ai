"""Database.connect and Database.migrate against a real sqlite file."""

import logging
import sqlite3
import threading
from pathlib import Path

import pytest

import local_ai.db.database as database_module
from local_ai.db.database import Database


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


def test_connect_creates_parent_and_shared_sqlite_settings(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """connect creates the parent directory, WAL, foreign keys, Row, and a lock."""
    database_path = tmp_path / "nested" / "local-ai.sqlite"
    database = Database(database_path)

    with caplog.at_level(logging.INFO, logger="local_ai.db.database"):
        connection = database.connect()

    assert database_path.parent.is_dir()
    assert database_path.is_file()
    assert connection.row_factory is sqlite3.Row
    row = connection.execute("SELECT 1 AS n").fetchone()
    assert row["n"] == 1
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert isinstance(database._lock, type(threading.Lock()))
    assert str(database_path) in caplog.text


def test_connect_allows_another_thread_to_use_the_connection(tmp_path: Path) -> None:
    """check_same_thread is False so the slot thread and the MCP thread can share it."""
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    errors: list[BaseException] = []

    def read_foreign_keys() -> None:
        try:
            connection.execute("PRAGMA foreign_keys").fetchone()
        except BaseException as exc:  # noqa: BLE001 — the test wants the thread's failure
            errors.append(exc)

    worker = threading.Thread(target=read_foreign_keys)
    worker.start()
    worker.join()
    assert errors == []


def test_connect_rejects_a_non_path() -> None:
    """connect fails fast when the constructor was given a plain string."""
    database = Database("data/local-ai.sqlite")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path must be Path"):
        database.connect()


def test_connect_returns_the_same_connection_and_lock(tmp_path: Path) -> None:
    """A second connect reuses the stored connection and lock."""
    database = Database(tmp_path / "local-ai.sqlite")
    first = database.connect()
    lock = database._lock
    second = database.connect()
    assert second is first
    assert database._lock is lock


def test_migrate_before_connect_raises(tmp_path: Path) -> None:
    """migrate refuses to run until connect has stored the connection."""
    database = Database(tmp_path / "local-ai.sqlite")
    with pytest.raises(RuntimeError, match="connect"):
        database.migrate()


def test_migrate_applies_schema_version_once(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The first migrate inserts version 1. The second returns 1 and keeps rows."""
    database_path = tmp_path / "data" / "local-ai.sqlite"
    database = Database(database_path)
    connection = database.connect()

    with caplog.at_level(logging.INFO, logger="local_ai.db.database"):
        assert database.migrate() == 1

    assert "version=1" in caplog.text
    assert _table_names(connection) >= {"schema_meta", "tasks", "plan_steps"}
    assert connection.execute("SELECT version FROM schema_meta").fetchone()["version"] == 1
    _insert_task(connection)

    original_schema = database_module.SCHEMA_SQL
    database_module.SCHEMA_SQL = "THIS IS NOT SQL;"
    try:
        assert database.migrate() == 1
    finally:
        database_module.SCHEMA_SQL = original_schema

    assert connection.execute("SELECT COUNT(*) AS n FROM schema_meta").fetchone()["n"] == 1
    assert connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO plan_steps (task_id, step_order, action, status) VALUES (?, ?, ?, ?)",
            ("missing-task", 1, "List chapter files", "pending"),
        )
    connection.rollback()
    assert connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_migrate_returns_existing_version_without_applying_schema(tmp_path: Path) -> None:
    """A schema_meta row is returned as-is, so tables are not rewritten."""
    database = Database(tmp_path / "local-ai.sqlite")
    connection = database.connect()
    connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
    connection.execute("INSERT INTO schema_meta (version) VALUES (2)")
    connection.commit()

    original_schema = database_module.SCHEMA_SQL
    database_module.SCHEMA_SQL = "THIS IS NOT SQL;"
    try:
        assert database.migrate() == 2
    finally:
        database_module.SCHEMA_SQL = original_schema

    assert "tasks" not in _table_names(connection)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    """Return sqlite table names. Example: {"schema_meta", "tasks"}."""
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row["name"]) for row in rows}


def _insert_task(connection: sqlite3.Connection) -> None:
    """Insert one tasks row so a later migrate must not wipe it."""
    connection.execute(
        """
        INSERT INTO tasks (
            id, priority, status, model, goal, done_when, access_json, phase, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _TASK_ID,
            40,
            "queued",
            "qwen-14b",
            "Summarize the book into 2-3 pages.",
            "summary.md exists",
            '{"mode": "job"}',
            "planning",
            "2026-10-04T18:40:00Z",
        ),
    )
    connection.commit()
