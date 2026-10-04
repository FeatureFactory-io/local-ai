"""Database.connect and Database.migrate against a real sqlite file."""

import logging
import sqlite3
import threading
from pathlib import Path

import pytest

from local_ai.db.database import Database


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
