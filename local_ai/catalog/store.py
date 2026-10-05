"""Phone book of JCL hosts. One row per advertise URL."""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

_STAMP = "%Y-%m-%dT%H:%M:%SZ"
_CREATE_HOSTS = """
CREATE TABLE IF NOT EXISTS hosts (
    advertise_url TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    capacity_json TEXT NOT NULL,
    models_json TEXT NOT NULL,
    last_seen TEXT NOT NULL
)
"""
_UPSERT = """
INSERT INTO hosts (advertise_url, name, tokens_json, capacity_json, models_json, last_seen)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(advertise_url) DO UPDATE SET
    name = excluded.name,
    tokens_json = excluded.tokens_json,
    capacity_json = excluded.capacity_json,
    models_json = excluded.models_json,
    last_seen = excluded.last_seen
"""
_LIST = """
SELECT advertise_url, name, tokens_json, capacity_json, models_json, last_seen
FROM hosts
ORDER BY advertise_url
"""


@dataclass(frozen=True)
class HostRegistration:
    """
    One catalog register body, after validation.

    :param name: host name as str. Example: "studio-mac"
    :param advertise_url: URL other machines curl as str. Example: "http://192.168.1.20:8770"
    :param tokens: priority and secret pairs as tuple[tuple[int, str], ...]. Example: ((50, "lane-50-secret"),)
    :param capacity_json: capacity tool body as str. Example: '{"status": "ok"}'
    :param models: logical model names as tuple[str, ...]. Example: ("qwen-14b",)
    """

    name: str
    advertise_url: str
    tokens: tuple[tuple[int, str], ...]
    capacity_json: str
    models: tuple[str, ...]


class CatalogStore:
    """
    SQLite phone book. The catalog process is the only writer.
    """

    def __init__(self, path: Path) -> None:
        """
        Open the catalog database and create the parent directory.

        :param path: sqlite file as Path. Example: Path("data/catalog.sqlite")
        :raises TypeError: when path is not a Path. Example: "data/catalog.sqlite"
        """
        _require_path(path)
        self._path = path
        self._lock = threading.Lock()
        self._connection = _open_connection(path)

    def migrate(self) -> None:
        """
        Create the hosts table when it is missing.

        :return: None. Example: None
        """
        logger.info("catalog.migrate | entry | path=%s", self._path.name)
        with self._lock:
            self._connection.execute(_CREATE_HOSTS)
            self._connection.commit()
        logger.info("catalog.migrate | exit | path=%s", self._path.name)

    def upsert(self, registration: HostRegistration, seen_at: str) -> None:
        """
        Insert or replace the row for one advertise URL.

        :param registration: validated body as HostRegistration. Example: HostRegistration("studio-mac", "http://192.168.1.20:8770", ((50, "lane-50-secret"),), "{}", ())
        :param seen_at: UTC stamp as str. Example: "2026-10-05T14:00:00Z"
        :return: None. Example: None
        :raises TypeError: when registration is not a HostRegistration. Example: {}
        :raises ValueError: when seen_at is blank. Example: "  "
        """
        _require_registration(registration)
        stamp = _require_stamp(seen_at)
        logger.info("catalog.upsert | entry | advertise_url=%s", registration.advertise_url)
        with self._lock:
            self._write(registration, stamp)
        logger.info("catalog.upsert | exit | advertise_url=%s", registration.advertise_url)

    def list_hosts(self, now: str, stale_after_s: int) -> tuple[dict[str, object], ...]:
        """
        Every row, oldest URL first, with a stale flag.

        :param now: UTC stamp as str. Example: "2026-10-05T14:03:01Z"
        :param stale_after_s: age limit in seconds as int. Example: 180
        :return: host bodies as tuple[dict[str, object], ...]. Example: ({"advertise_url": "http://192.168.1.20:8770", "stale": False},)
        :raises ValueError: when now is blank or stale_after_s is negative. Example: -1
        """
        stamp = _require_stamp(now)
        limit = _require_stale_after(stale_after_s)
        logger.info("catalog.list_hosts | entry | stale_after_s=%s", limit)
        with self._lock:
            rows = self._connection.execute(_LIST).fetchall()
        bodies = tuple(_host_body(row, stamp, limit) for row in rows)
        logger.info("catalog.list_hosts | exit | count=%s", len(bodies))
        return bodies

    def _write(self, registration: HostRegistration, seen_at: str) -> None:
        """
        Replace one host row. The caller holds the lock.

        :param registration: validated body as HostRegistration. Example: HostRegistration("studio-mac", "http://192.168.1.20:8770", ((50, "lane-50-secret"),), "{}", ())
        :param seen_at: UTC stamp as str. Example: "2026-10-05T14:00:00Z"
        :return: None. Example: None
        """
        self._connection.execute(_UPSERT, _row_values(registration, seen_at))
        self._connection.commit()


def _require_path(path: object) -> None:
    """
    Reject anything that is not a file path.

    :param path: candidate as object. Example: Path("data/catalog.sqlite")
    :return: None. Example: None
    :raises TypeError: when path is not a Path. Example: "data/catalog.sqlite"
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a Path. Example: Path('data/catalog.sqlite')")


def _open_connection(path: Path) -> sqlite3.Connection:
    """
    Open sqlite for the catalog process.

    :param path: sqlite file as Path. Example: Path("data/catalog.sqlite")
    :return: connection as sqlite3.Connection. Example: a connection with row_factory Row
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    logger.info("catalog.connect | path=%s", path.name)
    return connection


def _require_registration(registration: object) -> None:
    """
    Reject a body that skipped validation.

    :param registration: candidate as object. Example: HostRegistration(...)
    :return: None. Example: None
    :raises TypeError: when registration is not a HostRegistration. Example: {}
    """
    if not isinstance(registration, HostRegistration):
        raise TypeError("registration must be a HostRegistration. Example: HostRegistration(...)")


def _require_stamp(seen_at: object) -> str:
    """
    Require a non-empty timestamp string.

    :param seen_at: candidate stamp as object. Example: "2026-10-05T14:00:00Z"
    :return: stamp as str. Example: "2026-10-05T14:00:00Z"
    :raises ValueError: when seen_at is blank or not a string. Example: "  "
    """
    if not isinstance(seen_at, str) or not seen_at.strip():
        raise ValueError("seen_at must be a non-empty string. Example: '2026-10-05T14:00:00Z'")
    return seen_at


def _require_stale_after(stale_after_s: object) -> int:
    """
    Accept a non-negative second count. True is not a count.

    :param stale_after_s: candidate as object. Example: 180
    :return: seconds as int. Example: 180
    :raises ValueError: when the value is not an int or is negative. Example: -1
    """
    if type(stale_after_s) is not int or stale_after_s < 0:
        raise ValueError("stale_after_s must be an int >= 0. Example: 180")
    return stale_after_s


def _row_values(registration: HostRegistration, seen_at: str) -> tuple[str, str, str, str, str, str]:
    """
    Bind values for the upsert. Secrets stay inside tokens_json and are not logged here.

    :param registration: validated body as HostRegistration. Example: HostRegistration("studio-mac", "http://192.168.1.20:8770", ((50, "lane-50-secret"),), "{}", ())
    :param seen_at: UTC stamp as str. Example: "2026-10-05T14:00:00Z"
    :return: column values as tuple[str, str, str, str, str, str]. Example: ("http://192.168.1.20:8770", "studio-mac", "[]", "{}", "[]", "2026-10-05T14:00:00Z")
    """
    tokens = json.dumps([{"priority": priority, "token": token} for priority, token in registration.tokens])
    models = json.dumps(list(registration.models))
    return (
        registration.advertise_url,
        registration.name,
        tokens,
        registration.capacity_json,
        models,
        seen_at,
    )


def _host_body(row: sqlite3.Row, now: str, stale_after_s: int) -> dict[str, object]:
    """
    One list item, including the tokens a catalog reader needs in order to curl the host.

    :param row: sqlite row as sqlite3.Row. Example: a hosts row
    :param now: UTC stamp as str. Example: "2026-10-05T14:03:01Z"
    :param stale_after_s: age limit in seconds as int. Example: 180
    :return: host as dict[str, object]. Example: {"advertise_url": "http://192.168.1.20:8770", "stale": False}
    """
    last_seen = str(row["last_seen"])
    return {
        "name": row["name"],
        "advertise_url": row["advertise_url"],
        "tokens": _json_list(row["tokens_json"]),
        "capacity": _json_object(row["capacity_json"]),
        "models": _json_list(row["models_json"]),
        "last_seen": last_seen,
        "stale": _is_stale(last_seen, now, stale_after_s),
    }


def _json_list(raw: object) -> list:
    """
    Decode a JSON list. Corrupt text becomes an empty list.

    :param raw: stored JSON as object. Example: '[{"priority": 50, "token": "lane-50-secret"}]'
    :return: list as list. Example: [{"priority": 50, "token": "lane-50-secret"}]
    """
    parsed = _json_value(raw, [])
    if isinstance(parsed, list):
        return parsed
    return []


def _json_object(raw: object) -> dict:
    """
    Decode a JSON object. Corrupt text becomes an empty object.

    :param raw: stored JSON as object. Example: '{"status": "ok"}'
    :return: object as dict. Example: {"status": "ok"}
    """
    parsed = _json_value(raw, {})
    if isinstance(parsed, dict):
        return parsed
    return {}


def _json_value(raw: object, fallback: object) -> object:
    """
    Decode JSON text. A bad cell does not fail the whole listing.

    :param raw: stored text as object. Example: '{"status": "ok"}'
    :param fallback: value when decoding fails as object. Example: {}
    :return: decoded value as object. Example: {"status": "ok"}
    """
    if not isinstance(raw, str):
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("catalog.hosts | bad_json")
        return fallback


def _is_stale(last_seen: str, now: str, stale_after_s: int) -> bool:
    """
    True when the row is strictly older than the limit.

    An unreadable stamp is stale so a broken clock does not look fresh.

    :param last_seen: stored stamp as str. Example: "2026-10-05T14:00:00Z"
    :param now: current stamp as str. Example: "2026-10-05T14:03:01Z"
    :param stale_after_s: age limit in seconds as int. Example: 180
    :return: whether the row is stale as bool. Example: True
    """
    seen = _parse_stamp(last_seen)
    current = _parse_stamp(now)
    if seen is None or current is None:
        logger.info("catalog.stale | branch=unreadable")
        return True
    age = (current - seen).total_seconds()
    stale = age > stale_after_s
    logger.info("catalog.stale | age_s=%s stale=%s", int(age), stale)
    return stale


def _parse_stamp(stamp: str) -> datetime | None:
    """
    Parse one UTC stamp. None means the text is not the catalog format.

    :param stamp: timestamp as str. Example: "2026-10-05T14:00:00Z"
    :return: datetime as datetime | None. Example: datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)
    """
    try:
        parsed = datetime.strptime(stamp, _STAMP)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)
