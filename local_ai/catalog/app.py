"""Catalog HTTP API. Register is open. Listing the phone book requires a token."""

import hmac
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, current_app, jsonify, request

from local_ai.catalog.store import CatalogStore, HostRegistration
from local_ai.contracts import InvalidPriority, validate_priority


logger = logging.getLogger(__name__)

_HTTP_OK = 200
_HTTP_BAD = 400
_HTTP_UNAUTHORIZED = 401
_STATUS_REJECTED = "rejected"
_STATUS_OK = "ok"
_REASON_UNAUTHORIZED = "unauthorized"
_REASON_BAD_JSON = "bad_json"
_REASON_BAD_REQUEST = "bad_request"
_BEARER = "Bearer "
_CONFIG_STORE = "CATALOG_STORE"
_CONFIG_TOKEN = "CATALOG_TOKEN"
_CONFIG_STALE = "CATALOG_STALE_AFTER_S"
_CONFIG_CLOCK = "CATALOG_CLOCK"
_DEFAULT_STALE_S = 180
_NAME_MAX = 200
_URL_MAX = 2000


class _HttpReject(Exception):
    """
    A transport rejection with a stable reason.

    :param status_code: HTTP status as int. Example: 401
    :param reason: rejection reason as str. Example: "unauthorized"
    """

    def __init__(self, status_code: int, reason: str) -> None:
        """
        Store the status and reason.

        :param status_code: HTTP status as int. Example: 401
        :param reason: rejection reason as str. Example: "unauthorized"
        """
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def create_app(
    db_path: Path,
    catalog_token: str,
    stale_after_s: int = _DEFAULT_STALE_S,
    clock: Callable[[], str] | None = None,
) -> Flask:
    """
    Build the catalog app and migrate its database.

    :param db_path: sqlite file as Path. Example: Path("data/catalog.sqlite")
    :param catalog_token: bearer secret for GET /hosts as str. Example: "catalog-secret"
    :param stale_after_s: age limit in seconds as int. Example: 180
    :param clock: returns a UTC stamp as Callable[[], str] | None. Example: lambda: "2026-10-05T14:00:00Z"
    :return: app as Flask. Example: an app with POST /register and GET /hosts
    :raises TypeError: when db_path is not a Path. Example: "data/catalog.sqlite"
    :raises ValueError: when catalog_token is blank or stale_after_s is negative. Example: ""
    """
    token = _require_catalog_token(catalog_token)
    limit = _require_stale_after(stale_after_s)
    store = CatalogStore(db_path)
    store.migrate()
    application = Flask(__name__)
    application.config[_CONFIG_STORE] = store
    application.config[_CONFIG_TOKEN] = token
    application.config[_CONFIG_STALE] = limit
    application.config[_CONFIG_CLOCK] = _utc_now if clock is None else clock
    _register_routes(application)
    logger.info("catalog.create_app | exit | stale_after_s=%s", limit)
    return application


def _require_catalog_token(catalog_token: object) -> str:
    """
    Refuse to serve a catalog that has no read token.

    :param catalog_token: candidate secret as object. Example: "catalog-secret"
    :return: stripped secret as str. Example: "catalog-secret"
    :raises ValueError: when the secret is blank or not a string. Example: ""
    """
    if not isinstance(catalog_token, str) or not catalog_token.strip():
        raise ValueError("LOCAL_AI_CATALOG_TOKEN must be a non-empty secret. Example: 'catalog-secret'")
    return catalog_token.strip()


def _require_stale_after(stale_after_s: object) -> int:
    """
    Accept a non-negative second count.

    :param stale_after_s: candidate as object. Example: 180
    :return: seconds as int. Example: 180
    :raises ValueError: when the value is not an int or is negative. Example: -1
    """
    if type(stale_after_s) is not int or stale_after_s < 0:
        raise ValueError("stale_after_s must be an int >= 0. Example: 180")
    return stale_after_s


def _utc_now() -> str:
    """
    Current UTC time at whole seconds.

    :return: stamp as str. Example: "2026-10-05T14:00:00Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _register_routes(application: Flask) -> None:
    """
    Attach register, hosts, and the rejection handler.

    :param application: flask app as Flask. Example: Flask(__name__)
    :return: None. Example: None
    """
    application.post("/register")(_register)
    application.get("/hosts")(_hosts)
    application.register_error_handler(_HttpReject, _reject_response)


def _register() -> tuple[Response, int]:
    """
    Store one host. This route does not read the Authorization header.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok"}, 200)
    :raises _HttpReject: when the body is not a registration. Example: {}
    """
    registration = _registration(_payload())
    _store().upsert(registration, _stamp())
    logger.info(
        "catalog.register | ok | advertise_url=%s priorities=%s",
        registration.advertise_url,
        _priority_text(registration),
    )
    return _ok({"status": _STATUS_OK, "advertise_url": registration.advertise_url})


def _hosts() -> tuple[Response, int]:
    """
    List every host. The caller must present the catalog token.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok", "count": 0, "hosts": []}, 200)
    :raises _HttpReject: when the catalog token is missing or wrong. Example: no Authorization header
    """
    _authorize()
    hosts = _store().list_hosts(_stamp(), _stale_after())
    logger.info("catalog.hosts | ok | count=%s", len(hosts))
    return _ok({"status": _STATUS_OK, "count": len(hosts), "hosts": list(hosts)})


def _registration(payload: Mapping[str, object]) -> HostRegistration:
    """
    Validate a register body. Secrets are stored and not logged by this function.

    :param payload: JSON object as Mapping[str, object]. Example: {"name": "studio-mac", "advertise_url": "http://192.168.1.20:8770"}
    :return: registration as HostRegistration. Example: HostRegistration("studio-mac", "http://192.168.1.20:8770", ((50, "lane-50-secret"),), "{}", ("qwen-14b",))
    :raises _HttpReject: when a field is missing or the wrong type. Example: {}
    """
    try:
        return _registration_from(payload)
    except (InvalidPriority, ValueError, TypeError):
        logger.info("catalog.register | rejected | reason=%s", _REASON_BAD_REQUEST)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)


def _registration_from(payload: Mapping[str, object]) -> HostRegistration:
    """
    Build a registration from an object that already has the right shape.

    :param payload: JSON object as Mapping[str, object]. Example: {"name": "studio-mac"}
    :return: registration as HostRegistration. Example: HostRegistration(...)
    :raises _HttpReject: when a field is missing or the wrong type. Example: {}
    :raises InvalidPriority: when a token priority is outside 1..100. Example: 0
    """
    tokens = _token_rows(payload.get("tokens"))
    return HostRegistration(
        _name(payload.get("name")),
        _advertise_url(payload.get("advertise_url")),
        tokens,
        _capacity_json(payload.get("capacity")),
        _models(payload.get("models")),
    )


def _name(value: object) -> str:
    """
    Require a short non-empty host name.

    :param value: candidate as object. Example: "studio-mac"
    :return: name as str. Example: "studio-mac"
    :raises _HttpReject: when the name is blank or too long. Example: ""
    """
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > _NAME_MAX:
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return value.strip()


def _advertise_url(value: object) -> str:
    """
    Require an http or https URL.

    :param value: candidate as object. Example: "http://192.168.1.20:8770"
    :return: URL as str. Example: "http://192.168.1.20:8770"
    :raises _HttpReject: when the value is not an http URL. Example: "192.168.1.20"
    """
    if not isinstance(value, str):
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    url = value.strip()
    if not url.startswith(("http://", "https://")) or len(url) > _URL_MAX:
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return url


def _token_rows(value: object) -> tuple[tuple[int, str], ...]:
    """
    Require at least one priority token. Two rows may not share a secret.

    :param value: candidate as object. Example: [{"priority": 50, "token": "lane-50-secret"}]
    :return: pairs as tuple[tuple[int, str], ...]. Example: ((50, "lane-50-secret"),)
    :raises _HttpReject: when the list is empty or a secret repeats. Example: []
    :raises InvalidPriority: when a priority is outside 1..100. Example: 0
    """
    if not isinstance(value, list) or not value:
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    rows = tuple(sorted((_one_token(item) for item in value), key=_token_order))
    _require_unique_secrets(rows)
    return rows


def _one_token(item: object) -> tuple[int, str]:
    """
    Read one priority and secret.

    :param item: candidate as object. Example: {"priority": 50, "token": "lane-50-secret"}
    :return: pair as tuple[int, str]. Example: (50, "lane-50-secret")
    :raises _HttpReject: when the object is not a priority and a secret. Example: {}
    :raises InvalidPriority: when the priority is outside 1..100. Example: 0
    """
    if not isinstance(item, Mapping):
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    priority = item.get("priority")
    secret = item.get("token")
    if type(priority) is not int or not isinstance(secret, str) or not secret.strip():
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return validate_priority(priority), secret.strip()


def _token_order(row: tuple[int, str]) -> tuple[int, str]:
    """
    Sort by priority, then by secret, so the listing is stable.

    :param row: priority and secret as tuple[int, str]. Example: (50, "lane-50-secret")
    :return: sort key as tuple[int, str]. Example: (50, "lane-50-secret")
    """
    return row


def _require_unique_secrets(rows: tuple[tuple[int, str], ...]) -> None:
    """
    One secret maps to one priority inside a single registration.

    :param rows: pairs as tuple[tuple[int, str], ...]. Example: ((10, "lane-50-secret"), (50, "lane-50-secret"))
    :return: None. Example: None
    :raises _HttpReject: when the same secret appears twice. Example: two rows with "lane-50-secret"
    """
    secrets = [secret for _priority, secret in rows]
    if len(secrets) != len(set(secrets)):
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)


def _capacity_json(value: object) -> str:
    """
    Store the capacity object as JSON text.

    :param value: candidate as object. Example: {"status": "ok", "recommended": "qwen-14b"}
    :return: JSON as str. Example: '{"recommended": "qwen-14b", "status": "ok"}'
    :raises _HttpReject: when value is not a JSON object. Example: []
    """
    if not isinstance(value, dict):
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)


def _models(value: object) -> tuple[str, ...]:
    """
    Read logical model names. An empty list is a host with nothing to run.

    :param value: candidate as object. Example: ["qwen-14b"]
    :return: names as tuple[str, ...]. Example: ("qwen-14b",)
    :raises _HttpReject: when value is not a list of strings. Example: "qwen-14b"
    """
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return tuple(item.strip() for item in value)


def _authorize() -> None:
    """
    Compare the bearer token with the catalog secret. Neither value is logged.

    :return: None. Example: None
    :raises _HttpReject: when the token is missing or wrong. Example: ""
    """
    if _secrets_match(_bearer_token(), _catalog_token()):
        logger.info("catalog.hosts | auth | ok")
        return
    logger.info("catalog.hosts | rejected | reason=%s", _REASON_UNAUTHORIZED)
    raise _HttpReject(_HTTP_UNAUTHORIZED, _REASON_UNAUTHORIZED)


def _secrets_match(presented: str, expected: str) -> bool:
    """
    Compare two secrets in constant time when their lengths match.

    :param presented: bearer value as str. Example: "catalog-secret"
    :param expected: configured secret as str. Example: "catalog-secret"
    :return: whether they match as bool. Example: True
    """
    try:
        return hmac.compare_digest(presented, expected)
    except (TypeError, ValueError):
        return False


def _bearer_token() -> str:
    """
    Read the bearer secret. A missing header becomes an empty string.

    :return: secret as str. Example: "catalog-secret"
    """
    header = request.headers.get("Authorization", "")
    if not isinstance(header, str) or not header.startswith(_BEARER):
        return ""
    return header[len(_BEARER) :].strip()


def _payload() -> dict[str, object]:
    """
    Read a JSON object. An empty body is an empty object.

    :return: object as dict[str, object]. Example: {"name": "studio-mac"}
    :raises _HttpReject: when the body is not a JSON object. Example: "[]"
    """
    if request.get_data(cache=True) == b"":
        return {}
    parsed = request.get_json(silent=True)
    if not isinstance(parsed, dict):
        logger.info("catalog.register | rejected | reason=%s", _REASON_BAD_JSON)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_JSON)
    return parsed


def _priority_text(registration: HostRegistration) -> str:
    """
    Comma-separated priorities for the register log line.

    :param registration: stored body as HostRegistration. Example: HostRegistration(...)
    :return: priorities as str. Example: "10,50"
    """
    return ",".join(str(priority) for priority, _secret in registration.tokens)


def _stamp() -> str:
    """
    Read one UTC stamp from the app clock.

    :return: stamp as str. Example: "2026-10-05T14:00:00Z"
    :raises TypeError: when the clock does not return a string. Example: None
    """
    clock = current_app.config[_CONFIG_CLOCK]
    if not callable(clock):
        raise TypeError("clock must be callable. Example: lambda: '2026-10-05T14:00:00Z'")
    stamp = clock()
    if not isinstance(stamp, str) or not stamp.strip():
        raise TypeError("clock must return a non-empty string. Example: '2026-10-05T14:00:00Z'")
    return stamp


def _store() -> CatalogStore:
    """
    The phone book stored on the app.

    :return: store as CatalogStore. Example: CatalogStore(Path("data/catalog.sqlite"))
    """
    store = current_app.config[_CONFIG_STORE]
    if not isinstance(store, CatalogStore):
        raise TypeError("store must be a CatalogStore. Example: CatalogStore(Path('data/catalog.sqlite'))")
    return store


def _catalog_token() -> str:
    """
    The read secret stored on the app.

    :return: secret as str. Example: "catalog-secret"
    """
    token = current_app.config[_CONFIG_TOKEN]
    if not isinstance(token, str):
        raise TypeError("catalog token must be a string. Example: 'catalog-secret'")
    return token


def _stale_after() -> int:
    """
    The age limit stored on the app.

    :return: seconds as int. Example: 180
    """
    limit = current_app.config[_CONFIG_STALE]
    if type(limit) is not int:
        raise TypeError("stale_after_s must be an int. Example: 180")
    return limit


def _ok(body: dict) -> tuple[Response, int]:
    """
    A successful catalog result.

    :param body: response object as dict. Example: {"status": "ok"}
    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok"}, 200)
    """
    return jsonify(body), _HTTP_OK


def _reject_response(error: _HttpReject) -> tuple[Response, int]:
    """
    Render a transport rejection.

    :param error: rejection as _HttpReject. Example: _HttpReject(401, "unauthorized")
    :return: JSON and status as tuple[Response, int]. Example: ({"status": "rejected", "reason": "unauthorized"}, 401)
    """
    return jsonify({"status": _STATUS_REJECTED, "reason": error.reason}), error.status_code
