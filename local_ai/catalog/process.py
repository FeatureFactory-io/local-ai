"""Process entry for the catalog. The read token is loaded before the database opens."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from flask import Flask

from local_ai.catalog.app import create_app


logger = logging.getLogger(__name__)

_ENV_TOKEN = "LOCAL_AI_CATALOG_TOKEN"
_ENV_DB = "LOCAL_AI_CATALOG_DB"
_ENV_HOST = "LOCAL_AI_CATALOG_HOST"
_ENV_PORT = "LOCAL_AI_CATALOG_PORT"
_ENV_STALE = "LOCAL_AI_CATALOG_STALE_S"
_DEFAULT_DB = Path("data/catalog.sqlite")
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8771
_DEFAULT_STALE_S = 180
_MIN_PORT = 1
_MAX_PORT = 65535


class CatalogConfigError(ValueError):
    """Raised when the catalog environment cannot be loaded."""


@dataclass(frozen=True)
class CatalogSettings:
    """
    Where the catalog listens and which file it writes.

    :param host: bind address as str. Example: "127.0.0.1"
    :param port: bind port as int. Example: 8771
    :param db_path: sqlite file as Path. Example: Path("data/catalog.sqlite")
    :param stale_after_s: age limit in seconds as int. Example: 180
    """

    host: str
    port: int
    db_path: Path
    stale_after_s: int


def bind_settings(env: Mapping[str, str] | None = None) -> CatalogSettings:
    """
    Read the bind address, database path, and stale limit.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_CATALOG_PORT": "8771"}
    :return: settings as CatalogSettings. Example: CatalogSettings("127.0.0.1", 8771, Path("data/catalog.sqlite"), 180)
    :raises TypeError: when env is not a mapping. Example: []
    :raises CatalogConfigError: when the host is blank or the port is outside 1..65535. Example: "0"
    """
    source = _environment(env)
    settings = CatalogSettings(
        _host(source.get(_ENV_HOST)),
        _port(source.get(_ENV_PORT)),
        _db_path(source.get(_ENV_DB)),
        _stale_after(source.get(_ENV_STALE)),
    )
    logger.info(
        "catalog.bind_settings | exit | host=%s port=%s stale_after_s=%s",
        settings.host,
        settings.port,
        settings.stale_after_s,
    )
    return settings


def prepare(env: Mapping[str, str] | None = None) -> tuple[Flask, CatalogSettings]:
    """
    Load the catalog token, then open the database. A missing token does not create the file.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_CATALOG_TOKEN": "catalog-secret"}
    :return: app and settings as tuple[Flask, CatalogSettings]. Example: (app, CatalogSettings(...))
    :raises CatalogConfigError: when the token is missing or the port is illegal. Example: {}
    :raises TypeError: when env is not a mapping. Example: []
    """
    logger.info("catalog.prepare | entry")
    source = _environment(env)
    token = _catalog_token(source.get(_ENV_TOKEN))
    settings = bind_settings(source)
    application = create_app(settings.db_path, token, settings.stale_after_s)
    logger.info("catalog.prepare | exit | host=%s port=%s", settings.host, settings.port)
    return application, settings


def main() -> None:
    """
    Serve the catalog. Refuses to bind when LOCAL_AI_CATALOG_TOKEN is unset.

    :return: None. Example: None
    :raises CatalogConfigError: when the token is missing or the port is illegal. Example: {}
    """
    logger.info("catalog.main | entry")
    application, settings = prepare(None)
    _serve(application, settings)


def _environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """
    Use the passed mapping, or the process environment when env is omitted.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_CATALOG_TOKEN": "catalog-secret"}
    :return: mapping as Mapping[str, str]. Example: {"LOCAL_AI_CATALOG_TOKEN": "catalog-secret"}
    :raises TypeError: when env is not a mapping. Example: []
    """
    if env is None:
        return os.environ
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_CATALOG_TOKEN': 'catalog-secret'}")
    return env


def _catalog_token(raw: object) -> str:
    """
    Require the read secret. The value is not logged.

    :param raw: environment value as object. Example: "catalog-secret"
    :return: secret as str. Example: "catalog-secret"
    :raises CatalogConfigError: when the secret is missing or blank. Example: None
    """
    if not isinstance(raw, str) or not raw.strip():
        raise CatalogConfigError(
            "LOCAL_AI_CATALOG_TOKEN must be a non-empty secret. Example: 'catalog-secret'"
        )
    return raw.strip()


def _host(raw: object) -> str:
    """
    Bind address. A missing value stays on loopback.

    :param raw: environment value as object. Example: "0.0.0.0"
    :return: host as str. Example: "127.0.0.1"
    :raises CatalogConfigError: when raw is present and blank. Example: "  "
    """
    if raw is None:
        return _DEFAULT_HOST
    if not isinstance(raw, str) or not raw.strip():
        raise CatalogConfigError("LOCAL_AI_CATALOG_HOST must be a host. Example: '127.0.0.1'")
    return raw.strip()


def _port(raw: object) -> int:
    """
    Parse LOCAL_AI_CATALOG_PORT. A missing value is 8771.

    :param raw: port text as object. Example: "8771"
    :return: port as int. Example: 8771
    :raises CatalogConfigError: when the port is not an integer from 1 through 65535. Example: "0"
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_PORT
    if not isinstance(raw, str) or not raw.strip().isdigit():
        raise CatalogConfigError("LOCAL_AI_CATALOG_PORT must be an integer. Example: 8771")
    port = int(raw.strip())
    if port < _MIN_PORT or port > _MAX_PORT:
        raise CatalogConfigError("LOCAL_AI_CATALOG_PORT must be from 1 through 65535. Example: 8771")
    return port


def _db_path(raw: object) -> Path:
    """
    Catalog sqlite path. A missing value is data/catalog.sqlite.

    :param raw: environment value as object. Example: "data/catalog.sqlite"
    :return: path as Path. Example: Path("data/catalog.sqlite")
    """
    if not isinstance(raw, str) or not raw.strip():
        return _DEFAULT_DB
    return Path(raw.strip())


def _stale_after(raw: object) -> int:
    """
    Age limit in seconds. A missing value is 180.

    :param raw: environment value as object. Example: "180"
    :return: seconds as int. Example: 180
    :raises CatalogConfigError: when the value is not a non-negative integer. Example: "-1"
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_STALE_S
    if not isinstance(raw, str) or not raw.strip().isdigit():
        raise CatalogConfigError("LOCAL_AI_CATALOG_STALE_S must be an integer. Example: 180")
    return int(raw.strip())


def _serve(application: Flask, settings: CatalogSettings) -> None:
    """
    Bind the catalog. One request at a time keeps the sqlite writes ordered.

    :param application: flask app as Flask. Example: the app from prepare
    :param settings: bind address as CatalogSettings. Example: CatalogSettings("127.0.0.1", 8771, Path("data/catalog.sqlite"), 180)
    :return: None. Example: None
    """
    logger.info("catalog.main | bind | host=%s port=%s", settings.host, settings.port)
    application.run(
        host=settings.host,
        port=settings.port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )
