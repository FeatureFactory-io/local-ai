"""Process entry for the JCL API. Tokens are loaded before the socket opens."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from flask import Flask

from local_ai.dispatcher.server import compose
from local_ai.jcl.app import create_app
from local_ai.jcl.tokens import load_token_table


logger = logging.getLogger(__name__)

_ENV_HOST = "LOCAL_AI_JCL_HOST"
_ENV_PORT = "LOCAL_AI_JCL_PORT"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8770
_MIN_PORT = 1
_MAX_PORT = 65535


@dataclass(frozen=True)
class BindSettings:
    """
    Where the JCL process listens.

    :param host: bind address as str. Example: "127.0.0.1"
    :param port: bind port as int. Example: 8770
    """

    host: str
    port: int


def bind_settings(env: Mapping[str, str] | None = None) -> BindSettings:
    """
    Read the bind address. The default is loopback port 8770.

    A non-loopback host is legal because the routes require a bearer token.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_JCL_HOST": "0.0.0.0"}
    :return: settings as BindSettings. Example: BindSettings("127.0.0.1", 8770)
    :raises TypeError: when env is not a mapping. Example: []
    :raises ValueError: when the host is blank or the port is outside 1..65535. Example: "0"
    """
    source = _environment(env)
    settings = BindSettings(_host(source.get(_ENV_HOST)), _port(source.get(_ENV_PORT)))
    logger.info("bind_settings | exit | host=%s port=%s", settings.host, settings.port)
    return settings


def prepare(env: Mapping[str, str] | None = None) -> tuple[Flask, BindSettings]:
    """
    Load tokens, then compose the dispatcher. An empty token table does not open the database.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_JCL_TOKEN_50": "lane-50-secret"}
    :return: app and bind settings as tuple[Flask, BindSettings]. Example: (app, BindSettings("127.0.0.1", 8770))
    :raises TokenTableError: when no JCL token is set. Example: {}
    :raises ValueError: when the port is outside 1..65535. Example: "0"
    :raises TypeError: when env is not a mapping. Example: []
    """
    logger.info("prepare | entry")
    source = _environment(env)
    tokens = load_token_table(source)
    settings = bind_settings(source)
    server, _mcp = compose(source)
    logger.info("prepare | exit | host=%s port=%s", settings.host, settings.port)
    return create_app(server, tokens), settings


def main() -> None:
    """
    Serve the JCL API. Refuses to bind when no token is set.

    :return: None. Example: None
    :raises TokenTableError: when no JCL token is set. Example: {}
    :raises ValueError: when the port is outside 1..65535. Example: "0"
    """
    logger.info("main | entry")
    application, settings = prepare(None)
    _serve(application, settings)


def _environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """
    Use the passed mapping, or the process environment when env is omitted.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_JCL_PORT": "8770"}
    :return: mapping as Mapping[str, str]. Example: {"LOCAL_AI_JCL_PORT": "8770"}
    :raises TypeError: when env is not a mapping. Example: []
    """
    if env is None:
        return os.environ
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_JCL_PORT': '8770'}")
    return env


def _host(raw: object) -> str:
    """
    Bind address. A missing value stays on loopback.

    :param raw: environment value as object. Example: "0.0.0.0"
    :return: host as str. Example: "127.0.0.1"
    :raises ValueError: when raw is present and blank. Example: "  "
    """
    if raw is None:
        return _DEFAULT_HOST
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LOCAL_AI_JCL_HOST must be a host. Example: '127.0.0.1'")
    return raw.strip()


def _port(raw: object) -> int:
    """
    Parse LOCAL_AI_JCL_PORT. A missing value is 8770.

    :param raw: port text as object. Example: "8770"
    :return: port as int. Example: 8770
    :raises ValueError: when the port is not an integer from 1 through 65535. Example: "0"
    """
    if raw is None:
        return _DEFAULT_PORT
    if not isinstance(raw, str) or not raw.strip():
        return _DEFAULT_PORT
    text = raw.strip()
    if not text.isdigit():
        raise ValueError("LOCAL_AI_JCL_PORT must be an integer. Example: 8770")
    port = int(text)
    if port < _MIN_PORT or port > _MAX_PORT:
        raise ValueError("LOCAL_AI_JCL_PORT must be from 1 through 65535. Example: 8770")
    return port


def _serve(application: Flask, settings: BindSettings) -> None:
    """
    Bind one request at a time. The slot thread is the other user of the sqlite connection.

    :param application: flask app as Flask. Example: the app from prepare
    :param settings: bind address as BindSettings. Example: BindSettings("127.0.0.1", 8770)
    :return: None. Example: None
    """
    logger.info("jcl.main | bind | host=%s port=%s", settings.host, settings.port)
    application.run(
        host=settings.host,
        port=settings.port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )
