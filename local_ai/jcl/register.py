"""Publish this JCL host to a catalog. A failed post does not stop the API."""

import json
import logging
import socket
import threading
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from local_ai.dispatcher.server import LocalAiServer
from local_ai.jcl.tokens import TokenTable


logger = logging.getLogger(__name__)

CONFIG_REGISTER = "JCL_REGISTER"
_ENV_CATALOG_URL = "LOCAL_AI_CATALOG_URL"
_ENV_ADVERTISE_URL = "LOCAL_AI_ADVERTISE_URL"
_ENV_HOST_NAME = "LOCAL_AI_HOST_NAME"
_ENV_REFRESH = "LOCAL_AI_CATALOG_REFRESH_S"
_DEFAULT_REFRESH_S = 60
_MIN_REFRESH_S = 1
_POST_TIMEOUT_S = 5.0
_DEFAULT_NAME = "local-ai"
_REGISTER_PATH = "/register"


class Poster(Protocol):
    """Posts one JSON body to a URL."""

    def __call__(self, url: str, body: dict) -> None:
        """
        Perform one POST.

        :param url: register URL as str. Example: "http://192.168.1.40:8771/register"
        :param body: register body as dict. Example: {"name": "studio-mac"}
        :return: None. Example: None
        """


class RegisterConfigError(ValueError):
    """Raised when catalog registration settings cannot be loaded."""


@dataclass(frozen=True)
class RegisterSettings:
    """
    Where this host publishes itself. catalog_url None means do not register.

    :param catalog_url: catalog base URL as str | None. Example: "http://192.168.1.40:8771"
    :param advertise_url: URL other machines curl as str. Example: "http://192.168.1.20:8770"
    :param name: host name as str. Example: "studio-mac"
    :param refresh_s: seconds between posts as int. Example: 60
    """

    catalog_url: str | None
    advertise_url: str
    name: str
    refresh_s: int


def register_settings(env: Mapping[str, str]) -> RegisterSettings:
    """
    Read the optional catalog target. A catalog URL requires an advertise URL.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_CATALOG_URL": "http://192.168.1.40:8771"}
    :return: settings as RegisterSettings. Example: RegisterSettings(None, "", "studio-mac", 60)
    :raises TypeError: when env is not a mapping. Example: []
    :raises RegisterConfigError: when the catalog URL is set and the advertise URL is missing. Example: {"LOCAL_AI_CATALOG_URL": "http://192.168.1.40:8771"}
    """
    _require_mapping(env)
    catalog_url = _optional_http_url(env.get(_ENV_CATALOG_URL), _ENV_CATALOG_URL)
    if catalog_url is None:
        settings = _idle_settings(env)
        logger.info("register_settings | skip | reason=no_catalog_url")
        return settings
    advertise_url = _required_advertise_url(env.get(_ENV_ADVERTISE_URL))
    settings = RegisterSettings(
        catalog_url,
        advertise_url,
        _host_name(env.get(_ENV_HOST_NAME)),
        _refresh(env.get(_ENV_REFRESH)),
    )
    logger.info(
        "register_settings | exit | advertise_url=%s refresh_s=%s",
        advertise_url,
        settings.refresh_s,
    )
    return settings


def registration_body(name: str, advertise_url: str, tokens: TokenTable, capacity: dict) -> dict:
    """
    Build the catalog register body from the token table and one capacity reading.

    :param name: host name as str. Example: "studio-mac"
    :param advertise_url: URL other machines curl as str. Example: "http://192.168.1.20:8770"
    :param tokens: startup secrets as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :param capacity: capacity tool body as dict. Example: {"status": "ok", "models": ["qwen-14b"]}
    :return: register body as dict. Example: {"name": "studio-mac", "models": ["qwen-14b"]}
    :raises TypeError: when tokens is not a TokenTable or capacity is not a dict. Example: []
    """
    _require_tokens(tokens)
    reading = _require_capacity(capacity)
    models = _models(reading)
    logger.info("registration_body | advertise_url=%s models=%s", advertise_url, len(models))
    return {
        "name": name,
        "advertise_url": advertise_url,
        "tokens": list(tokens.registration_rows()),
        "capacity": reading,
        "models": models,
    }


def post_registration(catalog_url: str, body: dict, poster: Poster) -> None:
    """
    POST the body to the catalog register route.

    :param catalog_url: catalog base URL as str. Example: "http://192.168.1.40:8771"
    :param body: register body as dict. Example: {"name": "studio-mac"}
    :param poster: performs the POST as Poster. Example: urllib_poster
    :return: None. Example: None
    :raises ValueError: when catalog_url is blank. Example: ""
    :raises TypeError: when poster is not callable. Example: None
    """
    if not isinstance(catalog_url, str) or not catalog_url.strip():
        raise ValueError("catalog_url must be a non-empty URL. Example: 'http://192.168.1.40:8771'")
    if not callable(poster):
        raise TypeError("poster must be callable. Example: urllib_poster")
    poster(_register_url(catalog_url), body)


def register_once(
    server: LocalAiServer,
    tokens: TokenTable,
    settings: RegisterSettings,
    poster: Poster,
) -> None:
    """
    Read capacity and post one registration. A failure is logged and swallowed.

    :param server: dispatcher surface as LocalAiServer. Example: the composed server
    :param tokens: startup secrets as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :param settings: catalog target as RegisterSettings. Example: RegisterSettings("http://192.168.1.40:8771", "http://192.168.1.20:8770", "studio-mac", 60)
    :param poster: performs the POST as Poster. Example: urllib_poster
    :return: None. Example: None
    """
    if settings.catalog_url is None:
        logger.info("jcl.register | skip | reason=no_catalog_url")
        return
    try:
        _post_current(server, tokens, settings, poster)
    except Exception as exc:
        logger.error(
            "jcl.register | failed | advertise_url=%s error=%s",
            settings.advertise_url,
            type(exc).__name__,
        )


def start_refresh(
    server: LocalAiServer,
    tokens: TokenTable,
    settings: RegisterSettings,
    poster: Poster,
    stop: threading.Event | None = None,
) -> threading.Thread | None:
    """
    Start the daemon that repeats registration. No catalog URL starts nothing.

    :param server: dispatcher surface as LocalAiServer. Example: the composed server
    :param tokens: startup secrets as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :param settings: catalog target as RegisterSettings. Example: RegisterSettings(None, "", "studio-mac", 60)
    :param poster: performs the POST as Poster. Example: urllib_poster
    :param stop: set to end the loop as threading.Event | None. Example: threading.Event()
    :return: thread as threading.Thread | None. Example: None
    """
    if settings.catalog_url is None:
        logger.info("jcl.register | skip | reason=no_catalog_url")
        return None
    event = threading.Event() if stop is None else stop
    thread = threading.Thread(
        target=_refresh_loop,
        args=(server, tokens, settings, poster, event),
        daemon=True,
        name="jcl-catalog",
    )
    thread.start()
    logger.info(
        "jcl.register | thread | advertise_url=%s refresh_s=%s",
        settings.advertise_url,
        settings.refresh_s,
    )
    return thread


def urllib_poster(url: str, body: dict) -> None:
    """
    POST JSON with the standard library. HTTP errors raise.

    :param url: register URL as str. Example: "http://192.168.1.40:8771/register"
    :param body: register body as dict. Example: {"name": "studio-mac"}
    :return: None. Example: None
    :raises urllib.error.URLError: when the catalog is unreachable. Example: connection refused
    """
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_POST_TIMEOUT_S) as response:
        logger.info("urllib_poster | status=%s", getattr(response, "status", 200))


def _idle_settings(env: Mapping[str, str]) -> RegisterSettings:
    """
    Settings that do not register.

    :param env: environment as Mapping[str, str]. Example: {}
    :return: settings as RegisterSettings. Example: RegisterSettings(None, "", "studio-mac", 60)
    """
    return RegisterSettings(None, "", _host_name(env.get(_ENV_HOST_NAME)), _refresh(env.get(_ENV_REFRESH)))


def _post_current(
    server: LocalAiServer,
    tokens: TokenTable,
    settings: RegisterSettings,
    poster: Poster,
) -> None:
    """
    One successful post. The caller logs failures.

    :param server: dispatcher surface as LocalAiServer. Example: the composed server
    :param tokens: startup secrets as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :param settings: catalog target as RegisterSettings. Example: RegisterSettings("http://192.168.1.40:8771", "http://192.168.1.20:8770", "studio-mac", 60)
    :param poster: performs the POST as Poster. Example: urllib_poster
    :return: None. Example: None
    """
    if settings.catalog_url is None:
        return
    body = registration_body(settings.name, settings.advertise_url, tokens, server.capacity())
    post_registration(settings.catalog_url, body, poster)
    logger.info("jcl.register | ok | advertise_url=%s", settings.advertise_url)


def _refresh_loop(
    server: LocalAiServer,
    tokens: TokenTable,
    settings: RegisterSettings,
    poster: Poster,
    stop: threading.Event,
) -> None:
    """
    Post, then wait. The wait returns early when stop is set.

    :param server: dispatcher surface as LocalAiServer. Example: the composed server
    :param tokens: startup secrets as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    :param settings: catalog target as RegisterSettings. Example: RegisterSettings("http://192.168.1.40:8771", "http://192.168.1.20:8770", "studio-mac", 60)
    :param poster: performs the POST as Poster. Example: urllib_poster
    :param stop: set to end the loop as threading.Event. Example: threading.Event()
    :return: None. Example: None
    """
    while not stop.is_set():
        register_once(server, tokens, settings, poster)
        stop.wait(settings.refresh_s)


def _require_mapping(env: object) -> None:
    """
    Reject anything that is not a mapping.

    :param env: candidate as object. Example: {}
    :return: None. Example: None
    :raises TypeError: when env is not a mapping. Example: []
    """
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_CATALOG_URL': 'http://192.168.1.40:8771'}")


def _optional_http_url(raw: object, name: str) -> str | None:
    """
    A missing value means unset. A present value must be http or https.

    :param raw: environment value as object. Example: "http://192.168.1.40:8771"
    :param name: variable name as str. Example: "LOCAL_AI_CATALOG_URL"
    :return: URL as str | None. Example: "http://192.168.1.40:8771"
    :raises RegisterConfigError: when the value is present and not an http URL. Example: "ftp://pi"
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if not isinstance(raw, str) or not raw.strip().startswith(("http://", "https://")):
        raise RegisterConfigError(f"{name} must be an http URL. Example: 'http://192.168.1.40:8771'")
    return raw.strip().rstrip("/")


def _required_advertise_url(raw: object) -> str:
    """
    The URL other machines use. Required once a catalog URL is set.

    :param raw: environment value as object. Example: "http://192.168.1.20:8770"
    :return: URL as str. Example: "http://192.168.1.20:8770"
    :raises RegisterConfigError: when the URL is missing or not http. Example: None
    """
    url = _optional_http_url(raw, _ENV_ADVERTISE_URL)
    if url is None:
        raise RegisterConfigError(
            "LOCAL_AI_ADVERTISE_URL is required when LOCAL_AI_CATALOG_URL is set. "
            "Example: 'http://192.168.1.20:8770'"
        )
    return url


def _host_name(raw: object) -> str:
    """
    Host label. A missing value is the machine hostname.

    :param raw: environment value as object. Example: "studio-mac"
    :return: name as str. Example: "studio-mac"
    """
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    hostname = socket.gethostname().strip()
    if hostname:
        return hostname
    return _DEFAULT_NAME


def _refresh(raw: object) -> int:
    """
    Seconds between posts. A missing value is 60.

    :param raw: environment value as object. Example: "60"
    :return: seconds as int. Example: 60
    :raises RegisterConfigError: when the value is not an integer of at least 1. Example: "0"
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _DEFAULT_REFRESH_S
    if not isinstance(raw, str) or not raw.strip().isdigit():
        raise RegisterConfigError("LOCAL_AI_CATALOG_REFRESH_S must be an integer. Example: 60")
    seconds = int(raw.strip())
    if seconds < _MIN_REFRESH_S:
        raise RegisterConfigError("LOCAL_AI_CATALOG_REFRESH_S must be at least 1. Example: 60")
    return seconds


def _require_tokens(tokens: object) -> None:
    """
    Reject a token table that was not loaded by load_token_table.

    :param tokens: candidate as object. Example: TokenTable((("lane-50-secret", 50),))
    :return: None. Example: None
    :raises TypeError: when tokens is not a TokenTable. Example: {}
    """
    if not isinstance(tokens, TokenTable):
        raise TypeError("tokens must be a TokenTable. Example: TokenTable((('lane-50-secret', 50),))")


def _require_capacity(capacity: object) -> dict:
    """
    Reject a capacity reading that is not an object.

    :param capacity: candidate as object. Example: {"status": "ok"}
    :return: reading as dict. Example: {"status": "ok"}
    :raises TypeError: when capacity is not a dict. Example: []
    """
    if not isinstance(capacity, dict):
        raise TypeError("capacity must be a dict. Example: {'status': 'ok'}")
    return capacity


def _models(capacity: dict) -> list[str]:
    """
    Logical model names from the capacity reading. A missing list is empty.

    :param capacity: capacity tool body as dict. Example: {"models": ["qwen-14b"]}
    :return: names as list[str]. Example: ["qwen-14b"]
    """
    raw = capacity.get("models")
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _register_url(catalog_url: str) -> str:
    """
    The register route under the catalog base URL.

    :param catalog_url: catalog base URL as str. Example: "http://192.168.1.40:8771"
    :return: URL as str. Example: "http://192.168.1.40:8771/register"
    """
    return catalog_url.rstrip("/") + _REGISTER_PATH
