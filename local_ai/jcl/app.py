"""Flask job-control API. Each route calls LocalAiServer and returns its dict."""

import logging
from collections.abc import Callable, Mapping

from flask import Flask, Response, current_app, jsonify, request

from local_ai.contracts import InvalidAccess, InvalidModel, InvalidPriority
from local_ai.dispatcher.server import LocalAiServer
from local_ai.jcl.tokens import PriorityForbidden, TokenTable, priority_within_ceiling


logger = logging.getLogger(__name__)

_HTTP_OK = 200
_HTTP_BAD = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_STATUS_REJECTED = "rejected"
_REASON_UNAUTHORIZED = "unauthorized"
_REASON_PRIORITY_FORBIDDEN = "priority_forbidden"
_REASON_BAD_JSON = "bad_json"
_REASON_BAD_REQUEST = "bad_request"
_REASON_BAD_PRIORITY = "bad_priority"
_BEARER = "Bearer "
_CONFIG_SERVER = "JCL_SERVER"
_CONFIG_TOKENS = "JCL_TOKENS"


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


def create_app(server: LocalAiServer, tokens: TokenTable) -> Flask:
    """
    Build the token-gated job-control app.

    :param server: dispatcher surface as LocalAiServer. Example: the composed server
    :param tokens: startup secrets as TokenTable. Example: load_token_table(env)
    :return: app as Flask. Example: an app whose routes are POST /jcl/submit and the other six tools
    :raises TypeError: when server or tokens has the wrong type. Example: None
    """
    _require_server(server)
    _require_tokens(tokens)
    application = Flask(__name__)
    application.config[_CONFIG_SERVER] = server
    application.config[_CONFIG_TOKENS] = tokens
    _register_routes(application)
    logger.info("create_app | exit | routes=capacity,submit,read,reprioritize,list,cancel,purge")
    return application


def _require_server(server: object) -> None:
    """
    Reject a stand-in that is not the dispatcher surface.

    :param server: candidate as object. Example: LocalAiServer(...)
    :return: None. Example: None
    :raises TypeError: when server is not a LocalAiServer. Example: None
    """
    if not isinstance(server, LocalAiServer):
        raise TypeError("server must be a LocalAiServer. Example: LocalAiServer(...)")


def _require_tokens(tokens: object) -> None:
    """
    Reject a token table that was not loaded by load_token_table.

    :param tokens: candidate as object. Example: TokenTable((("lane-50-secret", 50),))
    :return: None. Example: None
    :raises TypeError: when tokens is not a TokenTable. Example: {}
    """
    if not isinstance(tokens, TokenTable):
        raise TypeError("tokens must be a TokenTable. Example: TokenTable((('lane-50-secret', 50),))")


def _register_routes(application: Flask) -> None:
    """
    Attach the seven POST routes and the rejection handler.

    :param application: flask app as Flask. Example: Flask(__name__)
    :return: None. Example: None
    """
    application.post("/jcl/capacity")(_capacity)
    application.post("/jcl/submit")(_submit)
    application.post("/jcl/read")(_read)
    application.post("/jcl/reprioritize")(_reprioritize)
    application.post("/jcl/list")(_list_jobs)
    application.post("/jcl/cancel")(_cancel)
    application.post("/jcl/purge")(_purge)
    application.register_error_handler(_HttpReject, _reject_response)


def _capacity() -> tuple[Response, int]:
    """
    Return the capacity tool body.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok"}, 200)
    :raises _HttpReject: when the bearer token is missing or unknown. Example: no Authorization header
    """
    _authorize()
    logger.info("jcl.capacity | entry")
    body = _call(lambda: _server().capacity())
    logger.info("jcl.capacity | exit | status=%s", body.get("status"))
    return _ok(body)


def _submit() -> tuple[Response, int]:
    """
    Pin a task at a priority the token may set.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "queued", "priority": 50}, 200)
    :raises _HttpReject: when auth, JSON, or the priority fails. Example: priority 10 with a ceiling of 50
    """
    ceiling = _authorize()
    payload = _payload()
    priority = _optional_priority(ceiling, payload)
    logger.info("jcl.submit | entry | priority=%s", priority)
    body = _call(lambda: _submit_with(payload, priority))
    logger.info("jcl.submit | exit | priority=%s status=%s", priority, body.get("status"))
    return _ok(body)


def _read() -> tuple[Response, int]:
    """
    Return one pin.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "running"}, 200)
    :raises _HttpReject: when auth, JSON, or task_id fails. Example: {}
    """
    _authorize()
    payload = _payload()
    task_id = _text_field(payload, "task_id")
    logger.info("jcl.read | entry | task_id=%s", task_id)
    body = _call(lambda: _server().read(task_id))
    logger.info("jcl.read | exit | status=%s", body.get("status"))
    return _ok(body)


def _reprioritize() -> tuple[Response, int]:
    """
    Change the priority of a queued pin when the token allows it.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "queued", "priority": 90}, 200)
    :raises _HttpReject: when auth or the new priority fails. Example: priority 10 with a ceiling of 50
    """
    ceiling = _authorize()
    payload = _payload()
    priority = _required_priority(ceiling, payload)
    task_id = _text_field(payload, "task_id")
    logger.info("jcl.reprioritize | entry | task_id=%s priority=%s", task_id, priority)
    body = _call(lambda: _server().reprioritize(task_id, priority))
    logger.info("jcl.reprioritize | exit | status=%s", body.get("status"))
    return _ok(body)


def _list_jobs() -> tuple[Response, int]:
    """
    Return compact rows.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok", "count": 0}, 200)
    :raises _HttpReject: when auth or JSON fails. Example: no Authorization header
    """
    _authorize()
    payload = _payload()
    status = _optional_text(payload, "status")
    logger.info("jcl.list | entry | filter=%s", status)
    body = _call(lambda: _server().list_jobs(status))
    logger.info("jcl.list | exit | status=%s", body.get("status"))
    return _ok(body)


def _cancel() -> tuple[Response, int]:
    """
    Delete one queued pin.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "cancelled"}, 200)
    :raises _HttpReject: when auth, JSON, or task_id fails. Example: {}
    """
    _authorize()
    payload = _payload()
    task_id = _text_field(payload, "task_id")
    logger.info("jcl.cancel | entry | task_id=%s", task_id)
    body = _call(lambda: _server().cancel(task_id))
    logger.info("jcl.cancel | exit | status=%s", body.get("status"))
    return _ok(body)


def _purge() -> tuple[Response, int]:
    """
    Delete finished pins.

    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "ok", "count": 0}, 200)
    :raises _HttpReject: when auth or JSON fails. Example: no Authorization header
    """
    _authorize()
    payload = _payload()
    task_id = _optional_text(payload, "task_id")
    status = _optional_text(payload, "status")
    logger.info("jcl.purge | entry | task_id=%s filter=%s", task_id, status)
    body = _call(lambda: _server().purge(task_id, status))
    logger.info("jcl.purge | exit | status=%s", body.get("status"))
    return _ok(body)


def _submit_with(payload: Mapping[str, object], priority: int) -> dict:
    """
    Call LocalAiServer.submit with the JSON fields.

    :param payload: request object as Mapping[str, object]. Example: {"goal": "Write a brief.", "done_when": "brief.md exists"}
    :param priority: ceiling-checked priority as int. Example: 50
    :return: queued or rejected body as dict. Example: {"status": "queued", "priority": 50}
    :raises _HttpReject: when goal or done_when is missing. Example: {}
    """
    return _server().submit(
        _text_field(payload, "goal"),
        _text_field(payload, "done_when"),
        priority,
        _optional_str(payload, "model"),
        _optional_str_list(payload, "inputs"),
        _optional_dict(payload, "access"),
        _optional_str_list(payload, "plan"),
    )


def _authorize() -> int:
    """
    Resolve the bearer token to its priority ceiling.

    The secret is not logged. A missing token and an unknown token share one reason.

    :return: ceiling as int. Example: 50
    :raises _HttpReject: when the token is missing or unknown. Example: ""
    """
    ceiling = _tokens().ceiling(_bearer_token())
    if ceiling is None:
        logger.info("jcl.auth | rejected | reason=%s route=%s", _REASON_UNAUTHORIZED, request.path)
        raise _HttpReject(_HTTP_UNAUTHORIZED, _REASON_UNAUTHORIZED)
    logger.info("jcl.auth | ok | route=%s ceiling=%s", request.path, ceiling)
    return ceiling


def _bearer_token() -> str:
    """
    Read the bearer secret. A missing header becomes an empty string.

    :return: secret as str. Example: "lane-50-secret"
    """
    header = request.headers.get("Authorization", "")
    if not isinstance(header, str) or not header.startswith(_BEARER):
        return ""
    return header[len(_BEARER) :].strip()


def _payload() -> dict[str, object]:
    """
    Read a JSON object. An empty body is an empty object.

    :return: object as dict[str, object]. Example: {"goal": "Write a brief."}
    :raises _HttpReject: when the body is not a JSON object. Example: "[]"
    """
    if request.get_data(cache=True) == b"":
        return {}
    parsed = request.get_json(silent=True)
    if not isinstance(parsed, dict):
        logger.info("jcl.body | rejected | reason=%s route=%s", _REASON_BAD_JSON, request.path)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_JSON)
    return parsed


def _optional_priority(ceiling: int, payload: Mapping[str, object]) -> int:
    """
    Use the token ceiling when the caller omits priority.

    :param ceiling: best priority as int. Example: 50
    :param payload: request object as Mapping[str, object]. Example: {}
    :return: priority as int. Example: 50
    :raises _HttpReject: when the priority is illegal for this token. Example: 10
    """
    if "priority" not in payload:
        return _checked_priority(ceiling, None)
    return _checked_priority(ceiling, payload["priority"])


def _required_priority(ceiling: int, payload: Mapping[str, object]) -> int:
    """
    Require a priority field, then apply the ceiling.

    :param ceiling: best priority as int. Example: 50
    :param payload: request object as Mapping[str, object]. Example: {"priority": 90}
    :return: priority as int. Example: 90
    :raises _HttpReject: when priority is missing or better than the ceiling. Example: {}
    """
    if "priority" not in payload:
        logger.info("jcl.priority | rejected | reason=%s", _REASON_BAD_PRIORITY)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_PRIORITY)
    return _checked_priority(ceiling, payload["priority"])


def _checked_priority(ceiling: int, requested: object) -> int:
    """
    Accept the ceiling or a worse priority.

    :param ceiling: best priority as int. Example: 50
    :param requested: caller priority as object. Example: 80
    :return: priority as int. Example: 80
    :raises _HttpReject: when the value is not an int, is outside 1..100, or is better than the ceiling. Example: 10
    """
    value = _priority_int_or_none(requested)
    try:
        return priority_within_ceiling(ceiling, value)
    except PriorityForbidden:
        logger.info(
            "jcl.priority | rejected | reason=%s ceiling=%s requested=%s",
            _REASON_PRIORITY_FORBIDDEN,
            ceiling,
            value,
        )
        raise _HttpReject(_HTTP_FORBIDDEN, _REASON_PRIORITY_FORBIDDEN)
    except InvalidPriority:
        logger.info("jcl.priority | rejected | reason=%s", _REASON_BAD_PRIORITY)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_PRIORITY)


def _priority_int_or_none(requested: object) -> int | None:
    """
    Accept an omitted priority or a real int. True is not a priority.

    :param requested: caller priority as object. Example: 80
    :return: priority as int | None. Example: 80
    :raises _HttpReject: when requested is not an int or None. Example: "50"
    """
    if requested is None:
        return None
    if type(requested) is not int:
        logger.info("jcl.priority | rejected | reason=%s", _REASON_BAD_PRIORITY)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_PRIORITY)
    return requested


def _text_field(payload: Mapping[str, object], name: str) -> str:
    """
    Read one required string field.

    :param payload: request object as Mapping[str, object]. Example: {"task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    :param name: field name as str. Example: "task_id"
    :return: field value as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :raises _HttpReject: when the field is missing or not a string. Example: {}
    """
    if name not in payload or not isinstance(payload[name], str):
        logger.info("jcl.body | rejected | reason=%s field=%s", _REASON_BAD_REQUEST, name)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return payload[name]


def _optional_text(payload: Mapping[str, object], name: str) -> str | None:
    """
    Read one optional string field.

    :param payload: request object as Mapping[str, object]. Example: {"status": "completed"}
    :param name: field name as str. Example: "status"
    :return: field value as str | None. Example: "completed"
    :raises _HttpReject: when the field is present and not a string or null. Example: {"status": 1}
    """
    if name not in payload or payload[name] is None:
        return None
    if not isinstance(payload[name], str):
        logger.info("jcl.body | rejected | reason=%s field=%s", _REASON_BAD_REQUEST, name)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)
    return payload[name]


def _optional_str(payload: Mapping[str, object], name: str) -> str | None:
    """
    Pass an optional string through. The server rejects a non-string.

    :param payload: request object as Mapping[str, object]. Example: {"model": "qwen-14b"}
    :param name: field name as str. Example: "model"
    :return: value as str | None. Example: "qwen-14b"
    """
    value = _optional_value(payload, name)
    if value is None:
        return None
    return value if isinstance(value, str) else _bad_field(name)


def _optional_dict(payload: Mapping[str, object], name: str) -> dict | None:
    """
    Pass an optional object through.

    :param payload: request object as Mapping[str, object]. Example: {"access": {"mode": "job"}}
    :param name: field name as str. Example: "access"
    :return: value as dict | None. Example: {"mode": "job"}
    """
    value = _optional_value(payload, name)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return _bad_field(name)


def _optional_str_list(payload: Mapping[str, object], name: str) -> list[str] | None:
    """
    Pass an optional list of strings through.

    :param payload: request object as Mapping[str, object]. Example: {"plan": ["Write marker.txt"]}
    :param name: field name as str. Example: "plan"
    :return: value as list[str] | None. Example: ["Write marker.txt"]
    """
    value = _optional_value(payload, name)
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return _bad_field(name)


def _optional_value(payload: Mapping[str, object], name: str) -> object:
    """
    Return a present field, or None when it is omitted or null.

    :param payload: request object as Mapping[str, object]. Example: {"model": "qwen-14b"}
    :param name: field name as str. Example: "model"
    :return: value as object. Example: "qwen-14b"
    """
    if name not in payload:
        return None
    return payload[name]


def _bad_field(name: str) -> None:
    """
    Reject a field whose JSON type is wrong.

    :param name: field name as str. Example: "inputs"
    :return: None. Example: None
    :raises _HttpReject: always. Example: inputs as a string
    """
    logger.info("jcl.body | rejected | reason=%s field=%s", _REASON_BAD_REQUEST, name)
    raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST)


def _call(action: Callable[[], dict]) -> dict:
    """
    Run a dispatcher call. A client mistake becomes bad_request.

    Domain rejections are ordinary dicts and stay HTTP 200.

    :param action: zero-arg call as Callable[[], dict]. Example: lambda: server.capacity()
    :return: tool body as dict. Example: {"status": "ok"}
    :raises _HttpReject: when the call raises a validation error. Example: goal ""
    """
    try:
        return action()
    except (ValueError, TypeError, InvalidPriority, InvalidModel, InvalidAccess, KeyError) as exc:
        logger.info("jcl.call | rejected | reason=%s error=%s", _REASON_BAD_REQUEST, type(exc).__name__)
        raise _HttpReject(_HTTP_BAD, _REASON_BAD_REQUEST) from exc


def _server() -> LocalAiServer:
    """
    The dispatcher surface stored on the app.

    :return: server as LocalAiServer. Example: the composed server
    """
    server = current_app.config[_CONFIG_SERVER]
    if not isinstance(server, LocalAiServer):
        raise TypeError("server must be a LocalAiServer. Example: LocalAiServer(...)")
    return server


def _tokens() -> TokenTable:
    """
    The token table stored on the app.

    :return: table as TokenTable. Example: TokenTable((("lane-50-secret", 50),))
    """
    tokens = current_app.config[_CONFIG_TOKENS]
    if not isinstance(tokens, TokenTable):
        raise TypeError("tokens must be a TokenTable. Example: TokenTable((('lane-50-secret', 50),))")
    return tokens


def _ok(body: dict) -> tuple[Response, int]:
    """
    A domain result. Rejected tool bodies stay HTTP 200.

    :param body: tool body as dict. Example: {"status": "queued"}
    :return: JSON and HTTP 200 as tuple[Response, int]. Example: ({"status": "queued"}, 200)
    """
    return jsonify(body), _HTTP_OK


def _reject_response(error: _HttpReject) -> tuple[Response, int]:
    """
    Render a transport rejection. The body matches the tool rejection shape.

    :param error: rejection as _HttpReject. Example: _HttpReject(401, "unauthorized")
    :return: JSON and status as tuple[Response, int]. Example: ({"status": "rejected", "reason": "unauthorized"}, 401)
    """
    return jsonify({"status": _STATUS_REJECTED, "reason": error.reason}), error.status_code
