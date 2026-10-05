"""Read-only Flask page for the dispatcher blackboard.

The page calls DispatcherService.list_jobs and DispatcherService.read_pin.
An unknown status query is rejected with reason bad_status and HTTP 400.
The access grant is left off the HTML. The process binds 127.0.0.1 only.
"""

import html
import logging
import os
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, current_app, request

from local_ai.blackboard.pins import Blackboard
from local_ai.capacity.reader import CapacityReader
from local_ai.db.database import Database
from local_ai.dispatcher.service import DispatcherService
from local_ai.plan.steps import PlanStore


logger = logging.getLogger(__name__)

_LOOPBACK_HOST = "127.0.0.1"
_ENV_PORT = "LOCAL_AI_BOARD_PORT"
_DEFAULT_PORT = 8766
_MIN_PORT = 1
_MAX_PORT = 65535
_ENV_DB = "LOCAL_AI_DB"
_DEFAULT_DB = Path("data/local-ai.sqlite")
_LOG_PATH = Path("logs/app.log")
_UNUSED_JOBS_ROOT = Path("data/jobs")
_FILTER_ALL = "all"
_STATUS_FILTERS = ("queued", "running", "completed", "failed")
_OPEN_STATUSES = frozenset({"queued", "running"})
_STATUS_REJECTED = "rejected"
_REASON_BAD_STATUS = "bad_status"
_REFRESH_SECONDS = 5
_EMPTY_MESSAGE = "No jobs on the blackboard."
_UNREADABLE_MESSAGE = "The blackboard database could not be read. This page does not migrate."
_NAV = (
    ("all", "/"),
    ("queued", "/?status=queued"),
    ("running", "/?status=running"),
    ("completed", "/?status=completed"),
    ("failed", "/?status=failed"),
)
_LOG_ATTACHED = False
_CSS = """
body { font: 16px/1.45 system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #1c2128; }
header, main { max-width: 52rem; margin: 0 auto; padding: 1.25rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
nav a { margin-right: 0.75rem; }
nav a[aria-current="page"] { font-weight: 700; }
section { background: #fff; border: 1px solid #d0d7de; border-left: 4px solid #6e7781;
  border-radius: 6px; margin: 0.75rem 0; padding: 0.5rem 0.9rem; }
section[data-status="running"] { border-left-color: #1a7f37; }
section[data-status="queued"] { border-left-color: #9a6700; }
section[data-status="completed"] { border-left-color: #0969da; }
section[data-status="failed"] { border-left-color: #cf222e; }
summary { cursor: pointer; }
pre, li, summary { white-space: pre-wrap; word-break: break-word; }
.meta { color: #57606a; }
"""


@dataclass(frozen=True)
class _BoardJob:
    """
    One list row plus the pin read_pin returned.

    :param summary: compact row as dict. Example: {"task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f", "status": "queued"}
    :param pin: full pin as dict. Example: {"status": "running", "current_plan": "1. List chapter files (running)."}
    """

    summary: dict[str, object]
    pin: dict[str, object]


@dataclass(frozen=True)
class _BoardView:
    """
    What the page renders.

    :param status_filter: active filter as str. Example: "all"
    :param jobs: rows as tuple[_BoardJob, ...]. Example: ()
    :param error: rejection text as str. Example: ""
    :param status_code: HTTP status as int. Example: 200
    """

    status_filter: str
    jobs: tuple[_BoardJob, ...]
    error: str
    status_code: int


def create_app(db_path: Path | None = None) -> Flask:
    """
    Build the read-only blackboard app.

    :param db_path: sqlite file as Path | None. Example: Path("data/local-ai.sqlite")
    :return: app as Flask. Example: an app whose only route is GET /
    :raises TypeError: when db_path is not a Path or None. Example: "data/local-ai.sqlite"
    """
    _require_optional_path(db_path)
    application = Flask(__name__)
    application.config["BOARD_DB"] = _db_from_env(os.environ) if db_path is None else db_path
    application.get("/")(_index)
    return application


def render_board(db_path: Path, status: str | None) -> tuple[str, int]:
    """
    Render the blackboard HTML for one status filter.

    Omitted status and all show every pin. Any other unknown value is HTTP 400
    with reason bad_status, the same rejection list returns. The database is not written.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :param status: filter as str | None. Example: "queued"
    :return: HTML and HTTP status as tuple[str, int]. Example: ("<!DOCTYPE html>...", 200)
    :raises TypeError: when db_path is not a Path or status is not a string. Example: 1
    """
    _require_path(db_path)
    _require_status(status)
    view = _load_view(db_path, status)
    return _render_page(view), view.status_code


def bind_settings(env: Mapping[str, str] | None = None) -> tuple[str, int]:
    """
    Loopback host and the board port. The host is not taken from the environment.

    :param env: environment as Mapping[str, str] | None. Example: {"LOCAL_AI_BOARD_PORT": "8766"}
    :return: host and port as tuple[str, int]. Example: ("127.0.0.1", 8766)
    :raises TypeError: when env is not a mapping. Example: []
    :raises ValueError: when the port is outside 1..65535. Example: "0"
    """
    source = os.environ if env is None else env
    _require_env(source)
    return _LOOPBACK_HOST, _port(source.get(_ENV_PORT))


def main() -> None:
    """
    Serve the blackboard on loopback.

    :return: None. Example: None
    :raises ValueError: when LOCAL_AI_BOARD_PORT is not a port. Example: "0"
    """
    _attach_log_file(_LOG_PATH)
    host, port = bind_settings(os.environ)
    logger.info("board.main | host=%s port=%s", host, port)
    create_app(_db_from_env(os.environ)).run(host=host, port=port, debug=False, use_reloader=False)


def _index() -> tuple[str, int]:
    """
    Render the blackboard for the status query parameter.

    :return: HTML and HTTP status as tuple[str, int]. Example: ("<!DOCTYPE html>...", 200)
    """
    status = request.args.get("status")
    logger.info("board.index | filter=%s", _filter_label(status))
    return render_board(current_app.config["BOARD_DB"], status)


def _load_view(db_path: Path, status: str | None) -> _BoardView:
    """
    Choose an empty page, a rejection, or the pins.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :param status: filter as str | None. Example: "queued"
    :return: view as _BoardView. Example: a view with status_code 200
    """
    label = _filter_label(status)
    if not db_path.is_file():
        return _view_for_missing_db(label, status)
    return _view_for_database(db_path, status, label)


def _view_for_missing_db(label: str, status: str | None) -> _BoardView:
    """
    Reject an unknown filter. A known filter on a missing file is an empty board.

    :param label: filter label as str. Example: "all"
    :param status: filter as str | None. Example: None
    :return: view as _BoardView. Example: a view with status_code 200
    """
    if not _known_filter(status):
        logger.info("board.render | rejected | reason=%s", _REASON_BAD_STATUS)
        return _rejected_view(status)
    logger.info("board.render | empty | reason=missing_db filter=%s", label)
    return _empty_view(label)


def _view_for_database(db_path: Path, status: str | None, label: str) -> _BoardView:
    """
    Open the file read-only and render list plus each pin.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :param status: filter as str | None. Example: "queued"
    :param label: filter label as str. Example: "queued"
    :return: view as _BoardView. Example: a view with status_code 200
    """
    logger.info("board.open | read_only | path=%s", db_path.name)
    try:
        connection = _open_readonly(db_path)
    except sqlite3.OperationalError:
        logger.info("board.render | unreadable | error=OperationalError")
        return _unreadable_view(label)
    try:
        return _view_from_connection(db_path, connection, status, label)
    finally:
        connection.close()


def _view_from_connection(
    db_path: Path,
    connection: sqlite3.Connection,
    status: str | None,
    label: str,
) -> _BoardView:
    """
    Ask list_jobs, then read_pin for each row. A rejection lists no jobs.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :param connection: read-only connection as sqlite3.Connection. Example: the board connection
    :param status: filter as str | None. Example: "done"
    :param label: filter label as str. Example: "rejected"
    :return: view as _BoardView. Example: a view with status_code 400
    """
    service = _service(db_path)
    try:
        body = service.list_jobs(status, connection)
    except sqlite3.OperationalError:
        logger.info("board.render | unreadable | error=OperationalError")
        return _unreadable_view(label)
    if body.get("status") == _STATUS_REJECTED:
        logger.info("board.render | rejected | reason=%s", body.get("reason"))
        return _rejected_view(status)
    jobs = _jobs_with_pins(service, connection, _job_rows(body))
    logger.info("board.render | ok | filter=%s count=%s", label, len(jobs))
    return _ok_view(label, jobs)


def _jobs_with_pins(
    service: DispatcherService,
    connection: sqlite3.Connection,
    rows: list[object],
) -> tuple[_BoardJob, ...]:
    """
    Attach a read_pin body to each list row.

    :param service: dispatcher as DispatcherService. Example: DispatcherService(...)
    :param connection: read-only connection as sqlite3.Connection. Example: the board connection
    :param rows: list rows as list[object]. Example: []
    :return: jobs as tuple[_BoardJob, ...]. Example: ()
    """
    return tuple(_job_with_pin(service, connection, row) for row in rows)


def _job_with_pin(
    service: DispatcherService,
    connection: sqlite3.Connection,
    row: object,
) -> _BoardJob:
    """
    Read one pin. A bad row becomes an unavailable plan and is not logged in full.

    :param service: dispatcher as DispatcherService. Example: DispatcherService(...)
    :param connection: read-only connection as sqlite3.Connection. Example: the board connection
    :param row: one list row as object. Example: {"task_id": "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"}
    :return: job as _BoardJob. Example: a job whose pin status is "running"
    """
    summary = dict(row) if isinstance(row, dict) else {}
    task_id = str(summary.get("task_id", ""))
    pin = _read_pin(service, connection, task_id)
    return _BoardJob(summary=summary, pin=pin)


def _read_pin(
    service: DispatcherService,
    connection: sqlite3.Connection,
    task_id: str,
) -> dict[str, object]:
    """
    Read one pin. Failures stay on this job and do not include the note text in the log.

    :param service: dispatcher as DispatcherService. Example: DispatcherService(...)
    :param connection: read-only connection as sqlite3.Connection. Example: the board connection
    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :return: pin body as dict[str, object]. Example: {"status": "running", "current_plan": ""}
    """
    try:
        pin = service.read_pin(task_id, connection)
    except (ValueError, sqlite3.OperationalError) as exc:
        logger.info("board.read | task_id=%s status=unreadable error=%s", task_id, type(exc).__name__)
        return {"status": _STATUS_REJECTED, "reason": "unreadable", "task_id": task_id}
    logger.info("board.read | task_id=%s status=%s", task_id, pin.get("status"))
    return pin


def _service(db_path: Path) -> DispatcherService:
    """
    Build a dispatcher that can list and read. This page never calls connect or migrate.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :return: service as DispatcherService. Example: DispatcherService(...)
    """
    database = Database(db_path)
    return DispatcherService(
        database,
        Blackboard(database),
        PlanStore(database),
        CapacityReader(),
        _UNUSED_JOBS_ROOT,
    )


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """
    Open sqlite in read-only mode. query_only blocks writes on this connection.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :return: connection as sqlite3.Connection. Example: a connection with row_factory Row
    :raises sqlite3.OperationalError: when the file cannot be opened. Example: a missing parent
    """
    connection = sqlite3.connect(_readonly_uri(db_path), uri=True, timeout=2.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _readonly_uri(db_path: Path) -> str:
    """
    Build a file URI that cannot create or write the database.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :return: URI as str. Example: "file:///tmp/local-ai.sqlite?mode=ro"
    """
    return f"{db_path.resolve().as_uri()}?mode=ro"


def _render_page(view: _BoardView) -> str:
    """
    Wrap the filter nav and the job list in one HTML document.

    :param view: page model as _BoardView. Example: a view with status_filter "all"
    :return: HTML as str. Example: "<!DOCTYPE html>..."
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="{_REFRESH_SECONDS}">\n'
        "<title>Local AI blackboard</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        "<header><h1>Blackboard</h1>\n"
        "<p>Read-only view of the same pins as list and read.</p>\n"
        f"{_render_nav(view.status_filter)}</header>\n"
        f"<main>{_render_body(view)}</main>\n</body></html>\n"
    )


def _render_nav(active: str) -> str:
    """
    Render the status links. The active filter is marked as the current page.

    :param active: filter label as str. Example: "all"
    :return: nav HTML as str. Example: '<nav aria-label="Status filter">...</nav>'
    """
    links = "".join(_nav_link(name, href, active) for name, href in _NAV)
    return f'<nav aria-label="Status filter">{links}</nav>'


def _nav_link(name: str, href: str, active: str) -> str:
    """
    Render one filter link. name and href come from _NAV, not from the query string.

    :param name: filter name as str. Example: "queued"
    :param href: link target as str. Example: "/?status=queued"
    :param active: current filter as str. Example: "all"
    :return: anchor HTML as str. Example: '<a href="/?status=queued" data-testid="filter-queued">queued</a>'
    """
    current = ' aria-current="page"' if name == active else ""
    return f'<a href="{href}" data-testid="filter-{name}"{current}>{name}</a>'


def _render_body(view: _BoardView) -> str:
    """
    Render the error, the empty board, or the job cards.

    :param view: page model as _BoardView. Example: a view with no jobs
    :return: main HTML as str. Example: '<p data-testid="board-empty">No jobs on the blackboard.</p>'
    """
    if view.error:
        return f'<p data-testid="board-error">{_esc(view.error)}</p>'
    if not view.jobs:
        return f'<p data-testid="board-empty">{_esc(_EMPTY_MESSAGE)}</p>'
    cards = "".join(_render_job(job) for job in view.jobs)
    return _render_progress(view.jobs) + cards


def _render_progress(jobs: tuple[_BoardJob, ...]) -> str:
    """
    Count running and queued pins so work in progress is visible above the list.

    :param jobs: rows as tuple[_BoardJob, ...]. Example: ()
    :return: paragraph HTML as str. Example: '<p data-testid="in-progress">In progress: 1 running, 0 queued.</p>'
    """
    running = sum(1 for job in jobs if job.summary.get("status") == "running")
    queued = sum(1 for job in jobs if job.summary.get("status") == "queued")
    return f'<p data-testid="in-progress">In progress: {running} running, {queued} queued.</p>'


def _render_job(job: _BoardJob) -> str:
    """
    Render one job. Running and queued cards start expanded.

    :param job: one job as _BoardJob. Example: a job with status "running"
    :return: section HTML as str. Example: '<section data-testid="job">...</section>'
    """
    summary = job.summary
    status = summary.get("status", "")
    return (
        f'<section data-testid="job" data-status="{_esc(status)}" '
        f'data-task-id="{_esc(summary.get("task_id"))}">'
        f"<details{_open_attr(status)}><summary>{_render_summary(summary)}</summary>"
        f"{_render_times(summary)}{_render_pin(job.pin)}</details></section>"
    )


def _render_summary(summary: dict[str, object]) -> str:
    """
    Render the fields that stay visible when the card is collapsed.

    :param summary: compact row as dict[str, object]. Example: {"status": "queued", "priority": 10}
    :return: summary HTML as str. Example: '<span data-testid="job-status">queued</span>'
    """
    parts = (
        _field("job-status", summary.get("status")),
        _field("job-priority", summary.get("priority")),
        _field("job-model", summary.get("model")),
        _field("job-phase", summary.get("phase")),
        _position_field(summary.get("position")),
        _field("job-goal", summary.get("goal")),
    )
    return " ".join(part for part in parts if part)


def _render_times(summary: dict[str, object]) -> str:
    """
    Render created, started, finished, and a non-empty error.

    :param summary: compact row as dict[str, object]. Example: {"created_at": "2026-10-04T18:40:00Z"}
    :return: paragraph HTML as str. Example: '<p class="meta">created ...</p>'
    """
    return (
        '<p class="meta">'
        f"created {_time_field('job-created', summary.get('created_at'))} "
        f"started {_time_field('job-started', summary.get('started_at'))} "
        f"finished {_time_field('job-finished', summary.get('finished_at'))}"
        f"{_error_field(summary.get('error'))}</p>"
    )


def _render_pin(pin: dict[str, object]) -> str:
    """
    Render the plan, steps, and reasoning. The access grant is omitted.

    :param pin: read body as dict[str, object]. Example: {"status": "running", "current_plan": ""}
    :return: pin HTML as str. Example: '<pre data-testid="job-plan"></pre>'
    """
    if pin.get("status") == _STATUS_REJECTED:
        return '<pre data-testid="job-plan">Plan unavailable.</pre>'
    return _render_plan(pin) + _render_steps(pin) + _render_notes(pin)


def _render_plan(pin: dict[str, object]) -> str:
    """
    Render current_plan.

    :param pin: read body as dict[str, object]. Example: {"current_plan": "1. List chapter files (running)."}
    :return: pre HTML as str. Example: '<pre data-testid="job-plan">1. List chapter files (running).</pre>'
    """
    return f'<pre data-testid="job-plan">{_esc(pin.get("current_plan", ""))}</pre>'


def _render_steps(pin: dict[str, object]) -> str:
    """
    Render ordered steps from read_pin.

    :param pin: read body as dict[str, object]. Example: {"steps": []}
    :return: list HTML as str. Example: '<ol data-testid="job-steps"></ol>'
    """
    steps = pin.get("steps")
    if not isinstance(steps, list) or not steps:
        return '<ol data-testid="job-steps"></ol>'
    items = "".join(_render_step(step) for step in steps)
    return f'<ol data-testid="job-steps">{items}</ol>'


def _render_step(step: object) -> str:
    """
    Render one step's order, action, and status.

    :param step: one step as object. Example: {"order": 1, "action": "List chapter files", "status": "running"}
    :return: item HTML as str. Example: '<li data-testid="job-step">1. List chapter files (running)</li>'
    """
    if not isinstance(step, dict):
        return "<li></li>"
    line = f"{_esc(step.get('order'))}. {_esc(step.get('action'))} ({_esc(step.get('status'))})"
    return f"<li data-testid=\"job-step\">{line}{_step_extra(step)}</li>"


def _step_extra(step: dict[str, object]) -> str:
    """
    Append a step error or synthesis when the pin has one.

    :param step: one step as dict[str, object]. Example: {"error": "", "llm_synthesis": ""}
    :return: extra HTML as str. Example: ""
    """
    error = step.get("error") or ""
    synthesis = step.get("llm_synthesis") or ""
    extra = f' <span class="error">{_esc(error)}</span>' if error else ""
    if synthesis:
        extra += f" — {_esc(synthesis)}"
    return extra


def _render_notes(pin: dict[str, object]) -> str:
    """
    Render reasoning notes, oldest first, as read_pin returns them.

    :param pin: read body as dict[str, object]. Example: {"reasoning": []}
    :return: list HTML as str. Example: '<ul data-testid="job-reasoning"></ul>'
    """
    notes = pin.get("reasoning")
    if not isinstance(notes, list) or not notes:
        return '<ul data-testid="job-reasoning"></ul>'
    items = "".join(_render_note(note) for note in notes)
    return f'<ul data-testid="job-reasoning">{items}</ul>'


def _render_note(note: object) -> str:
    """
    Render one reasoning note.

    :param note: one note as object. Example: {"at": "2026-10-04T18:40:01Z", "step": 1, "text": "Reading ch-04.md."}
    :return: item HTML as str. Example: "<li>2026-10-04T18:40:01Z step 1: Reading ch-04.md.</li>"
    """
    if not isinstance(note, dict):
        return "<li></li>"
    text = f"{_esc(note.get('at'))} step {_esc(note.get('step'))}: {_esc(note.get('text'))}"
    return f"<li>{text}</li>"


def _field(test_id: str, value: object) -> str:
    """
    Render one escaped field.

    :param test_id: data-testid value as str. Example: "job-status"
    :param value: field value as object. Example: "running"
    :return: span HTML as str. Example: '<span data-testid="job-status">running</span>'
    """
    return f'<span data-testid="{test_id}">{_esc(value)}</span>'


def _time_field(test_id: str, value: object) -> str:
    """
    Render a timestamp. A missing time is an em dash.

    :param test_id: data-testid value as str. Example: "job-started"
    :param value: timestamp as object. Example: None
    :return: span HTML as str. Example: '<span data-testid="job-started">—</span>'
    """
    shown = "—" if value is None or value == "" else value
    return _field(test_id, shown)


def _position_field(position: object) -> str:
    """
    Render queue position only for a queued pin.

    :param position: place as object. Example: 1
    :return: span HTML or empty as str. Example: '<span data-testid="job-position">1</span>'
    """
    if position is None:
        return ""
    return _field("job-position", position)


def _error_field(error: object) -> str:
    """
    Render a pin error when the row has one.

    :param error: failure summary as object. Example: ""
    :return: text and span, or empty, as str. Example: ""
    """
    if error is None or error == "":
        return ""
    return f" error {_field('job-error', error)}"


def _open_attr(status: object) -> str:
    """
    Expand running and queued cards. Finished cards stay collapsed.

    :param status: pin status as object. Example: "running"
    :return: attribute text as str. Example: " open"
    """
    if status in _OPEN_STATUSES:
        return " open"
    return ""


def _rejected_view(status: str | None) -> _BoardView:
    """
    Build the bad_status page. The raw filter is escaped later and is not logged.

    :param status: rejected filter as str | None. Example: "done"
    :return: view as _BoardView. Example: a view with status_code 400
    """
    shown = "" if status is None else status
    message = f"Unknown status filter '{shown}'. Rejected with reason {_REASON_BAD_STATUS}."
    return _BoardView("rejected", (), message, 400)


def _empty_view(label: str) -> _BoardView:
    """
    Build an empty board.

    :param label: filter label as str. Example: "all"
    :return: view as _BoardView. Example: a view with status_code 200
    """
    return _BoardView(label, (), "", 200)


def _ok_view(label: str, jobs: tuple[_BoardJob, ...]) -> _BoardView:
    """
    Build a page of jobs.

    :param label: filter label as str. Example: "queued"
    :param jobs: rows as tuple[_BoardJob, ...]. Example: ()
    :return: view as _BoardView. Example: a view with status_code 200
    """
    return _BoardView(label, jobs, "", 200)


def _unreadable_view(label: str) -> _BoardView:
    """
    Build the page shown when sqlite cannot be read. Nothing is migrated.

    :param label: filter label as str. Example: "all"
    :return: view as _BoardView. Example: a view with status_code 500
    """
    return _BoardView(label, (), _UNREADABLE_MESSAGE, 500)


def _job_rows(body: dict[str, object]) -> list[object]:
    """
    Return the list rows, or an empty list when the body has none.

    :param body: list response as dict[str, object]. Example: {"status": "ok", "jobs": []}
    :return: rows as list[object]. Example: []
    """
    rows = body.get("jobs")
    if not isinstance(rows, list):
        logger.info("board.render | empty | reason=missing_jobs")
        return []
    return rows


def _known_filter(status: str | None) -> bool:
    """
    True when list would accept the filter. Used when the database file is absent.

    :param status: filter as str | None. Example: "queued"
    :return: whether the filter is known as bool. Example: True
    """
    return status is None or status == _FILTER_ALL or status in _STATUS_FILTERS


def _filter_label(status: str | None) -> str:
    """
    Label a filter for logs. Unknown values become the word rejected, not the raw text.

    :param status: filter as str | None. Example: "done"
    :return: label as str. Example: "rejected"
    """
    if _known_filter(status):
        return _FILTER_ALL if status is None or status == _FILTER_ALL else str(status)
    return "rejected"


def _db_from_env(env: Mapping[str, str]) -> Path:
    """
    Read LOCAL_AI_DB. A missing or blank value keeps data/local-ai.sqlite.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_DB": "data/local-ai.sqlite"}
    :return: path as Path. Example: Path("data/local-ai.sqlite")
    """
    raw = env.get(_ENV_DB)
    if not isinstance(raw, str) or not raw.strip():
        return _DEFAULT_DB
    return Path(raw.strip())


def _port(raw: object) -> int:
    """
    Parse LOCAL_AI_BOARD_PORT. A missing value is 8766.

    :param raw: port text as object. Example: "8766"
    :return: port as int. Example: 8766
    :raises ValueError: when the port is not an integer from 1 through 65535. Example: "0"
    """
    if raw is None:
        return _DEFAULT_PORT
    if not isinstance(raw, str) or not raw.strip():
        return _DEFAULT_PORT
    text = raw.strip()
    if not text.isdigit():
        raise ValueError("LOCAL_AI_BOARD_PORT must be an integer. Example: 8766")
    port = int(text)
    if port < _MIN_PORT or port > _MAX_PORT:
        raise ValueError("LOCAL_AI_BOARD_PORT must be from 1 through 65535. Example: 8766")
    return port


def _attach_log_file(log_path: Path) -> None:
    """
    Append INFO lines to logs/app.log. This does not rotate that file.

    The dispatcher rotates logs/app.log when it starts. Renaming it here would
    pull the file out from under that process, so the board only appends.

    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: None. Example: None
    """
    global _LOG_ATTACHED
    if _LOG_ATTACHED:
        return
    _require_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.INFO)
    stderr.setFormatter(logging.Formatter("%(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(stderr)
    _LOG_ATTACHED = True


def _require_optional_path(db_path: Path | None) -> None:
    """
    Reject a database path that is not a Path or None.

    :param db_path: sqlite file as Path | None. Example: Path("data/local-ai.sqlite")
    :return: None. Example: None
    :raises TypeError: when db_path is not a Path or None. Example: "data/local-ai.sqlite"
    """
    if db_path is not None:
        _require_path(db_path)


def _require_path(db_path: Path) -> None:
    """
    Reject a value that is not a Path.

    :param db_path: sqlite file as Path. Example: Path("data/local-ai.sqlite")
    :return: None. Example: None
    :raises TypeError: when db_path is not a Path. Example: "data/local-ai.sqlite"
    """
    if not isinstance(db_path, Path):
        raise TypeError("db_path must be Path. Example: Path('data/local-ai.sqlite')")


def _require_status(status: str | None) -> None:
    """
    Reject a filter that is not a string or None.

    :param status: filter as str | None. Example: "queued"
    :return: None. Example: None
    :raises TypeError: when status is not a string or None. Example: 1
    """
    if status is not None and not isinstance(status, str):
        raise TypeError("status must be a string or omitted. Example: 'queued'")


def _require_env(env: Mapping[str, str]) -> None:
    """
    Reject an environment that is not a mapping.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_BOARD_PORT": "8766"}
    :return: None. Example: None
    :raises TypeError: when env is not a mapping. Example: []
    """
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_BOARD_PORT': '8766'}")


def _esc(value: object) -> str:
    """
    Escape text for HTML.

    :param value: field value as object. Example: "Write <script>"
    :return: escaped text as str. Example: "Write &lt;script&gt;"
    """
    if value is None:
        return ""
    return html.escape(str(value), quote=True)
