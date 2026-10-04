"""Process logging. Stdout stays empty. Records go to stderr and logs/app.log."""

import logging
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

_OWNED_HANDLER = "local_ai_owned_handler"
_LOG_FORMAT = "%(name)s | task_id=%(task_id)s | %(message)s"


class _TaskIdFilter(logging.Filter):
    """Supply task_id so the formatter never raises when a call omits it."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Default a missing task_id to an empty string.

        :param record: log record as logging.LogRecord. Example: a record with no task_id
        :return: always True as bool. Example: True
        """
        if not hasattr(record, "task_id"):
            record.task_id = ""
        return True


def configure_logging(log_path: Path) -> None:
    """
    Rotate the log file and attach stderr plus a fresh file handler.

    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: None. Example: None
    """
    _require_log_path(log_path)
    root = logging.getLogger()
    _detach_owned_handlers(root)
    rotated = _rotate_existing_log(log_path)
    _attach_process_handlers(root, log_path)
    logger.info("Starting configure_logging path=%s", log_path.name)
    if rotated:
        logger.info("Rotated previous log aside path=%s", log_path.name)
        return
    logger.info("Opened fresh log path=%s", log_path.name)


def _require_log_path(log_path: Path) -> None:
    """
    Reject anything that is not a file path.

    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: None. Example: None
    :raises TypeError: when log_path is not a Path. Example: "logs/app.log"
    """
    if not isinstance(log_path, Path):
        raise TypeError(f"log_path must be a Path, got {type(log_path).__name__}")


def _detach_owned_handlers(root: logging.Logger) -> None:
    """
    Close handlers this module attached so the log file can be renamed.

    :param root: process logger as logging.Logger. Example: logging.getLogger()
    :return: None. Example: None
    """
    for handler in list(root.handlers):
        if getattr(handler, _OWNED_HANDLER, False):
            root.removeHandler(handler)
            handler.close()


def _rotate_existing_log(log_path: Path) -> bool:
    """
    Rename an existing log to a .1 sibling so this start writes a new file.

    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: whether a previous file was renamed as bool. Example: False
    :raises ValueError: when log_path exists and is not a file. Example: Path("logs")
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        return False
    if not log_path.is_file():
        raise ValueError(f"log_path is not a file: {log_path.name}")
    log_path.replace(log_path.with_name(log_path.name + ".1"))
    return True


def _attach_process_handlers(root: logging.Logger, log_path: Path) -> None:
    """
    Attach stderr and a new file. stdout is left untouched.

    :param root: process logger as logging.Logger. Example: logging.getLogger()
    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: None. Example: None
    """
    root.setLevel(logging.INFO)
    root.addHandler(_owned_handler(logging.StreamHandler(sys.stderr)))
    root.addHandler(_owned_handler(logging.FileHandler(log_path, mode="w", encoding="utf-8")))


def _owned_handler(handler: logging.Handler) -> logging.Handler:
    """
    Mark a handler so the next configure can close it.

    :param handler: new handler as logging.Handler. Example: logging.StreamHandler(sys.stderr)
    :return: the same handler as logging.Handler. Example: logging.StreamHandler(sys.stderr)
    """
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_TaskIdFilter())
    setattr(handler, _OWNED_HANDLER, True)
    return handler
