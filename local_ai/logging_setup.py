"""Process logging. Stdout stays empty. Records go to stderr and logs/app.log."""

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def configure_logging(log_path: Path) -> None:
    """
    Rotate the log file and attach stderr plus a fresh file handler.

    :param log_path: log file as Path. Example: Path("logs/app.log")
    :return: None. Example: None
    """
    # If log_path exists, rename it to log_path with a ".1" suffix so this start has an empty file.
    # Create the parent directory.
    # Attach a stderr StreamHandler and a FileHandler at INFO.
    # Formatter includes task_id (empty string when the call has none), logger name, and message.
    # Leave stdout untouched.
    raise NotImplementedError
