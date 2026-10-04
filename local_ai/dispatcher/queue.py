"""Queue order. DispatcherService asks this before it returns a position."""

import logging

from local_ai.contracts import TaskPin


logger = logging.getLogger(__name__)


def queue_position(pin: TaskPin, queued: tuple[TaskPin, ...]) -> int:
    """
    One-based position among queued pins. A running pin is not a position.

    :param pin: the pin being placed as TaskPin. Example: a pin with priority 40
    :param queued: queued pins as tuple[TaskPin, ...]. Example: ()
    :return: position as int. Example: 1
    """
    _require_pin(pin)
    _require_queued(queued)
    logger.info("Starting queue_position task_id=%s", pin.task_id)
    ahead = sum(1 for other in queued if _is_ahead(other, pin))
    position = ahead + 1
    _log_rank(pin.task_id, position, queued)
    return position


def _require_pin(pin: TaskPin) -> None:
    """
    Reject a value that is not a pin.

    :param pin: the pin being placed as TaskPin. Example: a pin with priority 40
    :return: None. Example: None
    :raises TypeError: when pin is not a TaskPin. Example: None
    """
    if not isinstance(pin, TaskPin):
        raise TypeError(f"pin must be a TaskPin, got {type(pin).__name__}")


def _require_queued(queued: tuple[TaskPin, ...]) -> None:
    """
    Reject a queue that is not a tuple of pins.

    :param queued: queued pins as tuple[TaskPin, ...]. Example: ()
    :return: None. Example: None
    :raises TypeError: when queued is not a tuple of TaskPin. Example: []
    """
    if not isinstance(queued, tuple):
        raise TypeError(f"queued must be a tuple, got {type(queued).__name__}")
    for item in queued:
        if not isinstance(item, TaskPin):
            raise TypeError(f"queued items must be TaskPin, got {type(item).__name__}")


def _is_ahead(other: TaskPin, pin: TaskPin) -> bool:
    """
    True when other is a queued pin that should run before pin.

    :param other: candidate as TaskPin. Example: a pin with priority 1
    :param pin: the pin being placed as TaskPin. Example: a pin with priority 40
    :return: whether other is strictly ahead as bool. Example: True
    """
    if other.status != "queued" or other.task_id == pin.task_id:
        return False
    if other.priority != pin.priority:
        return other.priority < pin.priority
    # ISO-8601 timestamps sort lexicographically, so created_at ASC is string order.
    return other.created_at < pin.created_at


def _log_rank(task_id: str, position: int, queued: tuple[TaskPin, ...]) -> None:
    """
    Log the 1-based place and how many non-queued pins were ignored.

    :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
    :param position: 1-based place as int. Example: 1
    :param queued: pins considered as tuple[TaskPin, ...]. Example: ()
    :return: None. Example: None
    """
    skipped = sum(1 for other in queued if other.status != "queued")
    logger.info(
        "Ranked queued pins task_id=%s position=%s skipped_not_queued=%s",
        task_id,
        position,
        skipped,
    )
