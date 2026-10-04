"""Queue order. DispatcherService asks this before it returns a position."""

from local_ai.contracts import TaskPin


def queue_position(pin: TaskPin, queued: tuple[TaskPin, ...]) -> int:
    """
    One-based position among queued pins. A running pin is not a position.

    :param pin: the pin being placed as TaskPin. Example: a pin with priority 40
    :param queued: queued pins as tuple[TaskPin, ...]. Example: ()
    :return: position as int. Example: 1
    """
    # Order is priority ASC, then created_at ASC.
    # Count queued pins strictly ahead of this pin and add 1.
    # Do not count status running, completed, or failed.
    # Log the task_id and position at INFO.
    raise NotImplementedError
