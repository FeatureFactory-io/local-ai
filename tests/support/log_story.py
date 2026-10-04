"""Assert one log-story beat against captured records."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

LEVEL_NAMES: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def assert_log_story(
    records: Sequence[logging.LogRecord | Mapping[str, Any]],
    where: str,
    beat: str,
    must_include: Sequence[str],
    level: str = "INFO",
) -> None:
    """
    Assert one record at ``level`` contains where, beat, and every fragment.

    :param records: captured log records. Example: [{"message": "dispatcher.submit | queued", "level": "INFO"}]
    :param where: location fragment. Example: "dispatcher.submit"
    :param beat: decision fragment. Example: "queued"
    :param must_include: extra fragments in that same record. Example: ["task_id=7c1e"]
    :param level: level name that record must use. Example: "INFO"
    :return: None when a record matches. Example: None
    :raises TypeError: when records or must_include is not a sequence of text
    :raises ValueError: when where, beat, level, or a fragment is empty or unknown
    :raises AssertionError: when no record at that level contains every fragment
    """
    expected_level = _validated_level(level)
    needles = _story_needles(where, beat, must_include)
    seen = _messages_at_level(_as_record_list(records), expected_level)
    if any(all(needle in message for needle in needles) for message in seen):
        return
    raise AssertionError(
        f"log story missing where={where!r} beat={beat!r} level={level!r} "
        f"must_include={list(must_include)!r}; saw={seen!r}"
    )


def _validated_level(level: str) -> int:
    """
    Resolve a level name to a logging level number.

    :param level: level name. Example: "INFO"
    :return: logging level number. Example: 20
    :raises ValueError: when level is empty or not a known name
    """
    if not isinstance(level, str) or not level.strip():
        raise ValueError("level must be a non-empty string. Example: 'INFO'")
    key = level.upper()
    if key not in LEVEL_NAMES:
        known = ", ".join(sorted(LEVEL_NAMES))
        raise ValueError(f"level must be one of {known}. Example: 'INFO'")
    return LEVEL_NAMES[key]


def _story_needles(where: str, beat: str, must_include: Sequence[str]) -> list[str]:
    """
    Build the fragments that must share one log line.

    :param where: location fragment. Example: "worker.step"
    :param beat: decision fragment. Example: "acting"
    :param must_include: extra fragments. Example: ["tool=shell"]
    :return: fragments in match order. Example: ["worker.step", "acting", "tool=shell"]
    :raises TypeError: when must_include is a str
    :raises ValueError: when where, beat, or a fragment is blank
    """
    _require_text(where, "where")
    _require_text(beat, "beat")
    return [where, beat, *_text_sequence(must_include, "must_include")]


def _require_text(value: object, name: str) -> None:
    """
    Reject a blank or non-string fragment name.

    :param value: candidate text. Example: "dispatcher.submit"
    :param name: argument name used in the error. Example: "where"
    :return: None when value is usable text. Example: None
    :raises ValueError: when value is not a non-empty string
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string. Example: 'dispatcher.submit'")


def _text_sequence(values: Sequence[str], name: str) -> list[str]:
    """
    Copy a sequence of non-empty strings.

    :param values: fragments. Example: ["task_id=7c1e"]
    :param name: argument name used in the error. Example: "must_include"
    :return: the same strings. Example: ["task_id=7c1e"]
    :raises TypeError: when values is a str or not a sequence
    :raises ValueError: when an entry is blank or not a string
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings. Example: ['task_id=7c1e']")
    texts: list[str] = []
    for item in values:
        _require_text(item, f"{name} entry")
        texts.append(item)
    return texts


def _as_record_list(records: Iterable[Any]) -> list[Any]:
    """
    Materialize captured records once.

    :param records: log records. Example: [{"message": "dispatcher.submit | queued", "level": "INFO"}]
    :return: a list of those records. Example: [{"message": "dispatcher.submit | queued", "level": "INFO"}]
    :raises TypeError: when records is a string, a mapping, or not iterable
    """
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
        raise TypeError("records must be a sequence of log records. Example: caplog.records")
    return list(records)


def _messages_at_level(records: Sequence[Any], levelno: int) -> list[str]:
    """
    Keep messages whose level number equals the declared level.

    :param records: log records. Example: [LogRecord(...)]
    :param levelno: logging level number. Example: 20
    :return: messages at that exact level. Example: ["dispatcher.submit | queued"]
    :raises TypeError: when a record is neither a LogRecord nor a mapping
    """
    return [_message(record) for record in records if _levelno(record) == levelno]


def _levelno(record: logging.LogRecord | Mapping[str, Any]) -> int:
    """
    Read the level number from a LogRecord or a mapping.

    :param record: one captured record. Example: {"message": "queued", "level": "INFO"}
    :return: logging level number. Example: 20
    :raises TypeError: when the record has no readable level
    :raises ValueError: when a mapping level name is unknown
    """
    if isinstance(record, logging.LogRecord):
        return record.levelno
    if isinstance(record, Mapping):
        return _mapping_levelno(record)
    raise TypeError(
        "record must be a LogRecord or mapping. Example: caplog.records[0]; "
        f"got {type(record).__name__}"
    )


def _mapping_levelno(record: Mapping[str, Any]) -> int:
    """
    Read a level from a mapping's levelno, level, or levelname.

    :param record: mapping record. Example: {"message": "queued", "level": "INFO"}
    :return: logging level number. Example: 20
    :raises TypeError: when levelno is present and not an int
    :raises ValueError: when the level name is unknown
    """
    if "levelno" in record:
        return _int_levelno(record["levelno"])
    level = record.get("level", record.get("levelname", "INFO"))
    return _named_or_numeric_level(level)


def _int_levelno(levelno: object) -> int:
    """
    Accept a real int level number and reject bool.

    :param levelno: candidate level number. Example: 20
    :return: that number. Example: 20
    :raises TypeError: when levelno is not an int
    """
    if isinstance(levelno, bool) or not isinstance(levelno, int):
        raise TypeError("levelno must be an int. Example: 20")
    return levelno


def _named_or_numeric_level(level: object) -> int:
    """
    Accept a level name or a numeric level.

    :param level: level name or number. Example: "WARNING"
    :return: logging level number. Example: 30
    :raises ValueError: when the name is unknown or the value is not a level
    """
    if isinstance(level, bool):
        raise ValueError("record level must be a known name. Example: 'INFO'")
    if isinstance(level, int):
        return level
    if isinstance(level, str) and level.upper() in LEVEL_NAMES:
        return LEVEL_NAMES[level.upper()]
    raise ValueError(f"record level must be a known name. Example: 'INFO'; got {level!r}")


def _message(record: logging.LogRecord | Mapping[str, Any]) -> str:
    """
    Read the rendered message from a record.

    :param record: one captured record. Example: {"message": "dispatcher.submit | queued", "level": "INFO"}
    :return: message text. Example: "dispatcher.submit | queued"
    :raises TypeError: when the record has no message
    """
    if isinstance(record, logging.LogRecord):
        return record.getMessage()
    if isinstance(record, Mapping) and "message" in record:
        return str(record["message"])
    if isinstance(record, Mapping) and "msg" in record:
        return str(record["msg"])
    raise TypeError(
        "record needs a message. Example: {'message': 'dispatcher.submit | queued', 'level': 'INFO'}"
    )
