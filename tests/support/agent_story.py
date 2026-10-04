"""Assert ordered workflow, tool, and domain-effect beats on a run trace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

STORY_KEYS: tuple[str, ...] = ("workflow", "tool", "effect")


def assert_agent_story(
    trace: Sequence[Mapping[str, Any]],
    beats: Sequence[Mapping[str, Any]],
) -> None:
    """
    Assert each beat matches a later trace event, in order.

    A beat names a workflow, a tool, a domain effect, or a combination.
    The beat matches an event when every key equals the event's value.
    Each beat consumes its own event, so one event cannot cover the next beat.

    :param trace: ordered run events. Example: [{"workflow": "planning"}, {"tool": "shell"}]
    :param beats: ordered expected events. Example: [{"workflow": "planning"}, {"tool": "shell"}, {"effect": "step_completed"}]
    :return: None when every beat matches. Example: None
    :raises TypeError: when trace or beats is not a sequence of mappings
    :raises ValueError: when beats is empty or a beat omits workflow, tool, and effect
    :raises AssertionError: when a beat is missing from the remainder of the trace
    """
    events = _as_mappings(trace, "trace")
    expected = _validated_beats(beats)
    _require_ordered_beats(events, expected)


def _require_ordered_beats(
    events: Sequence[Mapping[str, Any]],
    beats: Sequence[Mapping[str, Any]],
) -> None:
    """
    Walk the trace forward and consume one event per beat.

    :param events: trace mappings. Example: [{"workflow": "planning"}]
    :param beats: validated beats. Example: [{"workflow": "planning"}]
    :return: None when every beat matches. Example: None
    :raises AssertionError: when a beat has no later match
    """
    cursor = 0
    for index, beat in enumerate(beats):
        found = _next_match(events, beat, cursor)
        if found is None:
            raise AssertionError(
                f"agent story missing beat {index} {dict(beat)!r} "
                f"after position {cursor}; trace={[dict(event) for event in events]!r}"
            )
        cursor = found + 1


def _next_match(
    events: Sequence[Mapping[str, Any]],
    beat: Mapping[str, Any],
    start: int,
) -> int | None:
    """
    Find the next event at or after start that equals the beat on every key.

    :param events: trace mappings. Example: [{"tool": "shell"}]
    :param beat: expected fields. Example: {"tool": "shell"}
    :param start: first index still available. Example: 0
    :return: matching index, or None when none remains. Example: 1
    """
    for index in range(start, len(events)):
        if _event_matches(events[index], beat):
            return index
    return None


def _event_matches(event: Mapping[str, Any], beat: Mapping[str, Any]) -> bool:
    """
    Return whether every beat field equals the event.

    :param event: one trace event. Example: {"workflow": "acting", "tool": "shell"}
    :param beat: required fields. Example: {"tool": "shell"}
    :return: True when the beat is a sub-mapping of the event. Example: True
    """
    return all(event.get(key) == expected for key, expected in beat.items())


def _validated_beats(beats: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """
    Require a non-empty sequence of story beats.

    :param beats: caller beats. Example: [{"effect": "step_completed"}]
    :return: those beats. Example: [{"effect": "step_completed"}]
    :raises TypeError: when beats is not a sequence of mappings
    :raises ValueError: when beats is empty or a beat has no story key
    """
    parsed = _as_mappings(beats, "beats")
    if not parsed:
        raise ValueError(
            "beats must name at least one workflow, tool, or effect. "
            "Example: [{'workflow': 'planning'}]"
        )
    for beat in parsed:
        _require_story_key(beat)
    return parsed


def _require_story_key(beat: Mapping[str, Any]) -> None:
    """
    Require workflow, tool, or effect, and require those values to be text.

    :param beat: one beat. Example: {"tool": "shell"}
    :return: None when the beat is usable. Example: None
    :raises ValueError: when no story key is present or a story value is blank
    """
    if not any(key in beat for key in STORY_KEYS):
        raise ValueError(
            "beat must include workflow, tool, or effect. Example: {'workflow': 'planning'}"
        )
    for key in STORY_KEYS:
        if key in beat:
            _require_text(beat[key], key)


def _require_text(value: object, name: str) -> None:
    """
    Reject a blank or non-string story value.

    :param value: candidate text. Example: "planning"
    :param name: field name used in the error. Example: "workflow"
    :return: None when value is usable text. Example: None
    :raises ValueError: when value is not a non-empty string
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string. Example: 'planning'")


def _as_mappings(values: object, name: str) -> list[Mapping[str, Any]]:
    """
    Copy a sequence of mappings.

    :param values: candidate sequence. Example: [{"workflow": "planning"}]
    :param name: argument name used in the error. Example: "trace"
    :return: the mappings. Example: [{"workflow": "planning"}]
    :raises TypeError: when values is not a sequence of mappings
    """
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings. Example: [{{'workflow': 'planning'}}]")
    parsed: list[Mapping[str, Any]] = []
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError(f"{name} entries must be mappings. Example: {{'tool': 'shell'}}")
        parsed.append(item)
    return parsed
