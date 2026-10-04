"""Contract for assert_agent_story. Beats stay in order and are not reused."""

import pytest

from tests.support.agent_story import assert_agent_story

HAPPY_BEATS = [
    {"workflow": "planning"},
    {"tool": "shell"},
    {"effect": "step_completed"},
]


def test_assert_agent_story_with_ordered_workflow_tool_and_effect_passes() -> None:
    """Noise events may sit between planning, the shell call, and the domain effect."""
    trace = [
        {"workflow": "planning"},
        {"workflow": "acting", "note": "list files"},
        {"tool": "shell"},
        {"effect": "step_completed", "order": 1},
    ]
    assert_agent_story(trace, HAPPY_BEATS)


def test_assert_agent_story_with_combined_beat_on_one_event_passes() -> None:
    """One event can satisfy one beat that names workflow, tool, and effect together."""
    trace = [{"workflow": "acting", "tool": "shell", "effect": "step_completed"}]
    beats = [{"workflow": "acting", "tool": "shell", "effect": "step_completed"}]
    assert_agent_story(trace, beats)


def test_assert_agent_story_with_tool_before_workflow_raises() -> None:
    """Once planning is consumed, an earlier shell call is no longer ahead of it."""
    trace = [
        {"tool": "shell"},
        {"workflow": "planning"},
        {"effect": "step_completed"},
    ]
    with pytest.raises(AssertionError, match="shell"):
        assert_agent_story(trace, HAPPY_BEATS)


def test_assert_agent_story_with_wrong_tool_name_raises() -> None:
    """The tool beat must name shell, not another tool."""
    trace = [
        {"workflow": "planning"},
        {"tool": "read"},
        {"effect": "step_completed"},
    ]
    with pytest.raises(AssertionError, match="shell"):
        assert_agent_story(trace, HAPPY_BEATS)


def test_assert_agent_story_with_missing_domain_effect_raises() -> None:
    """Workflow and tool without the domain effect is not a finished story."""
    trace = [
        {"workflow": "planning"},
        {"tool": "shell"},
    ]
    with pytest.raises(AssertionError, match="step_completed"):
        assert_agent_story(trace, HAPPY_BEATS)


def test_assert_agent_story_does_not_reuse_one_event_for_later_beats() -> None:
    """Each beat consumes its own later event. One collapsed event cannot cover three beats."""
    trace = [{"workflow": "planning", "tool": "shell", "effect": "step_completed"}]
    with pytest.raises(AssertionError, match="shell"):
        assert_agent_story(trace, HAPPY_BEATS)


def test_assert_agent_story_with_empty_beats_raises() -> None:
    """An empty beat list would pass a trace that proved nothing."""
    with pytest.raises(ValueError, match="beats"):
        assert_agent_story([{"workflow": "planning"}], [])


def test_assert_agent_story_with_beat_missing_story_key_raises() -> None:
    """A beat must name a workflow, a tool, or a domain effect."""
    with pytest.raises(ValueError, match="workflow"):
        assert_agent_story([{"status": "ok"}], [{"status": "ok"}])


def test_assert_agent_story_with_empty_trace_raises() -> None:
    """No events cannot satisfy the first beat."""
    with pytest.raises(AssertionError, match="planning"):
        assert_agent_story([], HAPPY_BEATS)


def test_assert_agent_story_with_mismatched_extra_field_raises() -> None:
    """Extra fields on a beat, such as phase, must equal the event."""
    trace = [{"workflow": "acting", "phase": "planning"}]
    with pytest.raises(AssertionError, match="acting"):
        assert_agent_story(trace, [{"workflow": "acting", "phase": "acting"}])


def test_assert_agent_story_with_non_mapping_event_raises() -> None:
    """A bare string is not a trace event."""
    with pytest.raises(TypeError, match="trace"):
        assert_agent_story(["planning"], [{"workflow": "planning"}])
