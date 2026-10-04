"""PromptAssembler builds foundation, shell brief, then the task layer."""

import logging

import pytest

from local_ai.contracts import AccessGrant
from local_ai.prompt.assemble import PromptAssembler


def test_foundation_states_identity_plan_and_recovery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The foundation tells the worker to plan, act, recover, and note why."""
    caplog.set_level(logging.INFO)

    text = PromptAssembler()._foundation()

    assert text.startswith("You are the local worker for one task.")
    assert "Plan, then act one step at a time." in text
    assert "After a failed command, revise the remaining plan." in text
    assert "Emit a short reasoning note before each command." in text
    assert any(record.levelno == logging.INFO and "_foundation" in record.message for record in caplog.records)


def test_shell_brief_names_binaries_and_the_grant(caplog: pytest.LogCaptureFixture) -> None:
    """The shell brief names the binaries, including mmdc, and the grant rule."""
    caplog.set_level(logging.INFO)

    text = PromptAssembler()._shell_brief()

    assert "python3, curl, pandoc, weasyprint, and mmdc are on PATH." in text
    for binary in ("python3", "curl", "pandoc", "weasyprint", "mmdc"):
        assert binary in text
    assert "access grant" in text
    assert "Declare output files" in text
    assert any(record.levelno == logging.INFO and "_shell_brief" in record.message for record in caplog.records)


def test_task_includes_goal_inputs_grant_plan_and_prior(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The task layer carries the pin fields and the grant object."""
    grant = AccessGrant(mode="home", paths=("/Users/denispetelin",))
    goal = "Turn the notes into a one-page brief."
    done_when = "brief.md exists"
    inputs = ("/Users/denispetelin/books/draft",)
    plan_text = "1. List chapter files (pending)."
    prior = "Found ch-01.md through ch-12.md"
    caplog.set_level(logging.INFO)

    text = PromptAssembler()._task(goal, done_when, grant, inputs, plan_text, prior)

    assert text.startswith("Goal: Turn the notes into a one-page brief.")
    assert f"Done when: {done_when}" in text
    assert inputs[0] in text
    assert repr(grant) in text
    assert plan_text in text
    assert prior in text
    assert all(goal not in record.message for record in caplog.records)


def test_task_rejects_a_grant_that_is_not_an_access_grant() -> None:
    """A plain dict is not a grant object the task layer can print."""
    with pytest.raises(TypeError):
        PromptAssembler()._task("goal", "done", {"mode": "job"}, (), "", "")  # type: ignore[arg-type]
