"""PromptAssembler builds foundation, shell brief, then the task layer."""

import logging

import pytest

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
