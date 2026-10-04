"""render_plan formats steps and does not touch the database."""

from pathlib import Path

import pytest

from local_ai.contracts import PlanStep
from local_ai.db.database import Database
from local_ai.plan.steps import PlanStore


_TASK_ID = "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"


def test_render_plan_puts_each_step_on_its_own_line(tmp_path: Path) -> None:
    """Each step is one line: "{order}. {action} ({status})." """
    database_path = tmp_path / "local-ai.sqlite"
    store = PlanStore(Database(database_path))
    steps = (
        PlanStep(_TASK_ID, 1, "List chapter files", "completed"),
        PlanStep(_TASK_ID, 2, "Write a thesis per chapter", "running"),
        PlanStep(_TASK_ID, 3, "Assemble summary.md", "pending"),
    )

    rendered = store.render_plan(steps)

    assert rendered == (
        "1. List chapter files (completed).\n"
        "2. Write a thesis per chapter (running).\n"
        "3. Assemble summary.md (pending)."
    )
    assert not database_path.exists()


def test_render_plan_matches_the_single_step_example() -> None:
    """One step matches the docstring example, and no steps yield an empty block."""
    store = PlanStore(Database(Path("data/local-ai.sqlite")))
    step = PlanStep("id", 1, "List chapter files", "completed")

    assert store.render_plan((step,)) == "1. List chapter files (completed)."
    assert store.render_plan(()) == ""


def test_render_plan_rejects_a_non_tuple() -> None:
    """render_plan fails fast when steps is not a tuple of PlanStep."""
    store = PlanStore(Database(Path("data/local-ai.sqlite")))
    with pytest.raises(TypeError, match="tuple"):
        store.render_plan([PlanStep("id", 1, "List chapter files", "completed")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PlanStep"):
        store.render_plan(("List chapter files",))  # type: ignore[arg-type]
