"""Acceptance: the local MCP accepts a task, runs it, updates the pin, and leaves the file."""

from pathlib import Path

from local_ai.contracts import Completion, ToolCall
from tests.api.test_mcp import _call_tool, _join_slot, _server


_THESIS = "The draft argues that local work stays on one pin."
_GOAL = "Write a one-paragraph brief in the job folder."
_DONE_WHEN = "brief.md exists and states the thesis in one paragraph."
_WRITE_BRIEF = (
    "python3 -c \"open('brief.md','w').write("
    + repr(_THESIS)
    + ")\""
)
_SCRIPT = (
    Completion(
        content="1. Write brief.md",
        thinking="The brief is one paragraph. I will write it in the job folder.",
    ),
    Completion(
        thinking="Writing brief.md with the thesis paragraph.",
        tool_calls=(ToolCall("shell", _WRITE_BRIEF),),
        final_answer="Wrote brief.md",
    ),
)


def test_local_mcp_accepts_a_task_and_finishes_the_brief(tmp_path: Path) -> None:
    """Submit through FastMCP, then read a completed pin and the brief file."""
    server = _server(tmp_path, script=_SCRIPT)
    mcp = server.build()

    capacity = _call_tool(mcp, "capacity", {})
    accepted = _call_tool(
        mcp,
        "submit",
        {
            "goal": _GOAL,
            "done_when": _DONE_WHEN,
            "priority": 40,
            "model": "qwen-14b",
        },
    )
    task_id = str(accepted["task_id"])
    _join_slot(server)
    pin = _call_tool(mcp, "read", {"task_id": task_id})
    brief = tmp_path / "jobs" / task_id / "brief.md"

    assert capacity["status"] == "ok"
    assert capacity["ollama"] == "up"
    assert accepted["status"] == "queued"
    assert accepted["priority"] == 40
    assert accepted["model"] == "qwen-14b"
    assert pin["status"] == "completed"
    assert pin["phase"] == "completed"
    assert pin["error"] == ""
    assert pin["goal"] == _GOAL
    assert pin["steps"][0]["status"] == "completed"
    assert pin["steps"][0]["llm_synthesis"] == "Wrote brief.md"
    assert any("thesis" in note["text"] for note in pin["reasoning"])
    assert "1. Write brief.md" in pin["current_plan"]
    assert brief.read_text(encoding="utf-8") == _THESIS
    assert (tmp_path / "jobs" / task_id / "shell.log").is_file()
