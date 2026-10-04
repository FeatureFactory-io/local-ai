"""Importing a skeleton must succeed. Calling a stub is a later fill slice."""

import sqlite3

import pytest


def test_sqlite_and_fastmcp_import() -> None:
    """The venv can see stdlib sqlite and FastMCP from the official SDK."""
    from mcp.server.fastmcp import FastMCP

    assert sqlite3.sqlite_version
    assert FastMCP.__name__ == "FastMCP"


@pytest.mark.parametrize(
    "module_name",
    [
        "local_ai.contracts",
        "local_ai.logging_setup",
        "local_ai.db",
        "local_ai.blackboard",
        "local_ai.plan",
        "local_ai.tools",
        "local_ai.prompt",
        "local_ai.capacity",
        "local_ai.llm",
        "local_ai.worker",
        "local_ai.dispatcher",
    ],
)
def test_skeleton_module_imports(module_name: str) -> None:
    """Each product module imports. Bodies still raise NotImplementedError when called."""
    module = __import__(module_name, fromlist=["*"])
    assert module is not None
