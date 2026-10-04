"""Package import directions from SAO section 1."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "local_ai"

FORBIDDEN = {
    "tools": {"dispatcher", "llm", "worker"},
    "llm": {"dispatcher", "worker", "blackboard", "plan"},
    "worker": {"dispatcher"},
    "blackboard": {"dispatcher", "worker", "llm"},
    "plan": {"dispatcher", "worker", "llm"},
    "prompt": {"dispatcher", "worker", "llm"},
    "capacity": {"dispatcher", "worker", "llm", "blackboard"},
    "service": {"worker"},
}


def _imported_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("local_ai."):
            found.add(node.module.split(".")[1])
    return found


def test_service_does_not_import_the_worker() -> None:
    """DispatcherService claims and finishes pins. Slot is the one that calls Worker."""
    imported = _imported_packages(ROOT / "dispatcher" / "service.py")
    assert "worker" not in imported


def test_worker_does_not_import_the_dispatcher() -> None:
    """The worker returns a status. The slot writes the finish."""
    imported = _imported_packages(ROOT / "worker" / "loop.py")
    assert "dispatcher" not in imported


def test_tools_do_not_import_dispatcher_or_llm() -> None:
    """The shell checks the grant and runs. It does not know the queue or the model."""
    for path in (ROOT / "tools").glob("*.py"):
        imported = _imported_packages(path)
        assert imported.isdisjoint({"dispatcher", "llm"})


def test_forbidden_edges_stay_closed() -> None:
    """Each package keeps the SAO dependency rule."""
    for path in ROOT.rglob("*.py"):
        package = path.relative_to(ROOT).parts[0]
        if package not in FORBIDDEN and path.stem not in FORBIDDEN:
            continue
        key = path.stem if path.stem in FORBIDDEN else package
        banned = FORBIDDEN.get(key, set())
        assert _imported_packages(path).isdisjoint(banned), path
