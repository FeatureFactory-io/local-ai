"""SAO section 18 tool examples load and keep their required keys."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contracts"

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "capacity_up.json": (
        "status",
        "memory_total_gb",
        "memory_free_gb",
        "memory_pressure",
        "ollama",
        "loaded_model",
        "recommended",
        "models",
    ),
    "capacity_down.json": (
        "status",
        "memory_total_gb",
        "memory_free_gb",
        "memory_pressure",
        "ollama",
        "loaded_model",
        "recommended",
        "models",
    ),
    "submit_request.json": ("goal", "done_when", "priority", "model", "inputs", "access"),
    "submit_queued.json": ("status", "task_id", "priority", "model", "position", "access"),
    "submit_rejected.json": ("status", "reason", "task_id"),
    "read_running.json": (
        "status",
        "task_id",
        "priority",
        "model",
        "phase",
        "goal",
        "access",
        "current_plan",
        "steps",
        "reasoning",
        "artifacts",
        "error",
    ),
    "reprioritize_request.json": ("task_id", "priority"),
    "reprioritize_queued.json": ("status", "task_id", "priority", "position"),
    "reprioritize_rejected.json": ("status", "reason", "task_id"),
    "list_request.json": ("status",),
    "list_all.json": ("status", "count", "jobs"),
    "list_rejected.json": ("status", "reason", "filter"),
    "cancel_request.json": ("task_id",),
    "cancel_cancelled.json": ("status", "task_id", "job_dir_removed"),
    "cancel_rejected.json": ("status", "reason", "task_id"),
    "purge_request.json": ("task_id", "status"),
    "purge_ok.json": ("status", "count", "purged"),
    "purge_not_terminal.json": ("status", "reason", "task_id"),
    "purge_not_status.json": ("status", "reason", "task_id", "filter"),
}


@pytest.mark.parametrize("filename", list(REQUIRED_KEYS))
def test_contract_fixture_loads_and_has_required_keys(filename: str) -> None:
    """Each section 18 example is JSON and contains the keys from that example."""
    path = FIXTURES / filename
    assert path.is_file(), filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    missing = [key for key in REQUIRED_KEYS[filename] if key not in payload]
    assert missing == []
