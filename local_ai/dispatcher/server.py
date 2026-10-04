"""FastMCP tools. Each tool calls DispatcherService or CapacityReader and returns the dict."""

import sqlite3

from local_ai.capacity.reader import CapacityReader
from local_ai.dispatcher.service import DispatcherService
from local_ai.dispatcher.slot import Slot
from local_ai.worker.loop import Worker


# Worker is constructed in main and passed to Slot. Service does not import it.
COMPOSER_WORKER = Worker


class LocalAiServer:
    """
    MCP surface. Cursor and Claude use stdio. ChatGPT uses loopback streamable HTTP.

    Tools call the service. They do not embed queue policy.
    """

    def __init__(
        self,
        service: DispatcherService,
        slot: Slot,
        capacity: CapacityReader,
        connection: sqlite3.Connection,
    ) -> None:
        """
        Store the wired objects. FastMCP registration happens in build.

        :param service: pin service as DispatcherService. Example: the composed service
        :param slot: worker slot as Slot. Example: Slot(service, worker, connection, clock)
        :param capacity: capacity reader as CapacityReader. Example: CapacityReader()
        :param connection: shared connection as sqlite3.Connection. Example: the Database connection
        """
        self._service = service
        self._slot = slot
        self._capacity = capacity
        self._connection = connection

    def build(self):
        """
        Register the four tools on a FastMCP server.

        :return: FastMCP server. Example: a server named "local-ai"
        """
        # Import FastMCP here only if module-level import is already done in this module.
        # Register capacity, submit, read, and reprioritize.
        # Each tool returns the dict from the matching method. A rejected dict is not an exception.
        raise NotImplementedError

    def capacity(self) -> dict:
        """
        Current machine load and a model-size recommendation.

        :return: capacity body as dict. Example: {"status": "ok", "ollama": "up", "recommended": "qwen-14b"}
        """
        # CapacityReader.read with the process memory reader, the Ollama HTTP reader, and OLLAMA_HOST.
        # Log the tool name at INFO. There is no task_id yet.
        raise NotImplementedError

    def submit(
        self,
        goal: str,
        done_when: str,
        priority: int | None = None,
        model: str | None = None,
        inputs: list[str] | None = None,
        access: dict | None = None,
    ) -> dict:
        """
        Pin a task and enqueue it.

        :param goal: worker goal as str. Example: "Summarize the book into 2-3 pages."
        :param done_when: completion condition as str. Example: "summary.md exists"
        :param priority: 1..100 or omitted as int | None. Example: 40
        :param model: qwen size or omitted as str | None. Example: "qwen-14b"
        :param inputs: files to copy as list[str] | None. Example: []
        :param access: grant object or omitted as dict | None. Example: {"mode": "job"}
        :return: queued or rejected body as dict. Example: {"status": "queued", "position": 1}
        """
        # Build SubmitRequest. DispatcherService.submit returns the queued snapshot.
        # Then Slot.kick. The returned dict stays the snapshot from before the claim.
        # Log the tool name at INFO.
        raise NotImplementedError

    def read(self, task_id: str) -> dict:
        """
        Return one pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :return: pin body as dict. Example: {"status": "running", "phase": "acting"}
        """
        # DispatcherService.read_pin.
        raise NotImplementedError

    def reprioritize(self, task_id: str, priority: int) -> dict:
        """
        Change priority of a queued pin.

        :param task_id: pin id as str. Example: "7c1e0a2e-3b4d-4f5a-9c8d-1a2b3c4d5e6f"
        :param priority: new priority as int. Example: 5
        :return: queued or rejected body as dict. Example: {"status": "rejected", "reason": "not_queued"}
        """
        # DispatcherService.reprioritize. Do not call the slot.
        raise NotImplementedError


def main() -> None:
    """
    Start the dispatcher. Stdio unless LOCAL_AI_HTTP_PORT is set.

    :return: None. Example: None
    """
    # configure_logging. Open Database, migrate, DispatcherService.recover_running.
    # Compose Blackboard, PlanStore, CapacityReader, GrantChecker, ShellTool,
    # PromptAssembler, LiteLLMAdapter, Worker, Slot, and LocalAiServer.
    # LocalAiServer.build.
    # If LOCAL_AI_HTTP_PORT is unset, FastMCP run transport stdio.
    # If it is set, FastMCP run transport streamable-http host 127.0.0.1 and that port.
    # Logs stay on stderr and logs/app.log.
    raise NotImplementedError


def http_port_from_env(env: dict[str, str] | None = None) -> int | None:
    """
    Read the loopback port. Unset means stdio.

    :param env: environment as dict[str, str] | None. Example: {"LOCAL_AI_HTTP_PORT": "8765"}
    :return: port or None as int | None. Example: 8765
    :raises ValueError: when the variable is not an integer. Example: "abc"
    """
    # Read LOCAL_AI_HTTP_PORT from env or os.environ.
    # Empty or missing returns None.
    # A present value must be an integer port. Do not bind a socket here.
    raise NotImplementedError
