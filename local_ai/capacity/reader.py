"""Memory, pressure, and Ollama. DispatcherService calls read before submit."""

from collections.abc import Callable


MemoryReader = Callable[[], tuple[float, float]]
HttpReader = Callable[[str], tuple[int, dict]]
PressureReader = Callable[[], str]


class CapacityReader:
    """
    Reports free memory and whether Ollama is up.

    Called by DispatcherService.submit and by the capacity MCP tool.
    """

    def recommend(self, free_gb: float) -> str:
        """
        Pick a model size from free memory.

        :param free_gb: free gigabytes as float. Example: 20.0
        :return: model name or none as str. Example: "qwen-24b"
        :raises ValueError: when free_gb is negative. Example: -1.0
        """
        # Reject negative free_gb with ValueError.
        # Bands: >=20 qwen-24b, >=12 qwen-14b, >=8 qwen-8b, else "none".
        # Log the band at INFO.
        raise NotImplementedError

    def read_pressure(self, reader: PressureReader) -> str:
        """
        Map a host pressure reading to a word.

        :param reader: zero-arg reader as PressureReader. Example: lambda: "normal"
        :return: pressure as str. Example: "normal"
        """
        # Call reader. Accept normal, warn, and critical.
        # If the reader raises, return "unknown" and log a WARNING.
        # Log the word at INFO.
        raise NotImplementedError

    def read(
        self,
        memory: MemoryReader,
        ollama: HttpReader,
        pressure: PressureReader,
        ollama_host: str,
    ) -> dict:
        """
        Build the capacity tool body.

        :param memory: returns (total_gb, free_gb) as MemoryReader. Example: lambda: (64.0, 40.2)
        :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
        :param pressure: pressure reader as PressureReader. Example: lambda: "normal"
        :param ollama_host: base URL as str. Example: "http://127.0.0.1:11434"
        :return: capacity body as dict. Example: {"status": "ok", "ollama": "up", "recommended": "qwen-14b"}
        """
        # Read total and free gigabytes. Call read_pressure.
        # GET {ollama_host}/api/tags. A non-200 or a connection error means ollama "down".
        # When down: loaded_model null, recommended "none", models [].
        # When up: GET /api/ps for loaded_model, call recommend(free_gb), models are the three logical names.
        # Log ollama state and recommended size at INFO.
        raise NotImplementedError
