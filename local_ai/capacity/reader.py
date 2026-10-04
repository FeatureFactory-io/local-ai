"""Memory, pressure, and Ollama. DispatcherService calls read before submit."""

import logging
import math
from collections.abc import Callable


logger = logging.getLogger(__name__)

MemoryReader = Callable[[], tuple[float, float]]
HttpReader = Callable[[str], tuple[int, dict]]
PressureReader = Callable[[], str]

FREE_GB_QWEN_24B = 20.0
FREE_GB_QWEN_14B = 12.0
FREE_GB_QWEN_8B = 8.0
MODEL_QWEN_24B = "qwen-24b"
MODEL_QWEN_14B = "qwen-14b"
MODEL_QWEN_8B = "qwen-8b"
RECOMMENDATION_NONE = "none"
PRESSURE_NORMAL = "normal"
PRESSURE_WARN = "warn"
PRESSURE_CRITICAL = "critical"
PRESSURE_UNKNOWN = "unknown"
PRESSURE_WORDS = frozenset({PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL})


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
        logger.info("Starting recommend free_gb=%s", free_gb)
        amount = _require_free_gigabytes(free_gb)
        band = _band_for_free_gigabytes(amount)
        logger.info("recommend chose band=%s free_gb=%s", band, amount)
        return band

    def read_pressure(self, reader: PressureReader) -> str:
        """
        Map a host pressure reading to a word.

        :param reader: zero-arg reader as PressureReader. Example: lambda: "normal"
        :return: pressure as str. Example: "normal"
        """
        logger.info("Starting read_pressure")
        word = _pressure_word_from(reader)
        logger.info("read_pressure word=%s", word)
        return word

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


def _pressure_word_from(reader: PressureReader) -> str:
    """
    Call the host probe and keep only normal, warn, or critical.

    :param reader: zero-arg reader as PressureReader. Example: lambda: "normal"
    :return: accepted word or unknown as str. Example: "unknown"
    """
    if not callable(reader):
        logger.warning("read_pressure reader is not callable")
        return PRESSURE_UNKNOWN
    try:
        raw = reader()
    except Exception as exc:
        logger.warning("read_pressure reader failed error_type=%s", type(exc).__name__)
        return PRESSURE_UNKNOWN
    if raw in PRESSURE_WORDS:
        return raw
    logger.warning("read_pressure unexpected word=%s", raw)
    return PRESSURE_UNKNOWN


def _require_free_gigabytes(free_gb: float) -> float:
    """
    Reject a reading that is not a finite, non-negative number of gigabytes.

    :param free_gb: candidate reading as float. Example: 20.0
    :return: the same number as float. Example: 20.0
    :raises ValueError: when free_gb is negative, non-finite, or not a number. Example: -1.0
    """
    if isinstance(free_gb, bool) or not isinstance(free_gb, (int, float)):
        raise ValueError(f"free_gb must be a number. Example: 20.0; got {type(free_gb).__name__}")
    amount = float(free_gb)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError(
            f"free_gb must be a non-negative finite number. Example: 20.0; got {free_gb}"
        )
    return amount


def _band_for_free_gigabytes(free_gb: float) -> str:
    """
    Map free gigabytes onto a model size.

    :param free_gb: non-negative gigabytes as float. Example: 20.0
    :return: model name or none as str. Example: "qwen-24b"
    """
    if free_gb >= FREE_GB_QWEN_24B:
        return MODEL_QWEN_24B
    if free_gb >= FREE_GB_QWEN_14B:
        return MODEL_QWEN_14B
    if free_gb >= FREE_GB_QWEN_8B:
        return MODEL_QWEN_8B
    return RECOMMENDATION_NONE
