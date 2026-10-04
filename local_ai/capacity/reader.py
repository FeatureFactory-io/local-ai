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
HTTP_OK = 200
OLLAMA_UP = "up"
OLLAMA_DOWN = "down"
CAPACITY_OK = "ok"
API_TAGS = "/api/tags"
API_PS = "/api/ps"
LOGICAL_MODELS: tuple[str, ...] = (MODEL_QWEN_8B, MODEL_QWEN_14B, MODEL_QWEN_24B)


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
        :return: capacity body as dict. Example: {"status": "ok", "memory_total_gb": 64.0, "memory_free_gb": 14.0, "memory_pressure": "normal", "ollama": "up", "loaded_model": "qwen2.5:14b", "recommended": "qwen-14b", "models": ["qwen-8b", "qwen-14b", "qwen-24b"]}
        :raises TypeError: when memory or ollama is not callable. Example: memory=None
        :raises ValueError: when memory is not (total_gb, free_gb) or ollama_host is blank. Example: ollama_host=""
        """
        logger.info("Starting read ollama_host=%s", ollama_host)
        host = _require_read_ports(ollama, ollama_host)
        total_gb, free_gb = _memory_gigabytes(memory)
        pressure_word = self.read_pressure(pressure)
        body = _body_after_tags(total_gb, free_gb, pressure_word, ollama, host, self.recommend)
        logger.info("capacity.read | %s | recommended=%s", body["ollama"], body["recommended"])
        return body


def _require_read_ports(ollama: HttpReader, ollama_host: str) -> str:
    """
    Reject a missing HTTP reader or a blank host before any GET.

    :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
    :param ollama_host: base URL as str. Example: "http://127.0.0.1:11434"
    :return: host without a trailing slash as str. Example: "http://127.0.0.1:11434"
    :raises TypeError: when ollama is not callable or the host is not str. Example: ollama=None
    :raises ValueError: when ollama_host is blank. Example: "  "
    """
    if not callable(ollama):
        raise TypeError("ollama must be callable. Example: lambda path: (200, {'models': []})")
    if not isinstance(ollama_host, str):
        raise TypeError("ollama_host must be str. Example: 'http://127.0.0.1:11434'")
    host = ollama_host.strip().rstrip("/")
    if not host:
        raise ValueError("ollama_host must be a non-empty URL. Example: 'http://127.0.0.1:11434'")
    return host


def _memory_gigabytes(memory: MemoryReader) -> tuple[float, float]:
    """
    Read total and free gigabytes from the injected probe.

    :param memory: returns (total_gb, free_gb) as MemoryReader. Example: lambda: (64.0, 40.2)
    :return: total and free gigabytes as tuple[float, float]. Example: (64.0, 40.2)
    :raises TypeError: when memory is not callable. Example: memory=None
    :raises ValueError: when the probe does not return two gigabytes. Example: (64.0,)
    """
    if not callable(memory):
        raise TypeError("memory must be callable. Example: lambda: (64.0, 40.2)")
    reading = memory()
    if not isinstance(reading, tuple) or len(reading) != 2:
        raise ValueError("memory must return (total_gb, free_gb). Example: (64.0, 40.2)")
    total_gb, free_gb = reading
    return _require_gigabytes(total_gb, "total_gb"), _require_gigabytes(free_gb, "free_gb")


def _body_after_tags(
    total_gb: float,
    free_gb: float,
    pressure_word: str,
    ollama: HttpReader,
    host: str,
    recommend: Callable[[float], str],
) -> dict:
    """
    GET /api/tags, then either the down body or ps plus recommend.

    :param total_gb: total gigabytes as float. Example: 64.0
    :param free_gb: free gigabytes as float. Example: 40.2
    :param pressure_word: pressure as str. Example: "normal"
    :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
    :param host: base URL without a trailing slash as str. Example: "http://127.0.0.1:11434"
    :param recommend: size picker as Callable[[float], str]. Example: CapacityReader().recommend
    :return: capacity body as dict. Example: {"status": "ok", "ollama": "down", "recommended": "none"}
    """
    # Tags is the readiness check. A failed ps still leaves Ollama up.
    if not _tags_ok(ollama, host):
        return _capacity_body(
            total_gb, free_gb, pressure_word, OLLAMA_DOWN, None, RECOMMENDATION_NONE, []
        )
    return _capacity_body(
        total_gb,
        free_gb,
        pressure_word,
        OLLAMA_UP,
        _loaded_model(ollama, host),
        recommend(free_gb),
        list(LOGICAL_MODELS),
    )


def _capacity_body(
    total_gb: float,
    free_gb: float,
    pressure_word: str,
    ollama_state: str,
    loaded_model: str | None,
    recommended: str,
    models: list[str],
) -> dict:
    """
    Assemble the capacity tool fields in contract order.

    :param total_gb: total gigabytes as float. Example: 64.0
    :param free_gb: free gigabytes as float. Example: 40.2
    :param pressure_word: pressure as str. Example: "normal"
    :param ollama_state: up or down as str. Example: "up"
    :param loaded_model: Ollama name or null as str | None. Example: "qwen2.5:14b"
    :param recommended: size or none as str. Example: "qwen-24b"
    :param models: logical names as list[str]. Example: ["qwen-8b", "qwen-14b", "qwen-24b"]
    :return: capacity body as dict. Example: {"status": "ok", "ollama": "up", "recommended": "qwen-24b"}
    """
    return {
        "status": CAPACITY_OK,
        "memory_total_gb": total_gb,
        "memory_free_gb": free_gb,
        "memory_pressure": pressure_word,
        "ollama": ollama_state,
        "loaded_model": loaded_model,
        "recommended": recommended,
        "models": models,
    }


def _tags_ok(ollama: HttpReader, host: str) -> bool:
    """
    True only when GET /api/tags returns 200.

    :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
    :param host: base URL as str. Example: "http://127.0.0.1:11434"
    :return: whether tags returned 200 as bool. Example: True
    """
    exchange = _get(ollama, f"{host}{API_TAGS}")
    if exchange is None:
        return False
    status, _body = exchange
    if status == HTTP_OK:
        return True
    logger.info("capacity.read tags status=%s treated as down", status)
    return False


def _loaded_model(ollama: HttpReader, host: str) -> str | None:
    """
    Read the first model name from GET /api/ps. A failed ps is null.

    :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
    :param host: base URL as str. Example: "http://127.0.0.1:11434"
    :return: loaded name or null as str | None. Example: "qwen2.5:14b"
    """
    exchange = _get(ollama, f"{host}{API_PS}")
    if exchange is None:
        return None
    status, body = exchange
    if status != HTTP_OK:
        logger.info("capacity.read ps status=%s loaded_model=null", status)
        return None
    return _first_loaded_name(body)


def _get(ollama: HttpReader, url: str) -> tuple[int, object] | None:
    """
    Call the injected reader. An exception or a bad pair is a failed GET.

    :param ollama: HTTP reader as HttpReader. Example: lambda path: (200, {"models": []})
    :param url: full URL as str. Example: "http://127.0.0.1:11434/api/tags"
    :return: status and body, or None when the call fails. Example: (200, {"models": []})
    """
    try:
        result = ollama(url)
    except Exception as exc:
        logger.warning("capacity.read request failed url=%s error_type=%s", url, type(exc).__name__)
        return None
    if not isinstance(result, tuple) or len(result) != 2:
        logger.warning("capacity.read malformed response url=%s", url)
        return None
    status, body = result
    if isinstance(status, bool) or not isinstance(status, int):
        logger.warning("capacity.read malformed status url=%s", url)
        return None
    return status, body


def _first_loaded_name(body: object) -> str | None:
    """
    Take the name, or else the model, of the first loaded row.

    :param body: /api/ps JSON as object. Example: {"models": [{"name": "qwen2.5:14b"}]}
    :return: model name or null as str | None. Example: "qwen2.5:14b"
    """
    if not isinstance(body, dict):
        return None
    models = body.get("models")
    if not isinstance(models, list) or not models:
        return None
    first = models[0]
    if not isinstance(first, dict):
        return None
    return _model_label(first)


def _model_label(row: dict) -> str | None:
    """
    Prefer a non-blank name, then a non-blank model field.

    :param row: one /api/ps model as dict. Example: {"name": "qwen2.5:14b"}
    :return: label or null as str | None. Example: "qwen2.5:14b"
    """
    for key in ("name", "model"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


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
    return _require_gigabytes(free_gb, "free_gb")


def _require_gigabytes(amount: float, label: str) -> float:
    """
    Reject a reading that is not a finite, non-negative number of gigabytes.

    :param amount: candidate reading as float. Example: 20.0
    :param label: field name used in the error as str. Example: "free_gb"
    :return: the same number as float. Example: 20.0
    :raises ValueError: when amount is negative, non-finite, or not a number. Example: -1.0
    """
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError(f"{label} must be a number. Example: 20.0; got {type(amount).__name__}")
    value = float(amount)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{label} must be a non-negative finite number. Example: 20.0; got {amount}"
        )
    return value


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
