"""Model port. Worker depends on LlmPort. The dispatcher does not."""

import logging
from collections.abc import Mapping

from local_ai.contracts import Completion, ScriptExhausted


logger = logging.getLogger(__name__)

MODEL_ENV_KEYS = {
    "qwen-8b": "LOCAL_AI_MODEL_8B",
    "qwen-14b": "LOCAL_AI_MODEL_14B",
    "qwen-24b": "LOCAL_AI_MODEL_24B",
}


class LlmPort:
    """Interface the worker calls. LiteLLMAdapter and ScriptedLLM both implement it."""

    def complete(self, prompt: str, model_name: str) -> Completion:
        """
        Ask the model for one turn.

        :param prompt: assembled prompt as str. Example: "Foundation\\nShell\\nTask"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: one turn as Completion. Example: Completion(final_answer="Found ch-01.md")
        """
        raise NotImplementedError


class ScriptedLLM(LlmPort):
    """
    Test double. Integration tests inject this. It is not used by the running server.

    Raises ScriptExhausted instead of returning an empty turn when the script ends.
    """

    def __init__(self, script: tuple[Completion, ...]) -> None:
        """
        Store the ordered turns.

        :param script: turns as tuple[Completion, ...]. Example: (Completion(final_answer="done"),)
        """
        # Remember the script and an index at 0.
        self._script = script
        self._index = 0

    def complete(self, prompt: str, model_name: str) -> Completion:
        """
        Return the next scripted turn.

        :param prompt: assembled prompt as str. Example: "Foundation\\nTask"
        :param model_name: logical size as str. Example: "qwen-8b"
        :return: next completion as Completion. Example: Completion(content="List the files.")
        :raises ValueError: when prompt is not str. Example: None
        :raises ScriptExhausted: when the script has no turn left. Example: a third call on a two-turn script
        """
        logger.info(
            "Starting ScriptedLLM.complete index=%s model_name=%s",
            self._index,
            model_name,
        )
        self._require_turn_request(prompt, model_name)
        turn = self._next_scripted_turn(model_name)
        logger.info(
            "ScriptedLLM.complete returned index=%s model_name=%s",
            self._index - 1,
            model_name,
        )
        return turn

    def _require_turn_request(self, prompt: str, model_name: str) -> None:
        """
        Reject a turn request that is not text.

        :param prompt: assembled prompt as str. Example: "Foundation\\nTask"
        :param model_name: logical size as str. Example: "qwen-8b"
        :return: None. Example: None
        :raises ValueError: when prompt or model_name is not a string. Example: None
        """
        if not isinstance(prompt, str):
            raise ValueError("prompt must be str. Example: 'Foundation\\nTask'")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty str. Example: 'qwen-8b'")

    def _next_scripted_turn(self, model_name: str) -> Completion:
        """
        Take the turn at the current index, or raise when the script is finished.

        :param model_name: logical size as str. Example: "qwen-8b"
        :return: next completion as Completion. Example: Completion(content="List the files.")
        :raises ScriptExhausted: when the index is past the script. Example: index 2 on a two-turn script
        """
        if self._index >= len(self._script):
            logger.info(
                "ScriptedLLM.complete exhausted index=%s model_name=%s",
                self._index,
                model_name,
            )
            raise ScriptExhausted(
                f"script has no turn left at index={self._index} model_name={model_name}"
            )
        turn = self._script[self._index]
        if not isinstance(turn, Completion):
            raise TypeError(
                "script turn must be Completion. Example: Completion(content='List the files.')"
            )
        self._index += 1
        return turn


class LiteLLMAdapter(LlmPort):
    """Calls LiteLLM against Ollama. The worker receives this in the running server."""

    def __init__(self, complete_fn, env: Mapping[str, str], ollama_host: str) -> None:
        """
        Store the LiteLLM function and the tag map.

        :param complete_fn: LiteLLM completion callable. Example: litellm.completion
        :param env: process environment as Mapping[str, str]. Example: {"LOCAL_AI_MODEL_14B": "qwen2.5:14b"}
        :param ollama_host: Ollama base URL as str. Example: "http://127.0.0.1:11434"
        """
        self._complete_fn = complete_fn
        self._env = env
        self._ollama_host = ollama_host

    def complete(self, prompt: str, model_name: str) -> Completion:
        """
        Call LiteLLM once and map the response onto Completion.

        :param prompt: assembled prompt as str. Example: "Foundation\\nTask"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: one turn as Completion. Example: Completion(thinking="Reading ch-04.md.")
        """
        # resolve_model_tag(model_name, env) to the Ollama tag.
        # Call complete_fn with that tag, the prompt, and the shell tool schema, against ollama_host.
        # Map thinking, content, and tool calls onto Completion.
        # Log model_name and token counts when Ollama returns them. Do not log the prompt.
        raise NotImplementedError


def split_note(completion: Completion) -> str:
    """
    Choose the reasoning note text for the blackboard.

    :param completion: one model turn as Completion. Example: Completion(thinking="The claim is in the last section.")
    :return: note text as str. Example: "The claim is in the last section."
    :raises TypeError: when completion is not a Completion. Example: None
    """
    logger.info("Starting split_note")
    _require_completion(completion)
    note = _note_text(completion)
    source = "thinking" if completion.thinking else "content"
    logger.info("split_note source=%s note_len=%s", source, len(note))
    return note


def _require_completion(completion: Completion) -> None:
    """
    Reject a value that is not one model turn.

    :param completion: candidate turn as Completion. Example: Completion(thinking="The claim is in the last section.")
    :return: None. Example: None
    :raises TypeError: when completion is not a Completion. Example: None
    """
    if not isinstance(completion, Completion):
        raise TypeError(
            "completion must be Completion. Example: Completion(thinking='The claim is in the last section.')"
        )


def _note_text(completion: Completion) -> str:
    """
    Prefer thinking, then content.

    :param completion: one model turn as Completion. Example: Completion(content="List the chapter files.")
    :return: note text as str. Example: "List the chapter files."
    """
    if completion.thinking:
        return completion.thinking
    return completion.content


def resolve_model_tag(model_name: str, env: Mapping[str, str]) -> str:
    """
    Map a logical model name to the Ollama tag in the environment.

    :param model_name: logical size as str. Example: "qwen-14b"
    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_MODEL_14B": "qwen2.5:14b"}
    :return: Ollama tag as str. Example: "qwen2.5:14b"
    :raises KeyError: when the matching LOCAL_AI_MODEL_* variable is missing. Example: "qwen-14b" with an empty env
    """
    logger.info("Starting resolve_model_tag model_name=%s", model_name)
    env_key = _env_key_for_model(model_name)
    tag = _tag_from_env(env, env_key)
    logger.info("resolve_model_tag model_name=%s env_key=%s", model_name, env_key)
    return tag


def _env_key_for_model(model_name: str) -> str:
    """
    Choose the environment variable for a logical model name.

    :param model_name: logical size as str. Example: "qwen-14b"
    :return: environment variable name as str. Example: "LOCAL_AI_MODEL_14B"
    :raises KeyError: when model_name is not one of the three sizes. Example: "gpt-4"
    :raises ValueError: when model_name is empty. Example: ""
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty str. Example: 'qwen-14b'")
    try:
        return MODEL_ENV_KEYS[model_name]
    except KeyError:
        raise KeyError(model_name) from None


def _tag_from_env(env: Mapping[str, str], env_key: str) -> str:
    """
    Read one Ollama tag. The value is not logged.

    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_MODEL_14B": "qwen2.5:14b"}
    :param env_key: variable name as str. Example: "LOCAL_AI_MODEL_14B"
    :return: Ollama tag as str. Example: "qwen2.5:14b"
    :raises KeyError: when env_key is absent. Example: "LOCAL_AI_MODEL_14B"
    """
    if not isinstance(env, Mapping):
        raise TypeError("env must be a mapping. Example: {'LOCAL_AI_MODEL_14B': 'qwen2.5:14b'}")
    try:
        tag = env[env_key]
    except KeyError:
        raise KeyError(env_key) from None
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"{env_key} must be a non-empty str. Example: 'qwen2.5:14b'")
    return tag
