"""Model port. Worker depends on LlmPort. The dispatcher does not."""

from collections.abc import Mapping

from local_ai.contracts import Completion, ScriptExhausted


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
        :raises ScriptExhausted: when the script has no turn left. Example: a third call on a two-turn script
        """
        # If the index is past the script, raise ScriptExhausted.
        # Otherwise return that turn and advance the index.
        # Log the index and model_name at INFO. Do not log the prompt body.
        raise NotImplementedError


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
    """
    # When thinking is non-empty, return it.
    # Otherwise return content, which the worker marks as a note.
    # An empty turn returns an empty string. The caller skips the append.
    raise NotImplementedError


def resolve_model_tag(model_name: str, env: Mapping[str, str]) -> str:
    """
    Map a logical model name to the Ollama tag in the environment.

    :param model_name: logical size as str. Example: "qwen-14b"
    :param env: environment as Mapping[str, str]. Example: {"LOCAL_AI_MODEL_14B": "qwen2.5:14b"}
    :return: Ollama tag as str. Example: "qwen2.5:14b"
    :raises KeyError: when the matching LOCAL_AI_MODEL_* variable is missing. Example: "qwen-14b" with an empty env
    """
    # qwen-8b reads LOCAL_AI_MODEL_8B. qwen-14b reads LOCAL_AI_MODEL_14B. qwen-24b reads LOCAL_AI_MODEL_24B.
    # Log the logical name at INFO. Do not log secrets. There is no API key.
    raise NotImplementedError
