"""Model port. Worker depends on LlmPort. The dispatcher does not."""

import json
import logging
from collections.abc import Mapping

from local_ai.contracts import Completion, ScriptExhausted, ToolCall


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
        :raises TypeError: when prompt is not str. Example: None
        :raises ValueError: when model_name is empty or the response has no message. Example: ""
        :raises KeyError: when the matching LOCAL_AI_MODEL_* variable is missing. Example: "LOCAL_AI_MODEL_14B"
        """
        logger.info("Starting LiteLLMAdapter.complete model_name=%s", model_name)
        _require_completion_request(prompt, model_name, self._ollama_host, self._complete_fn)
        response = self._call_litellm(prompt, model_name)
        completion = _completion_from_response(response)
        _log_token_counts(model_name, response)
        logger.info(
            "LiteLLMAdapter.complete returned model_name=%s tool_calls=%s",
            model_name,
            len(completion.tool_calls),
        )
        return completion

    def _call_litellm(self, prompt: str, model_name: str) -> object:
        """
        Resolve the Ollama tag and call the injected LiteLLM function once.

        :param prompt: assembled prompt as str. Example: "Foundation\\nTask"
        :param model_name: logical size as str. Example: "qwen-14b"
        :return: LiteLLM response as object. Example: {"choices": [{"message": {"content": ""}}]}
        :raises KeyError: when the matching LOCAL_AI_MODEL_* variable is missing. Example: "LOCAL_AI_MODEL_14B"
        """
        tag = resolve_model_tag(model_name, self._env)
        logger.info("LiteLLMAdapter.complete calling model_name=%s", model_name)
        return self._complete_fn(
            model=_ollama_chat_model(tag),
            messages=[{"role": "user", "content": prompt}],
            tools=[_shell_tool_schema()],
            api_base=self._ollama_host,
        )


def _require_completion_request(
    prompt: str,
    model_name: str,
    ollama_host: str,
    complete_fn: object,
) -> None:
    """
    Reject a turn request that cannot be sent.

    :param prompt: assembled prompt as str. Example: "Foundation\\nTask"
    :param model_name: logical size as str. Example: "qwen-14b"
    :param ollama_host: Ollama base URL as str. Example: "http://127.0.0.1:11434"
    :param complete_fn: LiteLLM completion callable. Example: litellm.completion
    :return: None. Example: None
    :raises TypeError: when prompt is not str or complete_fn is not callable. Example: None
    :raises ValueError: when model_name or ollama_host is empty. Example: ""
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be str. Example: 'Foundation\\nTask'")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty str. Example: 'qwen-14b'")
    if not isinstance(ollama_host, str) or not ollama_host:
        raise ValueError("ollama_host must be a non-empty str. Example: 'http://127.0.0.1:11434'")
    if not callable(complete_fn):
        raise TypeError("complete_fn must be callable. Example: litellm.completion")


def _ollama_chat_model(tag: str) -> str:
    """
    Prefix an Ollama tag so LiteLLM selects the chat provider.

    A bare tag such as qwen3:14b is rejected with "LLM Provider NOT provided".

    :param tag: Ollama tag as str. Example: "qwen3:14b"
    :return: LiteLLM model as str. Example: "ollama_chat/qwen3:14b"
    :raises ValueError: when tag is empty. Example: ""
    """
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("tag must be a non-empty str. Example: 'qwen3:14b'")
    if "/" in tag:
        return tag
    return f"ollama_chat/{tag}"


def _shell_tool_schema() -> dict[str, object]:
    """
    OpenAI-style schema for the one shell tool.

    :return: tool schema as dict[str, object]. Example: {"type": "function", "function": {"name": "shell"}}
    """
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run one shell command inside the job access grant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command. Example: ls *.md",
                    }
                },
                "required": ["command"],
            },
        },
    }


def _completion_from_response(response: object) -> Completion:
    """
    Map one LiteLLM response onto a Completion.

    :param response: LiteLLM response as object. Example: {"choices": [{"message": {"content": ""}}]}
    :return: one turn as Completion. Example: Completion(thinking="Reading ch-04.md.")
    :raises ValueError: when the response has no message. Example: {}
    """
    message = _message_from_response(response)
    return Completion(
        content=_text_field(message, "content"),
        thinking=_thinking_text(message),
        tool_calls=_tool_calls_from_message(message),
        final_answer=_text_field(message, "content"),
    )


def _message_from_response(response: object) -> object:
    """
    Read the first choice message.

    :param response: LiteLLM response as object. Example: {"choices": [{"message": {"content": ""}}]}
    :return: message as object. Example: {"content": "List the chapter files."}
    :raises ValueError: when choices or the message is missing. Example: {}
    """
    choices = _field(response, "choices")
    if not isinstance(choices, list | tuple) or len(choices) == 0:
        raise ValueError(
            "LiteLLM response must include choices. Example: {'choices': [{'message': {'content': ''}}]}"
        )
    message = _field(choices[0], "message")
    if message is None:
        raise ValueError(
            "LiteLLM choice must include a message. Example: {'content': 'List the chapter files.'}"
        )
    return message


def _thinking_text(message: object) -> str:
    """
    Prefer reasoning_content, then a raw thinking field.

    :param message: LiteLLM message as object. Example: {"reasoning_content": "Reading ch-04.md."}
    :return: thinking text as str. Example: "Reading ch-04.md."
    :raises TypeError: when the field is not str. Example: 1
    """
    reasoning = _text_field(message, "reasoning_content")
    if reasoning:
        return reasoning
    return _text_field(message, "thinking")


def _text_field(value: object, name: str) -> str:
    """
    Read a text field. Missing and null become an empty string.

    :param value: response node as object. Example: {"content": "List the chapter files."}
    :param name: field name as str. Example: "content"
    :return: field text as str. Example: "List the chapter files."
    :raises TypeError: when the field is not str. Example: 1
    """
    raw = _field(value, name)
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be str. Example: 'Reading ch-04.md.'")
    return raw


def _tool_calls_from_message(message: object) -> tuple[ToolCall, ...]:
    """
    Map message tool calls onto ToolCall values.

    :param message: LiteLLM message as object. Example: {"tool_calls": []}
    :return: tool calls as tuple[ToolCall, ...]. Example: (ToolCall("shell", "ls *.md"),)
    :raises TypeError: when tool_calls is not a list. Example: "shell"
    """
    raw_calls = _field(message, "tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list | tuple):
        raise TypeError("tool_calls must be a list. Example: []")
    return tuple(_tool_call_from_item(item) for item in raw_calls)


def _tool_call_from_item(item: object) -> ToolCall:
    """
    Read one function name and its command argument.

    :param item: tool call as object. Example: {"function": {"name": "shell", "arguments": "{\"command\": \"ls *.md\"}"}}
    :return: tool call as ToolCall. Example: ToolCall(name="shell", command="ls *.md")
    :raises ValueError: when function, name, or command is missing. Example: {}
    """
    function = _field(item, "function")
    if function is None:
        raise ValueError(
            "tool call must include function. Example: {'name': 'shell', 'arguments': '{\"command\": \"ls *.md\"}'}"
        )
    name = _text_field(function, "name")
    if not name:
        raise ValueError("tool call name must be a non-empty str. Example: 'shell'")
    command = _command_from_arguments(_field(function, "arguments"))
    return ToolCall(name=name, command=command)


def _command_from_arguments(arguments: object) -> str:
    """
    Read the command string from JSON text or a mapping.

    :param arguments: function arguments as object. Example: '{"command": "ls *.md"}'
    :return: command as str. Example: "ls *.md"
    :raises ValueError: when command is missing. Example: "{}"
    :raises TypeError: when arguments are not an object. Example: "ls *.md"
    """
    parsed = _arguments_object(arguments)
    command = parsed.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError("tool arguments must include command as str. Example: 'ls *.md'")
    return command


def _arguments_object(arguments: object) -> Mapping[str, object]:
    """
    Normalize tool arguments to a mapping.

    :param arguments: function arguments as object. Example: '{"command": "ls *.md"}'
    :return: arguments as Mapping[str, object]. Example: {"command": "ls *.md"}
    :raises TypeError: when arguments are not JSON or a mapping. Example: 1
    :raises ValueError: when a string is not JSON. Example: "ls *.md"
    """
    if isinstance(arguments, str):
        parsed = _load_arguments_json(arguments)
    elif isinstance(arguments, Mapping):
        parsed = arguments
    else:
        raise TypeError("tool arguments must be JSON or a mapping. Example: '{\"command\": \"ls *.md\"}'")
    if not isinstance(parsed, Mapping):
        raise TypeError("tool arguments must be an object. Example: {'command': 'ls *.md'}")
    return parsed


def _load_arguments_json(arguments: str) -> object:
    """
    Parse a tool-arguments JSON string.

    :param arguments: JSON text as str. Example: '{"command": "ls *.md"}'
    :return: parsed JSON as object. Example: {"command": "ls *.md"}
    :raises ValueError: when arguments is not JSON. Example: "ls *.md"
    """
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments must be JSON. Example: '{\"command\": \"ls *.md\"}'") from exc


def _log_token_counts(model_name: str, response: object) -> None:
    """
    Log model_name and token counts when the response includes them.

    :param model_name: logical size as str. Example: "qwen-14b"
    :param response: LiteLLM response as object. Example: {"usage": {"prompt_tokens": 12}}
    :return: None. Example: None
    """
    counts = _token_counts(response)
    if counts is None:
        logger.info("LiteLLMAdapter.complete model_name=%s usage=absent", model_name)
        return
    prompt_tokens, completion_tokens, total_tokens = counts
    logger.info(
        "LiteLLMAdapter.complete model_name=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def _token_counts(response: object) -> tuple[object, object, object] | None:
    """
    Read prompt, completion, and total tokens. None when the response has no counts.

    :param response: LiteLLM response as object. Example: {"usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
    :return: three counts as tuple[object, object, object] | None. Example: (12, 3, 15)
    """
    usage = _field(response, "usage")
    if usage is None:
        return None
    prompt_tokens = _field(usage, "prompt_tokens")
    completion_tokens = _field(usage, "completion_tokens")
    total_tokens = _field(usage, "total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return (prompt_tokens, completion_tokens, total_tokens)


def _field(value: object, name: str) -> object:
    """
    Read one field from a mapping or an object.

    :param value: response node as object. Example: {"content": "List the chapter files."}
    :param name: field name as str. Example: "content"
    :return: field value as object. Example: "List the chapter files."
    """
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


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
