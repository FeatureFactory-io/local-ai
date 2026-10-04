"""LiteLLMAdapter.complete maps one injected LiteLLM response. No network."""

import json
import logging
from collections.abc import Mapping

import pytest

from local_ai.contracts import Completion, ToolCall
from local_ai.llm.port import LiteLLMAdapter


PROMPT = "Foundation\nTask SECRET-PROMPT-BODY"
HOST = "http://127.0.0.1:11434"
TAG = "SECRET-TAG-VALUE"


def _adapter(
    complete_fn: object,
    env: Mapping[str, str] | None = None,
    ollama_host: str = HOST,
) -> LiteLLMAdapter:
    """Build an adapter whose LiteLLM function is the test stub."""
    return LiteLLMAdapter(
        complete_fn=complete_fn,
        env={"LOCAL_AI_MODEL_14B": TAG} if env is None else env,
        ollama_host=ollama_host,
    )


def _shell_call(command: str, *, as_dict: bool = False) -> dict[str, object]:
    """One shell tool call. Arguments are JSON text unless as_dict is set."""
    arguments: object = {"command": command} if as_dict else json.dumps({"command": command})
    return {"function": {"name": "shell", "arguments": arguments}}


def _response(
    *,
    content: str | None = "",
    reasoning: str | None = "",
    thinking: str | None = None,
    tool_calls: list[object] | None = None,
    usage: Mapping[str, int] | None = None,
    include_usage: bool = True,
) -> dict[str, object]:
    """A LiteLLM-shaped dict. reasoning_content is the mapped thinking field."""
    message: dict[str, object] = {"content": content, "reasoning_content": reasoning}
    if thinking is not None:
        message["thinking"] = thinking
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, object] = {"choices": [{"message": message}]}
    if include_usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15} if usage is None else dict(usage)
    return body


def test_complete_with_thinking_content_and_tool_call_maps_completion() -> None:
    """reasoning_content, content, and the shell command land on Completion."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(
            content="List the chapter files.",
            reasoning="Reading ch-04.md.",
            tool_calls=[_shell_call("ls *.md")],
        )

    completion = _adapter(complete_fn).complete(PROMPT, "qwen-14b")
    assert completion == Completion(
        content="List the chapter files.",
        thinking="Reading ch-04.md.",
        tool_calls=(ToolCall(name="shell", command="ls *.md"),),
        final_answer="",
    )


def test_complete_with_resolved_tag_calls_litellm_with_shell_schema_and_host() -> None:
    """The stub receives the Ollama tag, the prompt, one shell tool, and the host."""
    captured: dict[str, object] = {}

    def complete_fn(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _response()

    _adapter(complete_fn, ollama_host="http://127.0.0.1:11435").complete(PROMPT, "qwen-14b")
    assert set(captured) == {"model", "messages", "tools", "api_base"}
    assert captured["model"] == f"ollama_chat/{TAG}"
    assert captured["messages"] == [{"role": "user", "content": PROMPT}]
    assert captured["api_base"] == "http://127.0.0.1:11435"
    tools = captured["tools"]
    assert isinstance(tools, list) and len(tools) == 1
    function = tools[0]["function"]
    assert function["name"] == "shell"
    assert function["parameters"]["properties"]["command"]["type"] == "string"
    assert function["parameters"]["required"] == ["command"]


def test_complete_with_usage_logs_model_name_and_token_counts_not_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INFO names the logical model and token counts. The prompt and tag stay off the log."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(
            reasoning="SECRET-THINKING",
            tool_calls=[_shell_call("SECRET-COMMAND")],
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )

    _adapter(complete_fn).complete(PROMPT, "qwen-14b")
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "model_name=qwen-14b" in messages
    assert "prompt_tokens=12" in messages
    assert "completion_tokens=3" in messages
    assert "total_tokens=15" in messages
    assert "SECRET-PROMPT-BODY" not in messages
    assert TAG not in messages
    assert "SECRET-THINKING" not in messages
    assert "SECRET-COMMAND" not in messages


def test_complete_without_usage_logs_model_name_and_not_token_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A response with no usage still logs the logical model and skips token counts."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(include_usage=False)

    assert _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b") == Completion()
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "model_name=qwen-14b" in messages
    assert "usage=absent" in messages
    assert "prompt_tokens" not in messages


def test_complete_with_zero_token_counts_logs_them(caplog: pytest.LogCaptureFixture) -> None:
    """Zero is a count the response included, so it is logged."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "prompt_tokens=0" in messages
    assert "completion_tokens=0" in messages
    assert "total_tokens=0" in messages


def test_complete_with_empty_usage_object_skips_token_counts(caplog: pytest.LogCaptureFixture) -> None:
    """A usage object with no counts is the same as a missing usage for the log."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")

    def complete_fn(**kwargs: object) -> dict[str, object]:
        body = _response(include_usage=False)
        body["usage"] = {}
        return body

    _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "usage=absent" in messages
    assert "prompt_tokens" not in messages


def test_complete_with_missing_env_key_raises_and_does_not_call() -> None:
    """A missing LOCAL_AI_MODEL_* variable raises before the LiteLLM function runs."""
    called = False

    def complete_fn(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _response()

    with pytest.raises(KeyError):
        _adapter(complete_fn, env={}).complete("Foundation\nTask", "qwen-14b")
    assert called is False


def test_complete_with_non_string_prompt_raises_type_error() -> None:
    """A prompt that is not text is rejected before the LiteLLM function runs."""
    called = False

    def complete_fn(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _response()

    with pytest.raises(TypeError):
        _adapter(complete_fn).complete(None, "qwen-14b")  # type: ignore[arg-type]
    assert called is False


def test_complete_with_empty_model_name_raises_value_error() -> None:
    """An empty logical model name is rejected before the LiteLLM function runs."""
    called = False

    def complete_fn(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _response()

    with pytest.raises(ValueError):
        _adapter(complete_fn).complete("Foundation\nTask", "")
    assert called is False


def test_complete_with_thinking_field_maps_thinking_when_reasoning_is_empty() -> None:
    """A raw thinking field is used when reasoning_content is empty."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(reasoning="", thinking="The claim is in the last section.")

    completion = _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    assert completion.thinking == "The claim is in the last section."
    assert completion.content == ""


def test_complete_with_dict_arguments_maps_command() -> None:
    """Arguments that are already an object still yield the command string."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(tool_calls=[_shell_call("ls *.md", as_dict=True)])

    completion = _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    assert completion.tool_calls == (ToolCall(name="shell", command="ls *.md"),)


def test_complete_with_two_tool_calls_keeps_order() -> None:
    """Each tool call is mapped, and the order matches the response."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(tool_calls=[_shell_call("ls *.md"), _shell_call("wc -l ch-01.md")])

    completion = _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    assert completion.tool_calls == (
        ToolCall(name="shell", command="ls *.md"),
        ToolCall(name="shell", command="wc -l ch-01.md"),
    )


def test_complete_with_no_tool_calls_returns_empty_tuple() -> None:
    """A turn without tool_calls has an empty tuple, not None."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(content="No command this turn.", reasoning="")

    completion = _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    assert completion.tool_calls == ()
    assert completion.content == "No command this turn."
    assert completion.final_answer == ""


def test_complete_with_null_content_becomes_empty_string() -> None:
    """A null content field is an empty string on Completion."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(content=None, reasoning=None)

    assert _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b") == Completion()


def test_complete_with_attribute_response_maps_message_fields() -> None:
    """Objects shaped like a LiteLLM message map the same fields as a dict."""

    class Function:
        def __init__(self) -> None:
            self.name = "shell"
            self.arguments = '{"command": "ls *.md"}'

    class Tool:
        def __init__(self) -> None:
            self.function = Function()

    class Message:
        def __init__(self) -> None:
            self.content = "List the chapter files."
            self.reasoning_content = "Reading ch-04.md."
            self.tool_calls = [Tool()]

    class Choice:
        def __init__(self) -> None:
            self.message = Message()

    class Response:
        def __init__(self) -> None:
            self.choices = [Choice()]
            self.usage = None

    def complete_fn(**kwargs: object) -> Response:
        return Response()

    completion = _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")
    assert completion == Completion(
        content="List the chapter files.",
        thinking="Reading ch-04.md.",
        tool_calls=(ToolCall(name="shell", command="ls *.md"),),
    )


def test_complete_with_no_choices_raises_value_error() -> None:
    """A response without choices fails before a Completion is returned."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    with pytest.raises(ValueError):
        _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")


def test_complete_with_invalid_tool_json_raises_value_error() -> None:
    """Tool arguments that are not JSON fail with ValueError."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(tool_calls=[{"function": {"name": "shell", "arguments": "ls *.md"}}])

    with pytest.raises(ValueError):
        _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")


def test_complete_with_tool_call_missing_command_raises_value_error() -> None:
    """A function payload without command is not a shell call."""

    def complete_fn(**kwargs: object) -> dict[str, object]:
        return _response(tool_calls=[{"function": {"name": "shell", "arguments": "{}"}}])

    with pytest.raises(ValueError):
        _adapter(complete_fn).complete("Foundation\nTask", "qwen-14b")


def test_complete_with_non_callable_complete_fn_raises_type_error() -> None:
    """The adapter rejects a complete_fn that cannot be called."""
    with pytest.raises(TypeError):
        _adapter(complete_fn=None).complete("Foundation\nTask", "qwen-14b")


def test_complete_with_empty_host_raises_value_error() -> None:
    """An empty Ollama host is rejected before the LiteLLM function runs."""
    called = False

    def complete_fn(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return _response()

    with pytest.raises(ValueError):
        _adapter(complete_fn, ollama_host="").complete("Foundation\nTask", "qwen-14b")
    assert called is False
