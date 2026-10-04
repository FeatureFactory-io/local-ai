"""Leaf behavior for ScriptedLLM.complete, split_note, and resolve_model_tag."""

import logging

import pytest

from local_ai.contracts import Completion, ScriptExhausted
from local_ai.llm.port import LiteLLMAdapter, ScriptedLLM, split_note


def test_scripted_llm_complete_returns_turns_in_order_then_raises() -> None:
    """Each call returns the next Completion. The call after the last turn raises."""
    first = Completion(content="List the files.")
    second = Completion(final_answer="Found ch-01.md")
    llm = ScriptedLLM((first, second))
    assert llm.complete("Foundation\nTask", "qwen-8b") == first
    assert llm.complete("Foundation\nTask", "qwen-14b") == second
    with pytest.raises(ScriptExhausted):
        llm.complete("Foundation\nTask", "qwen-14b")


def test_scripted_llm_complete_on_empty_script_raises_not_empty_list() -> None:
    """Exhaustion raises ScriptExhausted. It does not return an empty list."""
    llm = ScriptedLLM(())
    outcome: object = None
    try:
        outcome = llm.complete("Foundation\nTask", "qwen-8b")
    except ScriptExhausted:
        outcome = "raised"
    assert outcome == "raised"


def test_scripted_llm_complete_logs_index_and_model_not_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """INFO carries the script index and model name. The prompt body stays off the log."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")
    prompt = "Foundation\nTask SECRET-PROMPT-BODY"
    ScriptedLLM((Completion(content="List the files."),)).complete(prompt, "qwen-8b")
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "index=0" in messages
    assert "qwen-8b" in messages
    assert "SECRET-PROMPT-BODY" not in messages


def test_scripted_llm_complete_rejects_a_non_string_prompt() -> None:
    """A prompt that is not text is rejected before a scripted turn is consumed."""
    llm = ScriptedLLM((Completion(content="List the files."),))
    with pytest.raises(ValueError):
        llm.complete(None, "qwen-8b")  # type: ignore[arg-type]
    assert llm.complete("Foundation\nTask", "qwen-8b").content == "List the files."


def test_litellm_adapter_complete_still_raises_not_implemented() -> None:
    """Plan A does not call LiteLLM. complete stays a stub until Plan B."""

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"live complete must not be called args={args} kwargs={kwargs}")

    adapter = LiteLLMAdapter(
        complete_fn=fail_if_called,
        env={"LOCAL_AI_MODEL_14B": "qwen2.5:14b"},
        ollama_host="http://127.0.0.1:11434",
    )
    with pytest.raises(NotImplementedError):
        adapter.complete("Foundation\nTask", "qwen-14b")


def test_split_note_returns_thinking_when_present() -> None:
    """A non-empty thinking stream is the blackboard note."""
    completion = Completion(
        thinking="The claim is in the last section.",
        content="List the chapter files.",
    )
    assert split_note(completion) == "The claim is in the last section."


def test_split_note_returns_content_when_thinking_is_empty() -> None:
    """Without thinking, the note is the assistant content."""
    completion = Completion(thinking="", content="List the chapter files.")
    assert split_note(completion) == "List the chapter files."


def test_split_note_returns_empty_string_when_turn_is_empty() -> None:
    """An empty turn returns an empty string so the caller can skip the append."""
    assert split_note(Completion()) == ""


def test_split_note_logs_source_and_not_the_note_text(caplog: pytest.LogCaptureFixture) -> None:
    """INFO names the source. The reasoning text stays off the log."""
    caplog.set_level(logging.INFO, logger="local_ai.llm.port")
    split_note(Completion(thinking="SECRET-NOTE-BODY"))
    messages = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)
    assert "source=thinking" in messages
    assert "SECRET-NOTE-BODY" not in messages
