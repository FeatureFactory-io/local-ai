"""Leaf behavior for validate_model. Other contract stubs stay uncalled."""

import logging

import pytest

from local_ai.contracts import MODEL_NAMES, InvalidModel, validate_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_validate_model_accepts_the_three_qwen_sizes(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """qwen-8b, qwen-14b, and qwen-24b are the only legal names."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    assert validate_model(name) == name
    assert name in _info_text(caplog)


def test_validate_model_qwen_70b_raises(caplog: pytest.LogCaptureFixture) -> None:
    """A larger tag than the three sizes is rejected."""
    caplog.set_level(logging.INFO, logger="local_ai.contracts")
    with pytest.raises(InvalidModel):
        validate_model("qwen-70b")
    assert "qwen-70b" in _info_text(caplog)


@pytest.mark.parametrize("name", ["", "Qwen-14b", "llama-8b"])
def test_validate_model_unknown_name_raises(name: str) -> None:
    """Empty, differently cased, and non-Qwen names are not model tags."""
    with pytest.raises(InvalidModel):
        validate_model(name)


def _info_text(caplog: pytest.LogCaptureFixture) -> str:
    """Join INFO messages. Example: 'Accepted model name=qwen-14b'."""
    return " ".join(record.message for record in caplog.records if record.levelno == logging.INFO)
