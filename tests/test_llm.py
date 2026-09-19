from __future__ import annotations

import pytest

from opscopilot.exceptions import ConfigurationError, InvalidJSONResponse
from opscopilot.llm import FakeLLM, complete_json, extract_json, parse_spec


def test_extracts_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_strips_markdown_fences_models_add_anyway():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_recovers_json_wrapped_in_prose():
    assert extract_json('Sure! Here it is:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_rejects_a_bare_json_list_since_callers_expect_an_object():
    with pytest.raises(InvalidJSONResponse):
        extract_json("[1, 2, 3]")


def test_retries_once_on_malformed_json_and_succeeds():
    llm = FakeLLM(["not json at all", '{"ok": true}'])
    assert complete_json(llm, "give me json", retries=1) == {"ok": True}
    assert len(llm.calls) == 2
    # The retry must tell the model what went wrong, or it just repeats itself.
    assert "not valid JSON" in llm.calls[1]["prompt"]


def test_gives_up_after_exhausting_retries():
    llm = FakeLLM(["nope", "still nope"])
    with pytest.raises(InvalidJSONResponse):
        complete_json(llm, "prompt", retries=1)


def test_provider_spec_parsing():
    assert parse_spec("mistral:mistral-small-latest") == ("mistral", "mistral-small-latest")
    with pytest.raises(ConfigurationError, match="Expected"):
        parse_spec("gpt-4o")
    with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
        parse_spec("acme:model-1")
