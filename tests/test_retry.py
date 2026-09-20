from __future__ import annotations

from types import SimpleNamespace

import pytest

from opscopilot.config import Settings
from opscopilot.exceptions import ConfigurationError, RateLimitError
from opscopilot.llm import build_llm, parse_spec
from opscopilot.llm.retry import RetryPolicy, is_retryable


class _HTTPError(Exception):
    """Minimal stand-in for httpx.HTTPStatusError: carries `.response.status_code/.headers`."""

    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__("boom")
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def _http_error(status: int, headers: dict[str, str] | None = None) -> _HTTPError:
    return _HTTPError(status, headers)


def _policy(sleeps: list[float], **kw) -> RetryPolicy:
    return RetryPolicy(sleep=sleeps.append, **kw)


def test_retries_a_429_then_succeeds():
    sleeps: list[float] = []
    outcomes = [_http_error(429), _http_error(429), "ok"]

    def fn():
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert _policy(sleeps).call("mistral", fn) == "ok"
    assert len(sleeps) == 2


def test_honours_retry_after_header():
    sleeps: list[float] = []
    outcomes = [_http_error(429, {"retry-after": "7"}), "ok"]

    def fn():
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    _policy(sleeps).call("mistral", fn)
    assert 7 <= sleeps[0] < 8.01  # 7s hint plus <1s jitter


def test_raises_rate_limit_error_when_attempts_run_out():
    def fn():
        raise _http_error(429)

    with pytest.raises(RateLimitError, match="rate limit exceeded after 3 attempts"):
        _policy([], max_attempts=3).call("mistral", fn)


def test_does_not_retry_auth_errors():
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        raise _http_error(401)

    with pytest.raises(_HTTPError):
        _policy([]).call("mistral", fn)
    assert calls == 1


def test_out_of_quota_is_not_retried():
    exc = _http_error(429)
    exc.code = "insufficient_quota"  # type: ignore[attr-defined]
    assert not is_retryable(exc)


def test_min_interval_spaces_out_calls():
    sleeps: list[float] = []
    now = [100.0]
    policy = RetryPolicy(min_interval=2.0, sleep=sleeps.append, clock=lambda: now[0])
    policy.call("x", lambda: 1)
    policy.call("x", lambda: 2)  # clock has not advanced, so it must wait the full gap
    assert sleeps == [2.0]


def test_gemini_is_a_known_provider_and_needs_a_key():
    assert parse_spec("gemini:gemini-flash-latest") == ("gemini", "gemini-flash-latest")
    # Pass the key explicitly: constructor values beat environment variables, so this test
    # does not depend on whether OPSCOPILOT_GEMINI_API_KEY happens to be set on the machine.
    with pytest.raises(ConfigurationError, match="OPSCOPILOT_GEMINI_API_KEY"):
        build_llm("gemini:gemini-flash-latest", Settings(_env_file=None, gemini_api_key=""))


class _ApiError(Exception):
    """Stands in for an SDK error that carries a status `code` and the provider's body text."""

    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"{code} RESOURCE_EXHAUSTED. {body}")
        self.code = code


_PER_MINUTE = (
    "{'error': {'message': 'quota', 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, {'retryDelay': '5s'}]}}"
)
_PER_DAY = (
    "{'error': {'message': 'You exceeded your current quota. limit: 20', 'details': "
    "[{'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, "
    "{'retryDelay': '34s'}]}}"
)


def test_reads_retry_delay_from_the_error_body():
    sleeps: list[float] = []
    outcomes: list = [_ApiError(429, _PER_MINUTE), "ok"]

    def fn():
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert _policy(sleeps).call("gemini", fn) == "ok"
    assert 5 <= sleeps[0] < 6.01  # Google said 5s, plus <1s jitter (not the 2s default)


def test_daily_quota_fails_fast_without_retrying():
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        raise _ApiError(429, _PER_DAY)

    sleeps: list[float] = []
    with pytest.raises(RateLimitError) as info:
        _policy(sleeps).call("gemini", fn)
    assert calls == 1 and sleeps == []
    assert info.value.quota_exhausted
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(info.value)


def test_zero_quota_fails_fast():
    def fn():
        raise _ApiError(429, "Quota exceeded for metric x, limit: 0, model: gemini-pro")

    with pytest.raises(RateLimitError) as info:
        _policy([]).call("gemini", fn)
    assert info.value.quota_exhausted


def test_a_wait_longer_than_max_delay_fails_fast():
    def fn():
        raise _ApiError(429, "quota. Please retry in 3000s.")

    with pytest.raises(RateLimitError, match="asked us to wait 3000s"):
        _policy([], max_delay=60).call("gemini", fn)


def test_retry_log_line_names_the_quota(caplog):
    outcomes: list = [_ApiError(429, _PER_MINUTE), "ok"]

    def fn():
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with caplog.at_level("WARNING", logger="opscopilot"):
        _policy([]).call("gemini", fn)
    assert "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" in caplog.text


def test_gemini_call_disables_afc_to_avoid_log_noise():
    pytest.importorskip("google.genai")
    from opscopilot.llm.gemini import GeminiProvider

    provider = GeminiProvider(api_key="fake-key")
    seen = {}

    def fake_generate(*, model, contents, config):
        seen["afc_disabled"] = config.automatic_function_calling.disable
        return SimpleNamespace(text=" hello ")

    provider._client.models.generate_content = fake_generate
    assert provider.complete("hi", temperature=0.0) == "hello"
    assert seen["afc_disabled"] is True
