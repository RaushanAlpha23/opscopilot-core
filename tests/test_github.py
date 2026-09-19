"""GitHub error-mapping tests, with `requests` stubbed out.

Worth testing without a network: the status-code mapping is real logic, and
403 in particular means two completely different things depending on the
rate-limit header.
"""

from __future__ import annotations

import sys
import types

import pytest

from opscopilot.exceptions import GitHubAPIError, GitHubAuthError, GitHubRateLimitError
from opscopilot.integrations.github import GitHubTicketProvider
from opscopilot.models import IncidentState


class FakeResponse:
    def __init__(self, status_code=201, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json


@pytest.fixture
def fake_requests(monkeypatch):
    module = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    module.RequestException = RequestException
    module.responses = []
    module.calls = []

    def post(url, headers=None, json=None, timeout=None):
        module.calls.append({"url": url, "json": json})
        result = module.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    module.post = post
    monkeypatch.setitem(sys.modules, "requests", module)
    monkeypatch.setattr("time.sleep", lambda s: None)  # no real backoff in tests
    return module


@pytest.fixture
def provider():
    return GitHubTicketProvider(token="t", repo="owner/name")


def test_repo_must_be_owner_slash_name():
    with pytest.raises(GitHubAPIError, match="owner/name"):
        GitHubTicketProvider(token="t", repo="justname")


def test_missing_token_is_rejected_up_front():
    with pytest.raises(GitHubAuthError):
        GitHubTicketProvider(token="", repo="o/n")


def test_successful_creation_returns_url_and_number(fake_requests, provider):
    fake_requests.responses = [
        FakeResponse(201, {"html_url": "https://github.com/o/n/issues/5", "number": 5})
    ]
    ticket = provider.create(IncidentState("inc-1", user_note="x"))

    assert ticket.number == 5
    assert fake_requests.calls[0]["json"]["labels"] == ["ai-triaged", "bug", "needs-review"]


def test_401_maps_to_auth_error(fake_requests, provider):
    fake_requests.responses = [FakeResponse(401)]
    with pytest.raises(GitHubAuthError, match="invalid or expired"):
        provider.create(IncidentState("i"))


def test_403_with_exhausted_quota_is_a_rate_limit_not_a_scope_problem(fake_requests, provider):
    fake_requests.responses = [FakeResponse(403, headers={"X-RateLimit-Remaining": "0"})]
    with pytest.raises(GitHubRateLimitError):
        provider.create(IncidentState("i"))


def test_403_with_quota_remaining_is_a_scope_problem(fake_requests, provider):
    fake_requests.responses = [FakeResponse(403, headers={"X-RateLimit-Remaining": "4999"})]
    with pytest.raises(GitHubAuthError, match="scope"):
        provider.create(IncidentState("i"))


def test_404_names_the_repo(fake_requests, provider):
    fake_requests.responses = [FakeResponse(404)]
    with pytest.raises(GitHubAPIError, match="owner/name"):
        provider.create(IncidentState("i"))


def test_server_error_is_retried_then_succeeds(fake_requests, provider):
    fake_requests.responses = [
        FakeResponse(502, text="bad gateway"),
        FakeResponse(201, {"html_url": "u", "number": 1}),
    ]
    assert provider.create(IncidentState("i")).number == 1
    assert len(fake_requests.calls) == 2


def test_network_error_is_retried_then_gives_up(fake_requests, provider):
    fake_requests.responses = [
        fake_requests.RequestException("conn reset"),
        fake_requests.RequestException("conn reset"),
        fake_requests.RequestException("conn reset"),
    ]
    with pytest.raises(GitHubAPIError, match="Network error"):
        provider.create(IncidentState("i"))
    assert len(fake_requests.calls) == 3  # initial attempt + 2 retries
