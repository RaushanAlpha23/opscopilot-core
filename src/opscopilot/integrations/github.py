"""GitHub Issues integration."""

from __future__ import annotations

import time

from ..exceptions import (
    GitHubAPIError,
    GitHubAuthError,
    GitHubRateLimitError,
    MissingDependencyError,
)
from ..models import IncidentState, Ticket
from .formatting import issue_body, issue_title

API_BASE = "https://api.github.com"
DEFAULT_LABELS = ["ai-triaged", "bug", "needs-review"]


class GitHubTicketProvider:
    name = "github"

    def __init__(
        self,
        token: str,
        repo: str,
        labels: list[str] | None = None,
        timeout: int = 15,
        max_retries: int = 2,
    ) -> None:
        if not token:
            raise GitHubAuthError("A GitHub token is required. Set OPSCOPILOT_GITHUB_TOKEN.")
        if "/" not in repo:
            raise GitHubAPIError(f"repo must be in 'owner/name' form, got '{repo}'.")
        self.token = token
        self.repo = repo
        self.labels = labels or list(DEFAULT_LABELS)
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, url: str, payload: dict) -> dict:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("requests", "github") from exc

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                # Transient network failure: retry with backoff rather than
                # losing an approved remediation to a blip.
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise GitHubAPIError(f"Network error calling GitHub: {exc}") from exc

            if response.status_code == 401:
                raise GitHubAuthError("GitHub returned 401 — token invalid or expired.")
            if response.status_code == 403:
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    reset = response.headers.get("X-RateLimit-Reset", "unknown")
                    raise GitHubRateLimitError(f"GitHub rate limit exceeded. Resets at {reset}.")
                raise GitHubAuthError(
                    f"GitHub returned 403 — token likely lacks 'repo' scope for {self.repo}."
                )
            if response.status_code == 404:
                raise GitHubAPIError(
                    f"Repo '{self.repo}' not found or inaccessible with this token."
                )
            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue
            if not response.ok:
                raise GitHubAPIError(
                    f"GitHub API error {response.status_code}: {response.text[:500]}"
                )

            return response.json()

        raise GitHubAPIError(f"GitHub request failed after retries: {last_exc}")

    def create(self, state: IncidentState) -> Ticket:
        data = self._post(
            f"{API_BASE}/repos/{self.repo}/issues",
            {
                "title": issue_title(state),
                "body": issue_body(state),
                "labels": self.labels,
            },
        )
        return Ticket(url=data.get("html_url"), number=data.get("number"))
