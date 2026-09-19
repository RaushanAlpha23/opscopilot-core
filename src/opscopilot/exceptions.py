"""Exception hierarchy.

Every error raised by this package descends from OpsCopilotError, so callers
can `except OpsCopilotError` once instead of catching provider-specific
exceptions leaking out of qdrant-client / redis-py / requests.
"""

from __future__ import annotations


class OpsCopilotError(Exception):
    """Base class for every error raised by opscopilot."""


class ConfigurationError(OpsCopilotError):
    """Missing or invalid configuration (absent API key, bad provider spec)."""


class MissingDependencyError(ConfigurationError):
    """An optional extra is required for this code path but isn't installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"'{package}' is required for this feature but is not installed. "
            f"Install it with: pip install 'opscopilot-core[{extra}]'"
        )
        self.package = package
        self.extra = extra


class ProviderError(OpsCopilotError):
    """The upstream LLM/embedding provider failed or returned something unusable."""


class InvalidJSONResponse(ProviderError):
    """A model was asked for JSON and did not produce parseable JSON."""

    def __init__(self, raw: str) -> None:
        preview = raw[:500] + ("..." if len(raw) > 500 else "")
        super().__init__(f"Model did not return valid JSON. Raw response:\n{preview}")
        self.raw = raw


class RetrievalError(OpsCopilotError):
    """Vector store operation failed."""


class StateNotFoundError(OpsCopilotError):
    """No persisted state for this incident (never created, or TTL expired)."""

    def __init__(self, incident_id: str) -> None:
        super().__init__(
            f"No stored state for incident '{incident_id}'. "
            "It may have expired, already been resolved, or never existed."
        )
        self.incident_id = incident_id


class InvalidTransitionError(OpsCopilotError):
    """An operation was attempted that the incident's current status doesn't allow."""


class TicketingError(OpsCopilotError):
    """Base for issue-tracker integration failures."""


class GitHubAuthError(TicketingError):
    """Token missing, invalid, expired, or lacking the required scope."""


class GitHubRateLimitError(TicketingError):
    """GitHub's API rate limit was hit."""


class GitHubAPIError(TicketingError):
    """Any other non-2xx response from the GitHub API."""
