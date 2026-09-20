"""opscopilot-core — an agentic SRE engine.

Screenshot-grounded bug triage, code-aware RAG diagnosis, and human-in-the-loop
remediation, with every backend (LLM, embeddings, vector store, state store,
issue tracker) pluggable.

Quickstart, with no API keys and no infrastructure::

    from opscopilot import OpsCopilot
    from opscopilot.llm import FakeLLM

    cop = OpsCopilot.from_env(llm="fake:demo")
    cop.index_repository("./my_app")
    state = cop.submit(user_note="Cart total shows NaN after removing an item")
    print(state.status, state.diagnosis.summary)
"""

from __future__ import annotations

from .config import Settings, load_settings
from .engine import OpsCopilot
from .events import Event, EventBus
from .exceptions import (
    ConfigurationError,
    GitHubAPIError,
    GitHubAuthError,
    GitHubRateLimitError,
    InvalidJSONResponse,
    InvalidTransitionError,
    MissingDependencyError,
    OpsCopilotError,
    ProviderError,
    RateLimitError,
    RetrievalError,
    StateNotFoundError,
    TicketingError,
)
from .models import (
    CodeChunk,
    Diagnosis,
    EvidenceExchange,
    IncidentState,
    IncidentStatus,
    Remediation,
    RetrievedChunk,
    RiskLevel,
    Severity,
    Ticket,
    Triage,
    VisionAnalysis,
)

__version__ = "0.2.3"

__all__ = [
    "__version__",
    "OpsCopilot",
    "Settings",
    "load_settings",
    "Event",
    "EventBus",
    # models
    "IncidentState",
    "IncidentStatus",
    "Severity",
    "RiskLevel",
    "Triage",
    "Diagnosis",
    "Remediation",
    "VisionAnalysis",
    "EvidenceExchange",
    "CodeChunk",
    "RetrievedChunk",
    "Ticket",
    # exceptions
    "OpsCopilotError",
    "ConfigurationError",
    "MissingDependencyError",
    "ProviderError",
    "RateLimitError",
    "InvalidJSONResponse",
    "RetrievalError",
    "StateNotFoundError",
    "InvalidTransitionError",
    "TicketingError",
    "GitHubAuthError",
    "GitHubRateLimitError",
    "GitHubAPIError",
]
