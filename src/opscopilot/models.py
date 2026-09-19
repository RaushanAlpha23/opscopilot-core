"""Domain types.

These are the package's public data contracts. They are dataclasses rather
than bare dicts so that:

  * consumers get autocomplete and mypy checking (the old SDK returned
    untyped dicts, which meant every caller re-guessed the key names);
  * `search()` can return structured hits with scores and metadata instead of
    a pre-formatted prompt string, which threw the metadata away;
  * state can round-trip through JSON (Redis, a file, a queue) losslessly via
    `to_dict()` / `from_dict()`.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Severity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class IncidentStatus(str, enum.Enum):
    """Lifecycle of one incident run.

    Terminal states are `rejected`, `ticketed` and `failed`; everything else
    means the pipeline is either mid-flight or parked waiting on a human.
    """

    received = "received"
    diagnosing = "diagnosing"
    awaiting_evidence = "awaiting_evidence"
    awaiting_approval = "awaiting_approval"
    rejected = "rejected"
    ticketed = "ticketed"
    failed = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (IncidentStatus.rejected, IncidentStatus.ticketed, IncidentStatus.failed)

    @property
    def is_waiting_on_human(self) -> bool:
        return self in (IncidentStatus.awaiting_evidence, IncidentStatus.awaiting_approval)


@dataclass(slots=True)
class CodeChunk:
    """One indexable unit of source code — ideally a whole function or class."""

    id: str
    text: str
    file_path: str
    language: str = "unknown"
    chunk_type: str = "window"
    symbol_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeChunk:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class RetrievedChunk:
    """A `CodeChunk` plus the similarity score it was retrieved with."""

    chunk: CodeChunk
    score: float

    def to_prompt(self, max_chars: int = 1200) -> str:
        """Render this hit for inclusion in an LLM prompt, truncated to a budget."""
        body = self.chunk.text
        if len(body) > max_chars:
            body = body[:max_chars] + "\n... (truncated)"
        symbol = self.chunk.symbol_name or "(unnamed)"
        return (
            f"--- {self.chunk.file_path} :: {symbol} (relevance {self.score:.3f}) ---\n{body}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": self.score}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedChunk:
        return cls(chunk=CodeChunk.from_dict(data["chunk"]), score=data["score"])


@dataclass(slots=True)
class VisionAnalysis:
    """Structured read of a screenshot, optionally cross-checked against an API response."""

    visible_symptom: str
    likely_layer: str = "unclear"  # "frontend" | "backend" | "unclear"
    api_response_consistent: bool | None = None
    api_response_notes: str | None = None
    raw_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionAnalysis:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class Triage:
    severity: Severity = Severity.medium
    category: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity.value, "category": self.category}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Triage:
        return cls(severity=Severity(data.get("severity", "medium")),
                   category=data.get("category", "unknown"))


@dataclass(slots=True)
class Diagnosis:
    summary: str
    confidence: float = 0.0
    suspected_file: str | None = None
    suspected_symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Diagnosis:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class Remediation:
    suggested_fix: str
    risk_level: RiskLevel = RiskLevel.medium
    patch: str | None = None  # optional unified diff, when the model can produce one

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggested_fix": self.suggested_fix,
            "risk_level": self.risk_level.value,
            "patch": self.patch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Remediation:
        return cls(
            suggested_fix=data["suggested_fix"],
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            patch=data.get("patch"),
        )


@dataclass(slots=True)
class EvidenceExchange:
    """One round of the clarifying-question loop."""

    question: str
    response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceExchange:
        return cls(question=data["question"], response=data.get("response"))


@dataclass(slots=True)
class Ticket:
    url: str | None = None
    number: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ticket:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class IncidentState:
    """The single object threaded through every node and persisted between turns.

    Replaces the old loose TypedDict. Every field has a default except
    `incident_id`, because at submission time only the inputs are known and
    everything else is filled in as the run progresses.
    """

    incident_id: str
    status: IncidentStatus = IncidentStatus.received

    # --- inputs ---
    user_note: str | None = None
    screenshot_path: str | None = None
    api_response: dict[str, Any] | None = None
    logs: str | None = None

    # --- produced by the pipeline ---
    triage: Triage | None = None
    vision: VisionAnalysis | None = None
    retrieved_code: list[RetrievedChunk] = field(default_factory=list)
    retrieved_schema: list[RetrievedChunk] = field(default_factory=list)
    diagnosis: Diagnosis | None = None
    evidence: list[EvidenceExchange] = field(default_factory=list)
    remediation: Remediation | None = None
    approval_decision: str | None = None  # "approved" | "rejected"
    ticket: Ticket | None = None

    # --- bookkeeping ---
    attempts: int = 0  # how many diagnosis passes this incident has had
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending_question(self) -> str | None:
        """The unanswered clarifying question, if the run is parked on one."""
        if self.evidence and self.evidence[-1].response is None:
            return self.evidence[-1].question
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "status": self.status.value,
            "user_note": self.user_note,
            "screenshot_path": self.screenshot_path,
            "api_response": self.api_response,
            "logs": self.logs,
            "triage": self.triage.to_dict() if self.triage else None,
            "vision": self.vision.to_dict() if self.vision else None,
            "retrieved_code": [r.to_dict() for r in self.retrieved_code],
            "retrieved_schema": [r.to_dict() for r in self.retrieved_schema],
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "evidence": [e.to_dict() for e in self.evidence],
            "remediation": self.remediation.to_dict() if self.remediation else None,
            "approval_decision": self.approval_decision,
            "ticket": self.ticket.to_dict() if self.ticket else None,
            "attempts": self.attempts,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IncidentState:
        return cls(
            incident_id=data["incident_id"],
            status=IncidentStatus(data.get("status", "received")),
            user_note=data.get("user_note"),
            screenshot_path=data.get("screenshot_path"),
            api_response=data.get("api_response"),
            logs=data.get("logs"),
            triage=Triage.from_dict(data["triage"]) if data.get("triage") else None,
            vision=VisionAnalysis.from_dict(data["vision"]) if data.get("vision") else None,
            retrieved_code=[RetrievedChunk.from_dict(r) for r in data.get("retrieved_code", [])],
            retrieved_schema=[
                RetrievedChunk.from_dict(r) for r in data.get("retrieved_schema", [])
            ],
            diagnosis=Diagnosis.from_dict(data["diagnosis"]) if data.get("diagnosis") else None,
            evidence=[EvidenceExchange.from_dict(e) for e in data.get("evidence", [])],
            remediation=(
                Remediation.from_dict(data["remediation"]) if data.get("remediation") else None
            ),
            approval_decision=data.get("approval_decision"),
            ticket=Ticket.from_dict(data["ticket"]) if data.get("ticket") else None,
            attempts=data.get("attempts", 0),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
