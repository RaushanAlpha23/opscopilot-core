"""The public facade.

Replaces the previous `OpsCopilotClient`. The important behavioural changes:

* The pipeline is explicit Python, not a graph re-invoked from its entry point
  with nodes that no-op on a status flag. The old `approve_and_resume` re-ran
  triage and remediation as no-ops to reach the ticket node, which meant the
  resume path's correctness depended on every node remembering to check
  `status == "approved"` and bail. Adding a node was a latent bug.
* Resuming is a real resume: `provide_evidence` re-runs only diagnosis onward.
* Every dependency is injectable, so tests and alternate deployments swap
  providers without patching module globals.
* Nothing is constructed at import time; a failure to reach Redis or Qdrant
  surfaces where it happens, with an actionable message.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Settings, load_settings
from .embeddings.base import Embedder
from .embeddings.registry import build_embedder
from .events import Event, EventBus
from .exceptions import InvalidTransitionError, StateNotFoundError, TicketingError
from .integrations.base import TicketProvider
from .llm.base import LLMProvider
from .llm.registry import build_llm
from .models import (
    Diagnosis,
    EvidenceExchange,
    IncidentState,
    IncidentStatus,
    Ticket,
)
from .nodes import (
    build_retrieval_query,
    run_diagnosis,
    run_evidence_request,
    run_remediation,
    run_triage,
    run_vision,
)
from .retrieval.base import VectorStore
from .retrieval.indexer import Indexer
from .retrieval.registry import build_vector_store
from .state.base import StateStore
from .state.registry import build_state_store

logger = logging.getLogger("opscopilot")


class OpsCopilot:
    """Entry point for indexing a codebase and running incidents through triage."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLMProvider | None = None,
        vision_llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        state_store: StateStore | None = None,
        ticket_provider: TicketProvider | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.events = EventBus()

        self._llm = llm
        self._vision_llm = vision_llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._state_store = state_store
        self._ticket_provider = ticket_provider
        self._indexer: Indexer | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> OpsCopilot:
        """Build from environment variables / .env, with keyword overrides."""
        return cls(load_settings(**overrides))

    # Components are built lazily and cached. Lazily, because constructing a
    # SentenceTransformer downloads a model and connecting to Redis opens a
    # socket — neither should happen just because someone imported the module
    # or wanted to call `index_repository`, which needs no LLM at all.

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = build_llm(self.settings.llm, self.settings)
        return self._llm

    @property
    def vision_llm(self) -> LLMProvider:
        if self._vision_llm is None:
            spec = self.settings.effective_vision_llm
            if spec == self.settings.llm:
                self._vision_llm = self.llm
            else:
                self._vision_llm = build_llm(spec, self.settings)
        return self._vision_llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings.embeddings, self.settings)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = build_vector_store(self.settings.vector_store, self.settings)
        return self._vector_store

    @property
    def state_store(self) -> StateStore:
        if self._state_store is None:
            self._state_store = build_state_store(self.settings.state_store, self.settings)
        return self._state_store

    @property
    def indexer(self) -> Indexer:
        if self._indexer is None:
            self._indexer = Indexer(self.vector_store, self.embedder)
        return self._indexer

    @property
    def ticket_provider(self) -> TicketProvider | None:
        has_github = self.settings.github_token and self.settings.github_repo
        if self._ticket_provider is None and has_github:
            from .integrations.github import GitHubTicketProvider

            self._ticket_provider = GitHubTicketProvider(
                token=self.settings.github_token,
                repo=self.settings.github_repo,
                labels=self.settings.github_labels,
                max_retries=self.settings.max_retries,
            )
        return self._ticket_provider

    def on_event(self, handler: Callable[[Event], None]) -> Callable[[], None]:
        """Subscribe to progress events. Returns an unsubscribe callable."""
        return self.events.subscribe(handler)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_repository(
        self,
        path: str | Path,
        *,
        collection: str | None = None,
        extensions: set[str] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Chunk, embed and store every supported source file under `path`."""
        return self.indexer.index_repository(
            path, collection or self.settings.collection_code, extensions, progress
        )

    def index_schema(self, sql: str, *, source_name: str = "schema.sql") -> int:
        """Index database DDL so the diagnosis step can reason about the schema."""
        return self.indexer.index_text(self.settings.collection_schema, sql, source_name)

    def search(self, query: str, *, top_k: int = 5, collection: str | None = None):
        """Retrieve code chunks directly — useful for debugging retrieval quality."""
        return self.indexer.search(collection or self.settings.collection_code, query, top_k)

    # ------------------------------------------------------------------
    # Incident lifecycle
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        user_note: str | None = None,
        screenshot_path: str | Path | None = None,
        api_response: dict[str, Any] | None = None,
        logs: str | None = None,
        incident_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IncidentState:
        """Run a new incident through the automated pipeline.

        Returns once the run parks at a human decision point (awaiting
        evidence or awaiting approval) or fails. The returned status says
        which, so callers branch on `state.status` rather than on a fixed
        string the old SDK returned unconditionally.
        """
        state = IncidentState(
            incident_id=incident_id or str(uuid.uuid4()),
            user_note=user_note,
            screenshot_path=str(screenshot_path) if screenshot_path else None,
            api_response=api_response,
            logs=logs,
            metadata=metadata or {},
        )
        self.events.emit(state.incident_id, "incident.received")
        self._save(state)

        try:
            self._run_triage(state)
            self._run_vision(state)
            self._diagnose(state)
            self._route(state)
        except Exception as exc:
            logger.exception("opscopilot: incident %s failed", state.incident_id)
            state.status = IncidentStatus.failed
            state.error = f"{type(exc).__name__}: {exc}"
            self.events.emit(state.incident_id, "incident.failed", error=state.error)

        self._save(state)
        return state

    def provide_evidence(self, incident_id: str, response: str) -> IncidentState:
        """Answer the pending clarifying question and resume from diagnosis."""
        state = self._require(incident_id)
        if state.status is not IncidentStatus.awaiting_evidence:
            raise InvalidTransitionError(
                f"Incident '{incident_id}' is '{state.status.value}', not awaiting evidence."
            )

        state.evidence[-1].response = response
        self.events.emit(incident_id, "evidence.received")

        try:
            # Re-diagnose with the new evidence folded in, then re-apply the
            # same gate the automated path uses — a real resume, not a replay.
            self._diagnose(state)
            self._route(state)
        except Exception as exc:
            logger.exception("opscopilot: resume-after-evidence failed for %s", incident_id)
            state.status = IncidentStatus.failed
            state.error = f"{type(exc).__name__}: {exc}"

        self._save(state)
        return state

    def approve(self, incident_id: str) -> IncidentState:
        """Approve the proposed remediation and file a ticket."""
        return self._decide(incident_id, "approved")

    def reject(self, incident_id: str, reason: str | None = None) -> IncidentState:
        """Reject the proposed remediation. Ends the run with no ticket filed."""
        return self._decide(incident_id, "rejected", reason)

    def get(self, incident_id: str) -> IncidentState:
        return self._require(incident_id)

    def list_incidents(self, limit: int = 100) -> list[str]:
        return self.state_store.list_ids(limit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save(self, state: IncidentState) -> None:
        self.state_store.save(state)

    def _require(self, incident_id: str) -> IncidentState:
        state = self.state_store.load(incident_id)
        if state is None:
            raise StateNotFoundError(incident_id)
        return state

    def _run_triage(self, state: IncidentState) -> None:
        self.events.emit(state.incident_id, "node.started", node="triage")
        state.triage = run_triage(self.llm, state.user_note, retries=self.settings.max_retries)
        state.status = IncidentStatus.diagnosing
        self.events.emit(
            state.incident_id, "node.finished", node="triage", **state.triage.to_dict()
        )

    def _run_vision(self, state: IncidentState) -> None:
        self.events.emit(state.incident_id, "node.started", node="vision")
        state.vision = run_vision(
            self.vision_llm,
            state.screenshot_path,
            api_response=state.api_response,
            user_note=state.user_note,
            logs=state.logs,
            retries=self.settings.max_retries,
        )
        self.events.emit(
            state.incident_id, "node.finished", node="vision",
            symptom=state.vision.visible_symptom,
        )

    def _diagnose(self, state: IncidentState) -> None:
        assert state.vision is not None
        self.events.emit(state.incident_id, "node.started", node="diagnosis")

        query = build_retrieval_query(state.vision, state.user_note)
        # Answered evidence joins the retrieval query too, not just the prompt:
        # the human's answer often contains the exact identifier that makes the
        # right chunk retrievable on the second pass.
        answered = [e.response for e in state.evidence if e.response]
        if answered:
            query = f"{query} {' '.join(answered)}"

        state.retrieved_code = self.indexer.search(
            self.settings.collection_code, query, self.settings.code_top_k
        )
        state.retrieved_schema = self.indexer.search(
            self.settings.collection_schema, query, self.settings.schema_top_k
        )

        state.diagnosis = run_diagnosis(
            self.llm,
            state.vision,
            state.retrieved_code,
            state.retrieved_schema,
            evidence=state.evidence,
            max_chunk_chars=self.settings.max_chunk_chars,
            retries=self.settings.max_retries,
        )
        state.attempts += 1
        self.events.emit(
            state.incident_id, "node.finished", node="diagnosis",
            confidence=state.diagnosis.confidence,
            suspected_file=state.diagnosis.suspected_file,
        )

    def _confidence_sufficient(self, diagnosis: Diagnosis | None) -> bool:
        # Fail closed: a missing diagnosis is not confidence.
        if diagnosis is None:
            return False
        return diagnosis.confidence >= self.settings.confidence_threshold

    def _route(self, state: IncidentState) -> None:
        """The confidence gate: remediate, or ask for more evidence."""
        if self._confidence_sufficient(state.diagnosis):
            self._remediate(state)
            return

        # Bounded loop. The old design could in principle ping-pong forever;
        # after max_evidence_rounds we proceed to remediation anyway and let
        # the human judge it, rather than stranding the incident.
        if len(state.evidence) >= self.settings.max_evidence_rounds:
            self.events.emit(
                state.incident_id, "evidence.exhausted", rounds=len(state.evidence)
            )
            self._remediate(state)
            return

        self._request_evidence(state)

    def _request_evidence(self, state: IncidentState) -> None:
        assert state.diagnosis is not None
        self.events.emit(state.incident_id, "node.started", node="request_evidence")
        exchange: EvidenceExchange = run_evidence_request(
            self.llm, state.diagnosis, state.retrieved_code, state.evidence
        )
        state.evidence.append(exchange)
        state.status = IncidentStatus.awaiting_evidence
        self.events.emit(
            state.incident_id, "awaiting_evidence", question=exchange.question
        )

    def _remediate(self, state: IncidentState) -> None:
        assert state.diagnosis is not None
        self.events.emit(state.incident_id, "node.started", node="remediation")
        state.remediation = run_remediation(
            self.llm, state.diagnosis, state.retrieved_code, retries=self.settings.max_retries
        )
        state.status = IncidentStatus.awaiting_approval
        self.events.emit(
            state.incident_id, "awaiting_approval",
            risk_level=state.remediation.risk_level.value,
        )

    def _decide(self, incident_id: str, decision: str, reason: str | None = None) -> IncidentState:
        state = self._require(incident_id)
        if state.status is not IncidentStatus.awaiting_approval:
            raise InvalidTransitionError(
                f"Incident '{incident_id}' is '{state.status.value}', not awaiting approval."
            )

        state.approval_decision = decision
        if reason:
            state.metadata["rejection_reason"] = reason

        if decision == "rejected":
            state.status = IncidentStatus.rejected
            self.events.emit(incident_id, "rejected", reason=reason)
            self._save(state)
            return state

        provider = self.ticket_provider
        if provider is None:
            # No tracker configured is not a failure: the approved remediation
            # is still recorded and retrievable. Surfacing it as `ticketed`
            # with an explanatory error beats raising and losing the decision.
            state.ticket = Ticket(
                error="No ticket provider configured (set github_token/github_repo)."
            )
            state.status = IncidentStatus.ticketed
        else:
            try:
                state.ticket = provider.create(state)
                self.events.emit(incident_id, "ticket.created", url=state.ticket.url)
            except TicketingError as exc:
                state.ticket = Ticket(error=str(exc))
                self.events.emit(incident_id, "ticket.failed", error=str(exc))
            state.status = IncidentStatus.ticketed

        self._save(state)
        return state
