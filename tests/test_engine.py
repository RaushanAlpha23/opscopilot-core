from __future__ import annotations

import pytest
from conftest import CONFIDENT_DIAGNOSIS, UNSURE_DIAGNOSIS, scripted_llm

from opscopilot.exceptions import InvalidTransitionError, StateNotFoundError
from opscopilot.models import IncidentStatus, RiskLevel, Severity, Ticket


def test_confident_run_parks_at_approval(engine):
    cop = engine()
    state = cop.submit(user_note="cart total is wrong")

    assert state.status is IncidentStatus.awaiting_approval
    assert state.triage.severity is Severity.high
    assert state.diagnosis.suspected_file == "cart.py"
    assert state.remediation.risk_level is RiskLevel.low
    assert state.attempts == 1


def test_low_confidence_asks_one_specific_question_instead_of_remediating(engine):
    cop = engine(llm=scripted_llm(diagnosis=UNSURE_DIAGNOSIS))
    state = cop.submit(user_note="something is off")

    assert state.status is IncidentStatus.awaiting_evidence
    assert state.pending_question
    assert state.remediation is None


def test_evidence_resumes_from_diagnosis_rather_than_replaying_the_run(engine):
    # First diagnosis is unsure; after the human answers, the second is confident.
    llm = scripted_llm(diagnoses=[UNSURE_DIAGNOSIS, CONFIDENT_DIAGNOSIS])
    cop = engine(llm=llm)

    parked = cop.submit(user_note="something is off")
    assert parked.status is IncidentStatus.awaiting_evidence

    resumed = cop.provide_evidence(parked.incident_id, "GET /api/cart returns total: null")

    assert resumed.status is IncidentStatus.awaiting_approval
    assert resumed.evidence[0].response == "GET /api/cart returns total: null"
    assert resumed.attempts == 2
    # Triage ran exactly once: resuming must not re-run completed nodes.
    assert sum("triage assistant" in c["prompt"] for c in llm.calls) == 1


def test_one_evidence_round_means_exactly_one_question(engine):
    """max_evidence_rounds is a budget on questions ASKED, not on resumes."""
    cop = engine(llm=scripted_llm(diagnosis=UNSURE_DIAGNOSIS), max_evidence_rounds=1)

    state = cop.submit(user_note="vague report")
    assert state.status is IncidentStatus.awaiting_evidence
    assert len(state.evidence) == 1

    state = cop.provide_evidence(state.incident_id, "here is a log")
    # Still unsure, but the budget is spent, so it remediates and lets the
    # human judge rather than stranding the incident in an endless loop.
    assert state.status is IncidentStatus.awaiting_approval
    assert len(state.evidence) == 1


def test_evidence_loop_is_bounded_and_remediates_anyway(engine):
    cop = engine(llm=scripted_llm(diagnosis=UNSURE_DIAGNOSIS), max_evidence_rounds=2)

    state = cop.submit(user_note="vague report")
    state = cop.provide_evidence(state.incident_id, "here is a log")
    assert state.status is IncidentStatus.awaiting_evidence  # second question

    state = cop.provide_evidence(state.incident_id, "and here is another")
    assert state.status is IncidentStatus.awaiting_approval
    assert len(state.evidence) == 2
    assert all(e.response for e in state.evidence)


def test_the_human_answer_is_fed_back_into_the_retrieval_query(engine):
    llm = scripted_llm(diagnoses=[UNSURE_DIAGNOSIS, CONFIDENT_DIAGNOSIS])
    cop = engine(llm=llm)
    state = cop.submit(user_note="something is off")

    queries = []
    original = cop.indexer.search
    cop.indexer.search = lambda c, q, k=5: (queries.append(q), original(c, q, k))[1]

    cop.provide_evidence(state.incident_id, "the quantity field is always 1")
    assert any("quantity field is always 1" in q for q in queries)


def test_approval_files_a_ticket(engine):
    class StubTracker:
        name = "stub"

        def create(self, state):
            return Ticket(url="https://github.com/o/r/issues/7", number=7)

    cop = engine()
    cop._ticket_provider = StubTracker()

    state = cop.submit(user_note="cart total is wrong")
    approved = cop.approve(state.incident_id)

    assert approved.status is IncidentStatus.ticketed
    assert approved.ticket.number == 7


def test_a_tracker_failure_is_recorded_not_raised(engine):
    from opscopilot.exceptions import GitHubRateLimitError

    class FailingTracker:
        name = "stub"

        def create(self, state):
            raise GitHubRateLimitError("rate limited")

    cop = engine()
    cop._ticket_provider = FailingTracker()

    state = cop.approve(cop.submit(user_note="cart total is wrong").incident_id)
    # The human's approval must not be lost because GitHub was briefly down.
    assert state.status is IncidentStatus.ticketed
    assert "rate limited" in state.ticket.error


def test_rejection_ends_the_run_without_a_ticket(engine):
    cop = engine()
    state = cop.submit(user_note="cart total is wrong")
    rejected = cop.reject(state.incident_id, reason="wrong file")

    assert rejected.status is IncidentStatus.rejected
    assert rejected.ticket is None
    assert rejected.metadata["rejection_reason"] == "wrong file"


def test_cannot_approve_an_incident_that_is_not_awaiting_approval(engine):
    cop = engine(llm=scripted_llm(diagnosis=UNSURE_DIAGNOSIS))
    state = cop.submit(user_note="vague")
    with pytest.raises(InvalidTransitionError, match="awaiting_evidence"):
        cop.approve(state.incident_id)


def test_cannot_approve_twice(engine):
    cop = engine()
    state = cop.submit(user_note="cart total is wrong")
    cop.approve(state.incident_id)
    with pytest.raises(InvalidTransitionError):
        cop.approve(state.incident_id)


def test_unknown_incident_raises_a_named_error(engine):
    with pytest.raises(StateNotFoundError, match="nope"):
        engine().get("nope")


def test_a_node_failure_marks_the_incident_failed_instead_of_propagating(engine):
    from opscopilot.llm import FakeLLM

    cop = engine(llm=FakeLLM(handler=lambda p: (_ for _ in ()).throw(RuntimeError("boom"))))
    state = cop.submit(user_note="cart total is wrong")

    assert state.status is IncidentStatus.failed
    assert "boom" in state.error
    # And the failure is persisted, so an operator can still inspect it.
    assert cop.get(state.incident_id).status is IncidentStatus.failed


def test_state_survives_a_reload_from_the_store(engine):
    cop = engine()
    state = cop.submit(user_note="cart total is wrong")
    reloaded = cop.get(state.incident_id)

    assert reloaded.diagnosis.summary == state.diagnosis.summary
    assert reloaded.remediation.suggested_fix == state.remediation.suggested_fix


def test_mutating_a_returned_state_does_not_corrupt_the_store(engine):
    cop = engine()
    state = cop.submit(user_note="cart total is wrong")
    state.diagnosis.summary = "tampered"

    assert cop.get(state.incident_id).diagnosis.summary != "tampered"


def test_events_are_emitted_for_each_stage(engine):
    cop = engine()
    seen = []
    cop.on_event(lambda e: seen.append(e.type))
    cop.submit(user_note="cart total is wrong")

    assert seen[0] == "incident.received"
    assert seen[-1] == "awaiting_approval"


def test_a_broken_event_handler_cannot_break_the_run(engine):
    cop = engine()
    cop.on_event(lambda e: (_ for _ in ()).throw(RuntimeError("subscriber exploded")))
    assert cop.submit(user_note="cart total is wrong").status is IncidentStatus.awaiting_approval


def test_incidents_can_be_listed(engine):
    cop = engine()
    first = cop.submit(user_note="cart total is wrong")
    second = cop.submit(user_note="cart total is wrong")
    assert {first.incident_id, second.incident_id} <= set(cop.list_incidents())
