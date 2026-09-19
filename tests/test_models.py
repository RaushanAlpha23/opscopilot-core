from __future__ import annotations

from opscopilot.models import (
    Diagnosis,
    EvidenceExchange,
    IncidentState,
    IncidentStatus,
    Severity,
    Triage,
)


def test_state_roundtrips_through_json_losslessly():
    state = IncidentState(
        incident_id="abc",
        status=IncidentStatus.awaiting_approval,
        user_note="broken",
        triage=Triage(Severity.critical, "checkout"),
        diagnosis=Diagnosis("summary", 0.9, "a.py", "f"),
        evidence=[EvidenceExchange("q?", "a.")],
    )
    assert IncidentState.from_dict(state.to_dict()).to_dict() == state.to_dict()


def test_terminal_and_waiting_status_flags():
    assert IncidentStatus.ticketed.is_terminal
    assert IncidentStatus.rejected.is_terminal
    assert not IncidentStatus.diagnosing.is_terminal
    assert IncidentStatus.awaiting_evidence.is_waiting_on_human
    assert not IncidentStatus.ticketed.is_waiting_on_human


def test_pending_question_only_when_unanswered():
    state = IncidentState("x", evidence=[EvidenceExchange("what url?")])
    assert state.pending_question == "what url?"
    state.evidence[-1].response = "/api/cart"
    assert state.pending_question is None
