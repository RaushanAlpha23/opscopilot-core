from __future__ import annotations

from opscopilot.integrations import issue_body, issue_title
from opscopilot.models import (
    CodeChunk,
    Diagnosis,
    EvidenceExchange,
    IncidentState,
    Remediation,
    RetrievedChunk,
    RiskLevel,
    Severity,
    Triage,
    VisionAnalysis,
)


def full_state():
    return IncidentState(
        incident_id="inc-1",
        user_note="cart total wrong",
        api_response={"total": None},
        triage=Triage(Severity.high, "cart"),
        vision=VisionAnalysis("total renders as NaN", api_response_notes="total is null"),
        diagnosis=Diagnosis(
            "get_total_price ignores quantity.", 0.85, "cart.py", "get_total_price"
        ),
        remediation=Remediation("Multiply by quantity.", RiskLevel.low, patch="- a\n+ b"),
        evidence=[EvidenceExchange("which endpoint?", "/api/cart")],
        retrieved_code=[
            RetrievedChunk(CodeChunk("1", "code", "cart.py", symbol_name="get_total_price"), 0.9)
        ],
    )


def test_title_is_short_and_labelled():
    title = issue_title(full_state())
    assert title.startswith("[Auto-triaged]")
    assert "(cart)" in title
    assert "\n" not in title


def test_body_contains_every_section_a_reviewer_needs():
    body = issue_body(full_state())
    for expected in [
        "get_total_price ignores quantity.",
        "`cart.py`",
        "**Confidence:** 0.85",
        "Multiply by quantity.",
        "**Risk level:** low",
        "total renders as NaN",
        "which endpoint?",
        "/api/cart",
        "inc-1",
        "```diff",
    ]:
        assert expected in body, f"missing from issue body: {expected}"


def test_body_survives_a_sparse_state():
    body = issue_body(IncidentState("inc-2"))
    assert "no diagnosis recorded" in body
    assert "inc-2" in body


def test_unanswered_evidence_is_not_presented_as_gathered():
    state = full_state()
    state.evidence = [EvidenceExchange("unanswered?")]
    assert "Additional evidence gathered" not in issue_body(state)
