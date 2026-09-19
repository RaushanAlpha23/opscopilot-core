from __future__ import annotations

import json

from opscopilot.llm import FakeLLM
from opscopilot.models import (
    CodeChunk,
    Diagnosis,
    RetrievedChunk,
    RiskLevel,
    Severity,
    VisionAnalysis,
)
from opscopilot.nodes import (
    build_retrieval_query,
    find_suspect_chunk,
    run_diagnosis,
    run_remediation,
    run_triage,
    run_vision,
)


def chunk(file_path="cart.py", symbol="get_total_price", score=0.9):
    return RetrievedChunk(
        chunk=CodeChunk(id=file_path, text="def f(): ...", file_path=file_path, symbol_name=symbol),
        score=score,
    )


def test_triage_defaults_when_no_note_and_never_calls_the_model():
    llm = FakeLLM([])
    result = run_triage(llm, None)
    assert result.severity is Severity.medium and result.category == "unknown"
    assert llm.calls == []


def test_triage_falls_back_on_an_unknown_severity_label():
    llm = FakeLLM([json.dumps({"severity": "catastrophic", "category": "Cart"})])
    result = run_triage(llm, "everything is on fire")
    assert result.severity is Severity.medium
    assert result.category == "cart"  # normalised to lowercase


def test_vision_works_without_a_screenshot():
    llm = FakeLLM([json.dumps({"visible_symptom": "blank price", "likely_layer": "frontend"})])
    result = run_vision(llm, None, user_note="price is blank")
    assert result.visible_symptom == "blank price"
    assert llm.calls[0]["has_image"] is False
    assert "No screenshot is available" in llm.calls[0]["prompt"]


def test_diagnosis_rejects_a_hallucinated_file_path():
    llm = FakeLLM([json.dumps({
        "diagnosis_summary": "it is broken",
        "suspected_file": "does/not/exist.py",
        "suspected_symbol": "ghost",
        "confidence_score": 0.9,
    })])
    result = run_diagnosis(llm, VisionAnalysis("symptom"), [chunk()], [])
    assert result.suspected_file is None  # not in the retrieved set, so discarded


def test_diagnosis_clamps_an_out_of_range_confidence():
    llm = FakeLLM([json.dumps({"diagnosis_summary": "x", "confidence_score": 42})])
    assert run_diagnosis(llm, VisionAnalysis("s"), [], []).confidence == 1.0


def test_retrieval_query_prefers_the_grounded_symptom():
    vision = VisionAnalysis("price field renders blank for product 4821",
                            api_response_notes="price is null")
    query = build_retrieval_query(vision, "prices broken")
    assert query.startswith("price field renders blank for product 4821")
    assert "price is null" in query


def test_suspect_chunk_falls_back_from_symbol_to_file_to_best_hit():
    exact = chunk(symbol="get_total_price")
    other = chunk(file_path="other.py", symbol="x", score=0.5)
    exact_diag = Diagnosis("s", 0.9, "cart.py", "get_total_price")
    assert find_suspect_chunk(exact_diag, [other, exact]) is exact
    assert find_suspect_chunk(Diagnosis("s", 0.9, "cart.py", "missing"), [other, exact]) is exact
    assert find_suspect_chunk(Diagnosis("s", 0.9, None, None), [other, exact]) is other
    assert find_suspect_chunk(Diagnosis("s", 0.9, "a.py", "b"), []) is None


def test_remediation_defaults_an_unknown_risk_level_to_medium():
    llm = FakeLLM([json.dumps({"suggested_fix": "do the thing", "risk_level": "apocalyptic"})])
    result = run_remediation(llm, Diagnosis("s", 0.9, "cart.py", "f"), [chunk()])
    assert result.risk_level is RiskLevel.medium
    assert result.patch is None
