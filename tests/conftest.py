from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from opscopilot import OpsCopilot  # noqa: E402
from opscopilot.llm import FakeLLM  # noqa: E402

TRIAGE = {"severity": "high", "category": "cart"}
VISION = {
    "visible_symptom": "cart total price renders as NaN",
    "likely_layer": "backend",
    "api_response_consistent": False,
    "api_response_notes": "total field is null in the response",
    "raw_confidence": 0.9,
}
CONFIDENT_DIAGNOSIS = {
    "diagnosis_summary": "get_total_price ignores item quantity.",
    "suspected_file": "cart.py",
    "suspected_symbol": "get_total_price",
    "confidence_score": 0.85,
}
UNSURE_DIAGNOSIS = {**CONFIDENT_DIAGNOSIS, "confidence_score": 0.2}
REMEDIATION = {
    "suggested_fix": "Multiply price by quantity inside get_total_price.",
    "risk_level": "low",
    "patch": None,
}


def scripted_llm(diagnosis: dict | None = None, diagnoses: list[dict] | None = None) -> FakeLLM:
    """An LLM that answers based on which prompt it is shown.

    Routing on prompt content rather than call order keeps these tests stable
    when the pipeline's node order changes.
    """
    queue = list(diagnoses or [])

    def handler(prompt: str) -> str:
        if "triage assistant" in prompt:
            return json.dumps(TRIAGE)
        if "analyzing a bug report" in prompt:
            return json.dumps(VISION)
        if "senior engineer diagnosing" in prompt:
            if queue:
                return json.dumps(queue.pop(0))
            return json.dumps(diagnosis or CONFIDENT_DIAGNOSIS)
        if "confidence too low" in prompt:
            return "Please paste the JSON response from GET /api/cart."
        if "writing a remediation" in prompt:
            return json.dumps(REMEDIATION)
        return "{}"

    return FakeLLM(handler=handler)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "cart.py").write_text(
        "def get_total_price(items):\n"
        "    # bug: quantity is ignored\n"
        "    return sum(i.price for i in items)\n",
        encoding="utf-8",
    )
    (tmp_path / "auth.py").write_text(
        "def login(user, password):\n    return verify(user, password)\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def engine(repo):
    def _make(**kwargs) -> OpsCopilot:
        llm = kwargs.pop("llm", None) or scripted_llm()
        cop = OpsCopilot.from_env(
            llm="fake:test",
            embeddings="hash:256",
            vector_store="memory://",
            state_store="memory://",
            **kwargs,
        )
        cop._llm = llm
        cop._vision_llm = llm
        cop.index_repository(repo)
        return cop

    return _make
