"""Runs the whole pipeline with no API keys and no infrastructure.

    python examples/quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from opscopilot import IncidentStatus, OpsCopilot
from opscopilot.llm import FakeLLM

BUGGY_APP = '''
def get_total_price(items):
    """Sum the cart. BUG: quantity is ignored."""
    return sum(item.price for item in items)


def apply_discount(total, code):
    return total * 0.9 if code else total
'''


def scripted(prompt: str) -> str:
    """Stands in for a real model so this runs offline."""
    if "triage assistant" in prompt:
        return json.dumps({"severity": "high", "category": "cart"})
    if "analyzing a bug report" in prompt:
        return json.dumps({
            "visible_symptom": "cart total price is too low when quantity exceeds one",
            "likely_layer": "backend",
            "api_response_consistent": False,
            "api_response_notes": "total does not reflect item quantity",
            "raw_confidence": 0.85,
        })
    if "senior engineer diagnosing" in prompt:
        return json.dumps({
            "diagnosis_summary": "get_total_price sums item.price without multiplying "
                                 "by item.quantity, so multi-quantity carts undercharge.",
            "suspected_file": "cart.py",
            "suspected_symbol": "get_total_price",
            "confidence_score": 0.88,
        })
    if "writing a remediation" in prompt:
        return json.dumps({
            "suggested_fix": "In get_total_price, multiply item.price by item.quantity.",
            "risk_level": "medium",
            "patch": "-    return sum(item.price for item in items)\n"
                     "+    return sum(item.price * item.quantity for item in items)",
        })
    return "{}"


def main() -> None:
    repo = Path(tempfile.mkdtemp())
    (repo / "cart.py").write_text(BUGGY_APP, encoding="utf-8")

    cop = OpsCopilot.from_env(llm="fake:demo", embeddings="hash:256")
    cop._llm = cop._vision_llm = FakeLLM(handler=scripted)
    cop.on_event(lambda e: print(f"  [event] {e.type} {e.data or ''}"))

    print(f"Indexed {cop.index_repository(repo)} chunks\n")

    state = cop.submit(user_note="Cart total is too low when I order more than one item")
    print()
    print(f"status     : {state.status.value}")
    print(f"severity   : {state.triage.severity.value} ({state.triage.category})")
    print(f"confidence : {state.diagnosis.confidence}")
    print(f"suspect    : {state.diagnosis.suspected_file}::{state.diagnosis.suspected_symbol}")
    print(f"diagnosis  : {state.diagnosis.summary}")
    print(f"fix        : {state.remediation.suggested_fix}")
    print(f"risk       : {state.remediation.risk_level.value}")

    if state.status is IncidentStatus.awaiting_approval:
        print()
        final = cop.approve(state.incident_id)
        print(f"final      : {final.status.value} — {final.ticket.url or final.ticket.error}")


if __name__ == "__main__":
    main()
