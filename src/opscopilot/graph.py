"""Optional LangGraph wiring.

The engine deliberately does not require LangGraph: the automated portion of
this pipeline is a short linear sequence with one conditional branch, which
plain Python expresses more clearly and tests more easily. This module exists
for consumers who want the pipeline as a `StateGraph` — to compose it into a
larger graph, or to use LangGraph's tracing and checkpointing.

Note it models only the automated portion. Both human decision points
terminate the graph, and resuming happens through the engine, because the
state store here is already the durable source of truth for what is paused;
adding LangGraph checkpointers would create a second one.
"""

from __future__ import annotations

from typing import Any

from .engine import OpsCopilot
from .exceptions import MissingDependencyError
from .models import IncidentState


def build_graph(engine: OpsCopilot):
    """Compile the automated pipeline as a LangGraph StateGraph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("langgraph", "graph") from exc

    def _state(payload: dict[str, Any]) -> IncidentState:
        return IncidentState.from_dict(payload["state"])

    def triage(payload: dict[str, Any]) -> dict[str, Any]:
        state = _state(payload)
        engine._run_triage(state)
        return {"state": state.to_dict()}

    def vision(payload: dict[str, Any]) -> dict[str, Any]:
        state = _state(payload)
        engine._run_vision(state)
        return {"state": state.to_dict()}

    def diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
        state = _state(payload)
        engine._diagnose(state)
        return {"state": state.to_dict()}

    def remediation(payload: dict[str, Any]) -> dict[str, Any]:
        state = _state(payload)
        engine._remediate(state)
        return {"state": state.to_dict()}

    def request_evidence(payload: dict[str, Any]) -> dict[str, Any]:
        state = _state(payload)
        engine._request_evidence(state)
        return {"state": state.to_dict()}

    def gate(payload: dict[str, Any]) -> str:
        state = _state(payload)
        if engine._confidence_sufficient(state.diagnosis):
            return "sufficient"
        if len(state.evidence) >= engine.settings.max_evidence_rounds:
            return "sufficient"
        return "insufficient"

    graph: Any = StateGraph(dict)
    graph.add_node("triage", triage)
    graph.add_node("vision", vision)
    graph.add_node("diagnosis", diagnosis)
    graph.add_node("remediation", remediation)
    graph.add_node("request_evidence", request_evidence)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "vision")
    graph.add_edge("vision", "diagnosis")
    graph.add_conditional_edges(
        "diagnosis", gate,
        {"sufficient": "remediation", "insufficient": "request_evidence"},
    )
    graph.add_edge("remediation", END)
    graph.add_edge("request_evidence", END)

    return graph.compile()
