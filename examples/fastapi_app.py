"""How the Ops-Copilot application should consume this package.

This is the structural point of the 0.2.0 restructuring. Today the application
and the SDK contain two divergent copies of the same engine — two triage
prompts, two chunkers, two approval gates, one on Mistral and one on OpenAI.
Every fix has to be made twice, and in practice only gets made once.

Here the application keeps only what is genuinely application-specific:

  * HTTP routing and request/response shapes
  * the relational audit trail (Postgres) and the trace store (Mongo)
  * SSE streaming to the browser
  * file upload handling

...and delegates everything else — vision, retrieval, diagnosis, the
confidence gate, remediation, ticketing — to `opscopilot`. The engine's
progress events are what the application subscribes to in order to write its
own persistence, which is why the package emits events rather than writing to
a database itself.

Run:  uvicorn fastapi_app:app --reload
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opscopilot import IncidentState, IncidentStatus, OpsCopilot
from opscopilot.exceptions import InvalidTransitionError, StateNotFoundError

SCREENSHOT_DIR = Path("data/screenshots")

cop = OpsCopilot.from_env()

# One queue per incident, fed by the engine's event bus and drained by the SSE
# endpoint. In a multi-worker deployment, replace this with Redis pub/sub —
# the subscription point stays exactly the same.
_streams: dict[str, asyncio.Queue] = {}


@cop.on_event
def _fan_out(event) -> None:
    queue = _streams.get(event.incident_id)
    if queue is not None:
        queue.put_nowait(event.to_dict())


@cop.on_event
def _write_audit_trail(event) -> None:
    """Application-owned persistence, driven by engine events.

    This is where the Postgres `incidents` / `remediation_logs` / `tickets`
    writes and the Mongo trace document belong — the package should not know
    those tables exist.
    """
    # persist_event(event)  # your db_mongo.append_node_execution(...)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    # init_db(); init_mongo_indexes()
    yield


app = FastAPI(title="AI Ops Copilot", lifespan=lifespan)


class EvidenceBody(BaseModel):
    response: str


class ApprovalBody(BaseModel):
    decision: str  # "approved" | "rejected"
    reason: str | None = None


def _serialize(state: IncidentState) -> dict[str, Any]:
    """The HTTP shape is the application's concern, not the package's."""
    return {
        "incident_id": state.incident_id,
        "status": state.status.value,
        "severity": state.triage.severity.value if state.triage else None,
        "category": state.triage.category if state.triage else None,
        "diagnosis": state.diagnosis.summary if state.diagnosis else None,
        "confidence": state.diagnosis.confidence if state.diagnosis else None,
        "suspected_file": state.diagnosis.suspected_file if state.diagnosis else None,
        "suggested_fix": state.remediation.suggested_fix if state.remediation else None,
        "risk_level": state.remediation.risk_level.value if state.remediation else None,
        "pending_question": state.pending_question,
        "ticket_url": state.ticket.url if state.ticket else None,
        "error": state.error,
    }


@app.post("/incidents")
async def submit_incident(
    screenshot: UploadFile | None = None,
    user_note: str | None = Form(default=None),
    api_response_json: str | None = Form(default=None),
):
    incident_id = str(uuid.uuid4())
    _streams[incident_id] = asyncio.Queue()

    screenshot_path = None
    if screenshot is not None:
        suffix = Path(screenshot.filename or ".png").suffix or ".png"
        screenshot_path = SCREENSHOT_DIR / f"{incident_id}{suffix}"
        screenshot_path.write_bytes(await screenshot.read())

    api_response = None
    if api_response_json:
        try:
            api_response = json.loads(api_response_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "api_response_json is not valid JSON") from None

    # The engine is synchronous and LLM-bound, so it runs in a worker thread
    # rather than blocking the event loop and stalling every open SSE stream.
    state = await asyncio.to_thread(
        cop.submit,
        incident_id=incident_id,
        user_note=user_note,
        screenshot_path=screenshot_path,
        api_response=api_response,
    )
    return _serialize(state)


@app.get("/incidents/{incident_id}/events")
async def incident_events(incident_id: str):
    queue = _streams.setdefault(incident_id, asyncio.Queue())

    async def generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # stops proxies closing an idle stream
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in {"awaiting_approval", "awaiting_evidence",
                                 "ticket.created", "rejected", "incident.failed"}:
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/incidents/{incident_id}/evidence")
async def submit_evidence(incident_id: str, body: EvidenceBody):
    try:
        state = await asyncio.to_thread(cop.provide_evidence, incident_id, body.response)
    except StateNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except InvalidTransitionError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _serialize(state)


@app.post("/incidents/{incident_id}/approval")
async def submit_approval(incident_id: str, body: ApprovalBody):
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(400, "decision must be 'approved' or 'rejected'")
    try:
        if body.decision == "approved":
            state = await asyncio.to_thread(cop.approve, incident_id)
        else:
            state = await asyncio.to_thread(cop.reject, incident_id, body.reason)
    except StateNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except InvalidTransitionError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _serialize(state)


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    try:
        return _serialize(cop.get(incident_id))
    except StateNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/incidents")
async def list_incidents(limit: int = 50):
    return {"incident_ids": cop.list_incidents(limit)}


@app.post("/index")
async def reindex(path: str = Form(...)):
    count = await asyncio.to_thread(cop.index_repository, path)
    return {"indexed_chunks": count}
