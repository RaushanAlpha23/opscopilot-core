# Migrating from 0.1.x

0.2.0 renames the entry point and returns typed objects. The changes are
mechanical; the table and examples below cover everything.

| 0.1.x | 0.2.0 |
|---|---|
| `OpsCopilotClient(...)` | `OpsCopilot(...)` or `OpsCopilot.from_env()` |
| `openai_api_key="sk-..."` | `OPSCOPILOT_OPENAI_API_KEY` env var, or `openai_api_key=...` |
| `redis_url=`, `qdrant_url=` | `state_store="redis://..."`, `vector_store="http://..."` |
| `ingest_codebase(path)` | `index_repository(path)` |
| `run_triage(id, text, image)` | `submit(user_note=..., screenshot_path=..., incident_id=...)` |
| `approve_and_resume(id)` | `approve(id)` |
| — | `reject(id, reason=...)` |
| — | `provide_evidence(id, response)` |
| `result["proposed_fix"]` | `state.remediation.suggested_fix` |
| `result["status"]` | `state.status` (an `IncidentStatus` enum) |
| `result["ticket_url"]` | `state.ticket.url` |

### Before

```python
from opscopilot import OpsCopilotClient

client = OpsCopilotClient(
    openai_api_key="sk-...",
    redis_url="redis://localhost:6379",
    qdrant_url="http://localhost:6333",
    github_token="ghp_...",
    github_repo="me/app",
)
client.ingest_codebase("./app")

result = client.run_triage("inc-1", "cart total is wrong", image_path="bug.png")
print(result["proposed_fix"])

final = client.approve_and_resume("inc-1")
print(final["ticket_url"])
```

### After

```python
from opscopilot import OpsCopilot, IncidentStatus

cop = OpsCopilot.from_env(
    llm="openai:gpt-4o-mini",
    embeddings="openai:text-embedding-3-small",
    state_store="redis://localhost:6379/0",
    vector_store="http://localhost:6333",
    github_repo="me/app",
)
cop.index_repository("./app")

state = cop.submit(
    incident_id="inc-1",
    user_note="cart total is wrong",
    screenshot_path="bug.png",
)

# New: the engine may ask a question instead of proposing a fix.
if state.status is IncidentStatus.awaiting_evidence:
    state = cop.provide_evidence("inc-1", answer_to(state.pending_question))

print(state.remediation.suggested_fix)
print(cop.approve("inc-1").ticket.url)
```

### Behavioural differences to plan for

1. **`submit` no longer always ends at approval.** It can return
   `awaiting_evidence`. Branch on `state.status`; 0.1.x reported
   `"paused_for_approval"` unconditionally, which was sometimes untrue.
2. **Re-indexing is idempotent.** Chunk IDs are content-derived, so running
   `index_repository` twice updates points rather than doubling the collection.
   If you have a 0.1.x collection, delete it once — its UUID-keyed points will
   never be updated in place.
3. **Errors are typed.** Catch `OpsCopilotError` rather than `ValueError`;
   `StateNotFoundError` replaces the bare `ValueError` on an expired incident.
4. **Approving twice now raises** `InvalidTransitionError` instead of quietly
   filing a second ticket.
