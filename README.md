# opscopilot-core

An agentic SRE engine: screenshot-grounded bug triage, code-aware RAG diagnosis, and human-in-the-loop remediation.

Point it at a repository, hand it a bug report and a screenshot, and it will read the screen, retrieve the code most likely responsible, propose a root cause with a confidence score, ask a clarifying question when it isn't sure, and file a reviewed GitHub issue once a human approves.

```bash
pip install opscopilot-core
```

---

## Quickstart

The base install pulls no LLM SDK, no vector database and no torch, so this runs offline in a couple of seconds:

```python
from opscopilot import OpsCopilot

cop = OpsCopilot.from_env(llm="mistral:mistral-small-latest")
cop.index_repository("./my_app")

state = cop.submit(
    user_note="Cart total shows NaN after removing an item",
    screenshot_path="bug.png",
    api_response={"items": [], "total": None},
)

print(state.status)               # awaiting_approval
print(state.diagnosis.summary)    # "get_total_price divides by len(items)..."
print(state.diagnosis.confidence) # 0.86
print(state.remediation.suggested_fix)

cop.approve(state.incident_id)    # files the GitHub issue
```

If the engine isn't confident enough, it asks instead of guessing:

```python
if state.status is IncidentStatus.awaiting_evidence:
    print(state.pending_question)   # "Can you paste the response from GET /api/cart?"
    state = cop.provide_evidence(state.incident_id, "total is null, items is []")
```

### From the command line

```bash
export OPSCOPILOT_MISTRAL_API_KEY=...

opscopilot --vector-store memory://./index.json index ./my_app
opscopilot submit --note "cart total is NaN" --screenshot bug.png
opscopilot evidence <incident-id> "GET /api/cart returns total: null"
opscopilot approve <incident-id>
```

---

## How it works

```
submit()
   │
   ├─ triage ............ severity + category from the written report
   ├─ vision ............ structured read of the screenshot, cross-checked
   │                      against the API response
   ├─ retrieval ......... query built from the vision model's grounded
   │                      symptom, not the reporter's wording
   ├─ diagnosis ......... one call correlating symptom + code + schema,
   │                      returning a root cause and a confidence score
   │
   └─ confidence gate
        ├─ below threshold ─→ ask ONE specific question ──→ awaiting_evidence
        │                     (provide_evidence resumes from diagnosis)
        └─ at/above ────────→ remediation ────────────────→ awaiting_approval
                                                              │
                                             approve() ───────┴──→ ticketed
                                             reject()  ──────────→ rejected
```

Both pauses are durable. State lives in the state store, not in a Python variable, so an incident parked on a human decision survives a restart or a deploy — which matters, because that gap is measured in hours.

---

## Configuration

Every backend is chosen by a string, so swapping one is a config change rather than a code change.

| Setting | Env var | Default | Options |
|---|---|---|---|
| `llm` | `OPSCOPILOT_LLM` | `mistral:mistral-small-latest` | `mistral:*`, `openai:*`, `fake:*` |
| `vision_llm` | `OPSCOPILOT_VISION_LLM` | falls back to `llm` | same |
| `embeddings` | `OPSCOPILOT_EMBEDDINGS` | `hash:384` | `hash:N`, `st:<model>`, `openai:<model>` |
| `vector_store` | `OPSCOPILOT_VECTOR_STORE` | `memory://` | `memory://`, `memory://path.json`, `http://host:6333` |
| `state_store` | `OPSCOPILOT_STATE_STORE` | `memory://` | `memory://`, `redis://host:6379/0` |
| `confidence_threshold` | `OPSCOPILOT_CONFIDENCE_THRESHOLD` | `0.6` | |
| `max_evidence_rounds` | `OPSCOPILOT_MAX_EVIDENCE_ROUNDS` | `2` | questions asked before proceeding anyway |
| `github_token` / `github_repo` | `OPSCOPILOT_GITHUB_TOKEN` / `_REPO` | unset | ticketing is skipped when unset |

A production setup:

```bash
OPSCOPILOT_LLM=mistral:mistral-small-latest
OPSCOPILOT_EMBEDDINGS=st:sentence-transformers/all-MiniLM-L6-v2
OPSCOPILOT_VECTOR_STORE=http://localhost:6333
OPSCOPILOT_STATE_STORE=redis://localhost:6379/0
OPSCOPILOT_GITHUB_REPO=your-org/your-repo
```

```bash
pip install 'opscopilot-core[mistral,qdrant,redis,local-embeddings,github]'
```

### Defaults worth knowing

`hash:384` is a dependency-free lexical embedder. It exists so the package installs and runs instantly, and so the test suite exercises real retrieval rather than a mock. It matches identifiers well and has no semantic understanding — `get_total_price` will match a query naming it, but "prices look wrong" will not find `calculate_subtotal`. **Use `st:` or `openai:` embeddings for real retrieval quality.**

`memory://` holds the index in the process. Use `memory://./index.json` to persist across CLI invocations, or Qdrant for anything concurrent.

---

## Extras

| Extra | Brings | Needed for |
|---|---|---|
| `mistral` | `langchain-mistralai` | Mistral models (incl. vision) |
| `openai` | `openai` | OpenAI chat + embeddings |
| `qdrant` | `qdrant-client` | Qdrant vector store |
| `redis` | `redis` | durable incident state |
| `local-embeddings` | `sentence-transformers` | local semantic embeddings (pulls torch) |
| `github` | `requests` | filing issues |
| `graph` | `langgraph` | the optional `StateGraph` wiring |
| `all` | everything above | |

---

## Extending it

Everything is a protocol, and the engine takes injected implementations:

```python
from opscopilot import OpsCopilot
from opscopilot.models import Ticket

class LinearTickets:
    name = "linear"
    def create(self, state) -> Ticket:
        ...

cop = OpsCopilot(ticket_provider=LinearTickets())
```

The same pattern works for `llm`, `embedder`, `vector_store` and `state_store`. Prompts live in one module (`opscopilot.nodes.prompts`) so they can be overridden without touching node logic.

### Progress events

```python
cop.on_event(lambda e: print(e.type, e.data))
# incident.received / node.started / node.finished / awaiting_approval / ticket.created
```

The engine never assumes where events go. Subscribe a handler that publishes to Redis pub/sub, writes a log line, or appends to a trace document — see `examples/fastapi_app.py` for streaming them to a browser over SSE.

---

## Testing against it

`FakeLLM` ships in the package so downstream tests need no API keys:

```python
from opscopilot import OpsCopilot
from opscopilot.llm import FakeLLM

cop = OpsCopilot.from_env(llm="fake:test", embeddings="hash:256")
cop._llm = cop._vision_llm = FakeLLM(handler=my_scripted_responses)
```

---

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
mypy src
```

## License

MIT
