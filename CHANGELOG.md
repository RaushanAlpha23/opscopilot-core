# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.3]

### Added
- Google Gemini provider: `llm="gemini:<model>"`, key via `OPSCOPILOT_GEMINI_API_KEY`
  (`pip install "opscopilot-core[gemini]"`).
- `RateLimitError` (a `ProviderError`) raised when a provider keeps answering HTTP 429.
- Settings `llm_max_attempts` (default 6) and `min_seconds_between_calls` (default 0.0).

### Fixed
- Indexing no longer prints `<unknown>:N: SyntaxWarning: invalid escape sequence` for Python files
  in the repository being indexed (those are the user's own files, not opscopilot's).
- HTTP 429 / 5xx from Mistral, OpenAI and Gemini are now retried with exponential
  backoff that honours `Retry-After`, instead of failing the incident on the first hit.
  `langchain-mistralai` only retries connection errors, never 429s.
- Gemini's own wait hint (`retryDelay` / "retry in Ns") is honoured, and 429s caused by a per-day or
  zero quota fail immediately with the quota name in the error instead of retrying for a minute.
- Retry log lines and `RateLimitError` now include the provider's quota id/limit.
- Gemini: silenced google-genai's per-call automatic-function-calling warning.
- OpenAI 429s caused by exhausted credit (`insufficient_quota`) fail fast instead of retrying.

## [0.2.2]

CI/type-checking fixes only — no runtime behaviour changes. 0.2.1 was published
from a tree that predated these fixes; if you're on 0.2.1, upgrade.

### Fixed
- `mistral.py`: `ChatMistralAI` is constructed with `model_name=` and a
  `SecretStr` api key, matching its actual pydantic field aliases
  (`model_name`/`api_key`). The previous `model=`/plain-`str` call worked at
  runtime but failed static type-checking.
- `openai.py`: chat messages are built via the SDK's own
  `ChatCompletionUserMessageParam` TypedDict instead of an untyped dict.
- `redis_store.py`: `list_ids` return value is explicitly cast, since
  redis-py's stub shares one return type across sync/async clients and every
  `withscores` overload and can never narrow to `list[str]` on its own.
- Removed the `python_version = "3.10"` pin from `[tool.mypy]`. It forced
  mypy to parse every third-party stub (including numpy, pulled in
  transitively by qdrant-client) as 3.10 syntax, which broke on newer numpy's
  use of the Python 3.12+ `type` statement — failing CI on the 3.12/3.13
  matrix jobs for reasons unrelated to this package's own code.
- Fixed 15 `ruff` violations (line length, unsorted imports, missing
  `zip(..., strict=True)`).
- `__version__` in `opscopilot/__init__.py` now matches `pyproject.toml`;
  they had drifted apart in 0.2.1.

## [0.2.0] — unreleased

A restructuring release. The 0.1.x SDK was a simplified reimplementation of the
Ops-Copilot application that had drifted away from it; 0.2.0 makes the package
the single engine both the application and external users run.

### Added
- `src/` layout, `py.typed`, and full type annotations exported to consumers.
- Pluggable backends behind protocols: `LLMProvider`, `Embedder`, `VectorStore`,
  `StateStore`, `TicketProvider`. Each is selected by a spec string.
- Mistral provider with native vision, alongside OpenAI.
- Vision analysis of screenshots, cross-referenced against the API response.
- Confidence gate with a bounded clarifying-question loop
  (`provide_evidence`), resumable from diagnosis.
- `InMemoryVectorStore` and `HashingEmbedder`, so the package installs and runs
  with no infrastructure and no API keys.
- Optional JSON persistence for the in-memory index.
- Schema indexing (`index_schema`) and SQL-aware chunking.
- `EventBus` for progress events, with SSE-friendly payloads.
- `opscopilot` CLI (index, search, submit, evidence, approve, reject, show, list).
- Typed exception hierarchy under `OpsCopilotError`.
- Optional LangGraph wiring in `opscopilot.graph`.
- 65 tests covering chunking, retrieval, nodes, lifecycle and GitHub error mapping.

### Changed
- **Breaking:** `OpsCopilotClient` is now `OpsCopilot`. See MIGRATION.md.
- **Breaking:** methods return typed `IncidentState` objects rather than
  loosely-shaped dicts. `run_triage` is `submit`; `approve_and_resume` is
  `approve`.
- Resuming after approval runs only the remaining work. Previously the graph
  was re-invoked from its entry point and every node had to check a status flag
  and no-op, which meant adding a node silently broke the resume path.
- `search()` returns `RetrievedChunk` objects with scores and metadata instead
  of a pre-formatted prompt string.
- Python chunking uses the stdlib `ast` module. Dropped `tree_sitter_languages`,
  which ships prebuilt binaries, is effectively unmaintained, and fails to
  install on recent Python versions.
- Chunk IDs are derived from content and location, so re-indexing a repository
  updates points in place instead of appending a duplicate set every run.
- Embeddings are batched. The previous indexer made one API call per chunk.
- Qdrant upserts are batched; the previous indexer buffered an entire
  repository in memory and sent a single request.
- Errors from every backend are wrapped in the package's own exception types.

### Fixed
- Qdrant retrieval uses `query_points`; `search()` is deprecated in
  qdrant-client ≥ 1.9.
- The embedding dimension is read from the loaded model rather than hardcoded
  to 1536/384, which silently corrupted a collection when the model changed.
- A diagnosis naming a file absent from the retrieved set is discarded, instead
  of sending a hallucinated path to the remediation step.
- Confidence scores are clamped to [0, 1].
- Transient GitHub 5xx and network failures are retried with backoff, so an
  approved remediation is not lost to a blip.
- Incidents without a screenshot, or run against a non-vision model, degrade to
  text-only analysis rather than raising.
- Tokenisation splits camelCase before lowercasing, so `getTotalPrice` matches
  a query saying "total price".

## [0.1.0]
- Initial release: Qdrant-backed AST chunking, a three-node LangGraph pipeline,
  a Redis approval gate, and GitHub issue creation.
