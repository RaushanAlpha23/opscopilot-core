"""Provider-agnostic LLM interface.

The previous SDK hardcoded `OpenAI(...)` and `model="gpt-4o"` inside the graph
nodes, while the main application hardcoded `ChatMistralAI`. That is the single
biggest reason the two codebases could not share an engine. Nodes here depend
only on the `LLMProvider` protocol, so the same node code runs against Mistral,
OpenAI, or a fake used in tests.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..exceptions import InvalidJSONResponse, ProviderError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(slots=True)
class Image:
    """An image to send alongside a prompt, already in memory."""

    data_base64: str
    media_type: str = "image/png"

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        p = Path(path)
        if not p.is_file():
            raise ProviderError(f"Screenshot not found: {p}")
        media_type = mimetypes.guess_type(p.name)[0] or "image/png"
        if not media_type.startswith("image/"):
            media_type = "image/png"
        return cls(base64.b64encode(p.read_bytes()).decode("utf-8"), media_type)

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data_base64}"


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal surface every provider adapter must implement.

    `supports_vision` is declared as a property, not a plain attribute: both
    shipped adapters compute it from `model` rather than storing it, so it
    must be read-only to type-check as an implementation of this protocol.
    """

    name: str
    model: str

    @property
    def supports_vision(self) -> bool: ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        image: Image | None = None,
    ) -> str:
        """Return the model's text response to a single prompt."""
        ...


def extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models ignore "no code fences" instructions often enough that stripping
    them defensively is cheaper than a retry. Falls back to locating the
    outermost brace pair when the model wraps JSON in prose.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise InvalidJSONResponse(raw) from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise InvalidJSONResponse(raw) from None

    if not isinstance(parsed, dict):
        raise InvalidJSONResponse(raw)
    return parsed


def complete_json(
    provider: LLMProvider,
    prompt: str,
    *,
    temperature: float = 0.2,
    image: Image | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    """Ask for JSON and parse it, re-prompting once if the model misbehaves.

    A single retry with an explicit corrective instruction recovers the large
    majority of malformed-JSON cases and is far cheaper than failing the run.
    """
    attempt_prompt = prompt
    last_error: InvalidJSONResponse | None = None

    for attempt in range(retries + 1):
        raw = provider.complete(attempt_prompt, temperature=temperature, image=image)
        try:
            return extract_json(raw)
        except InvalidJSONResponse as exc:
            last_error = exc
            attempt_prompt = (
                f"{prompt}\n\n"
                "Your previous response was not valid JSON. Respond with ONLY a "
                "single JSON object, no prose, no markdown fences."
            )
            # Nudge toward determinism on the retry.
            temperature = min(temperature, 0.1)
            if attempt == retries:
                break

    assert last_error is not None
    raise last_error
