"""An in-process provider for tests, demos and offline development.

Having this in the shipped package (not the test suite) is deliberate: it
lets users run the full pipeline end-to-end with zero API keys and zero
infrastructure, which is what the README quickstart relies on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from .base import Image


class FakeLLM:
    """Returns canned responses in order, or delegates to a callable.

    >>> llm = FakeLLM(["{\\"severity\\": \\"high\\", \\"category\\": \\"cart\\"}"])
    >>> llm.complete("classify this")
    '{"severity": "high", "category": "cart"}'
    """

    name = "fake"
    supports_vision = True

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        handler: Callable[[str], str] | None = None,
        model: str = "fake-1",
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self._handler = handler
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        image: Image | None = None,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "has_image": image is not None}
        )
        if self._handler is not None:
            return self._handler(prompt)
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"note": "fake provider exhausted its scripted responses"})
