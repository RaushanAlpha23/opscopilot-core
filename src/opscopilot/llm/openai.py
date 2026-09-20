"""OpenAI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError, MissingDependencyError
from .base import Image
from .retry import RetryPolicy

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionUserMessageParam


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        timeout: int = 60,
        retry: RetryPolicy | None = None,
    ):
        if not api_key:
            raise ConfigurationError(
                "An OpenAI API key is required. Set OPSCOPILOT_OPENAI_API_KEY "
                "or pass openai_api_key=... to OpsCopilot."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("openai", "openai") from exc

        self.model = model
        self._retry = retry or RetryPolicy()
        # max_retries=0: the SDK's own (2-try) retry would multiply with ours.
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    @property
    def supports_vision(self) -> bool:
        return not self.model.startswith(("gpt-3.5", "o1-mini"))

    def complete(
        self, prompt: str, *, temperature: float = 0.2, image: Image | None = None
    ) -> str:
        content: str | list[dict[str, Any]]
        if image is None:
            content = prompt
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ]

        # Built via the SDK's own TypedDict rather than a bare dict literal:
        # `messages` is a discriminated union keyed on "role", which a plain
        # `dict[str, object]` can't satisfy — mypy can't tell it apart from a
        # system or assistant message without the real type in hand.
        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,  # type: ignore[typeddict-item]
        }

        response = self._retry.call(
            "openai",
            lambda: self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[message],
            ),
        )
        return (response.choices[0].message.content or "").strip()
