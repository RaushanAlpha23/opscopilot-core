"""Mistral adapter (native vision on mistral-small-latest)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError, MissingDependencyError, ProviderError
from .base import Image
from .retry import RetryPolicy

if TYPE_CHECKING:
    from langchain_mistralai import ChatMistralAI

_VISION_MODELS = ("mistral-small", "mistral-medium", "mistral-large", "pixtral")


class MistralProvider:
    name = "mistral"

    def __init__(
        self,
        model: str = "mistral-small-latest",
        api_key: str = "",
        timeout: int = 60,
        retry: RetryPolicy | None = None,
    ):
        if not api_key:
            raise ConfigurationError(
                "A Mistral API key is required. Set OPSCOPILOT_MISTRAL_API_KEY "
                "or pass mistral_api_key=... to OpsCopilot."
            )
        try:
            from langchain_mistralai import ChatMistralAI  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingDependencyError("langchain-mistralai", "mistral") from exc

        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._retry = retry or RetryPolicy()
        self._cache: dict[float, ChatMistralAI] = {}

    @property
    def supports_vision(self) -> bool:
        return any(tag in self.model for tag in _VISION_MODELS)

    def _client(self, temperature: float) -> ChatMistralAI:
        # Cached per temperature: constructing a ChatModel is cheap but not free,
        # and nodes call with a small fixed set of temperatures.
        if temperature not in self._cache:
            from langchain_mistralai import ChatMistralAI
            from pydantic import SecretStr

            # ChatMistralAI's `model` field has validation_alias "model_name" and
            # its `mistral_api_key` field (aliased "api_key") requires a SecretStr,
            # not a plain str — both are pydantic v2 aliasing details that only
            # surface as a type error, never at runtime, so they're easy to miss
            # without checking the installed SDK directly.
            self._cache[temperature] = ChatMistralAI(
                api_key=SecretStr(self._api_key),
                model_name=self.model,
                temperature=temperature,
                timeout=self._timeout,
            )
        return self._cache[temperature]

    def complete(
        self, prompt: str, *, temperature: float = 0.2, image: Image | None = None
    ) -> str:
        from langchain_core.messages import HumanMessage

        if image is not None and not self.supports_vision:
            raise ProviderError(f"Model '{self.model}' does not support image input.")

        content: str | list[str | dict[str, Any]]
        if image is None:
            content = prompt
        else:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image.data_url},
            ]

        response = self._retry.call(
            "mistral",
            lambda: self._client(temperature).invoke([HumanMessage(content=content)]),
        )
        text = response.content
        if isinstance(text, list):  # some versions return content blocks
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return str(text).strip()
