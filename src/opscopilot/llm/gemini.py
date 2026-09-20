"""Google Gemini adapter (native vision on every current Gemini model)."""

from __future__ import annotations

import base64
from typing import Any

from ..exceptions import ConfigurationError, MissingDependencyError
from .base import Image
from .retry import RetryPolicy


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        api_key: str = "",
        timeout: int = 60,
        retry: RetryPolicy | None = None,
    ):
        if not api_key:
            raise ConfigurationError(
                "A Gemini API key is required. Set OPSCOPILOT_GEMINI_API_KEY "
                "or pass gemini_api_key=... to OpsCopilot."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise MissingDependencyError("google-genai", "gemini") from exc

        self.model = model
        self._retry = retry or RetryPolicy()
        # HttpOptions.timeout is in milliseconds, unlike the other adapters.
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000)
        )

    @property
    def supports_vision(self) -> bool:
        return True

    def complete(self, prompt: str, *, temperature: float = 0.2, image: Image | None = None) -> str:
        from google.genai import types

        contents: list[Any] = [prompt]
        if image is not None:
            contents.append(
                types.Part.from_bytes(
                    data=base64.b64decode(image.data_base64), mime_type=image.media_type
                )
            )
        # We never pass tools, and google-genai logs a WARNING about automatic function
        # calling on every generate_content call unless it is switched off.
        config = types.GenerateContentConfig(
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = self._retry.call(
            "gemini",
            lambda: self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            ),
        )
        return (response.text or "").strip()
