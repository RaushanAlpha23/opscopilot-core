"""Resolve a "<provider>:<model>" spec string into a live provider instance."""

from __future__ import annotations

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import LLMProvider
from .retry import RetryPolicy

_KNOWN = ("mistral", "openai", "gemini", "fake")


def parse_spec(spec: str) -> tuple[str, str]:
    """Split "mistral:mistral-small-latest" into ("mistral", "mistral-small-latest")."""
    if ":" not in spec:
        raise ConfigurationError(
            f"Invalid model spec '{spec}'. Expected '<provider>:<model>', "
            f"e.g. 'mistral:mistral-small-latest'. Known providers: {', '.join(_KNOWN)}."
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in _KNOWN:
        raise ConfigurationError(
            f"Unknown LLM provider '{provider}'. Known providers: {', '.join(_KNOWN)}."
        )
    return provider, model.strip()


def build_llm(spec: str, settings: Settings) -> LLMProvider:
    provider, model = parse_spec(spec)
    retry = RetryPolicy(
        max_attempts=settings.llm_max_attempts,
        min_interval=settings.min_seconds_between_calls,
    )

    if provider == "mistral":
        from .mistral import MistralProvider

        return MistralProvider(
            model=model,
            api_key=settings.mistral_api_key,
            timeout=settings.request_timeout_seconds,
            retry=retry,
        )

    if provider == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
            retry=retry,
        )

    if provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            model=model or "gemini-flash-latest",
            api_key=settings.gemini_api_key,
            timeout=settings.request_timeout_seconds,
            retry=retry,
        )

    from .fake import FakeLLM

    return FakeLLM(model=model or "fake-1")
