"""Resolve a state-store URL into a StateStore."""

from __future__ import annotations

from ..config import Settings
from ..exceptions import ConfigurationError
from .base import StateStore


def build_state_store(url: str, settings: Settings) -> StateStore:
    """Supported: 'memory://', 'redis://host:port/db', 'rediss://...'."""
    if url.startswith("memory://"):
        from .memory import InMemoryStateStore

        return InMemoryStateStore()

    if url.startswith(("redis://", "rediss://", "unix://")):
        from .redis_store import RedisStateStore

        return RedisStateStore(url, ttl_seconds=settings.state_ttl_seconds)

    raise ConfigurationError(
        f"Unrecognised state store URL '{url}'. Use 'memory://' or 'redis://localhost:6379/0'."
    )
