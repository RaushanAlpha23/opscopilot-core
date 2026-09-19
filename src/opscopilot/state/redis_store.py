"""Redis-backed state, so a paused incident survives a process restart.

That durability is the entire point of the human-in-the-loop design: the gap
between "agent proposes a fix" and "human approves it" is measured in hours,
and holding that state in a Python variable loses it on the next deploy.
"""

from __future__ import annotations

import json
from typing import cast

from ..exceptions import MissingDependencyError
from ..models import IncidentState

KEY_PREFIX = "opscopilot:incident:"
INDEX_KEY = "opscopilot:incidents"


class RedisStateStore:
    name = "redis"

    def __init__(self, url: str = "redis://localhost:6379/0", ttl_seconds: int = 86400) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("redis", "redis") from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, incident_id: str) -> str:
        return f"{KEY_PREFIX}{incident_id}"

    def save(self, state: IncidentState) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._key(state.incident_id), json.dumps(state.to_dict()), ex=self.ttl)
        # A sorted set keyed by insertion order gives `list_ids` without a
        # KEYS scan, which is O(n) and blocks the server on a large keyspace.
        pipe.zadd(INDEX_KEY, {state.incident_id: _now()})
        pipe.execute()

    def load(self, incident_id: str) -> IncidentState | None:
        raw = self._redis.get(self._key(incident_id))
        return IncidentState.from_dict(json.loads(raw)) if raw else None

    def delete(self, incident_id: str) -> None:
        pipe = self._redis.pipeline()
        pipe.delete(self._key(incident_id))
        pipe.zrem(INDEX_KEY, incident_id)
        pipe.execute()

    def list_ids(self, limit: int = 100) -> list[str]:
        # redis-py's stub shares one signature across sync and async clients and
        # every `withscores` overload, so its declared return type stays a wide
        # union no matter the call. At runtime it's always list[str] here because
        # the client is constructed with decode_responses=True and we never pass
        # withscores=True.
        ids = self._redis.zrevrange(INDEX_KEY, 0, max(0, limit - 1))
        return cast(list[str], ids)

    def ping(self) -> bool:
        return bool(self._redis.ping())


def _now() -> float:
    import time

    return time.time()
