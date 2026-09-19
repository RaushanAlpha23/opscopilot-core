"""Process-local state store. Fine for a CLI run or tests; not for multiple workers."""

from __future__ import annotations

import copy

from ..models import IncidentState


class InMemoryStateStore:
    name = "memory"

    def __init__(self) -> None:
        self._states: dict[str, IncidentState] = {}

    def save(self, state: IncidentState) -> None:
        # Deep-copied on the way in and out so a caller mutating a returned
        # state cannot silently corrupt what's "persisted" — this keeps the
        # in-memory store's semantics identical to the Redis one, which
        # serialises and therefore always hands back a fresh object.
        self._states[state.incident_id] = copy.deepcopy(state)

    def load(self, incident_id: str) -> IncidentState | None:
        state = self._states.get(incident_id)
        return copy.deepcopy(state) if state else None

    def delete(self, incident_id: str) -> None:
        self._states.pop(incident_id, None)

    def list_ids(self, limit: int = 100) -> list[str]:
        return list(self._states)[:limit]
