"""State store protocol.

The previous SDK persisted only a bare dict under one Redis key and exposed
save/load/clear. Two capabilities were missing that a real deployment needs:
listing incidents (an operator dashboard has nothing to render without it),
and atomic status transitions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import IncidentState


@runtime_checkable
class StateStore(Protocol):
    def save(self, state: IncidentState) -> None:
        ...

    def load(self, incident_id: str) -> IncidentState | None:
        ...

    def delete(self, incident_id: str) -> None:
        ...

    def list_ids(self, limit: int = 100) -> list[str]:
        ...
