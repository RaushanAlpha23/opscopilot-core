"""Progress events.

The application publishes these to Redis pub/sub and streams them to the
browser over SSE. The package must not assume that: it emits events to
registered callbacks, and the *application* decides whether a callback
publishes to Redis, writes a log line, or appends to a Mongo trace. That
inversion is what lets the same engine run in a CLI, a worker and a web app.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("opscopilot")


@dataclass(slots=True)
class Event:
    incident_id: str
    type: str  # e.g. "node.started", "node.finished", "awaiting_approval"
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register a handler; returns a function that unsubscribes it."""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    def emit(self, incident_id: str, type: str, **data: Any) -> Event:
        event = Event(incident_id=incident_id, type=type, data=data)
        for handler in list(self._handlers):
            try:
                handler(event)
            except Exception:
                # A broken subscriber must never abort an incident run.
                logger.exception("opscopilot: event handler raised, continuing")
        return event
