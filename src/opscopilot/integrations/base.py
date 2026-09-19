"""Issue-tracker protocol, so GitHub is a choice rather than an assumption."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import IncidentState, Ticket


@runtime_checkable
class TicketProvider(Protocol):
    name: str

    def create(self, state: IncidentState) -> Ticket:
        ...
