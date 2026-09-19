"""Classify severity and category from the user's written description."""

from __future__ import annotations

import logging

from ..exceptions import ProviderError
from ..llm.base import LLMProvider, complete_json
from ..models import Severity, Triage
from . import prompts

logger = logging.getLogger("opscopilot")


def run_triage(llm: LLMProvider, user_note: str | None, *, retries: int = 1) -> Triage:
    # No note is not an error: the screenshot alone may be enough for
    # diagnosis, so degrade to defaults rather than halting the pipeline.
    if not user_note or not user_note.strip():
        return Triage()

    try:
        parsed = complete_json(
            llm,
            prompts.TRIAGE.format(user_note=user_note),
            temperature=0.1,  # classification should be reproducible
            retries=retries,
        )
    except ProviderError:
        # A failed classification degrades to defaults rather than halting the
        # run — diagnosis is the valuable part and does not depend on this.
        # Deliberately narrow: a bare `except Exception` here once hid a
        # KeyError from a malformed prompt template, turning a code bug into a
        # silently wrong severity on every single incident.
        logger.warning("opscopilot: triage classification failed, using defaults", exc_info=True)
        return Triage()

    try:
        severity = Severity(str(parsed.get("severity", "medium")).lower())
    except ValueError:
        severity = Severity.medium

    category = str(parsed.get("category") or "unknown").strip().lower() or "unknown"
    return Triage(severity=severity, category=category)
