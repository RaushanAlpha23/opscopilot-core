"""Read the screenshot, cross-referenced against the API response."""

from __future__ import annotations

import json
from typing import Any

from ..llm.base import Image, LLMProvider, complete_json
from ..models import VisionAnalysis
from . import prompts


def run_vision(
    llm: LLMProvider,
    screenshot_path: str | None,
    *,
    api_response: dict[str, Any] | None = None,
    user_note: str | None = None,
    logs: str | None = None,
    retries: int = 1,
) -> VisionAnalysis:
    """Produce a structured read of the symptom.

    Degrades gracefully in two directions the old code did not handle: with no
    screenshot it falls back to text-only analysis, and with a non-vision model
    it does the same instead of raising. An incident reported without a
    screenshot is common and should still be diagnosable.
    """
    context_parts: list[str] = []
    if api_response is not None:
        context_parts.append(f"API response received by the frontend:\n{json.dumps(api_response)}")
    if user_note:
        context_parts.append(f"User's note: {user_note}")
    if logs:
        context_parts.append(f"Relevant logs:\n{logs[:4000]}")

    image: Image | None = None
    if screenshot_path and getattr(llm, "supports_vision", False):
        image = Image.from_path(screenshot_path)
        context_parts.insert(0, "Analyze the attached screenshot.")
    else:
        context_parts.insert(
            0,
            "No screenshot is available. Infer the visible symptom from the "
            "information below and lower raw_confidence accordingly.",
        )

    parsed = complete_json(
        llm,
        prompts.VISION.format(context="\n\n".join(context_parts)),
        temperature=0.1,  # literal reading, not creative
        image=image,
        retries=retries,
    )

    return VisionAnalysis(
        visible_symptom=str(parsed.get("visible_symptom") or user_note or "unspecified symptom"),
        likely_layer=str(parsed.get("likely_layer") or "unclear"),
        api_response_consistent=parsed.get("api_response_consistent"),
        api_response_notes=parsed.get("api_response_notes"),
        raw_confidence=parsed.get("raw_confidence"),
    )
