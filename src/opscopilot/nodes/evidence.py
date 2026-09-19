"""Ask one specific clarifying question when confidence is too low to act."""

from __future__ import annotations

from ..llm.base import LLMProvider
from ..models import Diagnosis, EvidenceExchange, RetrievedChunk
from . import prompts


def run_evidence_request(
    llm: LLMProvider,
    diagnosis: Diagnosis,
    code_chunks: list[RetrievedChunk],
    previous: list[EvidenceExchange] | None = None,
) -> EvidenceExchange:
    files = ", ".join(sorted({c.chunk.file_path for c in code_chunks})) or "none"
    # Passing prior questions prevents the loop asking the same thing twice,
    # which the single-round implementation could not guard against.
    asked = "; ".join(e.question for e in (previous or [])) or "none"

    question = llm.complete(
        prompts.EVIDENCE.format(
            summary=diagnosis.summary,
            confidence=diagnosis.confidence,
            files=files,
            asked=asked,
        ),
        temperature=0.3,
    ).strip()

    fallback = "Could you share the failing request and response?"
    return EvidenceExchange(question=question or fallback)
