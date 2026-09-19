"""Correlate symptom + retrieved code + schema into one root-cause hypothesis."""

from __future__ import annotations

from ..llm.base import LLMProvider, complete_json
from ..models import Diagnosis, EvidenceExchange, RetrievedChunk, VisionAnalysis
from . import prompts


def build_retrieval_query(vision: VisionAnalysis, user_note: str | None = None) -> str:
    """Build the search query from the model's grounded read of the screen.

    Preferring `visible_symptom` over `user_note` matters: the vision output is
    anchored to what is actually rendered ("price field renders blank for
    product 4821"), whereas the note is whatever the reporter typed ("price
    broken"). Better query in, better chunks out.
    """
    parts = [vision.visible_symptom]
    if vision.api_response_notes:
        parts.append(vision.api_response_notes)
    if user_note:
        parts.append(user_note)
    return " ".join(p for p in parts if p).strip()


def _format(chunks: list[RetrievedChunk], max_chars: int, empty: str) -> str:
    if not chunks:
        return empty
    return "\n\n".join(c.to_prompt(max_chars) for c in chunks)


def run_diagnosis(
    llm: LLMProvider,
    vision: VisionAnalysis,
    code_chunks: list[RetrievedChunk],
    schema_chunks: list[RetrievedChunk],
    *,
    evidence: list[EvidenceExchange] | None = None,
    max_chunk_chars: int = 1200,
    retries: int = 1,
) -> Diagnosis:
    evidence_text = ""
    if evidence:
        answered = [e for e in evidence if e.response]
        if answered:
            lines = "\n".join(f"Q: {e.question}\nA: {e.response}" for e in answered)
            evidence_text = f"\n## Additional evidence provided by the reporter\n{lines}"

    parsed = complete_json(
        llm,
        prompts.DIAGNOSIS.format(
            symptom=vision.visible_symptom,
            code_chunks=_format(code_chunks, max_chunk_chars, "(no code chunks retrieved)"),
            schema_chunks=_format(schema_chunks, max_chunk_chars, "(no schema chunks retrieved)"),
            evidence=evidence_text,
        ),
        temperature=0.2,
        retries=retries,
    )

    try:
        confidence = float(parsed.get("confidence_score") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    suspected_file = parsed.get("suspected_file")
    # Guard against the model inventing a path despite the instruction not to.
    # A hallucinated file makes the remediation step reference code that does
    # not exist, which is worse than admitting the file is unknown.
    if suspected_file:
        known = {c.chunk.file_path for c in code_chunks}
        if known and suspected_file not in known:
            suspected_file = None

    return Diagnosis(
        summary=str(parsed.get("diagnosis_summary") or "No diagnosis produced."),
        confidence=max(0.0, min(1.0, confidence)),
        suspected_file=suspected_file,
        suspected_symbol=parsed.get("suspected_symbol"),
    )
