"""Propose a concrete fix and rate its risk."""

from __future__ import annotations

from ..llm.base import LLMProvider, complete_json
from ..models import Diagnosis, Remediation, RetrievedChunk, RiskLevel
from . import prompts


def find_suspect_chunk(
    diagnosis: Diagnosis, chunks: list[RetrievedChunk]
) -> RetrievedChunk | None:
    """Locate the chunk the diagnosis pointed at: exact match, then file-only, then best hit."""
    for c in chunks:
        if (
            c.chunk.file_path == diagnosis.suspected_file
            and c.chunk.symbol_name == diagnosis.suspected_symbol
        ):
            return c
    for c in chunks:
        if c.chunk.file_path == diagnosis.suspected_file:
            return c
    return chunks[0] if chunks else None


def run_remediation(
    llm: LLMProvider,
    diagnosis: Diagnosis,
    code_chunks: list[RetrievedChunk],
    *,
    max_chunk_chars: int = 2000,
    retries: int = 1,
) -> Remediation:
    suspect = find_suspect_chunk(diagnosis, code_chunks)
    location = (
        f"{diagnosis.suspected_file or 'unknown file'} :: "
        f"{diagnosis.suspected_symbol or 'unknown symbol'}"
    )

    parsed = complete_json(
        llm,
        prompts.REMEDIATION.format(
            summary=diagnosis.summary,
            location=location,
            chunk=suspect.to_prompt(max_chunk_chars) if suspect else "(no code chunk available)",
        ),
        temperature=0.2,
        retries=retries,
    )

    try:
        risk = RiskLevel(str(parsed.get("risk_level", "medium")).lower())
    except ValueError:
        risk = RiskLevel.medium

    patch = parsed.get("patch")
    return Remediation(
        suggested_fix=str(parsed.get("suggested_fix") or "No fix suggested."),
        risk_level=risk,
        patch=patch if isinstance(patch, str) and patch.strip() else None,
    )
