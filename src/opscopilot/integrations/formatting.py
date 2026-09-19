"""Render an incident into issue markdown.

Separated from the GitHub client so the same body can be reused by a Jira,
Linear or Slack integration without copying the formatting logic.
"""

from __future__ import annotations

import json

from ..models import IncidentState

MAX_TITLE = 100


def issue_title(state: IncidentState) -> str:
    category = state.triage.category if state.triage else "bug"
    summary = state.diagnosis.summary if state.diagnosis else "Issue detected"
    short = summary.split(".")[0].strip()[:MAX_TITLE]
    return f"[Auto-triaged] {short} ({category})"


def issue_body(state: IncidentState) -> str:
    parts: list[str] = ["## Diagnosis (AI-generated, human-reviewed)"]

    if state.diagnosis:
        parts += [
            state.diagnosis.summary,
            "",
            f"**Suspected file:** `{state.diagnosis.suspected_file or 'unknown'}`",
            f"**Suspected symbol:** `{state.diagnosis.suspected_symbol or 'unknown'}`",
            f"**Confidence:** {state.diagnosis.confidence:.2f}",
        ]
    else:
        parts.append("_(no diagnosis recorded)_")

    if state.triage:
        parts += ["", f"**Severity:** {state.triage.severity.value}",
                  f"**Category:** {state.triage.category}"]

    if state.remediation:
        parts += ["", "## Suggested fix", state.remediation.suggested_fix, "",
                  f"**Risk level:** {state.remediation.risk_level.value}"]
        if state.remediation.patch:
            parts += ["", "<details><summary>Proposed patch</summary>", "",
                      f"```diff\n{state.remediation.patch}\n```", "", "</details>"]

    parts += ["", "## Evidence"]
    if state.vision:
        parts.append(f"**Visible symptom:** {state.vision.visible_symptom}")
        if state.vision.api_response_notes:
            parts.append(f"**API response notes:** {state.vision.api_response_notes}")
    if state.api_response:
        parts += ["", "**API response captured:**",
                  f"```json\n{json.dumps(state.api_response, indent=2)[:3000]}\n```"]

    answered = [e for e in state.evidence if e.response]
    if answered:
        parts += ["", "## Additional evidence gathered"]
        for i, exchange in enumerate(answered, 1):
            parts += [f"{i}. **Q:** {exchange.question}", f"   **A:** {exchange.response}"]

    if state.retrieved_code:
        files = sorted({c.chunk.file_path for c in state.retrieved_code})
        parts += ["", "## Files considered during retrieval",
                  "\n".join(f"- `{f}`" for f in files)]

    parts += ["", "---", f"_Incident ID: `{state.incident_id}` — filed by opscopilot._"]
    return "\n".join(parts)
