"""Command-line interface.

Uses argparse rather than typer so the CLI works on a bare
`pip install opscopilot-core` — a CLI that requires an extra to run is a
worse first experience than one with plainer help text.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .engine import OpsCopilot
from .exceptions import OpsCopilotError


def _engine(args: argparse.Namespace) -> OpsCopilot:
    overrides: dict[str, Any] = {}
    for key in ("llm", "embeddings", "vector_store", "state_store"):
        value = getattr(args, key, None)
        if value:
            overrides[key] = value
    return OpsCopilot.from_env(**overrides)


def _print_state(state: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(state.to_dict(), indent=2))
        return

    print(f"incident : {state.incident_id}")
    print(f"status   : {state.status.value}")
    if state.triage:
        print(f"severity : {state.triage.severity.value} ({state.triage.category})")
    if state.diagnosis:
        print(f"confidence: {state.diagnosis.confidence:.2f}")
        print(f"suspect  : {state.diagnosis.suspected_file} :: {state.diagnosis.suspected_symbol}")
        print(f"diagnosis: {state.diagnosis.summary}")
    if state.pending_question:
        print(f"\nQUESTION : {state.pending_question}")
        print(f"  reply with: opscopilot evidence {state.incident_id} \"your answer\"")
    if state.remediation:
        print(f"\nfix      : {state.remediation.suggested_fix}")
        print(f"risk     : {state.remediation.risk_level.value}")
        print(f"  approve with: opscopilot approve {state.incident_id}")
    if state.ticket:
        print(f"\nticket   : {state.ticket.url or state.ticket.error}")
    if state.error:
        print(f"\nerror    : {state.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opscopilot", description="Agentic SRE triage engine.")
    parser.add_argument("--version", action="version", version=f"opscopilot-core {__version__}")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument("--llm", help="override LLM spec, e.g. openai:gpt-4o-mini")
    parser.add_argument("--embeddings", help="override embeddings spec, e.g. hash:384")
    parser.add_argument("--vector-store", dest="vector_store", help="memory:// or http://host:6333")
    parser.add_argument("--state-store", dest="state_store", help="memory:// or redis://...")

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index a repository for retrieval")
    p_index.add_argument("path")
    p_index.add_argument("--collection")

    p_search = sub.add_parser("search", help="query the code index directly")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)

    p_submit = sub.add_parser("submit", help="submit a new incident")
    p_submit.add_argument("--note", help="short description of the bug")
    p_submit.add_argument("--screenshot")
    p_submit.add_argument("--api-response", help="path to a JSON file")
    p_submit.add_argument("--logs", help="path to a log file")

    p_evidence = sub.add_parser("evidence", help="answer a pending question")
    p_evidence.add_argument("incident_id")
    p_evidence.add_argument("response")

    p_approve = sub.add_parser("approve", help="approve a pending remediation")
    p_approve.add_argument("incident_id")

    p_reject = sub.add_parser("reject", help="reject a pending remediation")
    p_reject.add_argument("incident_id")
    p_reject.add_argument("--reason")

    p_show = sub.add_parser("show", help="print an incident's state")
    p_show.add_argument("incident_id")

    sub.add_parser("list", help="list known incident ids")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cop = _engine(args)

        if args.command == "index":
            def progress(done: int, total: int) -> None:
                print(f"\r  embedding {done}/{total} chunks", end="", file=sys.stderr)

            count = cop.index_repository(args.path, collection=args.collection, progress=progress)
            print(f"\rIndexed {count} chunks from {args.path}".ljust(60))
            return 0

        if args.command == "search":
            for hit in cop.search(args.query, top_k=args.top_k):
                print(f"{hit.score:.3f}  {hit.chunk.file_path}:{hit.chunk.start_line}  "
                      f"{hit.chunk.symbol_name or ''}")
            return 0

        if args.command == "submit":
            api_response = None
            if args.api_response:
                with open(args.api_response, encoding="utf-8") as handle:
                    api_response = json.load(handle)
            logs = None
            if args.logs:
                with open(args.logs, encoding="utf-8") as handle:
                    logs = handle.read()
            state = cop.submit(
                user_note=args.note,
                screenshot_path=args.screenshot,
                api_response=api_response,
                logs=logs,
            )
        elif args.command == "evidence":
            state = cop.provide_evidence(args.incident_id, args.response)
        elif args.command == "approve":
            state = cop.approve(args.incident_id)
        elif args.command == "reject":
            state = cop.reject(args.incident_id, args.reason)
        elif args.command == "show":
            state = cop.get(args.incident_id)
        elif args.command == "list":
            for incident_id in cop.list_incidents():
                print(incident_id)
            return 0
        else:  # pragma: no cover
            return 2

        _print_state(state, args.json)
        return 0

    except OpsCopilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
