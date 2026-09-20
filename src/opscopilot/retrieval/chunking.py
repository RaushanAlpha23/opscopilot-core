"""Split source files into retrievable units.

Carried over from the application (which chunks correctly) rather than the
old SDK (which did not). Two substantive differences from the SDK version:

* Python uses the stdlib `ast` module, not `tree_sitter_languages`. That
  dependency ships prebuilt binary grammars, is effectively unmaintained, and
  is a common install failure on newer Python versions — a hard dependency on
  it would make the package uninstallable for some users for no gain, since
  `ast` parses Python exactly.
* Chunks carry `file_path`, `symbol_name` and line numbers. The SDK stored
  only `{filepath, code}`, so a retrieved hit could not be pointed back at a
  line, which is what the remediation step needs to reference.
"""

from __future__ import annotations

import ast
import hashlib
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

from ..models import CodeChunk

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".next", ".nuxt", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "target", "vendor", "coverage", ".tox", "site-packages",
}

PYTHON_EXT = {".py", ".pyi"}
JS_LIKE_EXT = {".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"}
SQL_EXT = {".sql"}
SUPPORTED_EXT = PYTHON_EXT | JS_LIKE_EXT | SQL_EXT

MAX_FILE_BYTES = 1_000_000  # skip minified bundles and vendored blobs
FALLBACK_CHUNK_LINES = 60
FALLBACK_OVERLAP_LINES = 10

JS_BOUNDARY_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:\(|async|function)"
    r"|let\s+\w+\s*=\s*(?:\(|async|function)|\w+\s*\([^)]*\)\s*\{)",
    re.MULTILINE,
)
JS_NAME_RE = re.compile(r"(?:function|class|const|let)\s+(\w+)")


def _chunk_id(file_path: str, start_line: int, text: str) -> str:
    """Stable, content-derived id.

    The old SDK used `uuid4()` per chunk, which meant re-indexing the same
    repo appended a full duplicate set to the collection instead of updating
    it in place. A deterministic id makes re-indexing idempotent.
    """
    digest = hashlib.blake2b(
        f"{file_path}:{start_line}:{text}".encode(), digest_size=16
    ).hexdigest()
    # Format as a UUID so Qdrant accepts it as a point id.
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _make(text: str, file_path: str, language: str, chunk_type: str,
          symbol: str | None, start: int, end: int | None) -> CodeChunk:
    header = f"# File: {file_path}"
    if symbol:
        header += f"\n# {chunk_type}: {symbol}"
    body = f"{header}\n\n{text}"
    return CodeChunk(
        id=_chunk_id(file_path, start, text),
        text=body,
        file_path=file_path,
        language=language,
        chunk_type=chunk_type,
        symbol_name=symbol,
        start_line=start,
        end_line=end,
    )


def chunk_python(source: str, file_path: str) -> list[CodeChunk]:
    try:
        # We are parsing the *user's* repository, not our own code. Their invalid
        # escapes ("\\W" in a non-raw string) make the parser emit SyntaxWarning
        # with filename "<unknown>". That is noise for us, and unactionable
        # without a filename, so silence it and name the file for real errors.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return chunk_fallback(source, file_path, "python")

    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        # Include decorators: `@app.post("/incidents")` is exactly the kind of
        # line that makes a chunk findable, and node.lineno points past it.
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        end = getattr(node, "end_lineno", None) or (start + 1)
        body = "\n".join(lines[start:end])
        if not body.strip():
            continue
        chunks.append(
            _make(body, file_path, "python", type(node).__name__, node.name, start + 1, end)
        )

    return chunks or chunk_fallback(source, file_path, "python")


def chunk_js_like(source: str, file_path: str, language: str = "javascript") -> list[CodeChunk]:
    lines = source.splitlines()
    offsets = [m.start() for m in JS_BOUNDARY_RE.finditer(source)]
    if not offsets:
        return chunk_fallback(source, file_path, language)

    starts = sorted({source.count("\n", 0, o) for o in offsets})
    starts.append(len(lines))

    chunks: list[CodeChunk] = []
    for i in range(len(starts) - 1):
        start, end = starts[i], starts[i + 1]
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        match = JS_NAME_RE.search(lines[start] if start < len(lines) else "")
        symbol = match.group(1) if match else None
        chunks.append(_make(body, file_path, language, "declaration", symbol, start + 1, end))
    return chunks or chunk_fallback(source, file_path, language)


def chunk_sql(source: str, file_path: str) -> list[CodeChunk]:
    """Split on statement boundaries so each CREATE TABLE is its own chunk."""
    statements = [s.strip() for s in source.split(";") if s.strip()]
    if not statements:
        return []
    chunks: list[CodeChunk] = []
    line = 1
    for stmt in statements:
        match = re.search(r"(?:create|alter)\s+table\s+(?:if\s+not\s+exists\s+)?[`\"\[]?(\w+)",
                          stmt, re.IGNORECASE)
        symbol = match.group(1) if match else None
        chunks.append(_make(stmt, file_path, "sql", "statement", symbol, line, None))
        line += stmt.count("\n") + 1
    return chunks


def chunk_fallback(source: str, file_path: str, language: str) -> list[CodeChunk]:
    """Overlapping fixed-size windows for anything without a parser."""
    lines = source.splitlines()
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    step = max(1, FALLBACK_CHUNK_LINES - FALLBACK_OVERLAP_LINES)
    for start in range(0, len(lines), step):
        end = min(start + FALLBACK_CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append(_make(body, file_path, language, "window", None, start + 1, end))
        if end == len(lines):
            break
    return chunks


def chunk_source(source: str, file_path: str) -> list[CodeChunk]:
    """Dispatch to the right chunker based on the file's extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_EXT:
        return chunk_python(source, file_path)
    if suffix in JS_LIKE_EXT:
        language = "typescript" if suffix in {".ts", ".tsx"} else "javascript"
        return chunk_js_like(source, file_path, language)
    if suffix in SQL_EXT:
        return chunk_sql(source, file_path)
    return chunk_fallback(source, file_path, "unknown")


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".") for part in path.parts[:-1])


def walk_source_files(
    root: str | Path, extensions: set[str] | None = None
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute_path, path_relative_to_root) for every indexable file."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if root_path.is_file():
        yield root_path, root_path.name
        return

    allowed = extensions or SUPPORTED_EXT
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or should_skip(path.relative_to(root_path)):
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path, str(path.relative_to(root_path))


def chunk_repository(root: str | Path, extensions: set[str] | None = None) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for path, relative in walk_source_files(root, extensions):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if source.strip():
            chunks.extend(chunk_source(source, relative))
    return chunks
