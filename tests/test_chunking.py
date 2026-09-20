from __future__ import annotations

from opscopilot.retrieval.chunking import chunk_repository, chunk_source


def test_python_chunks_are_named_and_include_decorators():
    source = '@app.post("/x")\ndef handler(body):\n    return 1\n'
    chunks = chunk_source(source, "api.py")
    assert [c.symbol_name for c in chunks] == ["handler"]
    assert "@app.post" in chunks[0].text
    assert chunks[0].start_line == 1


def test_chunk_ids_are_deterministic_so_reindexing_is_idempotent():
    source = "def a():\n    return 1\n"
    assert chunk_source(source, "m.py")[0].id == chunk_source(source, "m.py")[0].id


def test_same_code_in_a_different_file_gets_a_different_id():
    source = "def a():\n    return 1\n"
    assert chunk_source(source, "one.py")[0].id != chunk_source(source, "two.py")[0].id


def test_syntax_error_falls_back_instead_of_raising():
    chunks = chunk_source("def broken(:\n", "bad.py")
    assert chunks and chunks[0].chunk_type == "window"


def test_jsx_components_are_detected():
    source = "export default function ProductCard({p}) {\n  return <div/>\n}\n"
    assert chunk_source(source, "ProductCard.jsx")[0].symbol_name == "ProductCard"


def test_sql_splits_per_table():
    source = "CREATE TABLE users (id INT);\nCREATE TABLE carts (id INT);"
    assert [c.symbol_name for c in chunk_source(source, "schema.sql")] == ["users", "carts"]


def test_repository_walk_skips_vendored_directories(tmp_path):
    (tmp_path / "app.py").write_text("def a():\n    return 1\n")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "index.js").write_text("function b(){}\n")
    assert {c.file_path for c in chunk_repository(tmp_path)} == {"app.py"}


def test_python_with_invalid_escapes_does_not_emit_warnings():
    import warnings

    from opscopilot.retrieval.chunking import chunk_python

    # The repo being indexed is user code; its invalid escapes must not leak
    # SyntaxWarning("<unknown>:N ...") into the host application's output.
    source = 'import re\n\n\ndef f():\n    return re.compile("\\W+")\n'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunks = chunk_python(source, "app/bad_escape.py")
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]
    assert any(c.symbol_name == "f" for c in chunks)  # parsed as Python, not fallback
