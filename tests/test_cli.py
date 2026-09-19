from __future__ import annotations

from opscopilot.cli import build_parser, main


def test_parser_requires_a_subcommand(capsys):
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_index_then_search_against_a_persisted_index(tmp_path, repo, capsys):
    index = tmp_path / "index.json"
    store = f"memory://{index}"

    assert main(["--embeddings", "hash:256", "--vector-store", store, "index", str(repo)]) == 0
    capsys.readouterr()

    # A separate invocation: this only works because the index was persisted,
    # which is the difference between 'memory://' and 'memory:///path'.
    assert main([
        "--embeddings", "hash:256", "--vector-store", store, "search", "total price",
    ]) == 0
    assert "cart.py" in capsys.readouterr().out


def test_unknown_incident_exits_nonzero_with_a_message(capsys):
    assert main(["--state-store", "memory://", "show", "missing-id"]) == 1
    assert "error:" in capsys.readouterr().err
