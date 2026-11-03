import json

import numpy as np

from mini_llm import prep


def test_doc_hash_is_stable_and_distinguishes_bytes():
    assert prep.doc_hash("hello") == prep.doc_hash("hello")
    assert prep.doc_hash("hello") != prep.doc_hash("hello ")
    assert len(prep.doc_hash("x")) == 32


def test_existing_val_returns_none_without_files(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "DATA_DIR", tmp_path)
    assert prep.existing_val(32768, "digit32k", "abc") == (None, 0)


def test_existing_val_reuses_a_matching_split(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "DATA_DIR", tmp_path)
    (tmp_path / "val_docs.json").write_text(
        json.dumps(
            {"vocab": 32768, "lineage": "digit32k", "tokenizer_sha256": "abc", "hashes": ["aa", "bb"]}
        )
    )
    np.asarray([1, 2, 3, 4], dtype=np.uint16).tofile(tmp_path / "val.bin")
    hashes, tokens = prep.existing_val(32768, "digit32k", "abc")
    assert hashes == {"aa", "bb"}
    assert tokens == 4


def test_existing_val_rebuilds_on_a_different_lineage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep, "DATA_DIR", tmp_path)
    (tmp_path / "val_docs.json").write_text(
        json.dumps({"vocab": 16384, "lineage": "", "tokenizer_sha256": "abc", "hashes": ["aa"]})
    )
    np.asarray([1, 2], dtype=np.uint16).tofile(tmp_path / "val.bin")
    assert prep.existing_val(32768, "digit32k", "abc") == (None, 0)
    assert "rebuilding val.bin" in capsys.readouterr().out


def test_existing_val_rebuilds_on_a_different_tokenizer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep, "DATA_DIR", tmp_path)
    (tmp_path / "val_docs.json").write_text(
        json.dumps({"vocab": 32768, "lineage": "digit32k", "tokenizer_sha256": "old", "hashes": ["aa"]})
    )
    np.asarray([1, 2], dtype=np.uint16).tofile(tmp_path / "val.bin")
    assert prep.existing_val(32768, "digit32k", "new") == (None, 0)
    assert "another tokenizer" in capsys.readouterr().out


def test_existing_val_stamps_the_hash_on_a_legacy_split(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep, "DATA_DIR", tmp_path)
    (tmp_path / "val_docs.json").write_text(
        json.dumps({"vocab": 32768, "lineage": "digit32k", "hashes": ["aa"]})
    )
    np.asarray([1, 2], dtype=np.uint16).tofile(tmp_path / "val.bin")
    assert prep.existing_val(32768, "digit32k", "new") == ({"aa"}, 2)
    stored = json.loads((tmp_path / "val_docs.json").read_text())
    assert stored["tokenizer_sha256"] == "new"
    assert "recording the current one" in capsys.readouterr().out


def test_mixture_report_prints_token_shares(capsys):
    meta = {
        "train_tokens": 1000,
        "train_bytes": 4000,
        "sources": [
            {"name": "a/one", "tokens": 700, "bytes": 2800},
            {"name": "b/two", "tokens": 300, "bytes": 1200},
        ],
    }
    prep.mixture_report(meta)
    out = capsys.readouterr().out
    assert "70.0%" in out and "30.0%" in out
    assert "4.00 bytes/token" in out
