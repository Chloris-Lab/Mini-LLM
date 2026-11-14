from tokenizers import Tokenizer, models as tokenizer_models

from mini_llm.checkpoints import load_tokenizer, resolve_checkpoint
from mini_llm.config import assistant_pattern


def make_tok(path, extra=None):
    vocab = {"<eos>": 0, **(extra or {})}
    tok = Tokenizer(tokenizer_models.WordLevel(vocab, unk_token="<eos>"))
    tok.save(str(path))
    return tok


def test_load_tokenizer_walks_up_to_the_lineage_folder(tmp_path):
    lineage = tmp_path / "32k"
    ckpt = lineage / "40M" / "mini-llm-base-40M-1steps.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    make_tok(lineage / "tok.json")
    assert load_tokenizer(ckpt).get_vocab_size() == 1


def test_load_tokenizer_prefers_the_sibling_tok(tmp_path):
    lineage = tmp_path / "32k"
    ckpt = lineage / "40M" / "mini-llm-base-40M-1steps.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    make_tok(lineage / "tok.json")
    make_tok(ckpt.parent / "tok.json", extra={"a": 1})
    assert load_tokenizer(ckpt).get_vocab_size() == 2


def test_resolve_checkpoint_honours_root_and_skips_states(tmp_path):
    lineage = tmp_path / "digit32k"
    for name in ("petal-1-150M/petal-1-150M-base-100steps.pt", "petal-1-150M/petal-1-150M-base-900steps.state.pt",
                 "petal-1-150M/petal-1-150M-base-500steps.pt", "other/petal-1-150M-base-700steps.pt"):
        path = lineage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert resolve_checkpoint(root=lineage / "petal-1-150M").name == "petal-1-150M-base-500steps.pt"
    assert resolve_checkpoint(root=lineage).name == "petal-1-150M-base-700steps.pt"


def test_resolve_checkpoint_assistant_picks_the_plain_release(tmp_path):
    lineage = tmp_path / "digit32k"
    for name in ("petal-1-150M.pt", "petal-1-150M-base-53975steps.pt", "petal-1-150M.state.pt"):
        path = lineage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert resolve_checkpoint(root=lineage, pattern=assistant_pattern()).name == "petal-1-150M.pt"


def test_resolve_checkpoint_assistant_ignores_base_and_state(tmp_path):
    lineage = tmp_path / "digit32k"
    for name in ("petal-1-150M-base-53375steps.pt", "petal-1-150M.state.pt"):
        path = lineage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    try:
        resolve_checkpoint(root=lineage, pattern=assistant_pattern())
        assert False, "expected no assistant checkpoint"
    except SystemExit:
        pass


def test_resolve_checkpoint_assistant_accepts_dotted_versions(tmp_path):
    lineage = tmp_path / "digit32k"
    path = lineage / "petal-1.1-150M.pt"
    lineage.mkdir()
    path.write_bytes(b"x")
    assert resolve_checkpoint(root=lineage, pattern=assistant_pattern()).name == "petal-1.1-150M.pt"


def test_resolve_checkpoint_falls_back_to_legacy_names(tmp_path):
    lineage = tmp_path / "32k"
    path = lineage / "40M" / "mini-llm-base-40M-500steps.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert resolve_checkpoint(root=lineage / "40M").name == "mini-llm-base-40M-500steps.pt"
