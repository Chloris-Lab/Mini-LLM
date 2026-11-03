from tokenizers import Tokenizer, models as tokenizer_models

from mini_llm.checkpoints import load_tokenizer, resolve_checkpoint


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
    for name in ("40M/mini-llm-base-40M-100steps.pt", "40M/mini-llm-base-40M-900steps.state.pt",
                 "40M/mini-llm-base-40M-500steps.pt", "150M/mini-llm-base-150M-700steps.pt"):
        path = lineage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert resolve_checkpoint(root=lineage / "40M").name == "mini-llm-base-40M-500steps.pt"
    assert resolve_checkpoint(root=lineage).name == "mini-llm-base-150M-700steps.pt"
