import pytest

from mini_llm.config import get_preset
from mini_llm.pretrain import resolve_cfg


def test_no_override_returns_preset_verbatim():
    assert resolve_cfg("40M") == get_preset("40M")
    assert resolve_cfg("150M") == get_preset("150M")


def test_partial_override_keeps_rest_of_preset():
    cfg = resolve_cfg("40M", ctx=2048)
    assert cfg.ctx == 2048
    assert cfg.vocab == 32768
    assert (cfg.dim, cfg.layers, cfg.heads, cfg.ffn) == (512, 8, 8, 1408)


def test_full_override_builds_custom_architecture():
    cfg = resolve_cfg("40M", dim=640, layers=12, heads=10, ffn=1728)
    assert (cfg.dim, cfg.layers, cfg.heads, cfg.ffn) == (640, 12, 10, 1728)
    assert cfg.vocab == 32768  



def test_heads_must_divide_dim():
    with pytest.raises(SystemExit, match="divisible"):
        resolve_cfg("40M", heads=10)


def test_overrides_must_be_positive():
    with pytest.raises(SystemExit, match="positive"):
        resolve_cfg("40M", layers=0)
    with pytest.raises(SystemExit, match="positive"):
        resolve_cfg("40M", ffn=-1408)


def test_preset_copy_is_not_mutated():
    before = get_preset("40M")
    resolve_cfg("40M", ctx=2048, vocab=16384)
    assert get_preset("40M") == before
