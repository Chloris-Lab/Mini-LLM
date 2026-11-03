from mini_llm.sft import cap_share, supervised_tokens


def example(supervised, length=None):
    n = supervised if length is None else length
    return (list(range(n)), [True] * supervised + [False] * (n - supervised))


def test_supervised_tokens_counts_the_mask():
    assert supervised_tokens([example(3, 5), example(2)]) == 5


def test_cap_share_holds_the_token_share():
    base = [example(10) for _ in range(10)]
    extras = [example(10) for _ in range(20)]
    kept = cap_share(base, extras, 0.3)
    assert 0 < len(kept) < len(extras)
    share = supervised_tokens(kept) / (supervised_tokens(base) + supervised_tokens(kept))
    assert share <= 0.3
    assert share > 0.2


def test_cap_share_is_deterministic():
    base = [example(10) for _ in range(4)]
    extras = [example(5) for _ in range(50)]
    assert cap_share(base, extras, 0.3) == cap_share(base, extras, 0.3)


def test_cap_share_keeps_everything_when_the_share_is_wide():
    base = [example(10)]
    extras = [example(1) for _ in range(10)]
    assert len(cap_share(base, extras, 0.9)) == 10
