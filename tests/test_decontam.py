from mini_llm.decontam import (
    contaminated,
    contamination_set_from_texts,
    normalize,
    ngram_hashes,
)


def test_normalize_strips_punctuation_and_case():
    assert normalize("Hello, WORLD! 42.") == "hello world 42"


def test_ngrams_are_deterministic():
    assert ngram_hashes("a b c d e f g h i j k l m n o p") == ngram_hashes("a b c d e f g h i j k l m n o p")


def test_detects_13_gram_overlap():
    bad = contamination_set_from_texts(
        ["the quick brown fox jumps over the lazy dog while the sun shines brightly on the hill"], n=13
    )
    assert contaminated("prefix text the quick brown fox jumps over the lazy dog while the sun shines suffix", bad, n=13)


def test_short_text_below_window_never_matches():
    bad = contamination_set_from_texts(["a b c d e f g h i j k l m n o p"], n=13)
    assert not contaminated("a b c", bad, n=13)


def test_unrelated_text_does_not_match():
    bad = contamination_set_from_texts(
        ["the quick brown fox jumps over the lazy dog while the sun shines brightly on the hill"], n=13
    )
    assert not contaminated("how to bake a sourdough bread with flour water salt and patience over the weekend", bad, n=13)


def test_exact_words_must_be_contiguous():
    bad = contamination_set_from_texts(["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"], n=13)
    assert contaminated("start alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu end", bad, n=13)
    assert not contaminated("start alpha gamma beta delta epsilon zeta eta theta iota kappa lambda mu nu end", bad, n=13)



