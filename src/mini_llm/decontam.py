import hashlib
import re

NGRAM_SIZE = 13
NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def normalize(text):
    return " ".join(NORMALIZE_RE.sub(" ", text.lower()).split())


def _ngram_hash(ngram):
    return int.from_bytes(hashlib.blake2b(ngram.encode(), digest_size=8).digest(), "little")


def ngram_hashes(text, n=NGRAM_SIZE):
    words = normalize(text).split()
    return {_ngram_hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)}


def contaminated(text, bad, n=NGRAM_SIZE):
    words = normalize(text).split()
    for i in range(len(words) - n + 1):
        if _ngram_hash(" ".join(words[i : i + n])) in bad:
            return True
    return False


PRODUCTION_SPECS = [
    ("Rowan/hellaswag", None, "test", lambda d: d.get("ctx") or f"{d.get('ctx_a', '')} {d.get('ctx_b', '')}"),
    ("allenai/ai2_arc", "ARC-Easy", "test", lambda d: d.get("question", "")),
    ("allenai/ai2_arc", "ARC-Challenge", "test", lambda d: d.get("question", "")),
    ("baber/piqa", None, "test", lambda d: d.get("goal", "")),
    ("allenai/winogrande", "winogrande_xl", "test", lambda d: d.get("sentence", "")),
    ("allenai/openbookqa", "main", "test", lambda d: d.get("question_stem", "")),
    ("allenai/sciq", None, "test", lambda d: d.get("question", "")),
    ("EleutherAI/lambada_openai", None, "test", lambda d: d.get("text", "")),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", lambda d: d.get("text", "")),
    ("openai/gsm8k", "main", "test", lambda d: d.get("question", "")),
]


def texts_from_specs(specs):
    for name, subset, split, extract in specs:
        from datasets import load_dataset

        ds = load_dataset(name, subset, split=split, streaming=True)
        for row in ds:
            text = extract(row)
            if text and str(text).strip():
                yield str(text)


def contamination_set_from_texts(texts, n=NGRAM_SIZE):
    bad = set()
    for text in texts:
        if str(text).strip():
            bad |= ngram_hashes(str(text), n)
    return bad


def contamination_set(specs=None, n=NGRAM_SIZE):
    if specs is None:
        specs = PRODUCTION_SPECS
    bad = set()
    for text in texts_from_specs(specs):
        bad |= ngram_hashes(text, n)
    return bad
