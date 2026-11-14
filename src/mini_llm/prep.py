import argparse
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.checkpoints import sha256_file
from mini_llm.config import DATA_DIR, MODELS_DIR, Cfg, get_preset, tokenizer_path
from mini_llm.decontam import contamination_set, contaminated

SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.7),
    ("wikimedia/wikipedia", "20231101.en", 0.3),
]
SEED = 1
TOKENIZER_DOCS = 30_000
TOKENIZER_DOCS_LARGE_VOCAB = 300_000
VAL_TOKENS = 1_000_000
MIN_DOC_TOKENS = 200
BATCH = 1_000
VAL_DOCS = "val_docs.json"


def digit_pre_tokenizer():
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )


def texts(exhausted=None):
    rng = random.Random(SEED)
    streams = [
        [
            name,
            iter(
                load_dataset(name, subset, split="train", streaming=True).shuffle(
                    seed=SEED, buffer_size=10_000
                )
            ),
        ]
        for name, subset, _ in SOURCES
    ]
    weights = [w for _, _, w in SOURCES]
    while streams:
        i = rng.choices(range(len(streams)), weights=weights)[0]
        try:
            yield streams[i][0], next(streams[i][1])["text"]
        except StopIteration:
            if exhausted is not None:
                exhausted.append(streams[i][0])
            del streams[i]
            del weights[i]


def batches(items, size):
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def prefetch(items, size=8):
    q = queue.Queue(maxsize=size)
    done = object()

    def run():
        try:
            for it in items:
                q.put(it)
        except BaseException as e:
            q.put(e)
        finally:
            q.put(done)

    threading.Thread(target=run, daemon=True).start()
    while True:
        it = q.get()
        if it is done:
            return
        if isinstance(it, BaseException):
            raise it
        yield it


def doc_hash(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def stale_prep(meta_path):
    if not meta_path.exists():
        return False
    try:
        prev = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return True
    return "skipped_contaminated_docs" not in prev


def tokenizer_reuse_guard(tok_path, meta_path, allow_stale):
    stale = stale_prep(meta_path)
    if stale and not allow_stale:
        raise SystemExit(
            f"data/meta.json was written before the decontamination filter (no skipped_contaminated_docs); "
            f"{tok_path} would be reused as-is and data/val.bin likewise. "
            "Move data/ and the lineage tokenizer aside and re-run prep, "
            "or pass --allow-stale to keep them (not acceptable for reported runs)."
        )
    print(f"Reusing existing tokenizer {tok_path}: it will not be retrained or re-verified against the decontamination filter.")
    if stale:
        print("warning: proceeding with artifacts from before the decontamination filter (--allow-stale); not acceptable for reported runs")


def train_tokenizer(vocab, n_docs, lineage="", bad=None):
    path = tokenizer_path(vocab, lineage)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = digit_pre_tokenizer() if lineage else pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t0 = time.time()
    docs = (t for _, t in zip(range(n_docs), texts()) if bad is None or not contaminated(t, bad))
    tok.train_from_iterator(docs, trainer)
    print(f"tokenizer: vocab={vocab} lineage={lineage or 'classic'} from {n_docs} docs in {time.time() - t0:.0f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


def existing_val(vocab, lineage, tok_hash):
    docs = DATA_DIR / VAL_DOCS
    val = DATA_DIR / "val.bin"
    if not docs.exists() or not val.exists():
        return None, 0
    stored = json.loads(docs.read_text())
    if stored.get("vocab") != vocab or stored.get("lineage", "") != lineage:
        print(
            f"data/{VAL_DOCS} holds vocab={stored.get('vocab')} lineage={stored.get('lineage', '')!r}: "
            f"rebuilding val.bin for vocab={vocab} lineage={lineage!r}"
        )
        return None, 0
    if "tokenizer_sha256" not in stored:
        print(f"data/{VAL_DOCS} has no tokenizer hash: recording the current one, keeping val.bin")
        stored["tokenizer_sha256"] = tok_hash
        docs.write_text(json.dumps(stored))
        return set(stored["hashes"]), val.stat().st_size // 2
    if stored["tokenizer_sha256"] != tok_hash:
        print(f"data/{VAL_DOCS} was built with another tokenizer: rebuilding val.bin")
        return None, 0
    return set(stored["hashes"]), val.stat().st_size // 2


def mixture_report(meta):
    total = meta["train_tokens"]
    if not total:
        return
    rows = " | ".join(
        f"{s['name'].split('/')[-1]} {s['tokens'] / total:5.1%} ({s['tokens'] / 1e6:.0f}M)"
        for s in meta["sources"]
    )
    print(
        f"train: {total / 1e6:.1f}M tokens | {meta['train_bytes'] / 1e9:.1f} GB | "
        f"{meta['train_bytes'] / total:.2f} bytes/token"
    )
    print(f"mixture by tokens: {rows}")


def prep(target_millions=120, size=None, vocab=None, tokenizer_docs=None, lineage="", decontam=True, allow_stale=False):
    if vocab is None:
        vocab = get_preset(size).vocab if size else Cfg().vocab
    if lineage is None:
        lineage = get_preset(size).lineage if size else ""
    if tokenizer_docs is None:
        tokenizer_docs = TOKENIZER_DOCS_LARGE_VOCAB if vocab > Cfg().vocab else TOKENIZER_DOCS
    tok_path = tokenizer_path(vocab, lineage)
    if tok_path.exists():
        existing = Tokenizer.from_file(str(tok_path)).get_vocab_size()
        if existing != vocab:
            raise SystemExit(
                f"{tok_path} has vocab={existing} but --vocab={vocab} was requested. "
                "Move it aside or drop --vocab."
            )
        tokenizer_reuse_guard(tok_path, DATA_DIR / "meta.json", allow_stale)
    bad = None
    if decontam:
        print("Loading benchmark test sets for the 13-gram decontamination index ...")
        bad = contamination_set()
        print(f"Decontamination index: {len(bad)} n-grams; overlapping documents will be dropped.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not tok_path.exists():
        print("Training the BPE tokenizer...")
        train_tokenizer(vocab, tokenizer_docs, lineage, bad=bad)
    tok = Tokenizer.from_file(str(tok_path))
    eos = tok.token_to_id("<eos>")
    tok_hash = sha256_file(tok_path)

    held, val_tokens = existing_val(vocab, lineage, tok_hash)
    writing_val = held is None
    val_hashes = []
    target = int(target_millions * 1_000_000) + (VAL_TOKENS if writing_val else 0)
    progress = 0
    train_tokens = 0
    train_bytes = 0
    docs_per_source = {}
    tokens_per_source = {}
    bytes_per_source = {}
    skipped = 0
    contaminated_docs = 0
    names = ", ".join(f"{name} ({w:.0%} of docs)" for name, _, w in SOURCES)
    print(f"Tokenizing {names} (target: {target / 1e6:.0f}M tokens, min {MIN_DOC_TOKENS} tokens/doc)...")
    if not writing_val:
        print(f"reusing data/val.bin ({val_tokens / 1e6:.1f}M tokens) and excluding its documents from train")
    t0 = time.time()
    nb = 0
    exhausted = []
    train_f = open(DATA_DIR / "train.bin", "wb")
    val_f = open(DATA_DIR / "val.bin", "wb") if writing_val else None
    try:
        for batch in batches(prefetch(texts(exhausted)), BATCH):
            nb += 1
            encs = tok.encode_batch([text for _, text in batch])
            val_ids, train_ids = [], []
            for (src, text), enc in zip(batch, encs):
                h = doc_hash(text)
                if bad is not None and contaminated(text, bad):
                    contaminated_docs += 1
                    continue
                if held is not None and h in held:
                    continue
                ids = enc.ids + [eos]
                if len(ids) < MIN_DOC_TOKENS:
                    skipped += 1
                    continue
                if val_f is not None and val_tokens + len(ids) <= VAL_TOKENS:
                    val_ids.extend(ids)
                    val_tokens += len(ids)
                    val_hashes.append(h)
                else:
                    train_ids.extend(ids)
                    train_tokens += len(ids)
                    nbytes = len(text.encode())
                    train_bytes += nbytes
                    bytes_per_source[src] = bytes_per_source.get(src, 0) + nbytes
                    tokens_per_source[src] = tokens_per_source.get(src, 0) + len(ids)
                    docs_per_source[src] = docs_per_source.get(src, 0) + 1
                progress += len(ids)
            if val_ids:
                np.asarray(val_ids, dtype=np.uint16).tofile(val_f)
            if train_ids:
                np.asarray(train_ids, dtype=np.uint16).tofile(train_f)
            if nb % 20 == 0:
                el = max(1e-6, time.time() - t0)
                rate = progress / el / 1e6
                eta = max(0, target - progress) / max(1e-6, rate * 1e6) / 60
                print(f"  {progress / 1e6:.1f}M / {target / 1e6:.0f}M tokens | {rate:.2f}M tok/s | ETA {eta:.1f}min", flush=True)
            if progress >= target:
                break
    finally:
        train_f.close()
        if val_f is not None:
            val_f.close()
    for src in exhausted:
        print(f"warning: source exhausted before target: {src}")
    if writing_val:
        (DATA_DIR / VAL_DOCS).write_text(
            json.dumps(
                {"vocab": vocab, "lineage": lineage, "tokenizer_sha256": tok_hash, "hashes": val_hashes}
            )
        )
    meta = {
        "vocab": vocab,
        "lineage": lineage,
        "tokenizer_sha256": tok_hash,
        "seed": SEED,
        "min_doc_tokens": MIN_DOC_TOKENS,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_bytes": train_bytes,
        "sources": [
            {
                "name": name,
                "subset": subset,
                "weight": weight,
                "docs": docs_per_source.get(name, 0),
                "tokens": tokens_per_source.get(name, 0),
                "bytes": bytes_per_source.get(name, 0),
            }
            for name, subset, weight in SOURCES
        ],
        "skipped_short_docs": skipped,
        "skipped_contaminated_docs": contaminated_docs,
        "complete": progress >= target,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    mixture_report(meta)
    print(f"val: {val_tokens / 1e6:.2f}M tokens | skipped {skipped} docs under {MIN_DOC_TOKENS} tokens | decontaminated {contaminated_docs} docs")
    if progress < target:
        print(f"STOPPED SHORT: {progress / 1e6:.1f}M / {target / 1e6:.0f}M tokens (a source ran dry)")
        sys.stdout.flush()
        os._exit(1)
    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("millions", type=float, nargs="?", default=120)
    ap.add_argument("--size", default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--tokenizer-docs", type=int, default=None)
    ap.add_argument("--lineage", default=None, choices=["", "digit32k"],
                    help="tokenizer lineage to prepare (default: follow --size)")
    ap.add_argument("--no-decontam", dest="decontam", action="store_false",
                    help="skip the 13-gram decontamination of the corpus (not allowed for reported runs)")
    ap.add_argument("--allow-stale", dest="allow_stale", action="store_true",
                    help="reuse a tokenizer and data written before the decontamination filter (not allowed for reported runs)")
    args = ap.parse_args()
    prep(args.millions, args.size, args.vocab, args.tokenizer_docs, args.lineage, args.decontam, args.allow_stale)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
