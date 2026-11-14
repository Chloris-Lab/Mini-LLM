import argparse
import math
import random
import re
import sys
import time
from contextlib import nullcontext

import torch
from datasets import load_dataset

from mini_llm.checkpoints import (
    load_model,
    load_tokenizer,
    read_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from mini_llm.config import model_stem, resolve_device, size_millions
from mini_llm import math_data
from mini_llm.identity import arithmetic_pairs, identity_pairs
from mini_llm.prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt, format_prompt
from mini_llm.decontam import contamination_set, contaminated
from mini_llm.system_prompts import (
    MATH_SYSTEM_PROMPT,
    SYSTEM_PARAPHRASES,
    date_pairs,
    sample_date,
    variation_pairs,
)

VAL_SIZE = 500
MAX_TURNS = 4
AI_MAX_SHARE = 0.25
MATH_MAX_SHARE = 0.3
IDENTITY_REPEAT = 5
VARIATION_REPEAT = 5
MATH_COUNT = 75_000
STORY_COUNT = 20_000
SKILL_COUNT = 10_000
DATE_COUNT = 400
DATE_SHARE = 0.9
PARAPHRASE_SHARE = 0.15
GSM8K_MAX = 4000
IDENTITY_LEAK = re.compile(
    r"open ?assistant|\blaion\b|chat ?gpt|openai"
    r"|as an ai( language)? model"
    r"|i('m| am) (an? )?(ai|artificial intelligence|language model|large language model|virtual assistant)",
    re.IGNORECASE,
)


def prompt_builder(with_system, seed):
    rng = random.Random(seed)

    def build(base=DEFAULT_SYSTEM_PROMPT):
        if not with_system:
            return None
        if base is DEFAULT_SYSTEM_PROMPT and rng.random() < PARAPHRASE_SHARE:
            base = rng.choice(SYSTEM_PARAPHRASES)
        if rng.random() >= DATE_SHARE:
            return base
        return build_system_prompt(base, date=sample_date(rng))

    return build


def dolly_pairs():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    plain, with_context = [], []
    for d in ds:
        if d["context"].strip():
            with_context.append((f"{d['context'].strip()}\n\n{d['instruction']}", d["response"]))
        else:
            plain.append((d["instruction"], d["response"]))
    return plain, with_context


def no_robots_pairs():
    ds = load_dataset("HuggingFaceH4/no_robots")
    split = "train_sft" if "train_sft" in ds else "train"
    pairs, conversations = [], []
    for d in ds[split]:
        msgs = [m for m in d["messages"] if m["role"] != "system"]
        turns = []
        for i in range(0, len(msgs) - 1, 2):
            question, answer = msgs[i], msgs[i + 1]
            if question["role"] != "user" or answer["role"] != "assistant" or not answer["content"].strip():
                break
            turns.append([question["content"], answer["content"]])
        if not turns:
            continue
        pairs.append(turns[0])
        if len(turns) >= 2:
            conversations.append(turns[:MAX_TURNS])
    return pairs, conversations


def oasst_pairs():
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    ds = [d for d in ds if d["lang"] == "en" and not d["deleted"] and d["review_result"]]
    by_parent = {}
    for d in ds:
        by_parent.setdefault(d["parent_id"], []).append(d)

    def top_reply(parent_id):
        replies = [c for c in by_parent.get(parent_id, []) if c["role"] == "assistant"]
        return min(replies, key=lambda r: r["rank"] if r["rank"] is not None else 999) if replies else None

    def chain(start):
        turns, node = [], start
        while node is not None and len(turns) < MAX_TURNS:
            reply = top_reply(node["message_id"])
            if reply is None or reply["rank"] != 0 or not reply["text"].strip() or not node["text"].strip():
                break
            turns.append([node["text"], reply["text"]])
            followups = [
                c for c in by_parent.get(reply["message_id"], [])
                if c["role"] == "prompter" and c["text"].strip()
            ]
            node = min(followups, key=lambda c: (len(c["text"]), c["message_id"])) if followups else None
        return turns

    pairs, conversations = [], []
    for prompter in [d for d in ds if d["role"] == "prompter" and d["parent_id"] is None]:
        turns = chain(prompter)
        if not turns:
            continue
        pairs.append(turns[0])
        if len(turns) >= 2:
            conversations.append(turns)
    return pairs, conversations


def alpaca_pairs():
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    return [(d["instruction"], d["output"]) for d in ds if not d["input"].strip()]


def gsm8k_pairs():
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception:
        ds = load_dataset("gsm8k", "main", split="train")
    out = []
    for d in ds:
        raw = d["answer"]
        if "####" not in raw:
            continue
        steps, final = raw.rsplit("####", 1)
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        final = final.strip().replace(",", "")
        if not steps or not final or final.count("-") > 0:
            continue
        if steps.count("\n") > 2 or len(d["question"].split()) > 80:
            continue
        out.append((d["question"].strip(), f"{steps} The answer is {final}."))
    random.Random(0).shuffle(out)
    return out[:GSM8K_MAX]


def conversation_segments(system, turns):
    segments = []
    if system:
        segments.append(("System: " + system.strip() + "\n", False))
    for question, answer in turns:
        segments.append((f"User: {question.strip()}\nAssistant:", False))
        segments.append((" " + answer.strip(), True))
    return segments


def pair_segments(system, question, answer):
    return [(format_prompt(question, system), False), (" " + answer.strip(), True)]


def encode_segments(tok, segments, eos, ctx):
    ids, mask = [], []
    for text, supervised in segments:
        enc = tok.encode(text)
        ids.extend(enc.ids)
        mask.extend([supervised] * len(enc.ids))
        if supervised:
            ids.append(eos)
            mask.append(True)
    if not ids or len(ids) > ctx:
        return None
    return ids, mask


def encode(name, items, tok, eos, ctx, clean=True):
    kept = []
    for segments in items:
        if clean:
            has_leak = any(
                IDENTITY_LEAK.search(text) for text, supervised in segments if supervised
            )
            if has_leak:
                continue
        enc = encode_segments(tok, segments, eos, ctx)
        if enc:
            kept.append(enc)
    print(f"{name}: {len(kept)}/{len(items)} kept")
    return kept


def load_examples(tok, eos, ctx, with_system, chat_only=False, decontam=True):
    bad = contamination_set() if decontam else None
    if bad is not None:
        print(f"decontamination index: {len(bad)} n-grams (SFT)")
    human, ai = [], []
    build = prompt_builder(with_system, 7)
    dolly_plain, dolly_context = dolly_pairs()
    no_robots, no_robots_multi = no_robots_pairs()
    oasst, oasst_multi = oasst_pairs()

    def clean(pairs):
        if bad is None:
            return pairs
        return [(q, a) for q, a in pairs if not contaminated(q, bad)]

    def clean_conversations(conversations):
        if bad is None:
            return conversations
        return [c for c in conversations if all(not contaminated(q, bad) for q, a in c)]

    sources = {"dolly": clean(dolly_plain), "no_robots": clean(no_robots), "oasst2": clean(oasst)}
    if with_system:
        sources["dolly with context"] = clean(dolly_context)
    for name, fn in sources.items():
        human += encode(name, [pair_segments(build(), q, a) for q, a in fn], tok, eos, ctx)
    if with_system and not chat_only:
        human += encode(
            "gsm8k",
            [pair_segments(build(MATH_SYSTEM_PROMPT), q, a) for q, a in clean(gsm8k_pairs())],
            tok, eos, ctx,
        )
    for name, conversations in (("no_robots multi-turn", no_robots_multi), ("oasst2 multi-turn", oasst_multi)):
        if not with_system:
            continue
        human += encode(
            name,
            [conversation_segments(build(), [tuple(t) for t in c]) for c in clean_conversations(conversations)],
            tok, eos, ctx, clean=False,
        )
    for name, fn in {"alpaca": lambda: clean(alpaca_pairs())}.items():
        ai += encode(name, [pair_segments(build(), q, a) for q, a in fn()], tok, eos, ctx)
    cap = int(len(human) * AI_MAX_SHARE / (1 - AI_MAX_SHARE))
    random.Random(0).shuffle(ai)
    ai = ai[:cap]
    print(f"AI-written examples used: {len(ai)} ({len(ai) / (len(human) + len(ai)):.0%} of the mix)")
    return human + ai


def format_examples(tok, eos, ctx, with_system, size, decontam=True):
    build = prompt_builder(with_system, 8)
    chat = load_examples(tok, eos, ctx, with_system, chat_only=True, decontam=decontam)
    extra = encode(
        "identity and arithmetic",
        [pair_segments(build(), q, a) for q, a in identity_pairs(size, repeat=IDENTITY_REPEAT) + arithmetic_pairs()],
        tok, eos, ctx, clean=False,
    )
    if with_system:
        extra += encode(
            "variations",
            [pair_segments(system, q, a) for system, q, a in variation_pairs(VARIATION_REPEAT)],
            tok, eos, ctx, clean=False,
        )
        extra += encode(
            "dates",
            [pair_segments(system, q, a) for system, q, a in date_pairs(DATE_COUNT)],
            tok, eos, ctx, clean=False,
        )
    return chat, extra


def supervised_tokens(items):
    return sum(sum(mask) for _, mask in items)


def cap_share(base, extras, share, seed=0):
    budget = int(supervised_tokens(base) * share / (1 - share))
    extras = list(extras)
    random.Random(seed).shuffle(extras)
    kept, used = [], 0
    for ex in extras:
        n = sum(ex[1])
        if used + n > budget:
            continue
        kept.append(ex)
        used += n
    return kept


def batch_tensors(items, eos, dev):
    xs, ys = [], []
    for ids, mask in items:
        labels = [i if m else -100 for i, m in zip(ids, mask)]
        xs.append(ids[:-1])
        ys.append(labels[1:])
    n = max(len(x) for x in xs)
    x = torch.tensor([x + [eos] * (n - len(x)) for x in xs])
    y = torch.tensor([y + [-100] * (n - len(y)) for y in ys])
    return x.to(dev), y.to(dev)


def sft(epochs=2, base=None, resume=False, device="auto", bs=None, accum=None,
        lr_max=None, lr_min=None, warmup=None, decontam=True):
    dev = resolve_device(device)
    base = resolve_checkpoint(base)
    print(f"base: {base.name} | device: {dev}")
    model, cfg = load_model(base, dev)
    tok = load_tokenizer(base)
    eos = tok.token_to_id("<eos>")
    with_system = cfg.lineage == "digit32k"

    n_params = sum(p.numel() for p in model.parameters())

    chat, extra = format_examples(tok, eos, cfg.ctx, with_system, size_millions(n_params), decontam=decontam)
    random.Random(0).shuffle(chat)
    val, train = chat[:VAL_SIZE], chat[VAL_SIZE:]
    train = train + extra

    maths = []
    if with_system:
        build = prompt_builder(with_system, 8)
        maths += encode(
            "math",
            [pair_segments(build(), q, a) for q, a in math_data.synthetic_pairs(MATH_COUNT)],
            tok, eos, cfg.ctx, clean=False,
        )
        maths += encode(
            "stories",
            [pair_segments(build(), q, a) for q, a in math_data.story_pairs(STORY_COUNT)],
            tok, eos, cfg.ctx, clean=False,
        )
        maths += encode(
            "skills",
            [pair_segments(build(), q, a) for q, a in math_data.skill_pairs(SKILL_COUNT)],
            tok, eos, cfg.ctx, clean=False,
        )
        kept = cap_share(train, maths, MATH_MAX_SHARE)
        print(f"math: {len(kept)}/{len(maths)} examples kept (cap {MATH_MAX_SHARE:.0%} of supervised tokens)")
        train = train + kept
    print(f"{len(train)} train examples, {len(val)} validation examples")

    def make_batch(items):
        return batch_tensors(items, eos, dev)

    dbs, dacc = (8, 2) if dev == "cuda" else (2, 8)
    bs = dbs if bs is None else bs
    accum = dacc if accum is None else accum
    per_epoch = len(train) // (bs * accum)
    total = per_epoch * epochs
    if lr_max is None:
        lr_max = 2e-4
    if lr_min is None:
        lr_min = 2e-5
    if warmup is None:
        warmup = 50
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = (s - warmup) / max(1, total - warmup)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for k in range(0, len(val), bs):
            x, y = make_batch(val[k : k + bs])
            with amp:
                losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    tag = f"{size_millions(n_params)}M"
    out_path = base.parent / f"{model_stem(tag)}.pt"
    state_path = base.parent / f"{model_stem(tag)}.state.pt"

    best = float("inf")
    step, start_epoch = 0, 0
    if resume:
        if not state_path.exists():
            sys.exit(f"Nothing to resume: {state_path.name} not found")
        state = read_checkpoint(state_path, dev)
        if state["total_steps"] != total:
            sys.exit(f"The saved run targets {state['total_steps']} steps, not {total}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, start_epoch, best = state["step"], state["epoch"], state["best_val"]
        print(f"resuming after epoch {start_epoch} (step {step})")
    elif out_path.exists():
        print(f"note: {out_path.name} will be overwritten")

    model.train()
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        random.shuffle(train)
        for i in range(per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step_loss = 0.0
            for j in range(accum):
                k = (i * accum + j) * bs
                x, y = make_batch(train[k : k + bs])
                with amp:
                    loss = model(x, y)[1] / accum
                loss.backward()
                step_loss += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(step_loss) or not math.isfinite(gnorm.item()):
                print(f"!! non-finite value at step {step}. Stopping.")
                return
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print(f"epoch {ep + 1}/{epochs} | step {step:4d}/{total} | loss {step_loss:.3f} | {time.time() - t0:.0f}s")
            step += 1
        val_loss = evaluate()
        print(f">> epoch {ep + 1} done | val loss {val_loss:.3f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(
                out_path, model, cfg,
                kind="assistant", step=step, epoch=ep + 1, total_steps=total, val_loss=val_loss,
            )
            print(f"   best so far, saved -> {out_path.name}")
        if ep + 1 < epochs:
            save_checkpoint(
                state_path, model, cfg,
                kind="assistant", step=step, epoch=ep + 1, total_steps=total,
                val_loss=val_loss, best_val=best, optimizer=opt.state_dict(),
            )
    state_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", type=int, nargs="?", default=2)
    ap.add_argument("base", nargs="?")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--lr-max", type=float, default=None, help="peak learning rate of the SFT cosine")
    ap.add_argument("--lr-min", type=float, default=None, help="final learning rate of the SFT cosine")
    ap.add_argument("--warmup", type=int, default=None, help="warmup steps before the cosine decay")
    ap.add_argument("--no-decontam", dest="decontam", action="store_false",
                    help="skip the 13-gram decontamination of the SFT mix (not allowed for reported runs)")
    args = ap.parse_args()
    sft(args.epochs, args.base, args.resume, args.device, args.bs, args.accum,
        args.lr_max, args.lr_min, args.warmup, args.decontam)


if __name__ == "__main__":
    main()
