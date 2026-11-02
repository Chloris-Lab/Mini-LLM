import argparse
import re

import torch

from mini_llm import math_data
from mini_llm.chat import default_system
from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import resolve_device
from mini_llm.prompt import format_prompt
from mini_llm.system_prompts import date_pairs

MAX_NEW = 32
NUMBER = re.compile(r"-?\d+")
OP_TOKEN = re.compile(r" ([+\-x/]) ")
LABELS = {"+": "add", "-": "sub", "x": "mul", "/": "div"}


def generate(model, tok, eos, dev, prompt, ctx):
    ids = tok.encode(prompt).ids[-ctx:]
    x = torch.tensor([ids], device=dev)
    n = max(1, min(MAX_NEW, ctx - len(ids)))
    out = model.generate(x, n=n, temp=1.0, top_k=1, eos=eos)
    return tok.decode(out[0, x.size(1):].tolist()).strip()


def final_number(text):
    found = NUMBER.findall(text)
    return int(found[-1]) if found else None


def label(answer):
    m = OP_TOKEN.search(answer)
    return LABELS[m[1]] if m else "other"


def math_report(model, tok, eos, dev, cfg, system, count, seed):
    pairs = math_data.synthetic_pairs(count, seed=seed)
    hits, totals, misses = {}, {}, []
    for question, answer in pairs:
        expected = final_number(answer)
        out = generate(model, tok, eos, dev, format_prompt(question, system), cfg.ctx)
        got = final_number(out)
        name = label(answer)
        totals[name] = totals.get(name, 0) + 1
        hits[name] = hits.get(name, 0) + int(got == expected)
        if got != expected and len(misses) < 5:
            misses.append((question, expected, got, out))
    print(f"math: {len(pairs)} held-out problems, seed {seed}")
    for name in sorted(totals):
        print(f"  {name:5s} {hits[name]:4d}/{totals[name]:<4d} {hits[name] / totals[name]:6.1%}")
    total = sum(totals.values())
    print(f"  total {sum(hits.values()):4d}/{total:<4d} {sum(hits.values()) / total:6.1%}")
    for question, expected, got, out in misses:
        print(f"  miss: {question} | expected {expected} | got {got} | {out!r}")
    return sum(hits.values()) / total


def date_report(model, tok, eos, dev, cfg, count, seed):
    rows = date_pairs(count, seed=seed)
    exact, misses = 0, []
    for system, question, answer in rows:
        out = generate(model, tok, eos, dev, format_prompt(question, system), cfg.ctx)
        if answer in out:
            exact += 1
        elif len(misses) < 5:
            misses.append((question, answer, out))
    print(f"dates: {count} prompts, seed {seed}")
    print(f"  exact {exact}/{count} {exact / count:6.1%}")
    for question, answer, out in misses:
        print(f"  miss: {question} | expected {answer!r} | got {out!r}")
    return exact / count


def main():
    ap = argparse.ArgumentParser(description="Held-out generation checks on a local assistant checkpoint")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--math", type=int, default=200, help="number of held-out arithmetic problems")
    ap.add_argument("--dates", type=int, default=50, help="number of dated system prompts to answer")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    path = resolve_checkpoint(args.checkpoint, pattern="mini-llm-assistant-*.pt")
    model, cfg = load_model(path, dev)
    model.eval()
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    system = default_system(getattr(cfg, "lineage", ""))
    print(f"[{path.name}] device {dev}, ctx {cfg.ctx}")
    if args.math:
        math_report(model, tok, eos, dev, cfg, system, args.math, args.seed)
    if args.dates:
        date_report(model, tok, eos, dev, cfg, args.dates, args.seed + 1)


if __name__ == "__main__":
    main()
