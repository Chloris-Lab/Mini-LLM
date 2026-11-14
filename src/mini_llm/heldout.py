import argparse
import re

import torch

from mini_llm import math_data
from mini_llm.chat import default_system
from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import assistant_pattern, base_pattern, resolve_device
from mini_llm.prompt import format_prompt
from mini_llm.system_prompts import UNCERTAINTY_PAIRS, VARIATIONS, date_pairs

MAX_NEW = 32
FORMAT_NEW = 96
NUMBER = re.compile(r"-?\d+")
OP_TOKEN = re.compile(r" ([+\-x/]) ")
LABELS = {"+": "add", "-": "sub", "x": "mul", "/": "div"}
BULLET = re.compile(r"^[-*\u2022]\s+")
NUMBERED = re.compile(r"^\d+[.)]\s+")
SENTENCE_END = re.compile(r"[.!?]")
UNCERTAIN = re.compile(
    r"don't know|do not know|not sure|no way of knowing|not confident"
    r"|can't know|cannot know|no reliable|don't have|no idea",
    re.IGNORECASE,
)


def generate(model, tok, eos, dev, prompt, ctx, n=MAX_NEW):
    ids = tok.encode(prompt).ids[-ctx:]
    x = torch.tensor([ids], device=dev)
    n = max(1, min(n, ctx - len(ids)))
    out = model.generate(x, n=n, temp=1.0, top_k=1, eos=eos)
    return tok.decode(out[0, x.size(1) :].tolist()).strip()


def final_number(text):
    found = NUMBER.findall(text)
    return int(found[-1]) if found else None


def label(answer):
    m = OP_TOKEN.search(answer)
    return LABELS[m[1]] if m else "other"


def shape(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if BULLET.match(line)]
    numbered = [line for line in lines if NUMBERED.match(line)]
    if bullets:
        return (f"bullets x{len(bullets)}", len(bullets))
    if numbered:
        return (f"numbered x{len(numbered)}", len(numbered))
    return (f"prose, {len(SENTENCE_END.findall(text))} sentence(s)", len(SENTENCE_END.findall(text)))


def shape_ok(expected, out):
    kind, n = expected
    got, got_n = shape(out)
    if kind.startswith("prose"):
        return got.startswith("prose") and got_n <= max(1, n)
    return got == kind


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


def story_report(model, tok, eos, dev, cfg, system, count, seed):
    pairs = math_data.story_pairs(count, seed=seed)
    hits, misses = 0, []
    for question, answer in pairs:
        expected = final_number(answer)
        out = generate(model, tok, eos, dev, format_prompt(question, system), cfg.ctx, FORMAT_NEW)
        got = final_number(out)
        hits += int(got == expected)
        if got != expected and len(misses) < 5:
            misses.append((question, expected, got, out))
    print(f"stories: {count} held-out word problems, seed {seed}")
    print(f"  exact {hits}/{count} {hits / count:6.1%}")
    for question, expected, got, out in misses:
        print(f"  miss: {question} | expected {expected} | got {got} | {out!r}")
    return hits / count


def format_report(model, tok, eos, dev, cfg, system, repeat=1):
    style = VARIATIONS * repeat
    misses = []
    for sys_prompt, question, reference in style:
        out = generate(model, tok, eos, dev, format_prompt(question, sys_prompt), cfg.ctx, FORMAT_NEW)
        if not shape_ok(shape(reference), out):
            misses.append((sys_prompt, question, shape(reference), out))
    print(f"format: {len(style)} style prompts, seed-free greedy")
    print(f"  style {len(style) - len(misses)}/{len(style)} {(len(style) - len(misses)) / len(style):6.1%}")
    for sys_prompt, question, expected, out in misses[:6]:
        print(f"  miss {expected[0]}: {sys_prompt!r} | {question!r} -> {out[:80]!r}")
    uncertainty = UNCERTAINTY_PAIRS * repeat
    unc_misses = []
    for sys_prompt, question, _ in uncertainty:
        out = generate(model, tok, eos, dev, format_prompt(question, sys_prompt), cfg.ctx, FORMAT_NEW)
        if not UNCERTAIN.search(out):
            unc_misses.append((question, out))
    hit = len(uncertainty) - len(unc_misses)
    print(f"  uncertainty {hit}/{len(uncertainty)} {hit / len(uncertainty):6.1%}")
    for question, out in unc_misses[:6]:
        print(f"  miss uncertainty: {question!r} -> {out[:80]!r}")
    return (len(style) - len(misses) + hit) / (len(style) + len(uncertainty))


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
    ap = argparse.ArgumentParser(description="Held-out generation checks on a local checkpoint")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--math", type=int, default=200, help="number of held-out arithmetic problems")
    ap.add_argument("--stories", type=int, default=50, help="number of held-out word problems")
    ap.add_argument("--dates", type=int, default=50, help="number of dated system prompts to answer")
    ap.add_argument("--format", type=int, default=1, help="repetitions of the format bench (0 disables)")
    ap.add_argument("--base", action="store_true", help="bench a base or mid-trained checkpoint instead of an assistant")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    pattern = base_pattern() if args.base else assistant_pattern()
    path = resolve_checkpoint(args.checkpoint, pattern=pattern)
    model, cfg = load_model(path, dev)
    model.eval()
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    system = default_system(getattr(cfg, "lineage", ""))
    print(f"[{path.name}] device {dev}, ctx {cfg.ctx}")
    if args.math:
        math_report(model, tok, eos, dev, cfg, system, args.math, args.seed)
    if args.stories:
        story_report(model, tok, eos, dev, cfg, system, args.stories, args.seed + 2)
    if args.dates:
        date_report(model, tok, eos, dev, cfg, args.dates, args.seed + 1)
    if args.format:
        format_report(model, tok, eos, dev, cfg, system, args.format)


if __name__ == "__main__":
    main()
