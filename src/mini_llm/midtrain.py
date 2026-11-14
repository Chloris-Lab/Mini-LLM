import argparse
import math
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from mini_llm.checkpoints import (
    load_model,
    load_tokenizer,
    read_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from mini_llm.config import DATA_DIR, MODELS_DIR, model_dir, model_stem, resolve_device, size_millions
from mini_llm.pretrain import check_data_matches, default_batch
from mini_llm.sft import batch_tensors, format_examples

REPLAY_SHARE = 0.3
DATA_SEED = 1234
EVAL_WINDOWS = 512


def midtrain(steps=600, init=None, replay=REPLAY_SHARE, lr_max=1e-4, lr_min=1e-5, warmup=50,
             batch=None, accum=None, device="auto"):
    if not 0.0 <= replay < 1.0:
        sys.exit("--replay must be in [0, 1)")
    dev = resolve_device(device)
    base = resolve_checkpoint(init, root=MODELS_DIR / "digit32k")
    ck = read_checkpoint(base, "cpu")
    if ck.get("kind") != "base":
        sys.exit(f"{base.name} is a {ck.get('kind')} checkpoint: the format stage starts from a base model")
    start = int(ck.get("step") or 0)
    del ck
    model, cfg = load_model(base, dev)
    if getattr(cfg, "lineage", "") != "digit32k":
        sys.exit("The format stage needs the 1.1 (digit32k) lineage: this base never saw a system turn")
    tok = load_tokenizer(base)
    eos = tok.token_to_id("<eos>")
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{size_millions(n_params)}M"
    out_dir = model_dir(cfg.vocab, tag, cfg.lineage)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_stem(tag)}-base-{start + steps}steps"
    weights_path = out_dir / f"{stem}.pt"
    if weights_path.exists():
        sys.exit(f"{stem} already exists. Delete it or ask for a different number of steps.")

    chat, extra = format_examples(tok, eos, cfg.ctx, True, size_millions(n_params))
    examples = chat + extra
    random.Random(0).shuffle(examples)
    print(f"format: {len(examples)} examples | {sum(sum(m) for _, m in examples) / 1e6:.2f}M supervised tokens")

    replay_data, n_win = None, 0
    if replay > 0:
        path = DATA_DIR / "train.bin"
        if not path.exists():
            sys.exit(f"--replay {replay:g} needs {path}")
        replay_data = np.memmap(str(path), dtype=np.uint16, mode="r")
        n_win = (len(replay_data) - 1) // cfg.ctx
        if n_win <= 0:
            sys.exit(f"{path} too small ({len(replay_data)} tokens) for ctx {cfg.ctx}")

    val_data = None
    val_path = DATA_DIR / "val.bin"
    if val_path.exists():
        val_data = np.memmap(str(val_path), dtype=np.uint16, mode="r")
    meta = check_data_matches(
        DATA_DIR / "meta.json", cfg,
        train_len=len(replay_data) if replay_data is not None else None,
        val_len=len(val_data) if val_data is not None else None,
    )
    bytes_per_token = None
    if meta.get("train_bytes") and meta.get("train_tokens"):
        bytes_per_token = meta["train_bytes"] / meta["train_tokens"]

    def bits_per_byte(loss):
        if not bytes_per_token or loss is None:
            return None
        return loss / (math.log(2) * bytes_per_token)

    db, da = default_batch(dev, tag, cfg.ctx)
    batch = db if batch is None else batch
    accum = da if accum is None else accum
    warmup = min(warmup, max(1, steps // 20))
    print(f"device: {dev} | {n_params / 1e6:.1f}M params")
    print(f"batch {batch}x{accum} | {batch * accum * cfg.ctx} tokens/step | {steps} steps of format {1 - replay:.0%}")
    print(f"lr {lr_max:g} -> {lr_min:g} | warmup {warmup} | replay {replay:.0%} of train.bin")

    decay, nodecay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr_max, betas=(0.9, 0.95), fused=(dev == "cuda"),
    )

    rng = random.Random(DATA_SEED)
    epoch, pos = 0, 0
    perm = None

    def format_batch():
        picks = [examples[rng.randrange(len(examples))] for _ in range(batch)]
        return batch_tensors(picks, eos, dev)

    def replay_batch():
        nonlocal epoch, pos, perm
        starts = []
        for _ in range(batch):
            if perm is None or pos >= n_win:
                perm = np.random.default_rng(DATA_SEED + epoch).permutation(n_win)
                epoch += 1
                pos = 0
            starts.append(int(perm[pos]) * cfg.ctx)
            pos += 1
        x = np.stack([replay_data[s : s + cfg.ctx] for s in starts]).astype(np.int64)
        y = np.stack([replay_data[s + 1 : s + 1 + cfg.ctx] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()
    cmodel = torch.compile(model) if dev == "cuda" else model

    @torch.no_grad()
    def evaluate():
        if val_data is None or (len(val_data) - 1) // cfg.ctx <= 0:
            return None
        model.eval()
        n_vwin = (len(val_data) - 1) // cfg.ctx
        k = min(EVAL_WINDOWS, n_vwin)
        starts = np.sort(np.random.default_rng(0).choice(n_vwin, size=k, replace=False)).astype(np.int64) * cfg.ctx
        ls = []
        for i in range(0, len(starts), batch):
            ss = starts[i : i + batch]
            x = torch.from_numpy(np.stack([val_data[s : s + cfg.ctx] for s in ss]).astype(np.int64)).to(dev)
            y = torch.from_numpy(np.stack([val_data[s + 1 : s + 1 + cfg.ctx] for s in ss]).astype(np.int64)).to(dev)
            with amp:
                ls.append(cmodel(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    t0 = time.time()
    val_loss = None
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        step_loss = torch.zeros((), device=dev)
        for _ in range(accum):
            if replay_data is not None and rng.random() < replay:
                x, y = replay_batch()
            else:
                x, y = format_batch()
            with amp:
                _, loss = cmodel(x, y)
            (loss / accum).backward()
            step_loss += loss.detach() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total = step_loss.item()
        if not math.isfinite(total) or not math.isfinite(gnorm.item()):
            print(f"!! non-finite value at step {step}. Stopping.")
            return
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(
                f"step {step:4d}/{steps} | loss {total:.3f} | gnorm {gnorm.item():.2f} "
                f"| lr {lr_at(step):.2e} | {time.time() - t0:.0f}s",
                flush=True,
            )
    val_loss = evaluate()
    save_checkpoint(
        weights_path, model, cfg,
        kind="base", step=start + steps, total_steps=start + steps, val_loss=val_loss,
        bits_per_byte=bits_per_byte(val_loss),
        phase_start=start, midtrain={"base": base.name, "steps": steps, "replay": replay},
    )
    if val_loss is not None:
        bpb = bits_per_byte(val_loss)
        print(f"  >> val loss {val_loss:.3f}" + (f" | bits/byte {bpb:.3f}" if bpb else ""))
    print(f"done -> {weights_path.name} (format stage after {base.name})")


def main():
    ap = argparse.ArgumentParser(description="Format stage: teach the chat markup before SFT")
    ap.add_argument("steps", type=int, nargs="?", default=600)
    ap.add_argument("--init", default=None, help="base checkpoint to start from")
    ap.add_argument("--replay", type=float, default=REPLAY_SHARE, help="share of raw-text batches")
    ap.add_argument("--lr-max", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    midtrain(
        args.steps, args.init, args.replay, args.lr_max, args.lr_min, args.warmup,
        args.batch, args.accum, args.device,
    )


if __name__ == "__main__":
    main()
