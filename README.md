# Mini-LLM

Small assistants (~40M and ~150M parameters on the 32k tokenizer lineage) trained
**end-to-end** with PyTorch — custom BPE tokenizer, transformer decoder architecture,
pretraining, a format stage and SFT, no pretrained backbone. The previous 1.0
generation (34M/70M,
16k tokenizer) is documented in the
[archive section](https://github.com/Mini-llm-lab/Documentation/blob/main/Mini-LLM/architecture.md#generation-10)
of the architecture page.

Training runs on NVIDIA GPUs (CUDA).

The 150M **1.1** (`models/digit32k/`) is the current cloud recipe: digit-split
tokenizer, 2048-token context, system-prompt format, a format stage before SFT,
arithmetic / date following / markdown / multi-turn SFT, ~7B-token pretraining
budget — see
[training](https://github.com/Mini-llm-lab/Documentation/blob/main/Mini-LLM/training.md) and
[cloud GPU](https://github.com/Mini-llm-lab/Documentation/blob/main/Mini-LLM/cloud-gpu.md).

> Training scale translates directly into output quality: base val loss drops from
> 3.030 (70M, ~1.4B tokens) to 2.941 (150M, ~4.5B tokens) — larger models are markedly
> more fluent and better formatted. Conformance and factual accuracy are tracked
> continuously with [Mini-Eval](https://github.com/Mini-llm-lab/Mini-Eval).

**Full documentation: [Mini-LLM documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-LLM)**

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```
git clone https://github.com/Mini-llm-lab/Mini-LLM
cd Mini-LLM
uv sync

uv run prep          # tokenizer + data
uv run pretrain 300  # short smoke run
uv run sft 1
uv run chat
```

To speed up Hugging Face downloads, create a free token and export it before running
`prep` or `sft`: `export HF_TOKEN=...`.

Full training runs go through the cloud GPU workflow — see
[Cloud GPU](https://github.com/Mini-llm-lab/Documentation/blob/main/Mini-LLM/cloud-gpu.md).
Everything else — architecture, checkpoints, data sources, Hugging Face export,
design notes — lives in the
[documentation](https://github.com/Mini-llm-lab/Documentation/tree/main/Mini-LLM).
