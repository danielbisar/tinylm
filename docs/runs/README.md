# Experiment index

## Current EOS-aware implementation

The bounded [CPU recurrence experiment](../../experiments/cpu_recurrence.py) uses
repository-authored synthetic documents, not downloaded TinyStories. It compares
shared-block passes at fixed parameter count and token budget, with current text/
sequence BPB, EOS diagnostics, isolated RNG streams, and fixed evaluation windows.
Run it from the root with `CUDA_VISIBLE_DEVICES="" python -m experiments.cpu_recurrence`.
The [measured EOS-aware report](2026-10-15_12-53-15_eos_cpu_recurrence.md) records
the exact clean revision, full configurations/environment, results, timing limits,
and [one-pass](2026-10-15_12-53-15_eos_cpu_r1.md) /
[two-pass](2026-10-15_12-53-17_eos_cpu_r2.md) transcripts. GPU memory is unavailable;
CPU peak memory was not measured. Do not compare these fixture scores to TinyStories.

## Historical/pre-EOS experiments

The [complete historical overview](HISTORY.md#run-overview) preserves every
comparison row, configuration, research note, and linked report from the earlier
README. These measurements precede EOS-aware documents and fixed evaluation
sampling. They are not current-code benchmarks.

| Topic | Entry points |
|---|---|
| Dense architecture progression | [First model](2026-10-09_22-36_43c5c8a.md), [dense width-384 baseline](2026-10-10_11-07_31ca79e.md), [width 512](2026-10-10_11-51_f633f44.md) |
| Sparse routing | [8 experts/top-2](2026-10-10_13-13-05_sparse-moe_8e_top2.md), [top-1](2026-10-10_sparse-moe_8e_top1.md), [16 experts/10k steps](2026-10-10_sparse-moe_16e_top2_10000steps.md) |
| Matched approximate byte-budget BPE | [2,048](2026-10-11_09-53-47_bpe_2048.md), [4,096](2026-10-11_10-33-38_bpe_4096.md), [8,192](2026-10-11_10-11-29_bpe_8192.md), [16,384](2026-10-11_10-47-52_bpe_16384.md) |
| Extra shared-block computation | [BPE 8,192 / three passes](2026-10-11_16-43-28_bpe_8192_r3.md) |
| Tokenizer workflow smoke | [CPU BPE smoke](2026-10-10_bpe_smoke.md) |

### Sampling transcript comparison

These four preserved invocations load the same historical BPE 8,192 best checkpoint
(step 2,440), use seed 42, and generate across the same five prompts. Timing and
generated lengths differ; these are illustrative single invocations.

| Decoding | Full transcript |
|---|---|
| Greedy | [16:32:44](2026-10-11_16-32-44-112871-bpe-8192-.md) |
| Temperature 0.8, top-p 0.9 | [16:33:54](2026-10-11_16-33-54-780639-bpe-8192-.md) |
| Temperature 0.7, top-p 0.9 | [16:34:50](2026-10-11_16-34-50-891440-bpe-8192-.md) |
| Temperature 0.9, top-p 0.9 | [16:35:48](2026-10-11_16-35-48-747672-bpe-8192-.md) |

## Provenance limits

Historical reports distinguish supplied measurements, reconstructed configuration,
inferred revisions, and missing environment metadata. Referenced local checkpoints
and private artifacts are not distributed. The eight-expert report's original
source patch is unavailable; source hashes do not reconstruct it.

Conversational recommendations are historical advice, not independently verified
conclusions. They should not override single-run/statistical caveats. Transcripts
retain measurements and generated text; personal path prefixes have been normalized
where present. Use the [current README](../../README.md) for current behavior and
the [template](TEMPLATE.md) when recording new experiments.
