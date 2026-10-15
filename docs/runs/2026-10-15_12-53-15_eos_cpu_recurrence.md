# Run: EOS-aware CPU recurrence comparison

## Purpose and decision

Compare one versus two passes through shared transformer blocks at the same
parameter count, initialization, sampled-token budget, and fixed evaluation windows.
This is a bounded **synthetic workflow experiment**, not a TinyStories quality
benchmark, GPU performance result, or statistically significant architecture study.

The extra pass slightly lowered validation text BPB in this run but almost doubled
training-step time. One pass is suitable for fast CPU workflow smoke checks; this
does not justify changing the project's two-pass default for real-data experiments.

## Exact provenance and reproduction

- Revision: `2eda754147c1e25d77959100ce2ee55f6845916d`; git dirty state **false**
  for both runs. The experiment script and all configuration were committed first.
- Started: 2026-10-15T12:53:15.725300+02:00 (one pass);
  2026-10-15T12:53:17.804351+02:00 (two passes).
- Environment: Linux; Python 3.14.8; PyTorch 2.13.0+cpu; NumPy 2.5.4;
  Datasets 5.1.0; Tokenizers 0.23.3; Hugging Face Hub 2.2.0.
- Device: CPU, one PyTorch thread, default dtype float32, matmul precision `highest`,
  deterministic-algorithm enforcement disabled. CUDA/cuDNN and GPU capacity are
  unavailable. CPU model/processor identity was not collected.
- Full provenance, effective configurations, measurements, routing diagnostics,
  source hashes, and validation curves are preserved in the
  [one-pass transcript](2026-10-15_12-53-15_eos_cpu_r1.md) and
  [two-pass transcript](2026-10-15_12-53-17_eos_cpu_r2.md).

From the repository root, use the pinned CPU setup in the [README](../../README.md)
and run the source-defined experiment:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m experiments.cpu_recurrence --output-directory=artifacts/cpu-recurrence-clean
```

Repeat from a clean checkout of the recorded revision with an unused output
directory. Checkpoints/results remain ignored local artifacts; **no model or dataset
binaries are distributed**. The transcripts' original artifact paths are location
records, not links to downloadable files. No access token or Hub request is needed.

## Configuration

| Setting | Value |
|---|---|
| Architecture | RMSNorm, causal RoPE SDPA, sparse SwiGLU MoE, untied output |
| Dim / heads / head dim | 32 / 4 / 8 |
| Unique blocks / recurrent passes | 2 / one versus two; weights shared across passes |
| MoE | 4 experts per block, top-2, no capacity limit/dropped tokens |
| Training objective | All-target next-token cross-entropy + 0.01 * mean router auxiliary loss |
| Tokenizer | Byte, 257 IDs, EOS=256; one EOS appended per document |
| Context / batch / steps | 32 tokens / 4 / 200; manual step budget |
| Sampled tokens / estimated source bytes | 25,600 / 25,190.4 per run, including repeated samples |
| Optimizer | AdamW, learning rate 0.0003; PyTorch defaults for other settings |
| Seeds | Initialization 42; separate training-window generator 42; evaluation generator 43 |
| Evaluation | Every 50 steps, 5 fixed-window batches; final 20 batches extends the same prefix |
| Best selection | Periodic validation text BPB; final evaluation does not replace best |
| Generation | Not run; this comparison measures training/evaluation/checkpoint behavior |

### Source-defined synthetic data

Dataset identifier: `tinylm-synthetic-robots-v1`, version 1. The documents are
constructed explicitly in [the experiment script](../../experiments/cpu_recurrence.py)
from a robot/shape/shelf sentence template. Training uses 16 color/shape combinations;
validation uses eight combinations with held-out color words, sharing the same
template. Nothing is downloaded.

| Split | Source UTF-8 bytes | Tokens, including EOS | Mean source bytes/token |
|---|---:|---:|---:|
| Training | 984 | 1,000 | 0.984000 |
| Validation | 504 | 512 | 0.984375 |

The synthetic dataset revision is the SHA-256 of canonical JSON for both splits
(not a Hub commit):
`a24345cc2daa4ccc89ccfc5b0bb33768aa96316456d3570fcffaca8e93d53b97`.
Split fingerprints and the hashing procedure are in the committed script/transcripts.

## Results

| Metric | One pass | Two passes |
|---|---:|---:|
| Parameters | 90,593 | 90,593 |
| Training-step time, rounded logged seconds | 1.24 | 2.30 |
| Training invocation wall time, seconds | 2.0580 | 2.5033 |
| Best periodic step | 200 | 200 |
| Best periodic text BPB, 5 batches | 3.040863 | 3.023242 |
| Best periodic sequence BPB, 5 batches | 3.103232 | 3.082575 |
| Best periodic token loss, nats | 2.114026 | 2.099954 |
| Final text BPB, 20 batches | 3.016653 | 2.997376 |
| Final sequence BPB, 20 batches | 3.069878 | 3.048083 |
| Final token loss, nats | 2.096292 | 2.081409 |
| Final EOS targets | 38 | 38 |
| Final EOS mean NLL, nats | 2.448533 | 2.332705 |
| Final EOS prediction accuracy | 0.00% | 26.32% |
| Peak GPU memory / capacity | Not applicable | Not applicable |
| Peak CPU memory | Not measured | Not measured |

Text BPB excludes EOS targets; sequence BPB includes their NLL over the same
source-byte denominator; token loss averages over all targets. These values are
**not** historical pre-EOS BPB reinterpreted under a new definition.

The CPU logger prints `Peak VRAM: 0.0 MiB`; that indicates no CUDA allocation,
not a zero CPU-memory footprint. Training-step time excludes validation, logs, and
checkpoint I/O. Invocation wall time includes preparation, model/optimizer setup,
and evaluation/I/O, but excludes the subsequent checkpoint-reload verification.
The first run also pays process-local lazy initialization overhead; compare step
time rather than the wall-time ratio, and treat both short timings as noisy.

## Verification, trade-off, and limitations

Both best and last checkpoints were reloaded. Re-evaluating each last checkpoint
on the same 20 fixed validation batches reproduced its text and sequence BPB
**exactly in this environment**.

The second pass adds no parameters, but adds block computation/activations.
Final text BPB fell by 0.019277 (~0.64%), while rounded logged training-step time
increased ~85.5%. This demonstrates a measured compute/quality trade-off on this
small fixture, not a general recommendation about recurrence.

Only one seed, one short CPU measurement per configuration, a highly repetitive
template corpus, and no natural-language quality assessment were used. Fixed
validation windows aid debugging but do not establish generalization or significance.
No memory improvement can be claimed without memory measurements.

**Still to run manually:** a current EOS-aware TinyStories experiment on the local
GPU, with an immutable dataset revision, captured environment, fresh quality and
VRAM results, and preferably repeated seeds. Historical GPU results remain separate.
