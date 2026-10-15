# TinyLM

**A small language-model experimentation platform focused on correctness testing, measurement, and explicit quality/runtime/memory trade-offs on consumer hardware.**

TinyLM implements a PyTorch transformer with recurrent/shared blocks, sparse Mixture-of-Experts (MoE) layers, and byte or locally trained byte-level BPE tokenization. My contribution is the model implementation and its engineering workflow: sparse dispatch, numerical correctness tests, tokenizer/evaluation debugging, reliable checkpoints, and measured experiments. The underlying ML techniques are established, not inventions of this project.

**Status:** educational and experimental, not production LLM infrastructure, a serving system, a distributed trainer, or a state-of-the-art model. The code favors inspectability over fused kernels. Historical consumer-GPU experiments are single runs, not statistically significant comparisons.

## Engineering highlights

- **Actual sparse execution:** experts receive only their assigned token rows; unselected experts do not run. Tests compare outputs **and gradients** with a dense reference and inspect expert inputs.
- **Checkpoint reliability:** self-contained architecture/tokenizer state, overwrite protection, atomic file replacement, and a regression test that simulates a failed save.
- **Byte-accurate, EOS-aware evaluation:** exact target-byte accounting, separate text/sequence BPB, and EOS prediction diagnostics, including split-Unicode cases.
- **Controlled sampling and provenance:** independent training/evaluation RNG streams, fixed evaluation windows, effective configuration, source hashes, versions, and dataset metadata in new runs/checkpoints.
- **Measured trade-offs:** reports retain quality, runtime, allocated VRAM, routing diagnostics, and imperfect generated samples rather than presenting loss alone as success.

## Selected historical results

These RTX 3060 results **predate EOS-aware document encoding and the current fixed evaluation sampling**. They are preserved as measured, not reinterpreted using current metric semantics.

| Experiment | Historical validation metric | Recorded training time | Peak allocated VRAM |
|---|---:|---:|---:|
| [Dense, width 384](docs/runs/2026-10-10_11-07_31ca79e.md) | 0.6338 token loss | 1,270.11 s, including periodic evaluation/I/O | 1,655.7 MiB |
| [8 experts, top-2](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md) | 0.5721 token loss | 2,312.09 s, including periodic evaluation/I/O | 3,296.9 MiB |
| [BPE 8,192, two passes](docs/runs/2026-10-11_10-11-29_bpe_8192.md) | 0.6519 BPB | 705.49 s, training steps only | 4,562.4 MiB |

The sparse run improved recorded loss but used more time, memory, and total parameters. BPE changes context coverage and evaluation windows; its BPB is **not** comparable to the byte rows' token losses. Generation still shows repetition. The [full historical overview](docs/runs/HISTORY.md#run-overview) retains every measurement, architecture comparison, and research note; the [experiment index](docs/runs/README.md) provides entry points and provenance limitations.

## Quick start and supported environment

Run from the repository root. The tested CPU environment is **Linux, Python 3.14, PyTorch 2.13.0**, with direct dependencies pinned in [requirements.txt](requirements.txt). Tests use standard-library `unittest`; pytest is not required. Transitive packages are resolved by pip, not fully locked.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

[CPU CI](.github/workflows/tests.yml) runs the complete suite with Hugging Face offline and no credentials. Tests use synthetic data and tiny optimizer/checkpoint workflows; they do not download TinyStories or launch full training.

For **local GPU training**, install a compatible PyTorch/CUDA build *before* installing `requirements.txt`, instead of the CPU wheel above. The historical environment used `torch==2.13.0` from `https://download.pytorch.org/whl/cu132`. Select a build compatible with your driver using the [PyTorch installation guide](https://pytorch.org/get-started/locally/); current GPU behavior is not validated by CPU CI.

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. Windows is not part of the tested CI environment.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python train.py --model-prefix=experiment-
python run.py --model-prefix=experiment-
```

Training downloads TinyStories on first use and selects CUDA when available, otherwise CPU. **The default configuration is a full experiment, not a quick CPU demo.** For a bounded, offline current-code comparison:

```bash
CUDA_VISIBLE_DEVICES="" python -m experiments.cpu_recurrence
```

That script compares one/two recurrent passes on repository-authored synthetic documents, stores outputs under ignored `artifacts/`, and refuses to overwrite checkpoints. Use `--output-directory=artifacts/another-run` to repeat it. It is workflow/trade-off evidence, not a TinyStories quality benchmark.

## Current architecture

The default model embeds **257 tokens: 256 UTF-8 byte values plus EOS**. BPE instead uses its actual trained vocabulary, also including EOS. Encoding appends one EOS ID after each document without inserting newline separators. Three unique blocks are reused twice, with width 384, 6 attention heads, context 256 tokens, batch size 32, sixteen independent SwiGLU experts per block, and top-2 routing. The current byte model has **58,611,329 parameters**; the historical 256-token model had 58,610,560.

```text
documents -> byte/BPE IDs + EOS per document -> token embedding
          -> [block A -> block B -> block C] repeated twice
          -> RMSNorm -> next-token logits
block: pre-norm causal RoPE attention + residual
       -> pre-norm sparse SwiGLU MoE + residual
```

Each block owns a learned bias-free router and sixteen independent SwiGLU experts with hidden width 1,024. Router probabilities and balancing statistics use float32. Each token selects two experts, renormalizes their probabilities, runs only gathered assigned rows, and scatters weighted outputs back. Empty experts are skipped; there is no capacity limit or token dropping. Recurrent passes reuse weights but route again.

Training remains next-token cross-entropy over **all targets, including EOS**, plus `0.01 * mean router auxiliary loss`. The Switch-style auxiliary loss is `E * sum(f * p)`: `f` is each expert's fraction of top-k assignments and `p` its mean full-router probability. Uniform routing gives 1. Averaging over all block calls keeps its coefficient's scale independent of recurrence/layer count. Logged assignment percentages describe the latest training batch summed over passes, not run-wide expert utilization.

The full-size top-2 experts use approximately twice a single dense SwiGLU's feed-forward arithmetic. Sparse dispatch saves work relative to evaluating all experts, not necessarily relative to a dense model. All expert weights and optimizer states remain resident. The Python gather/scatter loop, variable-sized batches, and recomputed attention during generation are intentional simplicity/performance trade-offs; no fused MoE kernel or KV cache is implemented.

## Evaluation and training budget

Evaluation excludes the router auxiliary term:

| Metric | Numerator | Denominator |
|---|---|---|
| Token loss | NLL of all targets, including EOS, in nats | All target tokens |
| Text BPB | NLL of non-EOS targets | `ln(2) * source bytes represented by those targets` |
| Sequence BPB | NLL of all targets, including EOS | The same source-byte denominator |

EOS has zero source-byte payload. Consequently, **byte-mode text BPB is not generally token loss / ln(2)**. Improving EOS predictions can change token/sequence loss without changing text BPB. An all-EOS evaluation has no byte-normalized metric and reports infinite BPB. EOS count, mean NLL, and prediction accuracy are reported separately.

Byte lengths come from the reversible vocabulary and are indexed by **shifted targets**, counting partial UTF-8 characters and whitespace merges exactly. The denominator uses each evaluated window, not a corpus-wide compression average. Evaluation samples overlapping token windows; it is not a full-corpus compression benchmark or a fixed-source-byte-context comparison.

The best checkpoint is selected by **periodic validation text BPB**. The final, larger evaluation measures the last model without replacing the best checkpoint. Evaluation restarts a CPU generator at `seed + 1` on every call: periodic windows are fixed, and final evaluation extends the same prefix. Training samples use a separate CPU generator at `seed`; initialization and evaluation no longer advance that stream. Tokenization still changes sampled source windows. Fixed windows improve repeatability but do not replace independent-seed experiments or a separate test set.

The default source-byte budget is 81,920,000, **not an unconditional 10,000 steps**:

```text
steps = round(target_training_bytes /
              (batch_size * context * measured_training_source_bytes_per_token))
validate_every = max(1, steps // 10)
```

EOS lowers source bytes/token in byte mode; BPE generally increases it. The budget counts repeated random samples, not unique corpus coverage. This is a nearest-whole-step **estimate**, not a measured count of every sampled byte. Budgets rounding to zero steps are rejected explicitly. Set `target_training_bytes=None` to use manual positive `steps` and `validate_every`.

## Tokenization, checkpoints, and generation

Edit `Config` in [core.py](core.py) for architecture, budget, validation, splits, and prompts. Architecture lives in [tinylm.py](tinylm.py); [train.py](train.py) and [run.py](run.py) are the entry points. Default splits are `train[:5%]` / `validation[:1%]`.

BPE uses Hugging Face Tokenizers' `models.BPE`, `ByteLevel(add_prefix_space=False, use_regex=True)`, and the ByteLevel decoder. It trains on **separate training documents only**, with all 256 byte symbols and EOS reserved, no normalization or dropout, and minimum frequency 2. Requested vocabulary includes EOS; actual vocabulary can be smaller. Validation uses the saved training tokenizer. Literal `<|endoftext|>` source text remains ordinary bytes; only appended EOS IDs mark boundaries. Context stays at 256 **tokens**, so BPE sees more source text and expands embedding/output/softmax work.

Training writes `model/experiment-best_model.pt` and `model/experiment-last_model.pt`; an empty prefix uses `best_model.pt` / `last_model.pt`. Both destinations are checked before data loading. Keep experiments separate with unique prefixes, or deliberately replace them:

```bash
python train.py --model-prefix=experiment- --force-overwrite
python train.py --model-prefix=bpe-4096- --tokenizer-type=bpe --bpe-vocab-size=4096
python run.py --model-prefix=experiment- --checkpoint=last
python run.py --model-prefix=experiment- --temperature=0.7 --top-p=0.9 --seed=42
python run.py --model-prefix=experiment- --greedy
```

Generation loads the best checkpoint by default and restores its architecture/tokenizer, independent of later architecture edits in `Config`. Prompts and decoding settings come from the generation invocation. Sampling defaults to temperature 0.7/top-p 0.9; `--greedy` overrides sampling. Temperature must be finite and positive; top-p must be in `(0, 1]`.

Generation stops on **EOS or approximately 1,000 newly decoded UTF-8 bytes per prompt**, excluding the prompt and EOS. The last token or decoder flush can exceed the budget. Invalid UTF-8 is replaced and replacement characters count toward output bytes. Stop reasons, seed, settings, and throughput are logged. Repeatability assumes the same checkpoint, prompts, settings, software, and device; cross-device/version bitwise reproducibility is not promised.

New checkpoints contain model/optimizer state, effective run configuration, tokenizer JSON/metadata, dataset statistics, provenance, and validation diagnostics. `validation_bpb` stores **text BPB**, `selection_metric` is `text_bpb`, and `validation_metrics` preserves token loss, text/sequence BPB, and EOS diagnostics. Older `bpb` labels retain their saved historical semantics; legacy metadata-free MoE byte checkpoints restore 256-token mode without EOS. Dense historical checkpoints have an incompatible parameter layout. There is no training-resume CLI.

## Run reporting and reproducibility

Both entry points mirror stdout/stderr into timestamped Markdown transcripts in `docs/runs/`. Source provenance is captured **before** a new transcript makes the worktree dirty. New training logs and checkpoints include:

- Git commit and dirty state, plus SHA-256 hashes of the four implementation files.
- Effective configuration after tokenizer/budget resolution, seed, and sampling policy.
- Python/PyTorch/NumPy/Datasets/Tokenizers/Hub versions; device type, thread count, GPU name/capacity and CUDA/cuDNN when applicable; precision/determinism settings.
- Dataset identifier, requested revision, splits, available dataset version and encoded-split fingerprints; tokenizer settings, vocabulary, and EOS metadata.

`Config.dataset_revision` or `train.py --dataset-revision=<immutable-commit>` pins an upstream dataset revision. With no revision requested, upstream remains unpinned; a dataset version/fingerprint is **not** a resolved Hub commit. Missing metadata is explicit rather than inferred. A dirty flag/source hash identifies a changed run, but does not preserve its source diff: commit source changes before a publishable experiment.

Generation records its effective prompts/decoding settings, restored model configuration, and available training provenance. Entry points are recorded without arbitrary command arguments; effective settings are captured structurally instead. No environment-variable dump, access token, hostname, or unrelated machine inventory is collected. Repository/home/temp path prefixes are normalized in public transcripts, while console output remains unchanged. Review logs before publication: custom prompts and third-party messages can themselves contain sensitive content.

Training-step time excludes evaluation, logging, and checkpoint I/O; total runtime includes preparation and that overhead. CUDA timings synchronize appropriately. Peak VRAM is PyTorch's maximum **allocated** memory, not reserved memory or total device usage. CPU VRAM readings do not measure CPU peak memory.

Use the [report template](docs/runs/TEMPLATE.md) for curated results. Preserve full output, exact committed configuration, dataset revision, metric semantics, timing boundaries, and limitations. Never infer missing historical provenance or compare old/new metric definitions as if unchanged.

## Dataset and license

**TinyStories**, by **Ronen Eldan and Yuanzhi Li**, is described in [*TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*](https://arxiv.org/abs/2305.07759). The [upstream dataset](https://huggingface.co/datasets/roneneldan/TinyStories) contains stories synthetically generated using GPT-3.5/GPT-4 and identifies its license as **[CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)**. It is downloaded at runtime and **not redistributed** here. Consult upstream provenance and terms before redistributing data or trained artifacts.

Generated examples are experimental outputs, not guarantees of originality: a model can reproduce training passages. They are not validated stories or production-ready content.

This repository's **source code** is licensed under [MIT](LICENSE). Third-party datasets and dependencies retain their own terms.

## Implementation and acknowledgments

The transformer blocks, sparse expert dispatch, workflow utilities, evaluation accounting, and tests are implemented in this repository. PyTorch supplies tensor/autograd/attention primitives; Hugging Face Datasets supplies dataset access, and Tokenizers supplies BPE training/serialization.

[Transformers](https://arxiv.org/abs/1706.03762), [RoPE](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [SwiGLU](https://arxiv.org/abs/2002.05202), BPE, and [Switch-style balancing](https://arxiv.org/abs/2101.03961) are established techniques. The [research reading list](docs/research/maybe-interesting.md) records related external projects; it is not a claim that their code was incorporated or that TinyLM contributed upstream.

This project demonstrates hands-on PyTorch implementation, sparse-routing correctness, numerical testing, tokenizer/evaluation debugging, checkpoint reliability, and disciplined performance/memory experiments. It makes no claim of novel ML research, production-scale deployment, model-serving expertise, or organizational impact.
