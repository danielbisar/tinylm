# TinyLM

A small byte-level language model built with PyTorch, exploring how model width, context length, attention heads, and shared transformer blocks affect quality, training time, and GPU memory use. Early experiments use a few repeated sentences; later runs train on TinyStories. The recorded experiments discuss an RTX 3060 with 12 GB VRAM.

The current model embeds 256 byte values, applies three unique transformer blocks twice (shared weights between passes), and predicts the next byte. Blocks use RMSNorm, causal attention with RoPE, and sparse MoE feed-forward layers with eight SwiGLU experts and top-2 routing. The current configuration is width 384, 6 heads, context 256, batch size 32, and 10,000 training steps. The historical dense width-512 model achieved best validation loss **0.6109** at step 9,500; the comparable dense width-384 baseline achieved **0.6338**. The full 10,000-step sparse MoE run achieved best LM validation loss **0.5721** at step 9,000; see its [run report](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md).

## Run overview

Historical dense report dates below are commit timestamps in UTC+02:00; actual run start times were not recorded. The new MoE report uses its actual run start time in CEST (UTC+02:00). Each report records its configuration and preserves the available terminal output and experiment notes. `Blocks × passes` distinguishes unique parameter sets from repeated computation. `—` means no reliable measurement is available.

Loss is held-out next-byte cross-entropy (lower is better): final validation for early runs, best periodic validation after checkpointing was introduced. The first three runs have no held-out validation. Dataset size, training budget, validation sampling, and implementation changed between runs, so results are not all directly comparable. Peak VRAM is PyTorch's maximum allocated CUDA memory, reported as “MB” by the scripts but calculated in MiB. Earlier timings exclude final validation; checkpointed timings include periodic validation and checkpoint writes.

| Report | Dim | Heads | Dim/head | Context | Blocks × passes | Norm | Feed-forward | Position encoding | Batch | Steps | Train split | Sampled training bytes | Parameters | Time | Bytes/sec | Peak VRAM (MiB) | Val loss | Comments |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| [2026-10-09 22:36](docs/runs/2026-10-09_22-36_43c5c8a.md) | 128 | 4 | 32 | 64 | 1 × 1 | LayerNorm | None | None | 1 | 1,000 | toy | 64,000 | — | — | — | — | — | Attention-only toy model |
| [2026-10-09 22:43](docs/runs/2026-10-09_22-43_556792e.md) | 128 | 4 | 32 | 64 | 1 × 1 | LayerNorm | GELU MLP | Learned position table | 1 | 1,000 | toy | 64,000 | — | — | — | — | — | Add transformer MLP and position table |
| [2026-10-09 22:51](docs/runs/2026-10-09_22-51_2bd1f09.md) | 128 | 4 | 32 | 64 | 4 × 1 | LayerNorm | GELU MLP | Learned position table | 1 | 1,000 | toy | 64,000 | — | — | — | — | — | Four unique blocks |
| [2026-10-09 23:02 (stale output)](docs/runs/2026-10-09_23-02_70a126b.md) | 128 | 4 | 32 | 64 | 4 × 1 | LayerNorm | GELU MLP | Learned position table | 1 | 1,000 | 1% | — | — | — | — | — | — | TinyStories code; retained toy output |
| [2026-10-09 23:06](docs/runs/2026-10-09_23-06_983a31f.md) | 128 | 4 | 32 | 64 | 4 × 1 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 1,280,000 | 867,328 | 140.86s | ~9,087 | 32.9 | 1.2925 | 20k-step TinyStories baseline |
| [2026-10-09 23:14](docs/runs/2026-10-09_23-14_25b29bb.md) | 128 | 4 | 32 | 64 | 1 × 4 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 1,280,000 | 272,512 | 141.00s | ~9,078 | 23.0 | 1.4159 | One shared block ×4 |
| [2026-10-09 23:19](docs/runs/2026-10-09_23-19_132b786.md) | 128 | 4 | 32 | 64 | 2 × 2 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 1,280,000 | 470,784 | 138.58s | ~9,237 | 26.0 | 1.3403 | Two shared blocks ×2 |
| [2026-10-09 23:25](docs/runs/2026-10-09_23-25_d75f856.md) | 128 | 4 | 32 | 64 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 1,280,000 | 470,784 | 194.78s | ~6,572 | 27.2 | 1.3039 | Two shared blocks ×3 |
| [2026-10-09 23:31](docs/runs/2026-10-09_23-31_4abf7da.md) | 128 | 4 | 32 | 64 | 2 × 4 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 1,280,000 | 470,784 | 247.96s | ~5,162 | 28.4 | 1.3072 | Two shared blocks ×4 |
| [2026-10-09 23:37](docs/runs/2026-10-09_23-37_a77460e.md) | 128 | 4 | 32 | 128 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 478,976 | 189.68s | ~13,496 | 31.8 | 1.2649 | Context 128 |
| [2026-10-09 23:42](docs/runs/2026-10-09_23-42_b5322b8.md) | 128 | 4 | 32 | 256 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 5,120,000 | 495,360 | 195.54s | ~26,184 | 43.5 | 1.2619 | Context 256 |
| [2026-10-09 23:49](docs/runs/2026-10-09_23-49_a52a788.md) | 256 | 4 | 64 | 128 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 1,744,128 | 198.15s | ~12,920 | 57.7 | 1.1105 | Width 256 |
| [2026-10-09 23:54](docs/runs/2026-10-09_23-54_6f57db8.md) | 384 | 4 | 96 | 128 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 3,795,712 | 204.99s | ~12,488 | 99.0 | 1.0384 | Width 384 |
| [2026-10-10 00:01](docs/runs/2026-10-10_00-01_3745c38.md) | 512 | 4 | 128 | 128 | 2 × 3 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 6,633,728 | 216.24s | ~11,839 | 150.3 | 1.0321 | Width 512 |
| [2026-10-10 00:09](docs/runs/2026-10-10_00-09_6770ac9.md) | 384 | 4 | 96 | 128 | 3 × 2 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 5,570,176 | 218.61s | ~11,710 | 126.1 | 1.0097 | Three shared blocks ×2 |
| [2026-10-10 00:15](docs/runs/2026-10-10_00-15_4f7e6f7.md) | 384 | 4 | 96 | 128 | 4 × 1 | LayerNorm | GELU MLP | Learned position table | 1 | 20,000 | 1% | 2,560,000 | 7,344,640 | 173.28s | ~14,774 | 158.0 | 1.0239 | Four unique blocks; one pass |
| [2026-10-10 00:21](docs/runs/2026-10-10_00-21_6f80b7e.md) | 384 | 4 | 96 | 128 | 3 × 2 | LayerNorm | GELU MLP | Learned position table | 32 | 625 | 1% | 2,560,000 | 5,570,176 | 41.06s | 62,348 | 745.4 | 1.3875 | Random batch sampling; seed 42 |
| [2026-10-10 00:31](docs/runs/2026-10-10_00-31_a03938b.md) | 384 | 4 | 96 | 128 | 3 × 2 | LayerNorm | GELU MLP | Learned position table | 32 | 5,000 | 1% | 20,480,000 | 5,570,176 | 328.62s | 62,321 | 745.4 | 0.8539 | 5k steps |
| [2026-10-10 00:56](docs/runs/2026-10-10_00-56_485b371.md) | 384 | 4 | 96 | 128 | 3 × 2 | LayerNorm | GELU MLP | Learned position table | 32 | 10,000 | 1% | 40,960,000 | 5,570,176 | 670.40s | 61,098 | 745.4 | 0.7820 | Periodic validation; best/last checkpoints |
| [2026-10-10 01:10](docs/runs/2026-10-10_01-10_8e5f76b.md) | 384 | 4 | 96 | 128 | 3 × 2 | LayerNorm | GELU MLP | Learned position table | 32 | 10,000 | 5% | 40,960,000 | 5,570,176 | 679.34s | 60,294 | 745.4 | 0.7826 | Train split increased to 5% |
| [2026-10-10 01:25](docs/runs/2026-10-10_01-25_377ff48.md) | 384 | 4 | 96 | 128 | 3 × 2 | RMSNorm | GELU MLP | Learned position table | 32 | 10,000 | 5% | 40,960,000 | 5,567,488 | 671.82s | 60,969 | 745.2 | 0.7821 | Switch to RMSNorm |
| [2026-10-10 01:41](docs/runs/2026-10-10_01-41_601bac6.md) | 384 | 4 | 96 | 128 | 3 × 2 | RMSNorm | SwiGLU | Learned position table | 32 | 10,000 | 5% | 40,960,000 | 5,561,728 | 695.64s | 58,881 | 845.3 | 0.7463 | Switch to dense SwiGLU |
| [2026-10-10 01:56](docs/runs/2026-10-10_01-56_a0c1968.md) | 384 | 4 | 96 | 128 | 3 × 2 | RMSNorm | SwiGLU | RoPE | 32 | 10,000 | 5% | 40,960,000 | 5,507,968 | 658.02s | 62,247 | 873.2 | 0.7280 | RoPE; custom causal attention |
| [2026-10-10 10:40](docs/runs/2026-10-10_10-40_918e40a.md) | 384 | 6 | 64 | 128 | 3 × 2 | RMSNorm | SwiGLU | RoPE | 32 | 10,000 | 5% | 40,960,000 | 5,507,968 | 628.93s | 65,126 | 873.2 | 0.7222 | Six heads (64 dimensions/head) |
| [2026-10-10 11:07](docs/runs/2026-10-10_11-07_31ca79e.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 5,507,968 | 1270.11s | 64,498 | 1655.7 | 0.6338 | Context 256; configurable block count |
| [2026-10-10 11:51](docs/runs/2026-10-10_11-51_f633f44.md) | 512 | 8 | 64 | 256 | 3 × 2 | RMSNorm | SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 9,701,632 | 2074.53s | 39,489 | 2227.9 | 0.6109 | Width 512; eight heads |
| [2026-10-10 13:13:05](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 30,289,792 | 2312.09s | 35,431 | 3296.9 | 0.5721 | 8 experts/block; top-2; aux weight 0.01; ~9.06M active parameters/traversal |

Sampled training bytes are `steps × batch size × context`, including repeated samples, rather than unique corpus size. Bytes/sec is copied from the terminal report when available; `~` marks throughput inferred from sampled bytes and the reported elapsed time. The stale-output revision has no reliable sampled-byte or throughput result.

The Comments column records the main change for each run and can describe future MoE configurations (expert count, top-k routing, total/active parameters), offloading (which components and to which device), or other architecture/runtime changes. The historical feed-forward networks in this table are dense. The new MoE row records the implemented sparse configuration. Full architecture details, validation curves, and generated samples remain in the linked reports. The final width-512 experiment also changes heads from 6 to 8 to preserve 64 dimensions per head.

## Setup and run

Use a Python environment with PyTorch (including `torch.nn.RMSNorm`) and Hugging Face Datasets. The script automatically selects CUDA when available and otherwise uses the CPU. The first run downloads TinyStories.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy datasets
python main.py
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. For CUDA, install the PyTorch build appropriate to your environment. The original experiment setup recorded `torch==2.13.0` from `https://download.pytorch.org/whl/cu132` (e.g. `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132`); this is historical setup information, not a verified dependency lock.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Edit configuration values at the top of [main.py](main.py) to change model dimensions, context, batch size, training steps, or validation frequency. The architecture is in [tinylm.py](tinylm.py). Training samples `train[:5%]`; validation samples `validation[:1%]`. Every 500 steps it validates on 20 random batches. It writes `best_model.pt` and `last_model.pt` in the working directory, then reloads the best checkpoint to generate 200 bytes from “Once upon a time”. Subsequent runs overwrite those checkpoint filenames.

## Sparse MoE

Each unique block owns a learned `384 → 8` router and eight independent SwiGLU experts with hidden width 1,024. Each token selects two experts, and their router probabilities are renormalized to sum to one. Experts run on gathered assigned tokens only; weighted outputs are added back to the original positions. Empty experts are skipped, and tokens are never dropped. Three unique blocks contain 24 experts in total; the second recurrent pass reuses those same weights and routes again.

`TinyLM.forward(tokens)` returns `(logits, router_aux_loss)`. Training uses `LM cross-entropy + 0.01 * router_aux_loss`. The Switch-style balancing loss is `E * sum(f * p)`, where `f` is each expert's fraction of top-k assignments and `p` is its mean probability before top-k selection. Uniform routing gives a value of 1. The auxiliary losses are averaged over all block calls, so the coefficient keeps the same scale when the layer count or recurrent passes change. Validation and best-checkpoint selection use only LM cross-entropy. Logged utilization percentages sum to 100% per block and describe the most recent training batch's assignments over all recurrent passes, rather than unique tokens or the entire run.

The benchmark configuration has **30,289,792 parameters**, compared with **5,507,968** for the dense baseline, and activates two of eight experts per token per block call. Feed-forward computation is approximately twice the dense baseline, because each selected expert is a full-size dense SwiGLU. All expert weights and optimizer states still consume memory. The explicit Python dispatch loop favors readability; gathers, scatters, and CUDA synchronization for variable-size expert batches add overhead compared with fused MoE kernels. There is no expert capacity limit, so imbalanced routing can also increase memory pressure. Dense checkpoints use a different feed-forward parameter layout and cannot be loaded directly into the MoE model. Both new checkpoints include the architecture settings and auxiliary-loss coefficient.

Run the routing, gradient, recurrence, and checkpoint checks with:

```bash
python -m unittest discover -s tests -v
```

These checks use CUDA when available and otherwise run on CPU. They compare outputs and gradients against a dense reference calculation used only in the tests, inspect the actual token inputs received by experts, and confirm that unselected experts are skipped.

A 10-step smoke run on the RTX 3060 with PyTorch `2.13.0+cu132` used the configuration above and the cached TinyStories splits. It completed training, validation, best/last checkpoint writes, best-checkpoint reload, and 200-byte generation. Peak allocated VRAM was **3,292.6 MiB**, throughput was **22,482 bytes/sec** over **3.64 seconds**, and LM validation loss was **2.6853** over two validation batches (training LM loss **2.7827**, router auxiliary loss **1.1661**). Forward hooks on the full configuration observed exactly **98,304 expert-token rows** across six block calls, versus **393,216** if all eight experts had run on every token. All four unit tests passed on both CPU and CUDA. The smoke timing includes validation and checkpoint writes, and its small training budget is insufficient to compare steady-state throughput or validation quality with the 10,000-step dense baseline. The subsequent [full 10,000-step report](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md) records best LM validation loss **0.5721**, **3,296.9 MiB** peak allocated VRAM, and **35,431 bytes/sec**.

## Recording future runs

Create `docs/runs/YYYY-MM-DD_HH-MM-SS_<experiment>.md` using the actual run start time and a short descriptive experiment name, for example `2026-10-10_13-13-05_sparse-moe_8e_top2.md`. Keep the timezone, exact code revision, and any uncommitted source changes inside the report. Existing historical report filenames retain their commit suffixes. If two runs start in the same second, append a run suffix.

Use the [run report template](docs/runs/TEMPLATE.md), save the complete terminal output and configuration, and add one row to the overview above. Record architecture changes (including MoE routing and total/active parameters or offloading details where applicable), dataset splits, sampled training bytes, throughput, seed, optimizer settings, environment/GPU, parameter count, elapsed time, peak VRAM, best/final validation loss, checkpoint step, and generated text. Keep each experiment's checkpoints under distinct names if you want to retain them.
