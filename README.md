# TinyLM

A small byte-level language model built with PyTorch, exploring how model width, context length, attention heads, and shared transformer blocks affect quality, training time, and GPU memory use. Early experiments use a few repeated sentences; later runs train on TinyStories. The recorded experiments discuss an RTX 3060 with 12 GB VRAM.

The baseline model embeds 256 byte values (optional locally trained BPE uses its actual vocabulary size), applies three unique transformer blocks twice (shared weights between passes), and predicts the next byte. Blocks use RMSNorm, causal attention with RoPE, and sparse MoE feed-forward layers with sixteen SwiGLU experts and top-2 routing. The current configuration is width 384, 6 heads, context 256, batch size 32, and an 81,920,000 source-byte training budget (10,000 steps in byte mode). The historical dense width-512 model achieved best validation loss **0.6109** at step 9,500; the comparable dense width-384 baseline achieved **0.6338**. The eight-expert 10,000-step sparse MoE run achieved best LM validation loss **0.5721** at step 9,000; see its [top-2 run report](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md). The [eight-expert top-1 run](docs/runs/2026-10-10_sparse-moe_8e_top1.md) achieved best LM validation loss **0.6515** at step 9,000, using **2,296.0 MiB** peak VRAM and **1,708.18 seconds** of training time. The [four-expert top-2 run](docs/runs/2026-10-10_sparse-moe_4e_top2.md) achieved **0.5863** at step 10,000 with **16,129,408 parameters**, **3,048.7 MiB** peak VRAM, and **2,149.92 seconds** of training time. The [six-expert top-2 run](docs/runs/2026-10-10_sparse-moe_6e_top2.md) achieved **0.5778** at step 10,000 with **23,209,600 parameters**, **3,177.9 MiB** peak VRAM, and **2,324.46 seconds** of training time. The earlier [sixteen-expert top-2 run](docs/runs/2026-10-10_sparse-moe_16e_top2.md) used a shorter 4,000-step budget and achieved **0.6412** at step 4,000 with **58,610,560 parameters**, **3,764.0 MiB** peak VRAM, and **1,100.73 seconds** of training time. At the same step, the six- and eight-expert top-2 runs had validation losses of **0.6734** and **0.6590**. The latest [10,000-step sixteen-expert top-2 run](docs/runs/2026-10-10_sparse-moe_16e_top2_10000steps.md) achieved best/final LM validation loss **0.5583** at step 10,000 with **58,610,560 parameters**, **3,762.7 MiB** peak VRAM, and **2,726.54 seconds** of training time. Its five longer greedy generation samples show substantial repetition.

## Run overview

Historical dense report dates below are commit timestamps in UTC+02:00; actual run start times were not recorded. The BPE reports use logged training start times in CEST (UTC+02:00). The eight-expert top-2 MoE report uses its actual run start time in CEST (UTC+02:00); the eight-expert top-1, four-expert top-2, six-expert top-2, and sixteen-expert top-2 reports use the dates they were recorded because their run start times were not supplied. Each report records its configuration and preserves the available terminal output and experiment notes. `Blocks × passes` distinguishes unique parameter sets from repeated computation. `—` means no reliable measurement is available.

Byte-row loss is held-out next-byte cross-entropy (lower is better): final validation for early runs, best periodic validation after checkpointing was introduced. BPE rows explicitly report best periodic validation BPB; their token losses are not comparable with byte losses. The first three runs have no held-out validation. Dataset size, training budget, validation sampling, and implementation changed between runs, so results are not all directly comparable. Peak VRAM is PyTorch's maximum allocated CUDA memory, reported as “MB” by the scripts but calculated in MiB. Earlier timings exclude final validation; checkpointed timings include periodic validation and checkpoint writes.

| Report | Dim | Heads | Dim/head | Context | Blocks × passes | Norm | Feed-forward | Position encoding | Batch | Steps | Train split | Sampled training bytes | Parameters | Time | Bytes/sec | Peak VRAM (MiB) | Val loss / BPB | Comments |
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
| [2026-10-10 (start time unknown)](docs/runs/2026-10-10_sparse-moe_8e_top1.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 30,289,792 | 1708.18s | 47,958 | 2296.0 | 0.6515 | 8 experts/block; top-1; aux weight 0.01; ~5.52M active parameters/traversal |
| [2026-10-10 (start time unknown)](docs/runs/2026-10-10_sparse-moe_4e_top2.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 16,129,408 | 2149.92s | 38,104 | 3048.7 | 0.5863 | 4 experts/block; top-2; aux weight 0.01; ~9.05M active parameters/traversal |
| [2026-10-10 (start time unknown)](docs/runs/2026-10-10_sparse-moe_6e_top2.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 23,209,600 | 2324.46s | 35,243 | 3177.9 | 0.5778 | 6 experts/block; top-2; aux weight 0.01; ~9.05M active parameters/traversal |
| [2026-10-10 (start time unknown)](docs/runs/2026-10-10_sparse-moe_16e_top2.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 4,000 | 5% | 32,768,000 | 58,610,560 | 1100.73s | 29,769 | 3764.0 | 0.6412 | 16 experts/block; top-2; aux weight 0.01; ~9.07M active parameters/traversal; shorter 4k-step budget |
| [2026-10-10 (start time unknown)](docs/runs/2026-10-10_sparse-moe_16e_top2_10000steps.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | 81,920,000 | 58,610,560 | 2726.54s | 30,045 | 3762.7 | 0.5583 | 16 experts/block; top-2; aux weight 0.01; ~9.07M active parameters/traversal; five 1,000-byte greedy generations |
| [2026-10-10 18:42:24](docs/runs/2026-10-10_18-42-24_bpe_4096.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 10,000 | 5% | ~322,361,363 | 61,563,520 | 2832.03s | 113,827 | 4030.4 | 0.5577 BPB | BPE 4,096; 16 experts/block; top-2; 10k steps; ~3.94× byte baseline source-byte budget; training-step timing |
| [2026-10-11 09:53:47](docs/runs/2026-10-11_09-53-47_bpe_2048.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 2,776 | 5% | ~81,929,445 | 59,988,608 | 738.56s | 110,932 | 3887.7 | 0.6636 BPB | BPE 2,048; 16 experts/block; top-2; ~81.92M source-byte budget; training-step timing |
| [2026-10-11 10:11:29](docs/runs/2026-10-11_10-11-29_bpe_8192.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 2,441 | 5% | ~81,905,513 | 64,713,344 | 705.49s | 116,097 | 4562.4 | 0.6519 BPB | BPE 8,192; 16 experts/block; top-2; ~81.92M source-byte budget; training-step timing |
| [2026-10-11 10:33:38](docs/runs/2026-10-11_10-33-38_bpe_4096.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 2,541 | 5% | ~81,912,022 | 61,563,520 | 696.16s | 117,663 | 4027.0 | 0.6558 BPB | BPE 4,096; 16 experts/block; top-2; ~81.92M source-byte budget; training-step timing |
| [2026-10-11 10:47:52](docs/runs/2026-10-11_10-47-52_bpe_16384.md) | 384 | 6 | 64 | 256 | 3 × 2 | RMSNorm | Sparse MoE SwiGLU | RoPE | 32 | 2,422 | 5% | ~81,918,924 | 71,012,992 | 765.10s | 107,070 | 5658.3 | 0.6494 BPB | BPE 16,384; 16 experts/block; top-2; ~81.92M source-byte budget; training-step timing |

For byte rows, sampled training bytes are `steps × batch size × context`, including repeated samples, rather than unique corpus size. For BPE rows, this product counts tokens; estimated source bytes additionally multiply by the training corpus average bytes/token. BPE throughput and training time cover training steps only, whereas historical byte timings include periodic validation and checkpoint writes. Bytes/sec is copied from the terminal report when available; `~` marks throughput inferred from sampled bytes and the reported elapsed time. The stale-output revision has no reliable sampled-byte or throughput result.

The Comments column records the main change for each run and can describe future MoE configurations (expert count, top-k routing, total/active parameters), offloading (which components and to which device), or other architecture/runtime changes. The historical feed-forward networks in this table are dense. The MoE rows record the implemented sparse configurations. Full architecture details, validation curves, and generated samples remain in the linked reports. The final width-512 experiment also changes heads from 6 to 8 to preserve 64 dimensions per head.

## Setup and run

Use a Python environment with PyTorch (including `torch.nn.RMSNorm`) and Hugging Face Datasets. The script automatically selects CUDA when available and otherwise uses the CPU. The first run downloads TinyStories.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy datasets tokenizers
python train.py --model-prefix=some-prefix-
python run.py --model-prefix=some-prefix-
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. For CUDA, install the PyTorch build appropriate to your environment. The original experiment setup recorded `torch==2.13.0` from `https://download.pytorch.org/whl/cu132` (e.g. `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132`); this is historical setup information, not a verified dependency lock.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Edit `Config` at the top of [core.py](core.py) to change model dimensions, attention head size, context, batch size, training steps, validation settings, dataset splits, or generation prompts. The architecture remains in [tinylm.py](tinylm.py). Training samples `train[:5%]`; validation samples `validation[:1%]`. With the default byte budget, every 1,000 steps it validates on 10 random batches, with a final larger 100-batch pass on the last model. The best checkpoint is selected using periodic validation BPB; the last checkpoint records the larger final validation token loss and BPB. For byte mode BPB equals token loss divided by `ln(2)`, so selection ordering is unchanged.

[train.py](train.py) writes `model/some-prefix-best_model.pt` and `model/some-prefix-last_model.pt` for the prefix above. With no prefix, the filenames are `model/best_model.pt` and `model/last_model.pt`. It checks both destinations before loading data or starting training, and aborts if either already exists. Use a different prefix to keep separate experiments, or explicitly allow replacement:

```bash
python train.py --model-prefix=some-prefix- --force-overwrite
```

[run.py](run.py) loads the best checkpoint by default, restoring its saved architecture settings even if `Config` has since changed. To generate from the last checkpoint instead:

```bash
python run.py --model-prefix=some-prefix- --checkpoint=last
```

Generation uses greedy decoding until approximately `max_output_bytes=1000` newly decoded UTF-8 bytes for each of the five configured prompts, possibly exceeding the budget by the final token. Checkpoint save/load durations are logged. Training reports time spent in training steps separately from total runtime; validation, logging, and checkpoint writes are excluded from training time. Each evaluation reports token loss and BPB; progress reports tokens processed, tokens/sec, estimated source-byte throughput based on the training corpus average, and remaining training-loop time. Generation reports generated tokens and decoded UTF-8 bytes, both rates excluding the prompt. Invalid generated UTF-8 is replaced, and replacement characters count toward the displayed byte budget; this differs from historical raw-byte generation counts.

Both commands immediately mirror console output into `docs/runs/<date-time>-<model-prefix>.md` (using `default` for an empty prefix). The timestamp includes microseconds to distinguish separate invocations. Logs include the timezone, command, configuration, measurements, and generated text; errors and interrupted runs retain the output already written.

## Optional byte-level BPE experiment

Byte mode remains the default (`Config.tokenizer_type="byte"`). To run the first
BPE benchmark with all existing model/training defaults unchanged:

```bash
python train.py --model-prefix=bpe-4096- --tokenizer-type=bpe --bpe-vocab-size=4096
python run.py --model-prefix=bpe-4096-
```

Alternatively set `tokenizer_type="bpe"` and `bpe_vocab_size=4096` in `Config`.
`bpe_min_frequency=2`. Training uses Hugging Face `tokenizers`' `models.BPE`,
`ByteLevel(add_prefix_space=False, use_regex=True)` pre-tokenization and the
ByteLevel decoder, without normalization, dropout, pretrained vocabulary, or
special tokens. The trainer includes all 256 ByteLevel alphabet symbols. The
factory trains exclusively on the newline-joined training split; validation is
encoded using that same tokenizer. The actual vocabulary can be smaller than
requested on small corpora, and it sets both model vocabulary layers.

The [CPU tokenizer/workflow smoke report](docs/runs/2026-10-10_bpe_smoke.md)
measured the intended TinyStories splits with an actual 4,096-entry vocabulary:

| Tokenizer | Training raw bytes | Training tokens | Training bytes/token | Validation bytes/token | Default model parameters |
|---|---:|---:|---:|---:|---:|
| Byte | 93,815,188 | 93,815,188 | 1.0000 | 1.0000 | 58,610,560 |
| BPE 4,096 | 93,815,188 | 23,840,761 | 3.9351 | 3.9504 | 61,563,520 |

This makes 256 BPE positions approximately 1,007 source bytes. The vocabulary
adds 2,952,960 parameters solely in the embedding/output layers.

The matched-budget BPE benchmarks are recorded in one report per vocabulary
size, each combining training and generation output. The earlier 10,000-step
4,096 benchmark remains a separate report:

| Vocabulary / report | Steps | Estimated sampled source bytes | Best BPB (step) | Final BPB | Training steps (s) | Peak VRAM (MiB) | Generated bytes/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| [2,048](docs/runs/2026-10-11_09-53-47_bpe_2048.md) | 2,776 | 81,929,445 | 0.6636 (2,770) | 0.6660 | 738.56 | 3,887.7 | 104.44 |
| [4,096 (earlier 10k steps)](docs/runs/2026-10-10_18-42-24_bpe_4096.md) | 10,000 | 322,361,363 | 0.5577 (10,000) | 0.5590 | 2,832.03 | 4,030.4 | 121.47 |
| [4,096](docs/runs/2026-10-11_10-33-38_bpe_4096.md) | 2,541 | 81,912,022 | 0.6558 (2,541) | 0.6641 | 696.16 | 4,027.0 | 111.07 |
| [8,192](docs/runs/2026-10-11_10-11-29_bpe_8192.md) | 2,441 | 81,905,513 | 0.6519 (2,440) | 0.6523 | 705.49 | 4,562.4 | 122.54 |
| [16,384](docs/runs/2026-10-11_10-47-52_bpe_16384.md) | 2,422 | 81,918,924 | 0.6494 (2,422) | 0.6508 | 765.10 | 5,658.3 | 121.85 |

The 2,048, 4,096, 8,192, and 16,384 runs use the same approximate 81.92M
source-byte budget. The 4,096 vocabulary trained fastest (696.16s); the 16,384
vocabulary achieved the lowest best/final BPB (0.6494 / 0.6508), with the highest
peak memory (5,658.3 MiB). Its improvement over 8,192 (0.6519 / 0.6523) is small
and comes with longer training time; these single-seed runs do not establish
statistical significance. The 16,384 invocation also encountered dataset-loading
timeouts, bringing total runtime to 2,182.58s; training-step timing excludes those
retries. The earlier 10,000-step 4,096 run used about 3.94 times the source-byte
budget, so its lower BPB does not isolate vocabulary size. The byte baseline's
0.5583 best token loss is approximately 0.8055 BPB. BPE also sees more source
bytes per 256-token context, and validation samples differ across tokenizations.
All recorded BPE runs still show repetition in their five greedy generation samples.

Every checkpoint embeds the complete tokenizer JSON, training settings, library
version, vocabulary size, architecture, run settings, validation metrics, selection
metric, and dataset byte/token statistics. Generation reconstructs this state
without tokenizer training or an external artifact. Old MoE byte checkpoints
without tokenizer metadata or vocabulary size load with the byte tokenizer and
256 vocabulary entries. Historical dense checkpoints still have incompatible
feed-forward weights. The Python generation helper now returns
`(text, generated_tokens, generated_utf8_bytes, elapsed)`; configuration uses
`max_output_bytes` instead of `max_new_bytes`.

BPB is the sum of target-token negative log likelihood in nats divided by
`ln(2) * target_source_bytes`, aggregated over validation batches. Exact source
bytes come from a precomputed vocabulary table: each reversible ByteLevel symbol
represents one original byte. Indexing this table with the shifted targets counts
partial UTF-8 characters and whitespace/newline merges exactly, without token
window decoding or overlapping Unicode offsets. Byte tokens have length one.
The denominator uses each evaluated window, not a global compression average.
Validation still samples uniformly in token positions and scores overlapping
windows; BPB is a sampled conditional prediction metric, rather than a full-corpus
compression benchmark with a fixed byte context.

Training defaults to `target_training_bytes=81_920_000`. After encoding the
training stream, it measures exact source bytes per token and sets
`steps = round(target_training_bytes / (batch_size * context * average_bytes_per_token))`.
This chooses the nearest whole-step estimate of the source-byte budget across
BPE vocabulary sizes. Validation runs every `max(1, steps // 10)` steps. Set
`target_training_bytes=None` in `Config` to use manual `steps` and `validate_every`.
The target, effective steps, tokens per step, estimated source bytes processed,
and validation interval are logged; checkpoints store the effective run settings.

Context remains **256 tokens**: byte mode sees 256 bytes, while BPE covers more
source text. Equal steps and batch/context sizes process equal token counts but
more source bytes for BPE; the default budget compensates by adjusting steps.
Vocabulary expansion also increases embedding/output
parameters and softmax computation. Token cross-entropies cannot be compared
across tokenizers; compare BPB, timings, memory, and generated text lengths.
Tokenizer preparation is included in total runtime but excluded from training-step
throughput. Random validation windows differ across tokenizations, and BPB-based
selection may pick a different BPE checkpoint than token-loss selection.

## Sparse MoE

Each unique block currently owns a learned `384 → 16` router and sixteen independent SwiGLU experts with hidden width 1,024. Each token selects `top_k` experts (currently two; earlier runs used four or six experts with top-2 or eight experts with top-1 or top-2 routing), and their router probabilities are renormalized to sum to one. For top-1, the selected expert has weight 1. Experts run on gathered assigned tokens only; weighted outputs are added back to the original positions. Empty experts are skipped, and tokens are never dropped. Three unique blocks currently contain 48 experts in total; the second recurrent pass reuses those same weights and routes again.

`TinyLM.forward(tokens)` returns `(logits, router_aux_loss)`. Training uses `LM cross-entropy + 0.01 * router_aux_loss`. The Switch-style balancing loss is `E * sum(f * p)`, where `f` is each expert's fraction of top-k assignments and `p` is its mean probability before top-k selection. Uniform routing gives a value of 1. The auxiliary losses are averaged over all block calls, so the coefficient keeps the same scale when the layer count or recurrent passes change. Validation reports LM token cross-entropy and BPB, excluding the router auxiliary term; best-checkpoint selection uses BPB. Logged utilization percentages sum to 100% per block and describe the most recent training batch's assignments over all recurrent passes, rather than unique tokens or the entire run.

The current sixteen-expert top-2 configuration has **58,610,560 parameters** and activates two of sixteen experts per token per block call. The earlier four-, six-, and eight-expert top-2 benchmarks have **16,129,408**, **23,209,600**, and **30,289,792 parameters**, respectively, compared with **5,507,968** for the dense baseline. All four top-2 configurations use approximately twice the dense baseline's feed-forward computation, because each selected expert is a full-size dense SwiGLU. All expert weights and optimizer states still consume memory. The explicit Python dispatch loop favors readability; gathers, scatters, and CUDA synchronization for variable-size expert batches add overhead compared with fused MoE kernels. There is no expert capacity limit, so imbalanced routing can also increase memory pressure. Dense checkpoints use a different feed-forward parameter layout and cannot be loaded directly into the MoE model. Best and last MoE checkpoints include the architecture settings and auxiliary-loss coefficient.

Run the routing, gradient, recurrence, and checkpoint checks with:

```bash
python -m unittest discover -s tests -v
```

These checks use CUDA when available and otherwise run on CPU. They compare outputs and gradients against a dense reference calculation used only in the tests, inspect the actual token inputs received by experts, and confirm that unselected experts are skipped.

A 10-step smoke run on the RTX 3060 with PyTorch `2.13.0+cu132` used the earlier eight-expert top-2 configuration and the cached TinyStories splits. It completed training, validation, best/last checkpoint writes, best-checkpoint reload, and 200-byte generation. Peak allocated VRAM was **3,292.6 MiB**, throughput was **22,482 bytes/sec** over **3.64 seconds**, and LM validation loss was **2.6853** over two validation batches (training LM loss **2.7827**, router auxiliary loss **1.1661**). Forward hooks on the full configuration observed exactly **98,304 expert-token rows** across six block calls, versus **393,216** if all eight experts had run on every token. All four unit tests passed on both CPU and CUDA. The smoke timing includes validation and checkpoint writes, and its small training budget is insufficient to compare steady-state throughput or validation quality with the 10,000-step dense baseline. The subsequent [full 10,000-step report](docs/runs/2026-10-10_13-13-05_sparse-moe_8e_top2.md) records best LM validation loss **0.5721**, **3,296.9 MiB** peak allocated VRAM, and **35,431 bytes/sec**.

## Recording future runs

The scripts automatically save transcripts under `docs/runs/`. For a curated experiment report, create `docs/runs/YYYY-MM-DD_HH-MM-SS_<experiment>.md` using the actual run start time and a short descriptive experiment name, for example `2026-10-10_13-13-05_sparse-moe_8e_top2.md`. Keep the timezone, exact code revision, and any uncommitted source changes inside the report. Existing historical report filenames retain their commit suffixes. If two reports use the same second, append a run suffix.

Use the [run report template](docs/runs/TEMPLATE.md), save the complete terminal output and configuration, and add one row to the overview above. Record architecture changes (including MoE routing and total/active parameters or offloading details where applicable), dataset splits, sampled training bytes, throughput, seed, optimizer settings, environment/GPU, parameter count, elapsed time, peak VRAM, best/final validation loss, checkpoint step, and generated text. Keep each experiment's checkpoints under distinct names if you want to retain them.
