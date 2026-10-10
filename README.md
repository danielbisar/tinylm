# TinyLM

A small byte-level language model built with PyTorch, exploring how model width, context length, attention heads, and shared transformer blocks affect quality, training time, and GPU memory use. Early experiments use a few repeated sentences; later runs train on TinyStories. The recorded experiments discuss an RTX 3060 with 12 GB VRAM.

The current model embeds 256 byte values, applies three unique transformer blocks twice (shared weights between passes), and predicts the next byte. Blocks use RMSNorm, causal attention with RoPE, and SwiGLU. The current configuration is width 512, 8 heads, context 256, batch size 32, and 10,000 training steps. It achieved best validation loss **0.6109** at step 9,500.

## Run overview

Dates below are commit timestamps in UTC+02:00; actual run start times were not recorded. Each report reconstructs the configuration at that revision and preserves its available terminal output and experiment notes. `Blocks × passes` distinguishes unique parameter sets from repeated computation. `—` means no reliable measurement is available.

Loss is held-out next-byte cross-entropy (lower is better): final validation for early runs, best periodic validation after checkpointing was introduced. The first three runs have no held-out validation. Dataset size, training budget, validation sampling, and implementation changed between runs, so results are not all directly comparable. Peak VRAM is PyTorch's maximum allocated CUDA memory, reported as “MB” by the scripts but calculated in MiB. Earlier timings exclude final validation; checkpointed timings include periodic validation and checkpoint writes.

| Report | Dim | Heads | Context | Blocks × passes | Batch | Steps | Train split | Parameters | Time | Peak VRAM (MiB) | Val loss |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| [2026-10-09 22:36](docs/runs/2026-10-09_22-36_43c5c8a.md) | 128 | 4 | 64 | 1 × 1 | 1 | 1,000 | toy | — | — | — | — |
| [2026-10-09 22:43](docs/runs/2026-10-09_22-43_556792e.md) | 128 | 4 | 64 | 1 × 1 | 1 | 1,000 | toy | — | — | — | — |
| [2026-10-09 22:51](docs/runs/2026-10-09_22-51_2bd1f09.md) | 128 | 4 | 64 | 4 × 1 | 1 | 1,000 | toy | — | — | — | — |
| [2026-10-09 23:02 (stale output)](docs/runs/2026-10-09_23-02_70a126b.md) | 128 | 4 | 64 | 4 × 1 | 1 | 1,000 | 1% | — | — | — | — |
| [2026-10-09 23:06](docs/runs/2026-10-09_23-06_983a31f.md) | 128 | 4 | 64 | 4 × 1 | 1 | 20,000 | 1% | 867,328 | 140.86s | 32.9 | 1.2925 |
| [2026-10-09 23:14](docs/runs/2026-10-09_23-14_25b29bb.md) | 128 | 4 | 64 | 1 × 4 | 1 | 20,000 | 1% | 272,512 | 141.00s | 23.0 | 1.4159 |
| [2026-10-09 23:19](docs/runs/2026-10-09_23-19_132b786.md) | 128 | 4 | 64 | 2 × 2 | 1 | 20,000 | 1% | 470,784 | 138.58s | 26.0 | 1.3403 |
| [2026-10-09 23:25](docs/runs/2026-10-09_23-25_d75f856.md) | 128 | 4 | 64 | 2 × 3 | 1 | 20,000 | 1% | 470,784 | 194.78s | 27.2 | 1.3039 |
| [2026-10-09 23:31](docs/runs/2026-10-09_23-31_4abf7da.md) | 128 | 4 | 64 | 2 × 4 | 1 | 20,000 | 1% | 470,784 | 247.96s | 28.4 | 1.3072 |
| [2026-10-09 23:37](docs/runs/2026-10-09_23-37_a77460e.md) | 128 | 4 | 128 | 2 × 3 | 1 | 20,000 | 1% | 478,976 | 189.68s | 31.8 | 1.2649 |
| [2026-10-09 23:42](docs/runs/2026-10-09_23-42_b5322b8.md) | 128 | 4 | 256 | 2 × 3 | 1 | 20,000 | 1% | 495,360 | 195.54s | 43.5 | 1.2619 |
| [2026-10-09 23:49](docs/runs/2026-10-09_23-49_a52a788.md) | 256 | 4 | 128 | 2 × 3 | 1 | 20,000 | 1% | 1,744,128 | 198.15s | 57.7 | 1.1105 |
| [2026-10-09 23:54](docs/runs/2026-10-09_23-54_6f57db8.md) | 384 | 4 | 128 | 2 × 3 | 1 | 20,000 | 1% | 3,795,712 | 204.99s | 99.0 | 1.0384 |
| [2026-10-10 00:01](docs/runs/2026-10-10_00-01_3745c38.md) | 512 | 4 | 128 | 2 × 3 | 1 | 20,000 | 1% | 6,633,728 | 216.24s | 150.3 | 1.0321 |
| [2026-10-10 00:09](docs/runs/2026-10-10_00-09_6770ac9.md) | 384 | 4 | 128 | 3 × 2 | 1 | 20,000 | 1% | 5,570,176 | 218.61s | 126.1 | 1.0097 |
| [2026-10-10 00:15](docs/runs/2026-10-10_00-15_4f7e6f7.md) | 384 | 4 | 128 | 4 × 1 | 1 | 20,000 | 1% | 7,344,640 | 173.28s | 158.0 | 1.0239 |
| [2026-10-10 00:21](docs/runs/2026-10-10_00-21_6f80b7e.md) | 384 | 4 | 128 | 3 × 2 | 32 | 625 | 1% | 5,570,176 | 41.06s | 745.4 | 1.3875 |
| [2026-10-10 00:31](docs/runs/2026-10-10_00-31_a03938b.md) | 384 | 4 | 128 | 3 × 2 | 32 | 5,000 | 1% | 5,570,176 | 328.62s | 745.4 | 0.8539 |
| [2026-10-10 00:56](docs/runs/2026-10-10_00-56_485b371.md) | 384 | 4 | 128 | 3 × 2 | 32 | 10,000 | 1% | 5,570,176 | 670.40s | 745.4 | 0.7820 |
| [2026-10-10 01:10](docs/runs/2026-10-10_01-10_8e5f76b.md) | 384 | 4 | 128 | 3 × 2 | 32 | 10,000 | 5% | 5,570,176 | 679.34s | 745.4 | 0.7826 |
| [2026-10-10 01:25](docs/runs/2026-10-10_01-25_377ff48.md) | 384 | 4 | 128 | 3 × 2 | 32 | 10,000 | 5% | 5,567,488 | 671.82s | 745.2 | 0.7821 |
| [2026-10-10 01:41](docs/runs/2026-10-10_01-41_601bac6.md) | 384 | 4 | 128 | 3 × 2 | 32 | 10,000 | 5% | 5,561,728 | 695.64s | 845.3 | 0.7463 |
| [2026-10-10 01:56](docs/runs/2026-10-10_01-56_a0c1968.md) | 384 | 4 | 128 | 3 × 2 | 32 | 10,000 | 5% | 5,507,968 | 658.02s | 873.2 | 0.7280 |
| [2026-10-10 10:40](docs/runs/2026-10-10_10-40_918e40a.md) | 384 | 6 | 128 | 3 × 2 | 32 | 10,000 | 5% | 5,507,968 | 628.93s | 873.2 | 0.7222 |
| [2026-10-10 11:07](docs/runs/2026-10-10_11-07_31ca79e.md) | 384 | 6 | 256 | 3 × 2 | 32 | 10,000 | 5% | 5,507,968 | 1270.11s | 1655.7 | 0.6338 |
| [2026-10-10 11:51](docs/runs/2026-10-10_11-51_f633f44.md) | 512 | 8 | 256 | 3 × 2 | 32 | 10,000 | 5% | 9,701,632 | 2074.53s | 2227.9 | 0.6109 |

Normalization, feed-forward layers, position encoding, validation curves, throughput, and generated samples are detailed in the linked reports. The final width-512 experiment also changes heads from 6 to 8 to preserve 64 dimensions per head. Proposed MoE experiments in its notes have not been implemented.

## Setup and run

Use a Python environment with PyTorch (including `torch.nn.RMSNorm`) and Hugging Face Datasets. The script automatically selects CUDA when available and otherwise uses the CPU. The first run downloads TinyStories.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy datasets
python main.py
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. For CUDA, install the PyTorch build appropriate to your environment. The original experiment setup recorded `torch==2.13.0` from `https://download.pytorch.org/whl/cu132`; this is historical setup information, not a verified dependency lock.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Edit configuration values at the top of [main.py](main.py) to change model dimensions, context, batch size, training steps, or validation frequency. The architecture is in [tinylm.py](tinylm.py). Training samples `train[:5%]`; validation samples `validation[:1%]`. Every 500 steps it validates on 20 random batches. It writes `best_model.pt` and `last_model.pt` in the working directory, then reloads the best checkpoint to generate 200 bytes from “Once upon a time”. Subsequent runs overwrite those checkpoint filenames.

## Recording future runs

Create `docs/runs/YYYY-MM-DD_HH-MM_<commit>.md` using the actual run start time for new experiments; include the timezone and exact code revision inside the report. A short commit suffix keeps filenames unique and makes the implementation easy to locate. For multiple runs of the same revision in one minute, append seconds or a run suffix.

Use the [run report template](docs/runs/TEMPLATE.md), save the complete terminal output and configuration, and add one row to the overview above. Record architecture changes, dataset splits, seed, optimizer settings, environment/GPU, parameter count, elapsed time, peak VRAM, best/final validation loss, checkpoint step, and generated text. Keep each experiment's checkpoints under distinct names if you want to retain them.
