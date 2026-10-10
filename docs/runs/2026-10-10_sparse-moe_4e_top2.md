# Run: Sparse MoE, four SwiGLU experts with top-2 routing

[Run overview](../../README.md)

- Report recorded: 2026-10-10; actual run start time and timezone were not supplied. The filename uses the report date without an invented start time.
- Code revision: The supplied configuration matches `e48ba71daa131a1872e3a85028fd311aa92cc68b` plus the current uncommitted changes in `main.py`, `num_experts = 8` → `num_experts = 4` and `top_k = 1` → `top_k = 2`. The exact revision used for the run was not supplied; configuration details below are reconstructed from the current source where absent from the transcript.
- Environment: CUDA, as reported; Python, PyTorch, CUDA runtime, and GPU details were not supplied for this run.
- Change being tested: Reduce the expert pool from eight to four per block while retaining top-2 routing and auxiliary-loss weight 0.01. Compare with the [eight-expert top-2 run](2026-10-10_13-13-05_sparse-moe_8e_top2.md), [eight-expert top-1 run](2026-10-10_sparse-moe_8e_top1.md), and [dense width-384/context-256 baseline](2026-10-10_11-07_31ca79e.md).

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | 256 byte values; UTF-8 text, no external tokenizer |
| Dim / heads / head dim | 384 / 6 / 64 |
| Context / batch / steps | 256 / 32 / 10,000 |
| Unique blocks / recurrent passes | 3 / 2; six block applications, shared weights between passes |
| Normalization / feed-forward / position encoding | RMSNorm / sparse SwiGLU MoE, expert hidden width 1,024 / RoPE on Q and K |
| Architecture/runtime comments (main change) | Causal scaled dot-product attention; pre-norm residual blocks; gather/expert/index-add dispatch; no capacity limit or dropped tokens; auxiliary loss averaged over all six block calls |
| MoE (if used): experts / top-k / routing / total and active parameters | 4 experts per block (12 unique total) / top-2 / learned bias-free router, normalized selected probabilities / 16,129,408 total; 9,051,520 active per single traversal, counting two experts per block plus all shared parameters (7,077,888 selected expert parameters + 1,973,632 shared). The recurrent pass can select different experts; two expert evaluations per token per block call. |
| Offloading (if used): components / destination device / policy | None; model and optimizer on CUDA, corpus tensors on CPU, sampled batches copied to CUDA |
| Dataset / train and validation splits / corpus bytes | TinyStories (`roneneldan/TinyStories`); `train[:5%]` / `validation[:1%]`; 93,815,188 training bytes / 187,001 validation bytes; stories joined with newlines |
| Seed / precision | 42 for CPU and CUDA / float32, no AMP, as configured in the current source |
| Optimizer / learning rate / other settings | AdamW; lr `0.0003`; other settings use PyTorch defaults; no scheduler, gradient clipping, or gradient accumulation |
| Validation frequency / samples | Every 500 steps, 20 randomly sampled batches of 32 × 256 positions (163,840 target bytes per validation); LM cross-entropy only; training objective = LM cross-entropy + 0.01 × router auxiliary loss |
| Generation method / prompt | Greedy argmax, last 256 bytes as context, 200 new bytes; prompt "Once upon a time"; reload best LM-validation checkpoint; UTF-8 decode with replacement |

## Results

| Metric | Value |
|---|---|
| Parameter count | 16,129,408 |
| Sampled training bytes | 81,920,000 = 10,000 × 32 × 256, including repeated samples |
| Training time (s; specify what is included) | 2149.92; current timing includes training, periodic validation, diagnostics, and best/final checkpoint writes; excludes dataset loading, model construction, best-checkpoint reload, and generation |
| Peak allocated CUDA memory (MiB) | 3048.7; script prints "MB" but divides allocated bytes by 1024² |
| Throughput (bytes/s) | 38,104 as reported; the displayed bytes/time give approximately 38,103.7 |
| Best validation loss / step | 0.5863 / 10,000; LM cross-entropy without router loss |
| Final validation loss | 0.5863 at step 10,000 |
| Checkpoint locations | `best_model.pt` and `last_model.pt`, both at step 10,000, as reported in the execution directory; no distinct archived locations supplied |

## Observations

| Metric | Dense baseline | 8 experts, top-2 | 8 experts, top-1 | 4 experts, top-2 |
|---|---:|---:|---:|---:|
| Total parameters | 5,507,968 | 30,289,792 | 30,289,792 | 16,129,408 |
| Active feed-forward networks/token/block call | 1 | 2 of 8 | 1 of 8 | 2 of 4 |
| Peak allocated CUDA memory (MiB) | 1,655.7 | 3,296.9 | 2,296.0 | 3,048.7 |
| Training time (s) | 1,270.11 | 2,312.09 | 1,708.18 | 2,149.92 |
| Throughput (bytes/s) | 64,498 | 35,431 | 47,958 | 38,104 |
| Best LM validation loss | 0.6338 | 0.5721 | 0.6515 | 0.5863 |

Relative to eight-expert top-2 routing, four experts reduced total parameters by **46.8%**,
training time by **7.0%**, and peak allocated VRAM by **7.5%**, while increasing throughput
by **7.5%**. Best validation loss increased by **0.0142** (2.5%). The number and size of
selected experts stay the same, so active feed-forward work per token is unchanged; the
smaller pool reduces stored expert parameters and optimizer state.

Relative to the dense baseline, best validation loss decreased by **0.0475** (7.5%), with
**69.3%** longer training time and **84.1%** more peak allocated VRAM. Best validation loss
was also **0.0652** lower than the eight-expert top-1 result, with greater time and memory use.

All experts received assignments at every logged checkpoint. Router auxiliary loss ranged
from **1.0019 to 1.0142**, approaching the uniform-routing value of 1. Final assignment shares
ranged from **21.7% to 28.8%**, compared with a uniform share of 25%:

| Block | E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| B0 | 28.8% | 23.7% | 24.0% | 23.4% |
| B1 | 27.9% | 25.3% | 21.7% | 25.1% |
| B2 | 25.4% | 23.9% | 25.7% | 25.0% |

These diagnostics describe the last training batch at each checkpoint, summed over both
recurrent passes per unique block; they are rounded and are not run-wide utilization statistics.

The run completed 10,000 steps, validated 20 times, saved best/final checkpoints, and reloaded
the step-10,000 checkpoint for generation. The supplied generation sample is preserved exactly,
including its ending at `park`. This is a single run with randomly sampled validation batches;
the baseline comparisons use historical results without repeated-seed variance estimates or a
fixed validation sweep. Changing the expert count also changes initialization RNG consumption
and subsequent batch samples. Checkpoints and environment metadata were not independently inspected.

### Validation curve

| Step | Train LM | Router auxiliary | Validation LM |
|---:|---:|---:|---:|
| 500 | 1.0251 | 1.0142 | 0.9767 |
| 1,000 | 0.9131 | 1.0136 | 0.8449 |
| 1,500 | 0.8328 | 1.0107 | 0.7925 |
| 2,000 | 0.7955 | 1.0090 | 0.7496 |
| 2,500 | 0.7803 | 1.0116 | 0.7202 |
| 3,000 | 0.7637 | 1.0106 | 0.7029 |
| 3,500 | 0.7562 | 1.0085 | 0.6921 |
| 4,000 | 0.7513 | 1.0114 | 0.6766 |
| 4,500 | 0.6785 | 1.0071 | 0.6715 |
| 5,000 | 0.7082 | 1.0060 | 0.6519 |
| 5,500 | 0.7346 | 1.0049 | 0.6425 |
| 6,000 | 0.6647 | 1.0079 | 0.6435 |
| 6,500 | 0.6906 | 1.0028 | 0.6285 |
| 7,000 | 0.6633 | 1.0035 | 0.6133 |
| 7,500 | 0.6504 | 1.0026 | 0.6122 |
| 8,000 | 0.6262 | 1.0030 | 0.6015 |
| 8,500 | 0.6504 | 1.0035 | 0.6045 |
| 9,000 | 0.6681 | 1.0034 | 0.5956 |
| 9,500 | 0.6592 | 1.0025 | 0.5877 |
| 10,000 | 0.6432 | 1.0019 | 0.5863 |

## Full terminal output

```text
python main.py
Device:     cuda
Context:    256
Batch size: 32
Steps:      10000
Validate every x steps: 500
Validate batches:       20
Transform Layers:       3
Target head dim:         64
Dim=384, Heads=6, Recurrent Steps=2
MoE: 4 experts/block, top-k=2, aux weight=0.01

Loading training data...
Training bytes: 93,815,188
Loading validation data...
Validation bytes: 187,001

Parameters: 16,129,408

    500: train_lm=1.0251  router_aux=1.0142  val_lm=0.9767
         expert assignment %: B0=[28.7,20.9,25.9,24.5]  B1=[30.1,23.7,23.1,23.1]  B2=[25.4,21.4,25.5,27.8]
       saved best_model.pt
   1000: train_lm=0.9131  router_aux=1.0136  val_lm=0.8449
         expert assignment %: B0=[28.7,21.9,25.3,24.2]  B1=[29.3,24.1,23.4,23.2]  B2=[25.7,22.8,24.2,27.2]
       saved best_model.pt
   1500: train_lm=0.8328  router_aux=1.0107  val_lm=0.7925
         expert assignment %: B0=[28.3,21.7,26.7,23.3]  B1=[29.0,24.9,21.7,24.4]  B2=[26.2,22.3,25.6,25.8]
       saved best_model.pt
   2000: train_lm=0.7955  router_aux=1.0090  val_lm=0.7496
         expert assignment %: B0=[29.1,22.2,25.5,23.2]  B1=[29.2,25.4,21.8,23.6]  B2=[25.4,22.5,26.1,26.1]
       saved best_model.pt
   2500: train_lm=0.7803  router_aux=1.0116  val_lm=0.7202
         expert assignment %: B0=[28.8,22.5,25.1,23.6]  B1=[29.2,25.1,22.0,23.7]  B2=[25.4,21.4,26.0,27.2]
       saved best_model.pt
   3000: train_lm=0.7637  router_aux=1.0106  val_lm=0.7029
         expert assignment %: B0=[28.2,22.3,25.6,23.9]  B1=[30.2,24.6,22.3,22.9]  B2=[24.8,22.4,25.8,27.0]
       saved best_model.pt
   3500: train_lm=0.7562  router_aux=1.0085  val_lm=0.6921
         expert assignment %: B0=[28.5,22.1,25.5,23.9]  B1=[30.2,25.4,22.0,22.4]  B2=[25.0,22.7,26.2,26.1]
       saved best_model.pt
   4000: train_lm=0.7513  router_aux=1.0114  val_lm=0.6766
         expert assignment %: B0=[28.1,21.7,25.6,24.6]  B1=[29.8,26.4,20.5,23.3]  B2=[24.6,21.8,27.2,26.3]
       saved best_model.pt
   4500: train_lm=0.6785  router_aux=1.0071  val_lm=0.6715
         expert assignment %: B0=[27.5,23.1,25.4,24.0]  B1=[28.5,25.5,21.2,24.8]  B2=[25.9,22.2,26.0,25.9]
       saved best_model.pt
   5000: train_lm=0.7082  router_aux=1.0060  val_lm=0.6519
         expert assignment %: B0=[27.8,21.8,26.5,23.9]  B1=[29.2,24.7,21.4,24.7]  B2=[26.0,22.6,25.1,26.2]
       saved best_model.pt
   5500: train_lm=0.7346  router_aux=1.0049  val_lm=0.6425
         expert assignment %: B0=[30.1,21.5,25.8,22.7]  B1=[27.8,25.5,22.6,24.1]  B2=[25.2,22.4,26.3,26.2]
       saved best_model.pt
   6000: train_lm=0.6647  router_aux=1.0079  val_lm=0.6435
         expert assignment %: B0=[28.8,22.7,24.6,23.9]  B1=[28.3,24.4,22.0,25.4]  B2=[26.1,22.0,25.5,26.4]
   6500: train_lm=0.6906  router_aux=1.0028  val_lm=0.6285
         expert assignment %: B0=[29.2,23.0,24.5,23.3]  B1=[28.8,26.1,21.4,23.7]  B2=[24.9,23.3,26.3,25.6]
       saved best_model.pt
   7000: train_lm=0.6633  router_aux=1.0035  val_lm=0.6133
         expert assignment %: B0=[27.9,23.3,25.2,23.6]  B1=[29.0,24.8,22.2,24.0]  B2=[24.0,25.1,25.6,25.2]
       saved best_model.pt
   7500: train_lm=0.6504  router_aux=1.0026  val_lm=0.6122
         expert assignment %: B0=[28.9,23.1,24.9,23.1]  B1=[25.7,24.5,23.0,26.8]  B2=[25.9,22.1,26.1,26.0]
       saved best_model.pt
   8000: train_lm=0.6262  router_aux=1.0030  val_lm=0.6015
         expert assignment %: B0=[27.9,23.8,25.1,23.2]  B1=[28.3,24.4,21.8,25.5]  B2=[26.3,22.8,25.0,25.9]
       saved best_model.pt
   8500: train_lm=0.6504  router_aux=1.0035  val_lm=0.6045
         expert assignment %: B0=[28.3,23.8,23.9,24.0]  B1=[28.5,26.2,21.8,23.5]  B2=[25.6,24.0,24.4,26.1]
   9000: train_lm=0.6681  router_aux=1.0034  val_lm=0.5956
         expert assignment %: B0=[27.1,24.2,24.6,24.1]  B1=[28.7,23.5,23.0,24.8]  B2=[25.5,23.4,25.6,25.5]
       saved best_model.pt
   9500: train_lm=0.6592  router_aux=1.0025  val_lm=0.5877
         expert assignment %: B0=[29.6,23.4,24.1,22.9]  B1=[28.0,24.5,22.0,25.4]  B2=[25.2,23.5,25.1,26.2]
       saved best_model.pt
  10000: train_lm=0.6432  router_aux=1.0019  val_lm=0.5863
         expert assignment %: B0=[28.8,23.7,24.0,23.4]  B1=[27.9,25.3,21.7,25.1]  B2=[25.4,23.9,25.7,25.0]
       saved best_model.pt
Saved last_model.pt

------------------------------
Results
------------------------------
Parameters:      16,129,408
Experts/token:   2 of 4 per block call
Training bytes:  81,920,000
Training time:   2149.92s
Peak VRAM:       3048.7 MB
Bytes/sec:       38,104
Best val LM loss: 0.5863
Using checkpoint from step 10000 (val=0.5863)

------------------------------
Generation
------------------------------

Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, they decided to play a game of hide and seek. Lily was so excited to see what was inside the park
```
