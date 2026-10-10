# Run: Sparse MoE, six SwiGLU experts with top-2 routing

[Run overview](../../README.md)

- Report recorded: 2026-10-10; actual run start time and timezone were not supplied. The filename uses the report date without an invented start time.
- Code revision: The supplied configuration matches `fa6f4d3aabf523f5f64abe8ff6d54e2fae7bf3b0` plus the current uncommitted change in `main.py`, `num_experts = 4` → `num_experts = 6`. The exact revision used for the run was not supplied; configuration details below are reconstructed from the current source where absent from the transcript.
- Environment: CUDA, as reported; Python, PyTorch, CUDA runtime, and GPU details were not supplied for this run.
- Change being tested: Increase the expert pool from four to six per block while retaining top-2 routing and auxiliary-loss weight 0.01. Compare with the [four-expert top-2 run](2026-10-10_sparse-moe_4e_top2.md), [eight-expert top-2 run](2026-10-10_13-13-05_sparse-moe_8e_top2.md), and [dense width-384/context-256 baseline](2026-10-10_11-07_31ca79e.md).

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | 256 byte values; UTF-8 text, no external tokenizer |
| Dim / heads / head dim | 384 / 6 / 64 |
| Context / batch / steps | 256 / 32 / 10,000 |
| Unique blocks / recurrent passes | 3 / 2; six block applications, shared weights between passes |
| Normalization / feed-forward / position encoding | RMSNorm / sparse SwiGLU MoE, expert hidden width 1,024 / RoPE on Q and K |
| Architecture/runtime comments (main change) | Causal scaled dot-product attention; pre-norm residual blocks; gather/expert/index-add dispatch; no capacity limit or dropped tokens; auxiliary loss averaged over all six block calls |
| MoE (if used): experts / top-k / routing / total and active parameters | 6 experts per block (18 unique total) / top-2 / learned bias-free router, normalized selected probabilities / 23,209,600 total; 9,053,824 active per single traversal, counting two experts per block plus all shared parameters (7,077,888 selected expert parameters + 1,975,936 shared). The recurrent pass can select different experts; two expert evaluations per token per block call. |
| Offloading (if used): components / destination device / policy | None; model and optimizer on CUDA, corpus tensors on CPU, sampled batches copied to CUDA |
| Dataset / train and validation splits / corpus bytes | TinyStories (`roneneldan/TinyStories`); `train[:5%]` / `validation[:1%]`; 93,815,188 training bytes / 187,001 validation bytes; stories joined with newlines |
| Seed / precision | 42 for CPU and CUDA / float32, no AMP, as configured in the current source |
| Optimizer / learning rate / other settings | AdamW; lr `0.0003`; other settings use PyTorch defaults; no scheduler, gradient clipping, or gradient accumulation |
| Validation frequency / samples | Every 500 steps, 20 randomly sampled batches of 32 × 256 positions (163,840 target bytes per validation); LM cross-entropy only; training objective = LM cross-entropy + 0.01 × router auxiliary loss |
| Generation method / prompt | Greedy argmax, last 256 bytes as context, 200 new bytes; prompt "Once upon a time"; reload best LM-validation checkpoint; UTF-8 decode with replacement |

## Results

| Metric | Value |
|---|---|
| Parameter count | 23,209,600 |
| Sampled training bytes | 81,920,000 = 10,000 × 32 × 256, including repeated samples |
| Training time (s; specify what is included) | 2324.46; current timing includes training, periodic validation, diagnostics, and best/final checkpoint writes; excludes dataset loading, model construction, best-checkpoint reload, and generation |
| Peak allocated CUDA memory (MiB) | 3177.9; script prints "MB" but divides allocated bytes by 1024² |
| Throughput (bytes/s) | 35,243 as reported; the displayed bytes/time give approximately 35,242.6 |
| Best validation loss / step | 0.5778 / 10,000; LM cross-entropy without router loss |
| Final validation loss | 0.5778 at step 10,000 |
| Checkpoint locations | `best_model.pt` and `last_model.pt`, both at step 10,000, as reported in the execution directory; no distinct archived locations supplied |

## Observations

| Metric | Dense baseline | 4 experts, top-2 | 6 experts, top-2 | 8 experts, top-2 |
|---|---:|---:|---:|---:|
| Total parameters | 5,507,968 | 16,129,408 | 23,209,600 | 30,289,792 |
| Active feed-forward networks/token/block call | 1 | 2 of 4 | 2 of 6 | 2 of 8 |
| Peak allocated CUDA memory (MiB) | 1,655.7 | 3,048.7 | 3,177.9 | 3,296.9 |
| Training time (s) | 1,270.11 | 2,149.92 | 2,324.46 | 2,312.09 |
| Throughput (bytes/s) | 64,498 | 38,104 | 35,243 | 35,431 |
| Best LM validation loss | 0.6338 | 0.5863 | 0.5778 | 0.5721 |

Relative to four-expert top-2 routing, six experts lowered best validation loss by **0.0085**
(1.4%), with **43.9%** more parameters, **8.1%** longer training time, and **4.2%** more peak
allocated VRAM. The number and size of selected experts stay the same, so active feed-forward
work per token is unchanged; the larger pool increases stored expert parameters and optimizer state.

Relative to eight-expert top-2 routing, six experts used **23.4%** fewer parameters and **3.6%**
less peak allocated VRAM. Best validation loss increased by **0.0057** (1.0%), and training time
was **0.5%** longer. This run did not show a speed improvement over eight experts. Relative to
the dense baseline, best validation loss decreased by **0.0560** (8.8%), with **83.0%** longer
training time and **91.9%** more peak allocated VRAM.

All experts received assignments at every logged checkpoint. Router auxiliary loss ranged
from **1.0096 to 1.0350**, ending close to the uniform-routing value of 1. Final assignment
shares ranged from **14.0% to 19.4%**, compared with a uniform share of approximately 16.7%:

| Block | E0 | E1 | E2 | E3 | E4 | E5 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 18.3% | 18.0% | 14.1% | 16.4% | 17.7% | 15.4% |
| B1 | 18.3% | 15.9% | 17.5% | 14.8% | 18.6% | 15.0% |
| B2 | 19.4% | 14.0% | 17.4% | 16.5% | 15.9% | 16.7% |

These diagnostics describe the last training batch at each checkpoint, summed over both
recurrent passes per unique block; they are rounded and are not run-wide utilization statistics.

The run completed 10,000 steps, validated 20 times, saved best/final checkpoints, and reloaded
the step-10,000 checkpoint for generation. The supplied generation sample is preserved exactly,
including its ending at `show`. This is a single run with randomly sampled validation batches;
the baseline comparisons use historical results without repeated-seed variance estimates or a
fixed validation sweep. Changing the expert count also changes initialization RNG consumption
and subsequent batch samples. Checkpoints and environment metadata were not independently inspected.

### Validation curve

| Step | Train LM | Router auxiliary | Validation LM |
|---:|---:|---:|---:|
| 500 | 0.9964 | 1.0350 | 0.9609 |
| 1,000 | 0.9303 | 1.0291 | 0.8334 |
| 1,500 | 0.8134 | 1.0283 | 0.7768 |
| 2,000 | 0.8085 | 1.0176 | 0.7445 |
| 2,500 | 0.7669 | 1.0248 | 0.7147 |
| 3,000 | 0.7468 | 1.0195 | 0.6927 |
| 3,500 | 0.7435 | 1.0200 | 0.6716 |
| 4,000 | 0.6957 | 1.0160 | 0.6734 |
| 4,500 | 0.7451 | 1.0132 | 0.6627 |
| 5,000 | 0.6897 | 1.0159 | 0.6505 |
| 5,500 | 0.6721 | 1.0164 | 0.6343 |
| 6,000 | 0.6944 | 1.0134 | 0.6340 |
| 6,500 | 0.6868 | 1.0185 | 0.6250 |
| 7,000 | 0.6906 | 1.0218 | 0.6219 |
| 7,500 | 0.6662 | 1.0133 | 0.6022 |
| 8,000 | 0.6383 | 1.0198 | 0.5954 |
| 8,500 | 0.6503 | 1.0120 | 0.5980 |
| 9,000 | 0.6189 | 1.0098 | 0.5898 |
| 9,500 | 0.6460 | 1.0148 | 0.5851 |
| 10,000 | 0.6381 | 1.0096 | 0.5778 |

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
MoE: 6 experts/block, top-k=2, aux weight=0.01

Loading training data...
Training bytes: 93,815,188
Loading validation data...
Validation bytes: 187,001

Parameters: 23,209,600

    500: train_lm=0.9964  router_aux=1.0350  val_lm=0.9609
         expert assignment %: B0=[19.2,18.6,15.2,15.9,13.5,17.6]  B1=[16.2,18.1,16.4,14.9,19.6,15.0]  B2=[20.2,13.2,18.8,18.6,17.5,11.7]
       saved best_model.pt
   1000: train_lm=0.9303  router_aux=1.0291  val_lm=0.8334
         expert assignment %: B0=[19.2,19.5,14.9,16.3,14.7,15.3]  B1=[17.7,17.5,17.0,15.2,20.1,12.5]  B2=[19.0,12.3,19.0,18.6,17.2,13.9]
       saved best_model.pt
   1500: train_lm=0.8134  router_aux=1.0283  val_lm=0.7768
         expert assignment %: B0=[17.5,18.9,14.5,18.0,15.1,16.0]  B1=[18.0,17.4,15.4,14.1,21.2,14.0]  B2=[19.7,13.6,18.3,18.5,16.2,13.8]
       saved best_model.pt
   2000: train_lm=0.8085  router_aux=1.0176  val_lm=0.7445
         expert assignment %: B0=[17.3,19.2,15.6,16.9,16.2,14.8]  B1=[16.7,16.7,16.8,14.8,19.7,15.3]  B2=[18.6,16.5,17.7,16.8,14.1,16.4]
       saved best_model.pt
   2500: train_lm=0.7669  router_aux=1.0248  val_lm=0.7147
         expert assignment %: B0=[17.7,18.4,16.2,16.4,16.0,15.3]  B1=[17.5,16.4,17.3,15.0,19.0,14.8]  B2=[19.8,14.3,18.4,17.1,16.1,14.2]
       saved best_model.pt
   3000: train_lm=0.7468  router_aux=1.0195  val_lm=0.6927
         expert assignment %: B0=[19.0,19.2,14.6,16.6,16.1,14.6]  B1=[18.6,15.5,16.3,14.8,19.4,15.4]  B2=[19.8,14.5,18.4,16.4,15.6,15.5]
       saved best_model.pt
   3500: train_lm=0.7435  router_aux=1.0200  val_lm=0.6716
         expert assignment %: B0=[16.8,17.9,15.2,16.5,17.5,16.0]  B1=[16.3,15.6,17.4,14.8,20.1,15.8]  B2=[19.2,13.5,18.2,17.0,15.6,16.5]
       saved best_model.pt
   4000: train_lm=0.6957  router_aux=1.0160  val_lm=0.6734
         expert assignment %: B0=[18.0,18.9,14.6,17.1,16.9,14.4]  B1=[15.5,16.2,16.2,17.0,19.3,15.8]  B2=[19.1,13.2,18.6,16.8,16.0,16.4]
   4500: train_lm=0.7451  router_aux=1.0132  val_lm=0.6627
         expert assignment %: B0=[17.6,18.2,15.7,16.5,16.7,15.4]  B1=[15.7,16.8,16.8,17.9,17.8,15.0]  B2=[18.7,13.6,18.2,18.1,15.1,16.2]
       saved best_model.pt
   5000: train_lm=0.6897  router_aux=1.0159  val_lm=0.6505
         expert assignment %: B0=[18.2,18.3,15.6,17.4,15.4,15.0]  B1=[17.3,14.8,16.7,15.8,19.2,16.1]  B2=[19.0,14.7,18.4,17.5,14.4,16.0]
       saved best_model.pt
   5500: train_lm=0.6721  router_aux=1.0164  val_lm=0.6343
         expert assignment %: B0=[18.1,18.5,15.7,16.4,16.2,15.2]  B1=[17.3,16.8,16.9,15.6,19.4,14.1]  B2=[19.6,13.4,17.7,17.0,14.9,17.4]
       saved best_model.pt
   6000: train_lm=0.6944  router_aux=1.0134  val_lm=0.6340
         expert assignment %: B0=[17.5,18.5,14.4,16.5,16.5,16.7]  B1=[16.5,17.5,16.1,15.6,18.8,15.6]  B2=[19.5,14.5,17.8,17.6,15.4,15.2]
       saved best_model.pt
   6500: train_lm=0.6868  router_aux=1.0185  val_lm=0.6250
         expert assignment %: B0=[16.2,17.7,17.6,17.3,16.1,15.1]  B1=[16.4,16.3,15.8,17.6,19.0,14.8]  B2=[19.3,14.2,18.0,17.8,14.7,15.9]
       saved best_model.pt
   7000: train_lm=0.6906  router_aux=1.0218  val_lm=0.6219
         expert assignment %: B0=[13.3,18.8,17.6,17.3,18.0,15.1]  B1=[18.1,16.3,15.9,15.1,19.5,15.2]  B2=[19.7,13.1,17.8,17.9,15.4,16.1]
       saved best_model.pt
   7500: train_lm=0.6662  router_aux=1.0133  val_lm=0.6022
         expert assignment %: B0=[17.2,18.1,15.7,16.7,16.9,15.4]  B1=[16.9,16.7,17.0,14.9,18.2,16.3]  B2=[19.4,14.7,18.8,17.2,13.8,16.1]
       saved best_model.pt
   8000: train_lm=0.6383  router_aux=1.0198  val_lm=0.5954
         expert assignment %: B0=[14.6,18.7,15.4,17.2,17.9,16.1]  B1=[17.0,16.9,16.0,15.4,20.0,14.8]  B2=[19.5,13.4,18.1,16.9,15.9,16.2]
       saved best_model.pt
   8500: train_lm=0.6503  router_aux=1.0120  val_lm=0.5980
         expert assignment %: B0=[16.6,17.3,15.7,17.0,18.2,15.2]  B1=[16.1,17.6,16.1,16.7,18.8,14.7]  B2=[18.7,15.0,17.6,17.5,15.4,15.9]
   9000: train_lm=0.6189  router_aux=1.0098  val_lm=0.5898
         expert assignment %: B0=[18.6,17.3,15.5,16.3,16.3,15.9]  B1=[16.6,16.7,16.8,16.8,18.8,14.4]  B2=[18.8,14.9,17.7,17.3,14.9,16.3]
       saved best_model.pt
   9500: train_lm=0.6460  router_aux=1.0148  val_lm=0.5851
         expert assignment %: B0=[16.4,17.2,17.0,16.7,17.3,15.4]  B1=[18.5,16.4,15.5,15.7,19.1,14.9]  B2=[19.2,14.3,18.5,16.8,14.6,16.7]
       saved best_model.pt
  10000: train_lm=0.6381  router_aux=1.0096  val_lm=0.5778
         expert assignment %: B0=[18.3,18.0,14.1,16.4,17.7,15.4]  B1=[18.3,15.9,17.5,14.8,18.6,15.0]  B2=[19.4,14.0,17.4,16.5,15.9,16.7]
       saved best_model.pt
Saved last_model.pt

------------------------------
Results
------------------------------
Parameters:      23,209,600
Experts/token:   2 of 6 per block call
Training bytes:  81,920,000
Training time:   2324.46s
Peak VRAM:       3177.9 MB
Bytes/sec:       35,243
Best val LM loss: 0.5778
Using checkpoint from step 10000 (val=0.5778)

------------------------------
Generation
------------------------------

Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she found a big box in the garage. It was a big box with a lock on it. Lily was so happy and show
```
