# Run: Sparse MoE, eight SwiGLU experts with top-1 routing

> **Historical/pre-EOS experiment.** Tokenization, metrics, sampling, and
> timing follow this report's recorded implementation, not current code.
> Conversational recommendations are historical advice, not verified conclusions.

[Run overview](HISTORY.md#run-overview)

- Report recorded: 2026-10-10; actual run start time and timezone were not supplied. The filename uses the report date without an invented start time.
- Code revision: The supplied configuration matches `1b4ae329e49747ca9cd10e6e3c4a354f461bd72a` plus the current uncommitted change in `main.py`, `top_k = 2` → `top_k = 1`. The exact revision used for the run was not supplied; configuration details below are reconstructed from the current source where absent from the transcript.
- Environment: CUDA, as reported; Python, PyTorch, CUDA runtime, and GPU details were not supplied for this run.
- Change being tested: Select one expert per token per block call instead of two, retaining eight experts per block and auxiliary-loss weight 0.01. Compare with the [top-2 MoE run](2026-10-10_13-13-05_sparse-moe_8e_top2.md) and [dense width-384/context-256 baseline](2026-10-10_11-07_31ca79e.md).

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | 256 byte values; UTF-8 text, no external tokenizer |
| Dim / heads / head dim | 384 / 6 / 64 |
| Context / batch / steps | 256 / 32 / 10,000 |
| Unique blocks / recurrent passes | 3 / 2; six block applications, shared weights between passes |
| Normalization / feed-forward / position encoding | RMSNorm / sparse SwiGLU MoE, expert hidden width 1,024 / RoPE on Q and K |
| Architecture/runtime comments (main change) | Causal scaled dot-product attention; pre-norm residual blocks; gather/expert/index-add dispatch; no capacity limit or dropped tokens; auxiliary loss averaged over all six block calls |
| MoE (if used): experts / top-k / routing / total and active parameters | 8 experts per block (24 unique total) / top-1 / learned bias-free router, normalized selected probability (weight 1 for top-1) / 30,289,792 total; 5,517,184 active per single traversal, counting one expert per block plus all shared parameters (3,538,944 selected expert parameters + 1,978,240 shared). The recurrent pass can select different experts. |
| Offloading (if used): components / destination device / policy | None; model and optimizer on CUDA, corpus tensors on CPU, sampled batches copied to CUDA |
| Dataset / train and validation splits / corpus bytes | TinyStories (`roneneldan/TinyStories`); `train[:5%]` / `validation[:1%]`; 93,815,188 training bytes / 187,001 validation bytes; stories joined with newlines |
| Seed / precision | 42 for CPU and CUDA / float32, no AMP, as configured in the current source |
| Optimizer / learning rate / other settings | AdamW; lr `0.0003`; other settings use PyTorch defaults; no scheduler, gradient clipping, or gradient accumulation |
| Validation frequency / samples | Every 500 steps, 20 randomly sampled batches of 32 × 256 positions (163,840 target bytes per validation); LM cross-entropy only; training objective = LM cross-entropy + 0.01 × router auxiliary loss |
| Generation method / prompt | Greedy argmax, last 256 bytes as context, 200 new bytes; prompt "Once upon a time"; reload best LM-validation checkpoint; UTF-8 decode with replacement |

## Results

| Metric | Value |
|---|---|
| Parameter count | 30,289,792 |
| Sampled training bytes | 81,920,000 = 10,000 × 32 × 256, including repeated samples |
| Training time (s; specify what is included) | 1708.18; current timing includes training, periodic validation, diagnostics, and best/final checkpoint writes; excludes dataset loading, model construction, best-checkpoint reload, and generation |
| Peak allocated CUDA memory (MiB) | 2296.0; script prints "MB" but divides allocated bytes by 1024² |
| Throughput (bytes/s) | 47,958 as reported; the displayed bytes/time give approximately 47,957.5 |
| Best validation loss / step | 0.6515 / 9,000; LM cross-entropy without router loss |
| Final validation loss | 0.6574 at step 10,000 |
| Checkpoint locations | `best_model.pt` (step 9,000) and `last_model.pt` (step 10,000), as reported in the execution directory; no distinct archived locations supplied |

## Observations

| Metric | Dense baseline | Top-2 MoE | Top-1 MoE |
|---|---:|---:|---:|
| Total parameters | 5,507,968 | 30,289,792 | 30,289,792 |
| Active feed-forward networks/token/block call | 1 | 2 of 8 | 1 of 8 |
| Peak allocated CUDA memory (MiB) | 1,655.7 | 3,296.9 | 2,296.0 |
| Training time (s) | 1,270.11 | 2,312.09 | 1,708.18 |
| Throughput (bytes/s) | 64,498 | 35,431 | 47,958 |
| Best LM validation loss | 0.6338 | 0.5721 | 0.6515 |

Top-1 routing reduced training time by **26.1%** and peak allocated VRAM by **30.4%** relative
to top-2, while increasing throughput by **35.4%**. Best validation loss increased by **0.0794**
(13.9%). Relative to the dense baseline, top-1 took **34.5%** longer, used **38.7%** more peak
VRAM, and had **0.0177** higher best validation loss (2.8%). Total parameter count is unchanged
from top-2; selecting fewer experts reduces active feed-forward work and activation memory.

All experts received assignments at every logged checkpoint. Router auxiliary loss ranged from
**1.0006 to 1.0045**, close to the uniform-routing value of 1. Final assignment shares ranged
from **10.2% to 14.3%**. These are diagnostics from the last training batch at each checkpoint,
summed over both recurrent passes per unique block; they are not run-wide utilization statistics.

The run completed 10,000 steps, validated 20 times, saved best/final checkpoints, and reloaded
the step-9,000 checkpoint for generation. The supplied generation sample is preserved exactly,
including its ending at `Li`. This is a single run with randomly sampled validation batches;
the baseline comparisons use historical results without repeated-seed variance estimates or a
fixed validation sweep. Checkpoints and environment metadata were not independently inspected.

### Validation curve

| Step | Train LM | Router auxiliary | Validation LM |
|---:|---:|---:|---:|
| 500 | 1.1248 | 1.0028 | 1.0396 |
| 1,000 | 0.9520 | 1.0045 | 0.9007 |
| 1,500 | 0.9190 | 1.0028 | 0.8452 |
| 2,000 | 0.8917 | 1.0020 | 0.8055 |
| 2,500 | 0.7865 | 1.0009 | 0.7732 |
| 3,000 | 0.8201 | 1.0013 | 0.7600 |
| 3,500 | 0.8402 | 1.0006 | 0.7399 |
| 4,000 | 0.7748 | 1.0015 | 0.7363 |
| 4,500 | 0.7640 | 1.0011 | 0.7251 |
| 5,000 | 0.7698 | 1.0008 | 0.7162 |
| 5,500 | 0.7669 | 1.0007 | 0.6999 |
| 6,000 | 0.7202 | 1.0014 | 0.6997 |
| 6,500 | 0.7295 | 1.0010 | 0.7010 |
| 7,000 | 0.6879 | 1.0016 | 0.6845 |
| 7,500 | 0.7171 | 1.0008 | 0.6878 |
| 8,000 | 0.7201 | 1.0019 | 0.6682 |
| 8,500 | 0.7435 | 1.0011 | 0.6631 |
| 9,000 | 0.7107 | 1.0029 | 0.6515 |
| 9,500 | 0.6911 | 1.0009 | 0.6604 |
| 10,000 | 0.6906 | 1.0020 | 0.6574 |

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
MoE: 8 experts/block, top-k=1, aux weight=0.01

Loading training data...
Training bytes: 93,815,188
Loading validation data...
Validation bytes: 187,001

Parameters: 30,289,792

    500: train_lm=1.1248  router_aux=1.0028  val_lm=1.0396
         expert assignment %: B0=[13.0,11.7,11.6,12.2,12.0,14.0,11.6,14.0]  B1=[14.3,10.9,13.7,10.5,12.5,11.7,12.7,13.7]  B2=[15.6,12.2,11.0,10.9,13.2,11.0,11.5,14.6]
       saved best_model.pt
   1000: train_lm=0.9520  router_aux=1.0045  val_lm=0.9007
         expert assignment %: B0=[11.9,14.7,10.9,10.6,13.2,12.3,13.4,13.0]  B1=[13.5,14.2,11.8,12.8,11.0,11.1,12.1,13.4]  B2=[13.2,13.9,13.4,10.8,13.1,10.2,13.0,12.2]
       saved best_model.pt
   1500: train_lm=0.9190  router_aux=1.0028  val_lm=0.8452
         expert assignment %: B0=[13.5,11.6,11.8,12.5,11.3,14.2,11.6,13.5]  B1=[11.1,12.3,11.6,12.3,14.5,12.4,12.7,13.1]  B2=[13.0,13.7,13.2,10.3,12.9,10.5,13.6,12.7]
       saved best_model.pt
   2000: train_lm=0.8917  router_aux=1.0020  val_lm=0.8055
         expert assignment %: B0=[11.3,11.6,13.8,10.9,12.8,13.5,12.5,13.6]  B1=[13.0,12.2,12.5,12.3,11.6,12.6,13.0,12.9]  B2=[12.4,13.1,11.8,11.5,13.1,12.1,13.1,12.9]
       saved best_model.pt
   2500: train_lm=0.7865  router_aux=1.0009  val_lm=0.7732
         expert assignment %: B0=[13.0,13.0,13.0,11.9,12.1,12.1,12.8,12.1]  B1=[11.6,12.1,14.6,13.2,12.4,12.4,12.1,11.6]  B2=[11.4,12.4,11.4,12.5,12.0,13.3,12.8,14.1]
       saved best_model.pt
   3000: train_lm=0.8201  router_aux=1.0013  val_lm=0.7600
         expert assignment %: B0=[12.0,13.8,11.9,12.8,12.0,12.4,12.8,12.4]  B1=[11.9,13.4,12.4,11.7,12.8,12.5,11.7,13.7]  B2=[12.2,11.8,13.4,11.2,12.4,13.5,12.0,13.5]
       saved best_model.pt
   3500: train_lm=0.8402  router_aux=1.0006  val_lm=0.7399
         expert assignment %: B0=[12.0,12.8,12.3,13.7,12.2,12.0,12.9,12.1]  B1=[11.9,12.4,13.4,12.4,13.5,13.2,12.5,10.7]  B2=[12.4,12.9,11.5,12.0,11.9,13.5,12.0,13.8]
       saved best_model.pt
   4000: train_lm=0.7748  router_aux=1.0015  val_lm=0.7363
         expert assignment %: B0=[14.1,13.2,11.6,12.1,12.0,11.9,13.2,12.0]  B1=[12.7,14.0,12.1,12.8,12.0,12.7,11.5,12.2]  B2=[10.8,12.4,12.6,13.2,12.7,12.6,13.3,12.4]
       saved best_model.pt
   4500: train_lm=0.7640  router_aux=1.0011  val_lm=0.7251
         expert assignment %: B0=[11.9,12.8,12.2,12.6,11.8,13.1,12.9,12.8]  B1=[11.5,11.8,12.4,13.5,12.1,13.8,12.0,12.8]  B2=[10.5,10.8,13.5,14.9,13.6,12.9,11.9,12.0]
       saved best_model.pt
   5000: train_lm=0.7698  router_aux=1.0008  val_lm=0.7162
         expert assignment %: B0=[12.5,12.0,11.7,13.8,12.1,13.0,13.5,11.4]  B1=[13.2,11.3,12.7,12.7,13.5,12.0,12.8,11.8]  B2=[11.6,12.0,12.4,13.1,11.5,13.6,12.6,13.2]
       saved best_model.pt
   5500: train_lm=0.7669  router_aux=1.0007  val_lm=0.6999
         expert assignment %: B0=[11.9,12.9,13.1,12.8,12.2,12.5,12.2,12.4]  B1=[12.8,11.2,11.8,13.1,13.2,12.5,11.9,13.4]  B2=[13.1,12.9,13.4,12.0,13.7,11.1,11.4,12.5]
       saved best_model.pt
   6000: train_lm=0.7202  router_aux=1.0014  val_lm=0.6997
         expert assignment %: B0=[14.7,12.9,11.4,10.8,12.4,12.9,12.2,12.6]  B1=[11.8,13.0,13.1,12.4,11.7,12.3,12.4,13.4]  B2=[12.6,12.2,11.9,13.7,13.5,11.2,12.7,12.2]
       saved best_model.pt
   6500: train_lm=0.7295  router_aux=1.0010  val_lm=0.7010
         expert assignment %: B0=[12.3,11.5,11.7,13.8,12.6,14.0,11.7,12.4]  B1=[13.1,13.4,13.0,11.7,12.6,12.6,12.0,11.5]  B2=[12.5,14.0,12.7,11.5,12.1,12.1,12.0,13.0]
   7000: train_lm=0.6879  router_aux=1.0016  val_lm=0.6845
         expert assignment %: B0=[13.5,12.0,14.2,12.0,12.4,12.8,11.2,11.9]  B1=[12.1,11.3,12.4,12.7,11.6,14.7,11.7,13.4]  B2=[12.7,13.8,11.2,12.0,13.0,12.2,12.6,12.5]
       saved best_model.pt
   7500: train_lm=0.7171  router_aux=1.0008  val_lm=0.6878
         expert assignment %: B0=[12.6,11.1,12.2,13.2,13.6,12.0,13.6,11.7]  B1=[12.2,11.4,12.3,12.9,12.4,13.9,14.1,10.8]  B2=[12.6,12.4,12.7,12.7,12.2,11.9,11.6,13.8]
   8000: train_lm=0.7201  router_aux=1.0019  val_lm=0.6682
         expert assignment %: B0=[13.0,12.8,13.2,11.0,12.4,12.9,12.4,12.3]  B1=[11.4,12.4,11.6,13.1,13.8,12.4,12.4,12.9]  B2=[12.1,13.6,12.8,13.4,11.6,11.4,14.1,11.0]
       saved best_model.pt
   8500: train_lm=0.7435  router_aux=1.0011  val_lm=0.6631
         expert assignment %: B0=[12.3,13.4,13.1,11.8,12.5,12.7,12.2,12.0]  B1=[12.9,13.8,11.3,11.9,12.7,12.9,12.7,11.7]  B2=[13.9,11.3,13.0,12.1,12.5,12.1,13.0,12.0]
       saved best_model.pt
   9000: train_lm=0.7107  router_aux=1.0029  val_lm=0.6515
         expert assignment %: B0=[12.6,11.6,11.8,11.9,12.5,14.6,12.5,12.4]  B1=[14.0,14.4,12.3,12.2,10.9,10.4,13.1,12.6]  B2=[13.9,13.7,13.8,11.3,13.5,11.7,11.2,10.9]
       saved best_model.pt
   9500: train_lm=0.6911  router_aux=1.0009  val_lm=0.6604
         expert assignment %: B0=[11.6,13.4,12.7,12.3,11.2,13.7,12.4,12.6]  B1=[11.6,13.1,11.9,13.0,12.7,13.2,12.0,12.5]  B2=[11.4,13.0,11.0,14.5,13.4,13.1,12.2,11.4]
  10000: train_lm=0.6906  router_aux=1.0020  val_lm=0.6574
         expert assignment %: B0=[12.4,13.3,12.8,12.5,12.8,10.2,13.5,12.4]  B1=[11.2,13.6,11.3,10.8,11.2,14.3,14.1,13.4]  B2=[11.7,12.8,13.3,14.0,11.4,12.0,13.1,11.7]
Saved last_model.pt

------------------------------
Results
------------------------------
Parameters:      30,289,792
Experts/token:   1 of 8 per block call
Training bytes:  81,920,000
Training time:   1708.18s
Peak VRAM:       2296.0 MB
Bytes/sec:       47,958
Best val LM loss: 0.6515
Using checkpoint from step 9000 (val=0.6515)

------------------------------
Generation
------------------------------

Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she went to the park to play with her friends. They played together and had lots of fun.

As they walked along, Li
```
