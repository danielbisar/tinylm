# Run: Sparse MoE, sixteen SwiGLU experts with top-2 routing, 4,000 steps

> **Historical/pre-EOS experiment.** Tokenization, metrics, sampling, and
> timing follow this report's recorded implementation, not current code.
> Conversational recommendations are historical advice, not verified conclusions.

[Run overview](HISTORY.md#run-overview)

- Report recorded: 2026-10-10; actual run start time and timezone were not supplied. The filename uses the report date without an invented start time.
- Code revision: The supplied configuration matches `cd1abe18e802073b1fe8de0c8952ec5ff6d170cf` plus the current uncommitted changes in `main.py`: `num_experts = 6` → `num_experts = 16` and `steps = 10000` → `steps = 4000`. The exact revision used for the run was not supplied; configuration details below are reconstructed from the current source where absent from the transcript.
- Environment: CUDA, as reported; Python, PyTorch, CUDA runtime, and GPU details were not supplied for this run.
- Change being tested: Increase the expert pool to sixteen per block while retaining top-2 routing and auxiliary-loss weight 0.01; reduce the training budget to 4,000 steps. Compare with the [six-expert top-2 run](2026-10-10_sparse-moe_6e_top2.md) and [eight-expert top-2 run](2026-10-10_13-13-05_sparse-moe_8e_top2.md) at the same step count.

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | 256 byte values; UTF-8 text, no external tokenizer |
| Dim / heads / head dim | 384 / 6 / 64 |
| Context / batch / steps | 256 / 32 / 4,000 |
| Unique blocks / recurrent passes | 3 / 2; six block applications, shared weights between passes |
| Normalization / feed-forward / position encoding | RMSNorm / sparse SwiGLU MoE, expert hidden width 1,024 / RoPE on Q and K |
| Architecture/runtime comments (main change) | Causal scaled dot-product attention; pre-norm residual blocks; gather/expert/index-add dispatch; no capacity limit or dropped tokens; auxiliary loss averaged over all six block calls |
| MoE (if used): experts / top-k / routing / total and active parameters | 16 experts per block (48 unique total) / top-2 / learned bias-free router, normalized selected probabilities / 58,610,560 total; 9,065,344 active per single traversal, counting two experts per block plus all shared parameters (7,077,888 selected expert parameters + 1,987,456 shared). The recurrent pass can select different experts; two expert evaluations per token per block call. |
| Offloading (if used): components / destination device / policy | None; model and optimizer on CUDA, corpus tensors on CPU, sampled batches copied to CUDA |
| Dataset / train and validation splits / corpus bytes | TinyStories (`roneneldan/TinyStories`); `train[:5%]` / `validation[:1%]`; 93,815,188 training bytes / 187,001 validation bytes; stories joined with newlines |
| Seed / precision | 42 for CPU and CUDA / float32, no AMP, as configured in the current source |
| Optimizer / learning rate / other settings | AdamW; lr `0.0003`; other settings use PyTorch defaults; no scheduler, gradient clipping, or gradient accumulation |
| Validation frequency / samples | Every 500 steps, 20 randomly sampled batches of 32 × 256 positions (163,840 target bytes per validation); LM cross-entropy only; training objective = LM cross-entropy + 0.01 × router auxiliary loss |
| Generation method / prompt | Greedy argmax, last 256 bytes as context, 200 new bytes; prompt "Once upon a time"; reload best LM-validation checkpoint; UTF-8 decode with replacement |

## Results

| Metric | Value |
|---|---|
| Parameter count | 58,610,560 |
| Sampled training bytes | 32,768,000 = 4,000 × 32 × 256, including repeated samples |
| Training time (s; specify what is included) | 1100.73; current timing includes training, periodic validation, diagnostics, and best/final checkpoint writes; excludes dataset loading, model construction, best-checkpoint reload, and generation |
| Peak allocated CUDA memory (MiB) | 3764.0; script prints "MB" but divides allocated bytes by 1024² |
| Throughput (bytes/s) | 29,769 as reported |
| Best validation loss / step | 0.6412 / 4,000; LM cross-entropy without router loss |
| Final validation loss | 0.6412 at step 4,000 |
| Checkpoint locations | `best_model.pt` and `last_model.pt`, both at step 4,000, as reported in the execution directory; no distinct archived locations supplied |

## Observations

| Metric | 6 experts, top-2 | 8 experts, top-2 | 16 experts, top-2 |
|---|---:|---:|---:|
| Total parameters | 23,209,600 | 30,289,792 | 58,610,560 |
| Active feed-forward networks/token/block call | 2 of 6 | 2 of 8 | 2 of 16 |
| Validation LM loss at step 4,000 | 0.6734 | 0.6590 | 0.6412 |
| Total recorded training steps | 10,000 | 10,000 | 4,000 |
| Throughput over each full recorded run (bytes/s) | 35,243 | 35,431 | 29,769 |
| Peak allocated CUDA memory over each full recorded run (MiB) | 3,177.9 | 3,296.9 | 3,764.0 |

At step 4,000, sixteen experts lowered validation LM loss by **0.0322** (4.8%) relative
to six experts and **0.0178** (2.7%) relative to eight experts. The number and size of
selected experts remain the same, so active feed-forward work per token is unchanged;
the larger pool increases stored expert parameters, optimizer state, and router width.
The recorded throughput is lower and peak allocated VRAM is higher than in either earlier run.

This run sampled 40% of the training bytes used by the earlier 10,000-step runs. Its
best/final loss should therefore be interpreted at that shorter budget; the earlier
six- and eight-expert best losses of 0.5778 and 0.5721 were reached after more training.
The transcript provides no evidence of the sixteen-expert result at 10,000 steps.

Validation LM loss improved at every logged checkpoint, from **0.9457** to **0.6412**.
All experts received assignments at every logged checkpoint. Router auxiliary loss
ranged from **1.0086 to 1.0350**, ending at **1.0093**, close to the uniform-routing
value of 1. Final assignment shares ranged from **4.1% to 8.8%**, compared with a
uniform share of 6.25%. These rounded diagnostics describe the last training batch
at each checkpoint, summed over both recurrent passes per unique block; they are
not run-wide utilization statistics.

The run completed 4,000 steps, validated eight times, saved best/final checkpoints,
and reloaded the step-4,000 checkpoint for generation. The supplied generation
sample is preserved exactly, including its ending at `mom i`. This is a single run
with randomly sampled validation batches; changing the expert count changes
initialization RNG consumption and subsequent batch samples. Comparisons use
historical results without repeated-seed variance estimates or a fixed validation
sweep. Checkpoints and environment metadata were not independently inspected.

### Validation curve

| Step | Train LM | Router auxiliary | Validation LM |
|---:|---:|---:|---:|
| 500 | 0.9881 | 1.0350 | 0.9457 |
| 1,000 | 0.8406 | 1.0306 | 0.8103 |
| 1,500 | 0.8215 | 1.0261 | 0.7644 |
| 2,000 | 0.7604 | 1.0165 | 0.7231 |
| 2,500 | 0.7317 | 1.0146 | 0.6977 |
| 3,000 | 0.7488 | 1.0086 | 0.6800 |
| 3,500 | 0.7223 | 1.0107 | 0.6505 |
| 4,000 | 0.6928 | 1.0093 | 0.6412 |

## Full terminal output

```text
python main.py
Device:     cuda
Context:    256
Batch size: 32
Steps:      4000
Validate every x steps: 500
Validate batches:       20
Transform Layers:       3
Target head dim:         64
Dim=384, Heads=6, Recurrent Steps=2
MoE: 16 experts/block, top-k=2, aux weight=0.01

Loading training data...
Training bytes: 93,815,188
Loading validation data...
Validation bytes: 187,001

Parameters: 58,610,560

    500: train_lm=0.9881  router_aux=1.0350  val_lm=0.9457
         expert assignment %: B0=[4.9,7.9,6.1,7.4,5.0,4.6,4.9,8.3,5.2,6.1,8.0,5.1,7.4,7.8,4.2,7.0]  B1=[5.2,6.2,8.1,7.4,7.6,5.4,6.4,6.9,4.8,5.5,5.6,6.1,4.7,7.6,5.9,6.7]  B2=[3.6,5.5,5.1,6.5,5.7,4.5,8.1,6.3,7.7,5.0,6.5,10.7,6.3,6.7,4.7,7.1]
       saved best_model.pt
   1000: train_lm=0.8406  router_aux=1.0306  val_lm=0.8103
         expert assignment %: B0=[5.0,7.4,5.9,7.5,4.3,5.9,6.4,9.2,5.2,5.5,7.5,4.6,6.8,7.9,5.1,5.8]  B1=[5.1,6.2,7.1,7.3,6.7,6.7,5.9,7.7,5.7,5.7,6.1,7.4,4.4,6.7,6.1,5.2]  B2=[3.8,5.2,4.5,5.9,6.4,5.3,8.6,6.4,8.5,4.6,6.8,9.6,6.2,5.4,5.0,7.9]
       saved best_model.pt
   1500: train_lm=0.8215  router_aux=1.0261  val_lm=0.7644
         expert assignment %: B0=[4.5,7.3,5.7,7.6,5.1,5.3,4.8,8.6,6.0,5.8,8.3,5.5,6.6,7.8,5.3,5.9]  B1=[6.0,6.7,7.9,7.4,6.1,5.5,5.9,7.2,5.4,5.2,6.1,6.4,5.1,7.5,6.5,5.2]  B2=[4.0,5.2,5.1,6.3,7.1,4.1,8.2,6.3,8.6,4.8,6.5,9.6,6.6,5.7,4.4,7.7]
       saved best_model.pt
   2000: train_lm=0.7604  router_aux=1.0165  val_lm=0.7231
         expert assignment %: B0=[4.6,7.8,5.9,7.8,5.2,4.8,5.7,9.0,5.3,5.9,6.9,5.0,6.9,7.9,5.5,5.8]  B1=[5.5,5.8,7.2,6.7,7.5,4.9,6.4,6.3,6.0,6.1,5.4,5.9,6.0,7.2,6.2,6.7]  B2=[4.5,5.0,5.2,5.9,6.4,4.1,7.4,5.8,8.2,5.2,7.0,8.7,6.3,6.3,5.4,8.8]
       saved best_model.pt
   2500: train_lm=0.7317  router_aux=1.0146  val_lm=0.6977
         expert assignment %: B0=[4.8,7.7,6.4,7.4,5.5,5.4,5.0,8.3,5.6,5.7,7.2,5.1,6.7,7.8,5.5,6.1]  B1=[5.6,6.3,7.6,7.3,6.2,6.2,6.0,7.4,5.1,5.9,5.1,6.2,5.4,7.8,6.0,5.8]  B2=[5.6,5.0,5.1,6.3,5.8,4.4,8.0,5.7,8.4,5.3,7.5,8.9,6.0,5.9,4.4,7.8]
       saved best_model.pt
   3000: train_lm=0.7488  router_aux=1.0086  val_lm=0.6800
         expert assignment %: B0=[4.8,7.5,6.2,7.7,5.0,5.0,5.3,8.0,6.3,5.8,7.0,5.1,6.6,8.1,5.8,5.8]  B1=[5.0,7.2,6.5,6.8,6.1,6.4,6.2,6.3,5.8,5.9,5.6,6.2,5.4,7.5,6.2,7.0]  B2=[5.0,5.1,5.6,6.0,6.4,5.2,8.4,7.1,7.9,4.7,6.3,8.5,5.6,6.0,4.9,7.2]
       saved best_model.pt
   3500: train_lm=0.7223  router_aux=1.0107  val_lm=0.6505
         expert assignment %: B0=[5.2,7.4,5.7,7.6,5.6,5.3,5.5,7.9,5.0,5.9,6.4,5.3,7.2,8.4,5.2,6.5]  B1=[6.0,6.4,6.7,6.4,6.7,5.9,6.5,6.6,5.8,5.9,5.3,5.9,5.5,7.4,6.6,6.3]  B2=[5.0,5.1,5.4,6.0,6.4,4.6,7.7,6.4,8.5,5.4,6.9,8.7,5.9,6.0,5.1,7.0]
       saved best_model.pt
   4000: train_lm=0.6928  router_aux=1.0093  val_lm=0.6412
         expert assignment %: B0=[4.7,8.5,5.9,7.1,5.1,5.0,4.8,8.8,5.9,5.5,6.8,5.4,6.6,7.8,5.9,6.2]  B1=[6.7,6.2,5.6,7.5,6.5,5.7,6.2,5.7,5.4,6.1,7.2,5.1,6.3,7.3,6.5,5.9]  B2=[5.5,5.4,5.6,6.6,6.8,4.9,7.3,5.4,8.1,6.2,6.9,7.9,6.6,5.2,4.1,7.6]
       saved best_model.pt
Saved last_model.pt

------------------------------
Results
------------------------------
Parameters:      58,610,560
Experts/token:   2 of 16 per block call
Training bytes:  32,768,000
Training time:   1100.73s
Peak VRAM:       3764.0 MB
Bytes/sec:       29,769
Best val LM loss: 0.6412
Using checkpoint from step 4000 (val=0.6412)

------------------------------
Generation
------------------------------

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, Lily went to the park with her mom. She saw a big box of candy and wanted to play with it. She asked her mom i
```
