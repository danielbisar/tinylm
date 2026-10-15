# Run: Sparse MoE, sixteen SwiGLU experts with top-2 routing, 10,000 steps

> **Historical/pre-EOS experiment.** Tokenization, metrics, sampling, and
> timing follow this report's recorded implementation, not current code.
> Conversational recommendations are historical advice, not verified conclusions.

[Run overview](HISTORY.md#run-overview)

- Report recorded: 2026-10-10; actual run start time and timezone were not supplied. The filename uses the report date without an invented start time.
- Code revision: The supplied configuration and generation prompts match `d731393b9118f9b31a2f98afedc38dec97ab5e2c` plus the current uncommitted changes in `main.py`: `steps = 4000` → `steps = 10000`, remove the default generation length, and generate 1,000 new bytes for each of five prompts instead of 200 bytes for one prompt. The exact revision used for the run was not supplied; configuration details below are reconstructed from the current source where absent from the transcript.
- Environment: CUDA, as reported; Python, PyTorch, CUDA runtime, and GPU details were not supplied for this run.
- Change being tested: Train the sixteen-expert top-2 configuration for 10,000 steps and inspect longer generation across five prompts. Compare with the earlier [4,000-step sixteen-expert run](2026-10-10_sparse-moe_16e_top2.md), [six-expert top-2 run](2026-10-10_sparse-moe_6e_top2.md), and [eight-expert top-2 run](2026-10-10_13-13-05_sparse-moe_8e_top2.md).

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | 256 byte values; UTF-8 text, no external tokenizer |
| Dim / heads / head dim | 384 / 6 / 64 |
| Context / batch / steps | 256 / 32 / 10,000 |
| Unique blocks / recurrent passes | 3 / 2; six block applications, shared weights between passes |
| Normalization / feed-forward / position encoding | RMSNorm / sparse SwiGLU MoE, expert hidden width 1,024 / RoPE on Q and K |
| Architecture/runtime comments (main change) | Causal scaled dot-product attention; pre-norm residual blocks; gather/expert/index-add dispatch; no capacity limit or dropped tokens; auxiliary loss averaged over all six block calls |
| MoE (if used): experts / top-k / routing / total and active parameters | 16 experts per block (48 unique total) / top-2 / learned bias-free router, normalized selected probabilities / 58,610,560 total; 9,065,344 active per single traversal, counting two experts per block plus all shared parameters (7,077,888 selected expert parameters + 1,987,456 shared). The recurrent pass can select different experts; two expert evaluations per token per block call. |
| Offloading (if used): components / destination device / policy | None; model and optimizer on CUDA, corpus tensors on CPU, sampled batches copied to CUDA |
| Dataset / train and validation splits / corpus bytes | TinyStories (`roneneldan/TinyStories`); `train[:5%]` / `validation[:1%]`; 93,815,188 training bytes / 187,001 validation bytes; stories joined with newlines |
| Seed / precision | 42 for CPU and CUDA / float32, no AMP, as configured in the current source |
| Optimizer / learning rate / other settings | AdamW; lr `0.0003`; other settings use PyTorch defaults; no scheduler, gradient clipping, or gradient accumulation |
| Validation frequency / samples | Every 500 steps, 20 randomly sampled batches of 32 × 256 positions (163,840 target bytes per validation); LM cross-entropy only; training objective = LM cross-entropy + 0.01 × router auxiliary loss |
| Generation method / prompt | Greedy argmax, last 256 bytes as context, 1,000 new bytes per prompt; prompts "Once upon a time", "Lily went to the park", "Tom found a strange box", "The dog was afraid because", and "One morning, Sarah discovered"; reload best LM-validation checkpoint; UTF-8 decode with replacement |

## Results

| Metric | Value |
|---|---|
| Parameter count | 58,610,560 |
| Sampled training bytes | 81,920,000 = 10,000 × 32 × 256, including repeated samples |
| Training time (s; specify what is included) | 2726.54; current timing includes training, periodic validation, diagnostics, and best/final checkpoint writes; excludes dataset loading, model construction, best-checkpoint reload, and generation |
| Peak allocated CUDA memory (MiB) | 3762.7; script prints "MB" but divides allocated bytes by 1024² |
| Throughput (bytes/s) | 30,045 as reported |
| Best validation loss / step | 0.5583 / 10,000; LM cross-entropy without router loss |
| Final validation loss | 0.5583 at step 10,000 |
| Checkpoint locations | `best_model.pt` and `last_model.pt`, both at step 10,000, as reported in the execution directory; no distinct archived locations supplied |

## Observations

| Metric | 6 experts, top-2 | 8 experts, top-2 | 16 experts, top-2 (this run) |
|---|---:|---:|---:|
| Total parameters | 23,209,600 | 30,289,792 | 58,610,560 |
| Active feed-forward networks/token/block call | 2 of 6 | 2 of 8 | 2 of 16 |
| Training steps | 10,000 | 10,000 | 10,000 |
| Best validation LM loss | 0.5778 | 0.5721 | 0.5583 |
| Final validation LM loss | 0.5778 | 0.5758 | 0.5583 |
| Training time (s) | 2324.46 | 2312.09 | 2726.54 |
| Throughput (bytes/s) | 35,243 | 35,431 | 30,045 |
| Peak allocated CUDA memory (MiB) | 3,177.9 | 3,296.9 | 3,762.7 |

At the same 10,000-step budget, sixteen experts achieved a best validation LM loss
**0.0195 lower (3.4%)** than six experts and **0.0138 lower (2.4%)** than eight
experts. Relative to eight experts, training took **17.9% longer** and peak allocated
VRAM was **14.1% higher**. Two full-size experts remain selected per token per block
call; the larger pool increases stored parameters, optimizer state, and router width.

This run's loss at step 4,000 was **0.6409**, close to the earlier sixteen-expert
4,000-step report's **0.6412**. By step 10,000 it reached **0.5583**. These are
separate supplied runs; the transcript does not show a checkpoint resume from the
earlier run.

Validation LM loss improved at every logged checkpoint, from **0.9519** to
**0.5583**. Router auxiliary loss ranged from **1.0054 to 1.0382**, ending at
**1.0054**, close to the uniform-routing value of 1. All experts received assignments
at every logged checkpoint; final assignment shares ranged from **4.7% to 8.3%**,
compared with a uniform share of 6.25%. These rounded diagnostics describe the
last training batch at each checkpoint, summed over both recurrent passes per
unique block; they are not run-wide utilization statistics.

All five greedy generation samples exhibit repetition. The first alternates nearly
identical dialogue between Lily and the boy; the coin sample repeats "We wanted to
see the coins"; the final sample repeatedly describes excitement and climbing a
tree. The park and dog samples also lose narrative consistency. The lower validation
loss therefore accompanies visible repetition in these longer outputs. All supplied
text, line breaks, and truncated endings are preserved below.

The run completed 10,000 steps, validated twenty times, saved best/final checkpoints,
and reloaded the step-10,000 checkpoint for generation. This is a single run with
randomly sampled validation batches. Changing the expert count changes initialization
RNG consumption and subsequent batch samples. Comparisons use historical results
without repeated-seed variance estimates or a fixed validation sweep. Checkpoints
and environment metadata were not independently inspected.

### Validation curve

| Step | Train LM | Router auxiliary | Validation LM |
|---:|---:|---:|---:|
| 500 | 0.9950 | 1.0382 | 0.9519 |
| 1,000 | 0.8439 | 1.0288 | 0.8103 |
| 1,500 | 0.8197 | 1.0302 | 0.7617 |
| 2,000 | 0.7533 | 1.0215 | 0.7235 |
| 2,500 | 0.7303 | 1.0185 | 0.6960 |
| 3,000 | 0.7463 | 1.0109 | 0.6811 |
| 3,500 | 0.7186 | 1.0211 | 0.6513 |
| 4,000 | 0.6988 | 1.0146 | 0.6409 |
| 4,500 | 0.6859 | 1.0139 | 0.6348 |
| 5,000 | 0.6775 | 1.0150 | 0.6260 |
| 5,500 | 0.6474 | 1.0122 | 0.6134 |
| 6,000 | 0.6262 | 1.0143 | 0.6073 |
| 6,500 | 0.6270 | 1.0128 | 0.6046 |
| 7,000 | 0.6172 | 1.0116 | 0.5915 |
| 7,500 | 0.6162 | 1.0100 | 0.5834 |
| 8,000 | 0.6603 | 1.0133 | 0.5788 |
| 8,500 | 0.6145 | 1.0069 | 0.5784 |
| 9,000 | 0.6290 | 1.0065 | 0.5662 |
| 9,500 | 0.6106 | 1.0099 | 0.5605 |
| 10,000 | 0.6056 | 1.0054 | 0.5583 |

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
MoE: 16 experts/block, top-k=2, aux weight=0.01

Loading training data...
Training bytes: 93,815,188
Loading validation data...
Validation bytes: 187,001

Parameters: 58,610,560

    500: train_lm=0.9950  router_aux=1.0382  val_lm=0.9519
         expert assignment %: B0=[4.9,4.1,5.8,7.1,4.6,5.2,4.9,8.0,6.3,4.8,11.1,5.9,7.1,8.4,6.1,5.7]  B1=[5.4,5.0,6.6,7.5,8.9,6.6,5.8,5.9,4.4,6.2,6.5,4.6,5.6,7.7,8.1,5.2]  B2=[3.9,6.7,4.7,6.7,5.3,5.3,6.0,5.5,8.3,6.1,3.5,7.4,6.6,8.0,4.4,11.7]
       saved best_model.pt
   1000: train_lm=0.8439  router_aux=1.0288  val_lm=0.8103
         expert assignment %: B0=[4.9,4.7,6.4,7.9,4.4,5.7,5.7,8.0,6.1,4.7,9.7,5.5,6.7,8.5,5.8,5.3]  B1=[5.2,5.4,7.2,7.8,7.7,7.2,5.7,6.3,5.1,5.6,8.1,4.5,5.0,7.1,6.9,5.3]  B2=[4.2,5.8,4.8,6.5,5.9,6.2,8.0,6.0,8.7,4.8,5.5,7.5,5.7,5.9,4.9,9.6]
       saved best_model.pt
   1500: train_lm=0.8197  router_aux=1.0302  val_lm=0.7617
         expert assignment %: B0=[4.5,5.2,6.4,7.6,4.7,5.1,5.4,6.9,7.0,5.5,9.6,6.2,6.6,8.4,6.3,4.7]  B1=[5.6,6.1,7.2,7.3,7.3,7.2,5.7,6.2,4.1,5.0,5.7,5.5,5.5,8.2,7.0,6.4]  B2=[4.4,5.3,4.2,6.7,7.4,4.9,7.4,5.4,8.6,5.8,4.7,7.9,6.2,5.9,5.1,10.1]
       saved best_model.pt
   2000: train_lm=0.7533  router_aux=1.0215  val_lm=0.7235
         expert assignment %: B0=[5.1,4.7,6.3,8.4,4.8,4.9,5.3,8.2,6.5,4.7,9.5,5.4,6.6,8.3,5.8,5.5]  B1=[5.1,5.9,7.0,8.2,7.8,6.5,5.7,5.8,4.5,6.4,5.4,5.0,5.6,7.3,7.1,6.6]  B2=[4.5,4.9,5.9,6.7,6.6,5.7,7.0,5.5,8.7,5.7,5.5,6.9,6.5,5.6,5.2,9.2]
       saved best_model.pt
   2500: train_lm=0.7303  router_aux=1.0185  val_lm=0.6960
         expert assignment %: B0=[4.4,4.7,6.1,7.9,5.3,4.6,5.9,8.1,5.5,5.3,9.5,5.7,6.5,8.4,6.3,5.7]  B1=[5.6,6.3,7.1,7.9,7.1,6.9,6.1,5.8,5.7,6.0,5.4,4.4,5.0,7.3,7.5,5.8]  B2=[5.1,6.7,5.7,5.9,6.7,5.8,7.1,4.8,8.1,5.3,4.3,7.4,5.8,6.5,5.4,9.4]
       saved best_model.pt
   3000: train_lm=0.7463  router_aux=1.0109  val_lm=0.6811
         expert assignment %: B0=[5.2,4.7,6.3,8.1,5.4,4.4,5.6,8.1,5.7,5.1,9.3,5.3,7.3,8.5,5.2,5.8]  B1=[5.4,6.5,6.8,6.5,8.0,6.7,6.1,5.4,5.0,6.1,5.4,5.5,5.9,7.8,6.7,6.1]  B2=[4.9,6.1,5.5,6.4,6.7,5.9,6.1,5.4,8.3,4.9,5.4,7.6,5.4,6.7,5.9,8.8]
       saved best_model.pt
   3500: train_lm=0.7186  router_aux=1.0211  val_lm=0.6513
         expert assignment %: B0=[5.2,4.4,6.6,7.7,5.1,6.1,5.1,7.6,6.0,5.0,9.2,5.0,6.7,8.6,5.8,6.0]  B1=[5.2,6.1,6.8,6.4,7.6,6.7,6.6,5.5,6.4,6.3,6.2,4.5,6.0,7.1,7.2,5.5]  B2=[5.3,5.6,5.2,6.2,7.5,6.2,6.4,5.9,8.9,5.4,4.5,7.2,5.8,6.2,4.8,8.8]
       saved best_model.pt
   4000: train_lm=0.6988  router_aux=1.0146  val_lm=0.6409
         expert assignment %: B0=[4.9,5.6,6.3,8.2,4.7,6.0,6.1,5.5,5.9,5.2,9.2,5.6,6.6,8.7,6.5,5.1]  B1=[5.7,5.4,7.0,7.3,7.3,6.4,5.9,5.4,5.4,6.3,6.5,5.7,5.4,7.3,7.1,5.8]  B2=[4.9,5.7,6.3,6.9,7.0,6.1,6.8,5.1,8.0,5.6,5.4,7.8,5.5,5.1,5.5,8.2]
       saved best_model.pt
   4500: train_lm=0.6859  router_aux=1.0139  val_lm=0.6348
         expert assignment %: B0=[6.3,4.7,6.2,8.3,4.8,5.6,6.5,6.7,6.1,5.1,8.7,5.9,6.8,8.0,5.6,4.8]  B1=[5.1,6.4,6.4,6.9,6.5,6.3,6.2,5.8,5.4,6.3,6.1,5.4,5.9,7.8,7.3,6.0]  B2=[5.7,6.0,4.8,6.4,6.8,5.6,7.3,4.9,8.2,5.0,5.5,6.7,6.1,5.7,5.7,9.5]
       saved best_model.pt
   5000: train_lm=0.6775  router_aux=1.0150  val_lm=0.6260
         expert assignment %: B0=[4.7,4.9,6.5,8.2,7.0,5.6,5.8,6.1,5.3,4.8,9.2,5.3,6.4,7.9,5.9,6.3]  B1=[5.3,5.3,7.0,6.9,7.8,7.4,6.0,5.6,5.0,7.0,5.5,5.0,5.6,7.3,7.3,6.0]  B2=[5.5,6.1,5.9,5.9,6.6,5.2,7.2,5.6,8.5,5.6,4.6,6.7,5.5,6.4,6.0,8.9]
       saved best_model.pt
   5500: train_lm=0.6474  router_aux=1.0122  val_lm=0.6134
         expert assignment %: B0=[4.7,5.8,6.6,8.1,5.8,5.6,5.5,6.5,5.7,4.3,9.6,5.4,6.5,8.2,6.2,5.4]  B1=[5.0,5.7,6.1,6.4,7.9,6.8,5.5,5.0,5.5,6.0,6.5,5.4,7.5,7.0,7.3,6.4]  B2=[6.1,6.2,5.9,6.6,6.6,4.9,7.2,5.1,8.6,5.2,4.8,7.1,5.7,5.6,5.7,8.7]
       saved best_model.pt
   6000: train_lm=0.6262  router_aux=1.0143  val_lm=0.6073
         expert assignment %: B0=[4.6,5.5,6.8,8.2,6.1,4.6,5.7,5.7,5.5,4.6,9.2,5.8,7.1,7.7,6.6,6.3]  B1=[5.4,5.6,6.5,7.4,6.9,7.6,6.2,6.0,5.5,5.7,5.1,5.5,5.5,7.9,7.1,6.0]  B2=[4.8,7.0,5.5,6.3,7.0,5.2,6.1,5.6,8.6,5.1,5.5,7.8,5.8,5.7,5.7,8.2]
       saved best_model.pt
   6500: train_lm=0.6270  router_aux=1.0128  val_lm=0.6046
         expert assignment %: B0=[4.7,6.2,6.9,8.2,5.0,5.3,5.9,6.0,5.9,5.3,9.3,5.2,6.7,8.7,5.3,5.5]  B1=[5.7,5.3,6.0,7.3,6.8,6.8,6.4,5.8,5.4,6.5,5.5,6.0,6.2,7.3,6.5,6.4]  B2=[5.1,6.0,6.1,6.0,7.1,5.3,7.9,5.0,8.2,6.2,4.9,6.3,6.2,5.7,5.2,8.7]
       saved best_model.pt
   7000: train_lm=0.6172  router_aux=1.0116  val_lm=0.5915
         expert assignment %: B0=[5.8,6.1,5.8,8.0,5.2,5.3,5.6,6.5,6.4,4.8,8.8,5.6,6.6,8.1,6.1,5.4]  B1=[5.5,6.1,6.9,6.8,7.5,6.8,5.4,5.9,4.9,6.0,6.2,6.4,5.8,7.2,6.9,5.8]  B2=[7.1,5.6,5.8,6.2,7.1,5.4,6.2,5.4,8.0,5.1,6.1,6.5,5.8,5.4,5.0,9.3]
       saved best_model.pt
   7500: train_lm=0.6162  router_aux=1.0100  val_lm=0.5834
         expert assignment %: B0=[4.9,5.7,6.4,8.2,5.5,5.2,5.5,6.3,5.8,5.0,8.9,5.5,6.5,8.4,5.8,6.4]  B1=[5.1,5.6,5.9,7.6,6.7,7.0,5.9,5.5,6.1,6.7,6.1,5.6,5.2,7.7,7.3,6.0]  B2=[6.1,6.3,5.4,6.4,6.6,6.1,6.9,5.2,7.9,4.9,5.8,7.2,5.7,5.4,5.5,8.7]
       saved best_model.pt
   8000: train_lm=0.6603  router_aux=1.0133  val_lm=0.5788
         expert assignment %: B0=[4.7,4.4,6.2,8.0,6.7,5.5,5.3,7.0,6.1,5.5,8.9,5.4,6.3,8.2,6.5,5.2]  B1=[5.9,5.9,6.4,7.3,7.6,5.9,5.5,5.8,5.2,6.1,5.5,5.3,6.9,7.2,7.8,5.8]  B2=[7.1,5.5,5.1,6.4,7.6,5.6,7.0,5.1,8.3,5.6,4.4,7.1,6.0,5.5,4.8,8.9]
       saved best_model.pt
   8500: train_lm=0.6145  router_aux=1.0069  val_lm=0.5784
         expert assignment %: B0=[5.2,5.3,6.0,7.9,5.2,6.2,6.0,5.9,6.7,4.8,8.2,5.4,6.7,8.1,6.6,5.7]  B1=[5.7,5.3,6.4,6.6,7.6,6.6,5.9,5.2,5.8,6.1,6.1,5.6,5.9,7.7,7.3,6.3]  B2=[6.0,6.3,6.1,6.7,7.4,4.9,5.7,5.7,8.0,4.9,5.0,6.7,6.4,5.8,5.8,8.6]
       saved best_model.pt
   9000: train_lm=0.6290  router_aux=1.0065  val_lm=0.5662
         expert assignment %: B0=[5.3,5.3,6.9,8.1,6.7,5.6,5.9,5.1,6.4,4.5,8.7,5.8,6.9,7.8,5.7,5.2]  B1=[5.8,6.3,6.9,6.2,7.3,6.3,5.8,5.9,5.4,6.5,6.3,5.4,5.4,7.2,7.3,6.1]  B2=[6.2,5.7,6.1,6.0,6.4,5.5,6.7,5.6,7.7,5.1,5.7,7.4,6.0,5.1,5.8,9.0]
       saved best_model.pt
   9500: train_lm=0.6106  router_aux=1.0099  val_lm=0.5605
         expert assignment %: B0=[5.7,5.9,6.6,7.9,5.3,6.0,5.7,5.0,6.4,5.4,9.1,5.7,6.5,8.3,5.7,5.0]  B1=[6.2,5.7,7.4,6.1,7.3,7.0,6.3,5.7,5.5,6.3,5.4,5.6,5.7,7.1,7.2,5.6]  B2=[6.5,5.6,6.0,6.3,7.0,5.3,7.0,5.6,8.2,5.0,5.1,6.5,5.8,5.6,5.9,8.7]
       saved best_model.pt
  10000: train_lm=0.6056  router_aux=1.0054  val_lm=0.5583
         expert assignment %: B0=[5.5,5.8,6.1,7.6,6.8,5.7,5.9,4.7,6.8,4.9,8.1,5.5,6.5,8.3,5.9,5.8]  B1=[5.1,6.3,6.7,6.6,6.9,6.8,6.6,5.7,5.2,6.5,5.8,6.4,5.5,7.1,6.8,6.1]  B2=[6.2,5.3,5.9,6.6,6.5,5.7,6.2,6.1,7.7,5.3,5.8,7.2,5.8,6.0,5.6,8.2]
       saved best_model.pt
Saved last_model.pt

------------------------------
Results
------------------------------
Parameters:      58,610,560
Experts/token:   2 of 16 per block call
Training bytes:  81,920,000
Training time:   2726.54s
Peak VRAM:       3762.7 MB
Bytes/sec:       30,045
Best val LM loss: 0.5583
Using checkpoint from step 10000 (val=0.5583)

------------------------------
Generation
------------------------------

Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she went to the park to play with her friends. They were having fun until they heard a loud noise.

Lily was scared and ran to her friends. She saw the big boy and said, "What are you doing, Lily?"

The boy replied, "I am trying to get the ball fixed."

Lily said, "I want to play with the ball."

The boy said, "No, I don't want to play with you."

Lily said, "No, I don't want to play with you. I want to play with you."

The boy said, "No, I don't want to play with you. I want to play with you."

Lily said, "No, I don't want to play with you. I want to play with you."

The boy said, "No, I don't want to play with you. I want to play with you."

Lily said, "No, I don't want to play with you. I want to play with you."

The boy said, "No, I don't want to play with you. I want to play with you."

Lily said, "No, I don't want to play with you. I want to play with you."

The boy
------------------------------
Lily went to the park to play with her friends. They played together and had fun. Lily was happy that her friends were her friends. They learned that they should not play with the toys anymore.
Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, she went to the park to play with her friends. They were having fun and were having fun.

As they were playing, Lily saw a big tree with lots of leaves. She wanted to climb it, but her mommy said no. Lily was sad because she wanted to play with her friends and her friends. So, Lily decided to play with her friends and take turns to play with
the tree.

As they were playing, Lily saw a big tree with a big branch on it. She wanted to climb it, but her mom said no. Lily was sad because she wanted to play with her toys and rest them on the tree. But her mom said they had to wait until they got home. Lily was sad because she loved her toy and was sad.

Lily felt bad and said she was sorry and wanted to h
------------------------------
Tom found a strange box of coins. He picked up the coins and put them in his pocket. He put them in his pocket and said, "This is my coin. It is mine. I want to see the coins."

Tom and Lily looked at each other and said, "We are sorry, Mom. We were just playing. We wanted to see the coins. We did not know. We wanted to see the coins. We wanted to see the coins. We did not know. We wanted to see the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did not know the coins. We wanted to see the coins. We did
not know the coins. We wanted to see the coins. We did not know the coins. We wante
------------------------------
The dog was afraid because he was so scared. He did not know what to do. He wanted to go back to the park. He wanted to go back to the park. He ran and ran and ran and ran around. He saw a big tree with many branches. He wanted to climb it, but he was too small. He tried to reach the tree, but it was too high. He tried and tried, but he could not reach the top. He was sad and scared. He wanted to get out of the tree.

Then, he heard a voice. It was his mom. She said, "Mom, I want to go to the tree. It is not safe to go there. It is not far away. It is a bad day. It is a bad day. It is not a bad day. It is a bad day. It is a bad day. It is not a bad day. It is a bad day.
Tom and Lily were playing in the park. They liked to run, jump and slide. They saw a big tree with many branches. They wanted to climb the tree and see what was on the tree.

"Let's go to the tree!" Tom said. He reached out his hand to touch the tree. But the tree was too high. The tree was too high up to the top.

"Oh no, our tree!" Tom said. "
------------------------------
One morning, Sarah discovered a big pile of shiny rocks. She was so excited to see what it was and she ran to it.

When she got to the top of the sky, she saw a beautiful light in the sky. She was so excited to see what it was and she ran to it.

When she got to the top of the hill, she saw a big tree. She was so excited to see what it was. She ran to it and started to climb it. She was so excited to see what it was and she ran to the top of the tree.

When she got to the top, she saw a big tree. She was so excited to see what it was. She ran to it and started to climb it. She was so excited to see what was inside.

The little girl was so excited to see what was inside. She ran to the tree and started to climb it. She was so excited to see what was inside.

The little girl was so excited to see what was inside. She ran to the tree and started to climb the tree. She was so excited to see what was inside.

The little girl was so excited to see what was inside. She ran to the tree and started to climb the tree.
------------------------------
```
