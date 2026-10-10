# Run: <experiment name>

- Run started: <YYYY-MM-DD HH:MM:SS and timezone>
- Code revision: <full commit; describe any uncommitted changes>
- Environment: <Python, PyTorch, CUDA, GPU and VRAM>
- Change being tested: <change and baseline report link>

## Configuration

| Setting | Value |
|---|---|
| Vocabulary / tokenizer | |
| Dim / heads / head dim | |
| Context / batch / steps | |
| Unique blocks / recurrent passes | |
| Normalization / feed-forward / position encoding | |
| Architecture/runtime comments (main change) | |
| MoE (if used): experts / top-k / routing / total and active parameters | |
| Offloading (if used): components / destination device / policy | |
| Dataset / train and validation splits / corpus bytes | |
| Seed / precision | |
| Optimizer / learning rate / other settings | |
| Validation frequency / samples | |
| Generation method / prompt | |

## Results

| Metric | Value |
|---|---|
| Parameter count | |
| Sampled training bytes | |
| Training time (s; specify what is included) | |
| Peak allocated CUDA memory (MiB) | |
| Throughput (bytes/s) | |
| Best validation loss / step | |
| Final validation loss | |
| Checkpoint locations | |

## Observations

<Comparison with baseline, limitations, and conclusions.>

## Full terminal output

```text
<paste complete output, including validation curve and generation>
```
