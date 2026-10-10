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
| Context (tokens; approximate source bytes) / batch / steps | |
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
| Training/validation raw UTF-8 bytes / tokens / average bytes per token | |
| Sampled training tokens / estimated source bytes | |
| Training time (s; specify what is included) | |
| Peak allocated CUDA memory (MiB) | |
| Throughput (tokens/s; estimated source bytes/s) | |
| Best periodic validation BPB / token loss / step | |
| Final validation BPB / token loss | |
| Generated tokens / decoded UTF-8 bytes / tokens/s / bytes/s | |
| Checkpoint selection metric / tokenizer state | |
| Checkpoint locations | |

## Observations

<Comparison with baseline, limitations, and conclusions.>

## Full terminal output

```text
<paste complete output, including validation curve and generation>
```
