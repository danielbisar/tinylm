# Run: <experiment name>

- Run started: <YYYY-MM-DD HH:MM:SS and timezone>
- Code revision: <full commit; describe any uncommitted changes>
- Implementation/metric version: <EOS-aware or historical; text/sequence BPB semantics>
- Provenance: <copy automatic metadata; source hashes; git dirty state>
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
| Dataset requested immutable revision / available version / split fingerprints | |
| Seed / precision | |
| Training/evaluation RNG seeds / fixed evaluation windows | |
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
| Validation text BPB / sequence BPB / EOS count, mean NLL, accuracy | |
| Generated tokens / decoded UTF-8 bytes / tokens/s / bytes/s | |
| Checkpoint selection metric / tokenizer state | |
| Checkpoint locations | |

## Observations

<Comparison with baseline, limitations, and conclusions.>

Keep historical and current semantics separate. State unavailable metadata/memory
measurements explicitly; do not infer provenance from timestamps. Publishable runs
should use committed source and configuration. A dirty flag/hash does not preserve
the source diff. CPU VRAM=0 does not measure CPU memory. Review transcripts for
private prompts, third-party messages, and filesystem paths before publication.

## Full terminal output

```text
<paste complete output, including validation curve and generation>
```
