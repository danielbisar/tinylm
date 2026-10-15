# Repository Guidelines

## Project Structure & Module Organization

TinyLM is an educational PyTorch language-model platform with shared transformer blocks,
sparse MoE layers, and byte or locally trained BPE tokenization. Both tokenizers reserve
EOS, and training appends EOS after each separate document.

- `tinylm.py`: model architecture, attention, RoPE, SwiGLU experts, and routing.
- `core.py`: editable `Config`, byte/BPE tokenizers, checkpoint storage, timing, provenance, and logging.
- `train.py` / `run.py`: training and checkpoint-based generation entry points.
- `tests/`: MoE correctness and workflow regression tests.
- `experiments/cpu_recurrence.py`: bounded offline recurrence comparison with synthetic documents.
- `docs/runs/`: invocation logs, curated reports, an index, `HISTORY.md`, and `TEMPLATE.md`; `docs/research/`: research references.
- `model/` and `artifacts/`: ignored checkpoints and experiment artifacts.

## Build, Test, and Development Commands

Run commands from the repository root. No separate build step is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" python -m experiments.cpu_recurrence
python train.py --model-prefix=experiment-
python run.py --model-prefix=experiment-
python run.py --model-prefix=experiment- --checkpoint=last
```

The tested environment is Linux/Python 3.14 with the pinned direct dependencies.
Install a compatible PyTorch CUDA build instead of the CPU wheel for local GPU work.
Training downloads TinyStories on first use and selects CUDA when available, otherwise CPU.
Generation restores the best checkpoint by default and stops on EOS or the output-byte budget.
Change model/run settings in `core.py`'s `Config`; effective steps derive from source bytes/token.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Follow existing multiline calls with trailing commas. Keep architecture in `tinylm.py` and shared workflow utilities in `core.py`. No formatter or linter is configured; match neighboring code and avoid unrelated reformatting.

## Testing Guidelines

Use standard-library `unittest`, naming files `test_*.py` and methods `test_*`. No numeric coverage threshold is configured. Add regression checks for changed behavior, especially routing outputs and gradients, recurrent auxiliary-loss averaging, checkpoint restoration, overwrite protection, and logging. Use small configurations, synthetic tensors, and temporary directories to keep tests independent of dataset downloads and full training runs. MoE tests select CUDA when available; workflow integration tests exercise CPU.

CPU CI runs the full suite offline. Training/evaluation use independent RNG streams;
evaluation resets fixed windows at seed + 1, with the same prefix in final evaluation.
Token loss and sequence BPB include EOS NLL; text BPB excludes EOS and selects the best
checkpoint. EOS contributes zero source bytes. Preserve historical/pre-EOS results and
their old metric semantics; do not silently reinterpret them as current measurements.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `Use RoPE` and `8 experts, top-2`, without a mandatory prefix convention. Describe the concrete change and include experiment settings when relevant. PRs should explain the purpose, tests run, and related issues if applicable.

For benchmark changes, use `docs/runs/TEMPLATE.md`, commit source/configuration first,
record provenance and environment, preserve terminal output, and update the experiment
index. Do not add current measurements to the historical/pre-EOS comparison table.

## Checkpoints & Local Configuration

Use unique model prefixes: training refuses existing checkpoint destinations unless `--force-overwrite` is supplied. Keep `.env`, `.venv/`, `model/`, and `artifacts/` out of commits.
