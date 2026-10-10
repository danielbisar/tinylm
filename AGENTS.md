# Repository Guidelines

## Project Structure & Module Organization

TinyLM is a byte-level PyTorch language model with shared transformer blocks and sparse MoE layers.

- `tinylm.py`: model architecture, attention, RoPE, SwiGLU experts, and routing.
- `core.py`: editable `Config`, byte tokenizer, checkpoint storage, timing, and logging.
- `train.py` / `run.py`: training and checkpoint-based generation entry points.
- `tests/`: MoE correctness and workflow regression tests.
- `docs/runs/`: automatically generated invocation logs, curated experiment reports, and `TEMPLATE.md`; `docs/research/`: research references.
- `model/` and `artifacts/`: ignored checkpoints and experiment artifacts.

## Build, Test, and Development Commands

Run commands from the repository root. No separate build step is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy datasets
python -m unittest discover -s tests -v
python train.py --model-prefix=experiment-
python run.py --model-prefix=experiment-
python run.py --model-prefix=experiment- --checkpoint=last
```

Use PyTorch with `torch.nn.RMSNorm` support. Training downloads TinyStories on first use and selects CUDA when available, otherwise CPU. Generation loads the best checkpoint by default. Change model and run settings in `core.py`'s `Config`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Follow existing multiline calls with trailing commas. Keep architecture in `tinylm.py` and shared workflow utilities in `core.py`. No formatter or linter is configured; match neighboring code and avoid unrelated reformatting.

## Testing Guidelines

Use standard-library `unittest`, naming files `test_*.py` and methods `test_*`. No numeric coverage threshold is configured. Add regression checks for changed behavior, especially routing outputs and gradients, recurrent auxiliary-loss averaging, checkpoint restoration, overwrite protection, and logging. Use small configurations, synthetic tensors, and temporary directories to keep tests independent of dataset downloads and full training runs. MoE tests select CUDA when available; workflow integration tests exercise CPU.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `Use RoPE` and `8 experts, top-2`, without a mandatory prefix convention. Describe the concrete change and include experiment settings when relevant. PRs should explain the purpose, tests run, and related issues if applicable. For benchmark changes, use `docs/runs/TEMPLATE.md`, record configuration and environment, preserve terminal output, and update the README comparison table.

## Checkpoints & Local Configuration

Use unique model prefixes: training refuses existing checkpoint destinations unless `--force-overwrite` is supplied. Keep `.env`, `.venv/`, `model/`, and `artifacts/` out of commits.
