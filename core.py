"""Shared configuration, tokenization, checkpoints, and run reporting."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shlex
import sys
import tempfile
import time
import traceback

import torch

from tinylm import TinyLM


# --------------------------------------------------
# Configuration: edit all model/run settings here.
# --------------------------------------------------

@dataclass
class Config:
    context: int = 256
    batch_size: int = 32
    steps: int = 10000

    dim: int = 384  # Size of the internal vector.
    target_head_dim: int = 64  # dim / heads: size of one attention head.
    recurrent_steps: int = 2
    transform_layers: int = 3
    num_experts: int = 16
    top_k: int = 2
    moe_aux_loss_weight: float = 0.01

    validate_every: int = 1000
    validation_batches: int = 10
    final_validation_batches: int = 100

    seed: int = 42
    learning_rate: float = 3e-4
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train[:5%]"
    validation_split: str = "validation[:1%]"

    max_new_bytes: int = 1000
    prompts: tuple[str, ...] = (
        "Once upon a time",
        "Lily went to the park",
        "Tom found a strange box",
        "The dog was afraid because",
        "One morning, Sarah discovered",
    )

    model_directory: str = "model"
    log_directory: str = "docs/run"

    @property
    def heads(self):
        return self.dim // self.target_head_dim

    def model_config(self):
        return {
            "dim": self.dim,
            "heads": self.heads,
            "target_head_dim": self.target_head_dim,
            "context": self.context,
            "recurrent_steps": self.recurrent_steps,
            "transform_layers": self.transform_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "moe_aux_loss_weight": self.moe_aux_loss_weight,
        }


class ByteTokenizer:
    """Keep text conversion separate so a future BPE tokenizer can replace it.

    Today one token is exactly one UTF-8 byte. BPE will also need vocabulary
    support in TinyLM and tokenizer state in checkpoints; neither changes here.
    """

    metadata = {"type": "byte", "encoding": "utf-8", "vocab_size": 256}

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")


def model_prefix(value):
    if "/" in value or "\\" in value or "\x00" in value:
        raise argparse.ArgumentTypeError(
            "Model prefix must be a filename prefix"
        )
    return value


def add_model_prefix_argument(parser):
    parser.add_argument(
        "--model-prefix",
        default="",
        type=model_prefix,
        help="Prefix for model/<prefix>best_model.pt and <prefix>last_model.pt",
    )


def seed_random(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    # CUDA work is asynchronous; finish it before reading a wall-clock timer.
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device):
    if torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


class TrainingTimer:
    """Time training intervals, excluding evaluation and checkpoint I/O."""

    def __init__(self, device):
        self.device = device
        self.elapsed = 0.0

    @contextmanager
    def measure(self):
        synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.elapsed += time.perf_counter() - started


def build_model(config, device):
    # One factory keeps architecture selection separate from training and
    # generation, including a future choice of dense or MoE blocks.
    return TinyLM(
        dim=config["dim"],
        heads=config["heads"],
        max_context=config["context"],
        recurrent_steps=config["recurrent_steps"],
        transform_layers=config["transform_layers"],
        num_experts=config["num_experts"],
        top_k=config["top_k"],
    ).to(device)


def print_model_info(model, config, device):
    print("Model")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context: {config['context']} bytes")
    print(f"  Dim: {config['dim']}; heads: {config['heads']}")
    print(f"  Head dim: {config['dim'] // config['heads']}")
    print(f"  Transformer blocks: {config['transform_layers']}")
    print(f"  Recurrent passes: {config['recurrent_steps']}")
    print(
        f"  MoE: {config['num_experts']} experts/block; "
        f"top-k: {config['top_k']}; aux weight: {config['moe_aux_loss_weight']}"
    )


class ModelRepository:
    def __init__(self, directory, prefix=""):
        self.directory = Path(directory)
        self.best_filename = f"{prefix}best_model.pt"
        self.last_filename = f"{prefix}last_model.pt"

    def check_overwrite(self, force=False):
        existing = [
            self.directory / filename
            for filename in (self.best_filename, self.last_filename)
            if (self.directory / filename).exists()
        ]
        if existing:
            print("WARNING: existing checkpoint(s):")
            for path in existing:
                print(f"  {path}")
            if not force:
                raise FileExistsError(
                    "Training aborted. Choose another --model-prefix or use "
                    "--force-overwrite."
                )
            print("  --force-overwrite enabled; these may be replaced.")

    def save(self, filename, model, optimizer, step, config, val_loss):
        self.directory.mkdir(parents=True, exist_ok=True)
        synchronize(next(model.parameters()).device)
        started = time.perf_counter()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "validation_loss": val_loss,
            "config": config.model_config(),
            "run_config": asdict(config),
            "tokenizer": ByteTokenizer.metadata,
        }
        path = self.directory / filename
        # Replace only after a complete write, preserving the previous best
        # checkpoint if serialization fails or the process is interrupted.
        with tempfile.NamedTemporaryFile(
            dir=self.directory,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        print(f"  Saved {path} (step {step:,}) in {elapsed:.2f}s")

    def load(self, filename, device):
        path = self.directory / filename
        synchronize(device)
        started = time.perf_counter()
        # Load on CPU first: generation does not need optimizer tensors on GPU.
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint.pop("optimizer", None)
        metadata = checkpoint.get("tokenizer", ByteTokenizer.metadata)
        if metadata != ByteTokenizer.metadata:
            raise ValueError("Unsupported tokenizer in checkpoint")
        model = build_model(checkpoint["config"], device)
        model.load_state_dict(checkpoint.pop("model"))
        model.eval()
        synchronize(device)
        elapsed = time.perf_counter() - started
        print(f"Loaded {path} in {elapsed:.2f}s (including model restoration)")
        return model, checkpoint


class Tee:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.flush()  # Preserve output immediately, even during long runs.
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()


@contextmanager
def log_run(directory, prefix, mode):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    # Microseconds distinguish training/generation started in the same second.
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = directory / f"{stamp}-{prefix or 'default'}.md"
    with path.open("x", encoding="utf-8") as log_file:
        log_file.write(f"# TinyLM {mode}\n\n")
        log_file.write(f"Started: {now.isoformat()}\n\n```text\n")
        with redirect_stdout(Tee(sys.stdout, log_file)):
            with redirect_stderr(Tee(sys.stderr, log_file)):
                print(f"Command: {shlex.join(sys.argv)}")
                print(f"Log: {path}\n")
                try:
                    yield path
                except BaseException:
                    # The usual traceback reaches the console after streams
                    # are restored; retain it in the run file as well.
                    traceback.print_exc(file=log_file)
                    raise
                finally:
                    log_file.write("\n```\n")
                    log_file.flush()
