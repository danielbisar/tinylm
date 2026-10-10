"""Generate text from a saved TinyLM checkpoint."""

import argparse
import time

import torch

from core import (
    ByteTokenizer,
    Config,
    ModelRepository,
    add_model_prefix_argument,
    log_run,
    print_model_info,
    seed_random,
    synchronize,
)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_bytes):
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    prompt_length = x.size(1)
    if prompt_length == 0:
        raise ValueError("Generation requires a nonempty prompt")

    synchronize(device)
    started = time.perf_counter()
    for _ in range(max_new_bytes):
        # Only the most recent context window is visible to the model.
        x_context = x[:, -model.max_context:]
        logits, _ = model(x_context)
        # Greedy decoding picks the most likely next byte, without sampling.
        next_byte = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, next_byte], dim=1)
    tokens = x[0].tolist()
    synchronize(device)
    elapsed = time.perf_counter() - started
    # Count new raw bytes, excluding the prompt. UTF-8 replacement characters
    # can change the displayed encoded length, so measure before decoding.
    output_bytes = len(tokens) - prompt_length
    return tokenizer.decode(tokens), output_bytes, elapsed


def run(config, repository, filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_random(config.seed)
    model, checkpoint = repository.load(filename, device)
    print_model_info(model, checkpoint["config"], device)
    print(f"  Checkpoint step: {checkpoint['step']:,}")
    # Older last_model.pt files did not store a validation loss.
    if "validation_loss" in checkpoint:
        print(f"  Validation LM loss: {checkpoint['validation_loss']:.4f}")
    tokenizer = ByteTokenizer()
    total_bytes = 0
    total_seconds = 0.0
    print("\nGeneration")
    print(f"  Greedy decoding; {config.max_new_bytes:,} new bytes per prompt")
    for prompt in config.prompts:
        print(f"\nPrompt: {prompt}\n")
        text, output_bytes, elapsed = generate(
            model, tokenizer, prompt, config.max_new_bytes,
        )
        print(text)
        print(
            f"\n  Generated {output_bytes:,} bytes in {elapsed:.2f}s; "
            f"{output_bytes / elapsed:,.2f} output bytes/sec"
        )
        total_bytes += output_bytes
        total_seconds += elapsed
    print(
        f"\nGeneration total: {total_bytes:,} new bytes; {total_seconds:.2f}s; "
        f"{total_bytes / total_seconds:,.2f} output bytes/sec"
    )


def main(argv=None):
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_prefix_argument(parser)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint to generate from (default: best)",
    )
    args = parser.parse_args(argv)
    config = Config()
    repository = ModelRepository(config.model_directory, args.model_prefix)
    filename = (
        repository.best_filename if args.checkpoint == "best"
        else repository.last_filename
    )
    with log_run(config.log_directory, args.model_prefix, "generation"):
        try:
            run(config, repository, filename)
        except FileNotFoundError as error:
            print(f"Checkpoint not found: {error.filename}")
            return 1
        print(f"Total runtime: {time.perf_counter() - total_started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
