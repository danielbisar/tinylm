"""Generate text from a saved TinyLM checkpoint."""

import argparse
import codecs
import math
import time

import torch

from core import (
    LEGACY_BYTE_METADATA,
    tokenizer_from_metadata,
    Config,
    ModelRepository,
    add_model_prefix_argument,
    log_run,
    print_model_info,
    seed_random,
    synchronize,
)


@torch.no_grad()
def generate(
    model, tokenizer, prompt, max_output_bytes, *, greedy=False,
    temperature=0.7, top_p=0.9, generator=None, return_stop_reason=False,
):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    model.eval()
    device = next(model.parameters()).device
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("Generation requires a nonempty prompt")
    x = torch.tensor(
        prompt_tokens[-model.max_context:], dtype=torch.long, device=device,
    ).unsqueeze(0)
    # Vocabulary payloads are raw bytes, including partial UTF-8 characters.
    # An incremental decoder handles character boundaries and replacements
    # exactly as decoding once at the end, with no growing-sequence decode.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fragments = []
    output_bytes = 0
    generated_tokens = 0
    synchronize(device)
    started = time.perf_counter()
    stop_reason = "max_output_bytes"
    while output_bytes < max_output_bytes:
        logits, _ = model(x)
        next_logits = logits[:, -1, :]
        if greedy:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(next_logits.float() / temperature, dim=-1)
            if top_p < 1:
                probabilities, indices = probabilities.sort(dim=-1, descending=True)
                # Keep the smallest prefix reaching top_p, including the token
                # that crosses the threshold and always at least one token.
                remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
                remove[..., 0] = False
                probabilities = probabilities.masked_fill(remove, 0)
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(probabilities, 1, generator=generator)
                next_token = indices.gather(-1, sampled)
            else:
                next_token = torch.multinomial(probabilities, 1, generator=generator)
        if next_token.item() == tokenizer.eos_token_id:
            stop_reason = "EOS"
            break
        payload = tokenizer.token_bytes(next_token.item())
        fragment = decoder.decode(payload)
        fragments.append(fragment)
        output_bytes += len(fragment.encode("utf-8"))
        generated_tokens += 1
        x = torch.cat([x, next_token], dim=1)[:, -model.max_context:]
    fragment = decoder.decode(b"", final=True)
    fragments.append(fragment)
    output_bytes += len(fragment.encode("utf-8"))
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = (prompt + "".join(fragments), generated_tokens, output_bytes, elapsed)
    return result + (stop_reason,) if return_stop_reason else result


def run(config, repository, filename, *, greedy=False, temperature=0.7, top_p=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_random(config.seed)
    model, checkpoint = repository.load(filename, device)
    generator = None if greedy else torch.Generator(device=device).manual_seed(config.seed)
    average_bytes_per_token = checkpoint.get("data_metadata", {}).get(
        "training", {},
    ).get("average_bytes_per_token", 1.0)
    print_model_info(model, checkpoint["config"], device, average_bytes_per_token)
    print(f"  Checkpoint step: {checkpoint['step']:,}")
    # Older last_model.pt files did not store a validation loss.
    if "validation_loss" in checkpoint:
        print(f"  Validation token loss: {checkpoint['validation_loss']:.4f}")
    if checkpoint.get("validation_bpb") is not None:
        print(f"  Validation BPB: {checkpoint['validation_bpb']:.4f}")
    print(f"  Selection metric: {checkpoint.get('selection_metric', 'token_loss (legacy)')}")
    tokenizer = tokenizer_from_metadata(checkpoint.get("tokenizer", LEGACY_BYTE_METADATA))
    print(f"  Tokenizer: {tokenizer.metadata['type']}")
    total_tokens = 0
    total_bytes = 0
    total_seconds = 0.0
    print("\nGeneration")
    decoding = "Greedy decoding" if greedy else (
        f"Sampling; temperature={temperature:g}; top-p={top_p:g}"
    )
    print(f"  {decoding}; seed={config.seed}; {config.max_output_bytes:,} new UTF-8 bytes per prompt (approximately)")
    for prompt in config.prompts:
        print(f"\nPrompt: {prompt}\n")
        text, output_tokens, output_bytes, elapsed, stop_reason = generate(
            model, tokenizer, prompt, config.max_output_bytes,
            greedy=greedy, temperature=temperature, top_p=top_p, generator=generator,
            return_stop_reason=True,
        )
        print(text)
        print(
            f"\n  Generated {output_tokens:,} tokens; {output_bytes:,} UTF-8 bytes "
            f"in {elapsed:.2f}s; {output_tokens / elapsed:,.2f} tokens/sec; "
            f"{output_bytes / elapsed:,.2f} output bytes/sec; stopped: {stop_reason}"
        )
        total_tokens += output_tokens
        total_bytes += output_bytes
        total_seconds += elapsed
    print(
        f"\nGeneration total: {total_tokens:,} tokens; {total_bytes:,} new UTF-8 bytes; "
        f"{total_seconds:.2f}s; {total_tokens / total_seconds:,.2f} tokens/sec; "
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
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--temperature", type=float,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p", type=float,
        help="Nucleus sampling with this probability in (0, 1] (default: 0.9)",
    )
    parser.add_argument("--seed", type=int, help="Random seed (default: Config.seed)")
    args = parser.parse_args(argv)
    temperature = 0.7 if args.temperature is None else args.temperature
    top_p = 0.9 if args.top_p is None else args.top_p
    if not math.isfinite(temperature) or temperature <= 0:
        parser.error("--temperature must be finite and greater than zero")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    greedy = args.greedy
    config = Config()
    if args.seed is not None:
        config.seed = args.seed
    repository = ModelRepository(config.model_directory, args.model_prefix)
    filename = (
        repository.best_filename if args.checkpoint == "best"
        else repository.last_filename
    )
    with log_run(config.log_directory, args.model_prefix, "generation"):
        try:
            run(
                config, repository, filename,
                greedy=greedy, temperature=temperature, top_p=top_p,
            )
        except FileNotFoundError as error:
            print(f"Checkpoint not found: {error.filename}")
            return 1
        print(f"Total runtime: {time.perf_counter() - total_started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
