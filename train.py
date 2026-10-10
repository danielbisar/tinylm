"""Train TinyLM and save best/last checkpoints in model/."""

import argparse
import math
import time

import torch
import torch.nn.functional as F

from core import (
    create_tokenizer,
    Config,
    ModelRepository,
    TrainingTimer,
    add_model_prefix_argument,
    build_model,
    log_run,
    peak_memory_mib,
    print_model_info,
    reset_peak_memory,
    seed_random,
    synchronize,
)


def load_source(config, split):
    from datasets import load_dataset

    print(f"Loading {config.dataset_name} ({split})...")
    dataset = load_dataset(config.dataset_name, split=split)
    return "\n".join(dataset["text"])


def load_data(config, split, tokenizer, text=None):
    if text is None:
        text = load_source(config, split)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    raw_bytes = len(text.encode("utf-8"))
    print(f"  Raw UTF-8 bytes: {raw_bytes:,}")
    print(f"  Tokens: {len(data):,}")
    print(f"  Average bytes/token: {raw_bytes / len(data):.4f}")
    return data


def source_statistics(data, tokenizer):
    lengths = torch.tensor(tokenizer.byte_lengths, dtype=torch.long)
    raw_bytes = int(lengths[data].sum())
    return {
        "raw_utf8_bytes": raw_bytes,
        "tokens": len(data),
        "average_bytes_per_token": raw_bytes / len(data),
    }


def get_batch(data, batch_size, context, device):
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    x = torch.stack([data[int(s):int(s) + context] for s in starts])
    # The target is shifted by one: each position learns to predict the next
    # token using only the tokens before it (causal attention prevents peeking).
    y = torch.stack([data[int(s) + 1:int(s) + context + 1] for s in starts])
    return x.to(device), y.to(device)


def language_model_loss(logits, targets):
    # Flatten batch and sequence axes; the last axis is the vocabulary.
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
    )


def train_step(model, optimizer, data, config, device):
    x, y = get_batch(data, config.batch_size, config.context, device)
    logits, router_aux_loss = model(x)
    lm_loss = language_model_loss(logits, y)
    # The auxiliary term encourages balanced expert use; validation reports
    # only prediction loss (and BPB), excluding the auxiliary term.
    loss = lm_loss + config.moe_aux_loss_weight * router_aux_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return lm_loss.detach(), router_aux_loss.detach()


def bits_per_byte(total_nll_nats, source_bytes):
    return total_nll_nats / (math.log(2) * source_bytes)


@torch.no_grad()
def validation_loss(model, data, config, device, batches, byte_lengths=None):
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    # The reversible ByteLevel vocabulary gives an exact byte length for
    # every token, even partial Unicode characters. Index this table with
    # shifted targets, not inputs. Offsets can overlap within a UTF-8 codepoint
    # and per-token decoding replaces partial codepoints, so neither is used.
    # Summing target payload lengths accounts exactly for arbitrary windows
    # in the actual joined encoded stream, including newline/space merges.
    if byte_lengths is None:
        if config.vocab_size != 256:
            raise ValueError("Validation requires tokenizer byte lengths")
        byte_lengths = (1,) * 256
    lengths = torch.tensor(byte_lengths, dtype=torch.long, device=device)
    try:
        for _ in range(batches):
            x, y = get_batch(data, config.batch_size, config.context, device)
            logits, _ = model(x)
            total_nll += language_model_loss(logits, y).item() * y.numel()
            total_tokens += y.numel()
            total_bytes += lengths[y].sum().item()
    finally:
        model.train(was_training)
    return total_nll / total_tokens, bits_per_byte(total_nll, total_bytes)


def evaluate(model, data, config, device, batches, byte_lengths=None):
    synchronize(device)
    started = time.perf_counter()
    loss, bpb = validation_loss(
        model, data, config, device, batches, byte_lengths,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    evaluated_tokens = batches * config.batch_size * config.context
    print(
        f"  Val token loss: {loss:.4f}; Val BPB: {bpb:.4f}; "
        f"{batches} batches; {elapsed:.2f}s; "
        f"{evaluated_tokens / elapsed:,.0f} tokens/sec"
    )
    return loss, bpb


def print_expert_usage(expert_counts):
    # Percentages of top-k assignments, per unique block, over both passes.
    utilization = 100 * expert_counts.float() / expert_counts.sum(
        dim=-1, keepdim=True,
    )
    print("  Expert assignment % (latest training batch):")
    for index, row in enumerate(utilization.cpu().tolist()):
        values = ", ".join(f"{value:.1f}" for value in row)
        print(f"    B{index}: [{values}]")


def print_progress(
    step, config, timer, total_started, training_started, average_bytes_per_token,
):
    now = time.perf_counter()
    elapsed = now - total_started
    # Estimate remaining loop time from observed wall time, including the
    # periodic validation/checkpoint overhead seen so far.
    remaining = (now - training_started) / step * (config.steps - step)
    processed_tokens = step * config.batch_size * config.context
    print(
        f"  Progress: {100 * step / config.steps:.1f}%; "
        f"total elapsed {elapsed:.2f}s; training {timer.elapsed:.2f}s"
    )
    print(
        f"  Training throughput: {processed_tokens / timer.elapsed:,.0f} "
        f"tokens/sec; estimated remaining training loop: {remaining:.2f}s"
    )

    print(
        f"  Tokens processed: {processed_tokens:,}; estimated source-byte "
        f"throughput: {processed_tokens * average_bytes_per_token / timer.elapsed:,.0f} "
        f"bytes/sec (training corpus average)"
    )


def print_training_config(config):
    print("Training")
    print(f"  Batch size: {config.batch_size}; steps: {config.steps:,}")
    print(f"  Validate every: {config.validate_every:,} steps")
    print(f"  Validation batches: {config.validation_batches}")
    print(f"  Final validation batches: {config.final_validation_batches}")
    print(f"  Seed: {config.seed}; AdamW learning rate: {config.learning_rate}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Splits: {config.train_split}; {config.validation_split}\n")


def train(config, repository, total_started):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_random(config.seed)
    print_training_config(config)
    training_text = load_source(config, config.train_split)
    tokenizer = create_tokenizer(config, training_text)
    config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {config.tokenizer_type}; actual vocabulary: {tokenizer.vocab_size:,}")
    print("Training source:")
    train_data = load_data(config, config.train_split, tokenizer, training_text)
    del training_text
    print("Validation source:")
    validation_data = load_data(config, config.validation_split, tokenizer)
    data_metadata = {
        "training": source_statistics(train_data, tokenizer),
        "validation": source_statistics(validation_data, tokenizer),
    }
    average_bytes_per_token = data_metadata["training"]["average_bytes_per_token"]
    model = build_model(config.model_config(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print_model_info(model, config.model_config(), device, average_bytes_per_token)
    print("  Best checkpoint selection: validation BPB")
    print()

    reset_peak_memory(device)
    timer = TrainingTimer(device)
    training_started = time.perf_counter()
    best_val_bpb = float("inf")
    best_token_loss = float("inf")
    model.train()

    # Time whole intervals to avoid introducing a CUDA sync at every step.
    # Validation, console/file output, and checkpoint writes happen outside.
    completed = 0
    while completed < config.steps:
        next_validation = min(completed + config.validate_every, config.steps)
        with timer.measure():
            for _ in range(completed + 1, next_validation + 1):
                lm_loss, aux_loss = train_step(
                    model, optimizer, train_data, config, device,
                )
        completed = next_validation

        # Snapshot training assignments before validation overwrites them.
        expert_counts = model.last_expert_counts.clone()
        print(f"\nStep {completed:,}/{config.steps:,}")
        print(
            f"  Training: LM loss {lm_loss.item():.4f}; "
            f"router aux {aux_loss.item():.4f}"
        )
        val_loss, val_bpb = evaluate(
            model, validation_data, config, device, config.validation_batches,
            tokenizer.byte_lengths,
        )
        print_expert_usage(expert_counts)
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            best_token_loss = val_loss
            repository.save(
                repository.best_filename,
                model,
                optimizer,
                completed,
                config,
                val_loss,
                tokenizer,
                val_bpb,
                data_metadata,
            )
        print_progress(
            completed, config, timer, total_started, training_started,
            average_bytes_per_token,
        )

    print("\nFinal validation (last model)")
    final_val_loss, final_val_bpb = evaluate(
        model, validation_data, config, device, config.final_validation_batches,
        tokenizer.byte_lengths,
    )
    # Keep best-checkpoint selection on the same periodic validation budget.
    # The larger final pass measures the last model, without changing that best.
    repository.save(
        repository.last_filename,
        model,
        optimizer,
        config.steps,
        config,
        final_val_loss,
        tokenizer,
        final_val_bpb,
        data_metadata,
    )
    print_progress(
        config.steps, config, timer, total_started, training_started,
        average_bytes_per_token,
    )

    print("\nResults")
    training_tokens = config.steps * config.batch_size * config.context
    print(f"  Sampled training tokens: {training_tokens:,}")
    print(f"  Training time (steps only): {timer.elapsed:.2f}s")
    print(f"  Total runtime: {time.perf_counter() - total_started:.2f}s")
    print(f"  Peak VRAM: {peak_memory_mib(device):.1f} MiB")
    print(f"  Best periodic validation BPB: {best_val_bpb:.4f}")
    print(f"  Corresponding validation token loss: {best_token_loss:.4f}")


def main(argv=None):
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_prefix_argument(parser)
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow replacement of existing best and last checkpoints",
    )
    parser.add_argument("--tokenizer-type", choices=("byte", "bpe"))
    parser.add_argument("--bpe-vocab-size", type=int)
    args = parser.parse_args(argv)
    config = Config()
    if args.tokenizer_type is not None:
        config.tokenizer_type = args.tokenizer_type
    if args.bpe_vocab_size is not None:
        config.bpe_vocab_size = args.bpe_vocab_size
    repository = ModelRepository(config.model_directory, args.model_prefix)
    with log_run(config.log_directory, args.model_prefix, "training"):
        try:
            # Check BOTH destinations before loading data or creating a model.
            repository.check_overwrite(args.force_overwrite)
        except FileExistsError as error:
            print(error)
            return 1
        train(config, repository, total_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
