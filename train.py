"""Train TinyLM and save best/last checkpoints in model/."""

import argparse
import time

import torch
import torch.nn.functional as F

from core import (
    ByteTokenizer,
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


def load_data(config, split, tokenizer):
    from datasets import load_dataset

    print(f"Loading {config.dataset_name} ({split})...")
    dataset = load_dataset(config.dataset_name, split=split)
    text = "\n".join(dataset["text"])
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"  Dataset bytes: {len(data):,}")
    return data


def get_batch(data, batch_size, context, device):
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    x = torch.stack([data[int(s):int(s) + context] for s in starts])
    # The target is shifted by one: each position learns to predict the next
    # byte using only the bytes before it (causal attention prevents peeking).
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
    # only the next-byte prediction loss so it remains easy to compare.
    loss = lm_loss + config.moe_aux_loss_weight * router_aux_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return lm_loss.detach(), router_aux_loss.detach()


@torch.no_grad()
def validation_loss(model, data, config, device, batches):
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = get_batch(data, config.batch_size, config.context, device)
            logits, _ = model(x)
            losses.append(language_model_loss(logits, y).item())
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def evaluate(model, data, config, device, batches):
    synchronize(device)
    started = time.perf_counter()
    loss = validation_loss(model, data, config, device, batches)
    synchronize(device)
    elapsed = time.perf_counter() - started
    evaluated_bytes = batches * config.batch_size * config.context
    print(
        f"  Validation: LM loss {loss:.4f}; {batches} batches; "
        f"{elapsed:.2f}s; {evaluated_bytes / elapsed:,.0f} bytes/sec"
    )
    return loss


def print_expert_usage(expert_counts):
    # Percentages of top-k assignments, per unique block, over both passes.
    utilization = 100 * expert_counts.float() / expert_counts.sum(
        dim=-1, keepdim=True,
    )
    print("  Expert assignment % (latest training batch):")
    for index, row in enumerate(utilization.cpu().tolist()):
        values = ", ".join(f"{value:.1f}" for value in row)
        print(f"    B{index}: [{values}]")


def print_progress(step, config, timer, total_started, training_started):
    now = time.perf_counter()
    elapsed = now - total_started
    # Estimate remaining loop time from observed wall time, including the
    # periodic validation/checkpoint overhead seen so far.
    remaining = (now - training_started) / step * (config.steps - step)
    # With byte tokenization, batch size * context is also processed bytes.
    processed_bytes = step * config.batch_size * config.context
    print(
        f"  Progress: {100 * step / config.steps:.1f}%; "
        f"total elapsed {elapsed:.2f}s; training {timer.elapsed:.2f}s"
    )
    print(
        f"  Training throughput: {processed_bytes / timer.elapsed:,.0f} "
        f"bytes/sec; estimated remaining training loop: {remaining:.2f}s"
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
    tokenizer = ByteTokenizer()
    train_data = load_data(config, config.train_split, tokenizer)
    validation_data = load_data(config, config.validation_split, tokenizer)
    model = build_model(config.model_config(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    print_model_info(model, config.model_config(), device)
    print()

    reset_peak_memory(device)
    timer = TrainingTimer(device)
    training_started = time.perf_counter()
    best_val_loss = float("inf")
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
        val_loss = evaluate(
            model, validation_data, config, device, config.validation_batches,
        )
        print_expert_usage(expert_counts)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            repository.save(
                repository.best_filename,
                model,
                optimizer,
                completed,
                config,
                val_loss,
            )
        print_progress(
            completed, config, timer, total_started, training_started,
        )

    print("\nFinal validation (last model)")
    final_val_loss = evaluate(
        model, validation_data, config, device, config.final_validation_batches,
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
    )
    print_progress(config.steps, config, timer, total_started, training_started)

    print("\nResults")
    training_bytes = config.steps * config.batch_size * config.context
    print(f"  Sampled training bytes: {training_bytes:,}")
    print(f"  Training time (steps only): {timer.elapsed:.2f}s")
    print(f"  Total runtime: {time.perf_counter() - total_started:.2f}s")
    print(f"  Peak VRAM: {peak_memory_mib(device):.1f} MiB")
    print(f"  Best periodic validation LM loss: {best_val_loss:.4f}")


def main(argv=None):
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_prefix_argument(parser)
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Allow replacement of existing best and last checkpoints",
    )
    args = parser.parse_args(argv)
    config = Config()
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
