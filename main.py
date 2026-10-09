import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from tinylm import TinyLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

context = 128
batch_size = 32

dim = 384
heads = 4
recurrent_steps = 2

# Previous experiment:
#
# 20,000 steps * batch 1 * 128 bytes
# = 2,560,000 training bytes
#
# New experiment:
#
# 625 steps * batch 32 * 128 bytes
# = 2,560,000 training bytes
#
# So both runs see approximately the same amount
# of training data.

steps = 5000
print_every_x_steps = 500

torch.manual_seed(42)

if device == "cuda":
    torch.cuda.manual_seed_all(42)

print(f"Device:     {device}")
print(f"Context:    {context}")
print(f"Batch size: {batch_size}")
print(f"Steps:      {steps}")
print(f"Print every x steps: {print_every_x_steps}")
print()


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("Loading training data...")

train_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1%]",
)

train_text = "\n".join(
    train_dataset["text"]
)

# Keep entire dataset in CPU RAM.
train_data = torch.tensor(
    list(train_text.encode("utf-8")),
    dtype=torch.long,
)

print(
    f"Training bytes: {len(train_data):,}"
)


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

print("Loading validation data...")

validation_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation[:1%]",
)

validation_text = "\n".join(
    validation_dataset["text"]
)

validation_data = torch.tensor(
    list(validation_text.encode("utf-8")),
    dtype=torch.long,
)

print(
    f"Validation bytes: {len(validation_data):,}"
)

print()


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = TinyLM(
    dim=dim,
    heads=heads,
    max_context=context,
    recurrent_steps=recurrent_steps,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {parameters:,}")
print()


# --------------------------------------------------
# Create a random batch
# --------------------------------------------------

def get_batch(data, batch_size):
    starts = torch.randint(
        0,
        len(data) - context - 1,
        (batch_size,),
    )

    x = torch.stack([
        data[
            int(start):
            int(start) + context
        ]
        for start in starts
    ])

    y = torch.stack([
        data[
            int(start) + 1:
            int(start) + context + 1
        ]
        for start in starts
    ])

    return (
        x.to(device),
        y.to(device),
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

@torch.no_grad()
def validation_loss(
    model,
    data,
    batches=20,
):
    model.eval()

    losses = []

    for _ in range(batches):
        x, y = get_batch(
            data,
            batch_size,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
        )

        losses.append(
            loss.item()
        )

    model.train()

    return sum(losses) / len(losses)


# --------------------------------------------------
# Training
# --------------------------------------------------

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_time = time.perf_counter()

model.train()

for step in range(steps):
    x, y = get_batch(
        train_data,
        batch_size,
    )

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, 256),
        y.reshape(-1),
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    loss.backward()

    optimizer.step()

    if step % print_every_x_steps == 0:
        print(
            f"{step:7d}: "
            f"train loss={loss.item():.4f}"
        )


# --------------------------------------------------
# Measurements
# --------------------------------------------------

if device == "cuda":
    torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - start_time
)

if device == "cuda":
    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
else:
    peak_vram = 0


# --------------------------------------------------
# Final validation
# --------------------------------------------------

val_loss = validation_loss(
    model,
    validation_data,
)


training_bytes = (
    steps
    * batch_size
    * context
)

bytes_per_second = (
    training_bytes
    / elapsed
)


print()
print("------------------------------")
print("Results")
print("------------------------------")

print(
    f"Parameters:        "
    f"{parameters:,}"
)

print(
    f"Training bytes:    "
    f"{training_bytes:,}"
)

print(
    f"Training time:     "
    f"{elapsed:.2f}s"
)

print(
    f"Peak VRAM:         "
    f"{peak_vram:.1f} MB"
)

print(
    f"Steps/sec:         "
    f"{steps / elapsed:.2f}"
)

print(
    f"Bytes/sec:         "
    f"{bytes_per_second:,.0f}"
)

print(
    f"Validation loss:   "
    f"{val_loss:.4f}"
)


# --------------------------------------------------
# Generation
# --------------------------------------------------

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_bytes=200,
):
    model.eval()

    x = torch.tensor(
        list(
            prompt.encode("utf-8")
        ),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_bytes):
        # Model can only see the latest
        # `context` bytes.
        x_context = x[
            :,
            -context:
        ]

        logits = model(
            x_context
        )

        # Prediction at final position
        next_logits = logits[
            :,
            -1,
            :
        ]

        # Greedy decoding for now
        next_byte = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        x = torch.cat(
            [
                x,
                next_byte,
            ],
            dim=1,
        )

    return bytes(
        x[0].tolist()
    ).decode(
        "utf-8",
        errors="replace",
    )


print()
print("------------------------------")
print("Generation")
print("------------------------------")
print()

print(
    generate(
        model,
        "Once upon a time",
        200,
    )
)
