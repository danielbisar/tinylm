import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

from tinylm import TinyLM


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load training data
# -------------------------

train_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1%]",
)

train_text = "\n".join(train_dataset["text"])

train_data = torch.tensor(
    list(train_text.encode("utf-8")),
    dtype=torch.long,
)


# -------------------------
# Load validation data
# -------------------------

validation_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation[:1%]",
)

validation_text = "\n".join(validation_dataset["text"])

validation_data = torch.tensor(
    list(validation_text.encode("utf-8")),
    dtype=torch.long,
)


# -------------------------
# Create model
# -------------------------
context = 128
model = TinyLM(
    dim=512,
    heads=4,
    max_context=context,
    recurrent_steps=3,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

parameters = sum(
    p.numel()
    for p in model.parameters()
)


# -------------------------
# Validation
# -------------------------

@torch.no_grad()
def validation_loss(model, data, samples=100):
    model.eval()

    losses = []

    for _ in range(samples):
        start = torch.randint(
            0,
            len(data) - context - 1,
            (1,),
        ).item()

        chunk = data[
            start:start + context + 1
        ].to(device)

        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# -------------------------
# Training
# -------------------------

steps = 20000

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_time = time.perf_counter()

model.train()

for step in range(steps):
    start = torch.randint(
        0,
        len(train_data) - context - 1,
        (1,),
    ).item()

    chunk = train_data[
        start:start + context + 1
    ].to(device)

    x = chunk[:-1].unsqueeze(0)
    y = chunk[1:].unsqueeze(0)

    logits = model(x)

    loss = F.cross_entropy(
        logits.reshape(-1, 256),
        y.reshape(-1),
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 2000 == 0:
        print(
            f"{step}: "
            f"train loss={loss.item():.4f}"
        )


# -------------------------
# Measurements
# -------------------------

if device == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start_time

if device == "cuda":
    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )
else:
    peak_vram = 0


# -------------------------
# Final validation
# -------------------------

val_loss = validation_loss(
    model,
    validation_data,
)


print()
print(f"Parameters:      {parameters:,}")
print(f"Training time:   {elapsed:.2f}s")
print(f"Peak VRAM:       {peak_vram:.1f} MB")
print(f"Steps/sec:       {steps / elapsed:.1f}")
print(f"Validation loss: {val_loss:.4f}")


# -------------------------
# Generation
# -------------------------

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_bytes=200,
):
    model.eval()

    x = torch.tensor(
        list(prompt.encode("utf-8")),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_bytes):
        x_context = x[:, -context:]

        logits = model(x_context)

        next_logits = logits[:, -1, :]

        next_byte = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        x = torch.cat(
            [x, next_byte],
            dim=1,
        )

    return bytes(
        x[0].tolist()
    ).decode(
        "utf-8",
        errors="replace",
    )


print()
print(generate(
    model,
    "Once upon a time",
    200,
))
