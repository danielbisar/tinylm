import time
import torch
import torch.nn.functional as F
from datasets import load_dataset

from tinylm import TinyLM


device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Configuration
# --------------------------------------------------

context = 256
batch_size = 32
steps = 10000

dim = 384 # size of internal vector
target_dimensions=64 # dim / heads (size of one head)
heads = dim // target_dimensions # number of attention heads
recurrent_steps = 2
transform_layers = 3

validate_every = 500
validation_batches = 20

torch.manual_seed(42)

if device == "cuda":
    torch.cuda.manual_seed_all(42)

print(f"Device:     {device}")
print(f"Context:    {context}")
print(f"Batch size: {batch_size}")
print(f"Steps:      {steps}")
print(f"Validate every x steps: {validate_every}")
print(f"Validate batches:       {validation_batches}")
print(f"Transform Layers:       {transform_layers}")
print(f"Target Dimensions:       {target_dimensions}")
print(f"Dim={dim}, Heads={heads}, Recurrent Steps={recurrent_steps}")
print()


# --------------------------------------------------
# Data
# --------------------------------------------------

print("Loading training data...")

train_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:5%]",
)

train_text = "\n".join(train_dataset["text"])

train_data = torch.tensor(
    list(train_text.encode("utf-8")),
    dtype=torch.long,
)

print(f"Training bytes: {len(train_data):,}")


print("Loading validation data...")

validation_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation[:1%]",
)

validation_text = "\n".join(validation_dataset["text"])

validation_data = torch.tensor(
    list(validation_text.encode("utf-8")),
    dtype=torch.long,
)

print(f"Validation bytes: {len(validation_data):,}")
print()


# --------------------------------------------------
# Model
# --------------------------------------------------

model = TinyLM(
    dim=dim,
    heads=heads,
    max_context=context,
    recurrent_steps=recurrent_steps,
    transform_layers=transform_layers
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
# Batch creation
# --------------------------------------------------

def get_batch(data, batch_size):
    starts = torch.randint(
        0,
        len(data) - context - 1,
        (batch_size,),
    )

    x = torch.stack([
        data[int(start):int(start) + context]
        for start in starts
    ])

    y = torch.stack([
        data[int(start) + 1:int(start) + context + 1]
        for start in starts
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# Validation
# --------------------------------------------------

@torch.no_grad()
def validation_loss(model, data):
    model.eval()

    losses = []

    for _ in range(validation_batches):
        x, y = get_batch(
            data,
            batch_size,
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, 256),
            y.reshape(-1),
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_loss = float("inf")

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_time = time.perf_counter()

model.train()

for step in range(1, steps + 1):
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

    # ----------------------------------------------
    # Periodic validation
    # ----------------------------------------------

    if step % validate_every == 0:
        val_loss = validation_loss(
            model,
            validation_data,
        )

        print(
            f"{step:7d}: "
            f"train={loss.item():.4f}  "
            f"val={val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "validation_loss": val_loss,
                    "config": {
                        "dim": dim,
                        "heads": heads,
                        "context": context,
                        "recurrent_steps": recurrent_steps,
                    },
                },
                "best_model.pt",
            )

            print("       saved best_model.pt")



torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": steps,
        "config": {
            "dim": dim,
            "heads": heads,
            "context": context,
            "recurrent_steps": recurrent_steps,
        },
    },
    "last_model.pt",
)

print("Saved last_model.pt")



# --------------------------------------------------
# Measurements
# --------------------------------------------------

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

print(f"Parameters:      {parameters:,}")
print(f"Training bytes:  {training_bytes:,}")
print(f"Training time:   {elapsed:.2f}s")
print(f"Peak VRAM:       {peak_vram:.1f} MB")
print(f"Bytes/sec:       {bytes_per_second:,.0f}")
print(f"Best val loss:   {best_val_loss:.4f}")


# --------------------------------------------------
# Load best checkpoint before generation
# --------------------------------------------------

checkpoint = torch.load(
    "best_model.pt",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model"]
)

print(
    f"Using checkpoint from step "
    f"{checkpoint['step']} "
    f"(val={checkpoint['validation_loss']:.4f})"
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
