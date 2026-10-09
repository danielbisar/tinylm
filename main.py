import torch
import torch.nn.functional as F

from tinylm import TinyLM


device = "cuda" if torch.cuda.is_available() else "cpu"

text = """
the cat sat on the mat.
the dog sat on the floor.
the cat likes milk.
the dog likes food.
""" * 100

data = torch.tensor(
    list(text.encode("utf-8")),
    dtype=torch.long,
    device=device,
)

context = 64

model = TinyLM(
    dim=128,
    heads=4,
    max_context=context,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



for step in range(1000):
    start = torch.randint(
        0,
        len(data) - context - 1,
        (1,),
    ).item()

    chunk = data[start:start + context + 1]

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

    if step % 100 == 0:
        print(f"{step}: loss={loss.item():.4f}")

def generate(model, prompt, max_new_bytes=100):
    model.eval()

    x = torch.tensor(
        list(prompt.encode("utf-8")),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_bytes):
            # only keep the context window
            x_context = x[:, -context:]

            logits = model(x_context)

            next_logits = logits[:, -1, :]
            next_byte = torch.argmax(next_logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_byte], dim=1)

    return bytes(x[0].tolist()).decode("utf-8", errors="replace")


print(generate(model, "the cat", 100))
