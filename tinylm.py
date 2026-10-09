import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.embedding = nn.Embedding(256, dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)

        self.output = nn.Linear(dim, 256)

    def forward(self, x):
        # x: [batch, sequence]
        x = self.embedding(x)

        # causal mask: no looking into the future
        length = x.size(1)
        mask = torch.triu(
            torch.ones(length, length, device=x.device),
            diagonal=1
        ).bool()

        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=mask
        )

        x = self.norm(x + attn_out)

        return self.output(x)
