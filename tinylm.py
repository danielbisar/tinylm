import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        length = x.size(1)

        mask = torch.triu(
            torch.ones(length, length, device=x.device),
            diagonal=1,
        ).bool()

        h = self.norm1(x)

        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=mask,
        )

        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        return x


class TinyLM(nn.Module):
    def __init__(self, dim=128, heads=4, max_context=128):
        super().__init__()

        self.token_embedding = nn.Embedding(256, dim)
        self.position_embedding = nn.Embedding(max_context, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads)
            for _ in range(4)
        ])

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 256)

    def forward(self, tokens):
        batch, length = tokens.shape

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.output(x)
