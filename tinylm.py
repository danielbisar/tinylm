import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = int(dim * 8 / 3)

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(
            F.silu(self.gate(x))
            * self.up(x)
        )


def apply_rope(x):
    """
    x shape:
        [batch, heads, sequence, head_dim]
    """

    batch, heads, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0

    half_dim = head_dim // 2

    positions = torch.arange(
        seq_len,
        device=x.device,
        dtype=torch.float32,
    )

    frequencies = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                half_dim,
                device=x.device,
                dtype=torch.float32,
            )
            / half_dim
        )
    )

    angles = positions[:, None] * frequencies[None, :]

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )

    return rotated.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=False,
        )

        self.output = nn.Linear(
            dim,
            dim,
            bias=False,
        )

    def forward(self, x):
        batch, length, dim = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        q = q.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        # RoPE applies to queries and keys
        q = apply_rope(q)
        k = apply_rope(k)

        # PyTorch handles causal masking internally.
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        x = x.transpose(
            1,
            2,
        ).contiguous()

        x = x.view(
            batch,
            length,
            dim,
        )

        return self.output(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim=128,
        heads=4,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)

        self.attn = Attention(
            dim,
            heads,
        )

        self.norm2 = nn.RMSNorm(dim)

        self.mlp = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class TinyLM(nn.Module):
    def __init__(
        self,
        dim=128,
        heads=4,
        max_context=128,
        recurrent_steps=2,
    ):
        super().__init__()

        # max_context is kept in the API so main.py
        # does not need to change.
        # RoPE itself has no learned position table.

        self.max_context = max_context

        # Byte values 0..255 -> learned vectors
        self.token_embedding = nn.Embedding(
            256,
            dim,
        )

        # Three unique Transformer blocks:
        # A -> B -> C
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads),
            TransformerBlock(dim, heads),
            TransformerBlock(dim, heads),
        ])

        # recurrent_steps=2:
        #
        # A -> B -> C -> A -> B -> C
        self.recurrent_steps = recurrent_steps

        self.norm = nn.RMSNorm(dim)

        self.output = nn.Linear(
            dim,
            256,
        )

    def forward(self, tokens):
        _, length = tokens.shape

        if length > self.max_context:
            raise ValueError(
                f"Sequence length {length} exceeds "
                f"max_context {self.max_context}"
            )

        # No position embedding anymore.
        x = self.token_embedding(tokens)

        for _ in range(self.recurrent_steps):
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)

        return self.output(x)
