import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = int(dim * 8 / 3)

        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(
            torch.nn.functional.silu(self.gate(x))
            * self.up(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm2 = nn.RMSNorm(dim)

        self.mlp = SwiGLU(dim)

    def forward(self, x):
        length = x.size(1)

        # Causal mask:
        # a position may only look at itself and earlier positions.
        mask = torch.triu(
            torch.ones(
                length,
                length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        h = self.norm1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=mask,
            need_weights=False,
        )

        # Residual connection
        x = x + attn_out

        # Feed-forward network + residual connection
        x = x + self.mlp(self.norm2(x))

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

        # Byte values 0..255 -> learned vectors
        self.token_embedding = nn.Embedding(
            256,
            dim,
        )

        # Learned position vectors
        self.position_embedding = nn.Embedding(
            max_context,
            dim,
        )

        # Three unique Transformer blocks:
        # A -> B -> C
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads),
            TransformerBlock(dim, heads),
            TransformerBlock(dim, heads),
        ])

        # With recurrent_steps=2:
        # A -> B -> C -> A -> B -> C
        self.recurrent_steps = recurrent_steps

        self.norm = nn.RMSNorm(dim)

        # Convert internal vector back into
        # probabilities/scores for 256 possible bytes.
        self.output = nn.Linear(
            dim,
            256,
        )

    def forward(self, tokens):
        _, length = tokens.shape

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        for _ in range(self.recurrent_steps):
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)

        return self.output(x)
