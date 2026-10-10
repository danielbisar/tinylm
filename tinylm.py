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


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()

        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("Require num_experts >= 1 and 1 <= top_k <= num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(dim) for _ in range(num_experts)
        ])
        self.register_buffer(
            "last_expert_counts",
            torch.zeros(num_experts, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x):
        tokens = x.reshape(-1, x.shape[-1])

        # Keep router probabilities and balancing statistics in float32.
        probabilities = F.softmax(self.router(tokens).float(), dim=-1)
        weights, selected = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        counts = torch.bincount(
            selected.reshape(-1), minlength=self.num_experts,
        )
        self.last_expert_counts.copy_(counts.detach())

        # Switch-style balancing, extended to top-k: f is each expert's
        # fraction of assignments (N * k total), p its mean full-router
        # probability. E * sum(f * p) is 1 for uniform routing. Discrete
        # counts have no gradient; p supplies gradients to every router logit.
        fractions = counts.to(probabilities.dtype) / selected.numel()
        aux_loss = self.num_experts * (
            fractions * probabilities.mean(dim=0)
        ).sum()

        output = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(selected == expert_index)
            if token_indices.numel() == 0:
                continue

            # Actual sparse execution: this expert sees only its assigned
            # tokens. No capacity limit or dropped tokens.
            expert_output = expert(tokens[token_indices])
            weighted_output = expert_output * weights[
                token_indices, slots,
            ].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_indices, weighted_output.to(output.dtype))

        return output.reshape_as(x), aux_loss


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
        num_experts=8,
        top_k=2,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)

        self.attn = Attention(
            dim,
            heads,
        )

        self.norm2 = nn.RMSNorm(dim)

        self.mlp = SparseMoE(dim, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )

        mlp_output, aux_loss = self.mlp(
            self.norm2(x)
        )
        x = x + mlp_output

        return x, aux_loss


class TinyLM(nn.Module):
    def __init__(
        self,
        dim=128,
        heads=4,
        max_context=128,
        recurrent_steps=2,
        transform_layers=3,
        num_experts=8,
        top_k=2,
        vocab_size=256,
    ):
        super().__init__()

        if transform_layers < 1 or recurrent_steps < 1:
            raise ValueError("transform_layers and recurrent_steps must be positive")

        # RoPE itself has no learned position table.

        self.max_context = max_context

        # Token IDs -> learned vectors
        self.token_embedding = nn.Embedding(
            vocab_size,
            dim,
        )

        # Unique Transformer blocks, reused on every recurrent pass.
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, num_experts, top_k)
            for _ in range(transform_layers)
        ])
        self.register_buffer(
            "last_expert_counts",
            torch.zeros(transform_layers, num_experts, dtype=torch.long),
            persistent=False,
        )

        # recurrent_steps=2:
        #
        # A -> B -> C -> A -> B -> C
        self.recurrent_steps = recurrent_steps

        self.norm = nn.RMSNorm(dim)

        self.output = nn.Linear(
            dim,
            vocab_size,
        )

    def forward(self, tokens):
        """Return token logits and mean router auxiliary loss over block calls.

        last_expert_counts holds per-block assignments from this forward,
        summed over recurrent passes; it is diagnostic, not checkpoint state.
        """
        _, length = tokens.shape

        if length > self.max_context:
            raise ValueError(
                f"Sequence length {length} exceeds "
                f"max_context {self.max_context}"
            )

        # No position embedding anymore.
        x = self.token_embedding(tokens)
        aux_loss = x.new_zeros((), dtype=torch.float32)
        self.last_expert_counts.zero_()

        for _ in range(self.recurrent_steps):
            for block_index, block in enumerate(self.blocks):
                x, block_aux_loss = block(x)
                aux_loss = aux_loss + block_aux_loss
                self.last_expert_counts[block_index].add_(
                    block.mlp.last_expert_counts
                )

        aux_loss = aux_loss / (self.recurrent_steps * len(self.blocks))

        x = self.norm(x)

        return self.output(x), aux_loss
