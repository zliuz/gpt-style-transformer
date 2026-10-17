"""Completed single-head and multi-head causal self-attention."""

import torch
from torch import nn


class CausalSelfAttentionHead(nn.Module):
    """Project [B, T, C] features into one attention head of width head_dim.

    Milestone 2 implements each attention step explicitly, without packaged
    attention operations. Milestone 3 reuses this completed head.
    """

    def __init__(self, embedding_dim: int, head_dim: int) -> None:
        super().__init__()
        for name, dimension in (
            ("embedding_dim", embedding_dim),
            ("head_dim", head_dim),
        ):
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output [B, T, head_dim] and attention weights [B, T, T].

        Require floating-point input [B, T, embedding_dim] with T >= 1;
        otherwise raise ValueError. Match the module's device and dtype as usual
        in PyTorch. Position i may attend to positions 0 through i, including i.
        Build the mask for the actual sequence length on x.device.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a three-dimensional torch.Tensor.")
        if not x.is_floating_point():
            raise ValueError("x must contain floating-point features.")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one sequence position.")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x's last dimension must equal embedding_dim.")

        # 1. Apply query, key, value to x: each result has shape [B, T, head_dim].
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2. Compare each query with all keys; transpose only the last two axes.
        #    Scale scores by sqrt(head_dim), not sqrt(embedding_dim).
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # 3. Mask future positions with negative infinity BEFORE softmax.
        #    Hint: torch.ones(..., device=x.device), tril, masked_fill.
        mask = torch.tril(
            torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~mask, float("-inf"))

        # 4. Normalize over keys (the last axis) to get attention_weights.
        # 5. Use those weights to combine values, then return both tensors.
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights


class MultiHeadCausalSelfAttention(nn.Module):
    """Milestone 3: combine independent heads and mix their features."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        for name, dimension in (
            ("embedding_dim", embedding_dim),
            ("num_heads", num_heads),
        ):
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            CausalSelfAttentionHead(embedding_dim, self.head_dim)
            for _ in range(num_heads)
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output [B, T, C] and per-head weights [B, H, T, T].

        Require floating-point features [B, T, embedding_dim], with T >= 1;
        otherwise raise ValueError. Match the module's device and dtype as usual
        in PyTorch. Keep heads in ModuleList order in both returned tensors.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a three-dimensional torch.Tensor.")
        if not x.is_floating_point():
            raise ValueError("x must contain floating-point features.")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one sequence position.")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x's last dimension must equal embedding_dim.")

        # 1. Call each head on the SAME full x; collect outputs and weights.
        #    Each head returns [B, T, head_dim] and [B, T, T].
        head_outputs = []
        head_weights = []
        for head in self.heads:
            output, weights = head(x)
            head_outputs.append(output)
            head_weights.append(weights)

        # 2. Concatenate head outputs along the feature axis to get [B, T, C].
        #    Hint: torch.cat joins an existing axis; do not sum or average heads.
        concatenated_output = torch.cat(head_outputs, dim=-1)

        # 3. Apply self.projection to mix features while preserving [B, T, C].
        mixed_output = self.projection(concatenated_output)

        # 4. Stack weights on a NEW head axis to get [B, H, T, T].
        #    Hint: torch.stack; retain the computation graph for both returns.
        stacked_weights = torch.stack(head_weights, dim=1)

        return mixed_output, stacked_weights
