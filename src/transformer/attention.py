"""Milestone 2: implement a single causal self-attention head."""

import torch
from torch import nn


class CausalSelfAttentionHead(nn.Module):
    """Project [B, T, C] features into one attention head of width head_dim.

    The constructor and input checks are provided. Complete forward's calculation
    without using nn.MultiheadAttention or scaled_dot_product_attention.
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
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))

        # 4. Normalize over keys (the last axis) to get attention_weights.
        # 5. Use those weights to combine values, then return both tensors.
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
