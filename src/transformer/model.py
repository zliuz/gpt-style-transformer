"""Completed milestone 4: connect token IDs to logits through a tiny Transformer.

Milestone 5 reuses this model for training without changing its forward API.
"""

import torch
from torch import nn

from transformer.attention import MultiHeadCausalSelfAttention


def _require_positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_token_ids(
    token_ids: torch.Tensor, vocab_size: int, context_length: int
) -> None:
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
        raise ValueError("token_ids must be a two-dimensional torch.Tensor.")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must have dtype torch.long.")
    if not 1 <= token_ids.shape[1] <= context_length:
        raise ValueError("Sequence length must be between 1 and context_length.")
    if ((token_ids < 0) | (token_ids >= vocab_size)).any():
        raise ValueError("Token IDs must satisfy 0 <= id < vocab_size.")


class TokenPositionEmbedding(nn.Module):
    """Combine learned character and absolute position embeddings."""

    def __init__(
        self, vocab_size: int, context_length: int, embedding_dim: int
    ) -> None:
        super().__init__()
        for name, dimension in (
            ("vocab_size", vocab_size),
            ("context_length", context_length),
            ("embedding_dim", embedding_dim),
        ):
            _require_positive_integer(name, dimension)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map torch.long IDs [B, T] to floating features [B, T, C].

        Positions start at zero for each sequence; use the actual T, not the
        maximum context_length. Input IDs must match the module's device.
        """
        _validate_token_ids(token_ids, self.vocab_size, self.context_length)
        # 1. Look up self.token_embedding for each token ID: [B, T, C].
        token_features = self.token_embedding(token_ids)

        # 2. Build positions 0..T-1 on token_ids.device (hint: torch.arange).
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)

        # 3. Look up self.position_embedding: [T, C].
        position_features = self.position_embedding(positions)

        # 4. Add the two; broadcasting shares positions across the batch.
        return token_features + position_features


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward with two residual paths."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        # The existing attention constructor validates dimensions/divisibility.
        self.attention = MultiHeadCausalSelfAttention(embedding_dim, num_heads)
        self.embedding_dim = embedding_dim
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        # Operates independently at each position: C -> 4C -> C.
        # Linear layers include biases; GELU provides the nonlinearity.
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve [B, T, C]; normalize over C, without reading future tokens."""
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a three-dimensional torch.Tensor.")
        if not x.is_floating_point():
            raise ValueError("x must contain floating-point features.")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one sequence position.")
        if x.shape[2] != self.embedding_dim:
            raise ValueError("x's last dimension must equal embedding_dim.")
        # 1. Normalize x with attention_norm, then call self.attention.
        #    Attention returns (features, weights); the block needs features only.
        attention_features, _ = self.attention(self.attention_norm(x))

        # 2. Add attention features to the ORIGINAL x (first residual).
        x = x + attention_features

        # 3. Normalize the UPDATED x with feed_forward_norm, then feed_forward.
        feed_forward_features = self.feed_forward(self.feed_forward_norm(x))

        # 4. Add feed-forward features to that UPDATED x (second residual).
        x = x + feed_forward_features

        # Return features only, not attention weights. Do not modify x in-place.
        return x


class TinyTransformerLM(nn.Module):
    """Assemble raw next-character logits; training is handled separately."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        _require_positive_integer("num_layers", num_layers)
        self.embedding = TokenPositionEmbedding(
            vocab_size, context_length, embedding_dim
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)
        )
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return raw logits [B, T, vocab_size] for input IDs [B, T].

        Return scores at EVERY position, without softmax, sampling, or loss.
        Input IDs and parameters must be on the same device as usual in PyTorch.
        """
        _validate_token_ids(token_ids, self.vocab_size, self.context_length)

        # 1. Get features from self.embedding(token_ids).
        x = self.embedding(token_ids)

        # 2. Pass features through EACH block in order; don't reuse the input.
        for block in self.blocks:
            x = block(x)

        # 3. Apply self.final_norm, then self.lm_head to produce logits.
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Return the logits tensor directly. Keep the full computation graph.
        return logits
