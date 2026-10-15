"""Milestone 1: implement the three exercises below, in order."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CharacterTokenizer:
    """A sorted vocabulary where each character's index is its token ID."""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Exercise 1: build a sorted, unique vocabulary; reject empty text."""
        if not text:
            raise ValueError("Input text cannot be empty.")
        unique_characters = sorted(set(text))
        return cls(tuple(unique_characters))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        """Exercise 2: map characters to IDs; unknown characters raise ValueError."""
        character_indices = {}
        for index, character in enumerate(self.characters):
            character_indices.setdefault(character, index)
        try:
            return [character_indices[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from None

    def decode(self, token_ids: list[int]) -> str:
        """Exercise 2: map IDs to text; negative/out-of-range IDs raise ValueError."""
        characters = []
        vocab_size = self.vocab_size
        for token_id in token_ids:
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"Token ID out of range: {token_id!r}")
            characters.append(self.characters[token_id])
        return "".join(characters)


def make_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise 3: sample next-character pairs on CPU.

    Require a one-dimensional CPU torch.long tensor, positive batch/context sizes,
    and at least context_length + 1 tokens; otherwise raise ValueError.
    Sample starts uniformly with replacement using the supplied generator.
    Return x and y with shape [batch_size, context_length] and dtype torch.long.
    At start s, x is tokens[s:s+T] and y is tokens[s+1:s+T+1].
    """
    if tokens.ndim != 1 or tokens.device.type != "cpu" or tokens.dtype != torch.long:
        raise ValueError("tokens must be a one-dimensional CPU torch.long tensor.")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive.")
    if tokens.numel() <= context_length:
        raise ValueError("At least context_length + 1 tokens are required.")

    starts = torch.randint(
        tokens.numel() - context_length, (batch_size,), generator=generator
    )
    indices = starts[:, None] + torch.arange(context_length)
    return tokens[indices], tokens[indices + 1]
