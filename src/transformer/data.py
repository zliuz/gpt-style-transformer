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
        raise NotImplementedError("Exercise 1: implement from_text")

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        """Exercise 2: map characters to IDs; unknown characters raise ValueError."""
        raise NotImplementedError("Exercise 2: implement encode")

    def decode(self, token_ids: list[int]) -> str:
        """Exercise 2: map IDs to text; negative/out-of-range IDs raise ValueError."""
        raise NotImplementedError("Exercise 2: implement decode")


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
    raise NotImplementedError("Exercise 3: implement make_batch")

