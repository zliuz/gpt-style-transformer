"""Run after completing milestone 1: uv run python -m transformer.inspect_data."""

import argparse
from pathlib import Path

import torch

from transformer.data import CharacterTokenizer, make_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect next-character batches")
    parser.add_argument("--text", type=Path, default=Path("data/toy.txt"))
    args = parser.parse_args()
    text = args.text.read_text(encoding="utf-8")
    try:
        tokenizer = CharacterTokenizer.from_text(text)
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        x, y = make_batch(
            tokens,
            batch_size=16,
            context_length=32,
            generator=torch.Generator().manual_seed(42),
        )
    except NotImplementedError as exc:
        parser.exit(1, f"Milestone 1 is unfinished: {exc}. See README.md.\n")
    print(f"Characters: {len(text)}; vocabulary: {tokenizer.vocab_size}")
    print(f"x: {tuple(x.shape)}, {x.dtype}; y: {tuple(y.shape)}, {y.dtype}")
    print(f"Input:  {tokenizer.decode(x[0].tolist())!r}")
    print(f"Target: {tokenizer.decode(y[0].tolist())!r}")


if __name__ == "__main__":
    main()
