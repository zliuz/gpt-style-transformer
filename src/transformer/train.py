"""Milestone 5 runner; core learning functions live in training.py."""

import argparse
import math
from pathlib import Path

import torch

from transformer.data import CharacterTokenizer, make_batch
from transformer.model import TinyTransformerLM
from transformer.training import evaluate_loss, split_tokens, train_step


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _learning_rate(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**63 - 1:
        raise argparse.ArgumentTypeError("must be an integer between 0 and 2**63 - 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny character Transformer")
    parser.add_argument("--mode", choices=("train", "overfit"), default="train")
    parser.add_argument("--text", type=Path, default=Path("data/toy.txt"))
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--context-length", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=_learning_rate, default=0.003)
    parser.add_argument("--seed", type=_seed, default=42)
    return parser


def make_evaluation_batches(
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    seed: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Cache five batches per partition without consuming training RNG state."""
    batches = {}
    for offset, (name, tokens) in enumerate(
        (("train", train_tokens), ("validation", validation_tokens)), start=1
    ):
        generator = torch.Generator().manual_seed(seed + offset)
        batches[name] = [
            make_batch(tokens, batch_size, context_length, generator=generator)
            for _ in range(5)
        ]
    return batches


def run_training(
    model: TinyTransformerLM,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Provided control flow: sampling, diagnostics, and reporting, no core solution."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    generator = torch.Generator().manual_seed(args.seed)
    if args.mode == "overfit":
        batch = make_batch(
            train_tokens, args.batch_size, args.context_length, generator=generator
        )
        before = evaluate_loss(model, [batch])
        print(f"Fixed batch before: loss={before:.6f}")
        for _ in range(args.steps):
            train_step(model, optimizer, *batch)
        after = evaluate_loss(model, [batch])
        print(f"Fixed batch after {args.steps} steps: loss={after:.6f}")
        print("固定批次诊断只检查能否记住训练数据，不证明泛化。")
        return

    batches = make_evaluation_batches(
        train_tokens,
        validation_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        seed=args.seed,
    )

    def report(step: int) -> None:
        train_loss = evaluate_loss(model, batches["train"])
        validation_loss = evaluate_loss(model, batches["validation"])
        print(
            f"Step {step}: train_loss={train_loss:.6f}; "
            f"validation_loss={validation_loss:.6f}"
        )

    report(0)
    for step in range(1, args.steps + 1):
        batch = make_batch(
            train_tokens, args.batch_size, args.context_length, generator=generator
        )
        train_step(model, optimizer, *batch)
        if step % 50 == 0 or step == args.steps:
            report(step)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    try:
        text = args.text.read_text(encoding="utf-8")
        tokenizer = CharacterTokenizer.from_text(text)
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        train_tokens, validation_tokens = split_tokens(
            tokens, context_length=args.context_length
        )
        model = TinyTransformerLM(
            tokenizer.vocab_size, context_length=args.context_length
        )
        print(
            f"CPU; vocabulary={tokenizer.vocab_size}; "
            f"train_tokens={len(train_tokens)}; validation_tokens={len(validation_tokens)}"
        )
        run_training(model, train_tokens, validation_tokens, args)
    except NotImplementedError as exc:
        parser.exit(1, f"Milestone 5 尚未完成：{exc}。请参阅 README.md。\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
