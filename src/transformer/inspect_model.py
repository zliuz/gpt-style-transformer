"""Inspect milestone 4 with: python -m transformer.inspect_model."""

import argparse
from pathlib import Path

import torch

from transformer.data import CharacterTokenizer, make_batch
from transformer.model import TinyTransformerLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect token-to-logit model forward")
    parser.add_argument("--text", type=Path, default=Path("data/toy.txt"))
    args = parser.parse_args()
    torch.manual_seed(42)
    text = args.text.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    x, y = make_batch(tokens, 2, 8, generator=torch.Generator().manual_seed(42))
    model = TinyTransformerLM(tokenizer.vocab_size, context_length=32)
    model.eval()
    try:
        with torch.no_grad():
            logits = model(x)
    except NotImplementedError as exc:
        parser.exit(
            1,
            f"Milestone 4 尚未完成：{exc}。\n"
            "请依次完成 src/transformer/model.py 的嵌入、Transformer 块、"
            "模型 forward，参阅 README.md。\n",
        )
    print(f"IDs: {tuple(x.shape)}, {x.dtype}; targets: {tuple(y.shape)}")
    print(f"Logits: {tuple(logits.shape)}, {logits.dtype}")
    print(f"Input:  {tokenizer.decode(x[0].tolist())!r}")
    print(f"Target: {tokenizer.decode(y[0].tolist())!r}")
    print(f"所有分数均有限：{bool(torch.isfinite(logits).all())}")
    print("最后位置的前五个原始分数：", logits[0, -1, :5].tolist())
    print("模型尚未训练；本入口只验证数据到 logits 的流程，不评价生成质量。")


if __name__ == "__main__":
    main()
