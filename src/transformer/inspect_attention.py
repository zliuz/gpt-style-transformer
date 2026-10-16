"""Inspect milestone 2 with: python -m transformer.inspect_attention."""

import argparse

import torch

from transformer.attention import CausalSelfAttentionHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect causal attention weights")
    parser.parse_args()
    torch.manual_seed(42)
    head = CausalSelfAttentionHead(embedding_dim=6, head_dim=3)
    x = torch.randn(1, 4, 6)
    head.eval()
    try:
        with torch.no_grad():
            output, weights = head(x)
    except NotImplementedError as exc:
        parser.exit(
            1,
            f"Milestone 2 尚未完成：{exc}。\n"
            "请完成 src/transformer/attention.py 中的 forward，参阅 README.md。\n",
        )
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"Input: {tuple(x.shape)}; output: {tuple(output.shape)}")
    print(f"Attention weights: {tuple(weights.shape)}")
    print("行 = query 位置，列 = key 位置：")
    print(weights[0])
    print(f"每行权重之和：{weights.sum(dim=-1)}")
    print(f"未来位置权重全为零：{bool((weights.triu(diagonal=1) == 0).all())}")


if __name__ == "__main__":
    main()
