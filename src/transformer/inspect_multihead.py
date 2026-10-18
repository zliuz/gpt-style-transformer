"""Inspect milestone 3 with: python -m transformer.inspect_multihead."""

import argparse

import torch

from transformer.attention import MultiHeadCausalSelfAttention


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect multi-head causal attention")
    parser.parse_args()
    torch.manual_seed(42)
    attention = MultiHeadCausalSelfAttention(embedding_dim=8, num_heads=2)
    x = torch.randn(1, 4, 8)
    attention.eval()
    try:
        with torch.no_grad():
            output, weights = attention(x)
    except NotImplementedError as exc:
        parser.exit(
            1,
            f"Milestone 3 尚未完成：{exc}。\n"
            "请完成 src/transformer/attention.py 中的 "
            "MultiHeadCausalSelfAttention.forward，参阅 README.md。\n",
        )
    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"Input: {tuple(x.shape)}; output: {tuple(output.shape)}")
    print(f"Attention weights [B, H, T, T]: {tuple(weights.shape)}")
    print("行 = query 位置，列 = key 位置；各头按 ModuleList 顺序排列：")
    for index in range(attention.num_heads):
        head_weights = weights[0, index]
        print(f"Head {index}:")
        print(head_weights)
        print(f"每行权重之和：{head_weights.sum(dim=-1)}")
        future_is_zero = bool((head_weights.triu(diagonal=1) == 0).all())
        print(f"未来位置权重全为零：{future_is_zero}")


if __name__ == "__main__":
    main()
