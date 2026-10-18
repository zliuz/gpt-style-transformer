"""Regression checks for completed milestone 4 embeddings, blocks, and logits."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transformer.attention import MultiHeadCausalSelfAttention
from transformer.model import (
    TinyTransformerLM,
    TokenPositionEmbedding,
    TransformerBlock,
)

pytestmark = pytest.mark.milestone4


@pytest.fixture
def embedding():
    with torch.random.fork_rng():
        torch.manual_seed(41)
        return TokenPositionEmbedding(7, 5, 8)


@pytest.fixture
def block():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return TransformerBlock(8, 2)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(43)
        return TinyTransformerLM(7, 5, embedding_dim=8, num_heads=2, num_layers=2)


def test_embedding_constructor_registers_two_learned_tables(embedding):
    assert embedding.token_embedding.weight.shape == (7, 8)
    assert embedding.position_embedding.weight.shape == (5, 8)
    assert set(dict(embedding.named_parameters())) == {
        "token_embedding.weight",
        "position_embedding.weight",
    }


def test_block_constructor_provides_norms_attention_and_positionwise_mlp(block):
    assert isinstance(block.attention, MultiHeadCausalSelfAttention)
    assert block.attention_norm.normalized_shape == (8,)
    assert block.feed_forward_norm.normalized_shape == (8,)
    assert block.attention_norm is not block.feed_forward_norm
    assert isinstance(block.feed_forward, nn.Sequential)
    first, activation, last = block.feed_forward
    assert (first.in_features, first.out_features) == (8, 32)
    assert isinstance(activation, nn.GELU)
    assert (last.in_features, last.out_features) == (32, 8)
    assert first.bias is not None and last.bias is not None


def test_model_constructor_registers_independent_blocks_and_unshared_output(model):
    assert isinstance(model.embedding, TokenPositionEmbedding)
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 2
    assert all(isinstance(item, TransformerBlock) for item in model.blocks)
    assert model.blocks[0] is not model.blocks[1]
    assert model.final_norm.normalized_shape == (8,)
    assert model.lm_head.weight.shape == (7, 8)
    assert model.lm_head.bias is None
    assert model.lm_head.weight is not model.embedding.token_embedding.weight
    parameters = dict(model.named_parameters())
    assert "blocks.0.attention.heads.0.query.weight" in parameters
    assert "blocks.1.feed_forward.0.weight" in parameters
    assert len({p.data_ptr() for p in parameters.values()}) == len(parameters)
    assert all(p.requires_grad for p in parameters.values())


def test_model_defaults_match_initial_architecture():
    model = TinyTransformerLM(7, 32)
    assert model.embedding_dim == 64
    assert model.num_heads == 4
    assert model.num_layers == 2


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
@pytest.mark.parametrize(
    "factory,defaults,parameter",
    [
        (
            TokenPositionEmbedding,
            {"vocab_size": 7, "context_length": 5, "embedding_dim": 8},
            "vocab_size",
        ),
        (
            TokenPositionEmbedding,
            {"vocab_size": 7, "context_length": 5, "embedding_dim": 8},
            "context_length",
        ),
        (
            TokenPositionEmbedding,
            {"vocab_size": 7, "context_length": 5, "embedding_dim": 8},
            "embedding_dim",
        ),
        (TransformerBlock, {"embedding_dim": 8, "num_heads": 2}, "embedding_dim"),
        (TransformerBlock, {"embedding_dim": 8, "num_heads": 2}, "num_heads"),
        (TinyTransformerLM, {"vocab_size": 7, "context_length": 5}, "vocab_size"),
        (TinyTransformerLM, {"vocab_size": 7, "context_length": 5}, "context_length"),
        (TinyTransformerLM, {"vocab_size": 7, "context_length": 5}, "embedding_dim"),
        (TinyTransformerLM, {"vocab_size": 7, "context_length": 5}, "num_heads"),
        (TinyTransformerLM, {"vocab_size": 7, "context_length": 5}, "num_layers"),
    ],
)
def test_constructor_rejects_invalid_dimensions(factory, defaults, parameter, value):
    kwargs = {**defaults, parameter: value}
    with pytest.raises(ValueError, match=parameter):
        factory(**kwargs)


@pytest.mark.parametrize("factory", [TransformerBlock, TinyTransformerLM])
def test_constructor_rejects_indivisible_head_dimensions(factory):
    kwargs = {"embedding_dim": 7, "num_heads": 2}
    if factory is TinyTransformerLM:
        kwargs.update(vocab_size=7, context_length=5)
    with pytest.raises(ValueError, match="divisible"):
        factory(**kwargs)


@pytest.mark.parametrize("module_name", ["embedding", "model"])
@pytest.mark.parametrize(
    "tokens",
    [
        None,
        [[0, 1]],
        torch.tensor([0, 1]),
        torch.zeros(1, 2, 3, dtype=torch.long),
        torch.zeros(2, 3),
        torch.zeros(2, 3, dtype=torch.int32),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.empty(2, 0, dtype=torch.long),
        torch.zeros(2, 6, dtype=torch.long),
        torch.tensor([[0, -1]]),
        torch.tensor([[0, 7]]),
    ],
)
def test_embedding_and_model_reject_invalid_ids(request, module_name, tokens):
    module = request.getfixturevalue(module_name)
    with pytest.raises(ValueError):
        module(tokens)


@pytest.mark.parametrize(
    "x",
    [
        None,
        [[1.0]],
        torch.zeros(2, 8),
        torch.zeros(2, 3, 8, 1),
        torch.zeros(2, 3, 8, dtype=torch.long),
        torch.zeros(2, 3, 8, dtype=torch.complex64),
        torch.empty(2, 0, 8),
        torch.zeros(2, 3, 7),
    ],
)
def test_block_rejects_invalid_features(block, x):
    with pytest.raises(ValueError):
        block(x)


def test_embedding_adds_exact_token_and_position_vectors():
    embedding = TokenPositionEmbedding(3, 4, 2).double()
    with torch.no_grad():
        embedding.token_embedding.weight.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]]))
        embedding.position_embedding.weight.copy_(
            torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])
        )
    tokens = torch.tensor([[0, 1, 0], [2, 0, 1]])
    expected = torch.tensor(
        [[[11, 22], [33, 44], [51, 62]], [[15, 26], [31, 42], [53, 64]]],
        dtype=torch.float64,
    )
    result = embedding(tokens)
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result, expected)


def test_embedding_handles_actual_lengths_and_restarts_positions(embedding):
    for length in (1, 5, 2):
        result = embedding(torch.zeros(3, length, dtype=torch.long))
        assert result.shape == (3, length, 8)
        torch.testing.assert_close(result[0], result[1])
        for position in range(length):
            expected = (
                embedding.token_embedding.weight[0]
                + embedding.position_embedding.weight[position]
            )
            torch.testing.assert_close(result[0, position], expected)


def test_embedding_backward_reaches_both_tables(embedding):
    embedding(torch.tensor([[0, 1, 2]])).square().mean().backward()
    for table in (embedding.token_embedding, embedding.position_embedding):
        assert table.weight.grad is not None
        assert torch.isfinite(table.weight.grad).all()
        assert table.weight.grad[:3].abs().sum() > 0


class _ProbeAttention(nn.Module):
    """A deterministic sublayer for checking block wiring, not attention math."""

    def forward(self, x):
        return 0.5 * x, None


class _ProbeFeedForward(nn.Module):
    def forward(self, x):
        return x.square()


def test_block_uses_pre_norm_and_updated_second_residual():
    block = TransformerBlock(4, 2).double()
    block.attention = _ProbeAttention()
    block.feed_forward = _ProbeFeedForward()
    x = torch.tensor([[[1, 2, 4, 8], [0, -2, 3, 1]]], dtype=torch.float64)
    original = x.clone()
    first_norm = F.layer_norm(x, (4,), eps=block.attention_norm.eps)
    residual = x + 0.5 * first_norm
    second_norm = F.layer_norm(residual, (4,), eps=block.feed_forward_norm.eps)
    expected = residual + second_norm.square()
    torch.testing.assert_close(block(x), expected)
    torch.testing.assert_close(x, original)


def test_block_zero_sublayers_preserve_identity_and_input_gradient(block):
    with torch.no_grad():
        block.attention.projection.weight.zero_()
        block.feed_forward[-1].weight.zero_()
        block.feed_forward[-1].bias.zero_()
    x = torch.randn(
        2, 3, 8, generator=torch.Generator().manual_seed(44), requires_grad=True
    )
    output = block(x)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_block_preserves_causality_and_batch_independence(block):
    x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(45))
    changed = x.clone()
    changed[0, 3:] += 10
    changed[1:] -= 10
    original = block(x)
    result = block(changed)
    assert result.shape == x.shape
    torch.testing.assert_close(result[0, :3], original[0, :3])


def test_block_backward_reaches_input_and_all_parameters(block):
    x = torch.randn(
        2, 4, 8, generator=torch.Generator().manual_seed(46), requires_grad=True
    )
    block(x).square().mean().backward()
    for tensor in (x, *block.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_model_outputs_raw_logits_at_every_position(model):
    with torch.no_grad():
        model.lm_head.weight.zero_()
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (2, 3, 7)
    assert logits.dtype == model.lm_head.weight.dtype
    assert logits.device == tokens.device
    # Softmax would turn zeros into 1/vocab_size, so this detects extra softmax.
    torch.testing.assert_close(logits, torch.zeros_like(logits))


def test_model_threads_features_through_each_layer_in_order(model):
    stages = [model.embedding, *model.blocks, model.final_norm, model.lm_head]
    outputs = []
    calls = []
    handles = []

    def pre_hook(index):
        def check_input(module, args):
            if index > 0:
                torch.testing.assert_close(args[0], outputs[-1])
            calls.append(index)

        return check_input

    def save_output(module, args, output):
        outputs.append(output.detach().clone())

    for index, stage in enumerate(stages):
        handles.append(stage.register_forward_pre_hook(pre_hook(index)))
        handles.append(stage.register_forward_hook(save_output))
    try:
        logits = model(torch.tensor([[0, 1, 2]]))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == list(range(len(stages)))
    torch.testing.assert_close(logits, outputs[-1])


def test_model_preserves_prefix_logits_and_batch_independence(model):
    tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    changed = tokens.clone()
    changed[0, 3:] = torch.tensor([5, 6])
    changed[1] = 6
    logits = model(tokens)
    torch.testing.assert_close(model(changed)[0, :3], logits[0, :3])
    torch.testing.assert_close(model(tokens[:, :3]), logits[:, :3])


def test_model_supports_single_position_and_full_context(model):
    for length in (1, 5, 2):
        logits = model(torch.zeros(2, length, dtype=torch.long))
        assert logits.shape == (2, length, 7)
        assert torch.isfinite(logits).all()


def test_model_backward_reaches_all_registered_parameters(model):
    # IDs are integer labels and do not require gradients; embeddings do.
    model(torch.tensor([[0, 1, 2], [3, 4, 5]])).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
