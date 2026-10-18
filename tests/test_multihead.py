"""Regression checks for completed milestone 3 multi-head attention."""

import math

import pytest
import torch
from torch import nn

from transformer.attention import CausalSelfAttentionHead, MultiHeadCausalSelfAttention

pytestmark = pytest.mark.milestone3


@pytest.fixture
def attention():
    with torch.random.fork_rng():
        torch.manual_seed(23)
        return MultiHeadCausalSelfAttention(8, 2)


def test_constructor_registers_independent_heads_and_output_projection(attention):
    assert attention.embedding_dim == 8
    assert attention.num_heads == 2
    assert attention.head_dim == 4
    assert isinstance(attention.heads, nn.ModuleList)
    assert len(attention.heads) == 2
    for head in attention.heads:
        assert isinstance(head, CausalSelfAttentionHead)
        assert head.embedding_dim == 8
        assert head.head_dim == 4
    parameters = dict(attention.named_parameters())
    assert set(parameters) == {
        "heads.0.query.weight",
        "heads.0.key.weight",
        "heads.0.value.weight",
        "heads.1.query.weight",
        "heads.1.key.weight",
        "heads.1.value.weight",
        "projection.weight",
    }
    assert all(parameter.requires_grad for parameter in parameters.values())
    assert len({parameter.data_ptr() for parameter in parameters.values()}) == 7
    before = attention.heads[1].query.weight.detach().clone()
    with torch.no_grad():
        attention.heads[0].query.weight.add_(1)
    torch.testing.assert_close(attention.heads[1].query.weight, before)
    assert attention.projection.weight.shape == (8, 8)
    assert attention.projection.bias is None


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, "2", None])
@pytest.mark.parametrize("parameter", ["embedding_dim", "num_heads"])
def test_constructor_rejects_non_positive_or_non_integer_dimensions(
    parameter, dimension
):
    dimensions = {"embedding_dim": 8, "num_heads": 2}
    dimensions[parameter] = dimension
    with pytest.raises(ValueError, match=parameter):
        MultiHeadCausalSelfAttention(**dimensions)


@pytest.mark.parametrize("embedding_dim,num_heads", [(7, 2), (2, 4)])
def test_constructor_rejects_indivisible_dimensions(embedding_dim, num_heads):
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadCausalSelfAttention(embedding_dim, num_heads)


@pytest.mark.parametrize("embedding_dim,num_heads", [(8, 1), (4, 4), (1, 1)])
def test_constructor_supports_one_head_and_unit_head_width(embedding_dim, num_heads):
    attention = MultiHeadCausalSelfAttention(embedding_dim, num_heads)
    assert len(attention.heads) == num_heads
    assert attention.head_dim == embedding_dim // num_heads


@pytest.mark.parametrize(
    "x",
    [
        None,
        [[[1.0]]],
        torch.ones(8),
        torch.ones(4, 8),
        torch.ones(1, 2, 4, 8),
        torch.ones(2, 4, 8, dtype=torch.long),
        torch.ones(2, 4, 8, dtype=torch.bool),
        torch.ones(2, 4, 8, dtype=torch.complex64),
        torch.empty(2, 0, 8),
        torch.ones(2, 4, 7),
    ],
)
def test_forward_rejects_invalid_features(attention, x):
    with pytest.raises(ValueError):
        attention(x)


def test_output_shapes_and_per_head_probability_rules(attention):
    # Distinct batch, head, and sequence sizes catch incorrect stack dimensions.
    x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(31))
    output, weights = attention(x)
    assert output.shape == (3, 5, 8)
    assert weights.shape == (3, 2, 5, 5)
    assert output.dtype == weights.dtype == x.dtype
    assert output.device == weights.device == x.device
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, 2, 5))
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    assert (weights.diagonal(dim1=-2, dim2=-1) > 0).all()
    for index, head in enumerate(attention.heads):
        _, expected_weights = head(x)
        torch.testing.assert_close(weights[:, index], expected_weights)


def test_fixed_example_preserves_head_order_and_applies_output_projection():
    attention = MultiHeadCausalSelfAttention(4, 2).double()
    with torch.no_grad():
        first, second = attention.heads
        first.query.weight.zero_()
        first.key.weight.zero_()
        first.value.weight.copy_(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]))
        second.query.weight.copy_(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]))
        second.key.weight.copy_(second.query.weight)
        second.value.weight.copy_(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]]))
        attention.projection.weight.copy_(
            torch.tensor([[1, 0, 2, 0], [0, 1, 0, 3], [1, 1, 0, 0], [0, 0, 1, -1]])
        )
    x = torch.tensor([[[1, 0, 2, 0], [0, 1, 0, 3]]], dtype=torch.float64)
    p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    # Head 0 reads uniformly; head 1's last row is [1-p, p].
    expected_weights = x.new_tensor([[[[1, 0], [0.5, 0.5]], [[1, 0], [1 - p, p]]]])
    # Concatenated last row: [0.5, 0.5, 2*(1-p), 3*p].
    expected_output = x.new_tensor(
        [[[5, 0, 1, 2], [0.5 + 4 * (1 - p), 0.5 + 9 * p, 1, 2 - 5 * p]]]
    )
    output, weights = attention(x)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_single_head_with_identity_projection_matches_completed_head():
    attention = MultiHeadCausalSelfAttention(4, 1)
    with torch.no_grad():
        attention.projection.weight.copy_(torch.eye(4))
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(32))
    expected_output, expected_weights = attention.heads[0](x)
    output, weights = attention(x)
    assert weights.shape == (2, 1, 3, 3)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weights[:, 0], expected_weights)


def test_changing_future_features_cannot_change_past_outputs(attention):
    x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(33))
    changed = x.clone()
    changed[:, 3:] += 100
    output, weights = attention(x)
    changed_output, changed_weights = attention(changed)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])
    torch.testing.assert_close(weights[:, :, :3], changed_weights[:, :, :3])


def test_batch_examples_do_not_influence_each_other(attention):
    x = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(34))
    changed = x.clone()
    changed[1:] += 100
    output, weights = attention(x)
    changed_output, changed_weights = attention(changed)
    torch.testing.assert_close(output[0], changed_output[0])
    torch.testing.assert_close(weights[0], changed_weights[0])


def test_one_position_combines_own_values_through_projection(attention):
    x = torch.randn(3, 1, 8, generator=torch.Generator().manual_seed(35))
    output, weights = attention(x)
    assert output.shape == (3, 1, 8)
    torch.testing.assert_close(weights, torch.ones(3, 2, 1, 1))
    # At T=1 each head returns its value. Check each projected feature block
    # separately, without reproducing the multi-head concatenation exercise.
    expected = torch.zeros_like(output)
    for index, head in enumerate(attention.heads):
        start = index * attention.head_dim
        block = attention.projection.weight[:, start : start + attention.head_dim]
        expected += torch.nn.functional.linear(head.value(x), block)
    torch.testing.assert_close(output, expected)


def test_same_module_handles_different_sequence_lengths(attention):
    for length in (4, 1, 7, 2):
        output, weights = attention(torch.zeros(3, length, 8))
        assert output.shape == (3, length, 8)
        assert weights.shape == (3, 2, length, length)
        torch.testing.assert_close(output, torch.zeros_like(output))
        for position in range(length):
            torch.testing.assert_close(
                weights[:, :, position, : position + 1],
                torch.full((3, 2, position + 1), 1 / (position + 1)),
            )
        assert torch.count_nonzero(weights.triu(diagonal=1)) == 0


def test_backward_reaches_input_all_heads_and_output_projection(attention):
    x = torch.randn(
        3, 4, 8, generator=torch.Generator().manual_seed(36), requires_grad=True
    )
    output, weights = attention(x)
    assert weights.requires_grad
    (output.square().mean() + weights.square().mean()).backward()
    for tensor in (x, *attention.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
