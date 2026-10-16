"""Regression checks for the completed milestone 2 attention head."""

import math

import pytest
import torch

from transformer.attention import CausalSelfAttentionHead

pytestmark = pytest.mark.milestone2


@pytest.fixture
def head():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return CausalSelfAttentionHead(6, 3)


def test_constructor_registers_three_trainable_bias_free_projections(head):
    assert head.embedding_dim == 6
    assert head.head_dim == 3
    for projection in (head.query, head.key, head.value):
        assert projection.weight.shape == (3, 6)
        assert projection.bias is None
        assert projection.weight.requires_grad
    assert set(dict(head.named_parameters())) == {
        "query.weight",
        "key.weight",
        "value.weight",
    }


@pytest.mark.parametrize("dimension", [0, -1, 1.5, True, "3", None])
@pytest.mark.parametrize("parameter", ["embedding_dim", "head_dim"])
def test_constructor_rejects_non_positive_or_non_integer_dimensions(
    parameter, dimension
):
    dimensions = {"embedding_dim": 6, "head_dim": 3}
    dimensions[parameter] = dimension
    with pytest.raises(ValueError, match=parameter):
        CausalSelfAttentionHead(**dimensions)


@pytest.mark.parametrize(
    "x",
    [
        None,
        [[[1.0]]],
        torch.ones(6),
        torch.ones(4, 6),
        torch.ones(1, 2, 4, 6),
        torch.ones(2, 4, 6, dtype=torch.long),
        torch.ones(2, 4, 6, dtype=torch.bool),
        torch.ones(2, 4, 6, dtype=torch.complex64),
        torch.empty(2, 0, 6),
        torch.ones(2, 4, 5),
    ],
)
def test_forward_rejects_invalid_features(head, x):
    with pytest.raises(ValueError):
        head(x)


def test_output_shapes_and_attention_probability_rules(head):
    x = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(5))
    output, weights = head(x)
    assert output.shape == (2, 4, 3)
    assert weights.shape == (2, 4, 4)
    assert output.dtype == weights.dtype == x.dtype
    assert output.device == weights.device == x.device
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4))
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
    assert (weights.diagonal(dim1=-2, dim2=-1) > 0).all()


def test_attention_matches_hand_calculated_scaled_masked_example():
    head = CausalSelfAttentionHead(3, 2).double()
    with torch.no_grad():
        head.query.weight.copy_(torch.tensor([[1, 0, 0], [0, 1, 0]]))
        head.key.weight.copy_(torch.tensor([[0, 1, 0], [0, 0, 1]]))
        head.value.weight.copy_(torch.tensor([[0, 0, 1], [1, 0, -1]]))
    x = torch.tensor([[[1, 0, 1], [0, 2, 1], [1, 1, 0]]], dtype=torch.float64)

    # Allowed unscaled scores: [0], [2, 2], [1, 3, 1].
    # Last row's middle score exceeds the others by 2 / sqrt(2).
    ratio = math.exp(math.sqrt(2))
    a, b = 1 / (2 + ratio), ratio / (2 + ratio)
    expected_weights = x.new_tensor([[[1, 0, 0], [0.5, 0.5, 0], [a, b, a]]])
    # Projected values are [1, 0], [1, -1], [0, 1].
    expected_output = x.new_tensor([[[1, 0], [1, -0.5], [a + b, a - b]]])
    output, weights = head(x)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


def test_changing_future_features_cannot_change_past_outputs(head):
    x = torch.randn(2, 5, 6, generator=torch.Generator().manual_seed(8))
    changed = x.clone()
    changed[:, 3:] += 100
    output, weights = head(x)
    changed_output, changed_weights = head(changed)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])
    torch.testing.assert_close(weights[:, :3], changed_weights[:, :3])


def test_batch_examples_do_not_influence_each_other(head):
    x = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(9))
    changed = x.clone()
    changed[1] += 100
    output, weights = head(x)
    changed_output, changed_weights = head(changed)
    torch.testing.assert_close(output[0], changed_output[0])
    torch.testing.assert_close(weights[0], changed_weights[0])


def test_one_position_attends_to_itself_and_returns_its_value(head):
    x = torch.randn(2, 1, 6, generator=torch.Generator().manual_seed(10))
    output, weights = head(x)
    torch.testing.assert_close(weights, torch.ones(2, 1, 1))
    torch.testing.assert_close(output, head.value(x))


def test_one_head_handles_different_sequence_lengths(head):
    for length in (4, 1, 7, 2):
        output, weights = head(torch.zeros(2, length, 6))
        assert output.shape == (2, length, 3)
        assert weights.shape == (2, length, length)
        # Zero features give equal scores over every allowed prefix.
        for position in range(length):
            torch.testing.assert_close(
                weights[:, position, : position + 1],
                torch.full((2, position + 1), 1 / (position + 1)),
            )
        assert torch.count_nonzero(weights.triu(diagonal=1)) == 0


def test_backward_reaches_input_and_all_projection_parameters(head):
    x = torch.randn(
        2, 4, 6, generator=torch.Generator().manual_seed(11), requires_grad=True
    )
    output, weights = head(x)
    assert weights.requires_grad
    loss = output.square().mean() + weights.square().mean()
    loss.backward()
    for tensor in (x, *head.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
