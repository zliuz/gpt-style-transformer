"""Milestone 5 contracts; completed infrastructure passes without the exercises."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from transformer.data import make_batch
from transformer.model import TinyTransformerLM
from transformer.training import (
    evaluate_loss,
    next_token_loss,
    split_tokens,
    train_step,
)

pytestmark = pytest.mark.milestone5


@pytest.mark.parametrize(
    "tokens",
    [None, [0, 1], torch.ones(30), torch.ones(3, 10, dtype=torch.long)],
)
def test_split_rejects_invalid_token_tensors(tokens):
    with pytest.raises(ValueError):
        split_tokens(tokens, context_length=2)


@pytest.mark.parametrize(
    "fraction", [0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.9"]
)
def test_split_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="train_fraction"):
        split_tokens(torch.arange(100), fraction, context_length=2)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_split_rejects_invalid_context_lengths(length):
    with pytest.raises(ValueError, match="context_length"):
        split_tokens(torch.arange(100), context_length=length)


@pytest.mark.parametrize("count,fraction", [(0, 0.5), (5, 0.5), (10, 0.9), (10, 0.1)])
def test_split_rejects_partitions_too_short_for_input_and_target(count, fraction):
    with pytest.raises(ValueError, match="Both partitions"):
        split_tokens(torch.arange(count), fraction, context_length=2)


@pytest.mark.parametrize(
    "logits,targets",
    [
        (None, torch.zeros(2, 3, dtype=torch.long)),
        (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.long)),
        (torch.zeros(2, 3, 4, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long)),
        (
            torch.zeros(2, 3, 4, dtype=torch.complex64),
            torch.zeros(2, 3, dtype=torch.long),
        ),
        (torch.zeros(0, 3, 4), torch.zeros(0, 3, dtype=torch.long)),
        (torch.zeros(2, 0, 4), torch.zeros(2, 0, dtype=torch.long)),
        (torch.zeros(2, 3, 0), torch.zeros(2, 3, dtype=torch.long)),
        (torch.zeros(2, 3, 4), None),
        (torch.zeros(2, 3, 4), torch.zeros(2, 3)),
        (torch.zeros(2, 3, 4), torch.zeros(2, 2, dtype=torch.long)),
        (torch.zeros(2, 3, 4), torch.zeros(6, dtype=torch.long)),
        (torch.zeros(2, 3, 4), torch.full((2, 3), -1)),
        (torch.zeros(2, 3, 4), torch.full((2, 3), 4)),
    ],
)
def test_loss_rejects_invalid_shapes_types_and_labels(logits, targets):
    with pytest.raises(ValueError):
        next_token_loss(logits, targets)


@pytest.fixture
def lookup_model():
    # A simple known-logit model makes update/evaluation reference math independent
    # of the Transformer and lets us inspect mode and autograd state on forward.
    class LookupModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(
                torch.tensor([[1.0, 0, -1], [0, 2, -1], [-1, 0, 1]])
            )
            self.calls = []

        def forward(self, x):
            self.calls.append((self.training, torch.is_grad_enabled()))
            return self.table[x]

    return LookupModel()


@pytest.mark.parametrize(
    "x,y",
    [
        (None, torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, dtype=torch.long), None),
        (torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long)),
        (torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2)),
        (torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(1, 2, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long)),
        (torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)),
    ],
)
def test_step_and_evaluation_reject_invalid_batches_before_mutation(lookup_model, x, y):
    lookup_model.eval()
    optimizer = torch.optim.SGD(lookup_model.parameters(), lr=0.1)
    before = lookup_model.table.detach().clone()
    with pytest.raises(ValueError):
        train_step(lookup_model, optimizer, x, y)
    with pytest.raises(ValueError):
        evaluate_loss(lookup_model, [(x, y)])
    assert not lookup_model.training
    assert not lookup_model.calls
    torch.testing.assert_close(lookup_model.table, before)


@pytest.mark.parametrize("batches", [None, [], (), [()], [(torch.ones(1, 1),)], [None]])
def test_evaluation_rejects_invalid_batch_collections(lookup_model, batches):
    with pytest.raises(ValueError):
        evaluate_loss(lookup_model, batches)


@pytest.mark.exercise
@pytest.mark.parametrize(
    "count,fraction,length", [(20, 0.6, 3), (6, 0.5, 2), (21, 0.6, 3)]
)
def test_split_preserves_exact_order_and_sampling_boundaries(count, fraction, length):
    tokens = torch.arange(count)
    before = tokens.clone()
    train, validation = split_tokens(tokens, fraction, context_length=length)
    boundary = int(count * fraction)
    torch.testing.assert_close(train, tokens[:boundary])
    torch.testing.assert_close(validation, tokens[boundary:])
    torch.testing.assert_close(torch.cat((train, validation)), before)
    torch.testing.assert_close(tokens, before)
    for partition, lower, upper in (
        (train, 0, boundary),
        (validation, boundary, count),
    ):
        x, y = make_batch(
            partition, 50, length, generator=torch.Generator().manual_seed(5)
        )
        assert x.min() >= lower and y.max() < upper
        torch.testing.assert_close(y, x + 1)


@pytest.mark.exercise
def test_loss_matches_hand_calculated_mean_over_all_positions():
    probabilities = torch.tensor(
        [[[0.25, 0.75], [0.8, 0.2]], [[0.6, 0.4], [0.1, 0.9]]], dtype=torch.float64
    )
    loss = next_token_loss(probabilities.log(), torch.tensor([[0, 1], [0, 1]]))
    expected = -sum(math.log(p) for p in (0.25, 0.2, 0.6, 0.9)) / 4
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(expected)


@pytest.mark.exercise
def test_uniform_logits_have_log_vocabulary_loss():
    assert next_token_loss(
        torch.zeros(2, 3, 7), torch.zeros(2, 3, dtype=torch.long)
    ).item() == pytest.approx(math.log(7))


@pytest.mark.exercise
def test_loss_handles_noncontiguous_inputs_and_correct_gradient():
    logits = (
        torch.tensor(
            [[[1.0, 0, -1], [0, 2, -1]], [[-1, 0, 1], [2, 0, 1]]],
            dtype=torch.float64,
        )
        .transpose(0, 1)
        .requires_grad_()
    )
    targets = torch.tensor([[0, 1], [2, 0]]).transpose(0, 1)
    assert not logits.is_contiguous()
    loss = next_token_loss(logits, targets)
    loss.backward()
    expected = (logits.detach().softmax(-1) - F.one_hot(targets, 3)) / targets.numel()
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.exercise
def test_train_step_clears_old_gradients_and_matches_reference_update(lookup_model):
    reference = copy.deepcopy(lookup_model)
    x, y = torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1, 2], [0, 0]])
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference_loss = F.cross_entropy(reference(x).movedim(-1, 1), y)
    reference_loss.backward()
    reference_optimizer.step()

    lookup_model.eval()
    lookup_model.table.grad = torch.full_like(lookup_model.table, 99)
    before = lookup_model.table.detach().clone()
    optimizer = torch.optim.SGD(lookup_model.parameters(), lr=0.1)
    loss = train_step(lookup_model, optimizer, x, y)
    assert type(loss) is float
    assert loss == pytest.approx(reference_loss.item())
    assert lookup_model.calls == [(True, True)]
    assert lookup_model.training
    assert not torch.equal(lookup_model.table, before)
    torch.testing.assert_close(lookup_model.table, reference.table)


@pytest.mark.exercise
@pytest.mark.parametrize("initial_training", [True, False])
def test_evaluation_weights_by_token_count_and_preserves_state(
    lookup_model, initial_training
):
    batches = [
        (torch.tensor([[0]]), torch.tensor([[2]])),
        (torch.tensor([[1, 2], [0, 1]]), torch.tensor([[1, 2], [0, 1]])),
    ]
    # Compute an independent summed reference without invoking the model.
    summed = sum(
        F.cross_entropy(lookup_model.table[x].movedim(-1, 1), y, reduction="sum").item()
        for x, y in batches
    )
    expected = summed / 5
    lookup_model.train(initial_training)
    lookup_model.table.grad = torch.full_like(lookup_model.table, 7)
    before = lookup_model.table.detach().clone()
    old_grad = lookup_model.table.grad.clone()
    loss = evaluate_loss(lookup_model, batches)
    assert type(loss) is float
    assert loss == pytest.approx(expected)
    assert lookup_model.calls == [(False, False), (False, False)]
    assert lookup_model.training is initial_training
    torch.testing.assert_close(lookup_model.table, before)
    torch.testing.assert_close(lookup_model.table.grad, old_grad)


@pytest.mark.exercise
@pytest.mark.parametrize("initial_training", [True, False])
@pytest.mark.parametrize("failure", ["forward", "loss"])
def test_evaluation_restores_mode_when_forward_or_loss_raises(
    lookup_model, monkeypatch, initial_training, failure
):
    from transformer import training

    def fail(*args, **kwargs):
        assert not lookup_model.training
        assert not torch.is_grad_enabled()
        raise LookupError("intentional evaluation failure")

    if failure == "forward":
        monkeypatch.setattr(lookup_model, "forward", fail)
    else:
        monkeypatch.setattr(training, "next_token_loss", fail)
    lookup_model.train(initial_training)
    with pytest.raises(LookupError, match="intentional evaluation failure"):
        evaluate_loss(lookup_model, [(torch.tensor([[0]]), torch.tensor([[1]]))])
    assert lookup_model.training is initial_training


@pytest.mark.exercise
def test_real_transformer_can_reduce_fixed_batch_loss():
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(47)
            model = TinyTransformerLM(3, 4, embedding_dim=8, num_heads=2, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        batch = (
            torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]),
            torch.tensor([[1, 2, 0, 1], [2, 0, 1, 2]]),
        )
        before = evaluate_loss(model, [batch])
        for _ in range(40):
            assert math.isfinite(train_step(model, optimizer, *batch))
        after = evaluate_loss(model, [batch])
        assert math.isfinite(after)
        assert after < before * 0.7
    finally:
        torch.set_num_threads(previous_threads)
