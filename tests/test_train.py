"""Checks for the supplied CLI/sampling control flow, independent of exercises."""

import pytest
import torch

import transformer.train as runner
from transformer.model import TinyTransformerLM

pytestmark = pytest.mark.milestone5


def test_parser_defaults_match_documented_run():
    args = runner.build_parser().parse_args([])
    assert (args.mode, args.steps, args.batch_size, args.context_length) == (
        "train",
        200,
        4,
        16,
    )
    assert args.learning_rate == 0.003
    assert args.seed == 42


@pytest.mark.parametrize(
    "option,value",
    [
        ("--steps", "0"),
        ("--steps", "1.5"),
        ("--batch-size", "-1"),
        ("--context-length", "0"),
        ("--learning-rate", "0"),
        ("--learning-rate", "nan"),
        ("--learning-rate", "inf"),
        ("--seed", "-1"),
        ("--seed", str(2**63)),
        ("--mode", "generate"),
    ],
)
def test_parser_rejects_invalid_run_options(option, value):
    with pytest.raises(SystemExit) as error:
        runner.build_parser().parse_args([option, value])
    assert error.value.code == 2


def test_parser_accepts_zero_seed():
    assert runner.build_parser().parse_args(["--seed", "0"]).seed == 0


def test_cached_evaluation_batches_are_reproducible_and_rng_independent():
    before_global = torch.random.get_rng_state().clone()
    kwargs = {"batch_size": 2, "context_length": 4, "seed": 42}
    first = runner.make_evaluation_batches(
        torch.arange(80), torch.arange(80, 100), **kwargs
    )
    second = runner.make_evaluation_batches(
        torch.arange(80), torch.arange(80, 100), **kwargs
    )
    assert set(first) == {"train", "validation"}
    for name in first:
        assert len(first[name]) == 5
        for a, b in zip(first[name], second[name]):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
            x, y = a
            assert x.min() >= (0 if name == "train" else 80)
            assert y.max() < (80 if name == "train" else 100)
    assert torch.equal(torch.random.get_rng_state(), before_global)


@pytest.mark.parametrize("mode", ["train", "overfit"])
def test_runner_control_flow_and_sampling_are_reproducible(monkeypatch, capsys, mode):
    # Stub the exercise functions only to inspect the provided orchestration.
    # These stubs supply no loss, backward, or parameter-update implementation.
    records = []
    reports = []
    optimizer_details = []

    def fake_step(model, optimizer, x, y):
        records.append((x.clone(), y.clone()))
        optimizer_details.append(optimizer.param_groups[0])
        return 1.0

    def fake_evaluate(model, batches):
        reports.append(batches)
        return 1.0

    monkeypatch.setattr(runner, "train_step", fake_step)
    monkeypatch.setattr(runner, "evaluate_loss", fake_evaluate)
    args = runner.build_parser().parse_args(
        ["--mode", mode, "--steps", "51", "--context-length", "4"]
    )
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = TinyTransformerLM(3, 4, embedding_dim=8, num_heads=2, num_layers=1)
    train, validation = torch.arange(80), torch.arange(80, 100)
    runner.run_training(model, train, validation, args)
    first = list(records)
    assert len(first) == 51
    expected_first = runner.make_batch(
        train,
        args.batch_size,
        args.context_length,
        generator=torch.Generator().manual_seed(args.seed),
    )
    assert all(torch.equal(a, b) for a, b in zip(first[0], expected_first))
    assert all(item[0].min() >= 0 and item[1].max() < 80 for item in first)
    assert optimizer_details[0]["lr"] == 0.003
    assert optimizer_details[0]["weight_decay"] == 0
    output = capsys.readouterr().out
    if mode == "overfit":
        assert len(reports) == 2
        assert reports[0][0] is reports[1][0]
        assert all(torch.equal(item[0], first[0][0]) for item in first)
        assert "不证明泛化" in output
    else:
        assert len(reports) == 6  # train/validation at steps 0, 50, 51
        assert reports[0] is reports[2] is reports[4]
        assert reports[1] is reports[3] is reports[5]
        assert all(f"Step {step}:" in output for step in (0, 50, 51))
    records.clear()
    runner.run_training(model, train, validation, args)
    for a, b in zip(first, records):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
