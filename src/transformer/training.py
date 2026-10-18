"""Milestone 5: implement splitting, loss, one update, and evaluation in order."""

import math

import torch
from torch import nn


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    for name, tensor in (("x", x), ("y", y)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.dtype != torch.long
            or tensor.device.type != "cpu"
            or tensor.numel() == 0
        ):
            raise ValueError(f"{name} must be a nonempty 2D CPU torch.long tensor.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same [B, T] shape.")


def split_tokens(
    tokens: torch.Tensor,
    train_fraction: float = 0.9,
    *,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise 1: return a contiguous train prefix and validation suffix.

    Require 1D CPU torch.long tokens, a finite fraction strictly between 0 and 1,
    and positive integer context_length. Split at int(N * train_fraction).
    Each partition needs at least context_length + 1 tokens, else ValueError.
    Do not shuffle, overlap partitions, or change the input tensor.
    """
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 1
        or tokens.dtype != torch.long
        or tokens.device.type != "cpu"
    ):
        raise ValueError("tokens must be a one-dimensional CPU torch.long tensor.")
    _positive_integer("context_length", context_length)
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not math.isfinite(train_fraction)
        or not 0 < train_fraction < 1
    ):
        raise ValueError("train_fraction must be finite and strictly between 0 and 1.")
    split_index = int(tokens.numel() * train_fraction)
    if min(split_index, tokens.numel() - split_index) < context_length + 1:
        raise ValueError("Both partitions need at least context_length + 1 tokens.")
    # Return two slices at split_index, preserving the original order.
    return tokens[:split_index], tokens[split_index:]

def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exercise 2: mean cross-entropy over ALL aligned positions, without softmax.

    Accept nonempty floating logits [B, T, V] and matching torch.long targets
    [B, T] on the same device, with 0 <= target < V; otherwise ValueError.
    Return a scalar tensor retaining its computation graph. make_batch already
    shifted the targets: do not shift them again here.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or not logits.is_floating_point()
        or logits.numel() == 0
    ):
        raise ValueError("logits must be nonempty floating features [B, T, V].")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.ndim != 2
        or targets.dtype != torch.long
        or targets.shape != logits.shape[:2]
        or targets.device != logits.device
    ):
        raise ValueError(
            "targets must be matching torch.long IDs [B, T] on the same device."
        )
    if ((targets < 0) | (targets >= logits.shape[-1])).any():
        raise ValueError("Target IDs must satisfy 0 <= id < vocab_size.")
    # Flatten B and T while retaining V as the class axis (hint: reshape).
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    # Use torch.nn.functional.cross_entropy with mean reduction on raw logits.
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    
    # Return the scalar tensor, not .item(); backward needs the graph.
    return loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Exercise 3: perform exactly one update; return the pre-update loss float.

    x/y must be nonempty matching 2D CPU torch.long tensors. Token range and
    context checks are delegated to the model and next_token_loss. Leave the
    model in training mode. The optimizer is supplied by the caller.
    """
    _validate_batch(x, y)
    # 1. model.train(), then optimizer.zero_grad() (set_to_none=True is fine).
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # 2. Run model(x), then next_token_loss(logits, y).
    logits = model(x)
    loss = next_token_loss(logits, y)

    # 3. Backpropagate the loss, then optimizer.step().
    loss.backward()
    optimizer.step()

    # 4. Return the pre-update loss as a Python float, without another forward.
    return loss.item()


def evaluate_loss(
    model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> float:
    """Exercise 4: return token-weighted mean loss, restoring the original mode.

    Accept a nonempty list of (x, y) batches, each satisfying train_step's tensor
    contract. Batch sizes and sequence lengths may differ. Do not change model
    parameters or existing .grad values. Restore model.training even if forward
    or loss raises; input validation itself must not change the mode.
    """
    if not isinstance(batches, list) or not batches:
        raise ValueError("batches must be a nonempty list of (x, y) pairs.")
    for batch in batches:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("Each batch must contain exactly (x, y).")
        _validate_batch(*batch)
    # 1. Remember model.training, then model.eval().
    original_training_mode = model.training
    try:
        model.eval()

        # 2. Under torch.no_grad(), get each batch's mean next_token_loss.
        total_loss = 0.0
        total_tokens = 0

        # 3. Weight each mean by y.numel(); divide total loss by total token count.
        with torch.no_grad():
            for x, y in batches:
                logits = model(x)
                loss = next_token_loss(logits, y)
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        mean_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")

        # Return a Python float; do not clear gradients or update parameters.
        return mean_loss
    finally:
        model.train(original_training_mode)
