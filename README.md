# Build a Tiny Transformer

Learn a GPT-style character-level text generator by implementing one milestone
at a time. This repository currently contains **milestone 1 exercises**, not a
finished model. You write the core functions; we review them before adding attention.

## Setup

Python 3.12 and `uv` are required. Run commands from the repository root:

```bash
uv sync
uv run pytest -m "not exercise"
uv run ruff check .
```

`uv sync` installs the package, NumPy, CPU-only PyTorch, pytest, and Ruff into `.venv/`.
NumPy enables PyTorch's tensor/array integration without initialization warnings.
The CPU index configuration follows the
[official uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
No GPU or dataset download is needed; initial setup needs package-index access.
`uv sync` updates `uv.lock`; commit it to fix dependency versions.
Use four spaces for indentation; `uv run ruff format .` formats Python files.

## Milestone 1: text to tensors

A language model predicts the **next token** at each position. For this project,
a token is one character, including spaces and newlines. A vocabulary maps each
character to an integer; these integers are labels, not numerical measurements.
Embeddings will turn them into learned vectors in a later milestone.

For the text `hello`, a context length of four gives:

```text
input x:  h e l l
target y: e l l o
```

Each position has its own target. Later, causal attention will prevent the model
from seeing future input characters while predicting that target.

Edit `src/transformer/data.py` in this order:

1. **Vocabulary:** implement `from_text`. Collect unique characters, sort them,
   and store them as a tuple. Reject empty training text with `ValueError`.
2. **Encoding and decoding:** implement `encode` and `decode`. For the vocabulary
   `('a', 'b', 'c')`, `encode('cab')` must return `[2, 0, 1]`.
   Reject unknown characters and IDs outside `0 <= id < vocab_size` with
   `ValueError`. Encoding or decoding an empty sequence returns an empty result.
3. **Batches:** implement `make_batch`. Validate the inputs described in its
   docstring, sample starting positions with replacement, slice inputs and
   targets, and stack the slices into two CPU `torch.long` tensors.
   Pass the optional generator into the random sampling operation.

Hints: `set`, `sorted`, and dictionaries help with vocabulary lookup.
`torch.randint` samples positions and `torch.stack` creates a batch dimension.
Its upper bound is exclusive: derive it so the final legal starting position
is included. A length `N` sequence needs `T + 1` characters for a context of `T`.

The output shape is `[B, T]`: `B` independent training examples, each with `T`
positions. Defaults in the inspection command are `B = 16` and `T = 32`.

## Check your work

```bash
uv run pytest -m exercise -x   # stop at the first unfinished behavior
uv run pytest                # all setup and exercise checks
uv run python -m transformer.inspect_data
```

Exercise tests intentionally fail with `NotImplementedError` until you implement
the functions. Setup checks should pass immediately. The inspection command
prints vocabulary size, tensor shapes, and one decoded input/target pair.
You can inspect another file with `--text path/to/file.txt`.

Before moving on, explain why targets shift by one position, why token IDs use
`torch.long`, and how many legal starts exist for `N = 10`, `T = 4`.
Share your implementation and any failing test output for review.

## Following milestones

After reviewing milestone 1, we will implement single-head causal attention,
multi-head attention, then embeddings and Transformer blocks. Next come training
with a contiguous train/validation split, a fixed-batch overfitting check, and
temperature-based generation with checkpoints. The initial model will use
64-dimensional embeddings, four heads, and two pre-normalized blocks.

Source lives in `src/transformer/`, automated checks in `tests/`, and an original
toy story in `data/toy.txt`. The story is deliberately tiny: it teaches the
pipeline, but fluent generation is not a goal for this corpus.
