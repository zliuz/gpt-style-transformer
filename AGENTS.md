# Repository Guidelines

## Purpose and Learning Workflow

This is a guided, character-level GPT-style Transformer learning project. Milestones 1–4 (data, attention, embeddings, blocks, and model forward) are implemented and reviewed. Milestone 5 provides four training exercise scaffolds and a complete sampling/reporting runner; splitting, loss, updates, and evaluation remain intentionally unfinished.

Default to scaffolds, incremental hints, tests, and conceptual review. Only supply or fill in the exercise's core solution when the user explicitly requests an answer or complete implementation. Preserve intentional `NotImplementedError` exercises when preparing learning material. Do not bypass exercise checks by skipping, deleting, or marking them xfail. After a learner completes a milestone, review the implementation and update its status and test markers before advancing.

## Project Structure

- `src/transformer/`: package source; `data.py` contains completed milestone 1, `attention.py` contains completed single/multi-head attention, `model.py` contains the completed model; `training.py` contains milestone 5 exercises and `train.py` provides the runnable training control flow, and `inspect_data.py` / `inspect_attention.py` / `inspect_multihead.py` / `inspect_model.py` are runnable teaching entry points.
- `tests/`: pytest checks mirroring source responsibilities.
- `data/toy.txt`: original tiny teaching corpus, not intended for fluent generation.
- `README.md`: learner-facing setup, concepts, exercises, validation, and roadmap; keep explanations primarily in Chinese with English code identifiers.
- `pyproject.toml` and `uv.lock`: tool configuration and locked dependencies.

The `.agents/`, `.codex/`, `.aws/`, and `.git/` directories are reserved for environment metadata, not application code. Add `assets/` only when static resources are needed.

## Setup, Development, and Build Commands

Use Python 3.12, uv, NumPy, CPU-only PyTorch, pytest, and Ruff. Run from the repository root:

```bash
uv sync --locked
uv run pytest -m "not exercise"
uv run pytest -m milestone2 -x
uv run pytest -m milestone3 -x
uv run pytest -m milestone4 -x
uv run pytest -m milestone5 -x
uv run pytest -m "milestone5 and exercise" -x
uv run pytest
uv run python -m transformer.inspect_data
uv run python -m transformer.inspect_attention
uv run python -m transformer.inspect_multihead
uv run python -m transformer.inspect_model
uv run python -m transformer.train --mode overfit
uv run python -m transformer.train
uv run ruff check .
uv run ruff format .
uv build
```

Initial dependency setup needs package-index access. No GPU or downloaded dataset is required. `uv build` writes distribution artifacts to `dist/`. Keep dependency changes intentional, update `uv.lock` when dependencies change, and document exact commands in README. Do not introduce dependencies for milestone 5.

## Coding Style and Teaching Interfaces

Use four-space indentation and the configured Ruff rules (Python 3.12, line length 88). Keep names descriptive and public interfaces small; document shapes and non-obvious behavior.

`CausalSelfAttentionHead(embedding_dim, head_dim)` owns three bias-free projections named `query`, `key`, and `value`. Its `forward(x)` accepts floating features `[B, T, embedding_dim]`, with `T >= 1`, and returns `(output, attention_weights)` of shapes `[B, T, head_dim]` and `[B, T, T]`. Validate dimensions and inputs with `ValueError`; use ordinary PyTorch device/dtype matching. The completed single head exposes Q/K/V projection, scaled scores, a diagonal-inclusive causal mask, softmax over keys, and value aggregation. Preserve the learner's calculation; only update stale documentation or formatting as needed. Do not replace attention steps with a packaged operation.

`MultiHeadCausalSelfAttention(embedding_dim, num_heads)` is complete. It registers independent heads in `heads: nn.ModuleList` and a bias-free `projection` mapping C to C, returning output `[B, T, C]` and weights `[B, H, T, T]`. Preserve the learner's calculation and existing tests; formatting and stale documentation may be updated.

Milestone 4 in `model.py` is complete. Preserve the learner's three forward methods:

- `TokenPositionEmbedding(vocab_size, context_length, embedding_dim)`: learned token plus absolute position tables, IDs `[B,T]` to features `[B,T,C]`. IDs must be `torch.long`, in vocabulary range, and `1 <= T <= context_length`. Positions restart at zero; input IDs must match the module device.
- `TransformerBlock(embedding_dim, num_heads)`: pre-norm attention and a provided `Linear(C,4C) -> GELU -> Linear(4C,C)` feed-forward with biases, each with a residual connection. Return features only, use the updated features in the second residual, and do not modify input in-place. Normalize over C only.
- `TinyTransformerLM(vocab_size, context_length, embedding_dim=64, num_heads=4, num_layers=2)`: embedding, independent registered blocks, final LayerNorm, and a bias-free, untied vocabulary head. Return raw logits `[B,T,V]` at every position, without softmax or loss.

All dimensions must be positive integers and C divisible by the head count. Model forward returns raw logits; keep training/loss outside the model. Preserve completed calculation behavior, updating only stale documentation and formatting as needed.

Milestone 5 in `training.py` provides validation, docstrings and hints, with core `NotImplementedError` exercises:

- `split_tokens(tokens, train_fraction=0.9, *, context_length)`: contiguous CPU-long 1D partitions at `int(N * fraction)`; finite fraction strictly between 0 and 1; each partition has at least T+1 tokens.
- `next_token_loss(logits, targets)`: mean cross-entropy over all B*T aligned positions, scalar tensor retaining its graph. Require nonempty floating logits `[B,T,V]`, matching same-device long targets `[B,T]`, and valid target range. Do not softmax or shift targets again.
- `train_step(model, optimizer, x, y)`: validate nonempty matching CPU-long `[B,T]` batches, then training mode, zero gradients, forward/loss, backward and exactly one optimizer update. Return the pre-update loss float.
- `evaluate_loss(model, batches)`: nonempty list of valid pairs; eval mode plus no_grad; token-weighted average as a float. Restore the caller's original mode even on exceptions. Do not change parameters or existing gradients.

`train.py` supplies CLI, data/tokenizer setup, sampling, AdamW construction and reporting. Default architecture 64 dimensions/4 heads/2 layers, CPU, seed 42, B=4/T=16, 200 steps, lr=0.003, weight_decay=0. Ordinary mode evaluates five fixed batches per partition at step 0, every 50 steps and the final step; independent generators must not consume training RNG state. Overfit mode samples only one training batch and reuses it. Vocabulary uses the whole corpus only to determine character IDs; sample/update training data exclusively from the training prefix. No dependencies, generation, checkpoints, schedulers, or GPU requirements are added.

## Testing and Validation

Use `exercise` only for unfinished milestone 5 calculations. Keep `milestone2`, `milestone3`, and `milestone4` for completed regressions; use `milestone5` for training and runner checks. Completed milestone 1–4 tests are ordinary regressions. No coverage threshold is configured.

At scaffold delivery, non-exercise tests, Ruff, changed-file formatting checks, and all four existing inspection commands must pass. Exercise tests intentionally fail at core `NotImplementedError`; both training modes must give a helpful unfinished message. Report expected exercise failures separately from regressions. Once completed, the full suite and both training modes must work.

Cover exact contiguous splits/boundaries; hand-calculated mean loss and gradients; clearing old gradients and reference updates; token-weighted evaluation, no state/gradient changes, and mode restoration on success/failure; independent reproducible RNG streams and fixed evaluation batches; fixed-batch learning with a small seeded real model. Keep learning tests short on CPU and use a broad relative loss reduction criterion, not zero loss or guaranteed validation improvement. Restore any thread-count changes made in tests. Preserve completed milestone 1–4 calculations and untracked learner files.

## Commits, Pull Requests, and Security

Use focused commits with short imperative subjects. PRs should explain the learning objective, resulting behavior, validation, and any deliberately unfinished exercises. Link related issues when available; document dependency or configuration changes.

Keep credentials, tokens, and local configuration out of committed files. Existing ignore rules cover virtual environments, caches, generated packages, checkpoints, and `.env`; document any future environment variables with placeholders.
