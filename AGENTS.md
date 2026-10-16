# Repository Guidelines

## Purpose and Learning Workflow

This is a guided, character-level GPT-style Transformer learning project. Milestone 1 (tokenization and next-character batches) is implemented. Milestone 2 provides a single-head causal self-attention exercise scaffold, tests, and an inspection command; its core forward calculation is intentionally unfinished.

Default to scaffolds, incremental hints, tests, and conceptual review. Only supply or fill in the exercise's core solution when the user explicitly requests an answer or complete implementation. Preserve intentional `NotImplementedError` exercises when preparing learning material. Do not bypass exercise checks by skipping, deleting, or marking them xfail. After a learner completes a milestone, review the implementation and update its status and test markers before advancing.

## Project Structure

- `src/transformer/`: package source; `data.py` contains completed milestone 1, `attention.py` contains the milestone 2 scaffold, and `inspect_data.py` / `inspect_attention.py` are runnable teaching entry points.
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
uv run pytest -m "milestone2 and exercise" -x
uv run pytest
uv run python -m transformer.inspect_data
uv run python -m transformer.inspect_attention
uv run ruff check .
uv run ruff format .
uv build
```

Initial dependency setup needs package-index access. No GPU or downloaded dataset is required. `uv build` writes distribution artifacts to `dist/`. Keep dependency changes intentional, update `uv.lock` when dependencies change, and document exact commands in README. Do not introduce dependencies for this attention exercise.

## Coding Style and Teaching Interfaces

Use four-space indentation and the configured Ruff rules (Python 3.12, line length 88). Keep names descriptive and public interfaces small; document shapes and non-obvious behavior.

`CausalSelfAttentionHead(embedding_dim, head_dim)` owns three bias-free projections named `query`, `key`, and `value`. Its `forward(x)` accepts floating features `[B, T, embedding_dim]`, with `T >= 1`, and returns `(output, attention_weights)` of shapes `[B, T, head_dim]` and `[B, T, T]`. Validate dimensions and inputs with `ValueError`; use ordinary PyTorch device/dtype matching. The exercise must expose Q/K/V projection, scaled scores, a diagonal-inclusive causal mask, softmax over keys, and value aggregation. Do not replace these steps with a packaged attention operation. No dropout, full model, training loop, or multi-head attention belongs to milestone 2.

## Testing and Validation

Use `exercise` only for unfinished core calculations; `milestone2` selects all single-head attention checks, including provided constructor and validation behavior. Completed milestone 1 tests are ordinary regression checks. No coverage threshold is configured.

At scaffold delivery, non-exercise tests and Ruff must pass. Exercise tests intentionally fail at the core `NotImplementedError`; the attention inspection command must exit with a helpful message. Report these expected failures explicitly, separately from regressions. Once the learner finishes the exercise, the full suite and both inspection commands must pass.

Cover shape/probability invariants, a hand-calculated reference, future-information isolation, batch independence, sequence-length boundaries, and gradient flow. Avoid tests that require a GPU or network. Do not change the completed milestone 1 implementation as part of preparing milestone 2.

## Commits, Pull Requests, and Security

Use focused commits with short imperative subjects. PRs should explain the learning objective, resulting behavior, validation, and any deliberately unfinished exercises. Link related issues when available; document dependency or configuration changes.

Keep credentials, tokens, and local configuration out of committed files. Existing ignore rules cover virtual environments, caches, generated packages, checkpoints, and `.env`; document any future environment variables with placeholders.
