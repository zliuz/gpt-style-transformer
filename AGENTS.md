# Repository Guidelines

## Purpose and Learning Workflow

This is a guided, character-level GPT-style Transformer learning project. Milestones 1–3 (data, single-head attention, multi-head attention) are implemented and reviewed. Milestone 4 provides embeddings, pre-norm Transformer blocks, and a language-model forward scaffold; its three core forward methods are intentionally unfinished.

Default to scaffolds, incremental hints, tests, and conceptual review. Only supply or fill in the exercise's core solution when the user explicitly requests an answer or complete implementation. Preserve intentional `NotImplementedError` exercises when preparing learning material. Do not bypass exercise checks by skipping, deleting, or marking them xfail. After a learner completes a milestone, review the implementation and update its status and test markers before advancing.

## Project Structure

- `src/transformer/`: package source; `data.py` contains completed milestone 1, `attention.py` contains completed single/multi-head attention, `model.py` contains the milestone 4 scaffold, and `inspect_data.py` / `inspect_attention.py` / `inspect_multihead.py` / `inspect_model.py` are runnable teaching entry points.
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
uv run pytest -m "milestone4 and exercise" -x
uv run pytest
uv run python -m transformer.inspect_data
uv run python -m transformer.inspect_attention
uv run python -m transformer.inspect_multihead
uv run python -m transformer.inspect_model
uv run ruff check .
uv run ruff format .
uv build
```

Initial dependency setup needs package-index access. No GPU or downloaded dataset is required. `uv build` writes distribution artifacts to `dist/`. Keep dependency changes intentional, update `uv.lock` when dependencies change, and document exact commands in README. Do not introduce dependencies for milestone 4.

## Coding Style and Teaching Interfaces

Use four-space indentation and the configured Ruff rules (Python 3.12, line length 88). Keep names descriptive and public interfaces small; document shapes and non-obvious behavior.

`CausalSelfAttentionHead(embedding_dim, head_dim)` owns three bias-free projections named `query`, `key`, and `value`. Its `forward(x)` accepts floating features `[B, T, embedding_dim]`, with `T >= 1`, and returns `(output, attention_weights)` of shapes `[B, T, head_dim]` and `[B, T, T]`. Validate dimensions and inputs with `ValueError`; use ordinary PyTorch device/dtype matching. The completed single head exposes Q/K/V projection, scaled scores, a diagonal-inclusive causal mask, softmax over keys, and value aggregation. Preserve the learner's calculation; only update stale documentation or formatting as needed. Do not replace attention steps with a packaged operation.

`MultiHeadCausalSelfAttention(embedding_dim, num_heads)` is complete. It registers independent heads in `heads: nn.ModuleList` and a bias-free `projection` mapping C to C, returning output `[B, T, C]` and weights `[B, H, T, T]`. Preserve the learner's calculation and existing tests; formatting and stale documentation may be updated.

Milestone 4 in `model.py` provides constructors and validation for three exercises:

- `TokenPositionEmbedding(vocab_size, context_length, embedding_dim)`: learned token plus absolute position tables, IDs `[B,T]` to features `[B,T,C]`. IDs must be `torch.long`, in vocabulary range, and `1 <= T <= context_length`. Positions restart at zero; input IDs must match the module device.
- `TransformerBlock(embedding_dim, num_heads)`: pre-norm attention and a provided `Linear(C,4C) -> GELU -> Linear(4C,C)` feed-forward with biases, each with a residual connection. Return features only, use the updated features in the second residual, and do not modify input in-place. Normalize over C only.
- `TinyTransformerLM(vocab_size, context_length, embedding_dim=64, num_heads=4, num_layers=2)`: embedding, independent registered blocks, final LayerNorm, and a bias-free, untied vocabulary head. Return raw logits `[B,T,V]` at every position, without softmax or loss.

All dimensions must be positive integers and C divisible by the head count. Validation is complete; only the three core forwards remain learner exercises. Preserve computation graphs. No packaged Transformer, dropout, training, generation, caching, or checkpointing belongs to milestone 4.

## Testing and Validation

Use `exercise` only for unfinished milestone 4 core calculations. Keep `milestone2` and `milestone3` for completed attention regressions, and `milestone4` for embedding/block/model checks including the provided constructor and validation behavior. Completed milestone 1–3 tests are ordinary regressions. No coverage threshold is configured.

At scaffold delivery, non-exercise tests, Ruff, and changed-file formatting checks must pass. Exercise tests intentionally fail at core `NotImplementedError` statements; the model inspection command must exit with a helpful message, while the existing three inspection commands must succeed. Report expected failures separately from regressions. After the learner completes milestone 4, the full suite and all four inspection commands must pass.

Cover exact embedding sums, position/batch broadcasting, pre-norm ordering, both residual paths, zero-sublayer identity, registered independent blocks, raw logits, causal/prefix and batch invariance, sequence/ID boundaries, and gradient flow. Keep tests on CPU without network access. Preserve completed milestone 1–3 calculations while preparing milestone 4.

## Commits, Pull Requests, and Security

Use focused commits with short imperative subjects. PRs should explain the learning objective, resulting behavior, validation, and any deliberately unfinished exercises. Link related issues when available; document dependency or configuration changes.

Keep credentials, tokens, and local configuration out of committed files. Existing ignore rules cover virtual environments, caches, generated packages, checkpoints, and `.env`; document any future environment variables with placeholders.
