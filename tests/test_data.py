import pytest
import torch

from transformer.data import CharacterTokenizer, make_batch


def test_vocabulary_is_sorted_and_unique():
    tokenizer = CharacterTokenizer.from_text("cabca\n")
    assert tokenizer.characters == ("\n", "a", "b", "c")


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        CharacterTokenizer.from_text("")


def test_encoding_uses_vocabulary_indices_and_round_trips():
    tokenizer = CharacterTokenizer(("\n", "a", "b"))
    assert tokenizer.encode("ba\n") == [2, 1, 0]
    assert tokenizer.decode([2, 1, 0]) == "ba\n"
    assert tokenizer.decode(tokenizer.encode("abba\n")) == "abba\n"
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_unknown_character_is_rejected():
    with pytest.raises(ValueError):
        CharacterTokenizer(("a",)).encode("b")


@pytest.mark.parametrize("token_id", [-1, 2])
def test_invalid_token_id_is_rejected(token_id):
    with pytest.raises(ValueError):
        CharacterTokenizer(("a", "b")).decode([token_id])


def test_batch_shapes_and_next_character_alignment():
    tokens = torch.arange(100, dtype=torch.long)
    x, y = make_batch(tokens, 16, 32, generator=torch.Generator().manual_seed(42))
    assert x.shape == y.shape == (16, 32)
    assert x.dtype == y.dtype == torch.long
    assert x.device.type == y.device.type == "cpu"
    assert torch.equal(y, x + 1)
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)
    assert int(x.min()) >= 0 and int(y.max()) < len(tokens)


def test_minimum_length_includes_last_valid_window():
    x, y = make_batch(torch.tensor([4, 7, 2, 9]), 2, 3)
    assert x.tolist() == [[4, 7, 2], [4, 7, 2]]
    assert y.tolist() == [[7, 2, 9], [7, 2, 9]]


def test_sampling_is_reproducible_with_a_seed():
    tokens = torch.arange(80)
    first = make_batch(tokens, 16, 32, generator=torch.Generator().manual_seed(7))
    second = make_batch(tokens, 16, 32, generator=torch.Generator().manual_seed(7))
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize(
    "tokens,batch_size,context_length",
    [
        (torch.arange(3), 2, 3),
        (torch.tensor([], dtype=torch.long), 2, 3),
        (torch.arange(8), 0, 3),
        (torch.arange(8), -1, 3),
        (torch.arange(8), 2, 0),
        (torch.arange(8), 2, -1),
        (torch.arange(8).reshape(2, 4), 2, 3),
        (torch.arange(8, dtype=torch.float32), 2, 3),
    ],
)
def test_invalid_batch_inputs_are_rejected(tokens, batch_size, context_length):
    with pytest.raises(ValueError):
        make_batch(tokens, batch_size, context_length)
