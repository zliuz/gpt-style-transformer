from pathlib import Path

import torch

from transformer.data import CharacterTokenizer


def test_cpu_tensor_operations_work():
    values = torch.tensor([1, 2, 3], dtype=torch.long)
    assert (values + 1).tolist() == [2, 3, 4]
    assert values.device.type == "cpu"


def test_package_is_importable():
    assert CharacterTokenizer(("a", "b")).vocab_size == 2


def test_bundled_corpus_has_enough_text():
    path = Path(__file__).resolve().parents[1] / "data" / "toy.txt"
    text = path.read_text(encoding="utf-8")
    assert len(text) > 320
    assert len(set(text)) > 10

