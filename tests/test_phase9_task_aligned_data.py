"""Tests for Phase 9 task-aligned data construction."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dataset.code_pipeline import (
    CodePipelineConfig,
    build_instruction_example,
    line_prefix_token_lengths,
)
from dataset.tinystories_lm import IGNORE_INDEX


class CharTokenizer:
    """Deterministic character tokenizer sufficient for packing tests."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def token_to_id(self, token: str) -> int:
        return self.special[token]

    def encode(self, text: str):
        return SimpleNamespace(ids=[4 + (ord(ch) % 250) for ch in text])

    def get_vocab_size(self) -> int:
        return self.vocab_size


def make_record(body: str) -> dict:
    return {
        "task_id": 7,
        "problem_text": "Write a function that returns its input plus one.",
        "code": f"def add_one(x):\n{body}",
        "test_list": ["assert add_one(1) == 2"],
        "test_setup_code": "",
    }


def test_prefix_example_masks_prefix_and_shifts_target():
    tokenizer = CharTokenizer()
    cfg = CodePipelineConfig(seq_len=128)
    record = make_record("    y = x + 1\n    return y\n")
    full = build_instruction_example(tokenizer, cfg, record, prefix_tokens=0)
    pref = build_instruction_example(tokenizer, cfg, record, prefix_tokens=4)
    assert full is not None and pref is not None
    assert pref["prefix_tokens"] == 4
    assert pref["target_tokens"] == full["target_tokens"] - 4
    assert pref["target_start"] == full["target_start"] + 4
    assert pref["valid_target_tokens"] == full["valid_target_tokens"] - 4
    labels = pref["labels"]
    inputs = pref["input_ids"]
    target_start = pref["target_start"]
    assert torch.all(labels[: target_start - 1] == IGNORE_INDEX)
    assert int(labels[target_start - 1]) == int(inputs[target_start])


def test_prefix_tokens_are_capped_below_full_body():
    tokenizer = CharTokenizer()
    cfg = CodePipelineConfig(seq_len=128)
    record = make_record("    return x + 1\n")
    example = build_instruction_example(tokenizer, cfg, record, prefix_tokens=10_000)
    assert example is not None
    assert example["target_tokens"] == 1
    assert example["prefix_tokens"] == example["full_body_tokens"] - 1


def test_line_prefix_boundaries_do_not_split_lines():
    tokenizer = CharTokenizer()
    body = "    a = 1\n    b = 2\n    return a + b\n"
    boundaries = line_prefix_token_lengths(body, tokenizer)
    assert boundaries
    decoded = tokenizer.encode(body).ids
    assert all(0 < value < len(decoded) for value in boundaries)
    # Each boundary must land exactly after a newline character.
    for value in boundaries:
        assert body[value - 1] == "\n"


def test_zero_prefix_is_bit_identical_to_original_packing():
    tokenizer = CharTokenizer()
    cfg = CodePipelineConfig(seq_len=128)
    record = make_record("    y = x + 1\n    return y\n")
    a = build_instruction_example(tokenizer, cfg, record)
    b = build_instruction_example(tokenizer, cfg, record, prefix_tokens=0)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])
    assert a["target_start"] == b["target_start"]
