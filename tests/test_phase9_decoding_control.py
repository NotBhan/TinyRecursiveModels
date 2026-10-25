"""Tests for the Phase 9 decoding-control generation modes."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from evaluators.recurrence_verification import generate_with_recurrence
from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


class FakeTokenizer:
    special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def token_to_id(self, token: str) -> int:
        return self.special[token]

    def encode(self, text: str):
        return SimpleNamespace(ids=[10 + (ord(ch) % 40) for ch in text])

    def decode(self, ids):
        return "".join(chr(40 + (int(i) % 40)) for i in ids)


def make_model() -> DualStateModel:
    torch.manual_seed(0)
    cfg = DualStateConfig(
        vocab_size=128,
        seq_len=64,
        hidden_size=64,
        num_heads=4,
        n_low_layers=1,
        n_high_layers=1,
        n_low_cycles=1,
        n_high_cycles=2,
        min_high_cycles=1,
        max_high_cycles=4,
    )
    return DualStateModel(cfg).eval()


def test_greedy_generation_is_deterministic():
    model = make_model()
    tok = FakeTokenizer()
    a = generate_with_recurrence(model, tok, "hello", n_high_cycles=2, max_new_tokens=8)
    b = generate_with_recurrence(model, tok, "hello", n_high_cycles=2, max_new_tokens=8)
    assert a["token_ids"] == b["token_ids"]


def test_sampling_is_seed_deterministic():
    model = make_model()
    tok = FakeTokenizer()
    a = generate_with_recurrence(
        model, tok, "hello", n_high_cycles=2, max_new_tokens=8,
        temperature=1.0, top_p=0.95, sampling_seed=123,
    )
    b = generate_with_recurrence(
        model, tok, "hello", n_high_cycles=2, max_new_tokens=8,
        temperature=1.0, top_p=0.95, sampling_seed=123,
    )
    assert a["token_ids"] == b["token_ids"]


def test_sampling_with_different_seeds_can_differ():
    model = make_model()
    tok = FakeTokenizer()
    sequences = {
        tuple(
            generate_with_recurrence(
                model, tok, "hello", n_high_cycles=1, max_new_tokens=12,
                temperature=2.0, top_p=1.0, sampling_seed=seed,
            )["token_ids"]
        )
        for seed in (1, 2, 3, 4)
    }
    assert len(sequences) > 1
