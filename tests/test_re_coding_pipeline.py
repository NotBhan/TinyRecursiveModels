"""Tests for the additive RE coding pipeline and adaptive inference additions.

These tests do not depend on experiment artifacts or the MBPP download. They
build a tiny byte-level BPE tokenizer from strings and use synthetic records.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from torch.utils.data import IterableDataset

from dataset.code_pipeline import (
    CodePipelineConfig,
    QuotaMixtureDataset,
    _target_function_name,
    build_instruction_example,
    pack_token_stream,
    split_target_function,
)
from dataset.tinystories_lm import IGNORE_INDEX, SPECIAL_TOKENS, count_valid_targets
from evaluators.coding_eval import extract_python_code
from evaluators.re_eval import (
    build_signature_prompt,
    difficulty_score,
    paired_deltas,
    tier_for_score,
)
from evaluators.re_eval import TaskResult
from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


def make_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = [
        "Once upon a time there was a small cat.",
        "def foo(x):\n    return x + 1\n",
        "assert foo(2) == 3",
        "Write a function to add one.\n### Problem:\n### Solution:\n",
    ]
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def make_record():
    return {
        "task_id": 1,
        "problem_text": "Write a function to add one to x.",
        "code": "def foo(x):\n    return x + 1\n",
        "test_list": ["assert foo(2) == 3", "assert foo(0) == 1"],
        "test_setup_code": "",
    }


def test_signature_split_excludes_body():
    record = make_record()
    name = _target_function_name(record["code"], record["test_list"])
    assert name == "foo"
    context, body = split_target_function(record["code"], name)
    assert "def foo(x):" in context
    assert "return x + 1" not in context
    assert "return x + 1" in body


def test_instruction_example_masks_prompt_and_aligns_targets():
    tokenizer = make_tokenizer()
    cfg = CodePipelineConfig(seq_len=128)
    record = make_record()
    example = build_instruction_example(tokenizer, cfg, record)
    assert example is not None
    assert example["expected_name"] == "foo"
    assert example["valid_target_tokens"] > 0
    labels = example["labels"]
    input_ids = example["input_ids"]
    target_start = example["target_start"]
    # Everything before the last prompt position is masked.
    assert torch.all(labels[: target_start - 1] == IGNORE_INDEX)
    # The last prompt position predicts the first target token.
    assert int(labels[target_start - 1]) == int(input_ids[target_start])
    # No body token appears in the prompt tokens.
    prompt_text = tokenizer.decode(input_ids[:target_start].tolist())
    assert "return x + 1" not in prompt_text
    assert "def foo(x):" in prompt_text


def test_pack_token_stream_drops_incomplete_block():
    tokens = list(range(10))
    assert pack_token_stream(tokens, seq_len=10, pad_id=0) == []
    blocks = pack_token_stream(tokens + [10], seq_len=10, pad_id=0)
    assert len(blocks) == 1
    assert blocks[0]["input_ids"].shape == (10,)
    assert int(blocks[0]["labels"][0]) == 1


def test_tier_for_score_respects_bounds_and_caps():
    assert tier_for_score(0.0, 1, 4) == (1, 128)
    assert tier_for_score(0.5, 1, 4) == (2, 192)
    assert tier_for_score(1.0, 1, 4) == (4, 256)
    # never exceeds explicit bounds
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        d, cap = tier_for_score(s, 2, 3)
        assert 2 <= d <= 3
        assert cap in (128, 192, 256)


def test_difficulty_score_is_bounded():
    for h in [0.0, 0.5, 10.0]:
        for m in [0.0, 0.5, 1.0]:
            for syn in [False, True]:
                s = difficulty_score(h, m, syn, 96, 8192, 96)
                assert 0.0 <= s <= 1.0


class FakeDataset(IterableDataset):
    def __init__(self, n: int, tokens_per_item: int, source: str):
        self.n = n
        self.t = tokens_per_item
        self.source = source

    def __iter__(self):
        for i in range(self.n):
            labels = torch.full((8,), IGNORE_INDEX, dtype=torch.int64)
            labels[: self.t] = 1
            yield {
                "input_ids": torch.zeros(8, dtype=torch.int64),
                "labels": labels,
                "attention_mask": torch.ones(8, dtype=torch.int64),
                "valid_target_tokens": self.t,
                "task_type": self.source,
                "task_id": i,
            }


def test_quota_mixture_respects_token_quotas():
    a = FakeDataset(100, 5, "a")
    b = FakeDataset(100, 5, "b")
    mixture = QuotaMixtureDataset(
        [("a", a, 10, False), ("b", b, 20, False)], max_tokens=25, seed=0
    )
    items = list(mixture)
    total = sum(i["valid_target_tokens"] for i in items)
    assert total <= 25
    assert mixture.token_accounting["a"] <= 10
    assert mixture.token_accounting["b"] <= 20
    assert total == mixture.token_accounting["a"] + mixture.token_accounting["b"]


def test_last_token_only_forward_and_generation_confidence():
    cfg = DualStateConfig(
        vocab_size=128,
        seq_len=32,
        hidden_size=64,
        num_heads=4,
        n_low_layers=1,
        n_high_layers=1,
        n_low_cycles=1,
        n_high_cycles=1,
        min_high_cycles=1,
        max_high_cycles=2,
    )
    model = DualStateModel(cfg)
    model.eval()
    inp = torch.randint(1, 100, (1, 8))
    mask = torch.ones_like(inp)
    full = model(inp, mask, mode="fast")
    last = model(inp, mask, mode="fast", last_token_only=True)
    assert full.logits.shape == (1, 8, 128)
    assert last.logits.shape == (1, 1, 128)
    gen = model.generate_with_diagnostics(
        inp, max_new_tokens=3, eos_token_id=None, mode="fast"
    )
    assert gen.new_token_count == 3
    assert len(gen.token_entropies) == 3
    assert len(gen.token_margins) == 3


def test_eval_prompt_contains_signature_not_body():
    record = make_record()
    prompt, name, body = build_signature_prompt(record)
    assert name == "foo"
    assert "def foo(x):" in prompt
    assert "return x + 1" not in prompt
    assert "return x + 1" in body


def _task(task_id, passed, syntax=True):
    return TaskResult(
        task_id=task_id,
        expected_name="foo",
        prompt_tokens=10,
        prompt_truncated=False,
        generated_tokens=10,
        raw_generation="x",
        assembled_code="x",
        nonempty=True,
        decodes=True,
        entry_point_present=True,
        syntax_valid=syntax,
        passed=passed,
        first_test_passed=passed,
        tests_total=3,
        tests_passed=3 if passed else 0,
        error_type=None if passed else "assertion_error",
        error_message="",
        latency_ms=1.0,
        total_transitions=2,
        mean_high_cycles=1.0,
        mean_token_entropy=1.0,
        mean_token_margin=0.5,
        mode="fast",
        requested_depth=None,
        context_cap=256,
    )


def test_extract_python_code_preserves_body_indent():
    raw = "  res = tuple(test_tup)\r\n  return (res)"
    preserved = extract_python_code(raw, preserve_leading_indent=True)
    assert preserved.startswith("  res")
    assert "  return (res)" in preserved
    # Legacy full-function behavior is unchanged.
    assert extract_python_code(raw) == "res = tuple(test_tup)\n  return (res)" or extract_python_code(raw).startswith("res")
    # A body may legitimately start with an indented nested definition. The
    # default preserve mode must not dedent it (MBPP tasks 215/317 regression).
    nested = "    def aux(g):\n        return g\n    return aux(x)\n"
    assert extract_python_code(nested, preserve_leading_indent=True).startswith("    def aux")
    # Callers that know the generation is a complete top-level function can
    # explicitly opt into the legacy dedent behavior.
    full = "  def foo(x):\n      return x + 1\n"
    assert extract_python_code(
        full, preserve_leading_indent=True, strip_first_def_indent=True
    ).startswith("def foo(x):")


def test_paired_deltas_counts_disagreements():
    baseline = [_task(1, True), _task(2, False), _task(3, True), _task(4, False)]
    treatment = [_task(1, True), _task(2, True), _task(3, False), _task(4, False)]
    out = paired_deltas(baseline, treatment)
    assert out["both_pass"] == 1
    assert out["only_treatment_pass"] == 1
    assert out["only_baseline_pass"] == 1
    assert out["both_fail"] == 1
    assert out["delta_pass_rate"] == 0.0
