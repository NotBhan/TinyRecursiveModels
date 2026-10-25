"""Tests for Coding Reasoning Dataset, Architecture Integration, and Evaluation.

Validates:
1. Dataset download integrity and schema.
2. Leakage detection and deduplication.
3. Tokenizer compatibility on Python code (byte-level BPE, 0 unk).
4. Task formatting and prompt-target loss masking.
5. Task mixture accounting and collation.
6. Controller causality and gradient separation.
7. Checkpoint save/reload fidelity.
8. Execution sandbox safety (timeout, syntax error, memory, isolation).
9. Metrics calculation (pass@1, CE, PPL).
"""

import ast
import json
import math
import os
import tempfile
import pytest
import torch
import torch.nn.functional as F

from dataset.coding_reasoning import (
    DELIMITER_PROBLEM,
    DELIMITER_SOLUTION,
    CodingReasoningConfig,
    CodingReasoningDataset,
    MBPPDatasetInspector,
    MixedCodingReasoningDataset,
    format_task_b_code_continuation,
    format_task_c_function_synthesis,
    format_task_d_output_prediction,
    normalize_text_for_dedup,
    pack_prompt_target,
    parse_assertion,
)
from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    count_valid_targets,
    load_tokenizer,
    next_token_cross_entropy,
    validate_next_token_alignment,
)
from evaluators.coding_eval import (
    check_syntax_validity,
    compare_thinking_off_vs_on,
    evaluate_function_synthesis,
    extract_python_code,
)
from evaluators.execution_sandbox import execute_sandboxed
from models.recursive_reasoning.trm_lm_adaptive import (
    STOP_LEARNED_HALT,
    STOP_MAX_REACHED,
    DualStateConfig,
    DualStateModel,
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("experiments/trm_lm_pilot_5m/r1")


@pytest.fixture(scope="module")
def small_dual_state_model(tokenizer):
    cfg = DualStateConfig(
        vocab_size=8000,
        hidden_size=64,
        num_heads=2,
        n_low_layers=1,
        n_high_layers=1,
        n_low_cycles=1,
        n_high_cycles=2,
        min_high_cycles=1,
        max_high_cycles=3,
        controller_hidden_size=32,
        pad_token_id=tokenizer.token_to_id("<pad>"),
    )
    return DualStateModel(cfg)


# ----------------------------------------------------------------------
# 1. Dataset download integrity and schema
# ----------------------------------------------------------------------
def test_mbpp_manifest_and_file_integrity():
    manifest_path = "data/coding/mbpp/manifest.json"
    assert os.path.exists(manifest_path), "MBPP manifest.json missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_name"] == "Mostly Basic Python Problems (MBPP)"
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["source_huggingface_revision"] == "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

    # Verify all files recorded in manifest exist and have non-zero size
    for rel_path, meta in manifest["files"].items():
        full_path = os.path.join("data/coding/mbpp", rel_path)
        assert os.path.exists(full_path), f"File {rel_path} does not exist"
        assert os.path.getsize(full_path) == meta["size_bytes"], f"Size mismatch for {rel_path}"


def test_mbpp_schema_and_row_counts():
    inspector = MBPPDatasetInspector(mbpp_dir="data/coding/mbpp", config_name="full")
    for split, count in [("train", 374), ("validation", 90), ("test", 500), ("prompt", 10)]:
        rows = inspector.load_raw_split(split)
        assert len(rows) == count
        for r in rows:
            assert "task_id" in r and isinstance(r["task_id"], int)
            assert "problem_text" in r and len(r["problem_text"]) > 0
            assert "code" in r and len(r["code"]) > 0
            assert "test_list" in r and isinstance(r["test_list"], list)


# ----------------------------------------------------------------------
# 2. Leakage detection and deduplication
# ----------------------------------------------------------------------
def test_leakage_detection_quarantines_cross_split_matches():
    inspector = MBPPDatasetInspector(mbpp_dir="data/coding/mbpp", config_name="full")
    audit = inspector.audit_and_deduplicate()

    # Verify that the 3 known cross-split duplicates in MBPP full are caught and quarantined
    quarantined_ids = {q["task_id"] for q in audit["quarantined_examples"]}
    assert 602 in quarantined_ids  # Matches test 217
    assert 704 in quarantined_ids  # Matches test 248 (both text and code)
    assert 872 in quarantined_ids  # Matches test 216

    # Verify retained train has no intersection with validation or test
    eval_texts = {r["problem_text"].strip() for r in audit["splits"]["validation"] + audit["splits"]["test"]}
    eval_codes = {r["code"].strip() for r in audit["splits"]["validation"] + audit["splits"]["test"]}

    for train_row in audit["splits"]["train"]:
        assert train_row["problem_text"].strip() not in eval_texts
        assert train_row["code"].strip() not in eval_codes


def test_normalize_text_dedup():
    t1 = "Write a python function to find the first repeated character in a given string."
    t2 = "  write a python function to find the first repeated character in a given string! \n"
    assert normalize_text_for_dedup(t1) == normalize_text_for_dedup(t2)


# ----------------------------------------------------------------------
# 3. Tokenizer compatibility
# ----------------------------------------------------------------------
def test_tokenizer_zero_unk_on_mbpp_code(tokenizer):
    unk_id = tokenizer.token_to_id("<unk>")
    inspector = MBPPDatasetInspector(mbpp_dir="data/coding/mbpp", config_name="full")
    train_rows = inspector.load_raw_split("train")

    for r in train_rows[:50]:
        code_ids = tokenizer.encode(r["code"]).ids
        text_ids = tokenizer.encode(r["problem_text"]).ids
        assert unk_id not in code_ids, f"Found <unk> in code for task {r['task_id']}"
        assert unk_id not in text_ids, f"Found <unk> in text for task {r['task_id']}"


def test_tokenizer_indentation_and_punctuation(tokenizer):
    sample_code = "def is_even(n):\n    return n % 2 == 0\n"
    encoded = tokenizer.encode(sample_code)
    decoded = tokenizer.decode(encoded.ids)
    assert "def is_even" in decoded
    assert "%" in decoded
    assert "==" in decoded


# ----------------------------------------------------------------------
# 4. Task formatting and loss masking
# ----------------------------------------------------------------------
def test_assertion_parser():
    stmt = "assert min_cost([[1, 2], [3, 4]], 1, 1) == 5"
    parsed = parse_assertion(stmt)
    assert parsed is not None
    call_expr, expected = parsed
    assert call_expr == "min_cost([[1, 2], [3, 4]], 1, 1)"
    assert expected == "5"


def test_task_b_code_continuation():
    code = "def foo(x):\n    a = x + 1\n    return a * 2\n"
    res = format_task_b_code_continuation(code, prefix_ratio=0.5)
    assert res is not None
    prefix, continuation = res
    assert "def foo(x):" in prefix
    assert "return" in continuation


def test_prompt_target_loss_masking(tokenizer):
    prompt = "### Problem:\nAdd two numbers.\n### Solution:\n"
    target = "def add(a, b):\n    return a + b\n"
    pad_id = tokenizer.token_to_id("<pad>")

    packed = pack_prompt_target(
        prompt=prompt,
        target=target,
        tokenizer=tokenizer,
        seq_len=64,
        pad_id=pad_id,
        task_type="func_synthesis",
        task_id=101,
    )
    assert packed is not None

    input_ids = packed["input_ids"]
    labels = packed["labels"]
    attention_mask = packed["attention_mask"]

    # Prompt tokens should be IGNORE_INDEX in labels
    prompt_len = len(tokenizer.encode(prompt).ids)
    # The prompt starts after <bos> (at index 0)
    for i in range(prompt_len):
        assert labels[i].item() == IGNORE_INDEX

    # Target tokens should have valid labels
    valid_count = count_valid_targets(labels)
    assert valid_count > 0
    # Next token alignment check must pass
    validate_next_token_alignment(input_ids.unsqueeze(0), labels.unsqueeze(0))


# ----------------------------------------------------------------------
# 5. Task mixture accounting and collation
# ----------------------------------------------------------------------
def test_mixed_dataset_batch_and_accounting(tokenizer):
    ts_cfg = TinyStoriesConfig(
        parquet_path="/home/chandrabhan/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet",
        workdir="experiments/trm_lm_pilot_5m/r1",
        seq_len=128,
    )
    with open("experiments/trm_lm_pilot_5m/r1/manifest.json") as f:
        manifest = json.load(f)

    ts_ds = TinyStoriesLMDataset(ts_cfg, manifest, tokenizer, split="train", max_docs=5)
    cr_cfg = CodingReasoningConfig(
        lang_weight=0.6,
        code_continuation_weight=0.1,
        func_synthesis_weight=0.2,
        output_prediction_weight=0.1,
        seq_len=128,
    )
    coding_ds = CodingReasoningDataset(cr_cfg, tokenizer, split="train", max_examples=10)
    mixed_ds = MixedCodingReasoningDataset(cr_cfg, tokenizer, ts_ds, coding_ds, max_train_tokens=1000)

    batch_items = []
    for item in mixed_ds:
        batch_items.append(item)
        if len(batch_items) == 4:
            break

    assert len(batch_items) == 4
    batch = collate_lm_batch(batch_items)
    assert batch["input_ids"].shape == (4, 128)
    assert batch["labels"].shape == (4, 128)
    assert batch["attention_mask"].shape == (4, 128)

    # Valid target tokens accounted
    assert sum(mixed_ds.token_accounting.values()) > 0


# ----------------------------------------------------------------------
# 6. Controller causality on coding inputs
# ----------------------------------------------------------------------
def test_controller_causality_on_coding_inputs(small_dual_state_model, tokenizer):
    """Ensure that modifying future tokens in a coding sequence does not alter earlier logits."""
    small_dual_state_model.eval()
    prompt = "### Problem:\nReverse a list.\n### Solution:\n"
    target_a = "def rev(l):\n    return l[::-1]\n"
    target_b = "def rev(l):\n    res = list(reversed(l))\n    return res\n"
    pad_id = tokenizer.token_to_id("<pad>")

    ex_a = pack_prompt_target(prompt, target_a, tokenizer, seq_len=64, pad_id=pad_id)
    ex_b = pack_prompt_target(prompt, target_b, tokenizer, seq_len=64, pad_id=pad_id)

    # Find first diverging position
    diff_pos = (ex_a["input_ids"] != ex_b["input_ids"]).nonzero()
    first_diff = int(diff_pos[0].item())
    assert first_diff > 5, "Differences should occur in the target body"

    with torch.no_grad():
        out_a = small_dual_state_model(
            ex_a["input_ids"].unsqueeze(0),
            ex_a["attention_mask"].unsqueeze(0),
            mode="adaptive",
        )
        out_b = small_dual_state_model(
            ex_b["input_ids"].unsqueeze(0),
            ex_b["attention_mask"].unsqueeze(0),
            mode="adaptive",
        )

    # Invariance check: positions before first_diff must be strictly identical
    logits_a_earlier = out_a.logits[0, :first_diff]
    logits_b_earlier = out_b.logits[0, :first_diff]
    max_dev = float((logits_a_earlier - logits_b_earlier).abs().max().item())
    assert max_dev == 0.0, f"Causality violation on coding inputs: max deviation = {max_dev}"


# ----------------------------------------------------------------------
# 7. Controller gradient flow on mixed batch
# ----------------------------------------------------------------------
def test_gradient_flow_on_mixed_batch(small_dual_state_model, tokenizer):
    small_dual_state_model.train()
    pad_id = tokenizer.token_to_id("<pad>")
    prompt = "### Problem:\nSquare a number.\n### Solution:\n"
    target = "def sq(x):\n    return x * x\n"

    ex = pack_prompt_target(prompt, target, tokenizer, seq_len=32, pad_id=pad_id)
    input_ids = ex["input_ids"].unsqueeze(0)
    attention_mask = ex["attention_mask"].unsqueeze(0)
    labels = ex["labels"].unsqueeze(0)

    depth_out = small_dual_state_model.forward_depths(
        input_ids, attention_mask, depths=[1, 2], max_depth=2, evaluate_controller=True
    )

    lm_loss = next_token_cross_entropy(depth_out.logits_by_depth[2], labels)
    ctrl_logits = depth_out.controller_logits_by_depth[1]
    # Synthetic target
    ctrl_target = torch.ones_like(ctrl_logits)
    ctrl_loss = F.binary_cross_entropy_with_logits(ctrl_logits, ctrl_target)

    total_loss = lm_loss + ctrl_loss
    total_loss.backward()

    # Controller gradients must be non-zero
    ctrl_grad_sum = sum(
        p.grad.abs().sum().item()
        for p in small_dual_state_model.controller.parameters()
        if p.grad is not None
    )
    assert ctrl_grad_sum > 0.0, "Controller received no gradients"

    # Backbone gradients must be non-zero
    embed_grad_sum = small_dual_state_model.embed_tokens.weight.grad.abs().sum().item()
    assert embed_grad_sum > 0.0, "Backbone received no gradients"


# ----------------------------------------------------------------------
# 8. Checkpoint save/reload fidelity
# ----------------------------------------------------------------------
def test_checkpoint_save_reload(small_dual_state_model, tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    ex = pack_prompt_target(
        "### Problem:\nTest\n### Solution:\n", "x = 1\n", tokenizer, seq_len=32, pad_id=pad_id
    )
    input_ids = ex["input_ids"].unsqueeze(0)
    mask = ex["attention_mask"].unsqueeze(0)

    small_dual_state_model.eval()
    with torch.no_grad():
        orig_out = small_dual_state_model(input_ids, mask, mode="adaptive")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        torch.save({"model_state_dict": small_dual_state_model.state_dict()}, tmp.name)

        # Load into new instance
        new_model = DualStateModel(small_dual_state_model.config)
        ckpt = torch.load(tmp.name, weights_only=True)
        new_model.load_state_dict(ckpt["model_state_dict"])
        new_model.eval()

        with torch.no_grad():
            new_out = new_model(input_ids, mask, mode="adaptive")

    max_diff = float((orig_out.logits - new_out.logits).abs().max().item())
    assert max_diff == 0.0, f"Checkpoint reload mismatch: {max_diff}"
    assert torch.equal(orig_out.stop_reason_codes, new_out.stop_reason_codes)


# ----------------------------------------------------------------------
# 9. Execution sandbox safety
# ----------------------------------------------------------------------
def test_sandbox_safety_and_resource_limits():
    # Pass case
    res_pass = execute_sandboxed("def f(x): return x + 1", ["assert f(1) == 2"])
    assert res_pass.passed is True
    assert res_pass.error_type is None

    # Fail assertion
    res_fail = execute_sandboxed("def f(x): return x + 1", ["assert f(1) == 3"])
    assert res_fail.passed is False
    assert res_fail.error_type == "assertion_error"

    # Syntax error
    res_syntax = execute_sandboxed("def f(x) return x + 1", ["assert f(1) == 2"])
    assert res_syntax.passed is False
    assert res_syntax.error_type == "syntax_error"

    # Prohibited network attempt (OS container isolation)
    res_network = execute_sandboxed("import socket\ns = socket.socket(); s.connect(('1.1.1.1', 80))", ["assert True"])
    assert res_network.passed is False
    assert res_network.error_type in ("runtime_error", "security_violation")

    # Timeout
    res_timeout = execute_sandboxed("def f():\n    while True: pass", ["assert f() == 1"], timeout_sec=1.0)
    assert res_timeout.passed is False
    assert res_timeout.error_type == "timeout"


# ----------------------------------------------------------------------
# 10. Metrics calculation
# ----------------------------------------------------------------------
def test_metrics_calculation():
    logits = torch.randn(2, 8, 100)
    labels = torch.randint(0, 100, (2, 8))
    labels[0, :3] = IGNORE_INDEX

    ce = next_token_cross_entropy(logits, labels)
    assert torch.isfinite(ce)
    assert ce.item() > 0.0
    ppl = math.exp(min(ce.item(), 50.0))
    assert ppl >= 1.0


def test_code_extraction():
    raw1 = "```python\ndef solve():\n    return 42\n```\nSome commentary"
    assert extract_python_code(raw1) == "def solve():\n    return 42"

    raw2 = "### Problem:\nFoo\n### Solution:\ndef solve():\n    return 42\n<eos>"
    assert extract_python_code(raw2) == "def solve():\n    return 42"
