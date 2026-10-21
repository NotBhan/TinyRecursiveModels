from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    SPLIT_TO_ID,
    TinyStoriesConfig,
    collate_lm_batch,
    count_valid_targets,
    drop_leading_valid_targets,
    encode_document,
    find_default_parquet,
    iter_packed_examples,
    iter_selected_documents,
    load_tokenizer,
    next_token_cross_entropy,
    pack_document_tokens,
    prepare_split_manifest,
    split_name_for_row,
    train_tokenizer,
    truncate_labels_to_budget,
    validate_next_token_alignment,
    validate_text,
    verify_parquet,
)
from models.recursive_reasoning.causal_transformer_lm import CausalTransformerLM
from models.recursive_reasoning.trm_lm import TRM_LM, TRM_LMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_HASHES_PATH = Path(__file__).resolve().parent / "reference_hashes.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def parquet_path() -> str:
    try:
        return find_default_parquet()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def bounded_data(parquet_path: str, tmp_path_factory) -> Dict[str, object]:
    pytest.importorskip("pyarrow")
    pytest.importorskip("tokenizers")
    workdir = tmp_path_factory.mktemp("trm_lm_data")
    config = TinyStoriesConfig(
        parquet_path=parquet_path,
        workdir=str(workdir),
        seq_len=64,
        tokenizer_vocab_size=1024,
        tokenizer_train_docs=500,
        seed=0,
        max_scan_rows=25_000,
        max_train_docs=50,
        max_val_docs=10,
    )
    manifest = prepare_split_manifest(config)
    tokenizer, tokenizer_config = train_tokenizer(config, manifest)
    return {
        "config": config,
        "manifest": manifest,
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config,
    }


@pytest.fixture()
def small_config() -> TRM_LMConfig:
    return TRM_LMConfig(
        vocab_size=128,
        seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        expansion=4.0,
        n_refine=2,
        tie_embeddings=True,
        pad_token_id=0,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
    )


@pytest.fixture()
def small_model(small_config: TRM_LMConfig) -> TRM_LM:
    torch.manual_seed(0)
    return TRM_LM(small_config)


def make_batch(
    batch_size: int = 2, seq_len: int = 16, vocab_size: int = 128
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randint(1, vocab_size, (batch_size, seq_len), generator=generator)
    labels = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
    labels[:, :-1] = inputs[:, 1:]
    mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


@pytest.fixture()
def batch() -> Dict[str, torch.Tensor]:
    return make_batch()


@pytest.fixture()
def padded_batch() -> Dict[str, torch.Tensor]:
    batch = make_batch(seq_len=16)
    real_len = 12
    batch["input_ids"][:, real_len:] = 0
    batch["attention_mask"][:, real_len:] = 0
    batch["labels"][:, real_len - 1 :] = IGNORE_INDEX
    return batch


# ---------------------------------------------------------------------------
# A. Dataset tests
# ---------------------------------------------------------------------------

def test_parquet_opens_and_matches_audit(parquet_path: str) -> None:
    info = verify_parquet(parquet_path)
    assert info["columns"] == ["text"]
    assert info["num_rows"] == 2_732_634
    assert info["size_bytes"] == 672_825_299
    assert info["matches_audit_rows"] is True
    assert info["matches_audit_size"] is True


def test_text_extraction_returns_valid_strings(bounded_data) -> None:
    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    docs = list(iter_selected_documents(config, manifest, "val", max_docs=5))
    assert len(docs) == 5
    for row_index, text in docs:
        assert 10_000 <= row_index < 20_000
        assert isinstance(text, str) and text.strip()
        assert "\x00" not in text


def test_malformed_record_validation() -> None:
    assert validate_text(None) == (False, "not_a_string")
    assert validate_text("") == (False, "empty_or_whitespace")
    assert validate_text("   \n\t") == (False, "empty_or_whitespace")
    assert validate_text("abc\x00def") == (False, "contains_nul")
    assert validate_text("Once upon a time.") == (True, None)


def test_split_policy_is_deterministic_and_disjoint() -> None:
    config = TinyStoriesConfig(parquet_path="unused", workdir="unused")
    assert split_name_for_row(0, config) == "test"
    assert split_name_for_row(9_999, config) == "test"
    assert split_name_for_row(10_000, config) == "val"
    assert split_name_for_row(19_999, config) == "val"
    assert split_name_for_row(20_000, config) == "train"
    assert split_name_for_row(2_732_633, config) == "train"


def test_manifest_counts_and_split_ids(bounded_data) -> None:
    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    split_ids = np.load(manifest["split_ids_path"])
    assert split_ids.shape[0] == manifest["scanned_rows"]
    for split_name, split_id in SPLIT_TO_ID.items():
        rows = np.where(split_ids == split_id)[0]
        assert len(rows) == manifest["counts"]["accepted"][split_name]
        if split_name == "test":
            assert rows.max() < 10_000
        elif split_name == "val":
            assert rows.min() >= 10_000 and rows.max() < 20_000
        else:
            assert rows.min() >= 20_000
    assert manifest["split_policy"]["cross_document_attention"] is False
    assert manifest["split_policy"]["document_level"] is True


def test_duplicate_and_malformed_policy(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    texts = [
        "docA",
        None,
        "docB",
        "docA",  # duplicate of row 0 (test -> val)
        "",
        "docC",
        "docB",  # duplicate of row 2 (test -> train)
        "docD",
        "\x00bad",
    ]
    path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"text": texts}), str(path))
    config = TinyStoriesConfig(
        parquet_path=str(path),
        workdir=str(tmp_path / "work"),
        test_rows=(0, 3),
        val_rows=(3, 6),
        train_rows=(6, None),
        max_scan_rows=9,
        tokenizer_train_docs=1,
    )
    manifest = prepare_split_manifest(config)
    counts = manifest["counts"]
    assert counts["examined"] == 9
    assert counts["accepted"] == {"test": 2, "val": 1, "train": 1}
    assert counts["retained_total"] == 4
    assert counts["rejected_total"] == 5
    assert counts["duplicate_cross_split"] == 2
    assert counts["duplicate_within_split"] == 0
    assert counts["reject_reasons"] == {
        "not_a_string": 1,
        "empty_or_whitespace": 1,
        "contains_nul": 1,
    }
    split_ids = np.load(manifest["split_ids_path"])
    assert split_ids[0] == SPLIT_TO_ID["test"]
    assert split_ids[2] == SPLIT_TO_ID["test"]
    assert split_ids[5] == SPLIT_TO_ID["val"]
    assert split_ids[7] == SPLIT_TO_ID["train"]
    assert split_ids[1] == -1 and split_ids[3] == -1 and split_ids[4] == -1
    assert split_ids[6] == -1 and split_ids[8] == -1


def test_tokenizer_is_trained_on_train_only(
    bounded_data, tmp_path, monkeypatch
) -> None:
    import dataset.tinystories_lm as lm_data

    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    new_config = TinyStoriesConfig(
        **{**config.__dict__, "workdir": str(tmp_path / "tok")}
    )
    calls = []
    original = lm_data.iter_selected_documents

    def wrapped(*args, **kwargs):
        calls.append(args[2])  # split argument
        yield from original(*args, **kwargs)

    monkeypatch.setattr(lm_data, "iter_selected_documents", wrapped)
    tokenizer, tokenizer_config = train_tokenizer(new_config, manifest)
    assert calls == ["train"]
    assert tokenizer_config["train_split"] == "train"
    assert tokenizer_config["train_docs"] <= new_config.tokenizer_train_docs
    assert tokenizer.get_vocab_size() == tokenizer_config["vocab_size"]


def test_token_ids_fit_vocabulary(bounded_data) -> None:
    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    tokenizer = bounded_data["tokenizer"]
    vocab_size = tokenizer.get_vocab_size()
    for _, text in iter_selected_documents(config, manifest, "train", max_docs=10):
        tokens = encode_document(text, tokenizer, vocab_size)
        assert tokens
        assert min(tokens) >= 0
        assert max(tokens) < vocab_size


def test_packing_boundaries_and_final_block() -> None:
    tokens = [2, 10, 11, 12, 13, 14, 15, 1]  # <bos> ... <eos>
    chunks = list(
        pack_document_tokens(
            tokens, document_hash="x", row_index=0, seq_len=4, pad_id=0
        )
    )
    assert len(chunks) == 2
    first, second = chunks
    assert first["input_ids"].tolist() == [2, 10, 11, 12]
    assert first["labels"].tolist() == [10, 11, 12, 13]
    assert first["attention_mask"].tolist() == [1, 1, 1, 1]
    assert second["input_ids"].tolist() == [13, 14, 15, 1]
    assert second["labels"].tolist() == [14, 15, 1, IGNORE_INDEX]
    assert second["attention_mask"].tolist() == [1, 1, 1, 1]


def test_packing_pads_final_incomplete_block() -> None:
    tokens = [2, 10, 11, 1]
    chunks = list(
        pack_document_tokens(
            tokens, document_hash="x", row_index=0, seq_len=8, pad_id=0
        )
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["input_ids"].tolist() == [2, 10, 11, 1, 0, 0, 0, 0]
    assert chunk["labels"].tolist() == [10, 11, 1, -100, -100, -100, -100, -100]
    assert chunk["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    validate_next_token_alignment(
        chunk["input_ids"].unsqueeze(0), chunk["labels"].unsqueeze(0)
    )


def test_misaligned_targets_are_detected() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    bad_labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]])
    with pytest.raises(ValueError, match="alignment"):
        validate_next_token_alignment(input_ids, bad_labels)


def test_loss_ignores_padding_and_final_target() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 16)
    labels = torch.tensor([[2, 3, IGNORE_INDEX, IGNORE_INDEX]])
    loss = next_token_cross_entropy(logits, labels)
    manual = F.cross_entropy(logits[:, :2, :].reshape(-1, 16), labels[:, :2].reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-6)
    changed = logits.clone()
    changed[:, 2:, :] = 1e6
    loss_changed = next_token_cross_entropy(changed, labels)
    assert torch.allclose(loss, loss_changed, atol=1e-6)


def test_loss_is_finite() -> None:
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))
    loss = next_token_cross_entropy(logits, labels)
    assert torch.isfinite(loss)


def test_count_valid_targets() -> None:
    labels = torch.tensor([[1, 2, -100], [-100, 3, 4]])
    assert count_valid_targets(labels) == 4


def test_truncate_labels_to_budget_keeps_row_major_prefix() -> None:
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    truncated, kept = truncate_labels_to_budget(labels, 4)
    assert kept == 4
    assert truncated.tolist() == [[1, 2, -100], [3, 4, -100]]
    # The input is not mutated.
    assert labels.tolist() == [[1, 2, -100], [3, 4, 5]]
    # A budget larger than the valid count keeps everything.
    same, kept_all = truncate_labels_to_budget(labels, 99)
    assert kept_all == 5
    assert torch.equal(same, labels)


def test_drop_leading_valid_targets() -> None:
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    dropped, remaining = drop_leading_valid_targets(labels, 3)
    assert remaining == 2
    assert dropped.tolist() == [[-100, -100, -100], [-100, 4, 5]]
    all_dropped, remaining_zero = drop_leading_valid_targets(labels, 99)
    assert remaining_zero == 0
    assert (all_dropped == IGNORE_INDEX).all()


def test_skip_tokens_consistency(bounded_data) -> None:
    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    tokenizer = bounded_data["tokenizer"]

    def flatten(examples):
        return torch.cat(
            [ex["labels"][ex["labels"] != IGNORE_INDEX] for ex in examples]
        )

    full_examples = list(
        iter_packed_examples(
            config, manifest, tokenizer, "train", max_docs=5, max_chunks=20
        )
    )
    full = flatten(full_examples)
    assert full.numel() > 50
    skip = 37
    skipped_examples = list(
        iter_packed_examples(
            config,
            manifest,
            tokenizer,
            "train",
            max_docs=5,
            max_chunks=20,
            skip_tokens=skip,
        )
    )
    skipped = flatten(skipped_examples)
    assert torch.equal(full[skip:], skipped)


# ---------------------------------------------------------------------------
# B. Model tests
# ---------------------------------------------------------------------------

def test_output_shapes_and_vocabulary(small_model: TRM_LM, batch) -> None:
    logits = small_model(batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (2, 16, 128)
    assert logits.shape[-1] == small_model.config.vocab_size


def test_causal_future_token_leakage(small_model: TRM_LM, batch) -> None:
    small_model.eval()
    with torch.no_grad():
        logits_a = small_model(batch["input_ids"], batch["attention_mask"])
        perturbed = batch["input_ids"].clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % small_model.config.vocab_size
        logits_b = small_model(perturbed, batch["attention_mask"])
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-6)


def test_padding_tokens_do_not_leak(small_model: TRM_LM, padded_batch) -> None:
    small_model.eval()
    with torch.no_grad():
        logits_a = small_model(
            padded_batch["input_ids"], padded_batch["attention_mask"]
        )
        changed = padded_batch["input_ids"].clone()
        changed[:, 12:] = 1  # different pad-like IDs, still masked
        logits_b = small_model(changed, padded_batch["attention_mask"])
    assert torch.allclose(logits_a[:, :12], logits_b[:, :12], atol=1e-6)


def test_puzzle_embedding_configuration(small_model: TRM_LM) -> None:
    assert small_model.config.puzzle_emb_ndim == 0
    assert small_model.config.puzzle_emb_len == 0
    assert not hasattr(small_model, "puzzle_emb")
    with pytest.raises(ValueError):
        TRM_LMConfig(
            vocab_size=16,
            seq_len=8,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            puzzle_emb_ndim=4,
            puzzle_emb_len=1,
        )


def test_forward_loss_is_finite(small_model: TRM_LM, batch) -> None:
    logits = small_model(batch["input_ids"], batch["attention_mask"])
    loss = next_token_cross_entropy(logits, batch["labels"])
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_gradients_are_finite_and_nonzero(small_model: TRM_LM, batch) -> None:
    small_model.zero_grad(set_to_none=True)
    logits = small_model(batch["input_ids"], batch["attention_mask"])
    loss = next_token_cross_entropy(logits, batch["labels"])
    loss.backward()
    grads = [p.grad for p in small_model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_optimizer_updates_a_parameter(small_model: TRM_LM, batch) -> None:
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
    before = small_model.embed_tokens.weight.detach().clone()
    logits = small_model(batch["input_ids"], batch["attention_mask"])
    loss = next_token_cross_entropy(logits, batch["labels"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, small_model.embed_tokens.weight.detach())


def test_n_refine_changes_computation_and_keeps_graph(small_model: TRM_LM, batch) -> None:
    calls = [0] * len(small_model.blocks)
    hooks = [
        block.register_forward_hook(
            lambda module, inputs, output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(small_model.blocks)
    ]
    try:
        with torch.no_grad():
            logits_one = small_model(
                batch["input_ids"], batch["attention_mask"], n_refine=1
            )
        assert calls == [1, 1]
        calls[:] = [0, 0]
        with torch.no_grad():
            logits_two = small_model(
                batch["input_ids"], batch["attention_mask"], n_refine=2
            )
        assert calls == [2, 2]
    finally:
        for hook in hooks:
            hook.remove()
    assert not torch.allclose(logits_one, logits_two, atol=1e-6)

    small_model.zero_grad(set_to_none=True)
    logits = small_model(batch["input_ids"], batch["attention_mask"], n_refine=2)
    loss = next_token_cross_entropy(logits, batch["labels"])
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in small_model.parameters()
        if p.requires_grad
    )


def test_checkpoint_roundtrip(small_model: TRM_LM, batch, tmp_path) -> None:
    small_model.eval()
    with torch.no_grad():
        expected = small_model(batch["input_ids"], batch["attention_mask"])
    path = tmp_path / "state.pt"
    torch.save(small_model.state_dict(), path)
    fresh = TRM_LM(small_model.config)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        actual = fresh(batch["input_ids"], batch["attention_mask"])
    assert torch.allclose(expected, actual, atol=1e-6)


def test_generation_produces_bounded_ids(small_model: TRM_LM) -> None:
    prompt = torch.tensor([[2, 10, 11, 12]])
    generated = small_model.generate(prompt, max_new_tokens=4, eos_token_id=None)
    assert generated.shape == (1, 8)
    assert generated.min() >= 0
    assert generated.max() < small_model.config.vocab_size


@pytest.fixture()
def baseline_config(small_config: TRM_LMConfig) -> TRM_LMConfig:
    return replace(small_config, n_refine=1)


@pytest.fixture()
def baseline_model(baseline_config: TRM_LMConfig) -> CausalTransformerLM:
    torch.manual_seed(0)
    return CausalTransformerLM(baseline_config)


def test_baseline_parameter_match(small_config: TRM_LMConfig) -> None:
    config = replace(small_config, n_refine=1)
    trm = TRM_LM(config)
    baseline = CausalTransformerLM(config)
    assert baseline.num_parameters == trm.num_parameters
    assert len(list(baseline.parameters())) == len(list(trm.parameters()))


def test_baseline_causal_leakage(baseline_model, batch) -> None:
    baseline_model.eval()
    with torch.no_grad():
        logits_a = baseline_model(batch["input_ids"], batch["attention_mask"])
        perturbed = batch["input_ids"].clone()
        perturbed[:, -1] = (
            perturbed[:, -1] + 1
        ) % baseline_model.config.vocab_size
        logits_b = baseline_model(perturbed, batch["attention_mask"])
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-6)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-6)


def test_baseline_forward_backward_updates(baseline_model, batch) -> None:
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3)
    before = baseline_model.embed_tokens.weight.detach().clone()
    logits = baseline_model(batch["input_ids"], batch["attention_mask"])
    loss = next_token_cross_entropy(logits, batch["labels"])
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in baseline_model.parameters()
        if p.requires_grad
    )
    optimizer.step()
    assert not torch.equal(before, baseline_model.embed_tokens.weight.detach())


def test_baseline_rejects_multiple_passes(small_config: TRM_LMConfig) -> None:
    with pytest.raises(ValueError):
        CausalTransformerLM(replace(small_config, n_refine=4))


def test_baseline_checkpoint_roundtrip(baseline_model, batch, tmp_path) -> None:
    baseline_model.eval()
    with torch.no_grad():
        expected = baseline_model(batch["input_ids"], batch["attention_mask"])
    path = tmp_path / "baseline.pt"
    torch.save(baseline_model.state_dict(), path)
    fresh = CausalTransformerLM(baseline_model.config)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        actual = fresh(batch["input_ids"], batch["attention_mask"])
    assert torch.allclose(expected, actual, atol=1e-6)


def test_baseline_generation(baseline_model) -> None:
    prompt = torch.tensor([[2, 10, 11, 12]])
    generated = baseline_model.generate(prompt, max_new_tokens=4, eos_token_id=None)
    assert generated.shape == (1, 8)
    assert generated.min() >= 0
    assert generated.max() < baseline_model.config.vocab_size


def _gpu_device() -> torch.device:
    if os.environ.get("TRM_LM_TEST_GPU") != "1":
        pytest.skip("set TRM_LM_TEST_GPU=1 to run the opt-in GPU tests")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", ""))
    if (
        "gfx1032" in arch
        and "gfx1032" not in torch.cuda.get_arch_list()
        and "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    ):
        pytest.skip(
            "gfx1032 needs an opt-in HSA_OVERRIDE_GFX_VERSION=10.3.0 runtime"
        )
    return torch.device("cuda:0")


@pytest.mark.skipif(
    os.environ.get("TRM_LM_TEST_GPU") != "1", reason="opt-in GPU test"
)
def test_gpu_forward_backward_fp32(small_config: TRM_LMConfig, batch) -> None:
    device = _gpu_device()
    props = torch.cuda.get_device_properties(device)
    print(
        f"\n[gpu] name={props.name} arch={getattr(props, 'gcnArchName', '?')} "
        f"hip={torch.version.hip} vram={props.total_memory / 1e9:.2f} GB",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    model = TRM_LM(small_config).to(device)
    batch_dev = {key: value.to(device) for key, value in batch.items()}
    logits = model(batch_dev["input_ids"], batch_dev["attention_mask"])
    loss = next_token_cross_entropy(logits, batch_dev["labels"])
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    print(
        f"[gpu] fp32 loss={float(loss.detach()):.4f} "
        f"peak_alloc={torch.cuda.max_memory_allocated(device) / 1e9:.3f} GB",
        flush=True,
    )


@pytest.mark.skipif(
    os.environ.get("TRM_LM_TEST_GPU") != "1", reason="opt-in GPU test"
)
def test_gpu_forward_backward_bf16(small_config: TRM_LMConfig, batch) -> None:
    device = _gpu_device()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this device")
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    model = TRM_LM(small_config).to(device=device, dtype=torch.bfloat16)
    batch_dev = {key: value.to(device) for key, value in batch.items()}
    logits = model(batch_dev["input_ids"], batch_dev["attention_mask"])
    loss = next_token_cross_entropy(logits, batch_dev["labels"])
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    print(
        f"[gpu] bf16 loss={float(loss.detach()):.4f} "
        f"peak_alloc={torch.cuda.max_memory_allocated(device) / 1e9:.3f} GB",
        flush=True,
    )


@pytest.mark.skipif(
    os.environ.get("TRM_LM_TEST_GPU") != "1", reason="opt-in GPU test"
)
def test_gpu_causal_and_padding_masks(small_config: TRM_LMConfig, batch) -> None:
    device = _gpu_device()
    torch.manual_seed(0)
    model = TRM_LM(small_config).to(device)
    batch_dev = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        baseline = model(batch_dev["input_ids"], batch_dev["attention_mask"])
        future = batch_dev["input_ids"].clone()
        future[:, -1] = (future[:, -1] + 1) % small_config.vocab_size
        perturbed = model(future, batch_dev["attention_mask"])
    causal_diff = float((baseline[:, :-1] - perturbed[:, :-1]).abs().max())

    padded_inputs = batch_dev["input_ids"].clone()
    padded_inputs[:, 12:] = 0
    padded_mask = batch_dev["attention_mask"].clone()
    padded_mask[:, 12:] = 0
    changed_pads = padded_inputs.clone()
    changed_pads[:, 12:] = 1
    with torch.no_grad():
        padded_a = model(padded_inputs, padded_mask)
        padded_b = model(changed_pads, padded_mask)
    pad_diff = float((padded_a[:, :12] - padded_b[:, :12]).abs().max())

    print(
        f"\n[gpu] causal_earlier_diff={causal_diff:.3e} "
        f"padding_real_diff={pad_diff:.3e}",
        flush=True,
    )
    assert causal_diff < 1e-5
    assert pad_diff < 1e-5


# ---------------------------------------------------------------------------
# C. Integration test
# ---------------------------------------------------------------------------
def test_end_to_end_bounded_smoke(bounded_data, tmp_path) -> None:
    config = bounded_data["config"]
    manifest = bounded_data["manifest"]
    tokenizer = bounded_data["tokenizer"]
    vocab_size = tokenizer.get_vocab_size()

    model_config = TRM_LMConfig(
        vocab_size=vocab_size,
        seq_len=64,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        expansion=4.0,
        n_refine=2,
        tie_embeddings=True,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
    )
    model = TRM_LM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    packed = list(
        iter_packed_examples(
            config, manifest, tokenizer, "train", max_docs=8, max_chunks=8
        )
    )
    assert len(packed) >= 2
    batch = collate_lm_batch(packed[:2])

    model.train()
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = next_token_cross_entropy(logits, batch["labels"])
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    before = model.embed_tokens.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.embed_tokens.weight.detach())

    checkpoint_path = tmp_path / "smoke.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.config_dict(),
            "step": 1,
        },
        checkpoint_path,
    )
    reloaded = TRM_LM(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(checkpoint["model"])
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        expected = model(batch["input_ids"], batch["attention_mask"])
        actual = reloaded(batch["input_ids"], batch["attention_mask"])
    assert torch.allclose(expected, actual, atol=1e-6)

    generated = reloaded.generate(
        batch["input_ids"][:1, :4], max_new_tokens=4,
        eos_token_id=tokenizer.token_to_id("<eos>"),
    )
    assert generated.shape[1] <= 8
    assert generated.max() < vocab_size


# ---------------------------------------------------------------------------
# D. Reference implementation regression
# ---------------------------------------------------------------------------

def test_reference_files_unchanged() -> None:
    with open(REFERENCE_HASHES_PATH, "r", encoding="utf-8") as handle:
        expected = json.load(handle)
    for relative_path, expected_hash in expected.items():
        path = REPO_ROOT / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected_hash, f"{relative_path} changed"


def test_reference_trm_forward_backward_still_works() -> None:
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    config = {
        "batch_size": 2,
        "seq_len": 32,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 64,
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 64,
        "expansion": 4,
        "num_heads": 2,
        "pos_encodings": "rope",
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "mlp_t": False,
        "puzzle_emb_len": 0,
        "no_ACT_continue": True,
    }
    torch.manual_seed(0)
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.train()
    inputs = torch.randint(0, 64, (2, 32))
    labels = torch.full((2, 32), IGNORE_INDEX, dtype=torch.long)
    labels[:, :-1] = inputs[:, 1:]
    batch = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32),
    }
    with torch.device("cpu"):
        carry = model.initial_carry(batch)
    _, outputs = model(carry=carry, batch=batch)
    loss = F.cross_entropy(
        outputs["logits"].reshape(-1, 64), labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # The q-head is not part of the logits loss when halt_max_steps=1, so it is
    # allowed to have no gradient; every participating parameter must be finite.
    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
