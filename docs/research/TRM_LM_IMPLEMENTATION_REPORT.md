# TRM-LM Implementation Report

> **Post-implementation update (2026-10-21):** A separate 5M-token pilot was
> subsequently authorized and completed. The statements below describing
> “substantive training not authorized/not run” are historical for the
> implementation phase; see `TRM_LM_PILOT_REPORT.md` for the pilot. The original
> Samsung TRM reference remains unchanged.

**Date:** 2026-10-21
**Repository:** `/home/chandrabhan/Documents/Chandrabhan/TinyRecursiveModels`
**Branch / commit:** `main` @ `c01103738605ba39d1430519b1ee0c62f4c707f8`
**Scope:** implementation + bounded verification only.
**Substantive training:** **NOT AUTHORIZED AND NOT RUN.**

---

## 1. Architecture and file changes

### 1.1 New files

| File | Lines | Purpose |
|---|---:|---|
| `models/recursive_reasoning/trm_lm.py` | 286 | Causal TRM-LM model variant: token embedding, masked causal attention, shared recursive block stack, tied LM head, greedy generation. |
| `dataset/tinystories_lm.py` | 541 | Dedicated TinyStories pipeline: Parquet streaming, split/duplicate manifest, BPE training, document-level chunking, next-token packing, collation, CE, alignment validation. |
| `train_trm_lm.py` | 508 | Minimal training/verification entrypoint: device/dtype checks, AdamW/SGD, gradient clipping, optional cosine LR, checkpoint/resume, validation, generation, run summary. |
| `tests/test_trm_lm.py` | 720 | Dataset, model, integration, GPU opt-in, and reference-regression tests. |
| `tests/reference_hashes.json` | 18 | SHA-256 baseline for 16 reference files; used by the preservation test. |
| `requirements-trm-lm.txt` | 16 | Supplemental deps for an isolated environment that inherits the system ROCm PyTorch build. |
| `docs/research/TRM_LM_IMPLEMENTATION_REPORT.md` | this file | Implementation evidence and limitations. |

**No reference implementation file was modified.** At implementation time `git diff --stat` was empty and the new files were untracked; the follow-up adaptation/pilot commit now contains them. No commit was created during the implementation phase.

**Pilot-phase additions (later):** `models/recursive_reasoning/causal_transformer_lm.py`, token-budget helpers in `dataset/tinystories_lm.py`, token-budget/interval logic in `train_trm_lm.py`, and baseline/token-accounting tests. See `TRM_LM_PILOT_REPORT.md`.

### 1.2 Execution path

```
Parquet row groups (pyarrow, streaming)
  → validate_text / document_id / split_name_for_row
  → manifest.json + split_ids.npy          (document-level split + duplicate policy)
  → byte-level BPE trained on train rows only
  → <bos> document <eos> tokens
  → fixed-length chunks of 256 tokens      (final chunk padded + masked)
  → labels[t] = tokens[chunk_start + t + 1] (final/pad positions = -100)
  → TRM_LM forward:
        x = Embedding(input_ids)
        z = x
        repeat n_refine times:
            z = Blocks(z + x)               (2 shared post-norm blocks)
        logits = tied LM head(RMSNorm(z))
  → F.cross_entropy(logits.float(), labels, ignore_index=-100)
  → backward → clip → AdamW/SGD step
  → checkpoint: model, optimizer, scheduler, step, configs, tokenizer reference,
                manifest, RNG states
```

### 1.3 Architecture details

- `TRM_LMCausalSelfAttention` uses `scaled_dot_product_attention` with an **explicit boolean mask** `[B, 1, L, L]`:
  `mask[b, i, j] = (j <= i) AND attention_mask[b, j]`.
  `is_causal=False` is passed deliberately; the explicit mask is the sole attention control, so mask polarity is testable. There is no alternate attention path.
- RoPE is applied to queries and keys (`RotaryEmbedding` / `apply_rotary_pos_emb` from the reference `models/layers.py`).
- `TRM_LMBlock` follows the reference block order: post-norm attention, then post-norm SwiGLU MLP.
- `TRM_LM` applies the **same** block stack `n_refine` times with input injection (`z <- Blocks(z + x)`). Parameters are shared across passes. All passes remain in the autograd graph; there is no `torch.no_grad()` and no detached carry.
- The LM head is tied to the token embedding (`lm_head.weight = embed_tokens.weight`).
- No ACT, no q-head, no q-halt loss, no puzzle embedding, no puzzle identifiers.
- `puzzle_emb_ndim` and `puzzle_emb_len` must both be 0; the config rejects nonzero values.
- `n_refine` is the number of **full block-stack applications** per forward pass. With `num_layers=2`, `n_refine=2` means four block applications and `n_refine=4` means eight.

---

## 2. Reference Samsung TRM vs `trm_lm`

| Property | Reference Samsung TRM (`models/recursive_reasoning/trm.py`) | `trm_lm` (`models/recursive_reasoning/trm_lm.py`) |
|---|---|---|
| Attention | `causal=False` in every block | Explicit causal + padding mask |
| Latent states | Two states, `z_H` and `z_L` | One latent `z` |
| Refinement | `H_cycles` / `L_cycles`; only final H cycle has gradients | `n_refine` full block-stack passes; every pass differentiable |
| Input injection | `z_H + input_embeddings` | `z + input_embeddings` at each pass |
| Puzzle embeddings | Supported / used in ARC/Sudoku configs | Removed; both dims must be 0 |
| ACT / q-head | Present | Removed |
| Loss | `stablemax_cross_entropy` + `0.5 * q_halt_loss` | Token CE only |
| Data | `PuzzleDataset` (`.npy` + `dataset.json`) | Dedicated TinyStories pipeline (Parquet → BPE → packed blocks) |
| Training entrypoint | `pretrain.py` (Hydra, AdamATan2) | `train_trm_lm.py` (argparse, AdamW/SGD) |

The reference model is untouched. The `trm_lm` module is a new research variant, not a claim that the reference TRM is language-capable. The single-latent design is a documented simplification: the original two-state hierarchy was motivated by structured-task search; the retained TRM property is repeated refinement through shared parameters with input injection. This is not a conventional stack of distinct transformer layers.

---

## 3. Parameter count and configuration

Default configuration measured in the target-scale probe:

| Field | Value |
|---|---|
| `vocab_size` | 8,192 |
| `seq_len` | 256 |
| `hidden_size` | 512 |
| `num_layers` | 2 |
| `num_heads` | 8 |
| `expansion` | 4.0 |
| `n_refine` | 1 (2 and 4 tested for compute/memory) |
| `rope_theta` | 10,000 |
| `rms_norm_eps` | 1e-5 |
| `tie_embeddings` | True |
| `pad_token_id` | 0 |
| `puzzle_emb_ndim` / `puzzle_emb_len` | 0 / 0 |
| **Trainable parameters** | **11,010,048** |
| Parameter tensors | 9 (tied embedding counted once) |

Parameter breakdown:

| Component | Parameters |
|---|---:|
| `embed_tokens.weight` (tied LM head) | 4,194,304 |
| Each block: QKV 786,432 + O 262,144 + gate_up 1,572,864 + down 786,432 | 3,407,872 |
| Two blocks | 6,815,744 |
| **Total** | **11,010,048** |

The 8–15M target is met without removing required components. A small 655,360-parameter configuration (`hidden=128`, `layers=2`, `vocab=1024`, `seq=64`) was used for fast CPU/GPU tests.

---

## 4. Dataset and tokenizer provenance

### 4.1 Dataset

| Item | Observed |
|---|---|
| Path | `/home/chandrabhan/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet` |
| Size | **672,825,299 bytes** (matches audit) |
| SHA-256 | `8bacd849e57784e06ebaaab3ef7b01077ca2d9d27ce7d95f2fbe465f1483548b` (matches audit / Hub `lfs_sha256`) |
| Rows | **2,732,634** (matches audit) |
| Columns | `["text"]` |
| Format | Parquet, zstd, 2,669 row groups |

`verify_parquet()` is called at the start of every run and reports size/row matches against the audit constants. No download was attempted.

### 4.2 Bounded manifest counts

| Scan | Examined | Test | Val | Train | Rejected | Duplicates |
|---|---:|---:|---:|---:|---:|---:|
| 25,000 rows (CPU/GPU small smoke) | 25,000 | 10,000 | 10,000 | 5,000 | 0 | 0 |
| 40,000 rows (target-scale smoke) | 40,000 | 10,000 | 10,000 | 20,000 | 0 | 0 |

The audit's full-file scan found 0 duplicates and 0 malformed rows; the bounded scans agree. The duplicate/malformed policy is also tested with a synthetic Parquet containing duplicate and malformed rows (`test_duplicate_and_malformed_policy`).

### 4.3 Tokenizer

- Byte-level BPE trained from scratch with `tokenizers`.
- Default target vocabulary: **8,192**; actual vocabulary recorded in `tokenizer_config.json` (the trainer may return fewer tokens on small corpora).
- Special tokens and IDs: `<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3`.
- Training split: **train only**; the tokenizer training counts are recorded (`train_docs=20,000` for the target-scale smoke; 500 for the bounded test fixture).
- No pretrained tokenizer and no pretrained model weights.
- Artifacts: `tokenizer.json` and `tokenizer_config.json` in the run workdir; the checkpoint embeds `tokenizer_config` and the tokenizer path. Temporary smoke workdirs were used, not committed artifacts.

**Deviation:** the target-scale smoke trained the tokenizer on 20,000 training documents, not the entire 2.7M-row train split, because the run is bounded. The config records this. The first authorized experiment should decide the final tokenizer budget.

---

## 5. Split and sequence-packing policy

### 5.1 Split policy

- **Document-level**, deterministic **row-index** split matching the dataset card:
  - test: rows `0..9,999`
  - val: rows `10,000..19,999`
  - train: rows `20,000..end`
- Recorded seed: `0`; split policy name: `row_index`.
- Every row is one document. No document is split across train/val/test.
- Validation documents are never used for tokenizer training (`test_tokenizer_is_trained_on_train_only`).
- Duplicate policy: first occurrence by row index wins; later occurrences are rejected, including cross-split duplicates. Counts and reasons are stored in `manifest.json`.
- Persisted artifacts: `manifest.json`, `split_ids.npy` (int8 per row: `-1` rejected, `0` test, `1` val, `2` train), a SHA-256 of `split_ids.npy`, and per-split samples of document hashes.

### 5.2 Packing policy

- Each document is tokenized independently as `<bos> document <eos>`.
- Consecutive chunks of `seq_len` tokens are emitted.
- **No cross-document attention** is possible: every sample contains tokens from exactly one document.
- **No cross-chunk attention**: each chunk is a separate sample. This is a deliberate limitation; context per prediction is at most 256 tokens, and long documents are represented as independent chunks.
- The final incomplete block is **padded**, not dropped:
  - inputs padded with `<pad>`
  - labels padded with `-100`
  - attention mask 0 for padded positions
- The model's explicit mask prevents padded keys from influencing real positions.

---

## 6. Autoregressive loss and target-shift semantics

For a document token sequence `T = [<bos>, d0, d1, ..., <eos>]` and a chunk starting at `s`:

```
input_ids[t]  = T[s + t]          for t in [0, L)
labels[t]     = T[s + t + 1]      for t in [0, L - 1)
labels[L-1]   = -100              if no next token exists
labels[t]     = -100              for padded positions
attention_mask[t] = 1             for real chunk positions, else 0
```

Shapes:

```
input_ids:      [B, L]     int64
labels:         [B, L]     int64   (-100 = ignore)
attention_mask: [B, L]     int64   (1 real, 0 pad)
logits:         [B, L, V]  float
```

Loss:

```python
F.cross_entropy(
    logits.float().reshape(-1, V),
    labels.reshape(-1),
    ignore_index=-100,
)
```

`F.cross_entropy` applies log-softmax internally; no softmax is applied before the loss. The mean is over valid (non-`-100`) targets. The last position without a target and all padding targets are excluded.

`validate_next_token_alignment()` enforces:

```
labels[:, :-1] == input_ids[:, 1:]   wherever labels[:, :-1] != -100
```

Tests covering this section:

- `test_packing_boundaries_and_final_block`
- `test_packing_pads_final_incomplete_block`
- `test_misaligned_targets_are_detected`
- `test_loss_ignores_padding_and_final_target`
- `test_loss_is_finite`

---

## 7. Causal leakage test results

The attention mask is an explicit boolean matrix passed to `scaled_dot_product_attention`:

```
causal   = tril(ones(L, L))
key_ok   = attention_mask.bool()
mask     = causal.unsqueeze(0) & key_ok.unsqueeze(1)   # [B, L, L] -> [B, 1, L, L]
```

`True` means the key participates. There is no `is_causal=True` shortcut and no alternative attention path.

Measured leakage:

| Check | Device / dtype | Result |
|---|---|---|
| Perturb final token; compare earlier logits (positions `0..L-2`) | CPU float32 | **max abs diff = 0.0** |
| Same perturbation; last position (sanity check that the perturbation is real) | CPU float32 | max abs diff = 0.8911 |
| Change masked pad token IDs; compare real-position logits | CPU float32 | **max abs diff = 0.0** |
| Perturb final token; compare earlier logits | ROCm GPU float32 | **max abs diff = 0.0** |
| Change masked pad token IDs; compare real-position logits | ROCm GPU float32 | **max abs diff = 0.0** |

The CPU tests use `atol=1e-6`; the GPU mask test uses `atol=1e-5` and measured exactly `0.000e+00`. This verifies the invariant on the actual ROCm SDPA backend, not only by inspecting the `causal=True` argument.

Test names:

- `test_causal_future_token_leakage`
- `test_padding_tokens_do_not_leak`
- `test_gpu_causal_and_padding_masks` (opt-in)

---

## 8. Gradient-flow and checkpoint tests

### 8.1 Gradient flow

- The refinement loop has no `torch.no_grad()` and no detached carry. Every `n_refine` pass is in the graph.
- `test_n_refine_changes_computation_and_keeps_graph` registers forward hooks on both blocks:
  - `n_refine=1` → each block called once (`[1, 1]`)
  - `n_refine=2` → each block called twice (`[2, 2]`)
  - logits differ between the two pass counts
  - after backward with `n_refine=2`, every trainable parameter has a finite gradient
- `test_gradients_are_finite_and_nonzero` checks finite, nonzero gradients for the shared model.
- `test_optimizer_updates_a_parameter` confirms an AdamW step changes a parameter.
- The target-scale probe confirmed 11,010,048 parameters and finite losses at `n_refine=1,2,4`.

### 8.2 Checkpointing

Checkpoint contents (`torch.save`):

- model state dict
- optimizer state dict
- scheduler state dict (or `None`)
- global step
- model config (includes `n_refine`, vocabulary, sequence length, layer/head counts, tied-embedding flag)
- training config (includes device, dtype, torch/HIP versions, parquet metadata)
- tokenizer config
- split manifest
- Python / NumPy / Torch / CUDA RNG states
- device string at save

Tests and smoke results:

- `test_checkpoint_roundtrip`: reload reproduces logits within `atol=1e-6`.
- CPU smoke: `checkpoint_max_abs_diff = 0.0`.
- GPU bf16 target-scale smoke: `checkpoint_max_abs_diff = 0.0`.
- Resume test: loaded `step_1.pt` at step 1, ran step 2, saved `step_2.pt`, round-trip diff `0.0`.
- Resume refuses a checkpoint whose `model_config` differs from the current config.

`torch.load(..., weights_only=False)` is used because the checkpoint intentionally contains RNG states and config dictionaries; checkpoints are local, trusted artifacts.

---

## 9. Hardware, backend, and environment results

### 9.1 Host and software

| Item | Observed |
|---|---|
| OS / kernel | CachyOS Linux, `7.2.8-1-cachyos`, x86_64 |
| CPU | AMD Ryzen 7 5800H, 16 threads |
| RAM | 14 GiB observed, ~5.4 GiB available, 12 GiB swap |
| GPU | AMD Radeon RX 6600M, `gfx1032`, 8.57 GB VRAM |
| Python | 3.14.7 (`/usr/bin/python3`), no system `pip` |
| PyTorch | 2.14.0 ROCm, `torch.version.hip = 7.2.53211` |
| Triton | 3.5.1 |
| Test env | isolated scratch venv with `--system-site-packages`; `pyarrow 26.0.0`, `tokenizers 0.23.3`, `einops 0.8.2`, `pytest 9.1.1` |
| Repo env file | `requirements-trm-lm.txt` (does not pin/reinstall torch) |

### 9.2 GPU compatibility

- `torch.cuda.is_available()` is `True` (ROCm exposes the CUDA API namespace).
- The installed Torch arch list contains `gfx1030`, `gfx1031`, `gfx1035`, `gfx1036`, but **not `gfx1032`**.
- Without an override, a trivial GPU kernel aborts with the HIP `StatCO::getStatFunc` assertion recorded in the audit.
- With the opt-in `HSA_OVERRIDE_GFX_VERSION=10.3.0`, the device reports as `gfx1030` and all GPU tests pass. This is an **unofficial ISA mapping**, not official RX 6600M support. It was set only in the test command's environment; no global shell or driver changes were made.
- The training entrypoint detects `gfx1032` without the override and refuses to start, instead of hard-aborting.

### 9.3 Memory and throughput

Target-scale forward + backward (no optimizer step), `batch=4`, `seq=256`, `hidden=512`, `vocab=8192`, bf16, GPU:

| `n_refine` | Loss | Time per step | Peak allocated | Reserved |
|---:|---:|---:|---:|---:|
| 1 | 9.2842 | 0.081 s | 0.281 GB | 0.336 GB |
| 2 | 9.1188 | 0.140 s | 0.355 GB | 0.390 GB |
| 4 | 9.1119 | 0.323 s | 0.493 GB | 0.528 GB |

Target-scale one-step training smoke (`batch=1`, `seq=256`, `n_refine=1`, bf16):

- parameters: **11,010,048**
- train loss: **9.264048**; val loss: **9.078260**
- peak allocated: **0.269 GB**; peak reserved: **0.294 GB**
- checkpoint round-trip: **0.0**
- total wall time: **1.19 s** measured from just before the training loop through one optimizer step, validation, checkpoint save, reload, and generation (tokenizer training and model initialization occur before this timer).
- generated text after one step is gibberish (expected); no language-quality claim is made.

Small configuration (`hidden=128`, `vocab=1024`, `seq=64`, `n_refine=2`):

- CPU float32: params 655,360; train loss 6.978647; val 6.946341; checkpoint diff 0.0; elapsed 0.321 s.
- GPU bf16: params 655,360; train loss 6.979235; val 6.946947; checkpoint diff 0.0.

All measurements are well within 8 GB VRAM. No microbatch-8 or longer-sequence run was performed; do not assume those fit without re-measurement.

---

## 10. Tests executed, pass/fail counts, known failures

### 10.1 Commands

```bash
# Environment used for execution (scratch path differs; this is the reproducible form)
uv venv --python /usr/bin/python3 --system-site-packages .venv-trm-lm
uv pip install --python .venv-trm-lm/bin/python -r requirements-trm-lm.txt

# Full suite, CPU
python -m pytest tests/test_trm_lm.py -q -p no:warnings

# Full suite, opt-in GPU tests
TRM_LM_TEST_GPU=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  python -m pytest tests/test_trm_lm.py -q -p no:warnings -s

# Bounded CPU smoke: one optimizer step, checkpoint reload, generation
python train_trm_lm.py --workdir /tmp/trmlm_cpu --device cpu \
  --max-steps 1 --max-train-docs 32 --max-val-docs 8 \
  --max-scan-rows 25000 --tokenizer-docs 300 --tokenizer-vocab-size 1024 \
  --seq-len 64 --batch-size 2 --hidden-size 128 --num-layers 2 \
  --num-heads 4 --n-refine 2 --smoke-test

# Bounded target-scale GPU smoke: one optimizer step, 11.0M params
HSA_OVERRIDE_GFX_VERSION=10.3.0 python train_trm_lm.py --workdir /tmp/trmlm_gpu \
  --device cuda --dtype bfloat16 --max-steps 1 --max-train-docs 8 \
  --max-val-docs 4 --max-scan-rows 40000 --tokenizer-docs 20000 \
  --tokenizer-vocab-size 8192 --seq-len 256 --batch-size 1 \
  --hidden-size 512 --num-layers 2 --num-heads 8 --n-refine 1 --smoke-test

# Resume verification
python train_trm_lm.py --workdir /tmp/trmlm_cpu --resume /tmp/trmlm_cpu/checkpoints/step_1.pt \
  --device cpu --max-steps 2 --max-train-docs 32 --max-val-docs 8 \
  --max-scan-rows 25000 --tokenizer-docs 300 --tokenizer-vocab-size 1024 \
  --seq-len 64 --batch-size 2 --hidden-size 128 --num-layers 2 \
  --num-heads 4 --n-refine 2 --smoke-test
```

### 10.2 Results

| Run | Result |
|---|---|
| CPU suite | **26 passed, 3 skipped** in 4.47 s (the 3 GPU tests are opt-in skips) |
| GPU suite | **29 passed** in 4.42 s |
| GPU fp32 forward/backward | loss 4.9129, peak 0.069 GB |
| GPU bf16 forward/backward | loss 4.9131, peak 0.068 GB |
| GPU causal/padding mask check | `causal_earlier_diff=0.000e+00`, `padding_real_diff=0.000e+00` |
| CPU smoke (1 step) | train 6.978647, val 6.946341, checkpoint diff 0.0, generation OK |
| CPU SGD smoke (1 step) | train 6.941362, val 6.949195, checkpoint diff 0.0, generation OK |
| GPU small smoke (1 step) | train 6.979235, val 6.946947, checkpoint diff 0.0 |
| GPU target-scale smoke (1 step) | 11,010,048 params, train 9.264048, val 9.078260, peak 0.269 GB, checkpoint diff 0.0 |
| Resume | step 1 loaded, step 2 ran, checkpoint diff 0.0 |
| Reference hashes | **16/16 unchanged**; `git diff --stat` empty |
| Reference TRM behavioral test | original model forward/backward finite; participating parameters have finite gradients |

### 10.3 Test inventory (29 collected)

Dataset (13): `test_parquet_opens_and_matches_audit`, `test_text_extraction_returns_valid_strings`, `test_malformed_record_validation`, `test_split_policy_is_deterministic_and_disjoint`, `test_manifest_counts_and_split_ids`, `test_duplicate_and_malformed_policy`, `test_tokenizer_is_trained_on_train_only`, `test_token_ids_fit_vocabulary`, `test_packing_boundaries_and_final_block`, `test_packing_pads_final_incomplete_block`, `test_misaligned_targets_are_detected`, `test_loss_ignores_padding_and_final_target`, `test_loss_is_finite`.

Model (10): `test_output_shapes_and_vocabulary`, `test_causal_future_token_leakage`, `test_padding_tokens_do_not_leak`, `test_puzzle_embedding_configuration`, `test_forward_loss_is_finite`, `test_gradients_are_finite_and_nonzero`, `test_optimizer_updates_a_parameter`, `test_n_refine_changes_computation_and_keeps_graph`, `test_checkpoint_roundtrip`, `test_generation_produces_bounded_ids`.

GPU opt-in (3): `test_gpu_forward_backward_fp32`, `test_gpu_forward_backward_bf16`, `test_gpu_causal_and_padding_masks`.

Integration (1): `test_end_to_end_bounded_smoke`.

Reference (2): `test_reference_files_unchanged`, `test_reference_trm_forward_backward_still_works`.

### 10.4 Known failures and limitations

- **No test failures** in the executed CPU or GPU suites.
- The repository shipped **no pre-existing test suite**; there was no original `tests/` directory. The reference regression coverage added here consists of the 16-file SHA-256 preservation test and a behavioral forward/backward test of the original `TinyRecursiveReasoningModel_ACTV1`. The historical Samsung test suite could not be run because it is not present in this checkout.
- Hydra 1.3.x remains incompatible with Python 3.14; `pretrain.py` was not used. This is why the new entrypoint uses `argparse`.
- `adam-atan2` still lacks its backend extension in the available PyPI install; `trm_lm` uses AdamW (with optional SGD) as authorized.
- GPU execution requires the opt-in `HSA_OVERRIDE_GFX_VERSION=10.3.0`; this is unofficial.
- No full training, convergence, language-quality, long-run stability, distributed, or memory-at-larger-batch test was run.

---

## 11. Deviations from the requested plan

1. **Single latent instead of `z_H`/`z_L`.** The reference hierarchy is collapsed to one latent `z`; the retained TRM mechanism is repeated shared-parameter refinement with input injection. This is documented in the module docstring and Section 2. It is not an ordinary transformer: the same block stack is applied recurrently, not a stack of distinct layers.
2. **No Hydra.** `train_trm_lm.py` uses `argparse` and dataclasses because the pinned Hydra versions fail on Python 3.14. The reference `pretrain.py` is untouched.
3. **No puzzle-dataset reuse.** A dedicated TinyStories pipeline was added instead of disguising text as `.npy` puzzle arrays.
4. **Tokenizer budget.** The target-scale smoke trained the tokenizer on 20,000 training documents, not the full train split, to keep the run bounded. The tokenizer config records the exact count. The final tokenizer budget is an experiment decision.
5. **Deterministic row order.** The pipeline does not shuffle. This favors reproducibility for bounded tests; a full experiment may want a seeded shuffle policy, which is not implemented.
6. **Row-index split only.** A hash-based split was not implemented; the card's deterministic row-index split is used and recorded.
7. **Per-document chunks, no cross-chunk attention.** Long documents are independent 256-token chunks. This avoids cross-document leakage but limits context. Cross-document attention is explicitly not permitted.
8. **Final block padded, not dropped.** Padding is masked; labels are `-100`.
9. **Tied embeddings.** The initial configuration ties the LM head to the token embedding, reducing parameters to 11.0M.
10. **`n_refine` semantics.** A pass is one full block-stack application (two layers), not one layer. The hook test verifies the call count.
11. **No ACT/q-halt.** The reference ACT objective is intentionally absent.
12. **AdamW initial optimizer.** SGD is available as a configurable alternative; no scheduler is used by default (optional cosine is implemented).
13. **No committed environment.** The executing venv lived in a scratch directory; `requirements-trm-lm.txt` documents the reproducible setup. No system Python, drivers, or GPU stack were changed.
14. **No config YAML.** The model configuration is stored in the checkpoint and `model_config.json`; a Hydra `config/arch/trm_lm.yaml` was not added.

---

## 12. Substantive training remains unauthorized

**No substantive pretraining or long-running training was started.** The only training activity was:

- one optimizer step in the CPU smoke test,
- one optimizer step in the GPU small smoke test,
- one optimizer step in the target-scale GPU smoke test,
- one additional resume step (step 2) in the CPU resume test.

These are bounded verification runs. They do not establish language competence, convergence, scaling behavior, or the value of recursive refinement. The generated text after one step is gibberish and is reported only to prove that autoregressive generation executes.

### Changed files

**Added at implementation time (subsequently committed):**
- `models/recursive_reasoning/trm_lm.py`
- `dataset/tinystories_lm.py`
- `train_trm_lm.py`
- `tests/test_trm_lm.py`
- `tests/reference_hashes.json`
- `requirements-trm-lm.txt`
- `docs/research/TRM_LM_IMPLEMENTATION_REPORT.md`

**Modified:** none. Reference hashes 16/16 unchanged; `git diff --stat` is empty.

### Remaining blockers before the first authorized training experiment

1. **Research authorization.** The current authorization covers implementation and bounded verification only. A substantive run requires a separate approval.
2. **Freeze the experiment configuration.** Decide and record:
   - data budget (e.g., 20–100M tokens rather than the full 534.9M-token corpus),
   - tokenizer training budget (more than 20,000 documents?),
   - `n_refine` (1/2/4),
   - sequence length/batch size (start at 256/1–4 and measure),
   - optimizer/LR schedule and validation cadence,
   - evaluation protocol and baseline.
3. **Environment confirmation.** Create the isolated environment from `requirements-trm-lm.txt`, verify the ROCm device, and decide whether to keep the Python 3.14 + system Torch + override path or use a supported Python 3.10–3.12 ROCm wheel. The current path is verified but the gfx1032 override is unofficial.
4. **No code blocker remains for a bounded pilot.** The data pipeline, causal model, loss, gradients, checkpointing, resume, and generation have all been exercised on the real local dataset.

**Exact next step:** freeze the experiment configuration above and obtain the separate training authorization; then run a short pilot (not the full corpus) with `train_trm_lm.py` without `--smoke-test` but with a small explicit step budget. Do not launch a substantive run before that approval.
