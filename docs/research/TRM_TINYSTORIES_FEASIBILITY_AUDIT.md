# TRM × TinyStories Feasibility Audit

> **Post-audit update (2026-10-21):** The recommended adaptation was implemented
> as the separate `trm_lm` variant, and a bounded 5M-token pilot was completed.
> This audit remains the historical pre-implementation assessment. See
> `TRM_LM_IMPLEMENTATION_REPORT.md` and `TRM_LM_PILOT_REPORT.md`. The original
> Samsung TRM reference implementation was not modified.

**Audit date:** 2026-10-21
**Scope:** verification and feasibility only. No architecture changes, no training run, no commits during the audit.
**Repository:** `/home/chandrabhan/Documents/Chandrabhan/TinyRecursiveModels`
**Upstream:** `https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git`
**Branch / commit:** `main` @ `c01103738605ba39d1430519b1ee0c62f4c707f8` (2026-03-31), working tree clean, 0 ahead / 0 behind `origin/main`.
**Dataset:** `karpathy/tinystories-gpt4-clean`, snapshot `0397e27157956705a0260709da3095bb9c43d6a7`.

---

## A. Executive verdict

**Primary outcome: `REQUIRES ARCHITECTURAL ADAPTATION`**

The Samsung TRM reference implementation cannot be trained on TinyStories in its current form. It is a **non-autoregressive, bidirectional, puzzle-structured model**, not a causal language model:

- Every TRM/HRM/transformer-baseline variant hardcodes `causal=False` in its attention block (`models/recursive_reasoning/trm.py:82`, `hrm.py:77`, `trm_singlez.py:82`, `trm_hier6.py:88`, `transformers_baseline.py:83`), and `pretrain.py:123` explicitly passes `causal=False  # Non-autoregressive`.
- A bounded probe on the real model showed that changing the final input token changed the logits at position 0 by **0.68 max-abs** (bf16). Future-token information therefore leaks into earlier positions. Training on shifted next-token targets with this attention would be label-leaking and would not produce a valid autoregressive model.
- The existing `Attention` class already supports causal mode (`models/layers.py:100`, `:132`); a scratch-only monkey-patch setting `causal=True` on the two attention modules reduced the position-0 leakage to **exactly 0.0**. The missing piece is wiring, not a new kernel.
- The data path is entirely non-textual: `PuzzleDataset` loads pre-built `.npy` arrays plus `dataset.json` (`puzzle_dataset.py:115-144`). It never reads Parquet, has no tokenizer, and its vocabularies are task-specific (11 tokens for Sudoku, 12 for ARC: `dataset/build_sudoku_dataset.py:132-133`, `dataset/build_arc_dataset.py:296-298`).
- The model does contain the reusable pieces: token-ID input, vocabulary embedding, positional encoding (RoPE/learned), an LM head, cross-entropy losses, backward propagation, and model-state checkpointing. The required work is adaptation, not a ground-up rewrite.
- **No pre-existing language-model adaptation was found.** All five model variants are non-causal; there is no tokenizer, no Parquet/text loader, and no LM evaluator. The only `TinyRecursiveReasoningModel_ACTV1` in the repository is the structured-task reference model.

The adaptation is small in line count (a causal flag plus a new data/tokenizer pipeline) but **architecturally decisive**: the resulting causal model is a new research variant, not the reference TRM. It must be separately named and must not be reported as "Samsung TRM trained on TinyStories".

Secondary blockers (resolvable, not the primary verdict):
- The system Python is 3.14.7; Hydra 1.3.x cannot parse arguments on Python 3.14, so the unmodified `pretrain.py` cannot start.
- `adam-atan2==0.0.3` from PyPI installs a pure-Python wrapper without the required `adam_atan2_backend` extension; `import pretrain` fails.
- The ROCm Torch build lacks `gfx1032` (RX 6600M) in its arch list and hard-aborts without `HSA_OVERRIDE_GFX_VERSION=10.3.0`.

---

## B. Dataset verification

### B.1 Location, size, integrity

| Item | Observed |
|---|---|
| Resolved snapshot path | `/home/chandrabhan/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet` |
| Blob path | `/home/chandrabhan/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/blobs/8bacd849e57784e06ebaaab3ef7b01077ca2d9d27ce7d95f2fbe465f1483548b` |
| File size | **672,825,299 bytes** (641.6 MiB) |
| SHA-256 | `8bacd849e57784e06ebaaab3ef7b01077ca2d9d27ce7d95f2fbe465f1483548b` — **matches** the Hub tree's `lfs_sha256` and `lfs_size` |
| Format | Parquet, `created_by parquet-cpp-arrow 21.0.0`, zstd, 2,669 row groups (nominal 1,024 rows/group; final group shorter) |
| Schema | one column: `text: string` |
| Row count | **2,732,634** |
| Nulls | 0 |
| Empty / whitespace-only | 0 |
| Non-ASCII rows | 0 (dataset is pure printable ASCII plus `\n`) |
| Rows with bad ending punctuation | 0 |
| Duplicate 64-bit content hashes | 0 |
| Multi-paragraph rows (contain `\n`) | 2,715,741 |
| Length min / p50 / p95 / max | 115 / 721 / 1,459 / 4,433 chars |
| Mean length | 801.9 chars |
| Total characters | **2,191,281,193** |

The full-file scan was streamed row-group by row-group with PyArrow; it completed in ~81 s without loading all text into memory. The local observations match the dataset card exactly (2,732,634 stories, ~2.19B chars, 115–4,433 chars, no duplicates).

### B.2 Structure and text field

- The only text column is **`text`**.
- Each row is a **complete cleaned story**, not a prompt/completion pair and not a raw document fragment.
- Rows are pre-shuffled. The card's suggested row-index splits are:
  - **test:** rows `0..9,999`
  - **validation:** rows `10,000..19,999`
  - **train:** rows `20,000..2,732,633`
- Preprocessing is required before training: the raw strings are not token IDs, there is no EOS marker, and no fixed-length packing.

### B.3 License and metadata availability

- The local snapshot contains `README.md` (dataset card), `clean.py`, `.gitattributes`, and the Parquet file.
- The card's front matter declares **`license: cdla-sharing-1.0`** locally. The full CDLA-Sharing-1.0 license text is not included as a separate local file; redistribution terms should be confirmed against the canonical license before publishing derived artifacts.
- The repository's MIT `LICENSE` covers the code only, not the dataset.

### B.4 Loading and streaming

- The system Python does **not** have `pyarrow`, `fastparquet`, `polars`, or `duckdb` installed. `pandas 2.3.3` is present but cannot read Parquet without an engine.
- The file was read successfully with an ephemeral `uv run --with pyarrow` environment. Row-group iteration is practical and memory-bounded.
- Loading all 2.19B characters into Python strings/pandas on this machine (14 GiB RAM, ~9.6 GiB already in use) is not advisable. Stream row groups and write tokenized outputs incrementally.

### B.5 Tokenizer and storage estimate

A bounded BPE feasibility check was run **only on the training split** (rows 20,480–43,007, 22,528 stories, 18.09M characters; strictly `>= 20,000`):

- Byte-level BPE, vocab **8,192**, special tokens `<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3`.
- Training time: **0.6 s**; max observed token ID: **8,191** (inside the 8,192 embedding/head dimension); sample round-trip decode was exact.
- Compression: **4.0968 characters/token**.
- Estimated full-dataset token count: **~534.9M tokens** (2,191,281,193 / 4.0968).

Storage implications (full dataset):

| Representation | Size |
|---|---|
| inputs + labels, `uint16` each (4 bytes/token) | **~2.14 GB** |
| inputs + labels, `int32` each (8 bytes/token) | **~4.28 GB** |
| tokenizer JSON | < 1 MB |
| model checkpoint, 9.8–12.8M params, fp32 state_dict | **~39–51 MB** per save |
| optimizer state, if added to checkpoints | roughly 3× parameter storage |

A controlled experiment should use a subset (e.g., 20–100M tokens) rather than the full 534.9M-token corpus. Disk free space observed: **21 GB** on `/home`.

---

## C. Architecture compatibility matrix

Status legend: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `NOT_SUPPORTED`, `NOT_VERIFIED`. The “Necessary change” column lists **recommendations only**; none of those changes were implemented during this audit.

| # | Required capability | Status | Evidence | Necessary change |
|---|---|---|---|---|
| 1 | Token IDs as inputs | **SUPPORTED** | `trm.py:162-164` `_input_embeddings`; `batch["inputs"]` cast to int32; probe accepted `[B,L]` integer batch. | None. |
| 2 | Vocabulary embedding layer | **SUPPORTED** | `trm.py:129` `CastedEmbedding(vocab_size, hidden_size)`. | Set `vocab_size` from metadata. |
| 3 | Positional information | **SUPPORTED** | `trm.py:140-149` RoPE or learned positions; `config/arch/trm.yaml` uses `pos_encodings: rope`. | None; `puzzle_emb_len=0` keeps RoPE length = `seq_len`. |
| 4 | Causal sequence modeling | **NOT_SUPPORTED** | `trm.py:82` hardcodes `Attention(..., causal=False)`; `layers.py:132` passes it to SDPA; `pretrain.py:123` hardcodes `causal=False`; all variants use `causal=False`. Probe: changing the last token changed position-0 logits by **0.68**. | Add a `causal` field to the model config and wire it into `TinyRecursiveReasoningModel_ACTV1Block`; set it true for the LM variant. The existing `Attention` already supports `causal=True`. |
| 5 | LM output head over vocabulary | **SUPPORTED** | `trm.py:130` `lm_head = CastedLinear(hidden_size, vocab_size)`; `trm.py:220` output `[B, seq_len, vocab]`. | None. |
| 6 | Next-token prediction with shifted targets | **PARTIALLY_SUPPORTED** | Model emits per-position logits, but `ACTLossHead` reads `labels` directly (`losses.py:59`); no shift is applied. Existing ARC/Sudoku builders set `labels` to the target grid, not to `inputs[1:]`. | Build `labels[t] = tokens[t+1]` in the new data pipeline. No model change required. |
| 7 | Cross-entropy LM objective | **SUPPORTED** | `losses.py:24-33` `stablemax_cross_entropy`; `losses.py:35-39` `softmax_cross_entropy`; `losses.py:87` `lm_loss` with `IGNORE_LABEL_ID` mask. Probe loss finite (11.53 tiny probe; 38.42 at target scale). | Keep, but remove/zero the auxiliary `q_halt_loss` for LM (`losses.py:88,102`). |
| 8 | Padding and EOS handling | **PARTIALLY_SUPPORTED** | Labels can be masked via `IGNORE_LABEL_ID`; `_collate_batch` pads labels with `-100` and inputs with `pad_id` (`puzzle_dataset.py:147-166`). **No attention padding mask.** Probe: changing pad tokens changed earlier logits by **1.69**. | Prefer fixed-length packing with no padding (recommended). If padding is retained, add a key-padding mask to `Attention`. Add `<eos>` as an ordinary token; no model change needed. |
| 9 | Differentiable recursive latent refinement | **PARTIALLY_SUPPORTED** | `trm.py:208-216`: `H_cycles-1` loops run under `torch.no_grad()`; only the final cycle builds a graph. `trm.py:219` detaches `z_H`/`z_L` in the new carry. | Accept last-cycle-only training, or remove the `no_grad` wrapper for full BPTT (more memory). Document the choice. |
| 10 | Gradient propagation through recursion | **PARTIALLY_SUPPORTED** | Probe: 12/12 trainable tensors had finite, nonzero gradients; an optimizer step changed 12/12 parameters. Gradients only reach the final H cycle by design. | Same decision as #9. |
| 11 | Variable-length sequences or fixed-length blocks | **PARTIALLY_SUPPORTED** | `seq_len` is fixed from metadata; `_collate_batch` pads batches but does not mask attention. Fixed-length blocks work; variable-length sequences would need a mask. | Use fixed-length packed blocks of 256 tokens; drop the final partial block; make eval split sizes multiples of the eval batch size. |
| 12 | Checkpoint saving and resumption | **PARTIALLY_SUPPORTED** | `pretrain.py:235-241` saves only `model.state_dict()`; `pretrain.py:244-262` loads with `map_location="cuda"` and `assign=True`; no optimizer/scheduler/step/carry. `load_checkpoint` unconditionally accesses `model.model.puzzle_emb.weights` (`pretrain.py:252-253`), which fails if puzzle embeddings are disabled. | Guard the puzzle-embedding block; add optimizer/scheduler/step/carry to the checkpoint if resumption is required. A raw `torch.save`/`load_state_dict(strict=True)` round-trip was verified with puzzle embeddings disabled; `pretrain.load_checkpoint` itself was not usable in that configuration. |
| 13 | Puzzle-embedding dependency | **PARTIALLY_SUPPORTED** | `config/arch/trm.yaml:19,25` sets `puzzle_emb_ndim: ${.hidden_size}` and `puzzle_emb_len: 16`. With `puzzle_emb_ndim=0` but `puzzle_emb_len=16`, the model fails: `RuntimeError: The size of tensor a (80) must match the size of tensor b (64)`. | Set **both** `arch.puzzle_emb_ndim=0` and `arch.puzzle_emb_len=0`. |
| 14 | ACT/halting objective suitable for LM | **NOT_SUPPORTED** | `trm.py:221` q-head reads `z_H[:, 0]` (first position); `losses.py:88,102` always adds `0.5 * q_halt_loss`, where the target is "entire sequence exactly correct". For LM this is a near-always-false auxiliary target. | Set `halt_max_steps=1` for LM; add a config switch to weight or disable `q_halt_loss`. Do not reuse ACT as the LM objective. |
| 15 | Distributed training | **NOT_VERIFIED / not required** | `pretrain.py` uses `backend="nccl"` only when `LOCAL_RANK` is set. Single-process runs avoid it. ROCm exposes RCCL under the `nccl` backend name. | Single-GPU run is sufficient for a first experiment; distributed is out of scope. |

---

## D. Training-path assessment

### D.1 The actual execution path

The repository's path is:

```
dataset/build_*.py  →  <split>/dataset.json + <set>__{inputs,labels,
                        puzzle_identifiers,puzzle_indices,group_indices}.npy
      → PuzzleDataset (mmap .npy, sample groups/puzzles, pad batch)
      → train_batch: batch.cuda(); initial_carry under torch.device("cuda")
      → TinyRecursiveReasoningModel_ACTV1: embeddings → recursive cycles
      → lm_head(z_H)[:, puzzle_emb_len:]
      → ACTLossHead: stablemax_cross_entropy(logits, labels) + 0.5*q_halt_loss
      → ((1/global_batch_size)*loss).backward()
      → AdamATan2 (+ sparse puzzle embedding optimizer if enabled)
      → save_train_state: torch.save(model.state_dict())
```

### D.2 Parquet-to-optimizer trace: missing or incompatible components

| Pipeline stage | Status | Evidence / problem |
|---|---|---|
| Parquet record | **READABLE** | Verified by streaming PyArrow; 2,732,634 rows. |
| Text extraction | **MISSING** | No Parquet reader anywhere in the repo. `puzzle_dataset.py:115-144` expects `.npy` + `dataset.json`. Pointing `PuzzleDataset` at the dataset snapshot fails immediately with `FileNotFoundError: .../train/dataset.json`. |
| Tokenizer | **MISSING** | No `tokenizer`, `sentencepiece`, `BPE`, or `tinystories` reference in any `.py`/`.yaml`. The only vocabularies are task-specific: ARC `vocab_size=12` (`build_arc_dataset.py:298`), Sudoku `vocab_size=11` (`build_sudoku_dataset.py:133`). |
| Token IDs | **MISSING** | Must be produced by the new tokenizer/builder. |
| Sequence packing | **MISSING** | No block-packing code. The loader can consume pre-packed fixed-length arrays, but the builder must create them. |
| Model forward | **PARTIAL** | The TRM class accepts `[B,L]` token IDs and returns `[B,L,V]` logits when `puzzle_emb_ndim=0` **and** `puzzle_emb_len=0`. The default `puzzle_emb_len=16` causes a shape error. |
| Next-token targets | **MISSING** | `ACTLossHead` uses `labels` as given (`losses.py:59,87`). No shift. Existing builders set labels to the task answer, not `inputs[1:]`. |
| Loss | **PARTIAL** | Token-level CE works with shifted labels. The loss head always adds the ACT q-halt term (`losses.py:88,102`), which is not an LM objective. |
| Backward | **SUPPORTED** | Probe: loss finite; all 12 parameter gradients finite and nonzero; optimizer step changed 12/12 parameters. |
| Optimizer update | **BLOCKED in current env** | `AdamATan2` import fails because `adam_atan2_backend` is missing from the PyPI package. The repo README recommends a source build with `--no-build-isolation`. With puzzle embeddings disabled, the sparse-embedding optimizer is not used. |
| Checkpoint | **PARTIAL** | Only model state is saved (`pretrain.py:236-241`); `map_location="cuda"` (`:249`); puzzle-embedding access is unconditional (`:252-253`), which breaks resumption when puzzle embeddings are disabled. |
| Evaluation | **MISSING for LM** | Only `evaluators/arc.py` exists; it requires ARC `identifiers.json`/`test_puzzles.json` and grid-specific logic (`evaluators/arc.py:42-62`). If no eval loader exists, `launch` sets `eval_loader=None` (`pretrain.py:570-574`) and `evaluate` later attempts to iterate `None`. Provide a validation split and use `evaluators=[]` for loss metrics, or add an LM evaluator. |

### D.3 Causal-masking verification

This is the critical check requested by the audit. **The current model does not satisfy it.**

Probe setup: real `TinyRecursiveReasoningModel_ACTV1` (hidden 128, 2 layers, `H_cycles=3`, `L_cycles=6`, bf16), batch `[2,64]`, vocab 256, deterministic carry. Change only the final input token, compare logits at position 0:

| Model state | `max abs(logit_pos0(x) - logit_pos0(x'))` | Interpretation |
|---|---|
| As shipped (`causal=False`) | **0.68359375** | Future token leaks into position 0. Not causal. |
| Scratch-only patch: `Attention.causal=True` on both blocks | **0.0** | Causal masking works when wired. Position-last diff remains 0.90, as expected. |

The patch was applied only in a temporary diagnostic script; **no repository file was modified**. It demonstrates that `models/layers.py` already has a correct causal path and that the missing change is configuration/wiring in the TRM block.

Padding probe: replacing the final 8 tokens with a different pad value changed logits at earlier positions by **1.6875**. The model has no padding mask. Packing fixed-length blocks without padding avoids this for training; the final partial eval batch would need care.

### D.4 Recursive gradient behavior

`trm.py:208-216` deliberately runs the first `H_cycles-1` passes under `torch.no_grad()`, then runs the last H cycle with gradients. `trm.py:219` detaches the carry returned to the caller. Therefore:

- Gradients flow only through the final H-cycle refinement, not through the full recursive history.
- This is the reference model's intended memory-saving schedule.
- For a causal LM variant it is still trainable, but it is a design choice that must be documented. Full BPTT would require removing the `no_grad` wrapper and increasing memory.

### D.5 Training-loop and config incompatibilities

- `create_model` passes `causal=False` explicitly (`pretrain.py:123`), but `TinyRecursiveReasoningModel_ACTV1Config` has no `causal` field; Pydantic ignores the extra key. Setting `arch.causal=true` in YAML would have no effect. The block hardcodes `causal=False`.
- `empty_carry` allocates `torch.empty(...)` on the **ambient default device**, not on the batch device (`trm.py:184-188`). `pretrain.py` compensates with `with torch.device("cuda")` (`:130`, `:299`). Any new training loop must replicate this or the carry will be on CPU while the weights are on GPU.
- `create_model` uses `torch.device("cuda")` and `train_batch` uses `.cuda()` (`pretrain.py:130,295`). On ROCm these are the correct API names; no string change is required.
- `torch.compile` is enabled unless `DISABLE_COMPILE` is set (`pretrain.py:134-135`). On this ROCm consumer GPU, set `DISABLE_COMPILE=1` for the first experiment.
- `wandb.init` is unconditional on rank 0 (`pretrain.py:590`). Use `WANDB_MODE=offline` or a stub for a local smoke test.
- `evaluate` assumes `eval_loader` is iterable. If no test split exists, the code sets it to `None` and then crashes at the first evaluation (default `min_eval_interval=0`). A validation/test split is required, or the guard must be changed.
- `total_steps` is computed as `epochs * total_groups * mean_puzzle_examples / global_batch_size` (`pretrain.py:217-220`). With one block per group and `mean_puzzle_examples=1`, this matches block-count training.

---

## E. Hardware feasibility

### E.1 Host and software versions (observed)

| Item | Observed |
|---|---|
| OS / kernel | CachyOS Linux, `7.2.8-1-cachyos`, x86_64 |
| CPU | AMD Ryzen 7 5800H, 16 threads |
| System RAM | 14 GiB total, ~9.6 GiB used, ~5.4 GiB available, 12 GiB swap |
| GPU 0 | AMD Radeon RX 6600M, `gfx1032`, 8.57 GB VRAM |
| GPU 1 | AMD Radeon Graphics (iGPU), `gfx90c`, 8.05 GB reported |
| Python | 3.14.7 (`/usr/bin/python3`); **no `pip` module** |
| PyTorch | **2.14.0** (ROCm build), `torch.version.hip = 7.2.53211` |
| Triton | 3.5.1 |
| NumPy / pandas / pydantic | 2.5.3 / 2.3.3 / 2.13.5 |
| Missing from system Python at audit start | `pyarrow`, `tokenizers`, `hydra-core`, `omegaconf`, `einops`, `wandb`, `argdantic`, `adam_atan2`, `coolname`, `numba` |

`torch.cuda.is_available()` is **True** (ROCm exposes the CUDA API namespace). `torch.cuda.get_arch_list()` contains `gfx1030`, `gfx1031`, `gfx1035`, `gfx1036`, but **not `gfx1032`** (the RX 6600M) and not `gfx90c`.

### E.2 GPU compatibility test

**Without an override**, even `torch.zeros(8, device="cuda:0")` aborts the process:

```
python3: /usr/src/debug/hip-runtime/rocm-systems-rocm-7.2.4/projects/clr/hipamd/src/hip_code_object.cpp:400:
hipError_t hip::StatCO::getStatFunc(ihipModuleSymbol_t**, const void*, int):
Assertion `err == hipSuccess' failed.
```

Exit code 134. This is a hard abort, not a Python exception.

**With the runtime override** `HSA_OVERRIDE_GFX_VERSION=10.3.0` (no driver or GPU-stack changes):

- tensor creation: OK
- bf16 512×512 matmul + backward: OK, gradients finite
- causal `scaled_dot_product_attention` + backward: OK, gradients finite
- real TRM forward/backward: OK (see E.4)

This is an **unofficial gfx1032→gfx1030 ISA mapping**. It works on this machine, but it is not an officially supported ROCm configuration; numerical or ISA edge cases are possible. It must be documented in any experiment log.

**CPU fallback:** a tiny TRM forward/backward on CPU completed successfully (0.54 s for hidden 128, `[2,64]`, 2 layers). CPU execution is technically possible but not practical for a 500M-token corpus.

### E.3 Memory and throughput at the proposed scale

Measured with the real TRM class, `seq_len=256`, `L_layers=2`, `H_cycles=3`, `L_cycles=6`, bf16, microbatch 4, on `cuda:0` with the override:

| `hidden_size` | `vocab_size` | Params | Peak allocated | Step time |
|---|---|---|---|---|
| 512 | 12,000 | **19,104,770** | **1.616 GB** | ~0.91 s/step (after warmup) |
| 384 | 12,000 | **12,755,714** | not separately measured | — |
| 384 | 8,192 | **9,831,170** | not separately measured | — |
| 320 | 8,192 | **8,028,802** | not separately measured | — |

The 8–15M parameter target is therefore reachable at `hidden_size=384` with an 8,192 or 12,000-token vocabulary. At 9.8–12.8M parameters, the embedding and LM head dominate because they are untied; this is expected for a from-scratch LM. The 7M ARC figure in the paper is not directly comparable (ARC vocab is 12 and puzzle embeddings are used).

The measured 1.6 GB peak at microbatch 4 leaves substantial headroom in 8 GB VRAM. This audit did not test microbatch 8+ or longer sequences; do not assume those fit without measurement.

### E.4 Bounded smoke test (component-level, not the full pipeline)

A bounded test was run against the **actual** `TinyRecursiveReasoningModel_ACTV1` and `ACTLossHead` classes, with synthetic token IDs (the full Parquet→tokenizer path does not exist):

- batch `[2,64]`, vocab 256, hidden 128, 2 layers, bf16, shifted labels (`labels[:, :-1] = inputs[:, 1:]`, last label `-100`)
- forward loss: **11.5319**, finite
- logits shape: **`(2, 64, 256)`**; q logits shape `(2,)`
- 12 trainable gradient tensors: **12 finite, 12 nonzero**
- gradient norms: embedding 0.4875, LM head 1.3607, first attention QKV 1.3848
- one `Adam` step changed **12/12** parameters
- checkpoint: `torch.save(model.state_dict())` → `load_state_dict(..., strict=True)` succeeded (2.37 MB for this tiny config)
- peak GPU memory: 0.115 GB
- future-token leakage (as shipped): 0.6836; padding leakage: 1.6875
- scratch-only `causal=True` patch: position-0 leakage **0.0**

At the target scale (`hidden 512`, `vocab 12,000`, `[4,256]`, bf16), four optimizer steps completed; the step-0 loss was finite (38.42), peak memory was 1.616 GB, and the last three steps averaged ~0.91 s/step.

**What this proves:** the existing model/loss/optimizer/checkpoint mechanics are technically trainable on this GPU with the override.
**What this does not prove:** that the Samsung TRM training path works end-to-end, that the recursive refinement is useful for language, or that TinyStories training would produce a competent model. The full path is missing the tokenizer/data pipeline and is blocked by the environment issues in E.5.

### E.5 Software blockers

1. **Hydra vs Python 3.14.** `python pretrain.py --cfg job` fails before any training:
   ```
   TypeError: argument of type 'LazyCompletionHelp' is not a container or iterable
   ValueError: badly formed help string
   ```
   This occurs with both `hydra-core 1.3.7` and the repo-pinned `1.3.2`. Calling `hydra._internal.utils.get_args_parser()` directly succeeds on Python 3.12.8, confirming this is a Python-3.14/argparse incompatibility, not a project bug. A Python 3.10–3.12 environment is required.
2. **`adam-atan2`.** `uv pip install adam-atan2==0.0.3` installs a `py3-none-any` pure-Python wheel whose `adam_atan2.py` begins with `import adam_atan2_backend`; the backend module is not in the wheel and `adam_atan2-backend` is not a resolvable package. `import pretrain` therefore fails. The repo README explicitly recommends `pip install --no-cache-dir --no-build-isolation adam-atan2` (source build). This was not attempted during the audit.
3. **`argdantic` at import time.** `puzzle_dataset.py:13` imports `argdantic` even though it is only needed by the dataset builder CLIs. `pretrain.py` cannot import without it. It is installable on Python 3.14.
4. **Tokenizer library.** `tokenizers` is not installed; the BPE feasibility check used an ephemeral `uv` environment.
5. **No pip in system Python.** `python3 -m pip` is unavailable. `uv` is available at `/home/chandrabhan/.local/bin/uv`; dependencies can be installed into a venv with `uv pip`.
6. **ROCm wheel for Python 3.12.** A dry-run resolved `torch-2.9.1+rocm6.3-cp312-cp312-manylinux_2_28_x86_64.whl` from the official PyTorch ROCm index, but the download could not be completed during the audit (transport timeout). A Python 3.12 venv with a ROCm wheel is the clean path, but **gfx1032 support in that wheel is unverified**; expect to need the same `HSA_OVERRIDE_GFX_VERSION` workaround.

---

## F. Minimum-change plan (recommendations; not implemented)

These changes were **not** made. They are ordered to preserve the distinction between the reference implementation and a new research variant.

### 1. Data-pipeline changes

- Add a new builder (e.g., `dataset/build_tinystories_dataset.py`); do not modify the ARC/Sudoku builders.
- Stream the Parquet file row-group by row-group; enforce the card's splits by row index (`test 0..9,999`, `val 10,000..19,999`, `train 20,000..end`).
- Train the BPE tokenizer **only** on the train split; save `tokenizer.json`; use it unchanged for val/test.
- Append `<eos>` between stories, concatenate the split's token stream, and pack into blocks of `seq_len=256`. Use `inputs = stream[i*L:(i+1)*L]` and `labels = stream[i*L+1:(i+1)*L+1]`; drop the final incomplete block. This yields a valid next-token target at every position and requires no padding mask.
- Write the five required `.npy` files per split (`all__inputs.npy`, `all__labels.npy`, `all__puzzle_identifiers.npy`, `all__puzzle_indices.npy`, `all__group_indices.npy`) and `dataset.json`, with one block per group/puzzle:
  - `seq_len=256`, `vocab_size=8192`
  - `pad_id=0`, `ignore_label_id=None` (or a dedicated pad ID if padding is used)
  - `blank_identifier_id=0`, `num_puzzle_identifiers=1`
  - `total_groups=N`, `mean_puzzle_examples=1.0`, `total_puzzles=N`, `sets=["all"]`
- Truncate val/test to multiples of the eval batch size, or add a drop-last path, to avoid the unpadded final batch.
- Start with a bounded token subset (e.g., 20–100M tokens) and a fixed seed. Do not scan the full corpus for a first controlled experiment.

### 2. Tokenizer and embedding changes

- No model code change: `vocab_size` is already taken from `PuzzleDatasetMetadata` (`pretrain.py:120`).
- Use an 8,192-token byte-level BPE (or a 12k/16k variant if desired). The bounded check produced max ID 8,191, so IDs stay within the embedding/head dimensions.
- Do not introduce a pretrained language model or pretrained tokenizer weights; the research direction is from-scratch training.
- Keep `<pad>`, `<eos>`, `<bos>`, `<unk>` as ordinary vocabulary entries. The model has no special EOS logic, but it can learn the token like any other.

### 3. Language-model head and causal objective

- Add `causal: bool = False` to `TinyRecursiveReasoningModel_ACTV1Config` (`trm.py:32-63`).
- Change `TinyRecursiveReasoningModel_ACTV1Block.__init__` to pass `causal=config.causal` to `Attention` (`trm.py:82`).
- Change `pretrain.py:123` so the LM variant passes `causal=True` (or remove the hardcoded value and read it from `arch`).
- Prefer creating a separately named `models/recursive_reasoning/trm_lm.py` (copy with the causal config) and `config/arch/trm_lm.yaml`, leaving `trm.py` untouched as the reference. The existing `Attention` kernel needs no change.
- Re-run the leakage probe with `causal=True` and require position-0 difference `== 0.0` before training.
- Set `arch.puzzle_emb_ndim=0` **and** `arch.puzzle_emb_len=0`; otherwise the model crashes with the 80-vs-64 shape error.
- Set `halt_max_steps=1` so ACT does not run; make the q-halt auxiliary loss weight configurable and set it to 0 for LM (`losses.py:88,102`).

### 4. Recursive-state and gradient-flow changes

- Decide explicitly between:
  - **Reference schedule (recommended for the first experiment):** keep the `H_cycles-1` no-grad passes; only the final H cycle trains. Lower memory, faithful to the reference design, but the recursion is not fully backpropagated.
  - **Full BPTT:** remove the `torch.no_grad()` wrapper at `trm.py:208`; this increases activation memory with `H_cycles` and should be measured before use.
- Keep `z_H`/`z_L` detached across calls (`trm.py:219`) unless truncated BPTT across batches is an explicit research question.
- Document that the causal variant's recursive refinement is a new hypothesis, not a validated language-modeling mechanism.

### 5. Training-loop and checkpoint changes

- Fix `load_checkpoint` to guard the puzzle-embedding resize block when `puzzle_emb_ndim=0` (`pretrain.py:252-253`); otherwise resumption raises `AttributeError`.
- Save optimizer state, scheduler state, global step, and carry if exact resumption is required. The current `save_train_state` saves only the model (`pretrain.py:236-241`).
- Guard `evaluate` against `eval_loader=None`, or guarantee a validation split exists (`pretrain.py:570-574`).
- Replace the `AdamATan2` dependency with a working build, or use `torch.optim.AdamW` in the separately named LM variant. Do not silently change the reference optimizer.
- Use `DISABLE_COMPILE=1` on this ROCm setup.
- Run `wandb` in offline mode or disable it for the first experiment (`pretrain.py:590`).
- Use a Python 3.10–3.12 environment with a ROCm PyTorch wheel; the current Python 3.14 environment cannot start Hydra. Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for the current system Torch build, and re-verify the override on any replacement wheel.
- Keep `torch.device("cuda")` / `.cuda()`; they are correct under ROCm. Preserve the ambient-device context around `initial_carry`.

### 6. Evaluation and baseline requirements

- Add a validation-language-model metric: token-level cross-entropy / perplexity on the held-out split, plus token accuracy. The existing ARC evaluator is not applicable (`evaluators/arc.py`).
- Report loss on the **val** split during training and reserve the **test** split for a final, preregistered evaluation.
- Define a baseline before training: at minimum, a simple n-gram/unigram baseline on the same tokenizer and splits; optionally a small causal transformer ablation. Do not replace TRM with the baseline and call it a TRM result.
- Preregister the comparison protocol, seeds, token budget, and evaluation prompts/slices. Do not alter frozen benchmarks or reference results.
- Do not claim that TinyStories performance demonstrates general-purpose intelligence or validates recursive reasoning.

---

## G. Evidence and reproducibility

### G.1 Verified facts

- Repository state: `main`, clean, `c01103738605ba39d1430519b1ee0c62f4c707f8`, 0 ahead/0 behind `origin/main`; upstream `SamsungSAILMontreal/TinyRecursiveModels`.
- Before this report, no `docs/`, `AGENTS.md`, research charter, preregistration, or governance record existed in the repository or the parent working directory. The only project documentation is `README.md` and the MIT `LICENSE`.
- Dataset file: 672,825,299 bytes; SHA-256 matches the Hub `lfs_sha256` exactly; 2,732,634 rows; one `text: string` column; 2,669 row groups; zstd.
- Full scan: 0 nulls, 0 empty/whitespace, 0 non-ASCII, 0 bad endings, 0 duplicate 64-bit hashes; lengths 115–4,433; total 2,191,281,193 chars.
- Strict-train BPE check: 22,528 stories, 18.09M chars, vocab 8,192, max ID 8,191, 4.0968 chars/token, ~534.9M full-dataset tokens.
- GPU: `torch 2.14.0`, HIP 7.2.53211, `torch.cuda.is_available()=True`, 2 devices, bf16 supported. Arch list lacks gfx1032. Without override, a trivial GPU allocation aborts with the HIP assertion shown in E.2. With `HSA_OVERRIDE_GFX_VERSION=10.3.0`, matmul/SDPA/TRM forward/backward all succeed.
- Real TRM probe: finite loss, 12/12 finite nonzero gradients, 12/12 parameters changed by one optimizer step, strict state_dict reload OK, peak 0.115 GB (small config).
- Target-scale probe: 19.1M params, 1.616 GB peak, ~0.91 s/step after warmup at hidden 512/vocab 12k/batch 4/seq 256/bf16.
- `PuzzleDataset` on the Parquet snapshot fails with `FileNotFoundError: .../train/dataset.json`.
- Default `puzzle_emb_ndim=0` + default `puzzle_emb_len=16` fails with `RuntimeError: The size of tensor a (80) must match the size of tensor b (64)`.
- Python 3.14 + Hydra 1.3.2/1.3.7 fails with `ValueError: badly formed help string`; Python 3.12.8 Hydra argparse succeeds.
- `adam-atan2==0.0.3` installed from PyPI lacks `adam_atan2_backend`; `import pretrain` fails. A diagnostic stub was used only to test the rest of the import path; it is not a repository change or a proposed fix.

### G.2 Assumptions / not verified

- The RX 6600M gfx1032→gfx1030 override is unofficial; long-run numerical stability was not tested.
- The Python 3.12 ROCm wheel (`torch 2.9.1+rocm6.3`) was resolved but not installed; its gfx1032 behavior is unverified.
- No end-to-end Parquet→tokenizer→`pretrain.py` run was performed, because the tokenizer/data pipeline does not exist and the entrypoint is blocked.
- No language competence, convergence, or recursive-reasoning benefit was tested.
- The exact memory at microbatch 8+ and longer sequences was not measured.
- The full CDLA-Sharing-1.0 license text is not in the local snapshot; only the card's `license:` declaration is local.

### G.3 Commands used (representative)

```bash
# Repository state
git -C TinyRecursiveModels status -sb
git -C TinyRecursiveModels log -1 --format='%H%n%ad%n%s'

# Dataset integrity
sha256sum .../blobs/8bacd849...   # matched Hub lfs_sha256
uv run --with pyarrow --with numpy python - <<'PY'
# streamed row groups; schema/row count/nulls/duplicates/statistics
PY

# Tokenizer feasibility (strict train split, rows >= 20,000)
uv run --with pyarrow --with tokenizers python - <<'PY'
# ByteLevel BPE, vocab 8192; 4.0968 chars/token
PY

# GPU compatibility
python3 -c "import torch; torch.zeros(8, device='cuda:0')"          # aborts
HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 -c "..."                    # succeeds

# Real TRM component probe
HSA_OVERRIDE_GFX_VERSION=10.3.0 <scratch-venv>/bin/python trm_probe.py
```

### G.4 Repository impact

No tracked file was modified during the audit. No commit was created during the audit; the later adaptation/pilot changes are committed separately. Python imports created `__pycache__/` directories, which were removed; `git status` was clean apart from the new audit report. The scratch venv and scripts were under the session scratchpad, not in the repository.

---

## H. Final decision

**Do not begin a bounded training experiment with the existing implementation.**

The reference Samsung TRM is not a causal language model, and its data path cannot read TinyStories. The correct next step is:

**Author a separately named recursive language-model adaptation** (e.g., `trm_lm`) that:
1. wires `causal=True` through the TRM attention block while leaving `trm.py` as the reference,
2. uses an LM-only objective (no ACT q-halt term),
3. consumes fixed-length packed blocks from a new TinyStories builder with an 8k BPE tokenizer, and
4. is trained from scratch, with a preregistered validation/test protocol.

Before any implementation or training, resolve the environment prerequisites: a Python 3.10–3.12 venv with a working ROCm PyTorch build, a functioning `adam-atan2` (or an approved AdamW substitution in the new variant), and the `HSA_OVERRIDE_GFX_VERSION=10.3.0` runtime workaround (or a wheel that natively supports gfx1032).

**Single recommended next action:** seek the separate research-governance authorization required by this audit to author the `trm_lm` variant and its TinyStories data pipeline. Do not train the unmodified Samsung TRM and do not report a conventional transformer as a TRM result.
