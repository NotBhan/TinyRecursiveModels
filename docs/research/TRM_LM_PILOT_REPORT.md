# TRM-LM 5M-Token Pilot Report

**Date:** 2026-10-21
**Authorization:** 5,000,000 eligible training target tokens per arm, for B1, R1, and R4.
**Status:** Pilot complete. No further training, seeds, or budget extension was started.
**Repository:** `/home/chandrabhan/Documents/Chandrabhan/TinyRecursiveModels`
**Git:** `main` @ `c01103738605ba39d1430519b1ee0c62f4c707f8` (base commit); the adaptation/pilot changes are in the follow-up repository commit.
**Remote:** `origin` → `https://github.com/NotBhan/TinyRecursiveModels.git`; `upstream` → the original Samsung repository.
**Experiment directory:** `experiments/trm_lm_pilot_5m/` (local; excluded from the commit by `.gitignore`).

---

## 1. Frozen configuration and dataset provenance

### 1.1 Dataset

| Item | Observed |
|---|---|
| Hugging Face repo | `karpathy/tinystories-gpt4-clean` |
| Local Parquet | `/home/chandrabhan/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/snapshots/0397e27157956705a0260709da3095bb9c43d6a7/tinystories_gpt4_clean.parquet` |
| Size | **672,825,299 bytes** (matches audit) |
| SHA-256 | `8bacd849e57784e06ebaaab3ef7b01077ca2d9d27ce7d95f2fbe465f1483548b` |
| Rows | **2,732,634** |
| Columns | `text` |
| Train subset | rows `20,000..59,999` (**40,000 documents**) |
| Validation subset | rows `10,000..19,999`; first **512 documents** used |
| Test subset | not used |
| Validation target tokens | **102,905** (same for all arms) |
| Rejected / duplicate rows in manifest | **0 / 0** over 60,000 scanned rows |
| Cross-document attention | **No** |
| Cross-chunk attention | **No**; each chunk is a separate sample |
| Final incomplete block | padded and masked, labels `-100` |

The dataset was verified at the start of every run with `verify_parquet()`: size and row count matched the audit constants. No download was attempted.

**Data-subset revision.** The first B1 attempt used a 20,000-document train stream and exhausted it at **3,941,142 eligible tokens**, short of the 5M budget. The train subset was expanded by appending rows `40,000..59,999` (40,000 documents total). The first 20,000 train documents and their data order are unchanged. The tokenizer and validation set are unchanged. The original frozen manifest is preserved as `FROZEN_CONFIG_v1_20k.json`; the expanded revision is `FROZEN_CONFIG.json` (revision 2). See Section 9.

### 1.2 Tokenizer

| Item | Value |
|---|---|
| Type | Byte-level BPE, trained from scratch |
| Vocabulary | **8,000** (authorized size) |
| Special IDs | `<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3` |
| Training split | train only |
| Training documents | first **20,000** train documents |
| Tokenizer SHA-256 | `7497806734d3e5849028d5184e29c2d7ba18202e097718518e751bede5c2f682` |
| Split manifest SHA-256 | `69f8c24d0e9829f684d040c93ba7578f858bc6f70150c8bac90eca45de03de6a` |

All three arms produced the **same** tokenizer and split hashes (verified from the one-step summaries). The tokenizer was never fitted on validation documents.

### 1.3 Model and training configuration

| Field | Value |
|---|---|
| Context length | 256 tokens |
| Hidden size | 512 |
| Layers | 2 |
| Heads | 8 |
| Expansion | 4.0 |
| Embeddings / LM head | tied |
| `n_refine` | B1: 1 (single pass), R1: 1, R4: 4 |
| Parameters | **10,911,744** for every arm |
| Optimizer | AdamW, lr `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1` |
| Gradient clipping | 1.0 |
| Microbatch | **4 sequences** |
| Precision | bfloat16 |
| Seed | **1337** |
| Training budget | **5,000,000 eligible target tokens per arm** |
| Checkpoint / validation interval | every **1,000,000 tokens**, plus completion |
| Device | AMD Radeon RX 6600M, `HSA_OVERRIDE_GFX_VERSION=10.3.0` (opt-in, unofficial gfx1032→gfx1030 mapping) |
| PyTorch / HIP | 2.14.0 ROCm / 7.2.53211 |

**Architecture note:** `trm_lm` is a single-latent adaptation, not the original
TRM’s two-state `z_H`/`z_L` + ACT recursion. R1 is a one-pass control; only R4
exercises repeated latent refinement. The original `trm.py` reference remains
untouched and was not used in the pilot. See
`TRM_LM_IMPLEMENTATION_REPORT.md` for the distinction.

The implementation report’s 11,010,048 parameter figure was for an 8,192-token vocabulary. The authorized 8,000-token tokenizer gives **10,911,744** parameters, exactly **98,304 fewer** (192 × 512). All three arms are exactly parameter-matched.

### 1.4 Learning-rate sanity check

A separate 20-step check at `lr=3e-4` used a 500-document sample (not part of the 5M budgets). All losses and gradients were finite and decreasing:

| Arm | Init val | Val at ~10,270 tokens | Final val at ~13,673 tokens |
|---|---:|---:|---:|
| B1 | 9.252786 | 6.349943 | 5.883913 |
| R1 | 9.250416 | 6.349744 | 5.883257 |
| R4 | 9.075796 | 6.624058 | 6.251244 |

The candidate `3e-4` was stable; no adjustment was made. Checkpoint reload in the sanity runs was exact.

### 1.5 One-step verification

A 500-token one-step check was run for each arm on the expanded stream:

| Arm | Tokens | Steps | Params | Checkpoint max abs diff | Tokenizer/split hashes |
|---|---:|---:|---:|---:|---|
| B1 | 500 | 1 | 10,911,744 | 0.0 | identical |
| R1 | 500 | 1 | 10,911,744 | 0.0 | identical |
| R4 | 500 | 1 | 10,911,744 | 0.0 | identical |

An earlier 1,000-token one-step check correctly failed with `Training stopped at 716 eligible tokens, expected exactly 1000`: one batch contained only 716 valid targets, and `max_steps=1` could not consume 1,000. The failure log is preserved at `logs/smoke_b1.log`. The corrected 500-token checks verify exact truncation.

---

## 2. Token-counting convention and verified budgets

The training budget is counted in **eligible next-token targets**: positions where `labels != -100`. For each batch, the count is `(labels != -100).sum()`. The final batch is truncated in row-major order to the remaining budget, and the same truncated labels are used for the loss and the count. Resume uses `skip_tokens` with the same row-major convention.

| Arm | Tokens processed | Optimizer steps | Mean targets/step | Effective batch |
|---|---:|---:|---:|---:|
| B1 | **5,000,000** | 7,170 | 697.35 | 4 sequences |
| R1 | **5,000,000** | 7,170 | 697.35 | 4 sequences |
| R4 | **5,000,000** | 7,170 | 697.35 | 4 sequences |

All three arms consumed exactly the authorized budget. The post-run guard `tokens_processed == max_train_tokens` passed for every arm.

---

## 3. Actual parameter counts

All arms: **10,911,744 trainable parameters**, 9 parameter tensors with tied embedding counted once.

| Component | Parameters |
|---|---:|
| Token embedding (tied LM head) | 4,096,000 |
| Two blocks: QKV 786,432 + O 262,144 + gate_up 1,572,864 + down 786,432 each | 6,815,744 |
| **Total** | **10,911,744** |

The checkpoint `state_dict` stores the tied embedding under both `embed_tokens.weight` and `lm_head.weight`; summing raw state-dict tensors gives 15,007,744, which double-counts the tied 4,096,000. The model’s `num_parameters` (10,911,744) is the correct count.

---

## 4. Training and validation losses

Validation was performed in `eval()` mode on the same 512 held-out documents (102,905 target tokens) for all arms; it did not update parameters or optimizer state.

| Tokens | B1 val CE | R1 val CE | R4 val CE |
|---:|---:|---:|---:|
| 0 | 9.255755 | 9.253309 | 9.078772 |
| 1,000,494 | 3.135253 | 3.135093 | 3.284817 |
| 2,000,494 | 2.917292 | 2.915295 | 2.963753 |
| 3,000,067 | 2.809023 | 2.805085 | 2.812072 |
| 4,000,060 | 2.727012 | 2.727647 | 2.722723 |
| **5,000,000** | **2.675637** | **2.674267** | **2.650498** |

Final perplexities (`exp(CE)`):

| Arm | Final CE | Final PPL |
|---|---:|---:|
| B1 | 2.675637 | **14.5216** |
| R1 | 2.674267 | **14.5017** |
| R4 | 2.650498 | **14.1611** |

Last single-batch training losses were 2.750863 (B1), 2.750862 (R1), and 2.737627 (R4); single-batch training loss is noisy and is not the comparison metric. All validation curves are still descending at 5M tokens.

The B1 curve combines the original 0→3M run and the resumed 3M→5M run. The resumed run’s initial validation at 3,000,067 tokens was **2.809023**, exactly matching the original run’s 3M validation loss, confirming correct checkpoint/data alignment.

---

## 5. Throughput, elapsed time, memory, and compute

### 5.1 Training time and throughput

| Arm | Successful training time | Tokens/s | Notes |
|---|---:|---:|---|
| B1 | **978.608 s** | **5,109.3** | 605.03 s for 0→3M + 373.58 s for 3M→5M |
| R1 | **1,017.708 s** | **4,913.0** | clean 0→5M run |
| R4 | **3,732.786 s** | **1,339.5** | clean 0→5M run; four refinement passes |

B1’s aborted 20k-document attempt added **192.537 s** of discarded compute (941,075 tokens), so B1’s total compute time including the aborted segment was **1,171.145 s**. The final B1 trajectory and reported metrics are the 5M-token successful trajectory; the aborted segment is not counted as an additional arm or budget extension.

**Compute ratios:**

- R4 / R1 training time: **3.668×**
- R4 / R1 forward latency: **4.269×**
- R4 / R1 generation throughput: **0.349×** (R4 is ~2.87× slower)
- R1 / B1 training time: **1.040×** (R1 is 4.0% slower)
- R1 / B1 forward latency: **1.145×**

### 5.2 Memory

| Arm | Peak training allocated | Peak training reserved | Peak generation allocated | Max RSS |
|---|---:|---:|---:|---:|
| B1 | 0.351 GB | 0.411 GB | 0.297 GB | 1,828.4 MB |
| R1 | 0.327 GB | 0.382 GB | 0.297 GB | 1,828.4 MB |
| R4 | 0.545 GB | 0.602 GB | 0.302 GB | 1,827.5 MB |

All values are far below the 8.57 GB VRAM and 14 GiB system RAM. No memory-pressure event occurred.

### 5.3 Inference

| Arm | Forward latency (batch 4, seq 256) | Generation throughput (5 prompts × 64 greedy tokens) |
|---|---:|---:|
| B1 | 14.824 ms | 257.173 tok/s |
| R1 | 16.978 ms | 258.413 tok/s |
| R4 | 72.479 ms | 90.077 tok/s |

A separate filtered 5-prompt evaluation measured 183.6 / 281.6 / 92.0 tok/s for B1 / R1 / R4; this shows run-to-run variance on the consumer GPU. The table above uses the end-of-run summary measurements. All generation used greedy argmax, `max_new_tokens=64`, `eos_token_id=1`, seed 1337.

### 5.4 Compute proxy

- Param-token product (all arms): `10,911,744 × 5,000,000 = 54,558,720,000,000` (`5.455872e13`).
- Refined pass product (R4): `2.1823488e14`; B1/R1: `5.455872e13`.
- Rough 6× training-FLOPs proxy: B1/R1 `3.2735232e14`; R4 `1.30940928e15`.

These are **proxies**, not measured hardware counters. R4 is token-matched but **not compute-matched**; the measured wall-clock ratios are the primary compute evidence.

---

## 6. Generation samples and limitations

Prompts were drawn from the fixed validation split and filtered to five distinct, non-degenerate texts. The full outputs are in `generation_distinct_filtered.json`; the earlier raw set (which included a degenerate `"!"` prompt) is preserved as `generation_distinct_all.json`.

Prompts:

1. `Once upon a time, there was`
2. `One day, a cat named Max`
3. `Once there was a very competitive little`
4. `Once there was a kid who was`
5. `Once upon a time, a little`

Representative outputs (truncated for the report; full text in the JSON):

| Prompt | B1 | R1 | R4 |
|---|---|---|---|
| 1 | “Once upon a time, there was a little boy named Tim. Tim had a toy car…” | “Once upon a time, there was a little boy named Tim. Tim had a toy car…” | “Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved very much. He would play with his toy car…” |
| 2 | “One day, a cat named Max went for a walk. He saw a big tree with a long stick…” | “One day, a cat named Max went for a walk. He saw a big tree with a long stick…” | “One day, a cat named Max went for a walk. He saw a big tree with a tree…” |
| 5 | “Once upon a time, a little boy named Tim went to the park with his mom…” | “Once upon a time, a little girl named Lily went to the park. She saw a big tree…” | “Once upon a time, a little girl named Lily went to the park with her mom…” |

Observations:

- All outputs are valid decoded text; no control characters, malformed tokens, or empty outputs.
- All generations reached the 64-token limit without emitting EOS; they are truncated by the decoding budget, not completed stories.
- Repetition is common in all arms (e.g., repeated “big tree”, “It is a tree” in R4 samples). B1 and R1 are similar; R4 is sometimes more descriptive, sometimes more repetitive.
- No meaningful qualitative winner is established by five greedy samples.
- The earlier raw prompt set included a punctuation-only prompt `"!"`; B1/R1 emitted almost nothing while R4 emitted a long repetitive continuation. That prompt is preserved in `generation_distinct_all.json` but excluded from the filtered comparison.
- These samples are not evidence of general language competence or intelligence.

---

## 7. Checkpoint reload results

| Arm | Final checkpoint | Reload step | Reload tokens | Max abs logit diff |
|---|---|---:|---:|---:|
| B1 | `experiments/trm_lm_pilot_5m/b1_resumed/checkpoints/tokens_5000000.pt` | 7,170 | 5,000,000 | **0.0** |
| R1 | `experiments/trm_lm_pilot_5m/r1/checkpoints/tokens_5000000.pt` | 7,170 | 5,000,000 | **0.0** |
| R4 | `experiments/trm_lm_pilot_5m/r4/checkpoints/tokens_5000000.pt` | 7,170 | 5,000,000 | **0.0** |

Checkpoints contain model, optimizer, scheduler state, step, token count, model/train/tokenizer configs, split manifest, RNG states, and device. The B1 resume from the 3M checkpoint produced an initial validation loss of 2.809023, identical to the original 3M validation loss, confirming data/model alignment. RNG states are saved; they are not restored on resume, but the pilot has no dropout or data shuffling, so the trajectory is deterministic.

---

## 8. Failures, deviations, and restarts

1. **B1 data exhaustion (v1).** The first B1 run used 20,000 train documents and stopped at **3,941,142 tokens** with the exact-budget guard:
   `Training stopped at 3941142 eligible tokens, expected exactly 5000000.`
   This was a data-subset limitation, not a numerical or GPU failure. The train stream was expanded to 40,000 documents; B1 was resumed from its last regular checkpoint at **3,000,067 tokens** (step 4,301) and completed to 5,000,000 tokens. The aborted 3M→3.94M segment cost 192.537 s and 941,075 tokens of discarded compute. The v1 log, metrics, and checkpoints are preserved under `b1/`.
2. **1,000-token one-step check.** The check failed by design because one batch had only 716 eligible targets and `max_steps=1` could not reach 1,000. The log is preserved; the corrected 500-token checks passed exactly.
3. **Degenerate raw generation prompt.** The first distinct-prompt set included a punctuation-only prompt. It is preserved in `generation_distinct_all.json`; the filtered set is used for the reported comparison.
4. **Vocabulary/parameter deviation.** The pilot uses the authorized 8,000-token tokenizer, not the implementation default 8,192. The resulting parameter count is 10,911,744 rather than 11,010,048 (0.9% lower). All arms are exactly matched.
5. **Single seed / no test split.** The pilot uses one seed (1337) and the validation split only. The test split was not used. No error bars or significance tests are available.

No non-finite loss, non-finite gradient, checkpoint failure, or GPU crash occurred in the three final arms. Across all 22,902 logged events (including the aborted B1 segment), there were **0 non-finite train losses, 0 non-finite grad norms, and 0 non-finite validation losses**.

---

## 9. Comparative interpretation

### 9.1 Token-matched comparison

- **B1 vs R1:** final CE 2.675637 vs 2.674267. R1 is better by **0.001370 CE (0.051% relative)**. The curves are essentially tied at every checkpoint. This is not a meaningful difference at one seed.
- **R4 vs B1/R1:** final CE 2.650498. R4 is better than R1 by **0.023769 CE (0.889% relative)** and better than B1 by **0.025139 CE (0.940% relative)**. R4 is worse at 1M and 2M tokens, narrows the gap by 3M (still slightly behind), and is slightly ahead at 4M and 5M. All curves are still descending at 5M.

### 9.2 Compute-aware comparison

- B1 and R1 have nearly identical training cost (R1 +4.0%) and forward cost (R1 +14.5%); their quality is indistinguishable. The input-injection/shared-stack difference at one pass does not produce a measurable benefit.
- R4 achieves a small final CE improvement at **3.668× the training time**, **4.269× the forward latency**, and **0.349× the generation throughput** of R1. The measured quality gain is small relative to this cost. **R4 does not demonstrate a compute advantage.**
- R4 is token-matched but not compute-matched. A fair compute-matched comparison would need a B1 variant with approximately 4× the training/forward compute (e.g., deeper/wider baseline or more tokens), which this pilot did not run.
- The R4 advantage may reflect the extra effective depth or a genuine refinement effect; this pilot cannot separate those explanations from compute alone, and a single seed cannot establish robustness.

### 9.3 Limitations

- 5M tokens is a small budget; all validation curves are still decreasing, so the ranking may change with more data.
- Single seed; differences of 0.001–0.024 CE are small and may be within seed noise.
- TinyStories is a narrow synthetic dataset; the results do not generalize to broader language modeling.
- The B1 arm includes a documented aborted segment and resume; its successful trajectory is 5M tokens, but total B1 compute was higher.
- Generation samples are greedy and short; they are not a language-quality benchmark.

**Bottom line:** At 5M tokens, B1 and R1 are effectively tied; R4 has the best token-matched validation CE, but the gain is small and comes at roughly 3.7× training and 4.3× forward cost. This pilot does **not** justify a claim that recursive refinement is compute-efficient, nor a claim of general language intelligence.

---

### 9.4 What this pilot establishes and does not establish

1. **Refinement is implemented, but only R4 exercises it.** The `trm_lm` forward
   carries a single latent through `n_refine` shared-block passes, with input
   injection and no detach; each pass influences the next and the final logits
   come from the final latent. This is not the original TRM H/L + ACT recursion.
   R1 uses one pass and is therefore a one-pass control, not a recursive test.
2. **Refinement’s benefit is not established.** R4 is better than R1 by
   0.023769 CE (0.889%) at the same token budget, but it costs 3.67× the training
   time and 4.27× the forward latency. The pilot is not compute-matched and uses
   one seed. The gain could come from extra effective depth, weight sharing, or
   the latent carry rather than refinement per se. A compute-matched baseline and
   no-carry/detached-carry ablations are required.
3. **Generalization is demonstrated only within TinyStories.** The validation
   documents are disjoint from training and the tokenizer is train-only; CE fell
   from ~9.25 to ~2.65 on 102,905 held-out target tokens. The models produce
   coherent continuations of held-out prompts, but the text is highly templated
   and repetitive. No out-of-distribution test, n-gram memorization audit, or
   exact-retrieval test was run, so broad generalization and freedom from
   memorization are not established.

## 10. Recommendation and exact next step

**Primary recommendation: a compute-matched, multi-seed comparison, not further pass scaling.**

The next experiment should:

1. Match B1’s measured training and forward compute to R4 (for example, a deeper/wider conventional baseline, or a larger token budget for B1) at the same evaluation protocol.
2. Run at least 3 seeds per arm to determine whether the ~0.024 CE R4 gain is robust.
3. Only if the gain survives should a larger data budget (e.g., 50–100M tokens) be considered.
4. An architectural revision (per-pass supervision, a different latent-state design, or adaptive passes) is a later option; the current evidence does not justify changing the architecture before resolving the compute-matching and seed-noise questions.

**Do not** simply increase `n_refine` further under this authorization. **Do not** launch a larger pretraining run, additional seeds, or distillation without separate approval.

**Exact next step:** obtain authorization for a compute-matched, multi-seed follow-up at the same 5M-token budget (or a larger budget if separately approved). This pilot’s training is complete; no further training was started.

---

## 11. Reproducibility and artifacts

Experiment root: `experiments/trm_lm_pilot_5m/` (2.0 GB total; local only, excluded from the commit by `.gitignore`).

| Artifact | Path |
|---|---|
| Frozen configuration (v2) | `FROZEN_CONFIG.json` |
| Original frozen configuration (v1, 20k docs) | `FROZEN_CONFIG_v1_20k.json` |
| Comparison metrics | `comparison.json` |
| B1 original/aborted run | `b1/` (log `logs/b1.log`) |
| B1 resumed final run | `b1_resumed/` (log `logs/b1_resumed.log`) |
| R1 run | `r1/` (log `logs/r1.log`) |
| R4 run | `r4/` (log `logs/r4.log`) |
| LR sanity runs | `lr_sanity/` (logs `logs/lr_sanity_*.log`) |
| One-step checks | `smoke/` (logs `logs/smoke_*.log`) |
| Filtered generations | `generation_distinct_filtered.json` |
| Raw generations | `generation_distinct_all.json` |
| Validation/training metrics | `{b1,b1_resumed,r1,r4}/metrics.jsonl` |
| Final summaries | `{b1_resumed,r1,r4}/summary.json` |

Representative training command (R4; B1/R1 differ only in `--arch`/`--n-refine` and B1’s resume):

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python train_trm_lm.py \
  --workdir experiments/trm_lm_pilot_5m/r4 \
  --arch trm_lm --n-refine 4 \
  --device cuda --dtype bfloat16 \
  --max-train-tokens 5000000 --max-steps 100000 \
  --max-train-docs 40000 --max-val-docs 512 --max-scan-rows 60000 \
  --tokenizer-docs 20000 --tokenizer-vocab-size 8000 \
  --seq-len 256 --batch-size 4 --hidden-size 512 --num-layers 2 --num-heads 8 \
  --lr 3e-4 --optimizer adamw --grad-clip 1.0 --seed 1337 \
  --save-every-tokens 1000000 --val-every-tokens 1000000 \
  --num-generation-prompts 5 --generate-new-tokens 64
```

`HSA_OVERRIDE_GFX_VERSION=10.3.0` is an opt-in, unofficial gfx1032→gfx1030 runtime mapping for this consumer GPU. It was set only in the test commands; no global shell, driver, or system GPU configuration was changed.

---

## 12. Code changes and verification status

**New/changed for the pilot:**
- `models/recursive_reasoning/causal_transformer_lm.py` — parameter-matched conventional decoder-only causal transformer baseline.
- `dataset/tinystories_lm.py` — exact token-budget truncation, resume `skip_tokens`, and validated helpers.
- `train_trm_lm.py` — token-budget enforcement, token-interval checkpoint/validation, metrics JSONL, baseline support, fixed-prompt generation, resume-aware summary.
- `tests/test_trm_lm.py` — baseline and token-accounting tests.

**Reference preservation:** the 16 reference code/config files remain byte-identical (`tests/reference_hashes.json`). The follow-up commit contains the new adaptation/pilot code, tests, research reports, and the README fork notice; it does not modify the Samsung reference implementation. The 2.0 GB `experiments/` directory is local-only and excluded from the commit.

**Tests executed before the pilot:**
- CPU suite: **36 passed, 3 skipped**.
- GPU suite: **39 passed** with the ROCm override.
- Baseline parameter-match, causal leakage, finite gradients/optimizer update, checkpoint round-trip, generation, token-budget, and resume-skip tests passed.
- One-step B1/R1/R4 checks: exactly 500 tokens, 10,911,744 params, checkpoint diff 0.0.
- LR sanity: 20 steps at 3e-4, stable.
- Main pilot: all three arms reached exactly 5,000,000 tokens; all final checkpoints reloaded with max abs diff 0.0.

**Known limitations / unresolved blockers:** none that prevented the pilot. The remaining limitations are scientific, not software: single seed, 5M-token budget, 11M parameters, narrow dataset, no compute-matched baseline, and an unofficial GPU ISA override. The B1 aborted-segment overhead is documented. No pre-existing Samsung test suite exists in this checkout; the reference regression tests added in the implementation phase were used instead.

---

## 13. Small-scale pilot statement

This is a **5M-token, single-seed, ~11M-parameter pilot** on the narrow TinyStories dataset. It is **not** evidence of general-purpose intelligence, general language competence, or a validated advantage for recursive reasoning. The generated stories are coherent within this simple domain but remain templated and repetitive. Any larger claim requires a compute-matched, multi-seed, larger-budget experiment under separate authorization.
