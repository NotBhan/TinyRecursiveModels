# RE:Inference — Coding Competence and Task-Adaptive VRAM

**Implementation and experiment report**
**Date:** 2026-10-22
**Branch:** `re-coding-adaptive` (base commit `2cffb3c`)
**Status:** Experiments executed on the local AMD ROCm GPU. All numbers below are measured unless labelled a hypothesis.

---

## 1. Executive summary

**The 0% syntax-validity result was not a property of the neural architecture. It was caused by four measurable defects in the data and evaluation path, plus severe under-training on code:**

1. The MBPP prompt did not contain the function name called by the tests. The name appears in only **21/500** test problems, so pass@1 was structurally near-zero even for a perfect body generator.
2. `extract_python_code` stripped the leading indentation of body-only completions. A valid body such as `'  res = tuple(test_tup)\r\n  return (res)'` was turned into an invalid program before parsing.
3. The tokenizer was trained only on TinyStories: **1.58 chars/token** on MBPP code, with `def` split into `d` + `ef`. The code-aware tokenizer reaches **2.42 chars/token**.
4. Only **38.7k function-synthesis target tokens** (7.7% of the 500k-token budget) reached the loss, and 12.9% of synthesis examples were truncated and forced to end with `<eos>`.

After correcting the harness, training a code-aware tokenizer, and training a code-heavy backbone, **syntax validity rose from 0% to 48.9% at fixed depth 2 (500k checkpoint) and 75.6% at depth 1 (multi-depth fine-tune)** on 45 held-out MBPP validation tasks. A bounded syntax-triggered repair policy reached **82.2% validation syntax** on the multi-depth checkpoint; after the stdlib-instruction fine-tune it reached **84.4% validation syntax** and **95.0% syntax on 100 held-out test tasks**. A controlled 500k-token ablation with the old TinyStories tokenizer measured **0.0% syntax** under the same data/harness, versus 48.9% with the code-aware tokenizer. **pass@1 remains 0%** on both validation (45 tasks) and test (100 tasks). Among syntax-valid programs, all failed the tests (assertion or runtime errors). The first milestone is met; coding competence is not demonstrated.

**Adaptive computation:** multi-depth training makes all depths coherent and the learned controller improves TinyStories CE (3.879 adaptive vs 4.088 fast), but the controller's LM-CE objective does not optimize code syntax. A syntax-triggered bounded repair policy is the only adaptive mechanism that improved a coding metric (syntax 75.6% → 82.2%) at lower transition count than fixed maximum depth.

**Adaptive VRAM:** in the full 45-task evaluation of the multi-depth checkpoint, measured peak allocated memory is **67.1 MB at depth 1, depth 2, and depth 4**, while executed transitions vary from 61 to 768. Depth adaptation changes latency and compute, **not peak VRAM**. Working memory changes by only about 0.4 MB on average (up to 0.75 MB on easy tasks) from a smaller context cap, and by about 2.2 MB from last-token-only logits; the tiered controller saves 2.5% (67.1 → 65.4 MB). The model-weight floor is **23.2 MB**. At this scale, the fixed per-cycle activation workspace dominates; this is exactly the distinction the research brief required us to verify.

---

## 2. What was implemented

### 2.1 New files

| File | Purpose |
|---|---|
| `dataset/code_pipeline.py` | Code-aware tokenizer, syntax-validated Python corpus, MBPP signature prompts, stdlib docstring instructions, token-quota mixture, no-corrupt-truncation packing. |
| `dataset/build_code_tokenizer.py` | CLI to train the mixed story+code BPE tokenizer. |
| `train_re_coding.py` | Backbone and multi-depth/controller training, exact token budgeting, checkpointing, evaluation. |
| `evaluators/re_eval.py` | Signature prompts, indentation-preserving extraction, full-code reconstruction, sandboxed execution, error taxonomy, peak-memory measurement, difficulty score, syntax repair. |
| `evaluators/run_re_experiments.py` | Quality, depth, difficulty, memory, and language-retention experiment driver. |
| `tests/test_re_coding_pipeline.py` | 10 tests for prompts, masking, tokenizer round-trip, quotas, tier bounds, last-token logits, and indentation preservation. |
| `docs/research/RE_CODING_ADAPTIVE_BASELINE.md` | Baseline reproduction and failure audit. |
| `docs/research/RE_CODING_ADAPTIVE_BASELINE.json` | Machine-readable baseline: environment, hashes, seeds, commands, metrics, supplemental runs, VRAM. |
| `docs/research/RE_CODING_ADAPTIVE_FAILURE_ANALYSIS.json` | Per-category and representative MBPP failure analysis after the fixes. |
| `docs/research/RE_CODING_ADAPTIVE_FINAL_REPORT.md` | This report. |

### 2.2 Modified files

| File | Change |
|---|---|
| `models/recursive_reasoning/trm_lm_adaptive.py` | Added `last_token_only` logits path and per-token entropy/margin generation diagnostics. No change to the recurrence or parameter count. |
| `evaluators/coding_eval.py` | Added `extract_python_code(..., preserve_leading_indent=True)`; legacy behavior unchanged by default. |

The frozen Samsung TRM reference, the historical `trm_lm_pilot_5m/` artifacts, and the historical `trm_lm_coding_reasoning/pilot_500k/` checkpoint were not modified.

### 2.3 Verification

```bash
TRM_LM_TEST_GPU=1 TRM_LM_ADAPTIVE_TEST_GPU=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  PYTHONPATH=. .venv-trm-lm/bin/pytest tests/ -q -p no:warnings
# 112 passed in 11.33 s
```

The suite includes 10 new tests (signature prompts, body/prompt masking, tokenizer round-trip, quota mixture, tier bounds, difficulty bounds, last-token-only logits, indentation preservation, paired deltas). The reference-hash test reports 16/16 unchanged. Checkpoint reload differences for the new runs were 0.03125–0.0625 in bf16 (the same order as the historical pilot's 0.03125).

---

## 3. Baseline and diagnosis

The full audit is in `RE_CODING_ADAPTIVE_BASELINE.md`. The reproduced baseline:

| Split | Mode | pass@1 | syntax valid |
|---|---|---:|---:|
| MBPP validation (43) | fast | 0.0% | 0.0% |
| MBPP validation (43) | adaptive | 0.0% | 0.0% |
| MBPP test (43) | fast | 0.0% | 0.0% |
| MBPP test (43) | adaptive | 0.0% | 0.0% |

Realized training tokens in the historical 500k pilot: 428,154 language (85.6%), 71,932 coding (14.4%), of which 38,716 were function-synthesis targets (7.7%). The model is 11.5M parameters trained from scratch.

Measured defects:

- **Function-name omission:** 21/500 test tasks mention the canonical function name.
- **Indentation stripping:** raw `'  res = tuple(test_tup)\r\n  return (res)'` became `'res = tuple(test_tup)\r\n  return (res)'` before parsing.
- **Tokenizer:** 1.58 chars/token on MBPP code; 42,164 tokens for the 66,715-character MBPP train code. New tokenizer: 2.42 chars/token, 27,565 tokens.
- **Truncation:** 48/371 (12.9%) synthesis examples were cut at 256 tokens and the final token was overwritten with `<eos>`.
- **Generation budget:** the old harness did not reserve new-token space explicitly; the corrected harness does. On the historical validation prompts this was not the dominant cause (mean prompt 42 tokens), but it is a real defect for longer prompts.

The harness corrections alone did not change the old checkpoint's score: 45 validation tasks with signature prompts and indentation-preserving extraction still measured **0% syntax and 0% pass**. The old model's raw bodies had no indentation to preserve and were not valid Python.

---

## 4. Experiment A — harness corrections

Corrected protocol:

- Prompt = problem statement + `### Solution:` + imports/helpers + the exact target `def <name>(<args>):` line.
- Target = function body only.
- Generation = body only; the prompt context is prepended before parsing/execution.
- Leading indentation of a body-only generation is preserved.
- Generation reserves `max_new_tokens` inside the context window; prompt truncation is recorded.
- Tests are hidden and never placed in the prompt.

The corrected harness is used for every result below. No test was changed to make a model pass.

---

## 5. Experiment B — data, tokenizer, and objective

### 5.1 Code-aware tokenizer

Trained on 4,000 TinyStories documents plus 8.0 MB of syntax-valid Python from the local standard library. Vocabulary 8,192, byte-level BPE, special IDs unchanged. MBPP code density improved from **1.58 to 2.42 chars/token**.

### 5.2 Data pipeline

- **Code LM:** 4,000 syntax-validated Python files, packed into 256-token causal blocks. Incomplete final blocks are dropped, never padded with a forced EOS.
- **MBPP instruction:** 371 deduplicated training tasks (validation/test matches quarantined), signature in the prompt, body as the target. Over-length examples are dropped (8/371).
- **Language:** TinyStories continuation.
- **Mixture:** token quotas, not example quotas. The 3M backbone realized exactly 2,100,224 code-LM tokens (70.0%), 599,592 instruction tokens (20.0%), and 300,204 language tokens (10.0%).

### 5.3 Training runs

| Run | Tokens | Steps | Duration | Throughput | Final TinyStories CE | Notes |
|---|---:|---:|---:|---:|---:|---|
| Historical pilot | 500k | 1,045 | 449 s | 1,113 tok/s | 3.53 (old tokenizer) | 0% syntax |
| New backbone, 500k checkpoint | 500,091 | — | — | — | — | fixed depth 2 |
| New backbone, 3M | 3,000,000 | 5,308 | 2,026 s | 1,481 tok/s | 4.302 | fixed depth 2 |
| New backbone, 1M checkpoint | 1,000,163 | — | — | — | 4.353 | fixed depth 2 |
| Multi-depth fine-tune (from 1M) | 1,000,000 | 1,773 | 1,047 s | 955 tok/s | 4.088 (fast) | depths 1–4 + controller |
| Stdlib-instruction fine-tune | 1,000,000 | 3,062 | 1,734 s | 577 tok/s | 4.034 (fast) | 35% stdlib docstrings |

**Fixed-depth-2 syntax on 45 MBPP validation tasks:**

| Checkpoint | Syntax valid | pass@1 |
|---|---:|---:|
| New tokenizer, 500k, seed 1337 | **48.9%** | 0.0% |
| New tokenizer, 500k, seed 2025 | 40.0% | 0.0% |
| New tokenizer, 1M | 35.6% | 0.0% |
| New tokenizer, 3M | 28.9% | 0.0% |

The 500k result is robust to seed at the level of "non-zero syntax" (40.0–48.9%) but not in its exact value; the gap is 8.9 points. Both seeds measured 0% pass@1. More fixed-depth-2 tokens are **not** monotonically better: the 500k checkpoint is the best, and the 3M checkpoint overfits to depth 2 (depth-1 syntax 0%, depth-4 syntax 0%). The multi-depth fine-tune from the 1M checkpoint repaired depth robustness: depth-1 syntax 75.6%, depth-4 syntax 35.6%. The stdlib-instruction fine-tune improved depth-1 syntax further to 77.8% and syntax-triggered repair to 84.4% on validation.

**Key result at fixed depth 2 (1M checkpoint):** syntax validity 35.6%, up from 0%. Error counts: 29 syntax, 7 runtime, 9 assertion. Of 16 syntactically valid programs, **0 passed**; 9 failed assertions and 7 raised runtime errors. The syntax failure is fixed; semantic correctness is not.

**Depth sweep on the multi-depth checkpoint (45 validation tasks):**

| Depth | Syntax valid | pass@1 | first-test pass | runtime errors | syntax errors | assertion errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **75.6%** | 0.0% | 0.0% | 24 | 11 | 10 |
| 2 | 35.6% | 0.0% | 4.4% | 8 | 29 | 8 |
| 3 | 31.1% | 0.0% | 6.7% | 6 | 31 | 8 |
| 4 | 35.6% | 0.0% | 4.4% | 10 | 29 | 6 |

Depth 1 has the best syntax; depths 2–4 produce a few first-test passes but break syntax more often. Extra refinement is not syntax-preserving. This is a measured trade-off, not a hypothesis.

### 5.4 Failure taxonomy after the data fixes

Representative depth-1 failures are mode-collapsed generic bodies:

```python
def find_Min_Sum(num):
    if (n == 0):
        return (n - 1)
    return (n - 1)
```

`n` is not in the signature, so the program raises `NameError`. Other tasks return a constant and fail assertions. The model has learned Python syntax and the shape of a function body, but not the problem-to-algorithm mapping. Depth 3 produces more task-shaped loops but still wrong values.

---

## 6. Experiment D — architecture decision

**No neural-architecture change was made.** The evidence does not justify one yet:

- The dual-state shared-weight model can represent valid Python bodies (82.2% syntax with repair) and multi-depth training makes all depths coherent.
- The remaining failure is semantic: generic bodies that ignore the signature/problem, not an inability to represent code.
- A larger or differently connected model might improve semantic mapping, but that hypothesis is not tested here. The highest-value next experiment is more diverse instruction data (the stdlib docstring fine-tune) or a larger parameter budget, not another recurrence variant.

The only model change is an inference-memory change (`last_token_only` logits), which does not alter the recurrence or parameter count.

---

## 7. Adaptive computation

### 7.1 Learned controller

The controller is a 65,793-parameter per-position MLP trained on the marginal utility of another refinement cycle:

```
target = 1[CE(depth d) - CE(depth d+1) > threshold]
```

It was trained for 1M tokens on the multi-depth run (controller loss ~0.48–0.63, from 0.693). On 45 MBPP validation tasks, adaptive mode measured:

| Metric | Fast (depth 1) | Fixed depth 2 | Fixed max (4) | Learned adaptive |
|---|---:|---:|---:|---:|
| Syntax valid | 75.6% | 35.6% | 35.6% | 35.6% |
| pass@1 | 0.0% | 0.0% | 0.0% | 0.0% |
| Mean transitions | 61.4 | 127.6 | 282.0 | 195.6 |
| Mean latency | 165.5 ms | 229.9 ms | 458.9 ms | 469.6 ms |
| Peak allocated | 67.1 MB | 67.1 MB | 67.1 MB | 67.1 MB |

On TinyStories, adaptive depth **improved** CE: 3.879 vs 4.088 fast. On code, the controller allocates more depth than depth 1 but does not recover depth-1 syntax. The controller optimizes LM cross-entropy, and LM CE is not a reliable proxy for code syntax or execution. After the stdlib-instruction fine-tune, adaptive mode measured 40.0% syntax, 4.4% first-test pass, 163.4 transitions, and 352.9 ms — better than the multi-depth adaptive result but still far below depth-1 syntax (77.8%).

### 7.2 Difficulty estimator validation

The label-free difficulty score combines normalized token entropy, top-1/top-2 margin, syntax failure, and generation length. Validated against teacher-forced CE improvement from depth 1 to depth 4 on 42 tasks:

- Mean `CE(depth 1) - CE(depth 4)` = **+0.138** (extra depth helps LM CE on average).
- Spearman correlation between the score and the depth gain = **-0.184**.
- Only a minority of tasks benefit from extra depth; the estimator does not identify them.

**The estimator is not validated as a difficulty predictor.** It should not be used to claim that easy tasks receive less computation.

### 7.3 Syntax-triggered bounded repair (the useful adaptive policy)

Policy: generate at depth 1; if the result fails `ast.parse`, retry at depth 2, then 3, then 4; stop at the first syntactically valid program; if none is valid, return the last attempt. Maximum depth is an explicit hard bound. Tests are never consulted.

| Policy | Syntax valid | pass@1 | mean attempts | mean transitions | mean latency | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| Depth 1 only | 75.6% | 0.0% | 1.00 | 61.4 | 165.5 ms | 67.1 MB |
| Fixed depth 2 | 35.6% | 0.0% | 1.00 | 127.6 | 229.9 ms | 67.1 MB |
| Fixed max 4 | 35.6% | 0.0% | 1.00 | 282.0 | 458.9 ms | 67.1 MB |
| **Syntax repair** | **82.2%** | 0.0% | 1.67 | 263.7 | 512.7 ms | 67.1 MB |

Attempt histogram: 34 tasks stopped at depth 1; 1 at depth 2; 1 at depth 3; 9 stopped at depth 4. Of those 9 depth-4 attempts, 1 became syntactically valid and 8 remained invalid. The repair fixed 3 additional syntactically invalid tasks (37/45 valid vs 34/45 at depth 1). It did not produce a passing program; all selected programs remain semantically wrong.

This is a genuine task-adaptive compute result: most tasks use one pass, a minority use more, and syntax validity improves. It is **not** a VRAM result: every attempt runs sequentially, so peak allocated memory is unchanged.

### 7.4 Post-training recurrence verification

A separate Phase-8 verification traced the executed tensor operations, broke each latent feedback pathway, ran matched-compute controls, perturbed intermediate states, and measured quality at depths 1/2/4. Full report: `experiments/trm_lm_coding_reasoning/RECURSION_VERIFICATION.md`.

- **State dependence is verified.** Normal depth 4 differs from depth 1 (max logit change 3.72, 50% top-1 changes on the fixed batch). Resetting both `z_H` and `z_L` each cycle makes depth 4 **exactly equal** to depth 1; resetting either pathway alone changes the output. Perturbations at depth 1/2 propagate monotonically to the final logits; the zero-perturbation control is exact.
- **Refinement benefit is unproven / task-dependent.** TinyStories CE improves from 4.0341 (depth 1) to 3.8448 (depth 2) and then worsens to 3.9880 (depth 4). MBPP syntax degrades from 77.8% (depth 1) to 46.7% (depth 2) and 35.6% (depth 4) on validation; test: 87.0% → 49.0% → 47.0%. First-test pass increases slightly at depth 4 (+3/45 validation, +4/100 test), but pass@1 remains 0%.
- **Compute efficiency is not demonstrated.** Depth 4 uses ~4× the transitions but loses 21 validation and 44 test syntax-valid tasks versus depth 1. The matched-compute no-feedback control reaches depth-1 syntax at the same 4× compute.

The model performs genuine recurrent state transitions, but the evidence does not justify calling the refinement reliably useful. The best code syntax is at depth 1; the best language CE is at depth 2.

---

## 8. VRAM measurements

### 8.1 Measurement method

- `torch.cuda.synchronize()` before and after.
- `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.empty_cache()` before each configuration.
- Report `max_memory_allocated` (live tensor memory) and `max_memory_reserved` (allocator cache) separately.
- Model weights counted separately from working memory.
- The model is 11,600,129 parameters; bf16 weights = **23.2 MB**.
- Host peak RSS reported by `resource.getrusage`: **1.59–1.82 GB** across configurations (interpreter/tokenizer/sandbox overhead), far below the 14 GiB system RAM. It did not vary systematically with depth and is not a limiting factor.

### 8.2 Depth does not change peak VRAM

Across the full 45-task evaluation of the multi-depth checkpoint, `fast` (depth 1, 61 transitions), `fixed_2` (128), and `fixed_max` (282) all measured **67.1 MB peak allocated** and **86.0 MB peak reserved**. The difference between depth 1 and depth 4 is **0.0 MB** in peak allocated memory, despite a 4.6× difference in executed transitions.

The reason is architectural: each high cycle allocates its workspace, uses it, and frees it before the next cycle. Peak live memory is one cycle's workspace plus the recurrent states, not the sum over cycles. The number of cycles changes time, not peak residency. **This confirms the brief's warning: adaptive recursion alone does not make VRAM adaptive.**

### 8.3 What does change working memory

Microbenchmark on 3 easy and 3 hard tasks, 64 new tokens per attempt:

| Configuration | Mean peak allocated | Mean working memory | Mean latency |
|---|---:|---:|---:|
| fast, context cap 128 | 59.54 MB | 36.34 MB | 196.0 ms |
| fast, context cap 256 | 59.92 MB | 36.72 MB | 194.2 ms |
| fixed max, cap 128 | 60.29 MB | 37.09 MB | 833.4 ms |
| fixed max, cap 256 | 60.66 MB | 37.46 MB | 838.7 ms |
| fixed max, cap 256, full logits | 62.90 MB | 39.70 MB | 818.2 ms |

- Last-token-only logits save **~2.2 MB** (3.6%) because the model otherwise materializes `[1, T, 8192]` logits at every generated token.
- Reducing the context cap from 256 to 128 saves **~0.4 MB on average** (up to ~0.75 MB on easy tasks) for these short prompts. The attention/MLP workspace is dominated by fixed per-cycle tensors at this scale, not by the 256-token maximum.
- The tiered controller measured 65.4 MB vs 67.1 MB peak allocated (−2.5%), driven by shorter contexts on some tasks, not by fewer refinement steps. This saving is **not free**: the tiered controller measured 35.6–40.0% syntax, far below depth-1's 75.6%. It is a memory/quality trade-off, not a quality-preserving optimization.

### 8.4 Weight floor and infeasible budgets

The unavoidable resident-model floor is **23.2 MB** (bf16 weights). Peak allocated is `weights + working`. At this scale, working memory is 36–44 MB and the floor is a minority of total VRAM. A budget below 23.2 MB is infeasible without offloading/paging; no such budget was requested or claimed. Offloading was not implemented because the floor is far below the 8.57 GB device capacity and would add latency without addressing the research question.

### 8.5 Honest interpretation

At 11.6M parameters and a 256-token context, **peak VRAM is not sensitive to recursive depth**. The measured adaptive-memory savings come from context length and last-token logits, not from fewer refinement steps. To obtain large task-adaptive VRAM savings, the working set must be dominated by sequence-length-dependent state (a KV cache, a larger context, a larger hidden size, or per-depth cached activations). Those are architecture/implementation choices, not consequences of halting.

---

## 9. Language retention

TinyStories validation (same new tokenizer, held-out documents):

| Checkpoint | Fast CE | Fast PPL | Adaptive CE | Adaptive PPL |
|---|---:|---:|---:|---:|
| 1M fixed-depth backbone | 4.353 | 77.7 | 4.495 | 89.6 |
| Multi-depth fine-tune | 4.088 | 59.8 | **3.879** | **48.4** |
| Stdlib-instruction fine-tune | 4.034 | 56.5 | **3.850** | **47.0** |

The multi-depth and stdlib models retain language modeling and improve CE with extra depth. The historical pilot's 2.65 CE is **not comparable** because it used a different tokenizer and data mixture; we do not claim a regression or improvement relative to it.

---

## 10. Ablations

| Change | Syntax validity (45 val) | pass@1 | Evidence |
|---|---:|---:|---|
| Historical 500k pilot, old harness | 0.0% | 0.0% | reproduced |
| Historical checkpoint + corrected signature/indentation harness | 0.0% | 0.0% | harness alone does not help an undertrained model |
| Old TinyStories tokenizer + new data/harness, 500k, depth 2 | 0.0% | 0.0% | tokenizer ablation; output collapses to whitespace |
| Code-aware tokenizer + code-heavy 500k backbone, fixed depth 2 | **48.9%** | 0.0% | data + training + harness |
| Same, 1M checkpoint, fixed depth 2 | 35.6% | 0.0% | overfitting begins |
| Same, 3M checkpoint, fixed depth 2 | 28.9% | 0.0% | overfit to depth 2 |
| Multi-depth fine-tune, depth 1 | **75.6%** | 0.0% | depth robustness |
| Multi-depth fine-tune, depth 3/4 | 31.1% / 35.6% | 0.0% | refinement can break syntax |
| Syntax-triggered repair, multi-depth | **82.2%** | 0.0% | adaptive compute |
| Stdlib-instruction fine-tune, depth 1 | **77.8%** | 0.0% | more instruction data |
| Stdlib-instruction fine-tune, syntax repair | **84.4%** | 0.0% | best validation syntax |
| Stdlib-instruction, held-out test 100, depth 1 | 87.0% | 0.0% | held-out |
| **Stdlib-instruction, held-out test 100, syntax repair** | **95.0%** | **0.0%** | best measured syntax |
| Learned adaptive controller | 35.6–40.0% | 0.0% | LM-CE objective does not optimize syntax |

The harness correction is necessary but not sufficient. The dominant syntax improvement comes from the combination of a code-aware tokenizer, a code-heavy loss distribution, syntax-valid training data, multi-depth training, and indentation-preserving evaluation. The 500k→1M→3M fixed-depth-2 sequence shows that more tokens are not automatically better: depth-specific overfitting regresses both shallow and deep behavior.

**Tokenizer ablation (executed):** the same new data mixture, harness, seed, and 500k-token budget with the old TinyStories tokenizer measured **0.0% syntax at depth 1 and 0.0% at depth 2** (11,501,825 parameters; 23.0 MB). Generations collapsed to whitespace:

```
'\t\t\t\t\t\t\t\t...'
'    return (num(nums)\r\n  return (res)'
```

The new-tokenizer 500k checkpoint under the identical protocol measured **48.9% syntax at depth 2**. This supports the tokenizer as a first-order factor, not a minor inefficiency: the old tokenizer splits Python keywords and identifiers into many fragments, and the model spends its 500k-token budget learning whitespace/subword statistics instead of code structure. The parameter counts differ by 98,304 (vocabulary size 8,000 vs 8,192), and the same token budget covers fewer code characters with the old tokenizer; both differences are disclosed. The data-mixture effect is still not isolated from the harness effect; the tokenizer effect is isolated only at the same token budget.

---

## 11. Test-split results

Held-out MBPP test evaluation was run **after** model/config selection on validation, on the first 100 test tasks. The test split was never used for training or model selection.

Checkpoint: `stdlib_instr_1m/checkpoint_final.pt`.

| Configuration | Syntax valid | pass@1 | first-test pass | mean transitions | mean latency | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| Depth 1 (fast) | 87.0% | 0.0% | 0.0% | 43.7 | 91.1 ms | 67.1 MB |
| Fixed max (4) | 47.0% | 0.0% | 4.0% | 191.6 | 263.5 ms | 67.1 MB |
| **Syntax repair (1→4)** | **95.0%** | **0.0%** | 1.0% | 97.0 | 169.0 ms | 67.1 MB |

Level-1 generation validity on the same test tasks: depth-1 nonempty rate 96.0%, entry-point-present rate 87.0%, syntax-valid rate 87.0%; fixed-max nonempty 98.0%, entry-point 47.0%, syntax 47.0%. The repair aggregate records syntax and execution, not nonempty/entry-point separately.

The repair policy reached **95/100 syntactically valid programs** on held-out MBPP test tasks, up from 0/43 in the historical pilot. One repair-selected program passed its first test but none passed all tests. With 0/100 full passes, the 95% upper confidence bound is about 3% (rule of three); the experiment does not establish non-zero coding competence. Depth 1 already produces syntactically valid programs for 87% of test tasks, so most of the syntax gain comes from the data/model, not from repair; repair adds ~8 points at 97 mean transitions (2.2× depth-1's 43.7, but 0.51× fixed-max's 191.6).

The attempt histogram and per-task rows are in `experiments/re_coding_adaptive/eval_stdlib_instr_1m_test100/results.json`.

**Honest reading:** syntax validity is solved at this benchmark/protocol. Execution correctness is not. A 95% syntax rate with 0% pass@1 means the model reliably writes parseable but wrong programs.

---

## 12. Limitations and hypotheses

**Measured limitations:**

- pass@1 is 0% on 45 validation tasks for every tested configuration. The 95% upper confidence bound for 0/45 is about 6.6% (rule of three); no claim of coding competence is made.
- The learned controller's LM-CE objective is misaligned with code syntax. Its difficulty estimator has near-zero/negative correlation with measured depth gains.
- Depth 1 is the best syntax configuration; deeper refinement is not syntax-preserving and only occasionally improves first-test correctness.
- Peak VRAM is depth-invariant in this architecture/scale. Measured memory savings are ~0.4 MB from a smaller context cap and ~2.2 MB from last-token-only logits, not from halting.
- The 3M continuation overfit to depth 2; more training is not monotonically better. The best fixed-depth-2 checkpoint was 500k, not 1M or 3M.
- The 500k backbone was run with two seeds (1337: 48.9% syntax, 2025: 40.0% syntax; both 0% pass). All other runs use one seed; their seed variance is not measured.
- The tokenizer effect is isolated (old tokenizer 0% vs new tokenizer 48.9% at 500k), but the data-mixture effect is not isolated from the harness effect.
- The stdlib-instruction fine-tune improved syntax (84.4% validation with repair) but not pass@1. Its test-split result is 95.0% syntax and 0% pass@1 on 100 tasks.
- The model is 11.6M parameters and the training budget is 0.5–1M tokens per final stage. This is far below the data/capacity needed for general code synthesis.

**Untested hypotheses:**

- More diverse instruction data (stdlib docstrings) improves semantic mapping.
- A larger model or more MBPP-style instruction tokens would produce non-zero pass@1.
- A syntax-aware halting objective (not LM CE) would allocate depth better.
- A KV cache or longer context would make working-memory adaptation materially larger.
- Multi-seed runs would preserve the ranking of depth 1 vs deeper refinement.

---

## 13. Reproducible commands

```bash
# Environment (ROCm; the HSA override is specific to the local RX 6600M)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=.

# Code-aware tokenizer
.venv-trm-lm/bin/python -m dataset.build_code_tokenizer \
  --workdir experiments/re_coding_adaptive/tokenizer \
  --tinystories-workdir experiments/trm_lm_pilot_5m/r1 \
  --vocab-size 8192 --story-docs 4000 --code-bytes 8000000 \
  --max-code-files 4000 --force

# 3M code-heavy backbone (the 1M checkpoint is .../checkpoint_tokens_1000163.pt)
.venv-trm-lm/bin/python train_re_coding.py \
  --stage backbone --workdir experiments/re_coding_adaptive/backbone_3m \
  --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
  --max-train-tokens 3000000 --fixed-depth 2 \
  --batch-size 4 --seq-len 256 --lr 3e-4 --warmup-steps 200 \
  --save-every-tokens 500000 --val-every-tokens 500000 --skip-code-eval

# Multi-depth + controller fine-tune from the 1M checkpoint
.venv-trm-lm/bin/python train_re_coding.py \
  --stage controller --workdir experiments/re_coding_adaptive/multidepth_1m_from_1m \
  --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
  --init-checkpoint experiments/re_coding_adaptive/backbone_3m/checkpoint_tokens_1000163.pt \
  --max-train-tokens 1000000 --max-high-cycles 4 --depth-sampling uniform \
  --controller-loss-weight 0.25 --lr 1e-4 --warmup-steps 100 \
  --save-every-tokens 250000 --val-every-tokens 250000 --skip-code-eval

# Quality + memory evaluation
.venv-trm-lm/bin/python -m evaluators.run_re_experiments \
  --checkpoint experiments/re_coding_adaptive/multidepth_1m_from_1m/checkpoint_final.pt \
  --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
  --workdir experiments/re_coding_adaptive/eval_multidepth_1m \
  --split validation --limit 45 --max-new-tokens 96 \
  --modes fast,fixed2,fixedmax,adaptive,tiered,repair \
  --memory-easy-hard 3 --sandbox-python /usr/bin/python3.14

# Tests
.venv-trm-lm/bin/pytest tests/test_re_coding_pipeline.py -q -p no:warnings
.venv-trm-lm/bin/pytest tests/ -q -p no:warnings -k "not gpu"
```

---

## 14. Final recommendation

**What works:**

- The corrected evaluation protocol (signature prompt, indentation-preserving extraction, context reservation) is sound and should be retained.
- The code-aware tokenizer and token-quota mixture deliver the intended coding-token share.
- Depth-1 / shallow inference is the best syntax configuration for this model.
- Syntax-triggered bounded repair is a real, measurable adaptive-compute improvement: 75.6% → 82.2% syntax at 1.67 mean passes.
- Multi-depth training makes deeper inference coherent and improves language CE.

**What does not work:**

- The model does not synthesize correct programs. pass@1 is 0% on 45 validation tasks; syntax-valid programs fail assertions or raise runtime errors.
- The learned controller optimizes LM CE, not code correctness; it allocates depth that degrades syntax.
- Adaptive depth does not reduce peak VRAM. Only context/logit workspace changes do, and the measured savings are modest at this scale.

**Next highest-value experiment:** train the same architecture on a substantially larger and more diverse instruction corpus (stdlib docstrings plus MBPP-style problems), with an explicit syntax-preserving objective or a syntax-aware halting target, at a larger token budget (10–50M) and at least 3 seeds. Evaluate pass@1 on the held-out test split. Do not claim coding competence until execution-based pass@1 is non-zero. For VRAM adaptation, implement a KV cache or longer-context path and measure whether task-adaptive context tiers produce material peak-memory differences; do not attribute depth-only savings to VRAM.
