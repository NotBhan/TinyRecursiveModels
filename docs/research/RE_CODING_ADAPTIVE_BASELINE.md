# RE:Inference — Coding Baseline and Failure Audit

**Date:** 2026-10-22
**Branch:** `re-coding-adaptive` (base `2cffb3c`)
**Scope:** reproduce the 0% MBPP result, audit the full evaluation and data path, and isolate harness defects from model defects before changing the architecture.

This report contains **measured results only** unless a sentence is explicitly labelled a hypothesis.

---

## 1. Reproduced baseline

Source run: `experiments/trm_lm_coding_reasoning/pilot_500k/`
Checkpoint: `checkpoint_final.pt` (step 1,045; 500,000 target tokens)
Model: `DualStateModel`, 11,501,825 parameters, 2 layers (1 low + 1 high), shared weights, hidden 512, 8 heads, vocab 8,000, bf16.

| Split | Mode | pass@1 | syntax validity | error breakdown |
|---|---|---:|---:|---|
| MBPP validation (43) | Thinking OFF (fast) | 0.0% | 0.0% | 23 syntax, 20 runtime |
| MBPP validation (43) | Thinking ON (adaptive) | 0.0% | 0.0% | 43 syntax, 0 runtime |
| MBPP test (43) | Thinking OFF | 0.0% | 0.0% | 27 syntax, 16 runtime |
| MBPP test (43) | Thinking ON | 0.0% | 0.0% | 43 syntax, 0 runtime |

The zero result reproduces exactly. The model does learn Python fragments: post-training fast generations contain `def count_list(tuple):`, and adaptive generations contain `def count_list(tuple, n):` and arithmetic fragments. It does **not** produce a valid body.

A fresh, corrected-harness run on the same checkpoint (5 validation tasks, signature in the prompt, full-code reconstruction, 48 new tokens) also measured **0% syntax validity and 0% pass@1**. The corrected harness therefore does not manufacture a score change; it removes a separate defect described in Section 3.

---

## 2. Effective training signal

The nominal mixture was 60% language / 40% coding. The realized **target-token** mixture was:

| Source | Target tokens | Share | Examples |
|---|---:|---:|---:|
| TinyStories continuation | 428,154 | 85.6% | 2,451 |
| MBPP code continuation | 22,014 | 4.4% | 374 |
| MBPP function synthesis | 38,716 | 7.7% | 376 |
| MBPP output prediction | 11,202 | 2.2% | 978 |
| **Total coding** | **71,932** | **14.4%** | 1,728 |

The model is 11.5M parameters trained from scratch on 500k total tokens. It received **38.7k function-synthesis target tokens**. That is not enough to learn a problem-to-program mapping, and the mixture was sampled by example rather than by token, which is why the nominal 40% coding share became 14.4% of the loss-bearing tokens.

Additional measured defects in target construction:

- 48/371 function-synthesis examples (12.9%) exceed `seq_len=256` after the prompt is prepended.
- `pack_prompt_target` truncates at 256 and then **overwrites the final token with `<eos>`**, so the model is trained to stop in the middle of an expression. This is a target-construction defect, not a model limitation.

---

## 3. Evaluation-protocol defect: the prompt does not identify the function

The canonical MBPP tests call a specific function name. The historical prompt was only:

```
### Problem:
<problem text>
### Solution:
```

Measured over all 500 MBPP test tasks:

| Check | Count |
|---|---:|
| Tasks with a `def` in the canonical solution | 500 / 500 |
| Canonical function name mentioned in the problem text | **21 / 500 (4.2%)** |

Examples:

- Task 511 prompt: “Write a python function to find minimum sum of factors of a given number.”
- Hidden tests: `assert find_Min_Sum(12) == 7`, etc.
- The prompt never contains `find_Min_Sum`.

A model that produced a perfect body under a different name would fail every test with `NameError`. This is a **harness defect that makes pass@1 structurally near-zero**, independent of coding capability. The old harness also executed only the generated text, not the prompt context, so a body-only completion could never be executable even if it were correct.

The corrected protocol (`evaluators/re_eval.py`) does three things and only these three things:

1. Includes the exact target `def <name>(<args>):` line in the prompt (body excluded). This is the minimum information needed to make the task well-posed.
2. Executes `prompt_code_context + generated_body` and parses/executes the result.
3. Preserves the leading indentation of a body-only generation.

The tests remain hidden and are never placed in the prompt. The canonical body is never placed in the prompt. Both facts are checked by `tests/test_re_coding_pipeline.py`.

The fresh corrected-harness run on the old checkpoint still measured 0% syntax validity, so the harness correction is necessary but not sufficient.

### 3.1 Second harness defect: indentation stripping

`extract_python_code` called `raw_generation.strip()` and `text.strip()`. For a
full-function generation that is harmless, but for a body-only completion it
deletes the first line's indentation. Measured example on MBPP validation task
513 (canonical body uses two spaces):

```
raw generation:  '  res = tuple(test_tup)\r\n  return (res)'
old extraction:  'res = tuple(test_tup)\r\n  return (res)'   # SyntaxError
```

The raw generation was already a syntactically valid body. The old extractor
made it invalid. `extract_python_code(..., preserve_leading_indent=True)` was
added and is used by the corrected evaluator. This defect alone can suppress
syntax validity for every body-first model.

### 3.2 Effect of the corrections at 1M code-heavy tokens

After the code-aware tokenizer and a 1M-token code-heavy backbone (Section 8),
the same corrected harness measured, on the first 20 MBPP validation tasks:

| Mode | Syntax validity | pass@1 | first-test pass |
|---|---:|---:|---:|
| fast (depth 1) | **40%** | 0% | 0% |
| fixed (depth 2) | **40%** | 0% | 0% |

Representative valid generations:

```
def find_Min_Sum(num):
    if (n == 0):
        return 0
    return result
```

```
def add_str(test_tup, K):
  res = tuple(test_tup)
  return (res)
```

Both parse as Python. Neither is semantically correct. This separates the two
failure modes: **syntax validity is fixable by the data/tokenizer/harness
changes; semantic correctness is not yet demonstrated.** The first milestone
(>0% syntax) is met at 1M tokens; the 3M-token run is the reported result.

---

## 4. Tokenizer defect

The historical tokenizer is a byte-level BPE trained **only on TinyStories**. Measured on MBPP training code:

| Tokenizer | MBPP chars/token | MBPP train tokens | `def` tokenization |
|---|---:|---:|---|
| Historical TinyStories BPE | **1.58** | 42,164 | `d` + `ef` |
| New code-aware BPE (stories + 8 MB syntax-valid Python) | **2.42** | 27,565 | `def` |

The historical tokenizer is 1.5× less efficient on code and does not represent common Python keywords as single tokens. This reduces the effective number of code examples that fit in 256 tokens and makes identifier copying harder. It is a contributing factor, not the sole cause.

---

## 5. Generation-harness audit

- Prompts are short: mean 42.2 tokens, p95 54, max 60 under the historical tokenizer. Reserving 64 new tokens inside a 256-token context is therefore not the main cause of failure on MBPP validation.
- The historical evaluator did not reserve generation space explicitly; it stopped when `seq_len > 256`. With longer prompts this silently truncates completions. The corrected harness caps the prompt to `context_cap - max_new_tokens` and records `prompt_truncated`.
- Greedy decoding was used. No sampling, no KV cache, and no code-fence-specific post-processing beyond `extract_python_code`.
- Sandboxed execution is genuinely isolated with `bwrap` (tested by 10 security probes). The sandbox is not the cause of the zero score.

---

## 6. Failure taxonomy from the 500k checkpoint

| Category | Evidence | Frequency |
|---|---|---:|
| Invalid/incomplete syntax | `def count_list(tuple):` with no body; `= (n) + 1)` | 100% of 43+43 validation/test tasks |
| Wrong function name | `count_list` / `check_list` / `sum_list` instead of the required name | 100% of tasks where a header was produced |
| Missing entry point | Empty generation or no `def` | 2/5 in the fresh corrected-harness probe |
| Semantic error | Fragments such as `sum_list(tuple, n): = (n) + 1)` | all nonempty generations |
| Repetition/mode collapse | Same four generic names across unrelated problems | all tasks |
| Runtime error | Empty body / undefined name | 20/43 val fast, 16/43 test fast |
| Truncation | 12.9% of synthesis targets forced to EOS | training data |

The model has learned the *surface form* of Python function definitions. It has not learned to bind identifiers from the problem/signature or to emit a valid body.

---

## 7. Diagnosis and experiment ordering

Ranked by evidence:

1. **Harness defect (pass@1):** the required function name is absent from 96% of prompts. Fix first, otherwise pass@1 is uninterpretable.
2. **Data/optimization defect (syntax):** 38.7k synthesis targets and 14.4% coding tokens for an 11.5M model trained from scratch. Fix by increasing code-token share, adding a syntax-valid Python corpus, and training longer.
3. **Target-construction defect:** trunctation + forced EOS in 12.9% of synthesis examples. Fix by dropping over-length examples rather than corrupting them.
4. **Tokenizer defect:** TinyStories-only BPE at 1.58 chars/token on code. Fix with a code-aware tokenizer.
5. **Architecture:** not implicated yet. The model already emits syntactically shaped headers and multi-argument signatures, so the shared recurrent stack can represent code syntax. Architecture changes are deferred until the data/objective fixes are measured. This is a decision, not a claim that architecture is irrelevant.

Hypotheses that remain untested at this stage:

- Whether a code-heavy 3M-token run raises syntax validity above zero.
- Whether the shared 2-layer recurrence can copy identifiers from a signature into a body.
- Whether adaptive depth improves code quality at matched token budgets.
- Whether depth adaptation changes peak VRAM at all (Section 8 of the final report must measure this independently from context/workspace changes).

---

## 8. Exact commands used

```bash
# Reproduce the historical baseline (read-only; existing checkpoint)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. .venv-trm-lm/bin/python - <<'PY'
# load experiments/trm_lm_coding_reasoning/pilot_500k/checkpoint_final.pt
# run evaluators.re_eval.evaluate_mbpp on MBPP validation
PY

# Function-name coverage
PYTHONPATH=. .venv-trm-lm/bin/python - <<'PY'
# count canonical def names appearing in MBPP problem text
PY

# Code-aware tokenizer
PYTHONPATH=. .venv-trm-lm/bin/python -m dataset.build_code_tokenizer \
  --workdir experiments/re_coding_adaptive/tokenizer \
  --tinystories-workdir experiments/trm_lm_pilot_5m/r1 \
  --vocab-size 8192 --story-docs 4000 --code-bytes 8000000 \
  --max-code-files 4000 --force

# Code-heavy backbone (3M target tokens)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. .venv-trm-lm/bin/python train_re_coding.py \
  --stage backbone --workdir experiments/re_coding_adaptive/backbone_3m \
  --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
  --max-train-tokens 3000000 --fixed-depth 2 \
  --batch-size 4 --seq-len 256 --lr 3e-4 --warmup-steps 200 \
  --save-every-tokens 500000 --val-every-tokens 500000 --skip-code-eval
```

The historical `experiments/trm_lm_pilot_5m/` and
`experiments/trm_lm_coding_reasoning/pilot_500k/` artifacts were not modified.
