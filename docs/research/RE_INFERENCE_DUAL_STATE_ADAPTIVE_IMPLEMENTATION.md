# RE:Inference — Dual-State Recursive Language Model with Adaptive Thinking

**Implementation and bounded verification report**
**Date:** 2026-10-21
**Status:** Implementation complete; **no substantive training was performed.**
**Repository:** `/home/chandrabhan/Documents/Chandrabhan/TinyRecursiveModels`
**Git:** `main` @ `2cffb3c232b9e8bc79c7613a117566e834908bb4` (local, ahead 1; origin = `NotBhan/TinyRecursiveModels`, upstream = Samsung).
**Reference:** original Samsung TRM is frozen; 16/16 reference hashes unchanged.

---

## 1. Research objective and hypotheses

The implementation enables, but does not answer, these questions:

- **RQ1 — Hierarchical recursion:** Does a dual-state (`z_H`, `z_L`) recursive model outperform the existing single-state recurrence under controlled compute?
- **RQ2 — Recurrent state:** Does carrying and refining latent state help beyond repeatedly applying shared layers to input embeddings?
- **RQ3 — Adaptive computation:** Can a learned controller stop refinement without unacceptable LM quality?
- **RQ4 — Compute efficiency:** Does adaptive inference improve the quality-versus-actual-compute frontier relative to fixed-depth inference?
- **RQ5 — Generalization:** Do gains persist across seeds, held-out documents, and less-template-like prompts?

**No empirical claim is made for any RQ.** The only training performed was a one-step smoke test; the controller has not been meaningfully trained or evaluated.

---

## 2. Original TRM source locations and frozen reference

Frozen reference files (byte-identical; see `tests/reference_hashes.json`):

- `models/recursive_reasoning/trm.py` — `TinyRecursiveReasoningModel_ACTV1_Inner.forward` (dual-state recursion, input injection, shared `L_level`, partial no-grad schedule) and `TinyRecursiveReasoningModel_ACTV1.forward` (ACT halting).
- `models/losses.py` — `ACTLossHead` and the puzzle q-halt loss.
- `config/arch/trm.yaml` — original `H_cycles`/`L_cycles`/`halt_max_steps` configuration.
- The other 13 reference files listed in `tests/reference_hashes.json`.

Reference commit: `c01103738605ba39d1430519b1ee0c62f4c707f8` (SamsungSAILMontreal upstream). The reference implementation was not modified. The historical pilot under `experiments/trm_lm_pilot_5m/` and its report were not modified.

---

## 3. Architecture and state-transition semantics

Module: `models/recursive_reasoning/trm_lm_adaptive.py`
Class: `DualStateModel`
Config: `DualStateConfig` (subclass of the existing `TRM_LMConfig` for block compatibility).

### 3.1 Dimensions and causal/padding convention

- `input_ids`: `[B, T]`; `attention_mask`: `[B, T]` with 1 for real tokens, 0 for padding.
- Embeddings: `E = Embed(X)`, `[B, T, d]`.
- `z_H`, `z_L`: `[B, T, d]`.
- Logits: `[B, T, V]`.
- All attention uses the existing `TRM_LMBlock` / `TRM_LMCausalSelfAttention`:
  an explicit boolean mask `(j <= i) AND key_is_real[j]`; future tokens and padding keys are excluded. No other attention path exists.

### 3.2 Dual-state initialization

Separate learned projections:

```
z_H = P_H(E)
z_L = P_L(E)
```

`P_H` and `P_L` are bias-free `d × d` linear projections, initialized with truncated normal `std = 1/sqrt(d)`. They add `2d²` parameters, reported separately in every parameter count. No future tokens are introduced beyond the normal causal token representation.

### 3.3 Low-level update

For each high cycle, `n_low_cycles` inner updates:

```
h = z_L + z_H + E
h = F_L(h, causal/padding mask)      # shared low-level block stack
z_L = RMSNorm(h)
```

`F_L` is a `ModuleList` of one or more shared `TRM_LMBlock`s, reused at every low update. The final `RMSNorm` is the transition norm; the block also has its internal post-norms. This is a deliberate language-model adaptation, not the exact original TRM equation.

### 3.4 High-level update

After the inner loop:

```
h = z_H + z_L + E
h = F_H(h, causal/padding mask)      # shared high-level block stack
z_H = RMSNorm(h)
```

`F_H` is either a separate shared block stack (default) or, with `share_high_low_blocks=True`, the same `F_L` stack (closer to the reference TRM, which uses one `L_level` for both states). The high update preserves `z_H` through the residual sum; neither state is silently reset between cycles.

### 3.5 Output

```
logits = LMHead(RMSNorm(z_H))
```

The output depends on `z_H`, which depends on `z_L` and `E`; tests verify that zeroing either `P_H` or `P_L` changes the logits.

### 3.6 Counts and transitions

- One **high cycle** = `n_low_cycles` low updates + 1 high update.
- `transitions_per_high_cycle = n_low_cycles + 1`.
- Total executed transitions = high cycles × (`n_low_cycles` + 1), reported per sequence.
- `n_high_cycles` is the fixed depth; `min_high_cycles`/`max_high_cycles` bound adaptive and fixed inference.
- `n_low_cycles` and `n_high_cycles` are independent.

---

## 4. Original TRM vs. pilot `trm_lm` vs. dual-state model

| Property | Original TRM | Pilot `trm_lm` | New dual-state model |
|---|---|---|---|
| Latent states | `z_H`, `z_L` | single `z` | `z_H`, `z_L` |
| Initialization | learned `H_init`/`L_init` buffers | token embeddings | separate `P_H(E)`, `P_L(E)` |
| Low-level recursion | `L_cycles` over shared `L_level` | `n_refine` shared block passes | `n_low_cycles` over shared `F_L` |
| High-level update | `H_cycles`, uses same `L_level` | none | `F_H` (separate or shared) |
| Parameter sharing | one shared `L_level` | shared block stack | shared `F_L`; shared `F_H` |
| Gradient schedule | `H_cycles-1` no-grad; final H cycle grad | all passes differentiable | configurable: full / truncated / last high cycle |
| ACT / halting | q-head, exact-answer target | none | per-sequence controller, LM marginal-utility target |
| Task | structured puzzles | causal LM | causal LM |
| Claim | reference | single-latent adaptation | TRM-inspired dual-state adaptation |

This new model is **not** “identical to original TRM.” It is a TRM-inspired causal language-model adaptation. The original reference remains untouched and is not used by the new module.

---

## 5. Parameter-sharing semantics

- `F_L` is instantiated once per layer and reused at every low update.
- `F_H` is instantiated once per layer and reused at every high update.
- `share_high_low_blocks=True` makes `F_H` reuse `F_L`; this reduces parameter count and approximates the original single-stack behavior.
- Recursion counts are runtime/config values; they do not instantiate new blocks. Parameter count is constant as recursion depth changes.
- Tests verify the same modules are called repeatedly (`F_L` calls = high cycles × low cycles; `F_H` calls = high cycles) and that parameter count is unchanged across `max_high_cycles=2` vs `4`.

---

## 6. Gradient-flow semantics

Three explicitly named strategies (`gradient_strategy`):

| Strategy | Transitions receiving gradients | Notes |
|---|---|---|
| `full` | All executed high cycles and their low updates | Default; full BPTT through the recursion |
| `truncated` | The current high cycle; states are detached after each high cycle | Truncated BPTT; `P_H`/`P_L` receive no gradient when the LM loss is taken at a later depth |
| `last_high_cycle` | Only the final high cycle and its low updates | Earlier cycles run under `torch.no_grad()`; intermediate LM losses are not used (training entrypoint requires `depth_sampling=max`) |

Additional semantics:

- `P_H`/`P_L` projections receive gradients under `full`; under `truncated`/`last_high_cycle` their outputs are detached before the loss-bearing cycle, so they have no gradient from the LM loss at a later depth.
- Controller features (pooled `z_H`/`z_L`) are detached by default (`controller_detach_features=True`). The controller parameters receive gradients from its BCE/compute objective; the backbone is not updated by the controller loss.
- Tests verify finite gradients to `P_H`/`P_L`, `F_L`/`F_H`, embeddings, and LM head under `full`, and the documented partial-gradient behavior.
- Activation checkpointing is **not** implemented. At the measured target scale, full BPTT fits in 8 GB (Section 13).

---

## 7. Halting controller design and training objective

### 7.1 Controller

`DualStateController` is a small MLP:

```
features = [masked_pool(z_H), masked_pool(z_L), depth/max_depth, remaining_budget_fraction]
q = sigmoid(MLP(features))
```

- Per-sequence, not per-token.
- Pooling is masked; padding positions cannot influence the controller summary.
- The controller sees the current depth's states only; it does not read labels, target tokens, or deeper/future refinement states.
- `q` is the probability of executing another high cycle.
- `controller_detach_features=True` by default so controller supervision does not update the backbone through the pooled features.

### 7.2 Primary strategy: marginal utility of additional refinement

Training uses ground-truth labels only to construct targets. For each decision depth `d`:

```
delta_b = CE_b(d) - CE_b(d+1)          # per sequence, token-averaged
target_b = 1[delta_b > cost_threshold]
loss_controller = BCEWithLogits(q_d, target_b)
```

This is Strategy 2: the controller learns whether the next refinement improves the sequence's LM loss enough to justify its cost. At inference, no labels are available; only `q` from the current state is used. The threshold is a configurable proxy, not a measured FLOPs cost.

### 7.3 Secondary/design comparison: differentiable expected computation

A differentiable expected-depth penalty is available:

```
expected_depth = min_high + Σ_{d=min}^{max-1} Π_{j=min}^{d} q_j
loss += compute_penalty * expected_depth
```

This is a ponder-style/compute-regularized formulation. It is implemented as a configurable penalty (`compute_penalty`) rather than a separate training algorithm. The report distinguishes it from the primary marginal-utility BCE; the two are not silently combined beyond this documented penalty.

### 7.4 Multiple-depth training

`forward_depths` runs up to `max_high_cycles` and returns logits/controller logits at each depth. The training entrypoint supports:

- `depth_sampling=uniform` — sample one LM depth per batch (uniform over `1..max_high_cycles`);
- `depth_sampling=all` — mean LM loss over all depths;
- `depth_sampling=max` — maximum-depth LM loss.

Controller targets are computed at every decision depth from `min_high_cycles` to `max_high_cycles-1`. The sampled depth, distribution, and seed are recorded in metrics. `last_high_cycle` requires `depth_sampling=max` because intermediate LM losses are outside the graph.

**Important:** these training configurations are implemented and smoke-tested only. No meaningful adaptive policy was trained.

---

## 8. Inference modes and stopping semantics

All modes are explicit in `DualStateModel.forward` / `generate`; no environment variables or hidden globals.

| Mode | Behavior | Stopping reason |
|---|---|---|
| `fast` | `min_high_cycles` high cycles; controller never invoked | `minimum_reached` |
| `fixed` | Exactly `n_high_cycles` (within min/max); controller never invoked | `fixed_depth` |
| `adaptive` | Start at min; after each high cycle evaluate `q`; continue if `q >= halt_threshold` and within `max_high_cycles`; stop otherwise | `learned_halt`, `max_reached` |
| `budgeted` | Same controller, plus hard `max_total_transitions`; stop when the next high cycle would exceed the budget | `budget_exhausted`, `learned_halt`, `max_reached` |

Per-sequence halting is implemented with an active mask: halted sequences freeze their state and logits while other sequences continue. **Batched wall-clock compute is not reduced for a mixed batch**, because the high cycle is executed for all rows and then masked. Actual compute savings are realized with batch-1 generation or by grouping sequences by depth. This is documented as a limitation, not a compute-efficiency claim.

Generation (`generate`) reports:

- generated tokens;
- transitions per generation step and total transitions;
- per-step high-cycle counts;
- stopping reasons;
- prompt length and new-token count;
- transitions in the last forward pass (per-forward budget) versus total generation compute.

Greedy decoding is implemented; sampling is not. No KV cache or recurrent-state cache is implemented; the recursive computation is recomputed per generated token. That distinction is explicit in the generation diagnostics.

---

## 9. Configuration

Typed schema: `DualStateConfig` (`models/recursive_reasoning/trm_lm_adaptive.py`) with startup validation for:

- vocabulary, sequence length, hidden size, heads;
- low/high layer counts, low/high cycles, min/fixed/max high cycles;
- controller hidden size, dropout, threshold;
- high/low block sharing;
- gradient strategy;
- optional total-transition budget.

Template: `config/trm_lm_adaptive.yaml` (target-scale candidate, not executed). Example fields:

```yaml
model:
  hidden_size: 512
  num_heads: 8
  n_low_layers: 1
  n_high_layers: 1
  n_low_cycles: 1
  n_high_cycles: 2
  min_high_cycles: 1
  max_high_cycles: 4
  controller_hidden_size: 64
  halt_threshold: 0.5
  share_high_low_blocks: false
  gradient_strategy: full
training:
  mode: adaptive
  depth_sampling: uniform
  controller_loss_weight: 1.0
  compute_penalty: 0.0
  cost_threshold: 0.0
  batch_size: 4
  dtype: bfloat16
  seed: 1337
```

The training entrypoint exposes the same fields as CLI flags. It loads the existing pilot manifest/tokenizer read-only from `experiments/trm_lm_pilot_5m/r1`.

---

## 10. Verification: commands, tests, and results

### 10.1 Test commands

```bash
# New dual-state tests (CPU)
python -m pytest tests/test_trm_lm_adaptive.py -q -p no:warnings

# New dual-state tests incl. opt-in GPU
TRM_LM_ADAPTIVE_TEST_GPU=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  python -m pytest tests/test_trm_lm_adaptive.py -q -p no:warnings -s

# Existing regressions
python -m pytest tests/test_trm_lm.py -q -p no:warnings
TRM_LM_TEST_GPU=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  python -m pytest tests/test_trm_lm.py -q -p no:warnings -s
```

### 10.2 Results

| Suite | Result |
|---|---|
| New adaptive tests, CPU | **26 passed, 1 skipped** (GPU opt-in) |
| New adaptive tests, GPU | **27 passed** |
| Existing `trm_lm` tests, CPU | **36 passed, 3 skipped** |
| Existing `trm_lm` tests, GPU | **39 passed** |
| Reference hashes | **16/16 unchanged** |
| Pilot artifacts | unchanged; read-only in smoke tests |

Coverage includes:

- shapes and both latent states;
- shared-parameter reuse and constant parameter count across recursion depth;
- transition counts and `n_low_cycles` overrides;
- causal future-token invariance and padding invariance;
- controller padding invariance;
- full/truncated/last-high-cycle gradients;
- controller gradients and detached-feature behavior;
- immediate halt, continue-to-max, budget exhaustion, fixed-mode controller independence;
- mixed-batch per-sequence halting;
- deterministic stopping decisions;
- all four generation modes;
- checkpoint round-trip;
- per-sequence CE and token-weighted aggregation;
- opt-in GPU forward/backward.

The mixed-batch test caught a real bug in which a halted sequence was advanced one extra high cycle because the update mask used the pre-decision active set. The update is now masked by the continuation decision; the full suite was rerun.

### 10.3 Bounded smoke runs

Stored under `experiments/trm_lm_adaptive_smoke/` (local; ignored by `.gitignore`), with logs `logs/cpu_full.log`, `logs/gpu_full.log`, `logs/cpu_partial.log`.

Common smoke config: vocab 8000, seq 64, hidden 128, 1 low layer, 1 high layer, 1 low cycle, min 1, max 3, controller hidden 32, 100 target tokens, 1 optimizer step, batch 2.

| Run | Device / grad | Tokens | Params | Train time | Peak GPU | Reload diff | LM loss | Controller loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cpu_full` | CPU / full | 100 | 1,589,377 | 4.064 s | — | 0.0 | 8.9900 | 0.6922 |
| `gpu_full` | RX 6600M / full, bf16 | 100 | 1,589,377 | 0.900 s | 0.111 GB | 0.0 | 8.9902 | 0.6921 |
| `cpu_partial` | CPU / last high cycle | 100 | 1,589,377 | 1.472 s | — | 0.0 | 8.9900 | 0.6922 |
| `cpu_full_after_fix` | CPU / full | 100 | 1,589,377 | 0.310 s | — | 0.0 | 8.9900 | 0.6922 |

CPU timing varies substantially with warm-up/IO (the first full run took 4.064 s; the post-fix rerun took 0.310 s). GPU timing is the more meaningful smoke measurement. The post-fix rerun was performed after the mixed-batch per-sequence halting correction and produced the same loss values.

- The controller received one optimizer step of training only; `controller_accuracy=0.5`, expected depth ≈1.75. This is **not** a trained adaptive policy.
- In the smoke, adaptive and budgeted inference halt at the minimum because the untrained controller outputs `q < 0.5`; adaptive quality equals fast quality. Fixed depth runs 3 high cycles.
- Generated text after one step is gibberish; no language-quality claim is made.

---

## 11. Hardware and dependencies

| Item | Observed |
|---|---|
| OS / kernel | CachyOS, `7.2.8-1-cachyos`, x86_64 |
| CPU | AMD Ryzen 7 5800H, 16 threads |
| RAM | 14 GiB observed, ~5.4 GiB available, 12 GiB swap |
| GPU | AMD Radeon RX 6600M, `gfx1032`, 8.57 GB VRAM |
| Runtime | `HSA_OVERRIDE_GFX_VERSION=10.3.0` (opt-in, unofficial gfx1032→gfx1030 mapping) |
| PyTorch / HIP | 2.14.0 ROCm / 7.2.53211 |
| Python | 3.14.7; no system pip; isolated scratch venv with system site packages |
| Supplemental deps | pyarrow 26.0.0, tokenizers 0.23.3, einops 0.8.2, pytest 9.1.1 |

No system packages, drivers, remotes, or branches were changed. No commit was created. The GPU override is not official RX 6600M support.

---

## 12. Known limitations and deliberately unimplemented items

**Limitations:**

- No substantive training. The adaptive controller is implemented and one-step smoke-tested, not meaningfully trained.
- Per-sequence halting uses a batched active mask; mixed-batch wall-clock compute is not reduced. Batch-1 generation realizes savings.
- No KV cache or recurrent-state cache; recursive states are recomputed for each generated token. Total generation transitions are reported separately from per-forward transitions.
- No per-token halting; per-sequence only.
- Controller targets use a scalar CE-improvement threshold, not measured FLOPs or latency. The threshold is untuned.
- No segment-ID attention mask. The current data pipeline uses one document chunk per sample, so cross-document attention is structurally absent; packing multiple documents would require segment masks.
- `DualStateConfig` inherits unused `num_layers` and `n_refine` fields for block compatibility.
- No activation checkpointing; full BPTT fits the measured target scale but was not tested at longer sequences/larger batches.
- The smoke uses the existing 8k pilot tokenizer/manifest; no new dataset or tokenizer was created.
- No multi-seed, quality-versus-compute, halting-distribution, or generalization experiments were run.

**Deliberately not implemented:** substantive training, new datasets, pretrained weights, distillation, per-token controllers, KV caching, activation checkpointing, external evaluation corpus, and the full experiment matrix.

---

## 13. Expected computational costs and estimates

### 13.1 Target configuration parameters

`vocab=8000`, `seq=256`, `hidden=512`, `heads=8`, 1 low layer, 1 high layer, separate `F_H`, tied head:

**11,501,825 parameters**

| Component | Parameters |
|---|---:|
| Embedding (tied LM head) | 4,096,000 |
| `P_H` + `P_L` | 524,288 |
| `F_L` | 3,407,872 |
| `F_H` | 3,407,872 |
| Controller | 65,793 |
| **Total** | **11,501,825** |

With `share_high_low_blocks=True`, `F_H` is removed, reducing the model by 3,407,872 parameters.

### 13.2 Full-BPTT memory and time

Batch 4, sequence 256, bf16, forward + backward, no optimizer step:

| `n_low_cycles` | `max_high_cycles` | Time | Peak allocated | Reserved |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.153 s | 0.291 GB | 0.342 GB |
| 1 | 2 | 0.245 s | 0.403 GB | 0.476 GB |
| 1 | 4 | 0.476 s | 0.601 GB | 0.703 GB |
| 2 | 1 | 0.214 s | 0.362 GB | 0.484 GB |
| 2 | 2 | 0.408 s | 0.481 GB | 0.545 GB |
| 2 | 4 | 0.708 s | 0.752 GB | 0.839 GB |

All configurations fit comfortably in 8.57 GB VRAM at this scale. Activation checkpointing was not required for these measurements and was not implemented.

### 13.3 Inference estimates

Batch 1, sequence 256, bf16, one forward pass:

| Mode | High cycles | Transitions | Forward latency | Stop reason |
|---|---:|---:|---:|---|
| fast | 1 | 2 | 3.26 ms | minimum_reached |
| fixed | 2 | 4 | 6.33 ms | fixed_depth |
| fixed | 4 | 8 | 10.30 ms | fixed_depth |
| adaptive (forced continue) | 4 | 8 | 14.84 ms | max_reached |
| budgeted (forced continue, budget 4) | 2 | 4 | 8.41 ms | budget_exhausted |

The adaptive/budgeted probe used a forced-continue mock controller to measure the upper bound; actual latency depends on the learned policy and controller overhead (≈4.5 ms for three controller calls in this probe).

Greedy generation of 8 tokens, batch 1, using the same target model: measured 0.063–0.133 s total (≈60–127 tokens/s), with run-to-run variance from a small sample. These are smoke estimates, not benchmark results.

---

## 14. Preregistered experiment arms and metrics

No experiments were launched. The following is the planned comparison:

| Arm | Configuration |
|---|---|
| A | Historical pilot B1/R1/R4 observations; not replicated by the new model |
| B | Conventional transformer baseline, documented parameter/compute budget |
| C | Existing single-state `trm_lm` control |
| D | New dual-state fixed-depth model |
| E | New dual-state adaptive model (same architecture/training as D) |
| F | Compute-matched non-recursive baseline (match measured training/forward compute) |

Additional ablations: no-carry/reinitialized state, unshared-depth (distinct layers), and `share_high_low_blocks` on/off.

Shared protocol: same tokenizer, split manifest, sequence length, target-token budget, evaluation documents, and checkpoint-selection rule. At least three seeds; report seed variation. Metrics: validation CE/PPL, valid target tokens, parameters, executed transitions, throughput, latency, peak memory, actual compute proxy, halting-step distribution, mean/percentile transitions, quality-versus-compute curves.

Do not compare the best checkpoint from one arm with the final checkpoint from another; freeze the checkpoint-selection rule in advance.

---

## 15. Verified behavior vs. untested hypotheses

**Verified implementation behavior:**

- dual-state initialization and updates;
- shared `F_L`/`F_H` parameter reuse;
- causal/padding invariance;
- full/truncated/last-high-cycle gradient semantics;
- controller gradient path;
- fast/fixed/adaptive/budgeted stopping semantics;
- checkpoint save/reload equivalence;
- configuration validation;
- reference integrity and existing regressions.

**Untested hypotheses:**

- RQ1: dual-state recursion outperforms single-state recurrence at controlled compute.
- RQ2: recurrent state provides value beyond extra shared-layer applications.
- RQ3: a learned controller can halt without unacceptable quality loss.
- RQ4: adaptive inference improves the quality/compute frontier.
- RQ5: gains persist across seeds, held-out documents, and less-template-like prompts.

**Bottom line:** the architecture, controller, inference modes, tests, and cost estimates are implemented. The adaptive controller has had one smoke optimizer step, not meaningful training. No claim of recursive benefit, adaptive efficiency, generalization, or intelligence is made.

---

## 16. Changed and new files

**New:**
- `models/recursive_reasoning/trm_lm_adaptive.py` — dual-state model, controller, modes, generation diagnostics.
- `train_trm_lm_adaptive.py` — fixed/adaptive training entrypoint, multi-depth sampling, controller objective, checkpoint validation, smoke generation.
- `tests/test_trm_lm_adaptive.py` — 27 tests (26 CPU + 1 opt-in GPU).
- `config/trm_lm_adaptive.yaml` — experiment configuration template.
- `docs/research/RE_INFERENCE_DUAL_STATE_ADAPTIVE_IMPLEMENTATION.md` — this report.
- `experiments/trm_lm_adaptive_smoke/` — local smoke artifacts and `REPRO_MANIFEST.json` (ignored by `.gitignore`; not committed).

**Modified:** none in this phase. The original Samsung reference, the pilot model, pilot artifacts, tests, and reports remain unchanged.

**Git status:** `main` @ `2cffb3c`, local ahead 1; origin `NotBhan/TinyRecursiveModels`, upstream Samsung. No commit was created for this phase. The new files are untracked.

**No substantive training was performed.** The only optimization steps were one per smoke run (CPU full, GPU full, CPU partial-gradient), 100 target tokens each.
