"""Non-invasive verification of recurrent latent refinement in RE:Inference.

This module does not modify the model or the normal inference path. It calls the
existing ``DualStateModel`` internals (``_init_states``, ``_high_cycle``,
``_logits``, ``controller``) under ``torch.no_grad()`` and compares the resulting
normal path against ``model.forward`` for parity.

Interventions:
  A. normal recurrence: z^{k+1} = f(z^k)
  B. broken feedback: reset z_H and/or z_L to their initial values at the start
     of every high cycle. ``both`` removes the feedback route entirely.
  C. repeated-compute control: K independent one-step evaluations. The
     deterministic version (C0) is mathematically identical to A_1; the noisy
     ensemble (C1) uses seeded initial-state perturbations and averages logits.
  D. depth ablations: normal recurrence at 1, 2, and 4 high cycles.

All outputs are written under the caller's output directory. No gradients are
computed, no optimizer exists, and the checkpoint is never written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset.code_pipeline import DELIMITER_SOLUTION, build_mbpp_eval_records, CodePipelineConfig
from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    next_token_cross_entropy,
)
from evaluators.coding_eval import (
    assemble_body_completion,
    check_syntax_validity,
    extract_python_code,
)
from evaluators.execution_sandbox import execute_sandboxed
from evaluators.re_eval import build_signature_prompt
from models.recursive_reasoning.trm_lm_adaptive import STOP_REASON_NAMES, DualStateModel


# ---------------------------------------------------------------------------
# Basic state statistics
# ---------------------------------------------------------------------------


def state_stats(z: torch.Tensor) -> Dict[str, float]:
    """Summary statistics of one latent state [B, T, d]."""
    zf = z.detach().float()
    norms = zf.norm(dim=-1)
    return {
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()) if norms.numel() > 1 else 0.0,
        "abs_mean": float(zf.abs().mean().item()),
    }


def pair_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Relative change and cosine similarity between matched states."""
    af = a.detach().float()
    bf = b.detach().float()
    diff = (bf - af).norm(dim=-1)
    denom = af.norm(dim=-1).clamp_min(1e-8)
    cos = F.cosine_similarity(af, bf, dim=-1)
    return {
        "rel_change_mean": float((diff / denom).mean().item()),
        "rel_change_max": float((diff / denom).max().item()),
        "cosine_mean": float(cos.mean().item()),
        "cosine_min": float(cos.min().item()),
    }


def logit_difference(clean: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    """Compare two logit tensors [B, T, V] or [B, 1, V]."""
    a = clean.detach().float()
    b = other.detach().float()
    if a.shape != b.shape:
        raise ValueError(f"logit shapes differ: {a.shape} vs {b.shape}")
    diff = (a - b).abs()
    logp_a = F.log_softmax(a, dim=-1)
    logp_b = F.log_softmax(b, dim=-1)
    # symmetric KL for a stable scalar
    kl = 0.5 * (
        (logp_a.exp() * (logp_a - logp_b)).sum(-1)
        + (logp_b.exp() * (logp_b - logp_a)).sum(-1)
    )
    top_a = a.argmax(dim=-1)
    top_b = b.argmax(dim=-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "l2": float((a - b).pow(2).mean().sqrt().item()),
        "sym_kl_mean": float(kl.mean().item()),
        "sym_kl_max": float(kl.max().item()),
        "top1_change_rate": float((top_a != top_b).float().mean().item()),
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Recurrence runner
# ---------------------------------------------------------------------------


def _make_perturbation(
    z: torch.Tensor,
    scale: float,
    seed: int,
    direction: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a perturbed state and the realized relative perturbation size."""
    if scale == 0.0:
        return z, {"scale_requested": 0.0, "realized_rel_mean": 0.0, "realized_rel_max": 0.0}
    zf = z.detach().float()
    norms = zf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if direction == "state":
        v = zf / norms
    else:
        generator = torch.Generator(device=z.device)
        generator.manual_seed(int(seed))
        v = torch.randn(zf.shape, generator=generator, device=z.device, dtype=torch.float32)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    delta = (float(scale) * norms) * v
    perturbed = (zf + delta).to(dtype=z.dtype)
    realized = (perturbed.float() - zf).norm(dim=-1) / norms.squeeze(-1)
    return perturbed, {
        "scale_requested": float(scale),
        "realized_rel_mean": float(realized.mean().item()),
        "realized_rel_max": float(realized.max().item()),
    }


@torch.no_grad()
def run_recurrence(
    model: DualStateModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    n_high_cycles: int = 4,
    reset_mode: str = "none",
    last_token_only: bool = True,
    record_states: bool = False,
    record_controller: bool = False,
    perturb_after_depth: Optional[int] = None,
    perturb_target: str = "both",
    perturb_scale: float = 0.0,
    perturb_seed: int = 0,
    perturb_direction: str = "random",
    return_state_tensors: bool = False,
) -> Dict[str, Any]:
    """Run the actual dual-state recurrence with explicit interventions.

    ``reset_mode``:
      - ``none``: normal recurrence
      - ``both``: reset z_H and z_L to their initial values at every cycle
      - ``high``: reset z_H only
      - ``low``: reset z_L only

    ``perturb_after_depth`` applies a perturbation to the output state of that
    depth, before the next cycle. The forward tensor operations are exactly
    ``model._init_states`` + ``model._high_cycle`` + ``model._logits``.
    """
    if reset_mode not in {"none", "both", "high", "low"}:
        raise ValueError("reset_mode must be none, both, high, or low")
    if perturb_target not in {"both", "z_H", "z_L"}:
        raise ValueError("perturb_target must be both, z_H, or z_L")
    if n_high_cycles < 1 or n_high_cycles > model.config.max_high_cycles:
        raise ValueError(
            f"n_high_cycles must be in [1, {model.config.max_high_cycles}]"
        )
    attention_mask = model._validate_inputs(input_ids, attention_mask)
    embeddings, z_H, z_L = model._init_states(input_ids, attention_mask)
    z_H0 = z_H.detach().clone()
    z_L0 = z_L.detach().clone()
    cos_sin = model.rotary_emb()
    counter = [0]
    perturbation_info: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = []
    state_tensors: List[Dict[str, Any]] = []

    def stats_for(zH, zL, prevH, prevL):
        out = {
            "z_H": state_stats(zH),
            "z_L": state_stats(zL),
            "z_H_change": pair_stats(prevH, zH),
            "z_L_change": pair_stats(prevL, zL),
            "cross_cosine_mean": float(
                F.cosine_similarity(zH.detach().float(), zL.detach().float(), dim=-1).mean().item()
            ),
        }
        return out

    initial_stats = {
        "z_H": state_stats(z_H),
        "z_L": state_stats(z_L),
        "cross_cosine_mean": float(
            F.cosine_similarity(z_H.detach().float(), z_L.detach().float(), dim=-1).mean().item()
        ),
    }
    prev_H = z_H
    prev_L = z_L
    final_logits = None
    for step in range(1, n_high_cycles + 1):
        if reset_mode in {"both", "high"}:
            z_H = z_H0.clone()
        if reset_mode in {"both", "low"}:
            z_L = z_L0.clone()
        step_prev_H = z_H
        step_prev_L = z_L
        z_H, z_L = model._high_cycle(
            z_H,
            z_L,
            embeddings,
            cos_sin,
            attention_mask,
            n_low_cycles=model.config.n_low_cycles,
            counter=counter,
            use_stable_update=(
                step >= model.config.stable_update_from_depth
            ),
        )
        if perturb_after_depth is not None and step == perturb_after_depth:
            if perturb_target in {"both", "z_H"}:
                z_H, info_h = _make_perturbation(
                    z_H, perturb_scale, perturb_seed, perturb_direction
                )
            else:
                info_h = {"scale_requested": perturb_scale, "realized_rel_mean": 0.0, "realized_rel_max": 0.0}
            if perturb_target in {"both", "z_L"}:
                z_L, info_l = _make_perturbation(
                    z_L, perturb_scale, perturb_seed + 1, perturb_direction
                )
            else:
                info_l = {"scale_requested": perturb_scale, "realized_rel_mean": 0.0, "realized_rel_max": 0.0}
            perturbation_info = {
                "after_depth": int(step),
                "target": perturb_target,
                "direction": perturb_direction,
                "seed": int(perturb_seed),
                "z_H": info_h,
                "z_L": info_l,
            }
        logits = model._logits(z_H, last_token_only=last_token_only)
        final_logits = logits
        if return_state_tensors:
            state_tensors.append(
                {
                    "step": step,
                    "kind": "state",
                    "z_H": z_H.detach().cpu(),
                    "z_L": z_L.detach().cpu(),
                    "logits": logits.detach().cpu(),
                }
            )
        if record_states or record_controller:
            step_record: Dict[str, Any] = {"step": step}
            if record_states:
                step_record.update(stats_for(z_H, z_L, step_prev_H, step_prev_L))
                step_record["relative_to_initial"] = {
                    "z_H": pair_stats(z_H0, z_H),
                    "z_L": pair_stats(z_L0, z_L),
                }
            if record_controller:
                if step < n_high_cycles:
                    q_logits = model.controller(
                        z_H,
                        z_L,
                        attention_mask,
                        depth=step,
                        max_depth=n_high_cycles,
                        remaining_budget_fraction=None,
                    )
                    q = torch.sigmoid(q_logits)
                    step_record["controller"] = {
                        "logit_mean": float(q_logits.mean().item()),
                        "logit_std": float(q_logits.std().item()) if q_logits.numel() > 1 else 0.0,
                        "prob_mean": float(q.mean().item()),
                        "prob_min": float(q.min().item()),
                        "prob_max": float(q.max().item()),
                        "prob_above_half": float((q >= 0.5).float().mean().item()),
                    }
                    if return_state_tensors:
                        state_tensors.append(
                            {
                                "step": step,
                                "kind": "controller",
                                "logits": q_logits.detach().cpu(),
                                "prob": q.detach().cpu(),
                            }
                        )
                else:
                    step_record["controller"] = {"invoked": False, "reason": "last_depth"}
            step_record["logits"] = {
                "norm": float(logits.detach().float().norm().item()),
                "top1_last": logits[:, -1, :].argmax(dim=-1).detach().cpu().tolist(),
            }
            trace.append(step_record)
    return {
        "final_logits": final_logits,
        "z_H": z_H,
        "z_L": z_L,
        "trace": trace,
        "state_tensors": state_tensors,
        "initial": initial_stats,
        "perturbation": perturbation_info,
        "block_calls": counter[0],
        "n_high_cycles": n_high_cycles,
        "reset_mode": reset_mode,
    }


@torch.no_grad()
def run_independent_ensemble(
    model: DualStateModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    n_passes: int = 4,
    noise_scale: float = 0.0,
    seed: int = 0,
    last_token_only: bool = True,
) -> Dict[str, Any]:
    """Matched-compute control: K independent one-step evaluations, mean logits."""
    attention_mask = model._validate_inputs(input_ids, attention_mask)
    embeddings, z_H0, z_L0 = model._init_states(input_ids, attention_mask)
    cos_sin = model.rotary_emb()
    logits_sum = None
    per_pass = []
    for i in range(n_passes):
        z_H = z_H0.clone()
        z_L = z_L0.clone()
        if noise_scale > 0:
            z_H, info_h = _make_perturbation(z_H, noise_scale, seed + 2 * i, "random")
            z_L, info_l = _make_perturbation(z_L, noise_scale, seed + 2 * i + 1, "random")
        else:
            info_h = info_l = {"realized_rel_mean": 0.0}
        z_H, z_L = model._high_cycle(
            z_H,
            z_L,
            embeddings,
            cos_sin,
            attention_mask,
            n_low_cycles=model.config.n_low_cycles,
            use_stable_update=(
                1 >= model.config.stable_update_from_depth
            ),
        )
        logits = model._logits(z_H, last_token_only=last_token_only)
        logits_sum = logits.float() if logits_sum is None else logits_sum + logits.float()
        per_pass.append(
            {
                "pass": i,
                "noise_scale": float(noise_scale),
                "z_H_realized_rel": info_h.get("realized_rel_mean", 0.0),
                "z_L_realized_rel": info_l.get("realized_rel_mean", 0.0),
                "logit_norm": float(logits.detach().float().norm().item()),
            }
        )
    mean_logits = (logits_sum / n_passes).to(dtype=z_H0.dtype)
    return {
        "final_logits": mean_logits,
        "per_pass": per_pass,
        "n_passes": n_passes,
        "noise_scale": noise_scale,
    }


# ---------------------------------------------------------------------------
# Generation with recurrence interventions
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_with_recurrence(
    model: DualStateModel,
    tokenizer: Any,
    prompt: str,
    n_high_cycles: int = 4,
    reset_mode: str = "none",
    max_new_tokens: int = 96,
    context_cap: int = 256,
    eos_token_id: Optional[int] = None,
    ensemble_passes: int = 0,
    ensemble_noise: float = 0.0,
    ensemble_seed: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    sampling_seed: int = 0,
) -> Dict[str, Any]:
    """Generation whose forward path is the explicit recurrence runner.

    ``temperature == 0`` is deterministic greedy argmax (the frozen primary
    protocol). ``temperature > 0`` samples with an optional nucleus filter;
    sampling is only used in the separately labeled decoding-control analysis.
    """
    device = next(model.parameters()).device
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>") if eos_token_id is None else eos_token_id
    context_cap = min(int(context_cap), int(model.config.seq_len))
    max_new_tokens = min(int(max_new_tokens), max(1, context_cap - 8))
    prompt_ids = [bos] + tokenizer.encode(prompt).ids
    max_prompt = max(8, context_cap - max_new_tokens)
    truncated = len(prompt_ids) > max_prompt
    if truncated:
        prompt_ids = [bos] + prompt_ids[-(max_prompt - 1):]
    sampling_generator = None
    if temperature > 0:
        sampling_generator = torch.Generator(device=device)
        sampling_generator.manual_seed(int(sampling_seed))
    generated: List[int] = []
    per_step = []
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(prompt_ids) + len(generated) >= model.config.seq_len:
                break
            ids = prompt_ids + generated
            inp = torch.tensor([ids], dtype=torch.int64, device=device)
            mask = torch.ones_like(inp)
            if ensemble_passes > 0:
                out = run_independent_ensemble(
                    model,
                    inp,
                    mask,
                    n_passes=ensemble_passes,
                    noise_scale=ensemble_noise,
                    seed=ensemble_seed,
                    last_token_only=True,
                )
            else:
                out = run_recurrence(
                    model,
                    inp,
                    mask,
                    n_high_cycles=n_high_cycles,
                    reset_mode=reset_mode,
                    last_token_only=True,
                    record_states=False,
                )
            logits = out["final_logits"][:, -1, :].float()
            if temperature > 0:
                logits = logits / max(1e-5, float(temperature))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    remove = cumulative > float(top_p)
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    probs = F.softmax(sorted_logits, dim=-1)
                    choice = torch.multinomial(
                        probs, 1, generator=sampling_generator
                    )
                    next_token = int(sorted_indices.gather(-1, choice).item())
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = int(
                        torch.multinomial(
                            probs, 1, generator=sampling_generator
                        ).item()
                    )
            else:
                next_token = int(logits.argmax(dim=-1).item())
            per_step.append(
                {
                    "token": next_token,
                    "logit_norm": float(out["final_logits"].detach().float().norm().item()),
                }
            )
            generated.append(next_token)
            if eos is not None and next_token == eos:
                break
    latency_ms = (time.perf_counter() - start) * 1000.0
    if generated and eos is not None and generated[-1] == eos:
        decode_ids = generated[:-1]
    else:
        decode_ids = generated
    text = tokenizer.decode(decode_ids)
    prompt_seen = tokenizer.decode(prompt_ids[1:]) if truncated else prompt
    return {
        "text": text,
        "token_ids": generated,
        "prompt_ids": prompt_ids,
        "prompt_truncated": truncated,
        "prompt_seen": prompt_seen,
        "per_step": per_step,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Code assembly and scoring (same protocol as re_eval)
# ---------------------------------------------------------------------------


def assemble_generated_code(prompt_seen: str, expected_name: str, raw_generation: str) -> str:
    """Assemble prompt context and generation with indentation preserved.

    Delegates to the shared helper so the recurrence evaluator and the general
    MBPP evaluator cannot drift. See
    :func:`evaluators.coding_eval.assemble_body_completion`.
    """
    return assemble_body_completion(prompt_seen, expected_name, raw_generation)


def score_generated_code(
    prompt_seen: str,
    expected_name: str,
    raw_generation: str,
    record: Dict[str, Any],
    sandbox_python: str,
    timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    assembled = assemble_generated_code(prompt_seen, expected_name, raw_generation)
    syntax = check_syntax_validity(assembled)
    tests = list(record.get("test_list", []))
    effective_timeout = float(record.get("timeout_sec", timeout_sec))
    first_passed = False
    passed = False
    error_type: Optional[str] = None
    error_message = ""
    if not syntax:
        error_type = "syntax_error"
        error_message = "generated program failed ast.parse"
    elif not tests:
        # No assertions means no executable evidence. Do not count a program as
        # passing a task for which the benchmark provides no test.
        error_type = "no_tests"
        error_message = "task has no test cases"
    else:
        # If the first assertion fails, the full suite cannot pass. Running the
        # first test alone is therefore exact for pass@1 and much cheaper than
        # executing every broken program against the whole suite.
        first = execute_sandboxed(
            assembled,
            [tests[0]],
            record.get("test_setup_code", ""),
            timeout_sec=effective_timeout,
            python_executable=sandbox_python,
        )
        first_passed = first.passed
        if first_passed:
            full = execute_sandboxed(
                assembled,
                tests,
                record.get("test_setup_code", ""),
                timeout_sec=effective_timeout,
                python_executable=sandbox_python,
            )
            passed = full.passed
            if not passed:
                error_type = full.error_type
                error_message = full.error_message
        else:
            error_type = first.error_type
            error_message = first.error_message
    return {
        "assembled_code": assembled,
        "syntax_valid": syntax,
        "passed": bool(passed),
        "first_test_passed": first_passed,
        "error_type": error_type,
        "error_message": error_message,
    }


@dataclass
class InterventionConfig:
    name: str
    n_high_cycles: int
    reset_mode: str = "none"
    description: str = ""
    ensemble_passes: int = 0
    ensemble_noise: float = 0.0


def aggregate_intervention_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"total_tasks": 0}
    errors: Dict[str, int] = {}
    for r in rows:
        key = r.get("error_type") or "passed"
        errors[key] = errors.get(key, 0) + 1
    return {
        "total_tasks": n,
        "syntax_valid_rate": sum(bool(r["syntax_valid"]) for r in rows) / n,
        "pass_at_1": sum(bool(r["passed"]) for r in rows) / n,
        "first_test_pass_rate": sum(bool(r["first_test_passed"]) for r in rows) / n,
        "nonempty_rate": sum(bool(r["nonempty"]) for r in rows) / n,
        "entry_point_present_rate": sum(bool(r["entry_point_present"]) for r in rows) / n,
        "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        "total_transitions": int(sum(r["transitions"] for r in rows)),
        "error_counts": errors,
    }


@torch.no_grad()
def evaluate_mbpp_intervention(
    model: DualStateModel,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    config: InterventionConfig,
    max_new_tokens: int = 96,
    context_cap: int = 256,
    limit: Optional[int] = None,
    sandbox_python: str = "/usr/bin/python3.14",
    timeout_sec: float = 3.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if limit is not None:
        records = list(records)[:limit]
    rows: List[Dict[str, Any]] = []
    per_high = model.config.n_low_cycles + 1
    for record in records:
        prompt, name, _ = build_signature_prompt(record)
        generation = generate_with_recurrence(
            model,
            tokenizer,
            prompt,
            n_high_cycles=config.n_high_cycles,
            reset_mode=config.reset_mode,
            max_new_tokens=max_new_tokens,
            context_cap=context_cap,
            ensemble_passes=config.ensemble_passes,
            ensemble_noise=config.ensemble_noise,
            ensemble_seed=1337 + int(record.get("task_id", 0)),
        )
        score = score_generated_code(
            generation["prompt_seen"], name, generation["text"], record, sandbox_python,
            timeout_sec=timeout_sec,
        )
        rows.append(
            {
                "task_id": int(record.get("task_id", -1)),
                "config": config.name,
                "n_high_cycles": config.n_high_cycles,
                "reset_mode": config.reset_mode,
                "ensemble_passes": config.ensemble_passes,
                "ensemble_noise": config.ensemble_noise,
                "nonempty": bool(generation["text"].strip()),
                "entry_point_present": bool(
                    re.search(
                        rf"^\s*(async\s+)?def\s+{re.escape(name)}\s*\(",
                        score["assembled_code"],
                        re.MULTILINE,
                    )
                ),
                "syntax_valid": score["syntax_valid"],
                "passed": score["passed"],
                "first_test_passed": score["first_test_passed"],
                "error_type": score["error_type"],
                "error_message": score["error_message"],
                "prompt_tokens": len(generation["prompt_ids"]),
                "prompt_truncated": bool(generation["prompt_truncated"]),
                "generated_tokens": len(generation["token_ids"]),
                "latency_ms": generation["latency_ms"],
                "transitions": len(generation["token_ids"]) * config.n_high_cycles * per_high,
                "raw_generation": generation["text"],
                "assembled_code": score["assembled_code"],
            }
        )
    return aggregate_intervention_rows(rows), rows


def paired_status(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    field: str,
) -> Dict[str, int]:
    a = {int(r["task_id"]): bool(r[field]) for r in rows_a}
    b = {int(r["task_id"]): bool(r[field]) for r in rows_b}
    ids = sorted(set(a) & set(b))
    both = sum(a[i] and b[i] for i in ids)
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(not a[i] and b[i] for i in ids)
    neither = sum(not a[i] and not b[i] for i in ids)
    return {
        "field": field,
        "paired_tasks": len(ids),
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "delta_b_minus_a": only_b - only_a,
    }


# ---------------------------------------------------------------------------
# TinyStories CE at matched target positions
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_tinystories_depths(
    model: DualStateModel,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    configs: Sequence[InterventionConfig],
) -> Dict[str, Any]:
    model.eval()
    results: Dict[str, Any] = {}
    per_batch: Dict[str, List[Dict[str, float]]] = {}
    for config in configs:
        total_loss = 0.0
        total_tokens = 0
        batches = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)
            n = int((labels != IGNORE_INDEX).sum().item())
            if n == 0:
                continue
            out = run_recurrence(
                model,
                input_ids,
                mask,
                n_high_cycles=config.n_high_cycles,
                reset_mode=config.reset_mode,
                last_token_only=False,
                record_states=False,
            )
            loss = float(next_token_cross_entropy(out["final_logits"], labels).item())
            total_loss += loss * n
            total_tokens += n
            batches.append({"ce": loss, "tokens": n})
        ce = total_loss / max(1, total_tokens)
        results[config.name] = {
            "ce": ce,
            "ppl": math.exp(min(20.0, ce)),
            "valid_target_tokens": total_tokens,
            "batches": len(batches),
            "config": {
                "n_high_cycles": config.n_high_cycles,
                "reset_mode": config.reset_mode,
                "ensemble_passes": config.ensemble_passes,
                "ensemble_noise": config.ensemble_noise,
            },
            "per_batch_ce": batches,
        }
        per_batch[config.name] = batches
    # Paired bootstrap between A_1 and A_4 (batch-weighted means).
    paired = {}
    if "A1" in per_batch and "A4" in per_batch:
        a = per_batch["A1"]
        b = per_batch["A4"]
        if len(a) == len(b) and a:
            diffs = []
            for x, y in zip(a, b):
                # CE difference on the same batch; positive means depth 1 worse
                diffs.append(x["ce"] - y["ce"])
            diffs = np.asarray(diffs, dtype=float)
            rng = np.random.default_rng(1337)
            boot = []
            for _ in range(2000):
                idx = rng.integers(0, len(diffs), len(diffs))
                boot.append(float(diffs[idx].mean()))
            paired["A1_minus_A4_ce"] = {
                "mean": float(diffs.mean()),
                "median": float(np.median(diffs)),
                "bootstrap_95_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "batches": len(diffs),
                "direction": "positive means depth 1 has higher CE (depth 4 better)",
            }
    results["paired_bootstrap"] = paired
    return results


def paired_bootstrap_ce(
    per_batch_a: Sequence[Dict[str, float]],
    per_batch_b: Sequence[Dict[str, float]],
    seed: int = 1337,
    n_boot: int = 2000,
) -> Dict[str, float]:
    """Paired batch bootstrap of CE(a) - CE(b); positive means b is better."""
    if not per_batch_a or len(per_batch_a) != len(per_batch_b):
        return {}
    diffs = np.asarray(
        [a["ce"] - b["ce"] for a, b in zip(per_batch_a, per_batch_b)], dtype=float
    )
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot.append(float(diffs[idx].mean()))
    return {
        "mean": float(diffs.mean()),
        "median": float(np.median(diffs)),
        "bootstrap_95_ci": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "batches": len(diffs),
        "direction": "positive means first config has higher CE (second is better)",
    }
