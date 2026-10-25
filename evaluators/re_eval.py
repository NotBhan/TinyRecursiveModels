"""Coding and adaptive-inference evaluation for RE:Inference.

The evaluation protocol in this module differs from the historical 500k pilot
in three documented ways:

1. The MBPP prompt includes the required function signature (but not its body).
   479/500 MBPP test problems never mention the function name called by the
   hidden tests, so the historical prompt could not yield a pass even for a
   perfect body generator.
2. The generated body is concatenated with the prompt's code context before
   parsing and execution. The context is exactly what the model was given.
3. Generation reserves room for new tokens inside the model context instead of
   silently truncating at ``seq_len``.

Peak GPU memory is measured with ``torch.cuda.max_memory_allocated`` and
``max_memory_reserved`` after synchronization. Allocator-cached memory is
reported separately and is never treated as live working memory.
"""

from __future__ import annotations

import ast
import contextlib
import gc
import json
import math
import os
import re
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset.code_pipeline import (
    DELIMITER_PROBLEM,
    DELIMITER_SOLUTION,
    _target_function_name,
    split_target_function,
)
from dataset.tinystories_lm import IGNORE_INDEX, count_valid_targets, next_token_cross_entropy
from evaluators.coding_eval import (
    assemble_body_completion,
    check_syntax_validity,
    extract_python_code,
)
from evaluators.execution_sandbox import ExecutionResult, execute_sandboxed


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


@dataclass
class MemoryMeasurement:
    device: str
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    weights_mb: float = 0.0
    working_allocated_mb: float = 0.0
    host_peak_rss_mb: float = 0.0
    includes_model_loading: bool = False
    notes: str = ""


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


@contextlib.contextmanager
def peak_memory_context(model: torch.nn.Module, device: torch.device):
    """Measure peak GPU memory inside the context.

    ``empty_cache`` is called before the measurement so ``max_memory_reserved``
    starts from the model's resident allocations rather than unrelated prior
    allocations. This is measurement hygiene, not an inference optimization.
    """
    weights = parameter_bytes(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_alloc = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        result = MemoryMeasurement(device="cuda")
        try:
            yield result
        finally:
            torch.cuda.synchronize(device)
            result.peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1e6
            result.peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1e6
            result.weights_mb = weights / 1e6
            result.working_allocated_mb = max(
                0.0, result.peak_allocated_mb - result.weights_mb
            )
            result.host_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            result.notes = (
                f"alloc_before_mb={before_alloc/1e6:.3f}; "
                f"reserved_before_mb={before_reserved/1e6:.3f}; "
                "peak_allocated is live tensor memory; peak_reserved includes allocator cache"
            )
    else:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        result = MemoryMeasurement(device="cpu")
        try:
            yield result
        finally:
            result.host_peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            result.weights_mb = weights / 1e6
            result.notes = f"host_rss_before_mb={before:.1f}; CPU peak RSS is process-wide"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_signature_prompt(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return ``(prompt, expected_function_name, canonical_body)``.

    The canonical body is returned only for diagnostics and is never placed in
    the prompt.
    """
    name = _target_function_name(record["code"], record.get("test_list", []))
    if name is None:
        raise ValueError(f"no target function for task {record.get('task_id')}")
    split = split_target_function(record["code"], name)
    if split is None:
        raise ValueError(f"cannot split target function for task {record.get('task_id')}")
    context, body = split
    problem_text = record.get("problem_text", record.get("text", ""))
    prompt = (
        f"{DELIMITER_PROBLEM}{str(problem_text).strip()}"
        f"{DELIMITER_SOLUTION}{context}"
    )
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt, name, body


def _entry_point_present(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# Generation and per-task evaluation
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: int
    expected_name: str
    prompt_tokens: int
    prompt_truncated: bool
    generated_tokens: int
    raw_generation: str
    assembled_code: str
    nonempty: bool
    decodes: bool
    entry_point_present: bool
    syntax_valid: bool
    passed: bool
    first_test_passed: bool
    tests_total: int
    tests_passed: int
    error_type: Optional[str]
    error_message: str
    latency_ms: float
    total_transitions: int
    mean_high_cycles: float
    mean_token_entropy: float
    mean_token_margin: float
    mode: str
    requested_depth: Optional[int]
    context_cap: int
    difficulty_score: float = 0.0
    predicted_depth: Optional[int] = None


def _decode_new_tokens(tokenizer: Any, ids: Sequence[int], eos_id: Optional[int]) -> str:
    values = list(ids)
    if values and eos_id is not None and values[-1] == eos_id:
        values = values[:-1]
    try:
        return tokenizer.decode(values)
    except Exception:
        return ""


def generate_for_record(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    mode: str = "fast",
    n_high_cycles: Optional[int] = None,
    max_new_tokens: int = 96,
    context_cap: int = 256,
    max_high_cycles: Optional[int] = None,
    timeout_sec: float = 3.0,
    sandbox_python: str = "/usr/bin/python",
    last_token_only: bool = True,
) -> TaskResult:
    prompt, expected_name, _ = build_signature_prompt(record)
    device = next(model.parameters()).device
    context_cap = min(int(context_cap), int(getattr(model.config, "seq_len", context_cap)))
    max_new_tokens = min(int(max_new_tokens), max(1, context_cap - 8))
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    prompt_ids = [bos] + tokenizer.encode(prompt).ids
    max_prompt = max(8, context_cap - max_new_tokens)
    truncated = len(prompt_ids) > max_prompt
    if truncated:
        # Keep the tail: the signature is the most important part of the prompt.
        prompt_ids = [bos] + prompt_ids[-(max_prompt - 1):]
    inp = torch.tensor([prompt_ids], dtype=torch.int64, device=device)

    t0 = time.perf_counter()
    forward_kwargs: Dict[str, Any] = {}
    if n_high_cycles is not None:
        forward_kwargs["n_high_cycles"] = n_high_cycles
    if max_high_cycles is not None:
        forward_kwargs["max_high_cycles"] = max_high_cycles
    generation = model.generate_with_diagnostics(
        inp,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        mode=mode,
        last_token_only=last_token_only,
        **forward_kwargs,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    new_ids = generation.tokens[0, len(prompt_ids):].detach().cpu().tolist()
    raw_generation = _decode_new_tokens(tokenizer, new_ids, eos)

    # The prompt contained imports/helpers and the exact def line. If the model
    # generated only a body, prepend the signature. If it generated a full
    # definition, prepend only the helpers; concatenating both defs would leave
    # the first one bodyless and make an otherwise valid program unparseable.
    # The shared helper preserves nested-def indentation (a body may start
    # with an indented helper definition).
    prompt_seen = prompt
    if truncated:
        prompt_seen = tokenizer.decode(prompt_ids[1:])
    cleaned = extract_python_code(raw_generation, preserve_leading_indent=True)
    assembled = assemble_body_completion(prompt_seen, expected_name, raw_generation)
    nonempty = bool(raw_generation.strip())
    entry = _entry_point_present(assembled, expected_name) or _entry_point_present(cleaned, expected_name)
    syntax_ok = check_syntax_validity(assembled)
    if syntax_ok:
        exec_result = execute_sandboxed(
            assembled,
            record.get("test_list", []),
            record.get("test_setup_code", ""),
            timeout_sec=timeout_sec,
            python_executable=sandbox_python,
        )
    else:
        exec_result = ExecutionResult(
            passed=False,
            error_type="syntax_error",
            error_message="generated program failed ast.parse",
            stdout="",
            stderr="",
            duration_sec=0.0,
        )
    tests = list(record.get("test_list", []))
    first_test_passed = False
    if tests:
        first = execute_sandboxed(
            assembled,
            [tests[0]],
            record.get("test_setup_code", ""),
            timeout_sec=timeout_sec,
            python_executable=sandbox_python,
        )
        first_test_passed = first.passed
    entropies = generation.token_entropies or [0.0]
    margins = generation.token_margins or [0.0]
    high = [max(v) for v in generation.high_cycles_per_step] if generation.high_cycles_per_step else [0]
    return TaskResult(
        task_id=int(record.get("task_id", -1)),
        expected_name=expected_name,
        prompt_tokens=len(prompt_ids),
        prompt_truncated=truncated,
        generated_tokens=len(new_ids),
        raw_generation=raw_generation,
        assembled_code=assembled,
        nonempty=nonempty,
        decodes=True,
        entry_point_present=entry,
        syntax_valid=syntax_ok,
        passed=exec_result.passed,
        first_test_passed=first_test_passed,
        tests_total=len(tests),
        tests_passed=0 if not exec_result.passed else len(tests),
        error_type=None if exec_result.passed else exec_result.error_type,
        error_message=exec_result.error_message,
        latency_ms=latency_ms,
        total_transitions=int(generation.total_transitions),
        mean_high_cycles=float(np.mean(high)) if high else 0.0,
        mean_token_entropy=float(np.mean(entropies)) if entropies else 0.0,
        mean_token_margin=float(np.mean(margins)) if margins else 0.0,
        mode=mode,
        requested_depth=n_high_cycles,
        context_cap=context_cap,
    )


def aggregate_results(results: Sequence[TaskResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total_tasks": 0}
    error_counts: Dict[str, int] = {}
    for r in results:
        key = r.error_type or "passed"
        error_counts[key] = error_counts.get(key, 0) + 1
    return {
        "total_tasks": n,
        "pass_at_1": sum(r.passed for r in results) / n,
        "first_test_pass_rate": sum(r.first_test_passed for r in results) / n,
        "syntax_valid_rate": sum(r.syntax_valid for r in results) / n,
        "nonempty_rate": sum(r.nonempty for r in results) / n,
        "entry_point_present_rate": sum(r.entry_point_present for r in results) / n,
        "prompt_truncated_rate": sum(r.prompt_truncated for r in results) / n,
        "mean_prompt_tokens": float(np.mean([r.prompt_tokens for r in results])),
        "mean_generated_tokens": float(np.mean([r.generated_tokens for r in results])),
        "mean_latency_ms": float(np.mean([r.latency_ms for r in results])),
        "p50_latency_ms": float(np.percentile([r.latency_ms for r in results], 50)),
        "p95_latency_ms": float(np.percentile([r.latency_ms for r in results], 95)),
        "mean_transitions": float(np.mean([r.total_transitions for r in results])),
        "mean_high_cycles": float(np.mean([r.mean_high_cycles for r in results])),
        "mean_token_entropy": float(np.mean([r.mean_token_entropy for r in results])),
        "mean_token_margin": float(np.mean([r.mean_token_margin for r in results])),
        "error_counts": error_counts,
    }


def evaluate_mbpp(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    mode: str = "fast",
    n_high_cycles: Optional[int] = None,
    max_new_tokens: int = 96,
    context_cap: int = 256,
    max_high_cycles: Optional[int] = None,
    limit: Optional[int] = None,
    sandbox_python: str = "/usr/bin/python",
) -> Tuple[Dict[str, Any], List[TaskResult]]:
    if limit is not None:
        records = list(records)[:limit]
    results: List[TaskResult] = []
    for record in records:
        try:
            result = generate_for_record(
                model,
                tokenizer,
                record,
                mode=mode,
                n_high_cycles=n_high_cycles,
                max_new_tokens=max_new_tokens,
                context_cap=context_cap,
                max_high_cycles=max_high_cycles,
                sandbox_python=sandbox_python,
            )
        except Exception as exc:  # keep the suite running; record the defect
            results.append(
                TaskResult(
                    task_id=int(record.get("task_id", -1)),
                    expected_name="",
                    prompt_tokens=0,
                    prompt_truncated=False,
                    generated_tokens=0,
                    raw_generation="",
                    assembled_code="",
                    nonempty=False,
                    decodes=False,
                    entry_point_present=False,
                    syntax_valid=False,
                    passed=False,
                    first_test_passed=False,
                    tests_total=len(record.get("test_list", [])),
                    tests_passed=0,
                    error_type="evaluation_exception",
                    error_message=f"{type(exc).__name__}: {exc}",
                    latency_ms=0.0,
                    total_transitions=0,
                    mean_high_cycles=0.0,
                    mean_token_entropy=0.0,
                    mean_token_margin=0.0,
                    mode=mode,
                    requested_depth=n_high_cycles,
                    context_cap=context_cap,
                )
            )
        else:
            results.append(result)
    return aggregate_results(results), results


def evaluate_mbpp_with_syntax_repair(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    depths: Sequence[int] = (1, 2, 3, 4),
    max_new_tokens: int = 96,
    context_cap: int = 256,
    limit: Optional[int] = None,
    sandbox_python: str = "/usr/bin/python",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Bounded syntax-triggered repair.

    Generate at the cheapest depth. If the program fails ``ast.parse``, retry at
    the next depth, up to the explicit maximum. The first syntactically valid
    program is selected. If none is valid, the final attempt is returned. Tests
    are never consulted during repair; only syntax is used.

    This policy is task-adaptive in compute: easy tasks stop after one pass,
    hard tasks consume more passes. It is not a VRAM-adaptation mechanism.
    """
    if limit is not None:
        records = list(records)[:limit]
    rows: List[Dict[str, Any]] = []
    for record in records:
        attempts = []
        chosen = None
        for depth in depths:
            result = generate_for_record(
                model,
                tokenizer,
                record,
                mode="fixed",
                n_high_cycles=int(depth),
                max_new_tokens=max_new_tokens,
                context_cap=context_cap,
                sandbox_python=sandbox_python,
            )
            attempts.append(
                {
                    "depth": int(depth),
                    "syntax_valid": result.syntax_valid,
                    "passed": result.passed,
                    "first_test_passed": result.first_test_passed,
                    "latency_ms": result.latency_ms,
                    "transitions": result.total_transitions,
                }
            )
            chosen = result
            if result.syntax_valid:
                break
        rows.append(
            {
                "task_id": chosen.task_id,
                "passed": chosen.passed,
                "first_test_passed": chosen.first_test_passed,
                "syntax_valid": chosen.syntax_valid,
                "error_type": chosen.error_type,
                "attempts": len(attempts),
                "attempt_depths": [a["depth"] for a in attempts],
                "final_depth": attempts[-1]["depth"],
                "total_transitions": int(sum(a["transitions"] for a in attempts)),
                "total_latency_ms": float(sum(a["latency_ms"] for a in attempts)),
                "attempt_details": attempts,
            }
        )
    n = len(rows)
    aggregate = {
        "total_tasks": n,
        "pass_at_1": sum(r["passed"] for r in rows) / max(1, n),
        "first_test_pass_rate": sum(r["first_test_passed"] for r in rows) / max(1, n),
        "syntax_valid_rate": sum(r["syntax_valid"] for r in rows) / max(1, n),
        "mean_attempts": float(np.mean([r["attempts"] for r in rows])) if rows else 0.0,
        "mean_total_transitions": float(np.mean([r["total_transitions"] for r in rows])) if rows else 0.0,
        "mean_total_latency_ms": float(np.mean([r["total_latency_ms"] for r in rows])) if rows else 0.0,
        "attempt_histogram": {
            str(k): sum(1 for r in rows if r["attempts"] == k)
            for k in range(1, len(depths) + 1)
        },
        "stopped_at_depth": {
            str(d): sum(1 for r in rows if r["final_depth"] == d) for d in depths
        },
    }
    return aggregate, rows


# ---------------------------------------------------------------------------
# Difficulty estimation and compute tiers
# ---------------------------------------------------------------------------


def difficulty_score(
    mean_entropy: float,
    mean_margin: float,
    syntax_valid: bool,
    generated_tokens: int,
    vocab_size: int,
    max_new_tokens: int = 96,
) -> float:
    """Small, label-free difficulty proxy in [0, 1].

    Components: normalized predictive entropy, low top-1/top-2 margin, syntax
    failure, and generation length. The weights are intentionally simple; the
    estimator is validated against measured depth gains in the experiment.
    """
    h = mean_entropy / max(1e-9, math.log(max(2, vocab_size)))
    low_margin = max(0.0, 1.0 - mean_margin)
    syn = 0.0 if syntax_valid else 1.0
    length = min(1.0, generated_tokens / max(1, max_new_tokens))
    return float(np.clip(0.45 * h + 0.20 * low_margin + 0.20 * syn + 0.15 * length, 0.0, 1.0))


def tier_for_score(score: float, min_depth: int, max_depth: int) -> Tuple[int, int]:
    """Map difficulty to (depth, context cap). Explicit hard bounds."""
    if score < 0.33:
        depth = min_depth
        cap = 128
    elif score < 0.66:
        depth = min(max_depth, min_depth + 1)
        cap = 192
    else:
        depth = max_depth
        cap = 256
    return int(np.clip(depth, min_depth, max_depth)), cap


def probe_difficulty(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    probe_tokens: int = 8,
    context_cap: int = 256,
) -> float:
    """Cheap fast-mode probe used by the heuristic controller."""
    result = generate_for_record(
        model,
        tokenizer,
        record,
        mode="fast",
        n_high_cycles=None,
        max_new_tokens=probe_tokens,
        context_cap=context_cap,
    )
    return difficulty_score(
        result.mean_token_entropy,
        result.mean_token_margin,
        result.syntax_valid,
        result.generated_tokens,
        getattr(model.config, "vocab_size", 8000),
        max_new_tokens=probe_tokens,
    )


# ---------------------------------------------------------------------------
# Language-model retention
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_language_modeling(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    mode: str = "fast",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        n = int((labels != IGNORE_INDEX).sum().item())
        if n == 0:
            continue
        out = model(input_ids, mask, mode=mode)
        loss = next_token_cross_entropy(out.logits, labels)
        total_loss += float(loss.item()) * n
        total_tokens += n
        batches += 1
    ce = total_loss / max(1, total_tokens)
    return {
        "ce": ce,
        "ppl": math.exp(min(20.0, ce)),
        "valid_target_tokens": total_tokens,
        "batches": batches,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def paired_deltas(
    baseline: Sequence[TaskResult], treatment: Sequence[TaskResult]
) -> Dict[str, Any]:
    by_id_base = {r.task_id: r for r in baseline}
    by_id_treat = {r.task_id: r for r in treatment}
    ids = sorted(set(by_id_base) & set(by_id_treat))
    both_pass = only_base = only_treat = both_fail = 0
    for task_id in ids:
        a = by_id_base[task_id].passed
        b = by_id_treat[task_id].passed
        if a and b:
            both_pass += 1
        elif a:
            only_base += 1
        elif b:
            only_treat += 1
        else:
            both_fail += 1
    return {
        "paired_tasks": len(ids),
        "both_pass": both_pass,
        "only_baseline_pass": only_base,
        "only_treatment_pass": only_treat,
        "both_fail": both_fail,
        "delta_pass_rate": (only_treat - only_base) / max(1, len(ids)),
    }
