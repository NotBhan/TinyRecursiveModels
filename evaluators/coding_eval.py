"""Evaluation harness for Language Modeling and Coding Capabilities.

Supports:
- Ordinary language modeling evaluation on held-out TinyStories.
- Code continuation evaluation (syntax validity, exact match).
- Function synthesis evaluation with sandboxed execution on hidden tests.
- Rigorous Thinking OFF vs Thinking ON comparative evaluation with latency,
  memory, and transition accounting.
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from dataset.coding_reasoning import (
    DELIMITER_PROBLEM,
    DELIMITER_SOLUTION,
    format_task_c_function_synthesis,
)
from dataset.tinystories_lm import (
    IGNORE_INDEX,
    count_valid_targets,
    next_token_cross_entropy,
)
from evaluators.execution_sandbox import ExecutionResult, execute_sandboxed


def extract_python_code(
    raw_generation: str,
    preserve_leading_indent: bool = False,
    strip_first_def_indent: bool = False,
) -> str:
    """Extract clean Python code from raw model generation.

    Handles:
    - Markdown code fences (```python ... ```)
    - Code starting with 'def ' or class definitions
    - Delimiter tokens

    ``preserve_leading_indent`` is required for body-only completions. In that
    mode the first line's spaces/tabs are part of the program and must not be
    stripped. A leading blank line is still removed.

    ``strip_first_def_indent`` is only for callers that know the generation is
    a complete top-level function whose indentation is an artifact. It must not
    be used for body-only completions: a body may legitimately start with an
    indented nested ``def``/``class``, and stripping that indent corrupts the
    program (MBPP test tasks 215 and 317 are canonical examples).
    """
    text = raw_generation if preserve_leading_indent else raw_generation.strip()

    # If code contains markdown fences
    fence_match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_generation, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
        if not preserve_leading_indent:
            return text.strip()
    elif DELIMITER_SOLUTION in text:
        text = text.split(DELIMITER_SOLUTION, 1)[1]

    # Stop at subsequent delimiters or stop markers
    for stop_seq in ["###", "<eos>", "<bos>", "```"]:
        if stop_seq in text:
            text = text.split(stop_seq, 1)[0]

    text = text.rstrip()
    if not preserve_leading_indent:
        return text.strip()

    # Drop leading blank lines, but keep the indentation of the first real line.
    lines = text.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""
    if strip_first_def_indent and re.match(r"^\s*(async\s+)?(def|class)\s+", lines[0]):
        lines[0] = lines[0].lstrip()
    return "".join(lines)


def assemble_body_completion(
    prompt_seen: str,
    expected_name: str,
    raw_generation: str,
) -> str:
    """Assemble a prompt context and a generated body into one program.

    The evaluation prompt ends with the target ``def`` line. Two generation
    shapes are accepted:

    1. Body-only completion (the intended shape): prepend the exact signature
       from the prompt. Indentation, including a nested ``def``/``class`` on
       the first body line, is preserved.
    2. Complete top-level function: use the generated definition and prepend
       only the pre-signature helpers, avoiding a bodyless duplicate ``def``.

    A complete function is detected only at column zero. An indented ``def`` is
    a nested body definition and must keep its indentation.
    """
    cleaned = extract_python_code(raw_generation, preserve_leading_indent=True)
    context = prompt_seen.split(DELIMITER_SOLUTION, 1)[-1]
    lines = context.splitlines(keepends=True)
    def_index = None
    for idx, line in enumerate(lines):
        if re.match(rf"^\s*(async\s+)?def\s+{re.escape(expected_name)}\s*\(", line):
            def_index = idx
    if def_index is None:
        prefix, signature = context, ""
    else:
        prefix = "".join(lines[:def_index])
        signature = "".join(lines[def_index:])
    generated_def = re.search(
        rf"^(async\s+)?def\s+{re.escape(expected_name)}\s*\(",
        cleaned,
        re.MULTILINE,
    )
    return (prefix + cleaned) if generated_def else (prefix + signature + cleaned)


def check_syntax_validity(code: str) -> bool:
    """Check if Python code parses into a valid AST."""
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


@dataclass
class CodingTaskEvaluation:
    task_id: int
    prompt: str
    generated_code: str
    extracted_code: str
    syntax_valid: bool
    passed: bool
    error_type: Optional[str]
    error_message: str
    latency_ms: float
    transitions: int
    high_cycles: int
    mode: str


def generate_code_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    mode: str = "fast",
    device: Optional[torch.device] = None,
    **forward_kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Autoregressive generation for a prompt under specified mode.

    Returns (generated_text, stats_dict).
    """
    if device is None:
        device = next(model.parameters()).device

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = [bos_id] + list(prompt_ids)

    curr_tensor = torch.tensor([input_ids], dtype=torch.int64, device=device)
    mask = torch.ones_like(curr_tensor)

    total_transitions = 0
    total_high_cycles = 0
    model.eval()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = curr_tensor.shape[1]
            max_seq = getattr(model.config, "max_position_embeddings", getattr(model.config, "seq_len", 256))
            if seq_len > max_seq:
                break

            out = model(curr_tensor, mask, mode=mode, **forward_kwargs)
            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            total_transitions += int(out.transitions[:, -1].item() if out.transitions.dim() > 1 else out.transitions[0].item())
            total_high_cycles += int(out.high_cycles[:, -1].item() if out.high_cycles.dim() > 1 else out.high_cycles[0].item())

            token_val = next_token.item()
            curr_tensor = torch.cat([curr_tensor, next_token], dim=1)
            mask = torch.ones_like(curr_tensor)

            if token_val == eos_id:
                break

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    generated_tokens = curr_tensor[0, len(input_ids):].cpu().tolist()
    # Strip eos if present
    if generated_tokens and generated_tokens[-1] == eos_id:
        generated_tokens = generated_tokens[:-1]

    generated_text = tokenizer.decode(generated_tokens)
    stats = {
        "latency_ms": latency_ms,
        "total_transitions": total_transitions,
        "total_high_cycles": total_high_cycles,
        "num_generated_tokens": len(generated_tokens),
        "mode": mode,
    }
    return generated_text, stats


def evaluate_function_synthesis(
    model: Any,
    tokenizer: Any,
    eval_records: Sequence[Dict[str, Any]],
    mode: str = "fast",
    max_new_tokens: int = 128,
    timeout_sec: float = 3.0,
    device: Optional[torch.device] = None,
    **forward_kwargs,
) -> Dict[str, Any]:
    """Evaluate Python function synthesis on held-out tasks.

    Runs generation, code extraction, static parsing, and sandboxed test
    execution. Hidden tests are never fed into the model prompt.
    """
    results: List[CodingTaskEvaluation] = []
    total_tasks = len(eval_records)
    passed_count = 0
    syntax_valid_count = 0
    total_latency_ms = 0.0
    total_transitions = 0
    total_high_cycles = 0

    error_category_counts: Dict[str, int] = {
        "syntax_error": 0,
        "assertion_error": 0,
        "timeout": 0,
        "prohibited_import": 0,
        "runtime_error": 0,
        "memory_limit": 0,
    }

    for record in eval_records:
        tid = record["task_id"]
        prob_text = record["problem_text"]
        test_list = record.get("test_list", [])
        test_setup = record.get("test_setup_code", "")

        # Format prompt
        prompt = f"{DELIMITER_PROBLEM}{prob_text.strip()}{DELIMITER_SOLUTION}"

        # Generate candidate
        raw_code, stats = generate_code_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            mode=mode,
            device=device,
            **forward_kwargs,
        )

        extracted = extract_python_code(raw_code)
        syntax_ok = check_syntax_validity(extracted)
        if syntax_ok:
            syntax_valid_count += 1

        exec_res = execute_sandboxed(
            candidate_code=extracted,
            test_list=test_list,
            test_setup_code=test_setup,
            timeout_sec=timeout_sec,
        )

        if exec_res.passed:
            passed_count += 1
        elif exec_res.error_type:
            error_category_counts[exec_res.error_type] = (
                error_category_counts.get(exec_res.error_type, 0) + 1
            )

        task_eval = CodingTaskEvaluation(
            task_id=tid,
            prompt=prob_text,
            generated_code=raw_code,
            extracted_code=extracted,
            syntax_valid=syntax_ok,
            passed=exec_res.passed,
            error_type=exec_res.error_type,
            error_message=exec_res.error_message,
            latency_ms=stats["latency_ms"],
            transitions=stats["total_transitions"],
            high_cycles=stats["total_high_cycles"],
            mode=mode,
        )
        results.append(task_eval)
        total_latency_ms += stats["latency_ms"]
        total_transitions += stats["total_transitions"]
        total_high_cycles += stats["total_high_cycles"]

    pass_rate = passed_count / max(1, total_tasks)
    syntax_rate = syntax_valid_count / max(1, total_tasks)
    mean_latency = total_latency_ms / max(1, total_tasks)
    mean_transitions = total_transitions / max(1, total_tasks)
    mean_high_cycles = total_high_cycles / max(1, total_tasks)

    return {
        "mode": mode,
        "total_tasks": total_tasks,
        "passed_count": passed_count,
        "pass_rate_pass_at_1": pass_rate,
        "syntax_valid_count": syntax_valid_count,
        "syntax_valid_rate": syntax_rate,
        "mean_latency_ms": mean_latency,
        "mean_transitions": mean_transitions,
        "mean_high_cycles": mean_high_cycles,
        "error_categories": error_category_counts,
        "per_task_results": results,
    }


def compare_thinking_off_vs_on(
    model: Any,
    tokenizer: Any,
    eval_records: Sequence[Dict[str, Any]],
    max_new_tokens: int = 128,
    timeout_sec: float = 3.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Paired per-problem comparison of Thinking OFF (fast) vs Thinking ON (adaptive)."""
    off_results = evaluate_function_synthesis(
        model=model,
        tokenizer=tokenizer,
        eval_records=eval_records,
        mode="fast",
        max_new_tokens=max_new_tokens,
        timeout_sec=timeout_sec,
        device=device,
    )
    on_results = evaluate_function_synthesis(
        model=model,
        tokenizer=tokenizer,
        eval_records=eval_records,
        mode="adaptive",
        max_new_tokens=max_new_tokens,
        timeout_sec=timeout_sec,
        device=device,
    )

    # Paired comparisons
    tasks_off = {t.task_id: t for t in off_results["per_task_results"]}
    tasks_on = {t.task_id: t for t in on_results["per_task_results"]}

    both_pass = 0
    only_on_pass = 0
    only_off_pass = 0
    both_fail = 0

    for tid in tasks_off:
        if tid in tasks_on:
            pass_off = tasks_off[tid].passed
            pass_on = tasks_on[tid].passed
            if pass_off and pass_on:
                both_pass += 1
            elif pass_on and not pass_off:
                only_on_pass += 1
            elif pass_off and not pass_on:
                only_off_pass += 1
            else:
                both_fail += 1

    comparison = {
        "thinking_off": {
            "mode": "fast (depth 1)",
            "pass_at_1": off_results["pass_rate_pass_at_1"],
            "syntax_valid_rate": off_results["syntax_valid_rate"],
            "mean_latency_ms": off_results["mean_latency_ms"],
            "mean_transitions": off_results["mean_transitions"],
            "error_categories": off_results["error_categories"],
            "per_task_results": [asdict(r) for r in off_results["per_task_results"]],
        },
        "thinking_on": {
            "mode": "adaptive (depths 1..max)",
            "pass_at_1": on_results["pass_rate_pass_at_1"],
            "syntax_valid_rate": on_results["syntax_valid_rate"],
            "mean_latency_ms": on_results["mean_latency_ms"],
            "mean_transitions": on_results["mean_transitions"],
            "error_categories": on_results["error_categories"],
            "per_task_results": [asdict(r) for r in on_results["per_task_results"]],
        },
        "paired_analysis": {
            "both_pass": both_pass,
            "only_thinking_on_pass": only_on_pass,
            "only_thinking_off_pass": only_off_pass,
            "both_fail": both_fail,
            "delta_pass_rate": on_results["pass_rate_pass_at_1"] - off_results["pass_rate_pass_at_1"],
            "latency_slowdown_ratio": (
                on_results["mean_latency_ms"] / max(0.001, off_results["mean_latency_ms"])
            ),
        },
    }
    return comparison
