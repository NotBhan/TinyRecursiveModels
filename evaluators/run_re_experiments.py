"""Run RE:Inference coding, adaptive-compute, and VRAM experiments.

The driver loads a checkpoint, runs quality and memory measurements, and writes
a machine-readable JSON plus a markdown summary. It never modifies checkpoints.

Example:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. python -m evaluators.run_re_experiments \
    --checkpoint experiments/re_coding_adaptive/backbone_3m/checkpoint_final.pt \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --workdir experiments/re_coding_adaptive/eval_backbone_3m \
    --split validation --limit 45 --modes fast,fixed2,fixedmax,adaptive,tiered
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.code_pipeline import (
    CodePipelineConfig,
    build_instruction_example,
    build_mbpp_eval_records,
)
from dataset.tinystories_lm import (
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    load_tokenizer,
    next_token_cross_entropy,
)
from evaluators.re_eval import (
    TaskResult,
    aggregate_results,
    build_signature_prompt,
    difficulty_score,
    evaluate_language_modeling,
    evaluate_mbpp,
    evaluate_mbpp_with_syntax_repair,
    generate_for_record,
    parameter_bytes,
    peak_memory_context,
    probe_difficulty,
    tier_for_score,
)
from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-workdir", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1")
    parser.add_argument("--mbpp-dir", default="data/coding/mbpp")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=45)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-cap", type=int, default=256)
    parser.add_argument("--sandbox-python", default="/usr/bin/python3.14")
    parser.add_argument(
        "--modes",
        default="fast,fixed2,fixedmax,adaptive,tiered",
        help="comma-separated: fast,fast128,fixed2,fixedmax,adaptive,tiered,repair",
    )
    parser.add_argument("--memory-easy-hard", type=int, default=3)
    parser.add_argument("--memory-max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def load_model(checkpoint: str, device: torch.device, dtype: torch.dtype) -> Tuple[DualStateModel, Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = DualStateConfig(**ckpt["model_config"])
    model = DualStateModel(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.rotary_emb.float()
    return model, ckpt


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = np.mean(ranks[order[i:j]])
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def teacher_forced_ce_by_depth(
    model: DualStateModel,
    tokenizer: Any,
    config: CodePipelineConfig,
    record: Dict[str, Any],
    depths: Sequence[int],
) -> Optional[Dict[str, float]]:
    example = build_instruction_example(tokenizer, config, record)
    if example is None:
        return None
    device = next(model.parameters()).device
    inp = example["input_ids"].unsqueeze(0).to(device)
    lab = example["labels"].unsqueeze(0).to(device)
    mask = example["attention_mask"].unsqueeze(0).to(device)
    max_depth = max(depths)
    out = model.forward_depths(
        inp,
        mask,
        depths=list(depths),
        max_depth=max_depth,
        evaluate_controller=False,
    )
    result = {}
    for d in depths:
        ce = float(next_token_cross_entropy(out.logits_by_depth[d], lab).item())
        result[f"ce_depth_{d}"] = ce
    return result


def run_tiered(
    model: DualStateModel,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    min_depth: int,
    max_depth: int,
) -> Tuple[Dict[str, Any], List[TaskResult]]:
    results: List[TaskResult] = []
    for record in records:
        score = probe_difficulty(
            model, tokenizer, record, probe_tokens=8, context_cap=args.context_cap
        )
        depth, cap = tier_for_score(score, min_depth, max_depth)
        generated = generate_for_record(
            model,
            tokenizer,
            record,
            mode="fixed",
            n_high_cycles=depth,
            max_new_tokens=args.max_new_tokens,
            context_cap=cap,
            sandbox_python=args.sandbox_python,
        )
        generated.difficulty_score = score
        generated.predicted_depth = depth
        results.append(generated)
    return aggregate_results(results), results


def memory_microbenchmark(
    model: DualStateModel,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    configs: Sequence[Tuple[str, str, Optional[int], int, bool]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        for name, mode, depth, cap, last_only in configs:
            with peak_memory_context(model, next(model.parameters()).device) as mem:
                result = generate_for_record(
                    model,
                    tokenizer,
                    record,
                    mode=mode,
                    n_high_cycles=depth,
                    max_new_tokens=args.memory_max_new_tokens,
                    context_cap=cap,
                    max_high_cycles=depth if mode == "adaptive" else None,
                    sandbox_python=args.sandbox_python,
                    last_token_only=last_only,
                )
            rows.append(
                {
                    "task_id": result.task_id,
                    "config": name,
                    "mode": mode,
                    "depth": depth,
                    "context_cap": cap,
                    "last_token_only": last_only,
                    "prompt_tokens": result.prompt_tokens,
                    "generated_tokens": result.generated_tokens,
                    "syntax_valid": result.syntax_valid,
                    "passed": result.passed,
                    "latency_ms": result.latency_ms,
                    "transitions": result.total_transitions,
                    "peak_allocated_mb": mem.peak_allocated_mb,
                    "peak_reserved_mb": mem.peak_reserved_mb,
                    "weights_mb": mem.weights_mb,
                    "working_allocated_mb": mem.working_allocated_mb,
                }
            )
    return rows


def lm_retention(
    model: DualStateModel,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    manifest_path = os.path.join(args.tinystories_workdir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cfg = TinyStoriesConfig(
        parquet_path=manifest["parquet"]["path"],
        workdir=args.tinystories_workdir,
        seq_len=model.config.seq_len,
        seed=0,
    )
    ds = TinyStoriesLMDataset(cfg, manifest, tokenizer, split="val", max_docs=256)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_lm_batch)
    fast = evaluate_language_modeling(model, loader, device, mode="fast", max_batches=None)
    adaptive = evaluate_language_modeling(model, loader, device, mode="adaptive", max_batches=None)
    return {"fast": fast, "adaptive": adaptive}


def main() -> None:
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32
    model, ckpt = load_model(args.checkpoint, device, dtype)
    tokenizer = load_tokenizer(args.tokenizer_workdir)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} != model vocab "
            f"{model.config.vocab_size}"
        )
    code_cfg = CodePipelineConfig(seq_len=model.config.seq_len, mbpp_dir=args.mbpp_dir)
    records = build_mbpp_eval_records(code_cfg)[args.split]
    print(
        f"[init] checkpoint tokens={ckpt.get('tokens_processed')} params={model.num_parameters:,} "
        f"weights={parameter_bytes(model)/1e6:.2f} MB records={len(records)}",
        flush=True,
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "checkpoint_tokens_processed": ckpt.get("tokens_processed"),
        "model_parameters": model.num_parameters,
        "weights_mb": parameter_bytes(model) / 1e6,
        "split": args.split,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "context_cap": args.context_cap,
        "modes": {},
    }

    # --- Quality + aggregate peak memory per configuration ---
    configs = []
    for mode in modes:
        if mode == "fast":
            configs.append(("fast_256", "fast", None, args.context_cap))
        elif mode == "fast128":
            configs.append(("fast_128", "fast", None, 128))
        elif mode == "fixed2":
            configs.append(("fixed_2", "fixed", 2, args.context_cap))
        elif mode == "fixedmax":
            configs.append(("fixed_max", "fixed", model.config.max_high_cycles, args.context_cap))
        elif mode == "adaptive":
            configs.append(("adaptive", "adaptive", None, args.context_cap))
        elif mode == "tiered":
            configs.append(("tiered", "tiered", None, args.context_cap))
        elif mode == "repair":
            configs.append(("syntax_repair", "repair", None, args.context_cap))
        else:
            raise ValueError(f"unknown mode {mode}")

    per_task_results: Dict[str, List[TaskResult]] = {}
    for name, mode, depth, cap in configs:
        with peak_memory_context(model, device) as mem:
            if mode == "tiered":
                agg, task_results = run_tiered(
                    model,
                    tokenizer,
                    records[: args.limit],
                    args,
                    model.config.min_high_cycles,
                    model.config.max_high_cycles,
                )
            elif mode == "repair":
                agg, task_results = evaluate_mbpp_with_syntax_repair(
                    model,
                    tokenizer,
                    records,
                    depths=list(range(1, model.config.max_high_cycles + 1)),
                    max_new_tokens=args.max_new_tokens,
                    context_cap=cap,
                    limit=args.limit,
                    sandbox_python=args.sandbox_python,
                )
            else:
                agg, task_results = evaluate_mbpp(
                    model,
                    tokenizer,
                    records,
                    mode=mode,
                    n_high_cycles=depth,
                    max_new_tokens=args.max_new_tokens,
                    context_cap=cap,
                    max_high_cycles=model.config.max_high_cycles if mode == "adaptive" else None,
                    limit=args.limit,
                    sandbox_python=args.sandbox_python,
                )
        results["modes"][name] = {"aggregate": agg, "memory": mem.__dict__}
        if mode == "repair":
            results["modes"][name]["rows"] = task_results
        per_task_results[name] = task_results
        latency = agg.get("mean_latency_ms", agg.get("mean_total_latency_ms", 0.0))
        transitions = agg.get("mean_transitions", agg.get("mean_total_transitions", 0.0))
        print(
            f"[{name}] syntax={agg.get('syntax_valid_rate',0):.1%} "
            f"pass@1={agg.get('pass_at_1',0):.1%} "
            f"latency={latency:.1f} ms "
            f"transitions={transitions:.1f} "
            f"peak_alloc={mem.peak_allocated_mb:.1f} MB",
            flush=True,
        )

    # --- Difficulty validation ---
    difficulty_rows = []
    for record, fast_result in zip(records[: args.limit], per_task_results.get("fast_256", [])):
        ce = teacher_forced_ce_by_depth(
            model,
            tokenizer,
            code_cfg,
            record,
            depths=[1, model.config.max_high_cycles],
        )
        score = difficulty_score(
            fast_result.mean_token_entropy,
            fast_result.mean_token_margin,
            fast_result.syntax_valid,
            fast_result.generated_tokens,
            model.config.vocab_size,
            args.max_new_tokens,
        )
        if ce is not None:
            difficulty_rows.append(
                {
                    "task_id": fast_result.task_id,
                    "difficulty_score": score,
                    "fast_syntax": fast_result.syntax_valid,
                    "fast_passed": fast_result.passed,
                    "fast_entropy": fast_result.mean_token_entropy,
                    "fast_margin": fast_result.mean_token_margin,
                    **ce,
                    "delta_ce_1_to_max": ce[f"ce_depth_1"] - ce[f"ce_depth_{model.config.max_high_cycles}"],
                }
            )
    scores = [r["difficulty_score"] for r in difficulty_rows]
    deltas = [r["delta_ce_1_to_max"] for r in difficulty_rows]
    results["difficulty_validation"] = {
        "rows": difficulty_rows,
        "spearman_score_vs_depth_gain": spearman(scores, deltas),
        "mean_delta_ce_1_to_max": float(np.mean(deltas)) if deltas else None,
        "tasks_where_refinement_helps": int(sum(d > 0 for d in deltas)),
        "tasks_total": len(deltas),
    }
    print(
        f"[difficulty] n={len(difficulty_rows)} "
        f"spearman={results['difficulty_validation']['spearman_score_vs_depth_gain']} "
        f"mean_delta_CE={results['difficulty_validation']['mean_delta_ce_1_to_max']}",
        flush=True,
    )

    # --- Memory microbenchmark: depth vs context vs logits ---
    if difficulty_rows and args.memory_easy_hard > 0:
        ranked = sorted(difficulty_rows, key=lambda r: r["delta_ce_1_to_max"])
        easy = ranked[: args.memory_easy_hard]
        hard = ranked[-args.memory_easy_hard :]
        by_id = {int(r["task_id"]): r for r in records}
        mem_configs = [
            ("fast_cap128", "fast", None, 128, True),
            ("fast_cap256", "fast", None, 256, True),
            ("fixedmax_cap128", "fixed", model.config.max_high_cycles, 128, True),
            ("fixedmax_cap256", "fixed", model.config.max_high_cycles, 256, True),
            ("fixedmax_cap256_full_logits", "fixed", model.config.max_high_cycles, 256, False),
        ]
        easy_rows = memory_microbenchmark(
            model, tokenizer, [by_id[int(r["task_id"])] for r in easy], mem_configs, args
        )
        hard_rows = memory_microbenchmark(
            model, tokenizer, [by_id[int(r["task_id"])] for r in hard], mem_configs, args
        )
        # tiered per-task measurements
        tiered_rows = []
        for label, group in [("easy", easy), ("hard", hard)]:
            for row in group:
                record = by_id[int(row["task_id"])]
                score = row["difficulty_score"]
                depth, cap = tier_for_score(
                    score, model.config.min_high_cycles, model.config.max_high_cycles
                )
                with peak_memory_context(model, device) as mem:
                    generated = generate_for_record(
                        model,
                        tokenizer,
                        record,
                        mode="fixed",
                        n_high_cycles=depth,
                        max_new_tokens=args.memory_max_new_tokens,
                        context_cap=cap,
                        sandbox_python=args.sandbox_python,
                    )
                tiered_rows.append(
                    {
                        "group": label,
                        "task_id": generated.task_id,
                        "difficulty_score": score,
                        "depth": depth,
                        "context_cap": cap,
                        "peak_allocated_mb": mem.peak_allocated_mb,
                        "peak_reserved_mb": mem.peak_reserved_mb,
                        "working_allocated_mb": mem.working_allocated_mb,
                        "latency_ms": generated.latency_ms,
                        "transitions": generated.total_transitions,
                        "syntax_valid": generated.syntax_valid,
                    }
                )
        results["memory_microbenchmark"] = {
            "easy_tasks": easy,
            "hard_tasks": hard,
            "easy_rows": easy_rows,
            "hard_rows": hard_rows,
            "tiered_rows": tiered_rows,
        }
        print(f"[memory] easy rows={len(easy_rows)} hard rows={len(hard_rows)}", flush=True)

    # --- Language retention ---
    if not args.skip_lm:
        results["language_retention"] = lm_retention(model, tokenizer, args, device)
        print(
            f"[lm] fast CE={results['language_retention']['fast']['ce']:.4f} "
            f"adaptive CE={results['language_retention']['adaptive']['ce']:.4f}",
            flush=True,
        )

    with open(os.path.join(args.workdir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(f"[done] wrote {os.path.join(args.workdir, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
