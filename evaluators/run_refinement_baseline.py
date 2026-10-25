"""Phase 0 baseline for the monotonic-refinement program.

Reproduces depth 1-4 quality, adjacent-depth stability, teacher-forced CE, and
the Phase-8 recurrence interventions on the current checkpoint. Writes only
under experiments/re_refinement_monotonicity/; it never writes a checkpoint.

Example:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. \
  .venv-trm-lm/bin/python -m evaluators.run_refinement_baseline \
    --checkpoint experiments/re_coding_adaptive/stdlib_instr_1m/checkpoint_final.pt \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --output-dir experiments/re_refinement_monotonicity/baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    load_tokenizer,
    next_token_cross_entropy,
)
from evaluators.re_eval import build_signature_prompt, parameter_bytes
from evaluators.recurrence_verification import (
    InterventionConfig,
    _sha256,
    evaluate_mbpp_intervention,
    evaluate_tinystories_depths,
    generate_with_recurrence,
    logit_difference,
    logit_difference as _logit_difference,
    pair_stats,
    run_recurrence,
    state_stats,
)
from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-workdir", required=True)
    parser.add_argument("--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1")
    parser.add_argument("--mbpp-dir", default="data/coding/mbpp")
    parser.add_argument("--output-dir", default="experiments/re_refinement_monotonicity/baseline")
    parser.add_argument("--val-limit", type=int, default=45)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-cap", type=int, default=256)
    parser.add_argument("--fixed-seq-len", type=int, default=64)
    parser.add_argument("--lm-max-docs", type=int, default=256)
    parser.add_argument("--prefix-tasks", type=int, default=20)
    parser.add_argument("--sandbox-python", default="/usr/bin/python3.14")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=1.0,
        help="default per-program sandbox wall-clock limit",
    )
    parser.add_argument(
        "--timeout-overrides",
        default=None,
        help="JSON map {task_id: seconds} for canonical slow-reference tasks",
    )
    parser.add_argument(
        "--exclude-task-ids",
        default="",
        help="comma-separated MBPP task ids excluded from the primary benchmark",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-lm", action="store_true")
    return parser.parse_args()


def load_model(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DualStateConfig(**ckpt["model_config"])
    model = DualStateModel(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.rotary_emb.float()
    tokenizer = load_tokenizer(args.tokenizer_workdir)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    return model, tokenizer, ckpt, device, dtype


def build_fixed_batch(records, tokenizer, seq_len, n=4):
    bos = tokenizer.token_to_id("<bos>")
    pad = tokenizer.token_to_id("<pad>")
    input_rows, mask_rows = [], []
    for record in records[:n]:
        prompt, _, _ = build_signature_prompt(record)
        ids = [bos] + tokenizer.encode(prompt).ids
        ids = ids[:seq_len]
        real = len(ids)
        input_rows.append(ids + [pad] * (seq_len - real))
        mask_rows.append([1] * real + [0] * (seq_len - real))
    return (
        torch.tensor(input_rows, dtype=torch.int64),
        torch.tensor(mask_rows, dtype=torch.int64),
    )


def state_trace(model, fixed_inp, fixed_mask, depths=(1, 2, 3, 4)):
    runs = {}
    for depth in depths:
        out = run_recurrence(
            model, fixed_inp, fixed_mask, depth, "none",
            last_token_only=True, record_states=True, record_controller=True,
        )
        runs[f"A{depth}"] = {
            "initial": out["initial"],
            "steps": out["trace"],
            "final_z_H_norm": float(out["z_H"].detach().float().norm().item()),
            "final_z_L_norm": float(out["z_L"].detach().float().norm().item()),
            "final_logits_norm": float(out["final_logits"].detach().float().norm().item()),
        }
    comparisons = {}
    ref = {d: run_recurrence(model, fixed_inp, fixed_mask, d, "none", last_token_only=True)["final_logits"] for d in depths}
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        comparisons[f"A{a}_vs_A{b}"] = logit_difference(ref[a], ref[b])
    broken = {}
    for mode in ["both", "high", "low"]:
        out = run_recurrence(model, fixed_inp, fixed_mask, 4, mode, last_token_only=True)
        broken[f"B4_{mode}"] = logit_difference(ref[1] if mode == "both" else ref[4], out["final_logits"])
    # For depth-conditioned stability updates, broken-both at depth 4 should
    # equal broken-both at depth 2 (the same gated one-step update repeated),
    # even if it no longer equals the original depth-1 output.
    b2_both = run_recurrence(model, fixed_inp, fixed_mask, 2, "both", last_token_only=True)
    b4_both = run_recurrence(model, fixed_inp, fixed_mask, 4, "both", last_token_only=True)
    broken["B4_both_vs_B2_both"] = logit_difference(b2_both["final_logits"], b4_both["final_logits"])
    return {"runs": runs, "comparisons": comparisons, "broken": broken}


def evaluate_mbpp_depths(model, tokenizer, args, records, split_name):
    results = {}
    rows_all = {}
    for depth in [1, 2, 3, 4]:
        cfg = InterventionConfig(f"A{depth}", depth, "none")
        agg, rows = evaluate_mbpp_intervention(
            model, tokenizer, records, cfg,
            max_new_tokens=args.max_new_tokens,
            context_cap=args.context_cap,
            limit=args.val_limit if split_name == "validation" else args.test_limit,
            sandbox_python=args.sandbox_python,
            timeout_sec=args.timeout_sec,
        )
        results[f"depth_{depth}"] = agg
        rows_all[f"depth_{depth}"] = rows
        print(
            f"[{split_name}:depth{depth}] syntax={agg['syntax_valid_rate']:.1%} "
            f"pass={agg['pass_at_1']:.1%} first={agg['first_test_pass_rate']:.1%}",
            flush=True,
        )
    paired = {}
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        ra = rows_all[f"depth_{a}"]
        rb = rows_all[f"depth_{b}"]
        def count(field):
            by_a = {r["task_id"]: bool(r[field]) for r in ra}
            by_b = {r["task_id"]: bool(r[field]) for r in rb}
            ids = sorted(set(by_a) & set(by_b))
            return {
                "both": sum(by_a[i] and by_b[i] for i in ids),
                "only_a": sum(by_a[i] and not by_b[i] for i in ids),
                "only_b": sum(not by_a[i] and by_b[i] for i in ids),
                "neither": sum(not by_a[i] and not by_b[i] for i in ids),
            }
        paired[f"A{a}_vs_A{b}"] = {
            "syntax": count("syntax_valid"),
            "first_test": count("first_test_passed"),
            "passed": count("passed"),
        }
    return {"aggregates": results, "paired": paired, "rows": rows_all}


def teacher_forced_ce(model, tokenizer, args, records):
    device = next(model.parameters()).device
    code_cfg = CodePipelineConfig(seq_len=model.config.seq_len, mbpp_dir=args.mbpp_dir)
    per_task = []
    for record in records:
        example = build_instruction_example(tokenizer, code_cfg, record)
        if example is None:
            continue
        inp = example["input_ids"].unsqueeze(0).to(device)
        lab = example["labels"].unsqueeze(0).to(device)
        mask = example["attention_mask"].unsqueeze(0).to(device)
        n_valid = int((lab != IGNORE_INDEX).sum().item())
        per_depth = {}
        logits_by_depth = {}
        for depth in [1, 2, 3, 4]:
            out = run_recurrence(
                model, inp, mask, depth, "none",
                last_token_only=False, record_states=False,
            )
            logits_by_depth[depth] = out["final_logits"]
            per_depth[depth] = float(next_token_cross_entropy(out["final_logits"], lab).item())
        adjacent = {}
        for a, b in [(1, 2), (2, 3), (3, 4)]:
            adjacent[f"{a}_to_{b}"] = {
                **logit_difference(logits_by_depth[a], logits_by_depth[b]),
                "ce_first": per_depth[a],
                "ce_second": per_depth[b],
                "delta_ce_first_minus_second": per_depth[a] - per_depth[b],
            }
        per_task.append(
            {
                "task_id": int(record["task_id"]),
                "valid_target_tokens": n_valid,
                "ce": {str(k): v for k, v in per_depth.items()},
                "adjacent": adjacent,
            }
        )
    summary = {}
    for depth in [1, 2, 3, 4]:
        vals = [r["ce"][str(depth)] for r in per_task]
        tokens = [r["valid_target_tokens"] for r in per_task]
        weighted = (
            float(np.average(vals, weights=tokens)) if vals and sum(tokens) > 0 else None
        )
        summary[f"depth_{depth}"] = {
            "mean_ce_unweighted": float(np.mean(vals)) if vals else None,
            "median_ce": float(np.median(vals)) if vals else None,
            "token_weighted_ce": weighted,
            "valid_target_tokens": int(sum(tokens)) if tokens else 0,
            "tasks": len(vals),
        }
    adjacent_summary = {}
    for pair in ["1_to_2", "2_to_3", "3_to_4"]:
        vals = [r["adjacent"][pair]["delta_ce_first_minus_second"] for r in per_task]
        changes = [r["adjacent"][pair]["top1_change_rate"] for r in per_task]
        adjacent_summary[pair] = {
            "mean_delta_ce": float(np.mean(vals)) if vals else None,
            "fraction_improved": float(np.mean([v > 0 for v in vals])) if vals else None,
            "fraction_degraded": float(np.mean([v <= 0 for v in vals])) if vals else None,
            "mean_top1_change_rate": float(np.mean(changes)) if changes else None,
            "tasks": len(vals),
        }
    return {"per_task": per_task, "summary": summary, "adjacent_summary": adjacent_summary}


def prefix_stability(model, tokenizer, args, records):
    rows = []
    for record in records[: args.prefix_tasks]:
        prompt, name, _ = build_signature_prompt(record)
        gens = {}
        for depth in [1, 2, 3, 4]:
            gen = generate_with_recurrence(
                model, tokenizer, prompt, depth, "none",
                max_new_tokens=args.max_new_tokens, context_cap=args.context_cap,
            )
            gens[depth] = gen
        for a, b in [(1, 2), (2, 3), (3, 4)]:
            ta = gens[a]["token_ids"]
            tb = gens[b]["token_ids"]
            prefix = 0
            for x, y in zip(ta, tb):
                if x != y:
                    break
                prefix += 1
            rows.append(
                {
                    "task_id": record["task_id"],
                    "pair": f"{a}_to_{b}",
                    "len_a": len(ta),
                    "len_b": len(tb),
                    "common_prefix": prefix,
                    "prefix_fraction_of_a": prefix / max(1, len(ta)),
                    "syntax_a": None,
                    "syntax_b": None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(1337)
    np.random.seed(1337)
    model, tokenizer, ckpt, device, dtype = load_model(args)
    sha = _sha256(args.checkpoint)
    print(f"[init] checkpoint={args.checkpoint} sha256={sha} params={model.num_parameters:,}", flush=True)

    code_cfg = CodePipelineConfig(seq_len=model.config.seq_len, mbpp_dir=args.mbpp_dir)
    all_records = build_mbpp_eval_records(code_cfg)
    excluded_task_ids = sorted(
        int(value) for value in args.exclude_task_ids.split(",") if value.strip()
    )
    if excluded_task_ids:
        all_records = {
            split: [r for r in rows if int(r["task_id"]) not in excluded_task_ids]
            for split, rows in all_records.items()
        }
        print(f"[exclude] task_ids={excluded_task_ids}", flush=True)
    timeout_overrides = {}
    if args.timeout_overrides:
        with open(args.timeout_overrides, "r", encoding="utf-8") as handle:
            timeout_overrides = {
                int(key): float(value)
                for key, value in json.load(handle).items()
                if str(key).isdigit()
            }
        print(f"[timeouts] overrides={timeout_overrides}", flush=True)
    for split, rows in all_records.items():
        for record in rows:
            record["timeout_sec"] = timeout_overrides.get(
                int(record["task_id"]), args.timeout_sec
            )
    fixed_inp, fixed_mask = build_fixed_batch(all_records["validation"], tokenizer, args.fixed_seq_len)
    fixed_inp = fixed_inp.to(device)
    fixed_mask = fixed_mask.to(device)

    print("[1/5] state trace and recurrence interventions", flush=True)
    trace = state_trace(model, fixed_inp, fixed_mask)
    print(f"  comparisons={json.dumps(trace['comparisons'])}", flush=True)
    print(f"  broken={json.dumps(trace['broken'])}", flush=True)

    print("[2/5] MBPP validation depth sweep", flush=True)
    val = evaluate_mbpp_depths(model, tokenizer, args, all_records["validation"], "validation")
    test = {}
    if not args.skip_test:
        print("[3/5] MBPP test depth sweep", flush=True)
        test = evaluate_mbpp_depths(model, tokenizer, args, all_records["test"], "test")

    print("[4/5] teacher-forced MBPP CE", flush=True)
    tf = teacher_forced_ce(model, tokenizer, args, all_records["validation"][: args.val_limit])
    print(f"  CE summary={json.dumps(tf['summary'])}", flush=True)
    print(f"  adjacent={json.dumps(tf['adjacent_summary'])}", flush=True)

    lm = {}
    if not args.skip_lm:
        print("[5/5] TinyStories depth CE", flush=True)
        manifest_path = os.path.join(args.tinystories_workdir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        cfg = TinyStoriesConfig(
            parquet_path=manifest["parquet"]["path"],
            workdir=args.tinystories_workdir,
            seq_len=model.config.seq_len,
            seed=0,
        )
        ds = TinyStoriesLMDataset(cfg, manifest, tokenizer, split="val", max_docs=args.lm_max_docs)
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_lm_batch)
        lm = evaluate_tinystories_depths(
            model, loader, device,
            [InterventionConfig("A1", 1), InterventionConfig("A2", 2), InterventionConfig("A3", 3), InterventionConfig("A4", 4), InterventionConfig("B4_both", 4, "both")],
        )
        print(f"  LM={json.dumps({k: v['ce'] for k, v in lm.items() if isinstance(v, dict) and 'ce' in v})}", flush=True)

    prefix = prefix_stability(model, tokenizer, args, all_records["validation"])

    results = {
        "schema": "re_refinement_monotonicity_baseline_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha,
        "model_parameters": model.num_parameters,
        "weights_mb": parameter_bytes(model) / 1e6,
        "device": str(device),
        "dtype": str(dtype),
        "args": vars(args),
        "excluded_task_ids": excluded_task_ids,
        "timeout_overrides": timeout_overrides,
        "trace": trace,
        "mbpp_validation": val,
        "mbpp_test": test,
        "teacher_forced_mbpp": tf,
        "language_modeling": lm,
        "prefix_stability": prefix,
    }
    out_json = os.path.join(args.output_dir, "baseline_results.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    with open(os.path.join(args.output_dir, "reproduction_command.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. .venv-trm-lm/bin/python "
            "-m evaluators.run_refinement_baseline "
            f"--checkpoint {args.checkpoint} --tokenizer-workdir {args.tokenizer_workdir} "
            f"--output-dir {args.output_dir} --val-limit {args.val_limit} --test-limit {args.test_limit}\n"
        )
    print(f"[done] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
