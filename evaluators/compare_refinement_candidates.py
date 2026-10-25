"""Compare monotonic-refinement candidate evaluations.

Example:
  PYTHONPATH=. .venv-trm-lm/bin/python -m evaluators.compare_refinement_candidates \
    --baseline experiments/re_refinement_monotonicity/baseline/baseline_results.json \
    --candidate control=.../diagnostic/control/eval/baseline_results.json \
    --candidate M1=.../diagnostic/M1_monotonic/eval/baseline_results.json \
    --output experiments/re_refinement_monotonicity/DEPTH_QUALITY_RESULTS.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--output", default="experiments/re_refinement_monotonicity/DEPTH_QUALITY_RESULTS.json")
    return parser.parse_args()


def summarize(path: str) -> Dict[str, Any]:
    d = json.load(open(path))
    out: Dict[str, Any] = {
        "path": path,
        "checkpoint": d.get("checkpoint"),
        "checkpoint_sha256": d.get("checkpoint_sha256"),
        "mbpp_validation": {},
        "language_modeling": {},
        "teacher_forced": {},
        "adjacent": {},
        "trace": {},
    }
    val = d.get("mbpp_validation", {}).get("aggregates", {})
    for depth in [1, 2, 3, 4]:
        agg = val.get(f"depth_{depth}", {})
        out["mbpp_validation"][f"depth_{depth}"] = {
            "syntax_valid_rate": agg.get("syntax_valid_rate"),
            "pass_at_1": agg.get("pass_at_1"),
            "first_test_pass_rate": agg.get("first_test_pass_rate"),
            "error_counts": agg.get("error_counts"),
        }
    lm = d.get("language_modeling", {})
    for name in ["A1", "A2", "A3", "A4", "B4_both"]:
        if name in lm and isinstance(lm[name], dict) and "ce" in lm[name]:
            out["language_modeling"][name] = lm[name]["ce"]
    tf = d.get("teacher_forced_mbpp", {})
    for depth in [1, 2, 3, 4]:
        s = tf.get("summary", {}).get(f"depth_{depth}", {})
        out["teacher_forced"][f"depth_{depth}"] = {
            "token_weighted_ce": s.get("token_weighted_ce"),
            "mean_ce_unweighted": s.get("mean_ce_unweighted"),
            "tasks": s.get("tasks"),
        }
    for pair in ["1_to_2", "2_to_3", "3_to_4"]:
        a = tf.get("adjacent_summary", {}).get(pair, {})
        out["adjacent"][pair] = {
            "mean_delta_ce": a.get("mean_delta_ce"),
            "fraction_improved": a.get("fraction_improved"),
            "fraction_degraded": a.get("fraction_degraded"),
            "mean_top1_change_rate": a.get("mean_top1_change_rate"),
        }
    trace = d.get("trace", {})
    out["trace"]["comparisons"] = trace.get("comparisons", {})
    out["trace"]["broken"] = trace.get("broken", {})
    out["trace"]["state_steps"] = {
        name: run.get("steps", []) for name, run in trace.get("runs", {}).items()
    }
    # Derived monotonicity indicators.
    lm_ce = out["language_modeling"]
    out["derived"] = {
        "lm_depth4_minus_depth1": (
            (lm_ce.get("A4") - lm_ce.get("A1"))
            if "A4" in lm_ce and "A1" in lm_ce else None
        ),
        "lm_depth4_lt_depth1": (
            lm_ce.get("A4") < lm_ce.get("A1")
            if "A4" in lm_ce and "A1" in lm_ce else None
        ),
        "lm_monotonic_1_2_3_4": (
            lm_ce.get("A1") > lm_ce.get("A2") > lm_ce.get("A3") > lm_ce.get("A4")
            if all(k in lm_ce for k in ["A1", "A2", "A3", "A4"]) else None
        ),
        "tf_ce_monotonic_1_2_3_4": (
            out["teacher_forced"]["depth_1"]["token_weighted_ce"]
            > out["teacher_forced"]["depth_2"]["token_weighted_ce"]
            > out["teacher_forced"]["depth_3"]["token_weighted_ce"]
            > out["teacher_forced"]["depth_4"]["token_weighted_ce"]
            if all(
                out["teacher_forced"][f"depth_{d}"]["token_weighted_ce"] is not None
                for d in [1, 2, 3, 4]
            ) else None
        ),
        "syntax_depth1_minus_depth4": (
            (out["mbpp_validation"]["depth_1"]["syntax_valid_rate"] or 0.0)
            - (out["mbpp_validation"]["depth_4"]["syntax_valid_rate"] or 0.0)
        ),
        "syntax_depth4_ge_depth1_minus_10pp": (
            out["mbpp_validation"]["depth_4"]["syntax_valid_rate"] is not None
            and out["mbpp_validation"]["depth_1"]["syntax_valid_rate"] is not None
            and out["mbpp_validation"]["depth_4"]["syntax_valid_rate"]
            >= out["mbpp_validation"]["depth_1"]["syntax_valid_rate"] - 0.10
        ),
        "max_pass_at_1": max(
            (out["mbpp_validation"][f"depth_{d}"]["pass_at_1"] or 0.0)
            for d in [1, 2, 3, 4]
        ),
        "b4_both_equals_a1": (
            out["trace"]["broken"].get("B4_both", {}).get("max_abs") == 0.0
            if out["trace"]["broken"].get("B4_both") else None
        ),
        "b4_both_equals_b2_both": (
            out["trace"]["broken"].get("B4_both_vs_B2_both", {}).get("max_abs") == 0.0
            if out["trace"]["broken"].get("B4_both_vs_B2_both") else None
        ),
    }
    return out


def main() -> None:
    args = parse_args()
    candidates = []
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        candidates.append((name, path))
    results = {
        "baseline": summarize(args.baseline),
        "candidates": {},
    }
    for name, path in candidates:
        if not os.path.exists(path):
            print(f"[warn] missing {name}: {path}")
            continue
        results["candidates"][name] = summarize(path)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    rows = [("baseline", results["baseline"])] + list(results["candidates"].items())
    print("| Run | Val syntax d1/d2/d3/d4 | First d1/d2/d3/d4 | pass@1 max | LM CE d1/d2/d3/d4 | TF CE d1/d2/d3/d4 | TF 1->2 / 2->3 / 3->4 improved |")
    print("|---|---|---|---|---|---|---|")
    for name, r in rows:
        syn = "/".join(
            f"{100*(r['mbpp_validation'][f'depth_{d}']['syntax_valid_rate'] or 0):.1f}"
            for d in [1, 2, 3, 4]
        )
        first = "/".join(
            f"{100*(r['mbpp_validation'][f'depth_{d}']['first_test_pass_rate'] or 0):.1f}"
            for d in [1, 2, 3, 4]
        )
        lm = "/".join(
            f"{r['language_modeling'].get(f'A{d}', float('nan')):.4f}" for d in [1, 2, 3, 4]
        )
        tf = "/".join(
            f"{r['teacher_forced'][f'depth_{d}']['token_weighted_ce'] or float('nan'):.4f}"
            for d in [1, 2, 3, 4]
        )
        imp = "/".join(
            f"{100*(r['adjacent'][p]['fraction_improved'] or 0):.1f}"
            for p in ["1_to_2", "2_to_3", "3_to_4"]
        )
        print(
            f"| {name} | {syn} | {first} | {100*r['derived']['max_pass_at_1']:.1f}% | "
            f"{lm} | {tf} | {imp} |"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
