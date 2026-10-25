"""Aggregate multi-seed refinement evaluations into one machine-readable file.

Each input is a ``baseline_results.json`` produced by
``evaluators.run_refinement_baseline``. Candidate and control runs are supplied
as repeated ``--candidate NAME=PATH`` / ``--control NAME=PATH`` arguments; the
seed is read from ``args.seed`` inside each evaluation file.

Example:
  PYTHONPATH=. .venv-trm-lm/bin/python -m evaluators.aggregate_refinement_results \
    --baseline experiments/re_refinement_monotonicity/baseline/baseline_results.json \
    --candidate experiments/re_refinement_monotonicity/final/A1_residual_seed1337/eval/baseline_results.json \
    --candidate experiments/re_refinement_monotonicity/final/A1_residual_seed2025/eval/baseline_results.json \
    --candidate experiments/re_refinement_monotonicity/final/A1_residual_seed3407/eval/baseline_results.json \
    --control experiments/re_refinement_monotonicity/final/control_seed1337/eval/baseline_results.json \
    --control experiments/re_refinement_monotonicity/final/control_seed2025/eval/baseline_results.json \
    --control experiments/re_refinement_monotonicity/final/control_seed3407/eval/baseline_results.json \
    --split validation \
    --output experiments/re_refinement_monotonicity/DEPTH_QUALITY_RESULTS.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from typing import Any, Dict, List, Optional


DEPTHS = [1, 2, 3, 4]
PAIRS = ["1_to_2", "2_to_3", "3_to_4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return float(statistics.fmean(clean)) if clean else None


def _std(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return float(statistics.stdev(clean)) if len(clean) > 1 else 0.0 if clean else None


def _ci95(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    # t critical at 95% for df = n - 1; only small n is used here.
    t_critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(
        len(clean) - 1, 1.96
    )
    return float(t_critical * statistics.stdev(clean) / math.sqrt(len(clean)))


def _bootstrap_ci(values: List[float], seed: int = 0, n: int = 10_000) -> Optional[List[float]]:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    size = len(values)
    for _ in range(n):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def _paired_stats(
    deltas: List[float], weights: List[float], seed: int
) -> Dict[str, Any]:
    """Positive delta means the later depth has lower CE (is better)."""
    clean = [(d, w) for d, w in zip(deltas, weights) if d is not None]
    values = [d for d, _ in clean]
    wsum = sum(w for _, w in clean)
    return {
        "n_pairs": len(values),
        "mean_delta": _mean(values),
        "weighted_mean_delta": (
            sum(d * w for d, w in clean) / wsum if wsum > 0 else None
        ),
        "median_delta": float(statistics.median(values)) if values else None,
        "fraction_improved": (
            float(sum(v > 0 for v in values) / len(values)) if values else None
        ),
        "fraction_degraded": (
            float(sum(v <= 0 for v in values) / len(values)) if values else None
        ),
        "t95_halfwidth": _ci95(values),
        "bootstrap95": _bootstrap_ci(values, seed=seed),
        "deltas": values,
    }


def paired_depth_stats(data: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"lm": {}, "teacher_forced": {}}
    lm = data.get("language_modeling", {})
    if all(f"A{d}" in lm and lm[f"A{d}"].get("per_batch_ce") for d in DEPTHS):
        for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
            rows_a = lm[f"A{a}"]["per_batch_ce"]
            rows_b = lm[f"A{b}"]["per_batch_ce"]
            n = min(len(rows_a), len(rows_b))
            deltas = [rows_a[i]["ce"] - rows_b[i]["ce"] for i in range(n)]
            weights = [
                (rows_a[i].get("tokens", 0) + rows_b[i].get("tokens", 0)) / 2.0
                for i in range(n)
            ]
            out["lm"][f"{a}_to_{b}"] = _paired_stats(deltas, weights, seed)
    per_task = data.get("teacher_forced_mbpp", {}).get("per_task", [])
    if per_task:
        for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
            deltas = [
                row["ce"][str(a)] - row["ce"][str(b)]
                for row in per_task
                if str(a) in row.get("ce", {}) and str(b) in row.get("ce", {})
            ]
            weights = [
                row.get("valid_target_tokens", 0)
                for row in per_task
                if str(a) in row.get("ce", {}) and str(b) in row.get("ce", {})
            ]
            out["teacher_forced"][f"{a}_to_{b}"] = _paired_stats(deltas, weights, seed)
    return out


def summarize(path: str, split: str) -> Dict[str, Any]:
    data = json.load(open(path, "r", encoding="utf-8"))
    block = data.get("mbpp_validation" if split == "validation" else "mbpp_test", {})
    aggregates = block.get("aggregates", {})
    tf = data.get("teacher_forced_mbpp", {})
    lm = data.get("language_modeling", {})
    trace = data.get("trace", {})

    depths: Dict[str, Any] = {}
    for depth in DEPTHS:
        agg = aggregates.get(f"depth_{depth}", {})
        depths[f"depth_{depth}"] = {
            "syntax_valid_rate": agg.get("syntax_valid_rate"),
            "pass_at_1": agg.get("pass_at_1"),
            "first_test_pass_rate": agg.get("first_test_pass_rate"),
            "entry_point_present_rate": agg.get("entry_point_present_rate"),
            "error_counts": agg.get("error_counts"),
            "teacher_forced_ce": tf.get("summary", {})
            .get(f"depth_{depth}", {})
            .get("token_weighted_ce"),
            "language_ce": (lm.get(f"A{depth}") or {}).get("ce"),
        }

    adjacent: Dict[str, Any] = {}
    for pair in PAIRS:
        row = tf.get("adjacent_summary", {}).get(pair, {})
        adjacent[pair] = {
            "mean_delta_ce": row.get("mean_delta_ce"),
            "fraction_improved": row.get("fraction_improved"),
            "fraction_degraded": row.get("fraction_degraded"),
            "mean_top1_change_rate": row.get("mean_top1_change_rate"),
        }

    lm_ce = {f"A{d}": (lm.get(f"A{d}") or {}).get("ce") for d in DEPTHS}
    tf_ce = {f"depth_{d}": depths[f"depth_{d}"]["teacher_forced_ce"] for d in DEPTHS}
    ordered_lm = [lm_ce[f"A{d}"] for d in DEPTHS]
    ordered_tf = [tf_ce[f"depth_{d}"] for d in DEPTHS]
    seed = data.get("args", {}).get("seed")
    if seed is None:
        match = re.search(r"seed[_-]?(\d+)", path)
        seed = int(match.group(1)) if match else None
    return {
        "path": path,
        "seed": seed,
        "checkpoint": data.get("checkpoint"),
        "checkpoint_sha256": data.get("checkpoint_sha256"),
        "depths": depths,
        "adjacent": adjacent,
        "language_modeling": lm_ce,
        "teacher_forced": tf_ce,
        "paired_depth": paired_depth_stats(data, seed=int(seed or 0)),
        "derived": {
            "lm_depth4_minus_depth1": (
                ordered_lm[3] - ordered_lm[0]
                if ordered_lm[0] is not None and ordered_lm[3] is not None
                else None
            ),
            "lm_monotonic_1_2_3_4": (
                all(a > b for a, b in zip(ordered_lm, ordered_lm[1:]))
                if all(v is not None for v in ordered_lm) else None
            ),
            "tf_monotonic_1_2_3_4": (
                all(a > b for a, b in zip(ordered_tf, ordered_tf[1:]))
                if all(v is not None for v in ordered_tf) else None
            ),
            "syntax_depth1_minus_depth4": (
                depths["depth_1"]["syntax_valid_rate"]
                - depths["depth_4"]["syntax_valid_rate"]
                if depths["depth_1"]["syntax_valid_rate"] is not None
                and depths["depth_4"]["syntax_valid_rate"] is not None
                else None
            ),
            "max_pass_at_1": max(
                (depths[f"depth_{d}"]["pass_at_1"] or 0.0) for d in DEPTHS
            ),
            "normal_parity_zero": all(
                (trace.get("comparisons", {}).get(key, {}).get("max_abs") is not None)
                for key in ["A1_vs_A2"]
            ),
            "broken": trace.get("broken"),
            "comparisons": trace.get("comparisons"),
        },
    }


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"runs": len(runs), "seeds": [r["seed"] for r in runs], "depths": {}}
    for depth in DEPTHS:
        key = f"depth_{depth}"
        out["depths"][key] = {}
        for metric in [
            "syntax_valid_rate",
            "pass_at_1",
            "first_test_pass_rate",
            "teacher_forced_ce",
            "language_ce",
        ]:
            values = [r["depths"][key][metric] for r in runs]
            out["depths"][key][metric] = {
                "mean": _mean(values),
                "std": _std(values),
                "ci95": _ci95(values),
                "min": min((v for v in values if v is not None), default=None),
                "max": max((v for v in values if v is not None), default=None),
                "values": values,
            }
    out["adjacent"] = {}
    for pair in PAIRS:
        out["adjacent"][pair] = {}
        for metric in ["fraction_improved", "fraction_degraded", "mean_delta_ce"]:
            values = [r["adjacent"][pair][metric] for r in runs]
            out["adjacent"][pair][metric] = {
                "mean": _mean(values),
                "std": _std(values),
                "values": values,
            }
    out["lm_monotonic_all_seeds"] = all(
        r["derived"]["lm_monotonic_1_2_3_4"] for r in runs
    ) if all(r["derived"]["lm_monotonic_1_2_3_4"] is not None for r in runs) else None
    out["tf_monotonic_all_seeds"] = all(
        r["derived"]["tf_monotonic_1_2_3_4"] for r in runs
    ) if all(r["derived"]["tf_monotonic_1_2_3_4"] is not None for r in runs) else None
    out["max_pass_at_1"] = max((r["derived"]["max_pass_at_1"] or 0.0) for r in runs)
    out["paired_depth"] = {}
    for family in ["lm", "teacher_forced"]:
        out["paired_depth"][family] = {}
        pairs = sorted({pair for r in runs for pair in r.get("paired_depth", {}).get(family, {})})
        for pair in pairs:
            per_seed = [
                r["paired_depth"][family][pair]["mean_delta"]
                for r in runs
                if pair in r.get("paired_depth", {}).get(family, {})
            ]
            raw = [
                d
                for r in runs
                for d in r.get("paired_depth", {}).get(family, {}).get(pair, {}).get("deltas", [])
            ]
            out["paired_depth"][family][pair] = {
                "per_seed_mean_delta": per_seed,
                "mean_of_seed_means": _mean(per_seed),
                "min_seed_mean": min((v for v in per_seed if v is not None), default=None),
                "max_seed_mean": max((v for v in per_seed if v is not None), default=None),
                "pooled_mean_delta": _mean(raw),
                "pooled_bootstrap95": _bootstrap_ci(raw, seed=0),
                "pooled_n_pairs": len(raw),
            }
    return out


def paired(candidates: List[Dict[str, Any]], controls: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed_candidate = {r["seed"]: r for r in candidates}
    by_seed_control = {r["seed"]: r for r in controls}
    shared = sorted(set(by_seed_candidate) & set(by_seed_control))
    out: Dict[str, Any] = {"shared_seeds": shared, "depths": {}}
    for depth in DEPTHS:
        key = f"depth_{depth}"
        out["depths"][key] = {}
        for metric in ["syntax_valid_rate", "first_test_pass_rate", "pass_at_1",
                       "teacher_forced_ce", "language_ce"]:
            deltas = []
            for seed in shared:
                c = by_seed_candidate[seed]["depths"][key][metric]
                b = by_seed_control[seed]["depths"][key][metric]
                deltas.append(None if c is None or b is None else c - b)
            out["depths"][key][metric] = {
                "mean_delta": _mean(deltas),
                "std": _std(deltas),
                "ci95": _ci95(deltas),
                "values": deltas,
            }
    return out


def print_table(name: str, summary: Dict[str, Any]) -> None:
    syn = "/".join(
        f"{100 * (summary['depths'][f'depth_{d}']['syntax_valid_rate']['mean'] or 0):.1f}"
        for d in DEPTHS
    )
    lm = "/".join(
        f"{summary['depths'][f'depth_{d}']['language_ce']['mean'] or float('nan'):.4f}"
        for d in DEPTHS
    )
    tf = "/".join(
        f"{summary['depths'][f'depth_{d}']['teacher_forced_ce']['mean'] or float('nan'):.4f}"
        for d in DEPTHS
    )
    print(f"{name:24s} syntax {syn}  LM {lm}  TF {tf}")


def main() -> None:
    args = parse_args()
    baseline = summarize(args.baseline, args.split)

    def load_all(specs: List[str]) -> List[Dict[str, Any]]:
        return [summarize(spec.split("=", 1)[-1], args.split) for spec in specs]

    candidates = load_all(args.candidate)
    controls = load_all(args.control)
    result = {
        "schema": "re_refinement_monotonicity_depth_quality_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "baseline": baseline,
        "arms": {
            "A1_residual": {"runs": candidates, "aggregate": aggregate(candidates)},
            "control": {"runs": controls, "aggregate": aggregate(controls)},
        },
        "paired_candidate_minus_control": paired(candidates, controls),
    }
    json.dump(result, open(args.output, "w", encoding="utf-8"), indent=2)
    print(f"split={args.split}")
    print_table("baseline", {"depths": {
        f"depth_{d}": {
            "syntax_valid_rate": {"mean": baseline["depths"][f"depth_{d}"]["syntax_valid_rate"]},
            "language_ce": {"mean": baseline["depths"][f"depth_{d}"]["language_ce"]},
            "teacher_forced_ce": {"mean": baseline["depths"][f"depth_{d}"]["teacher_forced_ce"]},
        } for d in DEPTHS
    }})
    for name in ["A1_residual", "control"]:
        print_table(name, result["arms"][name]["aggregate"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
