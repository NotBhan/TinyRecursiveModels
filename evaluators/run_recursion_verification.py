"""Run the Phase-8 recurrence verification.

Example:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. \
  .venv-trm-lm/bin/python -m evaluators.run_recursion_verification \
    --checkpoint experiments/re_coding_adaptive/stdlib_instr_1m/checkpoint_final.pt \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --tinystories-workdir experiments/trm_lm_pilot_5m/r1 \
    --output-dir experiments/trm_lm_coding_reasoning/recursion_verification \
    --val-limit 45 --test-limit 100

The driver never writes the checkpoint, never creates an optimizer, and runs
every forward call under ``torch.no_grad()``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.code_pipeline import (
    CodePipelineConfig,
    MBPPInstructionDataset,
    _load_mbpp_split,
    build_mbpp_eval_records,
    normalize_for_dedup,
)
from dataset.tinystories_lm import (
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    load_tokenizer,
)
from evaluators.re_eval import build_signature_prompt, parameter_bytes
from evaluators.recurrence_verification import (
    InterventionConfig,
    _sha256,
    aggregate_intervention_rows,
    evaluate_mbpp_intervention,
    evaluate_tinystories_depths,
    generate_with_recurrence,
    logit_difference,
    paired_bootstrap_ce,
    paired_status,
    pair_stats,
    run_independent_ensemble,
    run_recurrence,
    state_stats,
)
from models.recursive_reasoning.trm_lm_adaptive import (
    STOP_REASON_NAMES,
    DualStateConfig,
    DualStateModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-workdir", required=True)
    parser.add_argument("--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1")
    parser.add_argument("--mbpp-dir", default="data/coding/mbpp")
    parser.add_argument("--output-dir", default="experiments/trm_lm_coding_reasoning/recursion_verification")
    parser.add_argument("--val-limit", type=int, default=45)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-cap", type=int, default=256)
    parser.add_argument("--fixed-seq-len", type=int, default=64)
    parser.add_argument("--lm-max-docs", type=int, default=256)
    parser.add_argument("--sandbox-python", default="/usr/bin/python3.14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-lm", action="store_true")
    return parser.parse_args()


def load_trained(args: argparse.Namespace):
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
        raise ValueError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {model.config.vocab_size}"
        )
    return model, tokenizer, ckpt, device, dtype


def build_fixed_batch(records: Sequence[Dict[str, Any]], tokenizer: Any, seq_len: int, n: int = 4):
    bos = tokenizer.token_to_id("<bos>")
    pad = tokenizer.token_to_id("<pad>")
    input_rows = []
    mask_rows = []
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


def run_integrity(model, ckpt, tokenizer, fixed_inp, fixed_mask, args, sha_before):
    parity = []
    for k in [1, 2, model.config.max_high_cycles]:
        ref = model(fixed_inp, fixed_mask, mode="fixed", n_high_cycles=k).logits
        man = run_recurrence(
            model, fixed_inp, fixed_mask, n_high_cycles=k,
            reset_mode="none", last_token_only=False, record_states=False,
        )["final_logits"]
        parity.append({"n_high_cycles": k, **logit_difference(ref, man)})
        if parity[-1]["max_abs"] > 1e-4:
            raise RuntimeError(f"normal path parity failed at depth {k}: {parity[-1]}")

    # Generation parity against the existing generation path.
    generation_parity = []
    records = build_mbpp_eval_records(CodePipelineConfig(mbpp_dir=args.mbpp_dir))["validation"][:3]
    for record in records:
        prompt, _, _ = build_signature_prompt(record)
        bos = tokenizer.token_to_id("<bos>")
        ids = [bos] + tokenizer.encode(prompt).ids
        ids = ids[: model.config.seq_len - 16]
        inp = torch.tensor([ids], dtype=torch.int64, device=fixed_inp.device)
        mask = torch.ones_like(inp)
        ref = model.generate_with_diagnostics(
            inp, max_new_tokens=16, eos_token_id=tokenizer.token_to_id("<eos>"),
            mode="fixed", n_high_cycles=4,
        )
        man = generate_with_recurrence(
            model, tokenizer, prompt, n_high_cycles=4, reset_mode="none",
            max_new_tokens=16, context_cap=256,
        )
        ref_tokens = ref.tokens[0, len(ids):].detach().cpu().tolist()
        generation_parity.append(
            {
                "task_id": record.get("task_id"),
                "reference_tokens": ref_tokens,
                "manual_tokens": man["token_ids"],
                "exact_match": ref_tokens == man["token_ids"],
            }
        )

    no_grad = all(p.grad is None for p in model.parameters())
    sha_after = _sha256(args.checkpoint)

    # Provenance: the checkpoint was trained on MBPP train only, with the
    # documented validation/test quarantine.
    mbpp_config = ckpt.get("train_config", {}).get("mbpp_config", "full")
    code_cfg = CodePipelineConfig(
        seq_len=model.config.seq_len, mbpp_dir=args.mbpp_dir, mbpp_config=mbpp_config
    )
    train_ds = MBPPInstructionDataset(tokenizer, code_cfg, split="train")
    train_records = train_ds.records
    val_records = _load_mbpp_split(args.mbpp_dir, mbpp_config, "validation")
    test_records = _load_mbpp_split(args.mbpp_dir, mbpp_config, "test")
    train_text = {normalize_for_dedup(r["problem_text"]) for r in train_records}
    train_code = {r["code"].strip() for r in train_records}
    val_text = {normalize_for_dedup(r["problem_text"]) for r in val_records}
    val_code = {r["code"].strip() for r in val_records}
    test_text = {normalize_for_dedup(r["problem_text"]) for r in test_records}
    test_code = {r["code"].strip() for r in test_records}
    provenance = {
        "train_examples_retained": len(train_records),
        "train_quarantined": int(train_ds.quarantined),
        "train_val_text_overlap": len(train_text & val_text),
        "train_val_code_overlap": len(train_code & val_code),
        "train_test_text_overlap": len(train_text & test_text),
        "train_test_code_overlap": len(train_code & test_code),
        "checkpoint_train_config": ckpt.get("train_config", {}),
        "checkpoint_mixture_token_accounting": ckpt.get("mixture_token_accounting", {}),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_tokens_processed": ckpt.get("tokens_processed"),
    }
    return {
        "checkpoint_sha256_before": sha_before,
        "checkpoint_sha256_after": sha_after,
        "checkpoint_hash_unchanged": sha_before == sha_after,
        "normal_path_parity": parity,
        "generation_parity": generation_parity,
        "generation_exact_match_all": all(g["exact_match"] for g in generation_parity),
        "no_gradients": no_grad,
        "no_optimizer_created": True,
        "provenance": provenance,
    }


def summarize_trace(out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_high_cycles": out["n_high_cycles"],
        "reset_mode": out["reset_mode"],
        "block_calls": out["block_calls"],
        "initial": out["initial"],
        "trace": out["trace"],
        "perturbation": out["perturbation"],
        "final_z_H_norm": float(out["z_H"].detach().float().norm().item()),
        "final_z_L_norm": float(out["z_L"].detach().float().norm().item()),
        "final_logits_norm": float(out["final_logits"].detach().float().norm().item()),
    }


def run_trace(model, fixed_inp, fixed_mask, output_dir):
    configs = {
        "A1_normal": dict(n_high_cycles=1, reset_mode="none"),
        "A4_normal": dict(n_high_cycles=4, reset_mode="none"),
        "B4_broken_both": dict(n_high_cycles=4, reset_mode="both"),
        "B4_broken_high": dict(n_high_cycles=4, reset_mode="high"),
        "B4_broken_low": dict(n_high_cycles=4, reset_mode="low"),
    }
    runs = {}
    tensors = {}
    for name, kwargs in configs.items():
        out = run_recurrence(
            model, fixed_inp, fixed_mask,
            last_token_only=True, record_states=True, record_controller=True,
            return_state_tensors=True, **kwargs,
        )
        runs[name] = summarize_trace(out)
        tensors[name] = out["state_tensors"]
    comparisons = {
        "A4_vs_forward_fixed4": logit_difference(
            model(fixed_inp, fixed_mask, mode="fixed", n_high_cycles=4, last_token_only=True).logits,
            run_recurrence(model, fixed_inp, fixed_mask, 4, "none", last_token_only=True)["final_logits"],
        ),
        "A1_vs_A4": logit_difference(
            run_recurrence(model, fixed_inp, fixed_mask, 1, "none", last_token_only=True)["final_logits"],
            run_recurrence(model, fixed_inp, fixed_mask, 4, "none", last_token_only=True)["final_logits"],
        ),
        "B4_both_vs_A1": logit_difference(
            run_recurrence(model, fixed_inp, fixed_mask, 1, "none", last_token_only=True)["final_logits"],
            run_recurrence(model, fixed_inp, fixed_mask, 4, "both", last_token_only=True)["final_logits"],
        ),
        "B4_high_vs_A4": logit_difference(
            run_recurrence(model, fixed_inp, fixed_mask, 4, "none", last_token_only=True)["final_logits"],
            run_recurrence(model, fixed_inp, fixed_mask, 4, "high", last_token_only=True)["final_logits"],
        ),
        "B4_low_vs_A4": logit_difference(
            run_recurrence(model, fixed_inp, fixed_mask, 4, "none", last_token_only=True)["final_logits"],
            run_recurrence(model, fixed_inp, fixed_mask, 4, "low", last_token_only=True)["final_logits"],
        ),
    }
    trace_path = os.path.join(output_dir, "recurrence_trace_tensors.pt")
    torch.save(tensors, trace_path)
    return {"runs": runs, "comparisons": comparisons, "state_tensor_path": trace_path}


def run_sensitivity(model, fixed_inp, fixed_mask):
    clean = run_recurrence(
        model, fixed_inp, fixed_mask, 4, "none",
        last_token_only=True, record_states=True, record_controller=True,
    )
    rows = []
    for target in ["z_H", "z_L", "both"]:
        for depth in [1, 2]:
            for scale in [0.0, 0.02, 0.05, 0.1]:
                seeds = [0] if scale == 0.0 else [0, 1]
                for seed in seeds:
                    pert = run_recurrence(
                        model, fixed_inp, fixed_mask, 4, "none",
                        last_token_only=True, record_states=True, record_controller=True,
                        perturb_after_depth=depth, perturb_target=target,
                        perturb_scale=scale, perturb_seed=seed, perturb_direction="random",
                    )
                    diff = logit_difference(clean["final_logits"], pert["final_logits"])
                    state_h = pair_stats(clean["z_H"], pert["z_H"])
                    state_l = pair_stats(clean["z_L"], pert["z_L"])
                    ctrl_delta = None
                    if depth < 3:
                        # Controller call at step depth+1 is trace index depth.
                        c0 = clean["trace"][depth]["controller"]
                        c1 = pert["trace"][depth]["controller"]
                        ctrl_delta = {
                            "prob_mean_abs_delta": abs(c0["prob_mean"] - c1["prob_mean"]),
                            "logit_mean_abs_delta": abs(c0["logit_mean"] - c1["logit_mean"]),
                        }
                    realized_h = pert["perturbation"]["z_H"]["realized_rel_mean"] if pert["perturbation"] else 0.0
                    realized_l = pert["perturbation"]["z_L"]["realized_rel_mean"] if pert["perturbation"] else 0.0
                    if scale == 0.0:
                        if diff["max_abs"] != 0.0:
                            raise RuntimeError("zero-perturbation control changed the output")
                    rows.append(
                        {
                            "target": target,
                            "after_depth": depth,
                            "scale": scale,
                            "seed": seed,
                            "realized_rel_z_H": realized_h,
                            "realized_rel_z_L": realized_l,
                            "final_logit": diff,
                            "final_z_H_change": state_h,
                            "final_z_L_change": state_l,
                            "controller_delta": ctrl_delta,
                        }
                    )
    # Monotonicity of the max logit change with scale.
    monotonic = []
    for target in ["z_H", "z_L", "both"]:
        for depth in [1, 2]:
            values = [r["final_logit"]["max_abs"] for r in rows if r["target"] == target and r["after_depth"] == depth and r["seed"] == 0]
            monotonic.append(
                {
                    "target": target,
                    "after_depth": depth,
                    "scales": [0.0, 0.02, 0.05, 0.1],
                    "max_abs_values": values,
                    "non_decreasing": all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1)),
                }
            )
    return {"rows": rows, "monotonicity": monotonic}


def run_adaptive_trace(model, fixed_inp, fixed_mask):
    fast = model(fixed_inp, fixed_mask, mode="fast", last_token_only=True)
    fixed4 = model(fixed_inp, fixed_mask, mode="fixed", n_high_cycles=4, last_token_only=True)
    adaptive = model(fixed_inp, fixed_mask, mode="adaptive", max_high_cycles=4, last_token_only=True)
    real = fixed_mask.bool()
    reasons: Dict[str, int] = {}
    codes = adaptive.stop_reason_codes[real].detach().cpu().tolist()
    for code in codes:
        name = STOP_REASON_NAMES.get(int(code), str(int(code)))
        reasons[name] = reasons.get(name, 0) + 1
    hist = torch.bincount(
        adaptive.high_cycles_per_position[real].detach().cpu().long(), minlength=5
    ).tolist()
    halt_summaries = []
    for q in adaptive.halt_probs:
        halt_summaries.append(
            {
                "mean": float(q.mean().item()),
                "min": float(q.min().item()),
                "max": float(q.max().item()),
                "above_half": float((q >= 0.5).float().mean().item()),
            }
        )
    return {
        "active_counts_per_step": adaptive.active_counts_per_step,
        "mean_high_cycles": float(adaptive.high_cycles.float().mean().item()),
        "effective_depth_histogram": {str(i): int(v) for i, v in enumerate(hist)},
        "stop_reason_counts": reasons,
        "halt_prob_summaries": halt_summaries,
        "adaptive_vs_fast": logit_difference(fast.logits, adaptive.logits),
        "adaptive_vs_fixed4": logit_difference(fixed4.logits, adaptive.logits),
    }


def run_mbpp(model, tokenizer, args, val_records, test_records):
    configs = [
        InterventionConfig("A1_normal_d1", 1, "none", "normal recurrence, 1 cycle"),
        InterventionConfig("A2_normal_d2", 2, "none", "normal recurrence, 2 cycles"),
        InterventionConfig("A4_normal_d4", 4, "none", "normal recurrence, 4 cycles"),
        InterventionConfig("B4_broken_both", 4, "both", "both latents reset to initial each cycle"),
        InterventionConfig("B4_broken_high", 4, "high", "z_H reset each cycle; z_L carries"),
        InterventionConfig("B4_broken_low", 4, "low", "z_L reset each cycle; z_H carries"),
        InterventionConfig("C4_repeated_identical", 4, "none", "4 identical one-step evaluations averaged", ensemble_passes=4, ensemble_noise=0.0),
        InterventionConfig("C4_independent_noisy", 4, "none", "4 independent one-step evaluations with 0.02 initial-state noise averaged", ensemble_passes=4, ensemble_noise=0.02),
    ]
    val = {}
    val_rows = {}
    for cfg in configs:
        agg, rows = evaluate_mbpp_intervention(
            model, tokenizer, val_records, cfg,
            max_new_tokens=args.max_new_tokens, context_cap=args.context_cap,
            limit=args.val_limit, sandbox_python=args.sandbox_python,
        )
        val[cfg.name] = {"config": cfg.__dict__, "aggregate": agg}
        val_rows[cfg.name] = rows
        print(f"[val:{cfg.name}] syntax={agg['syntax_valid_rate']:.1%} pass={agg['pass_at_1']:.1%} first={agg['first_test_pass_rate']:.1%}", flush=True)
    test = {}
    test_rows = {}
    if not args.skip_test:
        test_configs = [
            InterventionConfig("A1_normal_d1", 1, "none"),
            InterventionConfig("A2_normal_d2", 2, "none"),
            InterventionConfig("A4_normal_d4", 4, "none"),
            InterventionConfig("B4_broken_both", 4, "both"),
            InterventionConfig("C4_repeated_identical", 4, "none", ensemble_passes=4, ensemble_noise=0.0),
        ]
        for cfg in test_configs:
            agg, rows = evaluate_mbpp_intervention(
                model, tokenizer, test_records, cfg,
                max_new_tokens=args.max_new_tokens, context_cap=args.context_cap,
                limit=args.test_limit, sandbox_python=args.sandbox_python,
            )
            test[cfg.name] = {"config": cfg.__dict__, "aggregate": agg}
            test_rows[cfg.name] = rows
            print(f"[test:{cfg.name}] syntax={agg['syntax_valid_rate']:.1%} pass={agg['pass_at_1']:.1%} first={agg['first_test_pass_rate']:.1%}", flush=True)

    def paired(rows_a, rows_b):
        return {
            "syntax": paired_status(rows_a, rows_b, "syntax_valid"),
            "first_test_passed": paired_status(rows_a, rows_b, "first_test_passed"),
            "passed": paired_status(rows_a, rows_b, "passed"),
        }

    pairs = {}
    for name_a, name_b in [
        ("A1_normal_d1", "A4_normal_d4"),
        ("A1_normal_d1", "B4_broken_both"),
        ("A1_normal_d1", "C4_repeated_identical"),
        ("A1_normal_d1", "C4_independent_noisy"),
        ("A2_normal_d2", "A4_normal_d4"),
    ]:
        if name_a in val_rows and name_b in val_rows:
            pairs[f"{name_a}_vs_{name_b}"] = paired(val_rows[name_a], val_rows[name_b])
    test_pairs = {}
    if test_rows:
        for name_a, name_b in [
            ("A1_normal_d1", "A4_normal_d4"),
            ("A1_normal_d1", "B4_broken_both"),
            ("A1_normal_d1", "C4_repeated_identical"),
        ]:
            if name_a in test_rows and name_b in test_rows:
                test_pairs[f"{name_a}_vs_{name_b}"] = paired(test_rows[name_a], test_rows[name_b])
    return {
        "validation": val,
        "validation_paired": pairs,
        "test": test,
        "test_paired": test_pairs,
        "validation_rows": val_rows,
        "test_rows": test_rows,
    }


def run_lm(model, tokenizer, args, device):
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
    configs = [
        InterventionConfig("A1", 1, "none"),
        InterventionConfig("A2", 2, "none"),
        InterventionConfig("A4", 4, "none"),
        InterventionConfig("B4_both", 4, "both"),
    ]
    results = evaluate_tinystories_depths(model, loader, device, configs)
    results["A4_vs_B4_both"] = paired_bootstrap_ce(
        results["A4"]["per_batch_ce"], results["B4_both"]["per_batch_ce"]
    )
    results["A2_vs_A4"] = paired_bootstrap_ce(
        results["A2"]["per_batch_ce"], results["A4"]["per_batch_ce"]
    )
    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model, tokenizer, ckpt, device, dtype = load_trained(args)
    sha_before = _sha256(args.checkpoint)
    print(f"[init] checkpoint={args.checkpoint} sha256={sha_before[:16]}... params={model.num_parameters:,} weights={parameter_bytes(model)/1e6:.2f} MB", flush=True)

    code_cfg = CodePipelineConfig(seq_len=model.config.seq_len, mbpp_dir=args.mbpp_dir)
    all_records = build_mbpp_eval_records(code_cfg)
    fixed_inp, fixed_mask = build_fixed_batch(
        all_records["validation"], tokenizer, args.fixed_seq_len, n=4
    )
    fixed_inp = fixed_inp.to(device)
    fixed_mask = fixed_mask.to(device)

    print("[1/5] integrity and parity", flush=True)
    integrity = run_integrity(
        model, ckpt, tokenizer, fixed_inp, fixed_mask, args, sha_before
    )
    print(f"  parity={integrity['normal_path_parity']} generation_match={integrity['generation_exact_match_all']} no_grad={integrity['no_gradients']} hash_unchanged={integrity['checkpoint_hash_unchanged']}", flush=True)

    print("[2/5] state/intervention trace", flush=True)
    trace = run_trace(model, fixed_inp, fixed_mask, args.output_dir)
    print(f"  comparisons={json.dumps(trace['comparisons'])}", flush=True)

    print("[3/5] causal sensitivity", flush=True)
    sensitivity = run_sensitivity(model, fixed_inp, fixed_mask)
    print(f"  rows={len(sensitivity['rows'])}", flush=True)

    print("[4/5] adaptive trace", flush=True)
    adaptive = run_adaptive_trace(model, fixed_inp, fixed_mask)

    print("[5/5] MBPP interventions", flush=True)
    mbpp = run_mbpp(
        model, tokenizer, args,
        all_records["validation"], all_records["test"],
    )
    lm = {}
    if not args.skip_lm:
        print("[lm] TinyStories depth CE", flush=True)
        lm = run_lm(model, tokenizer, args, device)

    sha_after = _sha256(args.checkpoint)
    integrity["checkpoint_sha256_after_final"] = sha_after
    integrity["checkpoint_hash_unchanged_final"] = sha_before == sha_after

    results = {
        "schema": "re_recursion_verification_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha_before,
        "model_parameters": model.num_parameters,
        "weights_mb": parameter_bytes(model) / 1e6,
        "device": str(device),
        "dtype": str(dtype),
        "args": vars(args),
        "fixed_batch": {
            "shape": list(fixed_inp.shape),
            "seq_len": args.fixed_seq_len,
            "n_sequences": int(fixed_inp.shape[0]),
        },
        "integrity": integrity,
        "trace": trace,
        "sensitivity": sensitivity,
        "adaptive": adaptive,
        "mbpp": mbpp,
        "language_modeling": lm,
    }
    out_json = os.path.join(args.output_dir, "recursion_verification_results.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    with open(os.path.join(args.output_dir, "reproduction_command.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. .venv-trm-lm/bin/python "
            "-m evaluators.run_recursion_verification "
            f"--checkpoint {args.checkpoint} "
            f"--tokenizer-workdir {args.tokenizer_workdir} "
            f"--tinystories-workdir {args.tinystories_workdir} "
            f"--output-dir {args.output_dir} "
            f"--val-limit {args.val_limit} --test-limit {args.test_limit}\n"
        )
    print(f"[done] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
