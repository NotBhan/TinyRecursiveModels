"""Training and preflight entry point for Language Modeling vs Coding Reasoning.

Implements:
- 60/40 mixed training of TinyStories and MBPP.
- Per-position causal adaptive computation.
- Survivor-aware controller supervision to resolve training/deployment mismatch.
- Execution-based function synthesis evaluation with safe sandboxing.
- Full provenance recording, checkpointing, and reload verification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.coding_reasoning import (
    CodingReasoningConfig,
    CodingReasoningDataset,
    MBPPDatasetInspector,
    MixedCodingReasoningDataset,
)
from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    count_valid_targets,
    load_tokenizer,
    next_token_cross_entropy,
    truncate_labels_to_budget,
)
from evaluators.coding_eval import (
    compare_thinking_off_vs_on,
    evaluate_function_synthesis,
)
from models.recursive_reasoning.trm_lm_adaptive import (
    STOP_BUDGET_EXHAUSTED,
    STOP_FIXED_DEPTH,
    STOP_LEARNED_HALT,
    STOP_MAX_REACHED,
    STOP_MINIMUM_REACHED,
    DualStateConfig,
    DualStateModel,
    per_position_cross_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate TRM-LM on Language Modeling vs Coding Reasoning."
    )
    # Experiment & Paths
    parser.add_argument(
        "--experiment-id", type=str, default="trm_lm_coding_reasoning_pilot"
    )
    parser.add_argument(
        "--workdir", type=str, default="experiments/trm_lm_coding_reasoning/run_smoke"
    )
    parser.add_argument(
        "--data-workdir", type=str, default="experiments/trm_lm_pilot_5m/r1"
    )
    parser.add_argument(
        "--mbpp-dir", type=str, default="data/coding/mbpp"
    )
    parser.add_argument(
        "--mbpp-config", type=str, default="full"
    )
    # Architecture
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--n-low-layers", type=int, default=1)
    parser.add_argument("--n-high-layers", type=int, default=1)
    parser.add_argument("--n-low-cycles", type=int, default=1)
    parser.add_argument("--n-high-cycles", type=int, default=2)
    parser.add_argument("--min-high-cycles", type=int, default=1)
    parser.add_argument("--max-high-cycles", type=int, default=4)
    parser.add_argument("--controller-hidden-size", type=int, default=64)
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--share-high-low-blocks", action="store_true")
    parser.add_argument("--gradient-strategy", type=str, default="full")
    # Training
    parser.add_argument("--mode", type=str, default="adaptive")
    parser.add_argument("--depth-sampling", type=str, default="uniform")
    parser.add_argument("--controller-loss-weight", type=float, default=1.0)
    parser.add_argument("--compute-penalty", type=float, default=0.0)
    parser.add_argument("--cost-threshold", type=float, default=0.0)
    parser.add_argument("--survivor-aware-controller", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    # Mixture weights
    parser.add_argument("--lang-weight", type=float, default=0.60)
    parser.add_argument("--code-continuation-weight", type=float, default=0.10)
    parser.add_argument("--func-synthesis-weight", type=float, default=0.20)
    parser.add_argument("--output-prediction-weight", type=float, default=0.10)
    # Budgets
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-train-tokens", type=int, default=100)
    parser.add_argument("--max-train-docs", type=int, default=40000)
    parser.add_argument("--max-val-docs", type=int, default=512)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--eval-coding-tasks", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device=device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def evaluate_language_modeling(
    model: DualStateModel,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    max_batches: Optional[int] = None,
    **forward_kwargs,
) -> Dict[str, Any]:
    """Evaluate next-token CE and PPL on held-out TinyStories."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    total_transitions = 0
    total_high_cycles = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(
                batch["input_ids"], batch["attention_mask"], mode=mode, **forward_kwargs
            )
            loss = next_token_cross_entropy(out.logits, batch["labels"])
            n_tokens = count_valid_targets(batch["labels"])
            total_loss += float(loss.detach()) * n_tokens
            total_tokens += n_tokens
            total_transitions += int(out.transitions.sum().item())
            total_high_cycles += int(out.high_cycles.sum().item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    model.train()
    if total_tokens == 0:
        return {"ce": float("nan"), "ppl": float("nan"), "tokens": 0}
    ce = total_loss / total_tokens
    return {
        "ce": ce,
        "ppl": math.exp(min(ce, 50.0)),
        "tokens": total_tokens,
        "mean_transitions": total_transitions / max(1, batches * loader.batch_size),
        "mean_high_cycles": total_high_cycles / max(1, batches * loader.batch_size),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    os.makedirs(args.workdir, exist_ok=True)

    # Resolve device and dtype
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; using CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32

    # Load tokenizer and TinyStories manifest
    tokenizer = load_tokenizer(args.data_workdir)
    ts_manifest_path = os.path.join(args.data_workdir, "manifest.json")
    with open(ts_manifest_path, "r", encoding="utf-8") as f:
        ts_manifest = json.load(f)

    ts_cfg = TinyStoriesConfig(
        parquet_path=ts_manifest["parquet"]["path"],
        workdir=args.data_workdir,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    # Initialize data pipelines
    ts_train_ds = TinyStoriesLMDataset(
        ts_cfg, ts_manifest, tokenizer, split="train", max_docs=args.max_train_docs
    )
    ts_val_ds = TinyStoriesLMDataset(
        ts_cfg, ts_manifest, tokenizer, split="val", max_docs=args.max_val_docs
    )
    ts_val_loader = DataLoader(
        ts_val_ds, batch_size=args.batch_size, collate_fn=collate_lm_batch
    )

    cr_cfg = CodingReasoningConfig(
        tinystories_workdir=args.data_workdir,
        mbpp_dir=args.mbpp_dir,
        seq_len=args.seq_len,
        seed=args.seed,
        lang_weight=args.lang_weight,
        code_continuation_weight=args.code_continuation_weight,
        func_synthesis_weight=args.func_synthesis_weight,
        output_prediction_weight=args.output_prediction_weight,
        mbpp_config=args.mbpp_config,
    )
    coding_train_ds = CodingReasoningDataset(cr_cfg, tokenizer, split="train")
    mixed_train_ds = MixedCodingReasoningDataset(
        cr_cfg,
        tokenizer,
        ts_train_ds,
        coding_train_ds,
        max_train_tokens=args.max_train_tokens,
    )
    mixed_train_loader = DataLoader(
        mixed_train_ds, batch_size=args.batch_size, collate_fn=collate_lm_batch
    )

    # Held-out MBPP validation and test records for coding evaluation
    inspector = MBPPDatasetInspector(mbpp_dir=args.mbpp_dir, config_name=args.mbpp_config)
    audit_data = inspector.audit_and_deduplicate()
    val_coding_records = audit_data["splits"]["validation"][: args.eval_coding_tasks]
    test_coding_records = audit_data["splits"]["test"][: args.eval_coding_tasks]

    # Model configuration
    model_cfg = DualStateConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        n_low_layers=args.n_low_layers,
        n_high_layers=args.n_high_layers,
        n_low_cycles=args.n_low_cycles,
        n_high_cycles=args.n_high_cycles,
        min_high_cycles=args.min_high_cycles,
        max_high_cycles=args.max_high_cycles,
        controller_hidden_size=args.controller_hidden_size,
        halt_threshold=args.halt_threshold,
        share_high_low_blocks=args.share_high_low_blocks,
        gradient_strategy=args.gradient_strategy,
        tie_embeddings=True,
        pad_token_id=tokenizer.token_to_id("<pad>"),
    )
    model = DualStateModel(model_cfg).to(device=device, dtype=dtype)
    model.rotary_emb.float()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"[init] Model params: {model.num_parameters:,} | Device: {device} | Dtype: {dtype}",
        flush=True,
    )

    # Initial Language Modeling evaluation
    init_val_fast = evaluate_language_modeling(
        model, ts_val_loader, device, mode="fast", max_batches=args.max_val_batches
    )
    init_val_adaptive = evaluate_language_modeling(
        model, ts_val_loader, device, mode="adaptive", max_batches=args.max_val_batches
    )
    print(
        f"[val:init] LM Fast: CE={init_val_fast['ce']:.4f}, PPL={init_val_fast['ppl']:.2f}",
        flush=True,
    )
    print(
        f"[val:init] LM Adaptive: CE={init_val_adaptive['ce']:.4f}, PPL={init_val_adaptive['ppl']:.2f}",
        flush=True,
    )

    # Initial Coding Function Synthesis evaluation (Thinking OFF vs ON)
    print("[val:init] Running initial Coding Synthesis evaluation...", flush=True)
    coding_comparison = compare_thinking_off_vs_on(
        model=model,
        tokenizer=tokenizer,
        eval_records=val_coding_records,
        max_new_tokens=64,
        device=device,
    )
    print(
        f"[val:init] Coding Thinking OFF pass@1: {coding_comparison['thinking_off']['pass_at_1']:.1%} | "
        f"Thinking ON pass@1: {coding_comparison['thinking_on']['pass_at_1']:.1%}",
        flush=True,
    )

    # Training loop
    model.train()
    step = 0
    tokens_processed = 0
    metrics_history: List[Dict[str, Any]] = []
    start_time = time.time()

    for batch in mixed_train_loader:
        if step >= args.max_steps:
            break
        if (
            args.max_train_tokens is not None
            and tokens_processed >= args.max_train_tokens
        ):
            break

        batch = move_batch(batch, device)
        n_valid = count_valid_targets(batch["labels"])
        if args.max_train_tokens is not None:
            remaining = args.max_train_tokens - tokens_processed
            if remaining <= 0:
                break
            if n_valid > remaining:
                batch["labels"], n_valid = truncate_labels_to_budget(
                    batch["labels"], remaining
                )
        if n_valid <= 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if args.mode == "fixed":
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                mode="fixed",
                n_high_cycles=args.n_high_cycles,
            )
            lm_loss = next_token_cross_entropy(out.logits, batch["labels"])
            loss = lm_loss
            ctrl_loss = None
        else:
            max_depth = args.max_high_cycles
            depths = list(range(1, max_depth + 1))
            depth_out = model.forward_depths(
                batch["input_ids"],
                batch["attention_mask"],
                depths=depths,
                max_depth=max_depth,
                evaluate_controller=True,
            )
            loss_by_depth = {
                d: next_token_cross_entropy(depth_out.logits_by_depth[d], batch["labels"])
                for d in depths
            }
            pos_loss_by_depth = {
                d: per_position_cross_entropy(
                    depth_out.logits_by_depth[d], batch["labels"]
                )
                for d in depths
            }

            if args.depth_sampling == "uniform":
                sampled_d = random.randint(1, max_depth)
                lm_loss = loss_by_depth[sampled_d]
            else:
                lm_loss = loss_by_depth[max_depth]

            valid = batch["labels"] != IGNORE_INDEX
            controller_terms = []
            cumulative_survival = torch.ones_like(batch["input_ids"], dtype=torch.float32)

            for d in range(args.min_high_cycles, max_depth):
                logits_d = depth_out.controller_logits_by_depth[d]
                delta = (pos_loss_by_depth[d] - pos_loss_by_depth[d + 1]).detach()
                target = (delta > args.cost_threshold).float()

                if not bool(valid.any()):
                    continue

                # Compute per-position BCE
                bce_pos = F.binary_cross_entropy_with_logits(
                    logits_d, target, reduction="none"
                )

                if args.survivor_aware_controller:
                    # Weight loss by cumulative survival probability up to depth d
                    weighted_bce = (bce_pos * cumulative_survival * valid.float()).sum() / (
                        (cumulative_survival * valid.float()).sum().clamp_min(1.0)
                    )
                    controller_terms.append(weighted_bce)
                    # Update survival probability for subsequent depths
                    cumulative_survival = cumulative_survival * torch.sigmoid(logits_d.detach().float())
                else:
                    controller_terms.append(bce_pos[valid].mean())

            ctrl_loss = torch.stack(controller_terms).mean() if controller_terms else None
            loss = lm_loss
            if ctrl_loss is not None:
                loss = loss + args.controller_loss_weight * ctrl_loss

        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite loss detected at step {step}: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        tokens_processed += n_valid

        metrics_history.append({
            "step": step,
            "lm_loss": float(lm_loss.item()),
            "ctrl_loss": float(ctrl_loss.item()) if ctrl_loss is not None else 0.0,
            "total_loss": float(loss.item()),
            "valid_tokens": n_valid,
            "tokens_processed": tokens_processed,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if step == 1 or step % 25 == 0 or (args.max_train_tokens and tokens_processed >= args.max_train_tokens):
            print(
                f"[step {step:4d}] LM Loss: {lm_loss.item():.4f} | "
                f"Ctrl Loss: {ctrl_loss.item() if ctrl_loss is not None else 0.0:.4f} | "
                f"Tokens: {tokens_processed:,} / {args.max_train_tokens or 0:,}",
                flush=True,
            )

    duration = time.time() - start_time
    throughput_tokens_sec = tokens_processed / max(0.001, duration)

    # Save atomic checkpoints
    ckpt_path = os.path.join(args.workdir, f"checkpoint_step_{step}.pt")
    final_ckpt_path = os.path.join(args.workdir, "checkpoint_final.pt")
    ckpt_payload = {
        "step": step,
        "tokens_processed": tokens_processed,
        "model_state_dict": model.state_dict(),
        "model_config": model.config_dict(),
        "token_accounting": mixed_train_ds.token_accounting,
        "example_accounting": mixed_train_ds.example_accounting,
    }
    torch.save(ckpt_payload, ckpt_path)
    torch.save(ckpt_payload, final_ckpt_path)

    # Verify reload max diff
    test_model = DualStateModel(model_cfg).to(device=device, dtype=dtype)
    test_ckpt = torch.load(final_ckpt_path, weights_only=True)
    test_model.load_state_dict(test_ckpt["model_state_dict"])
    test_model.eval()
    model.eval()
    with torch.no_grad():
        t_in = torch.randint(1, 100, (1, 16), device=device)
        t_mask = torch.ones_like(t_in)
        diff = float((model(t_in, t_mask).logits - test_model(t_in, t_mask).logits).abs().max().item())

    print(f"[eval] Running post-training evaluations on step {step}...", flush=True)

    # Final TinyStories LM evaluation (Fast vs Adaptive)
    final_val_fast = evaluate_language_modeling(
        model, ts_val_loader, device, mode="fast", max_batches=args.max_val_batches
    )
    final_val_adaptive = evaluate_language_modeling(
        model, ts_val_loader, device, mode="adaptive", max_batches=args.max_val_batches
    )
    print(
        f"[val:final] LM Fast: CE={final_val_fast['ce']:.4f}, PPL={final_val_fast['ppl']:.2f} | "
        f"LM Adaptive: CE={final_val_adaptive['ce']:.4f}, PPL={final_val_adaptive['ppl']:.2f}",
        flush=True,
    )

    # Final Coding evaluation on MBPP Validation (Thinking OFF vs ON)
    print("[val:final] Running post-training MBPP Validation evaluation...", flush=True)
    coding_comparison_final_val = compare_thinking_off_vs_on(
        model=model,
        tokenizer=tokenizer,
        eval_records=val_coding_records,
        max_new_tokens=64,
        device=device,
    )
    print(
        f"[val:final] Val Coding Thinking OFF pass@1: {coding_comparison_final_val['thinking_off']['pass_at_1']:.1%} | "
        f"Thinking ON pass@1: {coding_comparison_final_val['thinking_on']['pass_at_1']:.1%}",
        flush=True,
    )

    # Final Coding evaluation on MBPP Test (Thinking OFF vs ON)
    print("[val:final] Running post-training MBPP Test evaluation...", flush=True)
    coding_comparison_final_test = compare_thinking_off_vs_on(
        model=model,
        tokenizer=tokenizer,
        eval_records=test_coding_records,
        max_new_tokens=64,
        device=device,
    )
    print(
        f"[val:final] Test Coding Thinking OFF pass@1: {coding_comparison_final_test['thinking_off']['pass_at_1']:.1%} | "
        f"Thinking ON pass@1: {coding_comparison_final_test['thinking_on']['pass_at_1']:.1%}",
        flush=True,
    )

    # Forward latency and compute efficiency benchmarking
    print("[eval] Benchmarking inference compute efficiency...", flush=True)
    bench_seq_len = 128
    bench_input = torch.randint(1, 100, (1, bench_seq_len), device=device)
    bench_mask = torch.ones_like(bench_input)

    # Warmup
    for _ in range(5):
        _ = model(bench_input, bench_mask, mode="fast")
        _ = model(bench_input, bench_mask, mode="adaptive")

    # Fast mode benchmark
    fast_lats = []
    fast_trans = []
    for _ in range(20):
        t0 = time.perf_counter()
        out_f = model(bench_input, bench_mask, mode="fast")
        if device.type == "cuda":
            torch.cuda.synchronize()
        fast_lats.append((time.perf_counter() - t0) * 1000.0)
        fast_trans.append(float(out_f.transitions.float().mean().item()))

    # Adaptive mode benchmark
    adapt_lats = []
    adapt_trans = []
    adapt_cycles = []
    for _ in range(20):
        t0 = time.perf_counter()
        out_a = model(bench_input, bench_mask, mode="adaptive")
        if device.type == "cuda":
            torch.cuda.synchronize()
        adapt_lats.append((time.perf_counter() - t0) * 1000.0)
        adapt_trans.append(float(out_a.transitions.float().mean().item()))
        adapt_cycles.append(float(out_a.high_cycles.float().mean().item()))

    import numpy as np
    fast_lats_np = np.array(fast_lats)
    adapt_lats_np = np.array(adapt_lats)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )

    compute_efficiency = {
        "bench_config": {"batch_size": 1, "seq_len": bench_seq_len, "num_repeats": 20},
        "thinking_off": {
            "median_latency_ms": float(np.median(fast_lats_np)),
            "p25_latency_ms": float(np.percentile(fast_lats_np, 25)),
            "p75_latency_ms": float(np.percentile(fast_lats_np, 75)),
            "throughput_tok_per_sec": float(bench_seq_len / (np.median(fast_lats_np) / 1000.0)),
            "mean_transitions": float(np.mean(fast_trans)),
        },
        "thinking_on": {
            "median_latency_ms": float(np.median(adapt_lats_np)),
            "p25_latency_ms": float(np.percentile(adapt_lats_np, 25)),
            "p75_latency_ms": float(np.percentile(adapt_lats_np, 75)),
            "throughput_tok_per_sec": float(bench_seq_len / (np.median(adapt_lats_np) / 1000.0)),
            "mean_transitions": float(np.mean(adapt_trans)),
            "mean_high_cycles": float(np.mean(adapt_cycles)),
        },
        "latency_slowdown_ratio": float(np.median(adapt_lats_np) / max(0.001, np.median(fast_lats_np))),
        "peak_vram_mb": peak_vram_mb,
    }

    # Save metrics history
    with open(os.path.join(args.workdir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    # Save evaluation results
    eval_results = {
        "experiment_id": args.experiment_id,
        "completed_steps": step,
        "tokens_processed": tokens_processed,
        "duration_sec": duration,
        "tinystories_eval": {
            "init": {"fast": init_val_fast, "adaptive": init_val_adaptive},
            "final": {"fast": final_val_fast, "adaptive": final_val_adaptive},
        },
        "mbpp_coding_eval": {
            "val_tasks_evaluated": len(val_coding_records),
            "test_tasks_evaluated": len(test_coding_records),
            "init_val": coding_comparison,
            "final_val": coding_comparison_final_val,
            "final_test": coding_comparison_final_test,
        },
        "compute_efficiency": compute_efficiency,
    }
    with open(os.path.join(args.workdir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    summary = {
        "experiment_id": args.experiment_id,
        "completed_steps": step,
        "tokens_processed": tokens_processed,
        "duration_sec": duration,
        "throughput_tokens_per_sec": throughput_tokens_sec,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_reload_max_diff": diff,
        "realized_token_mixture": mixed_train_ds.token_accounting,
        "realized_example_mixture": mixed_train_ds.example_accounting,
        "eval_results": eval_results,
    }
    with open(os.path.join(args.workdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Training complete. Checkpoint reload max diff: {diff:.6f}", flush=True)
    return summary


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
