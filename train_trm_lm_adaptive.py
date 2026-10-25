#!/usr/bin/env python3
"""Training/verification entrypoint for the dual-state adaptive TRM-LM.

This is a new entrypoint.  It does not modify the existing ``train_trm_lm.py``
or the frozen Samsung reference.  It is intentionally capable of a bounded
smoke test with ``--max-steps 1``; substantive training requires separate
authorization.

Training modes
--------------
``fixed``
    Train a fixed high/low recursion depth.  The controller is not supervised.

``adaptive``
    Run ``max_high_cycles`` in the training graph and expose the model to
    multiple depths.  The LM loss is taken at a sampled depth (uniform), at all
    depths (mean), or at the maximum depth.  The controller is trained with a
    per-sequence marginal-utility target: continue if refinement at depth d+1
    lowers that sequence's CE by more than ``--cost-threshold``.  An optional
    differentiable expected-depth penalty (``--compute-penalty``) regularizes
    the controller.

The controller never receives labels at inference; labels are used only to
construct training-time targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    count_valid_targets,
    iter_packed_examples,
    load_tokenizer,
    next_token_cross_entropy,
    truncate_labels_to_budget,
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
from train_trm_lm import (
    git_commit,
    move_batch,
    resolve_device,
    resolve_dtype,
    rng_state,
    set_seed,
    sha256_file,
)

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-state adaptive TRM-LM training/smoke verification"
    )
    # Data
    parser.add_argument(
        "--data-workdir", type=str, default="experiments/trm_lm_pilot_5m/r1",
        help="Read-only pilot directory containing manifest.json and tokenizer.json",
    )
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default="dual_state_smoke")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-docs", type=int, default=16)
    parser.add_argument("--max-val-docs", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=4)
    # Model
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--n-low-layers", type=int, default=1)
    parser.add_argument("--n-high-layers", type=int, default=1)
    parser.add_argument("--n-low-cycles", type=int, default=1)
    parser.add_argument("--n-high-cycles", type=int, default=2)
    parser.add_argument("--min-high-cycles", type=int, default=1)
    parser.add_argument("--max-high-cycles", type=int, default=3)
    parser.add_argument("--controller-hidden-size", type=int, default=64)
    parser.add_argument("--halt-threshold", type=float, default=0.5)
    parser.add_argument("--share-high-low-blocks", action="store_true")
    parser.add_argument(
        "--gradient-strategy", type=str, default="full",
        choices=["full", "truncated", "last_high_cycle"],
    )
    # Training
    parser.add_argument("--mode", type=str, default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument(
        "--depth-sampling", type=str, default="uniform",
        choices=["uniform", "all", "max"],
    )
    parser.add_argument("--controller-loss-weight", type=float, default=1.0)
    parser.add_argument("--compute-penalty", type=float, default=0.0)
    parser.add_argument("--cost-threshold", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--save-every-tokens", type=int, default=None)
    parser.add_argument("--val-every-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=8)
    parser.add_argument("--generation-prompt-tokens", type=int, default=4)
    return parser.parse_args()


def next_multiple(current: int, interval: Optional[int]) -> Optional[int]:
    if interval is None or interval <= 0:
        return None
    return ((current // interval) + 1) * interval


def build_data(
    args: argparse.Namespace, device: torch.device, skip_tokens: int = 0
):
    manifest_path = os.path.join(args.data_workdir, "manifest.json")
    tokenizer_path = os.path.join(args.data_workdir, "tokenizer.json")
    if not os.path.exists(manifest_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Expected manifest.json and tokenizer.json in {args.data_workdir!r}"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tokenizer = load_tokenizer(args.data_workdir)
    config = TinyStoriesConfig(
        parquet_path=manifest["parquet"]["path"],
        workdir=args.data_workdir,
        seq_len=args.seq_len,
        tokenizer_vocab_size=tokenizer.get_vocab_size(),
        tokenizer_train_docs=20000,
        seed=args.seed,
        max_scan_rows=60000,
        max_train_docs=args.max_train_docs,
        max_val_docs=args.max_val_docs,
    )
    train_dataset = TinyStoriesLMDataset(
        config, manifest, tokenizer, "train",
        max_docs=args.max_train_docs, skip_tokens=skip_tokens,
    )
    val_dataset = TinyStoriesLMDataset(
        config, manifest, tokenizer, "val", max_docs=args.max_val_docs,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=collate_lm_batch, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_lm_batch, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    return config, manifest, tokenizer, train_loader, val_loader


def build_model(
    args: argparse.Namespace, tokenizer, device: torch.device, dtype: torch.dtype
) -> DualStateModel:
    config = DualStateConfig(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=1,
        num_heads=args.num_heads,
        expansion=4.0,
        n_refine=1,
        tie_embeddings=True,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
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
    )
    if (
        args.mode == "adaptive"
        and args.gradient_strategy == "last_high_cycle"
        and args.depth_sampling != "max"
    ):
        raise ValueError(
            "gradient_strategy='last_high_cycle' requires --depth-sampling max; "
            "intermediate LM losses are intentionally outside the graph"
        )
    model = DualStateModel(config).to(device=device, dtype=dtype)
    model.rotary_emb.float()
    return model


def evaluate_mode(
    model: DualStateModel,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    max_batches: Optional[int],
    **forward_kwargs,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    total_transitions = 0
    total_high_cycles = 0
    total_position_transitions = 0
    total_position_high_cycles = 0
    total_valid_positions = 0
    total_block_calls = 0
    total_controller_calls = 0
    total_active_positions = 0
    total_active_steps = 0
    reason_counts = {
        name: 0 for name in (
            "learned_halt", "max_reached", "budget_exhausted",
            "minimum_reached", "fixed_depth",
        )
    }
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch["input_ids"], batch["attention_mask"],
                mode=mode, **forward_kwargs,
            )
            loss = next_token_cross_entropy(output.logits, batch["labels"])
            n_tokens = count_valid_targets(batch["labels"])
            total_loss += float(loss.detach()) * n_tokens
            total_tokens += n_tokens
            total_sequences += batch["input_ids"].shape[0]
            total_transitions += int(output.transitions.sum().item())
            total_high_cycles += int(output.high_cycles.sum().item())
            total_position_transitions += int(
                output.transitions_per_position.sum().item()
            )
            total_position_high_cycles += int(
                output.high_cycles_per_position.sum().item()
            )
            valid = batch["attention_mask"].bool()
            total_valid_positions += int(valid.sum().item())
            total_block_calls += int(output.block_calls)
            total_controller_calls += int(output.controller_calls)
            total_active_positions += int(sum(output.active_counts_per_step))
            total_active_steps += len(output.active_counts_per_step)
            codes = output.stop_reason_codes[valid]
            for code, name in (
                (STOP_LEARNED_HALT, "learned_halt"),
                (STOP_MAX_REACHED, "max_reached"),
                (STOP_BUDGET_EXHAUSTED, "budget_exhausted"),
                (STOP_MINIMUM_REACHED, "minimum_reached"),
                (STOP_FIXED_DEPTH, "fixed_depth"),
            ):
                reason_counts[name] += int((codes == code).sum().item())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    model.train()
    if total_tokens == 0:
        raise RuntimeError("No validation target tokens were produced")
    ce = total_loss / total_tokens
    sequence_denom = max(1, total_sequences)
    position_denom = max(1, total_valid_positions)
    batch_denom = max(1, batches)
    step_denom = max(1, total_active_steps)
    return {
        "ce": ce,
        "ppl": math.exp(min(ce, 50.0)),
        "tokens": total_tokens,
        "sequences": total_sequences,
        "mean_transitions": total_transitions / sequence_denom,
        "mean_high_cycles": total_high_cycles / sequence_denom,
        "mean_position_transitions": (
            total_position_transitions / position_denom
        ),
        "mean_position_high_cycles": (
            total_position_high_cycles / position_denom
        ),
        "mean_block_calls": total_block_calls / batch_denom,
        "mean_controller_calls": total_controller_calls / batch_denom,
        "mean_active_positions_per_step": (
            total_active_positions / step_denom
        ),
        "frac_learned_halt": reason_counts["learned_halt"] / position_denom,
        "frac_max_reached": reason_counts["max_reached"] / position_denom,
        "frac_budget_exhausted": (
            reason_counts["budget_exhausted"] / position_denom
        ),
        "frac_minimum_reached": (
            reason_counts["minimum_reached"] / position_denom
        ),
        "frac_fixed_depth": reason_counts["fixed_depth"] / position_denom,
    }


def save_checkpoint(
    path: str,
    model: DualStateModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tokens_processed: int,
    model_config: DualStateConfig,
    train_config: Dict[str, object],
    tokenizer_sha256: str,
    manifest_sha256: str,
    experiment_id: str,
    device: torch.device,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "tokens_processed": tokens_processed,
            "model_config": model_config.config_dict(),
            "train_config": train_config,
            "tokenizer_sha256": tokenizer_sha256,
            "manifest_sha256": manifest_sha256,
            "experiment_id": experiment_id,
            "git_commit": train_config.get("git_commit"),
            "rng_state": rng_state(),
            "device_at_save": str(device),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: DualStateModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_config: DualStateConfig,
    tokenizer_sha256: str,
    manifest_sha256: str,
    experiment_id: str,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != model_config.config_dict():
        raise RuntimeError("Checkpoint model configuration does not match")
    if checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
        raise RuntimeError("Checkpoint tokenizer identity does not match")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Checkpoint dataset manifest identity does not match")
    if checkpoint.get("experiment_id") != experiment_id:
        raise RuntimeError(
            f"Checkpoint experiment_id {checkpoint.get('experiment_id')!r} "
            f"does not match {experiment_id!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    state = checkpoint.get("rng_state")
    if state:
        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if torch.cuda.is_available() and "cuda" in state:
                torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass
    return checkpoint


def write_metric(handle, record: Dict[str, object]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def collect_prompt(
    data_config, manifest, tokenizer, num_tokens: int
):
    for example in iter_packed_examples(
        data_config, manifest, tokenizer, "val", max_docs=1, max_chunks=1
    ):
        real = int(example["attention_mask"].sum().item())
        take = min(num_tokens, real)
        ids = example["input_ids"][:take].clone()
        return {
            "ids": ids.tolist(),
            "text": tokenizer.decode(ids.tolist()),
        }
    raise RuntimeError("No validation example was available for generation")


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 2)
        args.max_train_docs = min(args.max_train_docs, 16)
        args.max_val_docs = min(args.max_val_docs, 8)
        args.max_val_batches = min(args.max_val_batches, 4)
        print("[smoke-test] bounded run enforced", flush=True)
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    set_seed(args.seed)
    workdir = args.workdir
    if workdir is None:
        import tempfile
        workdir = tempfile.mkdtemp(prefix="trm_lm_adaptive_")
        print(f"[run] temporary workdir={workdir}", flush=True)
    else:
        if os.path.exists(workdir) and os.listdir(workdir) and not args.resume:
            raise SystemExit(
                f"Refusing to overwrite non-empty workdir {workdir!r}"
            )
        os.makedirs(workdir, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    data_config, manifest, tokenizer, train_loader, val_loader = build_data(
        args, device, skip_tokens=0
    )
    tokenizer_sha = sha256_file(
        os.path.join(args.data_workdir, "tokenizer.json")
    )
    manifest_sha = sha256_file(
        os.path.join(args.data_workdir, "manifest.json")
    )
    model = build_model(args, tokenizer, device, dtype)
    print(
        f"[model] params={model.num_parameters:,} "
        f"breakdown={model.parameter_breakdown()}",
        flush=True,
    )
    print(
        f"[model] config={model.config_dict()}",
        flush=True,
    )
    print(f"[model] device={device} dtype={dtype}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        max_steps = max(args.max_steps, 1)
        warmup = max(args.warmup_steps, 0)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, max_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler,
            model.config, tokenizer_sha, manifest_sha, args.experiment_id, device,
        )
        start_step = int(checkpoint["step"])
        tokens_processed = int(checkpoint["tokens_processed"])
        _, _, _, train_loader, _ = build_data(
            args, device, skip_tokens=tokens_processed
        )
        print(
            f"[resume] step={start_step} tokens={tokens_processed}",
            flush=True,
        )
    else:
        start_step = 0
        tokens_processed = 0

    train_config = {
        "args": vars(args),
        "model_config": model.config_dict(),
        "parameter_breakdown": model.parameter_breakdown(),
        "data_workdir": args.data_workdir,
        "tokenizer_sha256": tokenizer_sha,
        "manifest_sha256": manifest_sha,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": str(device),
        "dtype": str(dtype),
        "experiment_id": args.experiment_id,
        "halting_granularity": "per_position",
    }
    with open(os.path.join(workdir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)
    with open(os.path.join(workdir, "model_config.json"), "w", encoding="utf-8") as handle:
        json.dump(model.config_dict(), handle, indent=2, sort_keys=True)

    metrics_path = os.path.join(workdir, "metrics.jsonl")
    metrics = open(metrics_path, "a", encoding="utf-8")
    write_metric(metrics, {
        "event": "run_start", "step": start_step,
        "tokens": tokens_processed, "params": model.num_parameters,
        "experiment_id": args.experiment_id,
    })

    # Validation at initialization: fixed max depth, fast, adaptive.
    fixed_max = args.max_high_cycles
    val_fixed = evaluate_mode(
        model, val_loader, device, "fixed", args.max_val_batches,
        n_high_cycles=fixed_max, n_low_cycles=args.n_low_cycles,
    )
    val_fast = evaluate_mode(
        model, val_loader, device, "fast", args.max_val_batches,
        n_low_cycles=args.n_low_cycles,
    )
    val_adaptive = evaluate_mode(
        model, val_loader, device, "adaptive", args.max_val_batches,
        max_high_cycles=args.max_high_cycles,
        n_low_cycles=args.n_low_cycles,
        halt_threshold=args.halt_threshold,
    )
    for name, value in [
        ("fixed_max", val_fixed), ("fast", val_fast),
        ("adaptive", val_adaptive),
    ]:
        write_metric(metrics, {
            "event": "init_val", "mode": name, "step": start_step,
            "tokens": tokens_processed, **value,
        })
        print(f"[val] init {name}: {value}", flush=True)

    # Prompt for generation diagnostics.
    prompt = collect_prompt(
        data_config, manifest, tokenizer, args.generation_prompt_tokens
    )

    if args.mode == "adaptive" and args.max_high_cycles < args.min_high_cycles:
        raise ValueError("max_high_cycles must be >= min_high_cycles")

    model.train()
    step = start_step
    tokens_this_run = 0
    train_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    next_val = next_multiple(tokens_processed, args.val_every_tokens)
    next_save = next_multiple(tokens_processed, args.save_every_tokens)
    last_train = None
    last_controller = None
    last_expected_depth = None

    for batch in train_loader:
        if step >= args.max_steps:
            break
        if args.max_train_tokens is not None and tokens_processed >= args.max_train_tokens:
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

        if args.mode == "fixed":
            output = model(
                batch["input_ids"], batch["attention_mask"],
                mode="fixed", n_high_cycles=args.n_high_cycles,
                n_low_cycles=args.n_low_cycles,
            )
            lm_loss = next_token_cross_entropy(
                output.logits, batch["labels"]
            )
            controller_loss = None
            expected_depth = None
            sampled_depth = args.n_high_cycles
            mean_high = float(output.high_cycles.float().mean().item())
            loss = lm_loss
        else:
            max_depth = args.max_high_cycles
            depths = list(range(1, max_depth + 1))
            depth_output = model.forward_depths(
                batch["input_ids"], batch["attention_mask"],
                depths=depths, max_depth=max_depth,
                evaluate_controller=True, controller_max_depth=max_depth,
            )
            loss_by_depth = {
                d: next_token_cross_entropy(
                    depth_output.logits_by_depth[d], batch["labels"]
                )
                for d in depths
            }
            pos_loss_by_depth = {
                d: per_position_cross_entropy(
                    depth_output.logits_by_depth[d], batch["labels"]
                )
                for d in depths
            }
            if args.depth_sampling == "uniform":
                sampled_depth = random.randint(1, max_depth)
                lm_loss = loss_by_depth[sampled_depth]
            elif args.depth_sampling == "max":
                sampled_depth = max_depth
                lm_loss = loss_by_depth[max_depth]
            else:
                sampled_depth = max_depth
                lm_loss = torch.stack(
                    [loss_by_depth[d] for d in depths]
                ).mean()
            valid = batch["labels"] != IGNORE_INDEX
            controller_terms = []
            controller_acc = []
            for d in range(args.min_high_cycles, max_depth):
                logits_d = depth_output.controller_logits_by_depth[d]
                delta = (
                    pos_loss_by_depth[d] - pos_loss_by_depth[d + 1]
                ).detach()
                target = (delta > args.cost_threshold).float()
                if not bool(valid.any()):
                    continue
                controller_terms.append(
                    F.binary_cross_entropy_with_logits(
                        logits_d[valid], target[valid], reduction="mean"
                    )
                )
                controller_acc.append(
                    ((logits_d[valid] >= 0).float() == target[valid])
                    .float().mean()
                )
            if controller_terms:
                controller_loss = torch.stack(controller_terms).mean()
                controller_acc_t = torch.stack(controller_acc).mean()
            else:
                controller_loss = torch.zeros((), device=device)
                controller_acc_t = torch.zeros((), device=device)
            expected_depth = None
            if args.compute_penalty > 0 and controller_terms:
                valid_float = valid.float()
                expected = (
                    torch.full_like(
                        valid_float, float(args.min_high_cycles)
                    ) * valid_float
                )
                running = valid_float.clone()
                for d in range(args.min_high_cycles, max_depth):
                    running = running * torch.sigmoid(
                        depth_output.controller_logits_by_depth[d].float()
                    )
                    expected = expected + running
                expected_depth = (
                    expected.sum() / valid_float.sum().clamp_min(1)
                )
            loss = lm_loss
            if controller_loss is not None:
                loss = loss + args.controller_loss_weight * controller_loss
            if expected_depth is not None:
                loss = loss + args.compute_penalty * expected_depth
            mean_high = float(max_depth)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step + 1}: {loss}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite grad_norm at step {step + 1}")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        tokens_this_run += n_valid
        tokens_processed += n_valid
        last_train = float(lm_loss.detach().cpu())
        last_controller = (
            float(controller_loss.detach().cpu())
            if controller_loss is not None else None
        )
        last_expected_depth = (
            float(expected_depth.detach().cpu())
            if expected_depth is not None else None
        )
        record = {
            "event": "train", "step": step, "tokens": tokens_processed,
            "lm_loss": last_train, "total_loss": float(loss.detach().cpu()),
            "grad_norm": float(grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "sampled_depth": sampled_depth,
            "high_cycles": mean_high,
            "controller_loss": last_controller,
            "controller_accuracy": (
                float(controller_acc_t.detach().cpu())
                if args.mode == "adaptive" else None
            ),
            "expected_depth": last_expected_depth,
            "elapsed_sec": time.time() - train_start,
        }
        write_metric(metrics, record)
        print(f"[train] {record}", flush=True)

        while next_val is not None and tokens_processed >= next_val:
            for name, mode, kwargs in [
                ("fixed_max", "fixed", {
                    "n_high_cycles": fixed_max,
                    "n_low_cycles": args.n_low_cycles,
                }),
                ("fast", "fast", {"n_low_cycles": args.n_low_cycles}),
                ("adaptive", "adaptive", {
                    "max_high_cycles": args.max_high_cycles,
                    "n_low_cycles": args.n_low_cycles,
                    "halt_threshold": args.halt_threshold,
                }),
            ]:
                value = evaluate_mode(
                    model, val_loader, device, mode,
                    args.max_val_batches, **kwargs,
                )
                write_metric(metrics, {
                    "event": "val", "mode": name, "step": step,
                    "tokens": tokens_processed, **value,
                })
                print(f"[val] {name}: {value}", flush=True)
            next_val += args.val_every_tokens

        while next_save is not None and tokens_processed >= next_save:
            path = os.path.join(workdir, "checkpoints", f"tokens_{tokens_processed}.pt")
            save_checkpoint(
                path, model, optimizer, scheduler, step, tokens_processed,
                model.config, train_config, tokenizer_sha, manifest_sha,
                args.experiment_id, device,
            )
            write_metric(metrics, {
                "event": "checkpoint", "step": step,
                "tokens": tokens_processed, "path": path,
            })
            print(f"[checkpoint] {path}", flush=True)
            next_save += args.save_every_tokens

        if args.max_train_tokens is not None and tokens_processed >= args.max_train_tokens:
            break

    train_elapsed = time.time() - train_start
    if args.max_train_tokens is not None and tokens_processed != args.max_train_tokens:
        raise RuntimeError(
            f"Training stopped at {tokens_processed} tokens, expected "
            f"{args.max_train_tokens}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    else:
        peak_alloc = peak_reserved = None

    final_path = os.path.join(workdir, "checkpoints", f"tokens_{tokens_processed}.pt")
    if not os.path.exists(final_path):
        save_checkpoint(
            final_path, model, optimizer, scheduler, step, tokens_processed,
            model.config, train_config, tokenizer_sha, manifest_sha,
            args.experiment_id, device,
        )

    # Final fixed/adaptive/fast validation.
    final_val = {}
    for name, mode, kwargs in [
        ("fixed_max", "fixed", {
            "n_high_cycles": fixed_max,
            "n_low_cycles": args.n_low_cycles,
        }),
        ("fast", "fast", {"n_low_cycles": args.n_low_cycles}),
        ("adaptive", "adaptive", {
            "max_high_cycles": args.max_high_cycles,
            "n_low_cycles": args.n_low_cycles,
            "halt_threshold": args.halt_threshold,
        }),
        ("budgeted", "budgeted", {
            "max_high_cycles": args.max_high_cycles,
            "n_low_cycles": args.n_low_cycles,
            "halt_threshold": args.halt_threshold,
            "max_total_transitions": (
                args.max_high_cycles * (args.n_low_cycles + 1)
            ),
        }),
    ]:
        value = evaluate_mode(
            model, val_loader, device, mode, args.max_val_batches, **kwargs
        )
        final_val[name] = value
        write_metric(metrics, {
            "event": "final_val", "mode": name, "step": step,
            "tokens": tokens_processed, **value,
        })
        print(f"[val] final {name}: {value}", flush=True)

    # Checkpoint reload and deterministic output equivalence.
    reloaded = build_model(args, tokenizer, device, dtype)
    load_checkpoint(
        final_path, reloaded, optimizer, scheduler, model.config,
        tokenizer_sha, manifest_sha, args.experiment_id, device,
    )
    reloaded.eval()
    fixed_batch = next(iter(val_loader))
    fixed_batch = move_batch(fixed_batch, device)
    model.eval()
    with torch.no_grad():
        before = model(
            fixed_batch["input_ids"], fixed_batch["attention_mask"],
            mode="fixed", n_high_cycles=fixed_max,
            n_low_cycles=args.n_low_cycles,
        ).logits
        after = reloaded(
            fixed_batch["input_ids"], fixed_batch["attention_mask"],
            mode="fixed", n_high_cycles=fixed_max,
            n_low_cycles=args.n_low_cycles,
        ).logits
    reload_diff = float((before - after).abs().max().cpu())
    if not math.isfinite(reload_diff) or reload_diff > 1e-2:
        raise RuntimeError(f"Checkpoint reload mismatch: {reload_diff}")

    # Generation diagnostics for every inference mode.
    prompt_tensor = torch.tensor(
        [prompt["ids"]], dtype=torch.long, device=device
    )
    generation = {}
    for mode, kwargs in [
        ("fast", {"n_low_cycles": args.n_low_cycles}),
        ("fixed", {
            "n_high_cycles": fixed_max,
            "n_low_cycles": args.n_low_cycles,
        }),
        ("adaptive", {
            "max_high_cycles": args.max_high_cycles,
            "n_low_cycles": args.n_low_cycles,
            "halt_threshold": args.halt_threshold,
        }),
        ("budgeted", {
            "max_high_cycles": args.max_high_cycles,
            "n_low_cycles": args.n_low_cycles,
            "halt_threshold": args.halt_threshold,
            "max_total_transitions": (
                args.max_high_cycles * (args.n_low_cycles + 1)
            ),
        }),
    ]:
        out = reloaded.generate(
            prompt_tensor,
            max_new_tokens=args.generation_max_new_tokens,
            eos_token_id=tokenizer.token_to_id("<eos>"),
            mode=mode,
            **kwargs,
        )
        generation[mode] = {
            "input_ids": prompt["ids"],
            "input_text": prompt["text"],
            "output_ids": out.tokens[0].tolist(),
            "output_text": tokenizer.decode(out.tokens[0].tolist()),
            "transitions_per_step": out.transitions_per_step,
            "position_transitions_per_step": out.position_transitions_per_step,
            "total_transitions": out.total_transitions,
            "transitions_last_forward": out.transitions_last_forward,
            "stopping_reasons": out.stopping_reasons,
            "high_cycles_per_step": out.high_cycles_per_step,
            "position_high_cycles_per_step": out.position_high_cycles_per_step,
            "block_calls_per_step": out.block_calls_per_step,
            "controller_calls_per_step": out.controller_calls_per_step,
            "new_token_count": out.new_token_count,
        }
        print(f"[generation] {mode}: {generation[mode]['output_text']!r}", flush=True)

    max_rss_mb = None
    if resource is not None:
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {
        "experiment_id": args.experiment_id,
        "git_commit": train_config["git_commit"],
        "mode": args.mode,
        "gradient_strategy": args.gradient_strategy,
        "depth_sampling": args.depth_sampling,
        "halting_granularity": "per_position",
        "params": model.num_parameters,
        "parameter_breakdown": model.parameter_breakdown(),
        "model_config": model.config_dict(),
        "train_config": train_config,
        "tokens_processed": tokens_processed,
        "tokens_this_run": tokens_this_run,
        "optimizer_steps": step - start_step,
        "final_step": step,
        "train_elapsed_sec": round(train_elapsed, 3),
        "train_tokens_per_sec": (
            round(tokens_this_run / train_elapsed, 3)
            if train_elapsed > 0 else None
        ),
        "peak_training_memory_allocated_gb": (
            round(peak_alloc, 3) if peak_alloc is not None else None
        ),
        "peak_training_memory_reserved_gb": (
            round(peak_reserved, 3) if peak_reserved is not None else None
        ),
        "max_rss_mb": round(max_rss_mb, 1) if max_rss_mb is not None else None,
        "last_train_loss": last_train,
        "last_controller_loss": last_controller,
        "last_expected_depth": last_expected_depth,
        "initial_validation": {
            "fixed_max": val_fixed, "fast": val_fast,
            "adaptive": val_adaptive,
        },
        "final_validation": final_val,
        "checkpoint_path": final_path,
        "checkpoint_reload_max_abs_diff": reload_diff,
        "generation": generation,
        "tokenizer_sha256": tokenizer_sha,
        "manifest_sha256": manifest_sha,
        "substantive_training_authorized": False,
        "note": "Smoke/verification entrypoint; substantive training requires separate authorization.",
    }
    with open(os.path.join(workdir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    metrics.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
