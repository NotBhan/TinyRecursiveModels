"""Train objective-only monotonic-refinement candidates.

This script fine-tunes the existing dual-state checkpoint with all four depths
supervised in every step. It does not change the architecture.

Arms:
  mean          : L = mean_d L_d
  monotonic     : L = mean_d L_d + lambda * mean_d softplus(L_{d+1} - L_d)
  preservation  : L = mean_d L_d + mu * mean_d KL(p_d.detach() || p_{d+1})
  combined      : both penalties

The controller is not trained in these arms (controller_loss_weight=0), so the
comparison isolates the backbone objective. The test split is never used.

Example:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. \
  .venv-trm-lm/bin/python train_refinement_monotonic.py \
    --workdir experiments/re_refinement_monotonicity/diagnostic/M1_seed1337 \
    --init-checkpoint experiments/re_coding_adaptive/stdlib_instr_1m/checkpoint_final.pt \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --max-train-tokens 250000 --loss-mode monotonic --monotonic-weight 0.5 \
    --seed 1337
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    collate_lm_batch,
    count_valid_targets,
    load_tokenizer,
    next_token_cross_entropy,
    truncate_labels_to_budget,
)
from evaluators.re_eval import (
    evaluate_language_modeling,
    parameter_bytes,
)
from models.recursive_reasoning.trm_lm_adaptive import (
    DualStateConfig,
    DualStateModel,
    per_position_cross_entropy,
)
from train_re_coding import (
    build_datasets,
    lr_at,
    move_batch,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--tokenizer-workdir", default="experiments/re_coding_adaptive/tokenizer")
    parser.add_argument("--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1")
    parser.add_argument("--mbpp-dir", default="data/coding/mbpp")
    parser.add_argument("--mbpp-config", default="full")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Phase 9 JSON manifest; selects a pre-registered training task set",
    )
    parser.add_argument(
        "--manifest-key",
        default="train_clean_task_ids",
        help="manifest list to use for MBPP training task ids",
    )
    parser.add_argument(
        "--prefix-prob",
        type=float,
        default=0.0,
        help="probability of prefix-augmenting an MBPP example",
    )
    parser.add_argument(
        "--prefix-max-fraction",
        type=float,
        default=0.75,
        help="maximum canonical body prefix fraction for prefix augmentation",
    )
    parser.add_argument("--max-train-tokens", type=int, default=250_000)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--lr-schedule-steps",
        type=int,
        default=None,
        help=(
            "explicit cosine horizon in optimizer steps. Default behavior is "
            "the historical estimate from max_train_tokens/(batch*seq*0.85); "
            "Phase 9 sets this to the realized step count to avoid premature "
            "LR decay."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--loss-mode",
        choices=["mean", "monotonic", "preservation", "combined"],
        default="monotonic",
    )
    parser.add_argument("--monotonic-weight", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument(
        "--update-mode",
        choices=["overwrite", "residual", "gated"],
        default="overwrite",
    )
    parser.add_argument("--stable-update-from-depth", type=int, default=2)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--residual-alpha-init", type=float, default=0.5)
    parser.add_argument(
        "--stability-weight-decay",
        type=float,
        default=None,
        help="weight decay for gate/alpha parameters; defaults to --weight-decay",
    )
    parser.add_argument(
        "--self-generated-prefix-prob",
        type=float,
        default=0.0,
        help="replace this fraction of target input tokens with depth-1 predictions",
    )
    parser.add_argument("--save-every-tokens", type=int, default=125_000)
    parser.add_argument("--val-every-tokens", type=int, default=125_000)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--max-train-docs", type=int, default=4000)
    parser.add_argument("--max-val-docs", type=int, default=256)
    parser.add_argument("--max-code-files", type=int, default=4000)
    parser.add_argument("--code-lm-weight", type=float, default=0.30)
    parser.add_argument("--instruction-weight", type=float, default=0.25)
    parser.add_argument("--lang-weight", type=float, default=0.10)
    parser.add_argument("--stdlib-instruction-weight", type=float, default=0.35)
    parser.add_argument("--stdlib-max-functions", type=int, default=20_000)
    return parser.parse_args()


def parse_depths(value: str) -> List[int]:
    depths = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not depths or depths[0] < 1:
        raise ValueError("--depths must contain positive integers")
    return depths


def depth_losses(depth_out, labels, depths: List[int]) -> Dict[int, torch.Tensor]:
    return {
        d: next_token_cross_entropy(depth_out.logits_by_depth[d], labels)
        for d in depths
    }


def monotonic_penalty(losses: Dict[int, torch.Tensor], depths: List[int]) -> torch.Tensor:
    terms = []
    for a, b in zip(depths[:-1], depths[1:]):
        terms.append(F.softplus(losses[b] - losses[a]))
    return torch.stack(terms).mean()


def preservation_penalty(depth_out, labels, depths: List[int]) -> torch.Tensor:
    valid = (labels != IGNORE_INDEX).float()
    terms = []
    for a, b in zip(depths[:-1], depths[1:]):
        logp_a = F.log_softmax(depth_out.logits_by_depth[a].float(), dim=-1)
        logp_b = F.log_softmax(depth_out.logits_by_depth[b].float(), dim=-1)
        # One-way KL from the detached earlier distribution to the later one.
        kl = (logp_a.exp().detach() * (logp_a.detach() - logp_b)).sum(-1)
        terms.append((kl * valid).sum() / valid.sum().clamp_min(1.0))
    return torch.stack(terms).mean()


def main() -> None:
    args = parse_args()
    if args.stability_weight_decay is None:
        args.stability_weight_decay = args.weight_decay
    os.makedirs(args.workdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32
    depths = parse_depths(args.depths)

    tokenizer = load_tokenizer(args.tokenizer_workdir)
    tokenizer_info_path = os.path.join(args.tokenizer_workdir, "tokenizer_config.json")
    tokenizer_info = json.load(open(tokenizer_info_path, "r", encoding="utf-8")) if os.path.exists(tokenizer_info_path) else {}
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        args.mbpp_task_ids = [int(value) for value in manifest[args.manifest_key]]
        print(
            f"[manifest] {args.manifest} key={args.manifest_key} "
            f"tasks={len(args.mbpp_task_ids)}",
            flush=True,
        )
    else:
        args.mbpp_task_ids = None
    mixture, lang_val, code_cfg, ts_manifest = build_datasets(args, tokenizer)
    train_loader = DataLoader(mixture, batch_size=args.batch_size, collate_fn=collate_lm_batch, num_workers=0)
    val_loader = DataLoader(lang_val, batch_size=args.batch_size, collate_fn=collate_lm_batch, num_workers=0)

    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["model_config"])
    cfg_dict.update(
        {
            "update_mode": args.update_mode,
            "stable_update_from_depth": args.stable_update_from_depth,
            "gate_init_bias": args.gate_init_bias,
            "residual_alpha_init": args.residual_alpha_init,
        }
    )
    model_cfg = DualStateConfig(**cfg_dict)
    if max(depths) > model_cfg.max_high_cycles:
        raise ValueError(
            f"requested depth {max(depths)} exceeds checkpoint max_high_cycles={model_cfg.max_high_cycles}"
        )
    model = DualStateModel(model_cfg).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    bad_missing = [k for k in missing if not k.startswith(("gate_", "alpha_"))]
    bad_unexpected = [k for k in unexpected if not k.startswith(("gate_", "alpha_"))]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"non-stability parameter mismatch: missing={bad_missing} unexpected={bad_unexpected}"
        )
    print(
        f"[load] missing={len(missing)} (stability-only) unexpected={len(unexpected)} "
        f"update_mode={args.update_mode} stable_from={args.stable_update_from_depth}",
        flush=True,
    )
    model.rotary_emb.float()
    stability_params = [
        p for name, p in model.named_parameters()
        if name.startswith(("gate_", "alpha_")) and p.requires_grad
    ]
    stability_ids = {id(p) for p in stability_params}
    base_params = [p for p in model.parameters() if id(p) not in stability_ids]
    if stability_params:
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "weight_decay": args.weight_decay},
                {"params": stability_params, "weight_decay": args.stability_weight_decay},
            ],
            lr=args.lr,
        )
        print(
            f"[optim] stability params={sum(p.numel() for p in stability_params)} "
            f"weight_decay={args.stability_weight_decay}",
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    print(
        f"[init] {args.init_checkpoint} tokens={ckpt.get('tokens_processed')} "
        f"params={model.num_parameters:,} weights={parameter_bytes(model)/1e6:.2f} MB "
        f"depths={depths} loss={args.loss_mode} seed={args.seed}",
        flush=True,
    )

    init_lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=args.val_batches)
    print(f"[val:init] LM fast CE={init_lm['ce']:.4f} PPL={init_lm['ppl']:.2f}", flush=True)

    est_tokens_per_step = max(1.0, float(args.batch_size * args.seq_len * 0.85))
    est_steps = max(1, int(args.max_train_tokens / est_tokens_per_step))
    if args.lr_schedule_steps is not None:
        est_steps = max(1, int(args.lr_schedule_steps))
        print(f"[lr] explicit cosine horizon={est_steps} steps", flush=True)
    step = 0
    tokens_processed = 0
    history: List[Dict[str, Any]] = []
    start_time = time.time()
    last_val_tokens = 0
    last_save_tokens = 0
    model.train()

    for batch in train_loader:
        if step >= args.max_steps or tokens_processed >= args.max_train_tokens:
            break
        batch = move_batch(batch, device)
        n_valid = count_valid_targets(batch["labels"])
        remaining = args.max_train_tokens - tokens_processed
        if n_valid > remaining:
            batch["labels"], n_valid = truncate_labels_to_budget(batch["labels"], remaining)
        if n_valid <= 0:
            continue
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args, est_steps)
        optimizer.zero_grad(set_to_none=True)
        if args.self_generated_prefix_prob > 0:
            # Exposure-bias control: keep the canonical depth-1 path, but let
            # depths 2+ refine from a prefix containing depth-1 predictions.
            with torch.no_grad():
                depth1_out = model.forward_depths(
                    batch["input_ids"],
                    batch["attention_mask"],
                    depths=[depths[0]],
                    max_depth=depths[0],
                    evaluate_controller=False,
                )
                preds = depth1_out.logits_by_depth[depths[0]].argmax(dim=-1)
            modified = batch["input_ids"].clone()
            replace = (batch["labels"][:, :-1] != IGNORE_INDEX) & (
                torch.rand(batch["input_ids"].shape, device=device) < args.self_generated_prefix_prob
            )[:, 1:]
            modified[:, 1:] = torch.where(
                replace, preds[:, :-1], batch["input_ids"][:, 1:]
            )
            depth_out = model.forward_depths(
                modified,
                batch["attention_mask"],
                depths=depths[1:],
                max_depth=max(depths),
                evaluate_controller=False,
            )
            losses = {
                depths[0]: next_token_cross_entropy(
                    depth1_out.logits_by_depth[depths[0]], batch["labels"]
                )
            }
            for d in depths[1:]:
                losses[d] = next_token_cross_entropy(
                    depth_out.logits_by_depth[d], batch["labels"]
                )
            # Cross-input preservation KL is not meaningful here.
            pres = torch.zeros((), device=device)
        else:
            depth_out = model.forward_depths(
                batch["input_ids"],
                batch["attention_mask"],
                depths=depths,
                max_depth=max(depths),
                evaluate_controller=False,
            )
            losses = depth_losses(depth_out, batch["labels"], depths)
            pres = preservation_penalty(depth_out, batch["labels"], depths) if args.loss_mode in {"preservation", "combined"} else torch.zeros((), device=device)
        lm_loss = torch.stack([losses[d] for d in depths]).mean()
        mono = monotonic_penalty(losses, depths) if args.loss_mode in {"monotonic", "combined"} else torch.zeros((), device=device)
        loss = lm_loss
        if args.loss_mode in {"monotonic", "combined"}:
            loss = loss + args.monotonic_weight * mono
        if args.loss_mode in {"preservation", "combined"}:
            loss = loss + args.preservation_weight * pres
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        optimizer.step()
        step += 1
        tokens_processed += n_valid

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            throughput = tokens_processed / max(1e-9, elapsed)
            print(
                f"[step {step:5d}] loss={loss.item():.4f} depths="
                + ",".join(f"{d}:{losses[d].item():.3f}" for d in depths)
                + f" mono={mono.item():.4f} pres={pres.item():.4f} "
                f"grad={grad_norm:.3f} tokens={tokens_processed:,}/{args.max_train_tokens:,} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tok/s={throughput:.0f}",
                flush=True,
            )
        history.append(
            {
                "step": step,
                "tokens_processed": int(tokens_processed),
                "valid_tokens": int(n_valid),
                "loss": float(loss.item()),
                "lm_loss": float(lm_loss.item()),
                "depth_losses": {str(d): float(losses[d].item()) for d in depths},
                "monotonic_penalty": float(mono.item()),
                "preservation_penalty": float(pres.item()),
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if tokens_processed - last_save_tokens >= args.save_every_tokens:
            path = os.path.join(args.workdir, f"checkpoint_tokens_{tokens_processed}.pt")
            save_checkpoint(
                path, model, optimizer, None, step, tokens_processed, args,
                tokenizer_info, model_cfg, mixture.token_accounting,
                mixture.example_accounting, history,
            )
            last_save_tokens = tokens_processed

        if tokens_processed - last_val_tokens >= args.val_every_tokens:
            lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=args.val_batches)
            print(
                f"[val @ {tokens_processed:,}] LM fast CE={lm['ce']:.4f} PPL={lm['ppl']:.2f}",
                flush=True,
            )
            last_val_tokens = tokens_processed
            model.train()

    duration = time.time() - start_time
    final_ckpt = os.path.join(args.workdir, "checkpoint_final.pt")
    save_checkpoint(
        final_ckpt, model, optimizer, None, step, tokens_processed, args,
        tokenizer_info, model_cfg, mixture.token_accounting,
        mixture.example_accounting, history,
    )
    reload_model = DualStateModel(model_cfg).to(device=device, dtype=dtype)
    reload_model.load_state_dict(torch.load(final_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    reload_model.eval()
    model.eval()
    with torch.no_grad():
        probe = torch.randint(1, max(2, model_cfg.vocab_size - 1), (1, 16), device=device)
        probe_mask = torch.ones_like(probe)
        reload_diff = float(
            (model(probe, probe_mask).logits - reload_model(probe, probe_mask).logits).abs().max().item()
        )
    final_lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=None)
    summary = {
        "experiment": os.path.basename(args.workdir),
        "loss_mode": args.loss_mode,
        "seed": args.seed,
        "completed_steps": step,
        "tokens_processed": tokens_processed,
        "duration_sec": duration,
        "throughput_tokens_per_sec": tokens_processed / max(1e-9, duration),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "weights_mb": parameter_bytes(model) / 1e6,
        "checkpoint_reload_max_diff": reload_diff,
        "mixture_token_accounting": dict(mixture.token_accounting),
        "mixture_example_accounting": dict(mixture.example_accounting),
        "init_lm": init_lm,
        "final_lm": final_lm,
        "depths": depths,
        "monotonic_weight": args.monotonic_weight,
        "preservation_weight": args.preservation_weight,
        "train_config": vars(args),
    }
    with open(os.path.join(args.workdir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    with open(os.path.join(args.workdir, "metrics_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle)
    print(json.dumps({k: summary[k] for k in [
        "loss_mode", "seed", "completed_steps", "tokens_processed", "duration_sec",
        "checkpoint_reload_max_diff", "mixture_token_accounting", "final_lm",
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
