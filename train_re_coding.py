"""Train RE:Inference on a code-heavy mixture with a code-aware tokenizer.

Stages:
  --stage backbone   fixed-depth LM training on code + instructions + stories
  --stage controller adaptive multi-depth training with the halting objective

The script is additive. It does not modify the historical pilot checkpoints or
the frozen Samsung TRM reference.

Example (backbone):
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. python train_re_coding.py \
    --stage backbone --workdir experiments/re_coding_adaptive/backbone_3m \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --max-train-tokens 3000000 --fixed-depth 2

Example (controller fine-tune from a backbone checkpoint):
  HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONPATH=. python train_re_coding.py \
    --stage controller --workdir experiments/re_coding_adaptive/controller_500k \
    --tokenizer-workdir experiments/re_coding_adaptive/tokenizer \
    --init-checkpoint experiments/re_coding_adaptive/backbone_3m/checkpoint_final.pt \
    --max-train-tokens 500000 --max-high-cycles 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.code_pipeline import (
    CodePipelineConfig,
    MBPPInstructionDataset,
    PackedCodeDataset,
    QuotaMixtureDataset,
    build_mbpp_eval_records,
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
from evaluators.re_eval import (
    aggregate_results,
    evaluate_language_modeling,
    evaluate_mbpp,
    parameter_bytes,
    peak_memory_context,
)
from models.recursive_reasoning.trm_lm_adaptive import (
    DualStateConfig,
    DualStateModel,
    per_position_cross_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["backbone", "controller"], default="backbone")
    parser.add_argument("--workdir", default="experiments/re_coding_adaptive/backbone_smoke")
    parser.add_argument("--tokenizer-workdir", default="experiments/re_coding_adaptive/tokenizer")
    parser.add_argument("--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1")
    parser.add_argument("--mbpp-dir", default="data/coding/mbpp")
    parser.add_argument("--mbpp-config", default="full")
    parser.add_argument("--init-checkpoint", default=None)

    # Model
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
    parser.add_argument("--gradient-strategy", default="full")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="controller stage only: freeze all parameters except the controller",
    )

    # Data
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--code-lm-weight", type=float, default=0.70)
    parser.add_argument("--instruction-weight", type=float, default=0.20)
    parser.add_argument("--lang-weight", type=float, default=0.10)
    parser.add_argument("--stdlib-instruction-weight", type=float, default=0.0)
    parser.add_argument("--stdlib-max-functions", type=int, default=20_000)
    parser.add_argument("--max-code-files", type=int, default=4000)
    parser.add_argument("--max-train-docs", type=int, default=4000)
    parser.add_argument("--max-val-docs", type=int, default=256)

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-tokens", type=int, default=3_000_000)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--fixed-depth", type=int, default=2)
    parser.add_argument("--depth-sampling", choices=["uniform", "max"], default="uniform")
    parser.add_argument("--controller-loss-weight", type=float, default=0.25)
    parser.add_argument("--cost-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)

    # Device / dtype
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")

    # Evaluation / checkpointing
    parser.add_argument("--save-every-tokens", type=int, default=500_000)
    parser.add_argument("--val-every-tokens", type=int, default=500_000)
    parser.add_argument("--eval-coding-tasks", type=int, default=45)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--context-cap", type=int, default=256)
    parser.add_argument("--sandbox-python", default="/usr/bin/python3.14")
    parser.add_argument("--skip-code-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def lr_at(step: int, args: argparse.Namespace, total_steps_estimate: int) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps_estimate - args.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine)


def save_checkpoint(
    path: str,
    model: DualStateModel,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Optional[dict],
    step: int,
    tokens_processed: int,
    args: argparse.Namespace,
    tokenizer_info: dict,
    model_config: DualStateConfig,
    mixture_token_accounting: dict,
    mixture_example_accounting: dict,
    history: List[Dict[str, Any]],
) -> None:
    payload = {
        "step": step,
        "tokens_processed": tokens_processed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
        "model_config": model_config.config_dict(),
        "train_config": vars(args),
        "tokenizer_info": tokenizer_info,
        "mixture_token_accounting": dict(mixture_token_accounting),
        "mixture_example_accounting": dict(mixture_example_accounting),
        "metrics_history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_datasets(args: argparse.Namespace, tokenizer: Any):
    with open(os.path.join(args.tinystories_workdir, "manifest.json"), "r", encoding="utf-8") as handle:
        ts_manifest = json.load(handle)
    ts_cfg = TinyStoriesConfig(
        parquet_path=ts_manifest["parquet"]["path"],
        workdir=args.tinystories_workdir,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    lang_train = TinyStoriesLMDataset(
        ts_cfg, ts_manifest, tokenizer, split="train", max_docs=args.max_train_docs
    )
    lang_val = TinyStoriesLMDataset(
        ts_cfg, ts_manifest, tokenizer, split="val", max_docs=args.max_val_docs
    )
    code_cfg = CodePipelineConfig(
        seq_len=args.seq_len,
        mbpp_dir=args.mbpp_dir,
        mbpp_config=args.mbpp_config,
        max_code_files=args.max_code_files,
    )
    code_lm = PackedCodeDataset(tokenizer, code_cfg)
    instruction = MBPPInstructionDataset(
        tokenizer,
        code_cfg,
        split="train",
        task_ids=getattr(args, "mbpp_task_ids", None),
        prefix_prob=getattr(args, "prefix_prob", 0.0),
        prefix_max_fraction=getattr(args, "prefix_max_fraction", 0.75),
        prefix_seed=getattr(args, "seed", 1337),
    )
    quota = max(1, args.max_train_tokens)
    weights = {
        "code_lm": max(0.0, args.code_lm_weight),
        "instruction": max(0.0, args.instruction_weight),
        "lang": max(0.0, args.lang_weight),
        "stdlib_instruction": max(0.0, args.stdlib_instruction_weight),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    sources = [
        ("code_lm", code_lm, int(quota * weights["code_lm"] / total_weight), False),
        ("instruction", instruction, int(quota * weights["instruction"] / total_weight), True),
        ("lang", lang_train, int(quota * weights["lang"] / total_weight), False),
    ]
    if weights["stdlib_instruction"] > 0:
        from dataset.code_pipeline import StdlibInstructionDataset

        sources.append(
            (
                "stdlib_instruction",
                StdlibInstructionDataset(
                    tokenizer, code_cfg, max_functions=args.stdlib_max_functions
                ),
                int(quota * weights["stdlib_instruction"] / total_weight),
                True,
            )
        )
    mixture = QuotaMixtureDataset(sources, max_tokens=args.max_train_tokens, seed=args.seed)
    return mixture, lang_val, code_cfg, ts_manifest


def controller_loss(
    model: DualStateModel,
    batch: Dict[str, torch.Tensor],
    max_depth: int,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depths = list(range(1, max_depth + 1))
    if args.freeze_backbone:
        with torch.no_grad():
            depth_out = model.forward_depths(
                batch["input_ids"],
                batch["attention_mask"],
                depths=depths,
                max_depth=max_depth,
                evaluate_controller=False,
            )
            pos_loss = {
                d: per_position_cross_entropy(
                    depth_out.logits_by_depth[d], batch["labels"]
                )
                for d in depths
            }
            loss_by_depth = {
                d: next_token_cross_entropy(
                    depth_out.logits_by_depth[d], batch["labels"]
                )
                for d in depths
            }
        lm_loss = loss_by_depth[max_depth].detach()
        controller_logits_by_depth = {
            d: model.controller(
                depth_out.state_by_depth[d][0].detach(),
                depth_out.state_by_depth[d][1].detach(),
                batch["attention_mask"],
                depth=d,
                max_depth=max_depth,
                remaining_budget_fraction=None,
            )
            for d in range(1, max_depth)
        }
    else:
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
        if args.depth_sampling == "uniform":
            sampled = random.randint(1, max_depth)
        else:
            sampled = max_depth
        lm_loss = loss_by_depth[sampled]
        pos_loss = {
            d: per_position_cross_entropy(depth_out.logits_by_depth[d], batch["labels"])
            for d in depths
        }
        controller_logits_by_depth = depth_out.controller_logits_by_depth

    valid = batch["labels"] != IGNORE_INDEX
    terms = []
    survival = torch.ones_like(batch["input_ids"], dtype=torch.float32)
    for d in range(1, max_depth):
        logits_d = controller_logits_by_depth[d]
        delta = (pos_loss[d] - pos_loss[d + 1]).detach()
        target = (delta > args.cost_threshold).float()
        bce = F.binary_cross_entropy_with_logits(logits_d, target, reduction="none")
        weight = survival * valid.float()
        terms.append((bce * weight).sum() / weight.sum().clamp_min(1.0))
        survival = survival * torch.sigmoid(logits_d.detach().float())
    ctrl = torch.stack(terms).mean() if terms else torch.zeros((), device=lm_loss.device)
    return lm_loss, ctrl


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.workdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32

    tokenizer = load_tokenizer(args.tokenizer_workdir)
    tokenizer_info_path = os.path.join(args.tokenizer_workdir, "tokenizer_config.json")
    tokenizer_info = json.load(open(tokenizer_info_path, "r", encoding="utf-8")) if os.path.exists(tokenizer_info_path) else {}
    vocab_size = tokenizer.get_vocab_size()

    mixture, lang_val, code_cfg, ts_manifest = build_datasets(args, tokenizer)
    train_loader = DataLoader(
        mixture, batch_size=args.batch_size, collate_fn=collate_lm_batch, num_workers=0
    )
    val_loader = DataLoader(
        lang_val, batch_size=args.batch_size, collate_fn=collate_lm_batch, num_workers=0
    )

    model_cfg = DualStateConfig(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    tokens_processed = 0
    history: List[Dict[str, Any]] = []
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and args.stage == "controller":
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        # The init checkpoint supplies weights, not this stage's token budget.
        # A controller calibration/backbone adaptation starts a fresh budget.
        tokens_processed = 0
        print(f"[init] loaded {args.init_checkpoint} (checkpoint tokens={ckpt.get('tokens_processed')})", flush=True)

    if args.stage == "controller" and args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("controller.")
        model.eval()
        model.controller.train()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[init] frozen backbone; trainable controller params={trainable:,}", flush=True)

    print(
        f"[init] params={model.num_parameters:,} weights={parameter_bytes(model)/1e6:.2f} MB "
        f"vocab={vocab_size} device={device} dtype={dtype} stage={args.stage}",
        flush=True,
    )

    # Language-model retention baseline and coding baseline.
    init_lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=16)
    print(f"[val:init] LM fast CE={init_lm['ce']:.4f} PPL={init_lm['ppl']:.2f}", flush=True)

    coding_records = build_mbpp_eval_records(code_cfg)["validation"] if not args.skip_code_eval else []
    init_coding = None
    if coding_records:
        with peak_memory_context(model, device) as mem:
            init_agg, _ = evaluate_mbpp(
                model,
                tokenizer,
                coding_records,
                mode="fast",
                n_high_cycles=None,
                max_new_tokens=args.max_new_tokens,
                context_cap=args.context_cap,
                limit=min(args.eval_coding_tasks, len(coding_records)),
                sandbox_python=args.sandbox_python,
            )
        init_coding = {"aggregate": init_agg, "memory": mem.__dict__}
        print(
            f"[val:init] MBPP fast syntax={init_agg.get('syntax_valid_rate',0):.1%} "
            f"pass@1={init_agg.get('pass_at_1',0):.1%} "
            f"peak_alloc={mem.peak_allocated_mb:.1f} MB",
            flush=True,
        )

    # Estimate total steps for the LR schedule.
    est_tokens_per_step = max(1.0, float(args.batch_size * args.seq_len * 0.85))
    est_steps = max(1, int(args.max_train_tokens / est_tokens_per_step))
    step = start_step
    run_start_tokens = tokens_processed
    start_time = time.time()
    last_val_tokens = tokens_processed
    last_save_tokens = tokens_processed

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

        if args.stage == "backbone":
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                mode="fixed",
                n_high_cycles=args.fixed_depth,
            )
            lm_loss = next_token_cross_entropy(out.logits, batch["labels"])
            ctrl_loss = torch.zeros((), device=device)
        else:
            lm_loss, ctrl_loss = controller_loss(
                model, batch, args.max_high_cycles, args
            )
        loss = lm_loss + (args.controller_loss_weight * ctrl_loss if args.stage == "controller" else 0.0)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        tokens_processed += n_valid
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - start_time
            throughput = (tokens_processed - run_start_tokens) / max(1e-9, elapsed)
            print(
                f"[step {step:5d}] lm={lm_loss.item():.4f} ctrl={ctrl_loss.item():.4f} "
                f"tokens={tokens_processed:,}/{args.max_train_tokens:,} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tok/s={throughput:.0f}",
                flush=True,
            )
        history.append(
            {
                "step": step,
                "lm_loss": float(lm_loss.item()),
                "controller_loss": float(ctrl_loss.item()),
                "tokens_processed": int(tokens_processed),
                "valid_tokens": int(n_valid),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "mixture_source": batch.get("mixture_source", "unknown") if isinstance(batch, dict) else "unknown",
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
            lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=16)
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

    # Reload verification.
    reload_model = DualStateModel(model_cfg).to(device=device, dtype=dtype)
    reload_model.load_state_dict(
        torch.load(final_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    )
    reload_model.eval()
    model.eval()
    with torch.no_grad():
        probe = torch.randint(1, max(2, vocab_size - 1), (1, 16), device=device)
        probe_mask = torch.ones_like(probe)
        diff = float((model(probe, probe_mask).logits - reload_model(probe, probe_mask).logits).abs().max().item())

    final_lm = evaluate_language_modeling(model, val_loader, device, mode="fast", max_batches=None)
    print(f"[val:final] LM fast CE={final_lm['ce']:.4f} PPL={final_lm['ppl']:.2f}", flush=True)

    final_coding: Dict[str, Any] = {}
    if coding_records and not args.skip_code_eval:
        configurations = [
            ("fast", "fast", None, args.context_cap),
            ("fixed_2", "fixed", 2, args.context_cap),
            ("fixed_max", "fixed", args.max_high_cycles, args.context_cap),
            ("adaptive", "adaptive", None, args.context_cap),
        ]
        for name, mode, depth, cap in configurations:
            with peak_memory_context(model, device) as mem:
                agg, _ = evaluate_mbpp(
                    model,
                    tokenizer,
                    coding_records,
                    mode=mode,
                    n_high_cycles=depth,
                    max_new_tokens=args.max_new_tokens,
                    context_cap=cap,
                    max_high_cycles=args.max_high_cycles if mode == "adaptive" else None,
                    limit=min(args.eval_coding_tasks, len(coding_records)),
                    sandbox_python=args.sandbox_python,
                )
            final_coding[name] = {"aggregate": agg, "memory": mem.__dict__}
            print(
                f"[val:final:{name}] syntax={agg.get('syntax_valid_rate',0):.1%} "
                f"pass@1={agg.get('pass_at_1',0):.1%} "
                f"latency={agg.get('mean_latency_ms',0):.1f} ms "
                f"peak_alloc={mem.peak_allocated_mb:.1f} MB",
                flush=True,
            )

    summary = {
        "experiment_id": os.path.basename(args.workdir),
        "stage": args.stage,
        "completed_steps": step,
        "tokens_processed": tokens_processed,
        "duration_sec": duration,
        "throughput_tokens_per_sec": (tokens_processed - run_start_tokens) / max(1e-9, duration),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": model.num_parameters,
        "weights_mb": parameter_bytes(model) / 1e6,
        "checkpoint_reload_max_diff": diff,
        "mixture_token_accounting": dict(mixture.token_accounting),
        "mixture_example_accounting": dict(mixture.example_accounting),
        "init_lm": init_lm,
        "final_lm": final_lm,
        "init_coding": init_coding,
        "final_coding": final_coding,
        "model_config": model_cfg.config_dict(),
        "tokenizer_info": tokenizer_info,
        "train_config": vars(args),
    }
    with open(os.path.join(args.workdir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    with open(os.path.join(args.workdir, "metrics_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle)
    print(json.dumps({k: summary[k] for k in [
        "completed_steps", "tokens_processed", "duration_sec",
        "throughput_tokens_per_sec", "model_parameters",
        "checkpoint_reload_max_diff", "mixture_token_accounting",
    ]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
