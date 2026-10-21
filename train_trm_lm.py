#!/usr/bin/env python3
"""Bounded training/verification loop for TRM-LM and the causal baseline.

This entrypoint is deliberately independent of Hydra and ``pretrain.py``.  The
reference Samsung TRM training path is not modified and is not used here.

Token-budget convention
-----------------------
The training budget is measured in *eligible next-token targets*: positions
whose label is not ``-100``.  Each batch contributes
``(labels != -100).sum()``.  If the final batch would exceed the budget, its
labels are truncated in row-major order so that the run processes exactly the
requested number of targets.  The same truncated labels are used for the loss
and the token count.  All arms share the same deterministic data stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    TinyStoriesLMDataset,
    collate_lm_batch,
    count_valid_targets,
    find_default_parquet,
    iter_packed_examples,
    next_token_cross_entropy,
    prepare_split_manifest,
    train_tokenizer,
    truncate_labels_to_budget,
    verify_parquet,
)
from models.recursive_reasoning.causal_transformer_lm import CausalTransformerLM
from models.recursive_reasoning.trm_lm import TRM_LM, TRM_LMConfig

try:  # Linux; used only for a best-effort peak-RSS measurement.
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded TRM-LM / causal-transformer TinyStories run"
    )
    parser.add_argument("--parquet", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--arch", type=str, default="trm_lm", choices=["trm_lm", "transformer"])
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--n-refine", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--save-every-tokens", type=int, default=None)
    parser.add_argument("--val-every-tokens", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-train-docs", type=int, default=200)
    parser.add_argument("--max-val-docs", type=int, default=50)
    parser.add_argument("--max-scan-rows", type=int, default=40_000)
    parser.add_argument("--tokenizer-docs", type=int, default=5_000)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-generation-prompts", type=int, default=5)
    parser.add_argument("--generate-new-tokens", type=int, default=64)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA/ROCm device requested but torch.cuda.is_available() is False"
            )
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "unknown")
        arch_list = torch.cuda.get_arch_list()
        print(
            f"[device] {props.name} arch={arch} hip={torch.version.hip} "
            f"vram={props.total_memory / 1e9:.2f} GB",
            flush=True,
        )
        print(f"[device] torch arch list: {arch_list}", flush=True)
        if "gfx1032" in str(arch) and "gfx1032" not in arch_list:
            if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
                raise RuntimeError(
                    "The installed ROCm Torch build has no gfx1032 kernels and "
                    "would hard-abort. Re-run with "
                    "HSA_OVERRIDE_GFX_VERSION=10.3.0 in the test environment "
                    "(opt-in, unofficial ISA mapping), or use --device cpu."
                )
            print(
                "[device] gfx1032 is using HSA_OVERRIDE_GFX_VERSION="
                f"{os.environ['HSA_OVERRIDE_GFX_VERSION']} (unofficial override)",
                flush=True,
            )
    elif device.type != "cpu":
        raise RuntimeError(f"Unsupported device type: {device.type}")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested but is not supported on this GPU")
        return torch.bfloat16
    raise RuntimeError(f"Unsupported dtype: {requested}")


def build_model_config(args: argparse.Namespace, tokenizer) -> TRM_LMConfig:
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise RuntimeError("Tokenizer is missing <pad>")
    return TRM_LMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        expansion=args.expansion,
        n_refine=args.n_refine,
        tie_embeddings=True,
        pad_token_id=pad_id,
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,
    )


def build_model(
    args: argparse.Namespace,
    model_config: TRM_LMConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    if args.arch == "trm_lm":
        model = TRM_LM(model_config)
    elif args.arch == "transformer":
        model = CausalTransformerLM(model_config)
    else:  # pragma: no cover - argparse restricts the choices
        raise RuntimeError(f"Unsupported architecture: {args.arch}")
    model = model.to(device=device, dtype=dtype)
    # Keep the small RoPE cache in fp32 even when the model runs in bf16.
    model.rotary_emb.float()
    return model


def make_loader(
    config: TinyStoriesConfig,
    manifest: Dict[str, object],
    tokenizer,
    split: str,
    batch_size: int,
    max_docs: int,
    device: torch.device,
    skip_tokens: int = 0,
) -> DataLoader:
    dataset = TinyStoriesLMDataset(
        config=config,
        manifest=manifest,
        tokenizer=tokenizer,
        split=split,
        max_docs=max_docs,
        skip_tokens=skip_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_lm_batch,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = next_token_cross_entropy(logits, batch["labels"])
            total += float(loss.detach().cpu())
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    model.train()
    if count == 0:
        raise RuntimeError("No validation batches were produced")
    return total / count


def count_loader_targets(loader: DataLoader) -> int:
    total = 0
    for batch in loader:
        total += count_valid_targets(batch["labels"])
    return total


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return None


def next_multiple(current: int, interval: Optional[int]) -> Optional[int]:
    if interval is None or interval <= 0:
        return None
    return ((current // interval) + 1) * interval


def write_metric(handle, record: Dict[str, object]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path: str,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tokens_processed: int,
    model_config: TRM_LMConfig,
    train_config: Dict[str, object],
    tokenizer_config: Dict[str, object],
    manifest: Dict[str, object],
    device: torch.device,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "tokens_processed": tokens_processed,
        "model_config": model_config.config_dict(),
        "train_config": train_config,
        "tokenizer_config": tokenizer_config,
        "manifest": manifest,
        "rng_state": rng_state(),
        "device_at_save": str(device),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def first_batch(loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    for batch in loader:
        return move_batch(batch, device)
    raise RuntimeError("Loader produced no batches")


def collect_generation_prompts(
    data_config: TinyStoriesConfig,
    manifest: Dict[str, object],
    tokenizer,
    num_prompts: int,
) -> List[Dict[str, object]]:
    prompts: List[Dict[str, object]] = []
    for example in iter_packed_examples(
        data_config,
        manifest,
        tokenizer,
        "val",
        max_docs=num_prompts,
        max_chunks=num_prompts,
    ):
        real_len = int(example["attention_mask"].sum().item())
        take = min(8, real_len)
        ids = example["input_ids"][:take].clone()
        prompts.append(
            {
                "ids": ids.tolist(),
                "text": tokenizer.decode(ids.tolist()),
            }
        )
        if len(prompts) >= num_prompts:
            break
    if not prompts:
        raise RuntimeError("No generation prompts were produced")
    return prompts


def measure_forward_latency(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    iterations: int = 5,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(batch["input_ids"], batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            model(batch["input_ids"], batch["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
    model.train()
    return 1000.0 * elapsed / iterations


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 4)
        args.max_train_docs = min(args.max_train_docs, 512)
        args.max_val_docs = min(args.max_val_docs, 128)
        args.max_scan_rows = min(args.max_scan_rows or 40_000, 40_000)
        print("[smoke-test] bounded run enforced", flush=True)
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.arch == "transformer" and args.n_refine != 1:
        raise SystemExit("--arch transformer requires --n-refine 1")
    if args.max_train_tokens is not None and args.max_train_tokens <= 0:
        raise SystemExit("--max-train-tokens must be positive")

    total_start = time.time()
    set_seed(args.seed)
    parquet = args.parquet or find_default_parquet()
    parquet_info = verify_parquet(parquet)
    print(f"[data] parquet={parquet_info['path']}", flush=True)
    print(
        f"[data] size={parquet_info['size_bytes']} rows={parquet_info['num_rows']} "
        f"matches_audit_size={parquet_info['matches_audit_size']} "
        f"matches_audit_rows={parquet_info['matches_audit_rows']}",
        flush=True,
    )
    if parquet_info["size_bytes"] != 672_825_299:
        print("[data] WARNING: size differs from the audited 672,825,299 bytes", flush=True)

    workdir = args.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="trm_lm_run_")
        print(f"[run] no --workdir given; using temporary {workdir}", flush=True)
    else:
        if os.path.exists(workdir) and os.listdir(workdir) and not args.resume:
            raise SystemExit(
                f"Refusing to overwrite non-empty workdir {workdir!r}; "
                "choose a new directory or pass --resume"
            )
        os.makedirs(workdir, exist_ok=True)

    data_config = TinyStoriesConfig(
        parquet_path=parquet,
        workdir=workdir,
        seq_len=args.seq_len,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_train_docs=args.tokenizer_docs,
        seed=args.seed,
        max_scan_rows=args.max_scan_rows,
        max_train_docs=args.max_train_docs,
        max_val_docs=args.max_val_docs,
    )
    manifest = prepare_split_manifest(data_config)
    tokenizer, tokenizer_config = train_tokenizer(data_config, manifest)
    print(
        f"[tokenizer] vocab={tokenizer_config['vocab_size']} "
        f"requested={tokenizer_config['requested_vocab_size']} "
        f"train_docs={tokenizer_config['train_docs']} "
        f"special_ids={tokenizer_config['special_token_ids']}",
        flush=True,
    )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_config = build_model_config(args, tokenizer)
    model = build_model(args, model_config, device, dtype)
    print(
        f"[model] arch={args.arch} n_refine={args.n_refine} "
        f"params={model.num_parameters:,} config={model.config_dict()}",
        flush=True,
    )
    print(f"[model] device={device} dtype={dtype}", flush=True)

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported optimizer: {args.optimizer}")

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

    start_step = 0
    tokens_processed = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        saved_config = checkpoint.get("model_config")
        if saved_config is not None and saved_config != model_config.config_dict():
            raise RuntimeError(
                "Checkpoint model_config does not match the current model config; "
                "refusing to resume an incompatible run"
            )
        start_step = int(checkpoint["step"])
        tokens_processed = int(checkpoint.get("tokens_processed", 0))
        print(
            f"[resume] loaded {args.resume} at step {start_step}, "
            f"tokens {tokens_processed}",
            flush=True,
        )

    train_loader = make_loader(
        data_config, manifest, tokenizer, "train",
        args.batch_size, args.max_train_docs, device,
        skip_tokens=tokens_processed,
    )
    start_tokens = tokens_processed
    val_loader = make_loader(
        data_config, manifest, tokenizer, "val",
        args.batch_size, args.max_val_docs, device,
    )
    val_target_tokens = count_loader_targets(val_loader)
    print(f"[val] target_tokens={val_target_tokens}", flush=True)

    tokenizer_hash = sha256_file(os.path.join(workdir, "tokenizer.json"))
    manifest_hash = sha256_file(os.path.join(workdir, "manifest.json"))
    train_config = {
        "args": vars(args),
        "parquet": parquet_info,
        "workdir": workdir,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "arch": args.arch,
        "n_refine": args.n_refine,
        "seed": args.seed,
        "max_train_tokens": args.max_train_tokens,
        "val_target_tokens": val_target_tokens,
        "tokenizer_sha256": tokenizer_hash,
        "manifest_sha256": manifest_hash,
        "split_ids_sha256": manifest.get("split_ids_sha256"),
        "git_commit": git_commit(),
    }
    with open(os.path.join(workdir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(train_config, handle, indent=2, sort_keys=True)
    with open(os.path.join(workdir, "model_config.json"), "w", encoding="utf-8") as handle:
        json.dump(model_config.config_dict(), handle, indent=2, sort_keys=True)

    metrics_path = os.path.join(workdir, "metrics.jsonl")
    metrics_handle = open(metrics_path, "a", encoding="utf-8")
    write_metric(
        metrics_handle,
        {
            "event": "run_start",
            "step": start_step,
            "tokens": tokens_processed,
            "arch": args.arch,
            "n_refine": args.n_refine,
            "params": model.num_parameters,
            "val_target_tokens": val_target_tokens,
            "tokenizer_sha256": tokenizer_hash,
            "manifest_sha256": manifest_hash,
        },
    )

    init_val_loss = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
    write_metric(
        metrics_handle,
        {
            "event": "init_val",
            "step": start_step,
            "tokens": tokens_processed,
            "val_loss": init_val_loss,
            "val_ppl": math.exp(min(init_val_loss, 50.0)),
        },
    )
    print(f"[val] init loss={init_val_loss:.6f}", flush=True)

    fixed_val_batch = first_batch(val_loader, device)
    generation_prompts = collect_generation_prompts(
        data_config, manifest, tokenizer, args.num_generation_prompts
    )

    next_val = next_multiple(tokens_processed, args.val_every_tokens)
    next_save = next_multiple(tokens_processed, args.save_every_tokens)

    if tokens_processed == 0:
        initial_checkpoint = os.path.join(workdir, "checkpoints", "tokens_0.pt")
        save_checkpoint(
            initial_checkpoint, model, optimizer, scheduler, start_step,
            tokens_processed, model_config, train_config, tokenizer_config,
            manifest, device,
        )
        write_metric(
            metrics_handle,
            {"event": "checkpoint", "step": start_step, "tokens": 0, "path": initial_checkpoint},
        )
        print(f"[checkpoint] {initial_checkpoint}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    step = start_step
    last_train_loss = None
    last_val_loss = init_val_loss
    last_val_tokens = tokens_processed
    train_start = time.time()

    for batch in train_loader:
        if args.max_train_tokens is not None and tokens_processed >= args.max_train_tokens:
            break
        if step >= args.max_steps:
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

        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = next_token_cross_entropy(logits, batch["labels"])
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step + 1}, "
                f"tokens {tokens_processed}: {loss}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        tokens_processed += n_valid
        last_train_loss = float(loss.detach().cpu())
        elapsed = time.time() - train_start
        write_metric(
            metrics_handle,
            {
                "event": "train",
                "step": step,
                "tokens": tokens_processed,
                "train_loss": last_train_loss,
                "grad_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
            },
        )

        while next_val is not None and tokens_processed >= next_val:
            last_val_loss = evaluate(
                model, val_loader, device, max_batches=args.max_val_batches
            )
            last_val_tokens = tokens_processed
            write_metric(
                metrics_handle,
                {
                    "event": "val",
                    "step": step,
                    "tokens": tokens_processed,
                    "val_loss": last_val_loss,
                    "val_ppl": math.exp(min(last_val_loss, 50.0)),
                    "elapsed_sec": time.time() - train_start,
                },
            )
            print(
                f"[val] step={step} tokens={tokens_processed} "
                f"loss={last_val_loss:.6f}",
                flush=True,
            )
            next_val += args.val_every_tokens

        while next_save is not None and tokens_processed >= next_save:
            checkpoint_path = os.path.join(
                workdir, "checkpoints", f"tokens_{tokens_processed}.pt"
            )
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, step,
                tokens_processed, model_config, train_config, tokenizer_config,
                manifest, device,
            )
            write_metric(
                metrics_handle,
                {
                    "event": "checkpoint",
                    "step": step,
                    "tokens": tokens_processed,
                    "path": checkpoint_path,
                },
            )
            print(f"[checkpoint] {checkpoint_path}", flush=True)
            next_save += args.save_every_tokens

        if args.max_train_tokens is not None and tokens_processed >= args.max_train_tokens:
            break

    train_elapsed = time.time() - train_start
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_training_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_training_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    else:
        peak_training_alloc = None
        peak_training_reserved = None

    if args.max_train_tokens is not None and tokens_processed != args.max_train_tokens:
        raise RuntimeError(
            f"Training stopped at {tokens_processed} eligible tokens, expected "
            f"exactly {args.max_train_tokens}. Check the data budget/max-steps."
        )

    final_val_loss = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
    write_metric(
        metrics_handle,
        {
            "event": "final_val",
            "step": step,
            "tokens": tokens_processed,
            "val_loss": final_val_loss,
            "val_ppl": math.exp(min(final_val_loss, 50.0)),
            "elapsed_sec": time.time() - train_start,
        },
    )
    print(f"[val] final loss={final_val_loss:.6f}", flush=True)

    final_checkpoint = os.path.join(
        workdir, "checkpoints", f"tokens_{tokens_processed}.pt"
    )
    if not os.path.exists(final_checkpoint):
        save_checkpoint(
            final_checkpoint, model, optimizer, scheduler, step,
            tokens_processed, model_config, train_config, tokenizer_config,
            manifest, device,
        )
        write_metric(
            metrics_handle,
            {
                "event": "checkpoint",
                "step": step,
                "tokens": tokens_processed,
                "path": final_checkpoint,
            },
        )
        print(f"[checkpoint] {final_checkpoint}", flush=True)

    # --- checkpoint round-trip on a fixed batch -------------------------
    model.eval()
    with torch.no_grad():
        logits_before = model(
            fixed_val_batch["input_ids"], fixed_val_batch["attention_mask"]
        )
    reloaded = build_model(args, model_config, device, dtype)
    checkpoint = load_checkpoint(final_checkpoint, reloaded, None, None, device)
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(
            fixed_val_batch["input_ids"], fixed_val_batch["attention_mask"]
        )
    max_diff = float((logits_before - logits_after).abs().max().cpu())
    if not math.isfinite(max_diff) or max_diff > 1e-2:
        raise RuntimeError(
            f"Checkpoint reload failed: max_abs_diff={max_diff}"
        )
    reload_step = int(checkpoint["step"])
    reload_tokens = int(checkpoint["tokens_processed"])
    print(
        f"[checkpoint] reload step={reload_step} tokens={reload_tokens} "
        f"max_abs_diff={max_diff:.3e}",
        flush=True,
    )

    # --- bounded greedy generation from fixed prompts -------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generation_start = time.time()
    generations = []
    generated_token_count = 0
    eos_id = tokenizer.token_to_id("<eos>")
    for prompt in generation_prompts:
        prompt_ids = torch.tensor(
            [prompt["ids"]], dtype=torch.long, device=device
        )
        generated = reloaded.generate(
            prompt_ids,
            max_new_tokens=args.generate_new_tokens,
            eos_token_id=eos_id,
        )
        new_ids = generated[0, prompt_ids.shape[1] :].tolist()
        generated_token_count += len(new_ids)
        generations.append(
            {
                "prompt_ids": prompt["ids"],
                "prompt_text": prompt["text"],
                "output_ids": generated[0].tolist(),
                "output_text": tokenizer.decode(generated[0].tolist()),
                "new_token_count": len(new_ids),
            }
        )
    generation_seconds = time.time() - generation_start
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_generation_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_generation_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    else:
        peak_generation_alloc = None
        peak_generation_reserved = None

    forward_latency_ms = measure_forward_latency(
        reloaded, fixed_val_batch, device, iterations=5
    )

    max_rss_mb = None
    if resource is not None:
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    optimizer_steps = step - start_step
    tokens_this_run = tokens_processed - start_tokens
    params = reloaded.num_parameters
    summary = {
        "arch": args.arch,
        "n_refine": args.n_refine,
        "params": params,
        "model_config": model_config.config_dict(),
        "tokenizer_config": tokenizer_config,
        "manifest_counts": manifest["counts"],
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "start_step": start_step,
        "final_step": step,
        "start_tokens": start_tokens,
        "tokens_processed": tokens_processed,
        "tokens_processed_this_run": tokens_this_run,
        "resumed_from": args.resume,
        "max_train_tokens": args.max_train_tokens,
        "optimizer_steps": optimizer_steps,
        "optimizer_steps_total": step,
        "effective_batch_size_sequences": args.batch_size,
        "mean_target_tokens_per_step": (
            tokens_this_run / optimizer_steps if optimizer_steps else None
        ),
        "mean_target_tokens_per_step_total": (
            tokens_processed / step if step else None
        ),
        "val_target_tokens": val_target_tokens,
        "init_val_loss": init_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_ppl": math.exp(min(final_val_loss, 50.0)),
        "last_train_loss": last_train_loss,
        "train_elapsed_sec": round(train_elapsed, 3),
        "total_elapsed_sec": round(time.time() - total_start, 3),
        "train_tokens_per_sec": (
            round(tokens_this_run / train_elapsed, 3) if train_elapsed > 0 else None
        ),
        "peak_training_memory_allocated_gb": (
            round(peak_training_alloc, 3) if peak_training_alloc is not None else None
        ),
        "peak_training_memory_reserved_gb": (
            round(peak_training_reserved, 3) if peak_training_reserved is not None else None
        ),
        "peak_generation_memory_allocated_gb": (
            round(peak_generation_alloc, 3) if peak_generation_alloc is not None else None
        ),
        "max_rss_mb": round(max_rss_mb, 1) if max_rss_mb is not None else None,
        "forward_latency_ms": round(forward_latency_ms, 3),
        "generation_seconds": round(generation_seconds, 3),
        "generated_tokens_per_second": (
            round(generated_token_count / generation_seconds, 3)
            if generation_seconds > 0
            else None
        ),
        "generation_settings": {
            "decoding": "greedy_argmax",
            "do_sample": False,
            "seed": args.seed,
            "max_new_tokens": args.generate_new_tokens,
            "eos_token_id": eos_id,
            "num_prompts": len(generation_prompts),
        },
        "prompts": generation_prompts,
        "generations": generations,
        "checkpoint_path": final_checkpoint,
        "checkpoint_reload_step": reload_step,
        "checkpoint_reload_tokens": reload_tokens,
        "checkpoint_max_abs_diff": max_diff,
        "tokenizer_sha256": tokenizer_hash,
        "manifest_sha256": manifest_hash,
        "split_ids_sha256": manifest.get("split_ids_sha256"),
        "metrics_path": metrics_path,
        "compute_proxy": {
            "param_token_product": params * tokens_processed,
            "param_token_pass_product": params * tokens_processed * args.n_refine,
            "approx_training_flops_proxy_6x": 6 * params * tokens_processed * args.n_refine,
            "definition": (
                "Param-token product; the pass product multiplies by n_refine. "
                "The 6x value is a rough training FLOPs proxy, not a measured "
                "hardware counter. R4 is not compute-matched to B1/R1."
            ),
        },
        "authorized_pilot": args.max_train_tokens == 5_000_000,
        "substantive_training_authorized": (
            "5M-token pilot only"
            if args.max_train_tokens == 5_000_000
            else "bounded verification only"
        ),
    }
    with open(os.path.join(workdir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    metrics_handle.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if args.max_train_tokens is not None:
        # Hard postcondition: the authorized budget was consumed exactly once.
        assert summary["tokens_processed"] == args.max_train_tokens
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # clear failure instead of a silent fallback
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
