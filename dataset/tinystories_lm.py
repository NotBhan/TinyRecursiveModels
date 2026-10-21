"""Dedicated TinyStories language-model data pipeline.

This module is intentionally independent of ``PuzzleDataset``.  Text is not
disguised as puzzle arrays; it is validated, split at the document level,
tokenized with a from-scratch BPE tokenizer, and packed into fixed-length
next-token blocks.

Document-boundary policy
------------------------
Every document is tokenized independently as ``<bos> document <eos>`` and then
sliced into consecutive blocks of ``seq_len`` tokens.  Each block is a separate
sample, so no attention is ever allowed to cross a document boundary and no
document can cross the train/validation split boundary.  The final incomplete
block of a document is padded and masked, not dropped.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
SPLIT_TO_ID = {"test": 0, "val": 1, "train": 2}
ID_TO_SPLIT = {v: k for k, v in SPLIT_TO_ID.items()}
AUDIT_SIZE_BYTES = 672_825_299
AUDIT_ROWS = 2_732_634


@dataclass
class TinyStoriesConfig:
    parquet_path: str
    workdir: str
    seq_len: int = 256
    tokenizer_vocab_size: int = 8192
    tokenizer_train_docs: int = 5000
    seed: int = 0
    split_policy: str = "row_index"
    test_rows: Tuple[int, int] = (0, 10_000)
    val_rows: Tuple[int, int] = (10_000, 20_000)
    train_rows: Tuple[int, Optional[int]] = (20_000, None)
    max_scan_rows: Optional[int] = None
    max_train_docs: Optional[int] = None
    max_val_docs: Optional[int] = None
    max_test_docs: Optional[int] = None
    min_doc_chars: int = 1

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.split_policy != "row_index":
            raise ValueError(
                "This implementation records a deterministic row-index split. "
                "split_policy must be 'row_index'."
            )
        if self.tokenizer_train_docs <= 0:
            raise ValueError("tokenizer_train_docs must be positive")


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pyarrow is required for the TinyStories LM pipeline. Install it in "
            "a project-local environment (see requirements-trm-lm.txt)."
        ) from exc
    return pq


def _require_tokenizers():
    try:
        import tokenizers  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
        from tokenizers import decoders, models, pre_tokenizers, trainers  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "tokenizers is required for the TinyStories LM pipeline. Install it "
            "in a project-local environment (see requirements-trm-lm.txt)."
        ) from exc
    return Tokenizer, models, pre_tokenizers, trainers, decoders


def find_default_parquet() -> str:
    """Locate the already-downloaded TinyStories Parquet file.

    No download is attempted.  If several snapshots exist, prefer the newest
    local file and let the caller verify its size/row count.
    """
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--karpathy--tinystories-gpt4-clean/"
        "snapshots/*/tinystories_gpt4_clean.parquet"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            "Could not find tinystories_gpt4_clean.parquet under the Hugging "
            "Face cache. Pass --parquet explicitly."
        )
    return matches[0]


def verify_parquet(path: str) -> Dict[str, object]:
    """Open the Parquet metadata and verify the expected text column."""
    pq = _require_pyarrow()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    parquet_file = pq.ParquetFile(path)
    schema_names = list(parquet_file.schema_arrow.names)
    if "text" not in schema_names:
        raise ValueError(f"Expected a 'text' column, found {schema_names}")
    info = {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "num_rows": int(parquet_file.metadata.num_rows),
        "num_row_groups": int(parquet_file.metadata.num_row_groups),
        "columns": schema_names,
        "created_by": parquet_file.metadata.created_by,
        "matches_audit_size": os.path.getsize(path) == AUDIT_SIZE_BYTES,
        "matches_audit_rows": int(parquet_file.metadata.num_rows) == AUDIT_ROWS,
    }
    return info


def validate_text(text: object, min_doc_chars: int = 1) -> Tuple[bool, Optional[str]]:
    """Deterministic malformed-record policy."""
    if not isinstance(text, str):
        return False, "not_a_string"
    if text.strip() == "":
        return False, "empty_or_whitespace"
    if "\x00" in text:
        return False, "contains_nul"
    if len(text) < min_doc_chars:
        return False, "too_short"
    return True, None


def document_id(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def split_name_for_row(row_index: int, config: TinyStoriesConfig) -> str:
    """Deterministic document-level split by row index (the card's policy)."""
    test_start, test_end = config.test_rows
    val_start, val_end = config.val_rows
    train_start, _ = config.train_rows
    if test_start <= row_index < test_end:
        return "test"
    if val_start <= row_index < val_end:
        return "val"
    if row_index >= train_start:
        return "train"
    raise ValueError(f"row {row_index} is outside the declared split ranges")


def _iter_parquet_rows(
    path: str, batch_size: int = 1024, max_rows: Optional[int] = None
) -> Iterator[Tuple[int, object]]:
    pq = _require_pyarrow()
    parquet_file = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for text in texts:
            if max_rows is not None and row_index >= max_rows:
                return
            yield row_index, text
            row_index += 1


def prepare_split_manifest(config: TinyStoriesConfig, force: bool = False) -> Dict[str, object]:
    """Stream the Parquet once and create a deterministic split/duplicate manifest.

    Duplicate policy: the first occurrence by row index wins; later occurrences
    are rejected.  Because the audited card places test and validation rows
    before train rows, this also prevents a duplicated document from entering
    training after appearing in validation.  All rejection counts are recorded.
    """
    os.makedirs(config.workdir, exist_ok=True)
    manifest_path = os.path.join(config.workdir, "manifest.json")
    split_ids_path = os.path.join(config.workdir, "split_ids.npy")

    if not force and os.path.exists(manifest_path) and os.path.exists(split_ids_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    info = verify_parquet(config.parquet_path)
    scan_limit = config.max_scan_rows
    if scan_limit is not None:
        scan_limit = min(scan_limit, int(info["num_rows"]))

    split_ids = np.full(scan_limit if scan_limit is not None else int(info["num_rows"]), -1, dtype=np.int8)
    seen_hashes: Dict[str, str] = {}
    reject_reasons: Dict[str, int] = {}
    duplicate_within = 0
    duplicate_cross = 0
    accepted = {"test": 0, "val": 0, "train": 0}
    examined = 0
    doc_id_samples: Dict[str, List[str]] = {"test": [], "val": [], "train": []}

    for row_index, text in _iter_parquet_rows(
        config.parquet_path, max_rows=scan_limit
    ):
        examined += 1
        split = split_name_for_row(row_index, config)
        ok, reason = validate_text(text, config.min_doc_chars)
        if not ok:
            reject_reasons[reason or "unknown"] = (
                reject_reasons.get(reason or "unknown", 0) + 1
            )
            continue

        doc_hash = document_id(text)  # type: ignore[arg-type]
        previous_split = seen_hashes.get(doc_hash)
        if previous_split is not None:
            if previous_split == split:
                duplicate_within += 1
            else:
                duplicate_cross += 1
            continue

        seen_hashes[doc_hash] = split
        split_ids[row_index] = SPLIT_TO_ID[split]
        accepted[split] += 1
        if len(doc_id_samples[split]) < 20:
            doc_id_samples[split].append(doc_hash)

    np.save(split_ids_path, split_ids)
    with open(split_ids_path, "rb") as handle:
        split_ids_sha256 = hashlib.sha256(handle.read()).hexdigest()

    manifest: Dict[str, object] = {
        "format_version": 1,
        "parquet": info,
        "config": asdict(config),
        "split_policy": {
            "name": "row_index",
            "seed": config.seed,
            "test_rows": list(config.test_rows),
            "val_rows": list(config.val_rows),
            "train_rows": list(config.train_rows),
            "document_level": True,
            "cross_document_attention": False,
            "duplicate_policy": "first_occurrence_by_row_index",
        },
        "scanned_rows": int(split_ids.shape[0]),
        "full_num_rows": int(info["num_rows"]),
        "counts": {
            "examined": examined,
            "accepted": accepted,
            "retained_total": int(sum(accepted.values())),
            "rejected_total": int(
                sum(reject_reasons.values()) + duplicate_within + duplicate_cross
            ),
            "reject_reasons": reject_reasons,
            "duplicate_within_split": duplicate_within,
            "duplicate_cross_split": duplicate_cross,
        },
        "document_id_samples": doc_id_samples,
        "split_ids_path": split_ids_path,
        "split_ids_sha256": split_ids_sha256,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def iter_selected_documents(
    config: TinyStoriesConfig,
    manifest: Dict[str, object],
    split: str,
    max_docs: Optional[int] = None,
) -> Iterator[Tuple[int, str]]:
    """Yield accepted documents for one split in deterministic row order."""
    if split not in SPLIT_TO_ID:
        raise ValueError(f"unknown split {split!r}")
    split_id = SPLIT_TO_ID[split]
    split_ids = np.load(manifest["split_ids_path"], mmap_mode="r")  # type: ignore[arg-type]
    yielded = 0
    for row_index, text in _iter_parquet_rows(
        config.parquet_path, max_rows=int(split_ids.shape[0])
    ):
        if split_ids[row_index] != split_id:
            continue
        ok, reason = validate_text(text, config.min_doc_chars)
        if not ok:  # Defensive: the manifest should already have excluded these.
            raise RuntimeError(
                f"Accepted document at row {row_index} became invalid: {reason}"
            )
        yield row_index, text  # type: ignore[misc]
        yielded += 1
        if max_docs is not None and yielded >= max_docs:
            return


def train_tokenizer(
    config: TinyStoriesConfig, manifest: Dict[str, object], force: bool = False
) -> Tuple[object, Dict[str, object]]:
    """Train a byte-level BPE tokenizer on training documents only."""
    Tokenizer, models, pre_tokenizers, trainers, decoders = _require_tokenizers()
    os.makedirs(config.workdir, exist_ok=True)
    tokenizer_path = os.path.join(config.workdir, "tokenizer.json")
    tokenizer_config_path = os.path.join(config.workdir, "tokenizer_config.json")

    if not force and os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(tokenizer_config_path, "r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)
        return tokenizer, tokenizer_config

    train_docs = 0

    def document_iterator() -> Iterator[str]:
        nonlocal train_docs
        for _, text in iter_selected_documents(
            config, manifest, "train", max_docs=config.tokenizer_train_docs
        ):
            train_docs += 1
            yield text

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(document_iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)

    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in special_ids.values()):
        raise RuntimeError(f"Tokenizer did not create all special tokens: {special_ids}")
    actual_vocab = tokenizer.get_vocab_size()
    tokenizer_config = {
        "type": "byte_level_bpe",
        "requested_vocab_size": config.tokenizer_vocab_size,
        "vocab_size": actual_vocab,
        "special_token_ids": special_ids,
        "train_split": "train",
        "train_docs": train_docs,
        "seed": config.seed,
        "parquet_path": config.parquet_path,
        "tokenizer_path": tokenizer_path,
    }
    with open(tokenizer_config_path, "w", encoding="utf-8") as handle:
        json.dump(tokenizer_config, handle, indent=2, sort_keys=True)
    return tokenizer, tokenizer_config


def load_tokenizer(workdir: str):
    Tokenizer, _, _, _, _ = _require_tokenizers()
    path = os.path.join(workdir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Tokenizer.from_file(path)


def encode_document(text: str, tokenizer, vocab_size: int) -> List[int]:
    """Encode one document with explicit BOS/EOS and validate ID bounds."""
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing <bos> or <eos>")
    ids = tokenizer.encode(text).ids
    if not ids:
        return []
    tokens = [bos_id] + list(ids) + [eos_id]
    if max(tokens) >= vocab_size or min(tokens) < 0:
        raise ValueError(
            f"Tokenizer emitted an ID outside [0, {vocab_size}): "
            f"min={min(tokens)}, max={max(tokens)}"
        )
    return tokens


def validate_next_token_alignment(
    input_ids: torch.Tensor, labels: torch.Tensor
) -> None:
    """Check that labels[t] == inputs[t+1] wherever a target is valid."""
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape")
    aligned_labels = labels[:, :-1]
    valid = aligned_labels != IGNORE_INDEX
    expected = input_ids[:, 1:]
    if not torch.equal(aligned_labels[valid], expected[valid]):
        mismatch = (aligned_labels[valid] != expected[valid]).nonzero()
        raise ValueError(
            "Malformed next-token alignment: labels[:, :-1] must equal "
            f"input_ids[:, 1:] at valid positions; {mismatch.shape[0]} mismatches"
        )


def count_valid_targets(labels: torch.Tensor) -> int:
    """Number of eligible next-token targets (labels != IGNORE_INDEX)."""
    return int((labels != IGNORE_INDEX).sum().item())


def _valid_flat_indices(labels: torch.Tensor) -> torch.Tensor:
    valid = labels.reshape(-1) != IGNORE_INDEX
    return valid.nonzero(as_tuple=False).squeeze(1)


def truncate_labels_to_budget(
    labels: torch.Tensor, remaining: int
) -> Tuple[torch.Tensor, int]:
    """Keep the first ``remaining`` valid targets in row-major order.

    Excess labels are set to IGNORE_INDEX.  This is the documented final-batch
    token-budget convention; the kept count is returned.
    """
    if remaining < 0:
        raise ValueError("remaining must be >= 0")
    indices = _valid_flat_indices(labels)
    total = int(indices.numel())
    if remaining >= total:
        return labels, total
    output = labels.clone().reshape(-1)
    output[indices[remaining:]] = IGNORE_INDEX
    return output.view_as(labels), int(remaining)


def drop_leading_valid_targets(
    labels: torch.Tensor, drop: int
) -> Tuple[torch.Tensor, int]:
    """Drop the first ``drop`` valid targets in row-major order.

    Used for deterministic resume: labels before the resume point are set to
    IGNORE_INDEX while the inputs remain available as context.
    """
    if drop < 0:
        raise ValueError("drop must be >= 0")
    indices = _valid_flat_indices(labels)
    total = int(indices.numel())
    if drop >= total:
        return torch.full_like(labels, IGNORE_INDEX), 0
    output = labels.clone().reshape(-1)
    output[indices[:drop]] = IGNORE_INDEX
    return output.view_as(labels), total - drop


def pack_document_tokens(
    tokens: Sequence[int],
    *,
    document_hash: str,
    row_index: int,
    seq_len: int,
    pad_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> Iterator[Dict[str, object]]:
    """Yield fixed-length next-token blocks for one document.

    ``labels[t] = tokens[chunk_start + t + 1]``.  The last target of the final
    chunk is ``ignore_index`` because there is no token after ``<eos>``.
    """
    if len(tokens) == 0:
        return
    chunk_index = 0
    for start in range(0, len(tokens), seq_len):
        chunk = list(tokens[start : start + seq_len])
        target = list(tokens[start + 1 : start + seq_len + 1])
        real_len = len(chunk)

        input_ids = np.full(seq_len, pad_id, dtype=np.int64)
        labels = np.full(seq_len, ignore_index, dtype=np.int64)
        attention_mask = np.zeros(seq_len, dtype=np.int64)

        input_ids[:real_len] = chunk
        if target:
            labels[: len(target)] = target
        attention_mask[:real_len] = 1

        input_tensor = torch.from_numpy(input_ids)
        label_tensor = torch.from_numpy(labels)
        mask_tensor = torch.from_numpy(attention_mask)
        validate_next_token_alignment(
            input_tensor.unsqueeze(0), label_tensor.unsqueeze(0)
        )

        yield {
            "input_ids": input_tensor,
            "labels": label_tensor,
            "attention_mask": mask_tensor,
            "row_index": row_index,
            "document_id": document_hash,
            "chunk_index": chunk_index,
        }
        chunk_index += 1


def iter_packed_examples(
    config: TinyStoriesConfig,
    manifest: Dict[str, object],
    tokenizer,
    split: str,
    max_docs: Optional[int] = None,
    max_chunks: Optional[int] = None,
    skip_tokens: int = 0,
) -> Iterator[Dict[str, object]]:
    """Stream tokenized/packed blocks for one split.

    ``skip_tokens`` skips that many eligible target tokens from the front of
    the deterministic stream, using the same row-major convention as the
    training token budget.  This makes checkpoint resume exact.
    """
    if skip_tokens < 0:
        raise ValueError("skip_tokens must be >= 0")
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise RuntimeError("Tokenizer has no <pad> token")
    chunk_count = 0
    remaining_skip = int(skip_tokens)
    for row_index, text in iter_selected_documents(
        config, manifest, split, max_docs=max_docs
    ):
        tokens = encode_document(text, tokenizer, vocab_size)
        for example in pack_document_tokens(
            tokens,
            document_hash=document_id(text),
            row_index=row_index,
            seq_len=config.seq_len,
            pad_id=pad_id,
        ):
            if remaining_skip > 0:
                valid = count_valid_targets(example["labels"])
                if remaining_skip >= valid:
                    remaining_skip -= valid
                    continue
                example["labels"], _ = drop_leading_valid_targets(
                    example["labels"], remaining_skip
                )
                remaining_skip = 0
            yield example
            chunk_count += 1
            if max_chunks is not None and chunk_count >= max_chunks:
                return


class TinyStoriesLMDataset(IterableDataset):
    """Streaming packed dataset; deterministic row order, no cross-doc packing."""

    def __init__(
        self,
        config: TinyStoriesConfig,
        manifest: Dict[str, object],
        tokenizer,
        split: str,
        max_docs: Optional[int] = None,
        max_chunks: Optional[int] = None,
        skip_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.split = split
        self.max_docs = max_docs
        self.max_chunks = max_chunks
        self.skip_tokens = skip_tokens

    def __iter__(self) -> Iterator[Dict[str, object]]:
        yield from iter_packed_examples(
            self.config,
            self.manifest,
            self.tokenizer,
            self.split,
            max_docs=self.max_docs,
            max_chunks=self.max_chunks,
            skip_tokens=self.skip_tokens,
        )


def collate_lm_batch(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """Stack tensor fields; discard provenance strings for model consumption."""
    if not batch:
        raise ValueError("empty batch")
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),  # type: ignore[list-item]
        "labels": torch.stack([item["labels"] for item in batch]),  # type: ignore[list-item]
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch]  # type: ignore[list-item]
        ),
    }


def next_token_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Numerically stable CE over vocabulary logits with shifted labels."""
    if logits.dim() != 3:
        raise ValueError("logits must be [batch, seq_len, vocab_size]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits' first two dimensions")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
