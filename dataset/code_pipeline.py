"""Code-focused data pipeline for RE:Inference coding experiments.

This module is additive. It does not modify the historical TinyStories pipeline
or the original Samsung TRM reference. It provides:

- a byte-level BPE tokenizer trained on a mixture of code and stories;
- a syntax-validated Python source corpus packed into causal LM blocks;
- MBPP instruction examples whose prompt includes the target function
  signature (without the body), so the task is well-posed;
- a deterministic token-quota mixture that reports realized token counts.

The signature-in-prompt protocol exists because 479/500 MBPP test problems do
not mention the function name that the hidden tests call. Without a signature,
pass@1 is not a meaningful measure of code synthesis. The body is never placed
in the prompt, and evaluation tests are never placed in the prompt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import sysconfig
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    count_valid_targets,
    iter_selected_documents,
    validate_next_token_alignment,
)

DELIMITER_PROBLEM = "### Problem:\n"
DELIMITER_SOLUTION = "\n### Solution:\n"

ASSERT_NAME_RE = re.compile(r"^\s*assert\s+([A-Za-z_]\w*)\s*\(")


def _parse_python(text: str) -> ast.AST:
    """Parse Python while suppressing stdlib invalid-escape warnings."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(text)


@dataclass
class CodePipelineConfig:
    seq_len: int = 256
    tokenizer_vocab_size: int = 8192
    tokenizer_story_docs: int = 4000
    tokenizer_code_bytes: int = 8_000_000
    tokenizer_seed: int = 1337
    code_roots: Tuple[str, ...] = (
        "/usr/lib/python3.14",
    )
    max_code_files: int = 4000
    min_code_file_bytes: int = 200
    max_code_file_bytes: int = 200_000
    code_seed: int = 1337
    mbpp_dir: str = "data/coding/mbpp"
    mbpp_config: str = "full"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _require_tokenizers():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "tokenizers is required; install requirements-trm-lm.txt"
        ) from exc
    return Tokenizer, models, pre_tokenizers, trainers, decoders


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_python_sources(
    roots: Sequence[str],
    max_files: int,
    min_bytes: int,
    max_bytes: int,
    seed: int,
) -> Iterator[Tuple[str, str]]:
    """Yield deterministic, syntax-valid Python source files.

    Only files that parse with ``ast.parse`` are returned. Paths are sorted
    before sampling so runs are reproducible; the sample itself is seeded.
    """
    candidates: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in {"__pycache__", "site-packages", "dist-packages", ".git"}
            )
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if min_bytes <= size <= max_bytes:
                    candidates.append(path)
    candidates.sort()
    rng = random.Random(seed)
    rng.shuffle(candidates)
    yielded = 0
    for path in candidates:
        if yielded >= max_files:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            _parse_python(text)
        except (SyntaxError, ValueError):
            continue
        yielded += 1
        yield path, text


def train_code_tokenizer(
    workdir: str,
    config: CodePipelineConfig,
    tinystories_parquet: str,
    tinystories_manifest: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """Train a byte-level BPE tokenizer on stories plus syntax-valid Python."""
    Tokenizer, models, pre_tokenizers, trainers, decoders = _require_tokenizers()
    os.makedirs(workdir, exist_ok=True)
    tokenizer_path = os.path.join(workdir, "tokenizer.json")
    tokenizer_config_path = os.path.join(workdir, "tokenizer_config.json")
    if not force and os.path.exists(tokenizer_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    story_chars = 0
    code_chars = 0
    code_files: List[str] = []

    def iterator() -> Iterator[str]:
        nonlocal story_chars, code_chars
        # Stories first, bounded.
        for _, text in iter_selected_documents(
            _ManifestConfig(tinystories_parquet, config.seq_len),
            tinystories_manifest,
            "train",
            max_docs=config.tokenizer_story_docs,
        ):
            story_chars += len(text)
            if story_chars > config.tokenizer_code_bytes:
                break
            yield text
        # Code, bounded by characters.
        for path, text in iter_python_sources(
            config.code_roots,
            config.max_code_files,
            config.min_code_file_bytes,
            config.max_code_file_bytes,
            config.code_seed,
        ):
            if code_chars >= config.tokenizer_code_bytes:
                break
            code_files.append(path)
            code_chars += len(text)
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
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    info = {
        "type": "byte_level_bpe_code_mixed",
        "requested_vocab_size": config.tokenizer_vocab_size,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": special_ids,
        "story_chars": story_chars,
        "code_chars": code_chars,
        "code_files": len(code_files),
        "code_file_sample": code_files[:20],
        "seed": config.tokenizer_seed,
        "tinystories_parquet": tinystories_parquet,
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": _sha256_file(tokenizer_path),
    }
    with open(tokenizer_config_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
    return info


class _ManifestConfig:
    """Minimal config object accepted by iter_selected_documents."""

    def __init__(self, parquet_path: str, seq_len: int) -> None:
        self.parquet_path = parquet_path
        self.seq_len = seq_len
        self.min_doc_chars = 1
        self.max_scan_rows = None


# ---------------------------------------------------------------------------
# Packed code LM corpus
# ---------------------------------------------------------------------------


def pack_token_stream(
    tokens: Sequence[int], seq_len: int, pad_id: int
) -> List[Dict[str, Any]]:
    """Pack one document into full causal-LM blocks.

    Incomplete final blocks are dropped, not padded with a forced EOS. Dropping
    avoids training the model to emit EOS in the middle of a token sequence.
    """
    blocks: List[Dict[str, Any]] = []
    for start in range(0, len(tokens) - seq_len, seq_len):
        chunk = list(tokens[start : start + seq_len])
        labels = list(tokens[start + 1 : start + seq_len + 1])
        blocks.append(
            {
                "input_ids": torch.tensor(chunk, dtype=torch.int64),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "attention_mask": torch.ones(seq_len, dtype=torch.int64),
            }
        )
    return blocks


class PackedCodeDataset(IterableDataset):
    """Causal LM over syntax-validated Python files."""

    def __init__(
        self,
        tokenizer: Any,
        config: CodePipelineConfig,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_tokens = max_tokens
        self.seed = config.code_seed if seed is None else seed
        self.stats: Dict[str, int] = {
            "files_seen": 0,
            "files_used": 0,
            "target_tokens": 0,
            "blocks": 0,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        pad = self.tokenizer.token_to_id("<pad>")
        emitted = 0
        for path, text in iter_python_sources(
            self.config.code_roots,
            self.config.max_code_files,
            self.config.min_code_file_bytes,
            self.config.max_code_file_bytes,
            self.seed,
        ):
            self.stats["files_seen"] += 1
            ids = self.tokenizer.encode(text).ids
            if len(ids) < 8:
                continue
            tokens = [bos] + ids + [eos]
            blocks = pack_token_stream(tokens, self.config.seq_len, pad)
            if blocks:
                self.stats["files_used"] += 1
            for item in blocks:
                n = count_valid_targets(item["labels"])
                item["task_type"] = "code_lm"
                item["task_id"] = -1
                item["source"] = path
                item["valid_target_tokens"] = n
                self.stats["target_tokens"] += n
                self.stats["blocks"] += 1
                emitted += n
                yield item
                if self.max_tokens is not None and emitted >= self.max_tokens:
                    return


# ---------------------------------------------------------------------------
# MBPP instruction data
# ---------------------------------------------------------------------------


def _load_mbpp_split(mbpp_dir: str, config_name: str, split: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    path = os.path.join(mbpp_dir, config_name, f"{split}-00000-of-00001.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    text_col = "text" if config_name == "full" else "prompt"
    rows: List[Dict[str, Any]] = []
    for row in table.to_pylist():
        setup = row.get("test_setup_code")
        if setup is None and row.get("test_imports"):
            setup = "\n".join(row["test_imports"])
        rows.append(
            {
                "task_id": int(row["task_id"]),
                "problem_text": str(row[text_col]).strip(),
                "code": str(row["code"]).strip(),
                "test_list": list(row.get("test_list") or []),
                "test_setup_code": str(setup or ""),
            }
        )
    return rows


def _target_function_name(code: str, tests: Sequence[str]) -> Optional[str]:
    for test in tests:
        match = ASSERT_NAME_RE.match(test)
        if match:
            return match.group(1)
    try:
        tree = _parse_python(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def split_target_function(code: str, name: str) -> Optional[Tuple[str, str]]:
    """Return ``(context_with_signature, body)`` for the named function.

    ``context_with_signature`` contains everything before the body, including
    imports/helpers and the exact ``def`` line. ``body`` never appears in the
    prompt.
    """
    try:
        tree = _parse_python(code)
    except SyntaxError:
        return None
    target = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            target = node
            break
    if target is None:
        return None
    lines = code.splitlines(keepends=True)
    if target.lineno is None or target.end_lineno is None:
        return None
    signature_and_before = "".join(lines[: target.lineno])
    body = "".join(lines[target.lineno : target.end_lineno])
    if not body.strip():
        return None
    return signature_and_before, body


def normalize_for_dedup(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_instruction_example(
    tokenizer: Any,
    config: CodePipelineConfig,
    record: Dict[str, Any],
    prefix_tokens: int = 0,
) -> Optional[Dict[str, Any]]:
    """Build one prompt-masked MBPP function-completion example.

    ``prompt`` contains the problem statement, imports/helpers, and the exact
    target ``def`` line. ``target`` is the function body only. Returns ``None``
    when the target cannot be identified or the example exceeds ``seq_len``.

    ``prefix_tokens`` inserts that many canonical body tokens into the input
    (visible context) and masks them from the loss. This is prefix-augmented
    supervised fine-tuning: the model learns to continue a correct partial
    solution as well as to start one. ``prefix_tokens=0`` is the ordinary
    body-only objective and is bit-for-bit unchanged.
    """
    name = _target_function_name(record["code"], record.get("test_list", []))
    if name is None:
        return None
    split = split_target_function(record["code"], name)
    if split is None:
        return None
    context, body = split
    prompt = (
        f"{DELIMITER_PROBLEM}{record['problem_text'].strip()}"
        f"{DELIMITER_SOLUTION}{context}"
    )
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(body).ids
    if not target_ids:
        return None
    prefix_tokens = max(0, min(int(prefix_tokens), len(target_ids) - 1))
    supervised_ids = target_ids[prefix_tokens:]
    tokens = [bos] + list(prompt_ids) + list(target_ids) + [eos]
    if len(tokens) > config.seq_len:
        return None
    real_len = len(tokens)
    input_ids = np.full(config.seq_len, pad, dtype=np.int64)
    labels = np.full(config.seq_len, IGNORE_INDEX, dtype=np.int64)
    attention_mask = np.zeros(config.seq_len, dtype=np.int64)
    input_ids[:real_len] = tokens
    attention_mask[:real_len] = 1
    target_start = 1 + len(prompt_ids) + prefix_tokens
    for t in range(target_start - 1, real_len - 1):
        labels[t] = tokens[t + 1]
    inp = torch.from_numpy(input_ids)
    lab = torch.from_numpy(labels)
    mask = torch.from_numpy(attention_mask)
    validate_next_token_alignment(inp.unsqueeze(0), lab.unsqueeze(0))
    return {
        "input_ids": inp,
        "labels": lab,
        "attention_mask": mask,
        "task_type": "func_synthesis",
        "task_id": int(record.get("task_id", -1)),
        "valid_target_tokens": count_valid_targets(lab),
        "expected_name": name,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(supervised_ids),
        "full_body_tokens": len(target_ids),
        "prefix_tokens": prefix_tokens,
        "target_start": target_start,
    }


def line_prefix_token_lengths(body: str, tokenizer: Any) -> List[int]:
    """Cumulative token lengths at line boundaries, excluding the full body.

    Prefix-augmented training must not split a physical line, because the
    resulting partial statement is not a useful continuation target. The full
    body length is excluded so at least one supervised token always remains.
    """
    lines = body.splitlines(keepends=True)
    if len(lines) <= 1:
        return []
    cumulative = []
    running = 0
    for line in lines[:-1]:
        running += len(tokenizer.encode(line).ids)
        if running > 0:
            cumulative.append(running)
    total = len(tokenizer.encode(body).ids)
    return [value for value in sorted(set(cumulative)) if value < total]


class MBPPInstructionDataset(IterableDataset):
    """MBPP function synthesis with the signature in the prompt.

    The prompt is::

        ### Problem:
        <problem text>
        ### Solution:
        <imports/helpers>
        def <required_name>(<args>):

    The target is the function body. The body is never shown in the prompt.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: CodePipelineConfig,
        split: str = "train",
        quarantine_from: Sequence[str] = ("validation", "test"),
        include_full_code: bool = False,
        task_ids: Optional[Sequence[int]] = None,
        prefix_prob: float = 0.0,
        prefix_max_fraction: float = 0.75,
        prefix_seed: int = 1337,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.include_full_code = include_full_code
        self.task_ids = set(int(value) for value in task_ids) if task_ids else None
        self.prefix_prob = max(0.0, min(1.0, float(prefix_prob)))
        self.prefix_max_fraction = max(0.0, min(1.0, float(prefix_max_fraction)))
        self.prefix_seed = int(prefix_seed)
        self._epoch = 0
        self.records = _load_mbpp_split(config.mbpp_dir, config.mbpp_config, split)
        if self.task_ids is not None:
            self.records = [
                row for row in self.records if int(row["task_id"]) in self.task_ids
            ]
        if split == "train":
            eval_texts = set()
            eval_codes = set()
            for other in quarantine_from:
                for row in _load_mbpp_split(config.mbpp_dir, config.mbpp_config, other):
                    eval_texts.add(normalize_for_dedup(row["problem_text"]))
                    eval_codes.add(row["code"].strip())
            kept = []
            self.quarantined = 0
            for row in self.records:
                if normalize_for_dedup(row["problem_text"]) in eval_texts:
                    self.quarantined += 1
                    continue
                if row["code"].strip() in eval_codes:
                    self.quarantined += 1
                    continue
                kept.append(row)
            self.records = kept
        self.stats: Dict[str, int] = {
            "records": len(self.records),
            "examples": 0,
            "dropped_too_long": 0,
            "dropped_no_target": 0,
            "target_tokens": 0,
            "prefix_examples": 0,
        }

    def _choose_prefix(self, body: str, rng: random.Random) -> int:
        if self.prefix_prob <= 0.0 or rng.random() >= self.prefix_prob:
            return 0
        boundaries = line_prefix_token_lengths(body, self.tokenizer)
        if not boundaries:
            return 0
        body_ids = self.tokenizer.encode(body).ids
        max_prefix = max(1, int(len(body_ids) * self.prefix_max_fraction))
        candidates = [value for value in boundaries if value <= max_prefix]
        if not candidates:
            return 0
        return int(rng.choice(candidates))

    def _pack(
        self,
        prompt: str,
        target: str,
        task_id: int,
        prefix_tokens: int = 0,
    ) -> Optional[Dict[str, Any]]:
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        pad = self.tokenizer.token_to_id("<pad>")
        prompt_ids = self.tokenizer.encode(prompt).ids
        target_ids = self.tokenizer.encode(target).ids
        if not target_ids:
            return None
        prefix_tokens = max(0, min(int(prefix_tokens), len(target_ids) - 1))
        tokens = [bos] + list(prompt_ids) + list(target_ids) + [eos]
        if len(tokens) > self.config.seq_len:
            return None
        real_len = len(tokens)
        input_ids = np.full(self.config.seq_len, pad, dtype=np.int64)
        labels = np.full(self.config.seq_len, IGNORE_INDEX, dtype=np.int64)
        attention_mask = np.zeros(self.config.seq_len, dtype=np.int64)
        input_ids[:real_len] = tokens
        attention_mask[:real_len] = 1
        target_start = 1 + len(prompt_ids) + prefix_tokens
        for t in range(target_start - 1, real_len - 1):
            labels[t] = tokens[t + 1]
        inp = torch.from_numpy(input_ids)
        lab = torch.from_numpy(labels)
        mask = torch.from_numpy(attention_mask)
        validate_next_token_alignment(inp.unsqueeze(0), lab.unsqueeze(0))
        return {
            "input_ids": inp,
            "labels": lab,
            "attention_mask": mask,
            "task_type": "func_synthesis",
            "task_id": task_id,
            "valid_target_tokens": count_valid_targets(lab),
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids) - prefix_tokens,
            "full_body_tokens": len(target_ids),
            "prefix_tokens": prefix_tokens,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.prefix_seed + self._epoch)
        self._epoch += 1
        for row in self.records:
            code = row["code"]
            name = _target_function_name(code, row["test_list"])
            if name is None:
                self.stats["dropped_no_target"] += 1
                continue
            split = split_target_function(code, name)
            if split is None:
                self.stats["dropped_no_target"] += 1
                continue
            context, body = split
            prompt = (
                f"{DELIMITER_PROBLEM}{row['problem_text']}"
                f"{DELIMITER_SOLUTION}{context}"
            )
            prefix_tokens = self._choose_prefix(body, rng)
            item = self._pack(prompt, body, row["task_id"], prefix_tokens=prefix_tokens)
            if item is None:
                self.stats["dropped_too_long"] += 1
                continue
            self.stats["examples"] += 1
            self.stats["target_tokens"] += item["valid_target_tokens"]
            self.stats["prefix_examples"] += int(prefix_tokens > 0)
            yield item

            if self.include_full_code:
                full_prompt = f"{DELIMITER_PROBLEM}{row['problem_text']}{DELIMITER_SOLUTION}"
                full_item = self._pack(full_prompt, code, row["task_id"])
                if full_item is not None:
                    full_item["task_type"] = "full_code"
                    self.stats["examples"] += 1
                    self.stats["target_tokens"] += full_item["valid_target_tokens"]
                    self.stats["dropped_too_long"] += int(full_item is None)
                    yield full_item


class StdlibInstructionDataset(IterableDataset):
    """Synthetic instruction data from documented stdlib functions.

    The docstring is the problem statement; the function signature is in the
    prompt; the body (docstring removed) is the target. This increases natural
    language -> code supervision beyond the 371 MBPP training tasks without
    using any evaluation data. Only top-level functions with a docstring and a
    nonempty body are emitted. The function is reconstructed as a standalone
    definition, so a method that depends on ``self`` is skipped if it contains
    ``self`` in its body.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: CodePipelineConfig,
        max_functions: int = 20_000,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_functions = max_functions
        self.stats: Dict[str, int] = {
            "files": 0,
            "functions_seen": 0,
            "examples": 0,
            "dropped": 0,
            "target_tokens": 0,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        seen: set = set()
        emitted = 0
        for path, text in iter_python_sources(
            self.config.code_roots,
            self.config.max_code_files,
            self.config.min_code_file_bytes,
            self.config.max_code_file_bytes,
            self.config.code_seed,
        ):
            self.stats["files"] += 1
            try:
                tree = _parse_python(text)
            except SyntaxError:
                continue
            lines = text.splitlines(keepends=True)
            imports = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    try:
                        imports.append(ast.unparse(node))
                    except Exception:
                        pass
            import_text = "\n".join(imports)
            if import_text:
                import_text += "\n"
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.decorator_list:
                    continue
                doc = ast.get_docstring(node, clean=True)
                if not doc:
                    continue
                problem = doc.split("\n\n", 1)[0].strip()
                if not (20 <= len(problem) <= 500):
                    continue
                body_nodes = list(node.body)
                if body_nodes and isinstance(body_nodes[0], ast.Expr):
                    value = body_nodes[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        body_nodes = body_nodes[1:]
                if not body_nodes:
                    continue
                if len(body_nodes) > 30:
                    continue
                try:
                    args = ast.unparse(node.args)
                    signature = f"def {node.name}({args}):"
                except Exception:
                    continue
                start = body_nodes[0].lineno - 1
                end = node.end_lineno
                if start is None or end is None or end <= start:
                    continue
                body_text = "".join(lines[start:end])
                if not body_text.strip() or len(body_text) > 2000:
                    continue
                if "self." in body_text or "self," in args:
                    continue
                key = (node.name, normalize_for_dedup(problem))
                if key in seen:
                    continue
                seen.add(key)
                self.stats["functions_seen"] += 1
                record = {
                    "task_id": -1,
                    "problem_text": problem,
                    "code": f"{import_text}{signature}\n{body_text}",
                    "test_list": [],
                    "test_setup_code": "",
                }
                item = build_instruction_example(self.tokenizer, self.config, record)
                if item is None:
                    self.stats["dropped"] += 1
                    continue
                item["task_type"] = "stdlib_instruction"
                item["source"] = path
                self.stats["examples"] += 1
                self.stats["target_tokens"] += item["valid_target_tokens"]
                emitted += 1
                yield item
                if emitted >= self.max_functions:
                    return


# ---------------------------------------------------------------------------
# Token-quota mixture
# ---------------------------------------------------------------------------


class QuotaMixtureDataset(IterableDataset):
    """Interleave datasets until each source's target-token quota is met.

    Quotas are token quotas, not example quotas. This prevents long language
    sequences from silently dominating a nominal example-level mixture.
    """

    def __init__(
        self,
        sources: Sequence[Tuple[str, IterableDataset, int, bool]],
        max_tokens: Optional[int] = None,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("at least one source is required")
        self.sources = list(sources)
        self.max_tokens = max_tokens
        self.seed = seed
        self.token_accounting: Dict[str, int] = {name: 0 for name, *_ in sources}
        self.example_accounting: Dict[str, int] = {name: 0 for name, *_ in sources}
        self.quota: Dict[str, int] = {name: int(q) for name, _, q, _ in sources}
        self.allow_repeat: Dict[str, bool] = {name: repeat for name, _, _, repeat in sources}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        iterators = {name: iter(ds) for name, ds, _, _ in self.sources}
        exhausted: set = set()
        total_tokens = 0
        rng = random.Random(self.seed)
        while True:
            if self.max_tokens is not None and total_tokens >= self.max_tokens:
                return
            active = [
                name for name in iterators
                if self.token_accounting[name] < self.quota[name]
                and (name not in exhausted or self.allow_repeat[name])
            ]
            if not active:
                return
            # Weight by remaining fraction so the stream interleaves rather
            # than emitting one source as a contiguous block.
            weights = []
            for name in active:
                remaining = max(0, self.quota[name] - self.token_accounting[name])
                weights.append(remaining / max(1, self.quota[name]))
            name = rng.choices(active, weights=weights, k=1)[0]
            try:
                item = next(iterators[name])
            except StopIteration:
                if self.allow_repeat[name]:
                    iterators[name] = iter(dict((n, d) for n, d, _, _ in self.sources)[name])
                    try:
                        item = next(iterators[name])
                    except StopIteration:
                        exhausted.add(name)
                        continue
                else:
                    exhausted.add(name)
                    continue
            if "valid_target_tokens" not in item:
                item["valid_target_tokens"] = count_valid_targets(item["labels"])
                item["task_type"] = name
                item["task_id"] = item.get("row_index", -1)
            n = int(item["valid_target_tokens"])
            item["mixture_source"] = name
            self.token_accounting[name] += n
            self.example_accounting[name] += 1
            total_tokens += n
            yield item


def build_mbpp_eval_records(config: CodePipelineConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Load MBPP validation/test records for evaluation (never used for training)."""
    return {
        "validation": _load_mbpp_split(config.mbpp_dir, config.mbpp_config, "validation"),
        "test": _load_mbpp_split(config.mbpp_dir, config.mbpp_config, "test"),
    }
