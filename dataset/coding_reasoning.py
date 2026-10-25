"""Coding and deliberate reasoning dataset pipeline for RE:Inference.

Integrates Mostly Basic Python Problems (MBPP) alongside TinyStories language
modeling. Implements:
- Leakage-proof deduplication and splitting.
- Task A: Ordinary language continuation (TinyStories).
- Task B: Python code continuation (MBPP prefix -> continuation).
- Task C: Python function synthesis (MBPP problem -> implementation).
- Task D: Program behavior / output prediction (MBPP function + query -> output).
- Explicit prompt-target loss masking (prompt tokens get IGNORE_INDEX).
- Configurable mixed-task sampling with exact token accounting.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from dataset.tinystories_lm import (
    IGNORE_INDEX,
    TinyStoriesConfig,
    collate_lm_batch,
    count_valid_targets,
    drop_leading_valid_targets,
    iter_selected_documents,
    load_tokenizer,
    truncate_labels_to_budget,
    validate_next_token_alignment,
)

DELIMITER_PROBLEM = "### Problem:\n"
DELIMITER_SOLUTION = "\n### Solution:\n"
DELIMITER_CODE = "### Code:\n"
DELIMITER_QUERY = "\n### Query:\n"
DELIMITER_OUTPUT = "\n### Output:\n"


@dataclass
class CodingReasoningConfig:
    """Configuration for coding and reasoning data mixture."""

    tinystories_workdir: str = "experiments/trm_lm_pilot_5m/r1"
    mbpp_dir: str = "data/coding/mbpp"
    seq_len: int = 256
    seed: int = 1337
    # Proposed starting mixture: 60% language continuation, 40% coding tasks
    lang_weight: float = 0.60
    code_continuation_weight: float = 0.10
    func_synthesis_weight: float = 0.20
    output_prediction_weight: float = 0.10
    mbpp_config: str = "full"  # "full" or "sanitized"
    min_code_lines: int = 2


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text to detect near-duplicate problem statements."""
    lowered = text.lower().strip()
    # Normalize whitespace and punctuation
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_assertion(assert_stmt: str) -> Optional[Tuple[str, str]]:
    """Parse an assert statement into (call_expression, expected_output).

    Example:
        assert min_cost([[1, 2], [3, 4]], 1, 1) == 5
        -> ('min_cost([[1, 2], [3, 4]], 1, 1)', '5')
    """
    assert_stmt = assert_stmt.strip()
    if not assert_stmt.startswith("assert"):
        return None
    try:
        parsed = ast.parse(assert_stmt)
        if not parsed.body or not isinstance(parsed.body[0], ast.Assert):
            return None
        test_expr = parsed.body[0].test
        if not isinstance(test_expr, ast.Compare):
            return None
        if len(test_expr.ops) == 1 and isinstance(test_expr.ops[0], ast.Eq):
            left_src = ast.unparse(test_expr.left).strip()
            right_src = ast.unparse(test_expr.comparators[0]).strip()
            return left_src, right_src
    except Exception:
        # Fallback to simple regex if ast unparse fails
        m = re.match(r"^assert\s+(.+?)\s*==\s*(.+)$", assert_stmt)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


class MBPPDatasetInspector:
    """Inspects, deduplicates, and splits MBPP examples."""

    def __init__(self, mbpp_dir: str = "data/coding/mbpp", config_name: str = "full") -> None:
        self.mbpp_dir = mbpp_dir
        self.config_name = config_name
        self.split_files = {
            "train": os.path.join(mbpp_dir, config_name, "train-00000-of-00001.parquet"),
            "validation": os.path.join(mbpp_dir, config_name, "validation-00000-of-00001.parquet"),
            "test": os.path.join(mbpp_dir, config_name, "test-00000-of-00001.parquet"),
            "prompt": os.path.join(mbpp_dir, config_name, "prompt-00000-of-00001.parquet"),
        }
        self.text_col = "text" if config_name == "full" else "prompt"

    def load_raw_split(self, split: str) -> List[Dict[str, Any]]:
        path = self.split_files[split]
        if not os.path.exists(path):
            raise FileNotFoundError(f"MBPP split file not found: {path}")
        table = pq.read_table(path)
        rows: List[Dict[str, Any]] = []
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            # Normalize text field
            row["problem_text"] = str(row[self.text_col]).strip()
            row["code"] = str(row["code"]).strip()
            row["task_id"] = int(row["task_id"])
            rows.append(row)
        return rows

    def audit_and_deduplicate(self) -> Dict[str, Any]:
        """Perform comprehensive deduplication and cross-split leakage check.

        Returns split manifests where any training example matching a test or
        validation example (exact or normalized text, or exact code) is flagged
        and quarantined from the training partition.
        """
        raw_splits = {s: self.load_raw_split(s) for s in ["train", "validation", "test", "prompt"]}
        
        # Build evaluation lookup tables (validation + test)
        eval_exact_texts: Dict[str, List[Tuple[str, int]]] = {}
        eval_norm_texts: Dict[str, List[Tuple[str, int]]] = {}
        eval_exact_codes: Dict[str, List[Tuple[str, int]]] = {}

        for s in ["validation", "test"]:
            for r in raw_splits[s]:
                tid = r["task_id"]
                txt = r["problem_text"]
                norm_txt = normalize_text_for_dedup(txt)
                code = r["code"]
                eval_exact_texts.setdefault(txt, []).append((s, tid))
                eval_norm_texts.setdefault(norm_txt, []).append((s, tid))
                eval_exact_codes.setdefault(code, []).append((s, tid))

        # Check train examples for leakage
        retained_train: List[Dict[str, Any]] = []
        quarantined_train: List[Dict[str, Any]] = []

        for r in raw_splits["train"]:
            tid = r["task_id"]
            txt = r["problem_text"]
            norm_txt = normalize_text_for_dedup(txt)
            code = r["code"]

            leak_reasons: List[str] = []
            if txt in eval_exact_texts:
                leak_reasons.append(f"exact_text_match:{eval_exact_texts[txt]}")
            elif norm_txt in eval_norm_texts:
                leak_reasons.append(f"near_text_match:{eval_norm_texts[norm_txt]}")
            if code in eval_exact_codes:
                leak_reasons.append(f"exact_code_match:{eval_exact_codes[code]}")

            if leak_reasons:
                r_copy = dict(r)
                r_copy["quarantine_reasons"] = leak_reasons
                quarantined_train.append(r_copy)
            else:
                retained_train.append(r)

        return {
            "config_name": self.config_name,
            "counts_raw": {s: len(rows) for s, rows in raw_splits.items()},
            "train_retained_count": len(retained_train),
            "train_quarantined_count": len(quarantined_train),
            "quarantined_examples": quarantined_train,
            "splits": {
                "train": retained_train,
                "validation": raw_splits["validation"],
                "test": raw_splits["test"],
                "prompt": raw_splits["prompt"],
            },
        }


def format_task_b_code_continuation(
    code: str, prefix_ratio: float = 0.5
) -> Optional[Tuple[str, str]]:
    """Format Task B: Python code continuation.

    Returns (prefix, continuation) where prefix is non-empty and continuation
    completes the code block.
    """
    lines = code.splitlines(keepends=True)
    if len(lines) < 2:
        return None
    split_idx = max(1, int(len(lines) * prefix_ratio))
    prefix = "".join(lines[:split_idx])
    continuation = "".join(lines[split_idx:])
    if not continuation.strip():
        return None
    return prefix, continuation


def format_task_c_function_synthesis(
    problem_text: str, code: str
) -> Tuple[str, str]:
    """Format Task C: Python function synthesis.

    Returns (prompt, target_code) with explicit structural delimiters.
    """
    prompt = f"{DELIMITER_PROBLEM}{problem_text.strip()}{DELIMITER_SOLUTION}"
    return prompt, code.strip()


def format_task_d_output_prediction(
    code: str, test_list: Sequence[str]
) -> List[Tuple[str, str]]:
    """Format Task D: Program behavior / output prediction from test assertions."""
    examples: List[Tuple[str, str]] = []
    for test_stmt in test_list:
        parsed = parse_assertion(test_stmt)
        if parsed is not None:
            call_expr, expected_out = parsed
            prompt = f"{DELIMITER_CODE}{code.strip()}{DELIMITER_QUERY}{call_expr}{DELIMITER_OUTPUT}"
            target = expected_out
            examples.append((prompt, target))
    return examples


def pack_prompt_target(
    prompt: str,
    target: str,
    tokenizer: Any,
    seq_len: int,
    pad_id: int,
    ignore_index: int = IGNORE_INDEX,
    task_type: str = "coding",
    task_id: int = -1,
) -> Optional[Dict[str, Any]]:
    """Tokenize and pack a prompt-target pair with prompt loss masking.

    Prompt tokens are assigned IGNORE_INDEX in labels so that gradients and
    supervision only occur on the target completion.
    Next-token alignment is enforced: labels[t] == input_ids[t+1] for valid targets.
    """
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer missing <bos> or <eos>")

    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(target).ids

    # Full sequence: <bos> + prompt + target + <eos>
    tokens = [bos_id] + list(prompt_ids) + list(target_ids) + [eos_id]
    if len(tokens) > seq_len:
        # If longer than seq_len, truncate from the end of target, but ensure
        # prompt and at least 2 target tokens remain.
        min_needed = 1 + len(prompt_ids) + 2
        if seq_len < min_needed:
            return None
        tokens = tokens[:seq_len]
        # Ensure final token is <eos> if truncated
        tokens[-1] = eos_id

    real_len = len(tokens)
    input_ids = np.full(seq_len, pad_id, dtype=np.int64)
    labels = np.full(seq_len, ignore_index, dtype=np.int64)
    attention_mask = np.zeros(seq_len, dtype=np.int64)

    input_ids[:real_len] = tokens
    attention_mask[:real_len] = 1

    # Prompt boundary:
    # prompt covers index 0 (<bos>) through len(prompt_ids).
    # Target begins at index 1 + len(prompt_ids).
    target_start_idx = 1 + len(prompt_ids)
    for t in range(target_start_idx - 1, real_len - 1):
        labels[t] = tokens[t + 1]

    input_tensor = torch.from_numpy(input_ids)
    label_tensor = torch.from_numpy(labels)
    mask_tensor = torch.from_numpy(attention_mask)

    validate_next_token_alignment(
        input_tensor.unsqueeze(0), label_tensor.unsqueeze(0)
    )

    return {
        "input_ids": input_tensor,
        "labels": label_tensor,
        "attention_mask": mask_tensor,
        "task_type": task_type,
        "task_id": task_id,
        "valid_target_tokens": count_valid_targets(label_tensor),
    }


class CodingReasoningDataset(IterableDataset):
    """Dataset streaming coding examples (Tasks B, C, D) from MBPP."""

    def __init__(
        self,
        config: CodingReasoningConfig,
        tokenizer: Any,
        split: str = "train",
        max_examples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.max_examples = max_examples
        self.inspector = MBPPDatasetInspector(
            mbpp_dir=config.mbpp_dir, config_name=config.mbpp_config
        )
        self.audit = self.inspector.audit_and_deduplicate()
        self.raw_records = self.audit["splits"][split]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        pad_id = self.tokenizer.token_to_id("<pad>")
        yielded = 0

        # Construct examples for each record
        for row in self.raw_records:
            tid = row["task_id"]
            problem_text = row["problem_text"]
            code = row["code"]
            test_list = row.get("test_list", [])

            # Task C: Function Synthesis
            prompt_c, target_c = format_task_c_function_synthesis(problem_text, code)
            ex_c = pack_prompt_target(
                prompt_c,
                target_c,
                self.tokenizer,
                self.config.seq_len,
                pad_id,
                task_type="func_synthesis",
                task_id=tid,
            )
            if ex_c is not None:
                yield ex_c
                yielded += 1
                if self.max_examples and yielded >= self.max_examples:
                    return

            # Task B: Code Continuation
            continuation_pair = format_task_b_code_continuation(code)
            if continuation_pair is not None:
                prefix_b, cont_b = continuation_pair
                ex_b = pack_prompt_target(
                    prefix_b,
                    cont_b,
                    self.tokenizer,
                    self.config.seq_len,
                    pad_id,
                    task_type="code_continuation",
                    task_id=tid,
                )
                if ex_b is not None:
                    yield ex_b
                    yielded += 1
                    if self.max_examples and yielded >= self.max_examples:
                        return

            # Task D: Output Prediction
            output_pairs = format_task_d_output_prediction(code, test_list)
            for prompt_d, target_d in output_pairs:
                ex_d = pack_prompt_target(
                    prompt_d,
                    target_d,
                    self.tokenizer,
                    self.config.seq_len,
                    pad_id,
                    task_type="output_prediction",
                    task_id=tid,
                )
                if ex_d is not None:
                    yield ex_d
                    yielded += 1
                    if self.max_examples and yielded >= self.max_examples:
                        return


class MixedCodingReasoningDataset(IterableDataset):
    """Reproducible mixture of TinyStories (Task A) and MBPP coding tasks (B, C, D).

    Interleaves examples according to configurable sampling weights and records
    actual token consumption per task.
    """

    def __init__(
        self,
        config: CodingReasoningConfig,
        tokenizer: Any,
        tinystories_dataset: IterableDataset,
        coding_dataset: CodingReasoningDataset,
        max_train_tokens: Optional[int] = None,
        skip_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.tinystories_dataset = tinystories_dataset
        self.coding_dataset = coding_dataset
        self.max_train_tokens = max_train_tokens
        self.skip_tokens = skip_tokens
        self.token_accounting: Dict[str, int] = {
            "lang_continuation": 0,
            "code_continuation": 0,
            "func_synthesis": 0,
            "output_prediction": 0,
        }
        self.example_accounting: Dict[str, int] = {
            "lang_continuation": 0,
            "code_continuation": 0,
            "func_synthesis": 0,
            "output_prediction": 0,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.RandomState(self.config.seed)
        ts_iter = iter(self.tinystories_dataset)

        # Pre-materialize coding examples for the split to allow looping
        coding_examples = list(self.coding_dataset)
        if not coding_examples:
            raise RuntimeError("No valid coding examples produced from MBPP.")

        coding_idx = 0
        tokens_yielded = 0
        remaining_skip = int(self.skip_tokens)

        # Compute sampling probability for TinyStories
        p_lang = self.config.lang_weight / (
            self.config.lang_weight
            + self.config.code_continuation_weight
            + self.config.func_synthesis_weight
            + self.config.output_prediction_weight
        )

        while True:
            if self.max_train_tokens and tokens_yielded >= self.max_train_tokens:
                break

            use_lang = rng.rand() < p_lang
            if use_lang:
                try:
                    raw_ts = next(ts_iter)
                    item = {
                        "input_ids": raw_ts["input_ids"],
                        "labels": raw_ts["labels"],
                        "attention_mask": raw_ts["attention_mask"],
                        "task_type": "lang_continuation",
                        "task_id": raw_ts.get("row_index", -1),
                        "valid_target_tokens": count_valid_targets(raw_ts["labels"]),
                    }
                except StopIteration:
                    break
            else:
                item = coding_examples[coding_idx % len(coding_examples)]
                coding_idx += 1

            valid_tokens = item["valid_target_tokens"]
            if remaining_skip > 0:
                if remaining_skip >= valid_tokens:
                    remaining_skip -= valid_tokens
                    continue
                item["labels"], _ = drop_leading_valid_targets(
                    item["labels"], remaining_skip
                )
                valid_tokens = count_valid_targets(item["labels"])
                remaining_skip = 0

            self.token_accounting[item["task_type"]] += valid_tokens
            self.example_accounting[item["task_type"]] += 1
            tokens_yielded += valid_tokens

            yield item
