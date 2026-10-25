"""Phase 9 evaluator-audit regressions.

These tests pin the three defects found while auditing the executable-coding
pipeline and the ordinary sandbox error classifications. They do not depend on
model checkpoints.
"""

from __future__ import annotations

import os

import pytest

from evaluators.coding_eval import (
    assemble_body_completion,
    check_syntax_validity,
)
from evaluators.execution_sandbox import execute_sandboxed, is_bwrap_available
from evaluators.re_eval import build_signature_prompt


pytestmark = pytest.mark.skipif(
    not is_bwrap_available(), reason="Bubblewrap (bwrap) is required"
)


def _record(code: str, tests: list[str], setup: str = "") -> dict:
    return {
        "task_id": 1,
        "problem_text": "Write a function that satisfies the tests.",
        "code": code,
        "test_list": tests,
        "test_setup_code": setup,
    }


def test_nested_def_first_body_line_keeps_indent_and_executes():
    record = _record(
        "def apply_all(items):\n"
        "    def inc(x):\n"
        "        return x + 1\n"
        "    return [inc(x) for x in items]\n",
        ["assert apply_all([1, 2]) == [2, 3]"],
    )
    prompt, name, body = build_signature_prompt(record)
    assembled = assemble_body_completion(prompt, name, body)
    assert assembled.startswith("def apply_all(items):\n    def inc(x):")
    assert check_syntax_validity(assembled)
    result = execute_sandboxed(assembled, record["test_list"])
    assert result.passed, result.stderr


def test_full_function_generation_is_not_duplicated():
    record = _record("def add_one(x):\n    return x + 1\n", ["assert add_one(1) == 2"])
    prompt, name, _ = build_signature_prompt(record)
    assembled = assemble_body_completion(prompt, name, "def add_one(x):\n    return x + 1\n")
    assert assembled.count("def add_one") == 1
    result = execute_sandboxed(assembled, record["test_list"])
    assert result.passed, result.stderr


def test_setup_code_runs_after_candidate_definitions():
    # MBPP test task 367 pattern: test_setup_code instantiates a class that the
    # candidate solution defines. Setup-before-solution makes this unsolvable.
    code = (
        "class Node:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
        "def read_value(node):\n"
        "    return node.value\n"
    )
    setup = "root = Node(7)\n"
    result = execute_sandboxed(code, ["assert read_value(root) == 7"], setup)
    assert result.passed, result.stderr


def test_error_classification_synthetic_fixtures():
    assert execute_sandboxed("def f(:", ["assert True"]).error_type == "syntax_error"
    assert execute_sandboxed("def f(): return 1", ["assert f() == 2"]).error_type == "assertion_error"
    assert execute_sandboxed("def f(): return 1 / 0", ["assert f()"]).error_type == "runtime_error"
    timeout = execute_sandboxed(
        "def f():\n    while True:\n        pass", ["assert f()"], timeout_sec=1.0
    )
    assert timeout.error_type == "timeout"


def test_canonical_mbpp_regressions_for_defect_tasks():
    parquet = "data/coding/mbpp/full/test-00000-of-00001.parquet"
    if not os.path.exists(parquet):
        pytest.skip("MBPP parquet not available")
    pyarrow = pytest.importorskip("pyarrow.parquet")
    from dataset.code_pipeline import CodePipelineConfig, build_mbpp_eval_records

    records = build_mbpp_eval_records(
        CodePipelineConfig(seq_len=256, mbpp_dir="data/coding/mbpp")
    )["test"]
    by_id = {int(r["task_id"]): r for r in records}
    # 215 and 317 previously lost the indentation of a nested first body line.
    # 367 previously ran test_setup_code before the candidate defined Node.
    for task_id in (215, 317, 367):
        record = by_id[task_id]
        prompt, name, body = build_signature_prompt(record)
        assembled = assemble_body_completion(prompt, name, body)
        assert check_syntax_validity(assembled)
        result = execute_sandboxed(
            assembled, record["test_list"], record["test_setup_code"]
        )
        assert result.passed, (task_id, result.error_type, result.stderr[-500:])
