"""Build a code-aware BPE tokenizer for RE coding experiments.

Example:
  PYTHONPATH=. python -m dataset.build_code_tokenizer \
      --workdir experiments/re_coding_adaptive/tokenizer \
      --tinystories-workdir experiments/trm_lm_pilot_5m/r1 \
      --vocab-size 8192
"""

from __future__ import annotations

import argparse
import json
import os

from dataset.code_pipeline import CodePipelineConfig, train_code_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", default="experiments/re_coding_adaptive/tokenizer"
    )
    parser.add_argument(
        "--tinystories-workdir", default="experiments/trm_lm_pilot_5m/r1"
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--story-docs", type=int, default=4000)
    parser.add_argument("--code-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-code-files", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = os.path.join(args.tinystories_workdir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    parquet_path = manifest["parquet"]["path"]
    config = CodePipelineConfig(
        tokenizer_vocab_size=args.vocab_size,
        tokenizer_story_docs=args.story_docs,
        tokenizer_code_bytes=args.code_bytes,
        max_code_files=args.max_code_files,
    )
    info = train_code_tokenizer(
        workdir=args.workdir,
        config=config,
        tinystories_parquet=parquet_path,
        tinystories_manifest=manifest,
        force=args.force,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
