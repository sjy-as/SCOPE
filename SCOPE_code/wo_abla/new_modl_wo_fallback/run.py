"""
Ablation: new_model w/o Fallback.

Disables the fallback-source retry in stage3_execute: when the primary source
returns empty answers, the pipeline does NOT re-try with the fallback source.
Decomposition, Routing (semantic list), Semantic Parsing, and Operator Planning
all run as normal.

Copyable runs:

python /root/autodl-tmp/baseline/new_modl_wo_fallback/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_modl_wo_fallback/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_modl_wo_fallback/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/table-doc-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/table-doc-160.jsonl \
  --kb table,doc \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEW_MODEL_DIR = Path("/root/autodl-tmp/new_model")
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "result"

if str(NEW_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_MODEL_DIR))

if "--no-fallback" not in sys.argv:
    sys.argv.append("--no-fallback")

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
