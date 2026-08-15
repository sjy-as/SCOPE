"""
Ablation: SCOPE w/o Consolidation.

Disables the runtime consolidation memory while keeping decomposition, routing, semantic parsing, operator planning, execution, and fallback-source reflection/retry unchanged.

Copyable runs:

python SCOPE_code/wo_abla/SCOPE_wo_consolidation/run.py \
  --input ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --gold  ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_consolidation/run.py \
  --input ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --gold  ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_consolidation/run.py \
  --input ../../CMQA/qa_bench/table-doc-1120.jsonl \
  --gold  ../../CMQA/qa_bench/table-doc-1120.jsonl \
  --kb table,doc \
  --workers 8 \
  --api-key "$LLM_API_KEY"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEW_MODEL_DIR = Path(os.getenv("SCOPE_NEW_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "SCOPE"))).resolve()
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "result"

if str(NEW_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_MODEL_DIR))

if "--no-consolidation" not in sys.argv:
    sys.argv.append("--no-consolidation")

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
