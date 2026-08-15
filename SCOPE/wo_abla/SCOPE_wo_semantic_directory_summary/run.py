"""
Ablation: SCOPE w/o semantic list, with DeepSieve source routing.

This is a thin wrapper around SCOPE/run.py that forces
--routing-mode=deepsieve. Under that mode the router returns no
`matched_concept`, so `step2_decompose.semantic.parse_semantic` and
`step2_decompose.operator_plan.build_plan_for_subquery` both receive
`matched_info=None`, which means the semantic-list content is NOT injected
into the query-parsing prompt or the operator-tree-generation prompt. Only
source routing runs (DeepSieve uses profiles, not the catalog).

Default output dir is co-located with this wrapper. Override with --output-dir.
--profiles-path defaults to SCOPE/data_sources/source_profiles.json.

Copyable runs:

python SCOPE_code/wo_abla/SCOPE_wo_semlist_deepsieve/run.py \
  --input ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --gold  ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_semlist_deepsieve/run.py \
  --input ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --gold  ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_semlist_deepsieve/run.py \
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

# Force routing-mode=deepsieve; reject user override since this script's
# whole point is the DeepSieve ablation.
if "--routing-mode" in sys.argv:
    i = sys.argv.index("--routing-mode")
    val = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    if val != "deepsieve":
        raise SystemExit(
            f"[SCOPE_wo_semlist_deepsieve] --routing-mode is fixed to 'deepsieve', "
            f"got '{val}'. Use SCOPE/run.py directly if you want a different mode."
        )
else:
    sys.argv.extend(["--routing-mode", "deepsieve"])

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
