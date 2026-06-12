"""
Ablation: new_model w/o semantic list, with DeepSieve source routing.

This is a thin wrapper around /root/autodl-tmp/new_model/run.py that forces
--routing-mode=deepsieve. Under that mode the router returns no
`matched_concept`, so `step2_decompose.semantic.parse_semantic` and
`step2_decompose.operator_plan.build_plan_for_subquery` both receive
`matched_info=None`, which means the semantic-list content is NOT injected
into the query-parsing prompt or the operator-tree-generation prompt. Only
source routing runs (DeepSieve uses profiles, not the catalog).

Default output dir is co-located with this wrapper. Override with --output-dir.
--profiles-path defaults to new_model/data_sources/source_profiles.json.

Copyable runs:

python /root/autodl-tmp/baseline/new_modl_wo_semlist_deepsieve/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_modl_wo_semlist_deepsieve/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_modl_wo_semlist_deepsieve/run.py \
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

# Force routing-mode=deepsieve; reject user override since this script's
# whole point is the DeepSieve ablation.
if "--routing-mode" in sys.argv:
    i = sys.argv.index("--routing-mode")
    val = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    if val != "deepsieve":
        raise SystemExit(
            f"[new_modl_wo_semlist_deepsieve] --routing-mode is fixed to 'deepsieve', "
            f"got '{val}'. Use new_model/run.py directly if you want a different mode."
        )
else:
    sys.argv.extend(["--routing-mode", "deepsieve"])

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
