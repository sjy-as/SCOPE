# 🧪 Evaluation Metric Robustness

> **Reviewer A4 — Judge reliability / benchmark dependence**
> We re-evaluate all methods using **Exact Match (EM)** and **token-level F1**, and retain the LLM Judge only as a supplementary metric.

---

## ✨ Key Findings

| Metric      | **SCOPE** |     Best Baseline |        Gain |
| ----------- | --------: | ----------------: | ----------: |
| **Avg. EM** | **0.517** | 0.480 (DeepSieve) | **+3.7 pp** |
| **Avg. F1** | **0.637** | 0.564 (DeepSieve) | **+7.3 pp** |

* 🏆 **SCOPE remains the best method overall under EM and F1.**
* 🔍 On **100** SCOPE cases with `Judge=1` but `EM=0`, **83%** are semantically equivalent answers missed by strict string matching.
* ✅ In the revised paper, **EM/F1 are the primary metrics**; the **LLM Judge is supplementary only**.

---

## ⚙️ Experimental Setup

### Datasets

| Split       | # Questions | Sources          |
| ----------- | ----------: | ---------------- |
| `kg_doc`    |       1,154 | KG + Document    |
| `kg_table`  |       1,147 | KG + Table       |
| `table_doc` |       1,120 | Table + Document |

### Compared Methods

**Without Retrieval:** Standard Prompt, CoT, Self-ASK
**With Retrieval:** Standard RAG, CoK, IRCoT, HydraRAG, DeepSieve, AtomR
**Ours:** SCOPE

### Metrics

* **EM** — exact string match after basic normalization.
* **F1** — token-level F1 against the best matching gold answer.
* **LLM Judge** — retained only for supplementary analysis.

---

## 📊 Overall EM / F1 Results

| Method           | KG-Doc EM | KG-Doc F1 | KG-Table EM | KG-Table F1 | Table-Doc EM | Table-Doc F1 | **Avg. EM** | **Avg. F1** |
| ---------------- | --------: | --------: | ----------: | ----------: | -----------: | -----------: | ----------: | ----------: |
| **SCOPE (ours)** | **0.577** | **0.710** |       0.471 |   **0.586** |    **0.504** |    **0.615** |   **0.517** |   **0.637** |
| DeepSieve        |     0.498 |     0.594 |   **0.473** |       0.564 |        0.468 |        0.533 |       0.480 |       0.564 |
| AtomR            |     0.414 |     0.584 |       0.321 |       0.452 |        0.400 |        0.517 |       0.378 |       0.518 |
| IRCoT            |     0.334 |     0.460 |       0.401 |       0.523 |        0.407 |        0.486 |       0.381 |       0.490 |
| HydraRAG         |     0.323 |     0.472 |       0.291 |       0.389 |        0.351 |        0.446 |       0.322 |       0.435 |
| CoT              |     0.246 |     0.408 |       0.160 |       0.300 |        0.190 |        0.280 |       0.199 |       0.329 |
| CoK              |     0.215 |     0.392 |       0.148 |       0.283 |        0.178 |        0.266 |       0.180 |       0.314 |
| Self-ASK         |     0.244 |     0.407 |       0.155 |       0.289 |        0.173 |        0.265 |       0.191 |       0.320 |
| Standard RAG     |     0.179 |     0.237 |       0.235 |       0.307 |        0.255 |        0.290 |       0.223 |       0.278 |
| Standard Prompt  |     0.162 |     0.309 |       0.106 |       0.222 |        0.115 |        0.184 |       0.128 |       0.238 |

> **Note.** SCOPE achieves the best F1 on all three CMQA splits and the best EM on two of three splits. On KG-Table EM, DeepSieve is higher by only **0.002**.

---

## 🔎 LLM Judge vs. EM

For SCOPE, the LLM Judge marked **637** predictions as exact while strict EM returned 0.

| Dataset   | Judge = Exact | Judge = Exact ∧ EM = 1 | Judge = Exact ∧ EM = 0 |
| --------- | ------------: | ---------------------: | ---------------------: |
| KG-Doc    |           757 |                    590 |                    167 |
| KG-Table  |           719 |                    479 |                    240 |
| Table-Doc |           718 |                    488 |                    230 |
| **Total** |     **2,194** |              **1,557** |                **637** |

### 👀 Manual Audit

We manually inspected **100** randomly sampled `Judge=exact ∧ EM=0` SCOPE cases.

| Verdict                       |   Count |    Share |
| ----------------------------- | ------: | -------: |
| ✅ Semantically equivalent     |  **83** |  **83%** |
| ❌ Not semantically equivalent |      17 |      17% |
| **Total**                     | **100** | **100%** |

Typical EM misses include:

* `27` → `27 points`
* `44.1` → `44.1 seconds`
* `Guard` → `point guard`
* `22` → `22nd pick`

These cases show why strict string matching can undercount correct free-form answers, especially in document QA.

---

<details>
<summary><b>📁 Experiment Files</b></summary>

```text
R2_LLM_Judge_Reliability_Experiment/
├── runs_results/
│   ├── kg_doc/deepseek-chat/SCOPE/
│   ├── kg_table/deepseek-chat/new_model/      # SCOPE alias
│   ├── table_doc/deepseek-chat/new_model/     # SCOPE alias
│   └── <dataset>/deepseek-chat/<method>/
│
└── analysis_outputs/
    ├── conservative_em_f1_summary.json
    ├── judge_em_agreement_summary.json
    ├── per_example/<dataset>/<method>.conservative.json
    ├── judge_em_scope_manual_audit_100.json
    └── judge_em_scope_manual_audit_100_summary.json
```

</details>

<details>
<summary><b>▶️ Reproduce</b></summary>

```bash
python3 scripts/conservative_em_f1.py \
    --runs-root runs_results \
    --out analysis_outputs
```

</details>

---

# 📈 Dataset Performance

## CMQA

> EM / F1 results in the same comparison style as the main paper table.

| Category              | Model           | KG-Doc EM | KG-Doc F1 | KG-Table EM | KG-Table F1 | Table-Doc EM | Table-Doc F1 |
| --------------------- | --------------- | --------: | --------: | ----------: | ----------: | -----------: | -----------: |
| **Without Retrieval** | Standard Prompt |     0.162 |     0.309 |       0.106 |       0.222 |        0.115 |        0.184 |
|                       | CoT             |     0.246 |     0.408 |       0.160 |       0.300 |        0.190 |        0.280 |
|                       | Self-ASK        |     0.244 |     0.407 |       0.155 |       0.289 |        0.173 |        0.265 |
| **With Retrieval**    | Standard RAG    |     0.179 |     0.237 |       0.235 |       0.307 |        0.255 |        0.290 |
|                       | CoK             |     0.215 |     0.392 |       0.148 |       0.283 |        0.178 |        0.266 |
|                       | IRCoT           |     0.334 |     0.460 |       0.401 |       0.523 |        0.407 |        0.486 |
|                       | HydraRAG        |     0.323 |     0.472 |       0.291 |       0.389 |        0.351 |        0.446 |
|                       | DeepSieve       |     0.498 |     0.594 |   **0.473** |       0.564 |        0.468 |        0.533 |
|                       | AtomR           |     0.414 |     0.584 |       0.321 |       0.452 |        0.400 |        0.517 |
| **Ours**              | **SCOPE**       | **0.577** | **0.710** |       0.471 |   **0.586** |    **0.504** |    **0.615** |

---

> 📌 **More results coming soon.** We will progressively upload the new **EM/F1 evaluation results on CMDBench, CompMix, and HybridQA**.

---

## ✅ Takeaway

**SCOPE remains the strongest method after replacing the LLM Judge with EM/F1.** The ranking is preserved under strict string-based evaluation, while the manual audit shows that many Judge–EM disagreements come from semantically correct answers expressed in different surface forms.
