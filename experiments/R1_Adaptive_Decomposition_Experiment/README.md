# 🔀 Adaptive Decomposition on Full CMQA

## 📌 Overview

This experiment examines whether **SCOPE depends on a fixed two-subquestion decomposition**.

The original decomposition prompt always produces exactly two sub-questions (`q1 + q2`). We replace it with an **adaptive decomposition prompt**, where the LLM decides:

- whether decomposition is needed;
- how many sub-questions are necessary;
- the minimum dependency chain required by the query.

The experiment is conducted on the **full CMQA benchmark (3,421 questions)**.

---

## 🧩 Adaptive Decomposition

| Setting | Fixed Decomposition | Adaptive Decomposition |
|---|---|---|
| Need to decompose? | Always | Decided by the LLM |
| Number of sub-questions | Exactly 2 | 1 / 2 / 3 / ... as needed |
| Directly executable query | Still split | Returned as one sub-query |
| Dependency structure | Fixed `q1 → q2` | Minimum sufficient dependency chain |

### Core Prompt Logic

```text
Decide adaptively whether decomposition is needed and how many
sub-questions are necessary.

- If the query can be executed directly, return exactly one
  sub-question containing the original query.
- If decomposition is needed, return the minimum sufficient
  number of sub-questions (q1, q2, ...).
- Order sub-questions so every dependency is produced before it is consumed.
- A dependent sub-question uses [k] to refer to the answer of qk.
```

The decomposition stage only determines the **reasoning structure**; it does not perform source routing or source selection.

---

## 📊 Decomposition Statistics

| Dataset | Total | 1 Subquery | 2 Subqueries | 3 Subqueries | 4 Subqueries |
|---|---:|---:|---:|---:|---:|
| **KG–Doc** | 1,154 | 137 (11.9%) | **1,000 (86.7%)** | 17 (1.5%) | 0 |
| **KG–Table** | 1,147 | 255 (22.2%) | **878 (76.5%)** | 13 (1.1%) | 1 (0.1%) |
| **Table–Doc** | 1,120 | 54 (4.8%) | **1,044 (93.2%)** | 22 (2.0%) | 0 |
| **Overall** | **3,421** | **446 (13.0%)** | **2,922 (85.4%)** | **52 (1.5%)** | **1 (<0.1%)** |

> 🔎 Without enforcing a fixed two-subquestion structure, **85.41% (2,922 / 3,421)** of CMQA questions still naturally produce exactly two sub-questions.

This suggests that the dominant two-hop structure mainly comes from the intrinsic dependency structure of CMQA questions rather than from the decomposition prompt.

---

## 🔹 Direct Single-Query Cases

`should_decompose = false` corresponds to questions that the LLM considers directly executable and therefore keeps as a single sub-query.

| Dataset | Single-Query Cases | Correct | Lenient Acc. |
|---|---:|---:|---:|
| **KG–Doc** | 137 | 48 | **35.04%** |
| **KG–Table** | 255 | 167 | **65.49%** |
| **Table–Doc** | 54 | 26 | **48.15%** |
| **Overall** | **446 (13.0%)** | **241** | **54.04%** |

---

## 🎯 Final Accuracy

For the reported result, we use the following conservative rule:

> **For decomposed queries, fallback answers are counted as incorrect. For non-decomposed single-query cases (`should_decompose = false`), correct fallback answers are retained.**

| Dataset | Total | Correct | Accuracy |
|---|---:|---:|---:|
| **KG–Doc** | 1,154 | 830 | **71.92%** |
| **KG–Table** | 1,147 | 768 | **66.96%** |
| **Table–Doc** | 1,120 | 757 | **67.59%** |
| **Overall** | **3,421** | **2,355** | **68.84%** |

---

## 🏆 Comparison with Baselines

Only the **Acc.** columns from the original CMQA table are used below; the `Used` metric is not included.

| Model | KG–Doc Acc. | KG–Table Acc. | Table–Doc Acc. | Avg. Acc. |
|---|---:|---:|---:|---:|
| Standard Prompt | 0.281 | 0.243 | 0.246 | 0.257 |
| CoT | 0.428 | 0.344 | 0.358 | 0.377 |
| Self-ASK | 0.426 | 0.327 | 0.316 | 0.356 |
| Standard RAG | 0.261 | 0.374 | 0.303 | 0.313 |
| CoK | 0.489 | 0.603 | 0.540 | 0.544 |
| IRCoT | 0.419 | 0.330 | 0.345 | 0.365 |
| ToG2 | 0.487 | 0.419 | 0.374 | 0.427 |
| HydraRAG | 0.482 | 0.460 | 0.515 | 0.486 |
| **DeepSieve** | 0.630 | **0.604** | 0.577 | **0.604** |
| **AtomR** | **0.675** | 0.505 | **0.611** | 0.597 |
| **SCOPE (Adaptive)** | **0.719** | **0.670** | **0.676** | **0.688** |

### Improvement over the strongest baseline

| Comparison | Gain |
|---|---:|
| KG–Doc: 0.719 vs. 0.675 (AtomR) | **+4.42 pp** |
| KG–Table: 0.670 vs. 0.604 (DeepSieve) | **+6.56 pp** |
| Table–Doc: 0.676 vs. 0.611 (AtomR) | **+6.49 pp** |
| Avg.: 0.688 vs. 0.604 (DeepSieve) | **+8.46 pp** |

> ✅ Under adaptive decomposition, SCOPE still **clearly outperforms all compared baselines** on each CMQA source pair and by **8.46 percentage points on average** over the strongest baseline method.

---

## ✅ Conclusion

The full-benchmark experiment shows that removing the fixed two-subquestion constraint does **not** fundamentally change the decomposition pattern:

- **85.41%** of all questions still naturally form exactly two sub-questions;
- **13.04%** are judged directly executable as a single query;
- SCOPE achieves **68.84%** final accuracy under the conservative evaluation rule;
- it remains stronger than all compared baselines, with an **8.46 pp average gain** over the strongest baseline.

These results show that **SCOPE does not rely on a hard-coded two-step decomposition prompt**. The two-step reasoning structure largely emerges from the questions themselves.
