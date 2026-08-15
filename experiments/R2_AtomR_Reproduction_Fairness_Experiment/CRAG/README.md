# 🧪 CRAG Generalization & AtomR Reproduction

This experiment evaluates **SCOPE on CRAG**, a benchmark used in AtomR's original evaluation, and validates the **fairness of our AtomR reproduction**.

> **Reviewer W3 — Fairness of AtomR Reproduction**
> We reproduce AtomR with its original reasoning procedure and few-shot prompting, introducing only necessary retrieval-interface adaptations.

---

## ✨ Key Findings

| Method                       |      F1 ↑ |      EM ↑ | LLM Judge Acc. ↑ | Hallucination ↓ |
| ---------------------------- | --------: | --------: | ---------------: | --------------: |
| **AtomR (Reported)**         |     0.690 |     0.614 |        **0.752** |       **0.242** |
| **AtomR (Our Reproduction)** | **0.707** | **0.634** |            0.726 |           0.274 |
| **SCOPE (Ours)**             |     0.672 |     0.599 |            0.690 |           0.254 |

### 🔍 Main Observations

* ✅ **AtomR is faithfully reproduced:** our reproduction achieves **0.707 F1**, close to the **0.690 F1** reported result.
* 🌐 **SCOPE transfers well to CRAG:** SCOPE achieves **0.672 F1**, comparable to AtomR on its original benchmark.
* 📊 The reproduction slightly exceeds the reported AtomR result in both **F1 (+1.7 pp)** and **EM (+2.1 pp)**.
* 🧩 These results suggest that AtomR's weaker performance in our heterogeneous QA setting is unlikely to result from reproduction quality.

---

## 📚 Experimental Setup

### Dataset

| Property              | Setting              |
| --------------------- | -------------------- |
| **Benchmark**         | CRAG                 |
| **Knowledge Sources** | KG + Web             |
| **Evaluated Domains** | Finance, Movie, Open |
| **Total Questions**   | **339**              |

### Domain Distribution

| Domain    | # Questions |
| --------- | ----------: |
| Finance   |          36 |
| Movie     |         183 |
| Open      |         120 |
| **Total** |     **339** |

We evaluate the three CRAG domains available in our experimental setup under the same evaluation protocol.

---

## 📊 Detailed Results

### AtomR — Reported Result

| Metric             |     Result |
| ------------------ | ---------: |
| F1                 | **0.6903** |
| Exact Match        | **0.6136** |
| LLM Judge Accuracy | **0.7522** |
| LLM Judge Score    | **0.5103** |
| Hallucination Rate | **24.19%** |

---

### AtomR — Our Reproduction

| Metric             |     Result |
| ------------------ | ---------: |
| F1                 | **0.7073** |
| Exact Match        | **0.6342** |
| LLM Judge Accuracy | **0.7257** |
| LLM Judge Score    | **0.4513** |
| Hallucination Rate | **27.43%** |

Our reproduction preserves AtomR's original procedure and few-shot prompting, with only necessary adaptations to the retrieval interfaces.

The resulting **0.707 F1** is close to — and slightly higher than — the reported **0.690 F1**, supporting the validity of our reproduction.

---

### SCOPE — Ours

| Metric             |     Result |
| ------------------ | ---------: |
| F1                 | **0.6721** |
| Exact Match        | **0.5988** |
| LLM Judge Accuracy | **0.6903** |
| LLM Judge Score    | **0.4366** |
| Hallucination Rate | **25.37%** |

SCOPE achieves **0.672 F1** on CRAG, showing that its reasoning framework can transfer beyond CMQA to a heterogeneous **KG + Web** environment.

---

## 🔬 Reproduction Fairness

The CRAG experiment primarily serves as a **sanity check for the AtomR reproduction**.

Our reproduced AtomR obtains:

> **0.707 F1 vs. 0.690 reported F1**

Since the reproduced result closely matches the reported performance, AtomR's weaker performance on our multi-source heterogeneous QA benchmark is unlikely to be caused by an unfavorable reproduction.

A likely factor is **limited transferability across heterogeneous settings**: AtomR mainly relies on few-shot prompting for routing and planning, whereas SCOPE introduces systematic semantic guidance through the **Semantic Directory**.

---

## 📁 Result Files

```text
CRAG_new/results/Atomr/
├── crag_test.jsonl
└── evaluation_deepseek.json
```

**AtomR reported results**

```text
result/atomr_crag_scope_aligned/
├── crag_test.jsonl
├── evaluation_summary.json
├── finance/predictions.jsonl
├── movie/predictions.jsonl
└── open/predictions.jsonl
```

**AtomR reproduction results**

```text
result/scope_crag_by_domain/
├── open/
│   ├── predictions.jsonl
│   └── summary.json
└── finance/
    ├── predictions.jsonl
    └── summary.json
```

**SCOPE results**

---

## 📝 Takeaway

> **Our AtomR reproduction reaches 0.707 F1, closely matching the reported 0.690 F1, while SCOPE achieves a comparable 0.672 F1 on CRAG.**

This experiment validates the **fairness of the AtomR reproduction** and provides additional evidence that SCOPE generalizes beyond CMQA.
