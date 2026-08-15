# 🌐 All-Source Availability Experiment

## 🎯 Overview

This experiment evaluates whether SCOPE depends on the **pre-specified source-pair setting** of CMQA.

During inference, **KG, Table, and Document are simultaneously available**, and no source-pair information is provided. The system must identify the relevant sources through its normal routing process.

We evaluate the **full CMQA benchmark (3,421 questions)**.

### Key Results

| Evaluation                          |     Result |
| ----------------------------------- | ---------: |
| Full CMQA questions                 |  **3,421** |
| All-source final-answer accuracy    | **71.53%** |
| Original source-pair accuracy       | **73.50%** |
| Semantic Directory routing accuracy | **74.69%** |
| Few-shot Prompting routing accuracy |     63.88% |
| Source Summaries routing accuracy   |     60.13% |

The all-source setting introduces only a **1.97-point** decrease in end-to-end accuracy, while the **Semantic Directory** remains substantially more accurate than the other routing strategies.

---

## ⚙️ Experimental Setting

### Source Availability

| Setting                | Available Sources         |
| ---------------------- | ------------------------- |
| Original CMQA          | Relevant source pair      |
| **All-source setting** | **KG + Table + Document** |

The source-pair labels below are used **only for analysis** and are never provided during inference.

### Configuration

| Item            | Setting             |
| --------------- | ------------------- |
| Dataset         | Full CMQA           |
| Questions       | **3,421**           |
| Knowledge bases | KG, Table, Document |
| Routing         | Semantic Directory  |
| LLM             | `deepseek-chat`     |
| Failed runs     | **0**               |

---

# 📊 1. End-to-End Accuracy

We rerun SCOPE on the **entire CMQA benchmark** with all three sources simultaneously available.

For comparison, the **Original** column uses the accuracy reported in the main paper.

| CMQA Subset | Questions | All-Source Exact | **All-Source Acc.** | **Original Acc.** |         Δ |
| ----------- | --------: | ---------------: | ------------------: | ----------------: | --------: |
| KG–Doc      |     1,154 |           63.17% |          **76.08%** |            77.10% |     −1.02 |
| KG–Table    |     1,147 |           59.37% |          **73.67%** |            73.70% |     −0.03 |
| Table–Doc   |     1,120 |           57.50% |          **64.64%** |            69.60% |     −4.96 |
| **Overall** | **3,421** |       **60.04%** |          **71.53%** |        **73.50%** | **−1.97** |

> **Result:** SCOPE retains comparable end-to-end performance when KG, Table, and Document are all available, without knowing the relevant source pair in advance.

---

# 🧭 2. Routing Accuracy

We further evaluate source selection on the **full 3,421-question CMQA benchmark**.

Three routing strategies are compared:

* **Semantic Directory** — fine-grained semantic guidance used by SCOPE.
* **Few-shot Prompting** — source selection based on few-shot demonstrations.
* **Source Summaries** — source selection based on source-level summaries.

### Overall Routing Results

| Routing Strategy       | **Avg. Primary** | **Joint Primary** |
| ---------------------- | ---------------: | ----------------: |
| **Semantic Directory** |       **74.69%** |        **53.49%** |
| Few-shot Prompting     |           63.88% |            35.57% |
| Source Summaries       |           60.13% |            33.12% |

The **Semantic Directory** improves average routing accuracy by **10.81 points** over Few-shot Prompting and **14.56 points** over Source Summaries.

---

## 📌 Routing Results by Source Pair

| Source Pair   | Routing Strategy       | R1 Primary | R2 Primary | Joint Primary | **Avg. Primary** |
| ------------- | ---------------------- | ---------: | ---------: | ------------: | ---------------: |
| **KG–Table**  | **Semantic Directory** | **91.19%** | **91.80%** |    **84.83%** |       **91.50%** |
|               | Few-shot Prompting     |     86.05% |     76.29% |        63.91% |           81.17% |
|               | Source Summaries       |     85.27% |     77.07% |        66.70% |           81.17% |
| **KG–Doc**    | **Semantic Directory** | **72.01%** | **76.26%** |    **50.78%** |       **74.13%** |
|               | Few-shot Prompting     |     55.55% |     74.44% |        32.15% |           64.99% |
|               | Source Summaries       |     53.38% |     62.22% |        21.14% |           57.80% |
| **Table–Doc** | **Semantic Directory** | **41.70%** | **74.38%** |    **24.20%** |       **58.04%** |
|               | Few-shot Prompting     |     21.52% |     68.57% |        10.09% |           45.04% |
|               | Source Summaries       |     26.61% |     55.36% |        11.07% |           40.98% |

---

# ✅ 3. Takeaway

When **KG, Table, and Document are all simultaneously available**, SCOPE achieves **71.53%** final-answer accuracy on the full CMQA benchmark, compared with **73.50%** in the original source-pair setting.

More importantly, the **Semantic Directory** achieves the highest routing accuracy (**74.69%**), outperforming both **Few-shot Prompting (63.88%)** and **Source Summaries (60.13%)**.

These results show that SCOPE does **not rely on pre-specified source pairs** and can discover the appropriate heterogeneous sources online.

---

# 📁 4. Directory Layout

| Path                                       | Content                               |
| ------------------------------------------ | ------------------------------------- |
| `result/final_answer_accuracy/kg-doc/`                      | Full all-source results for KG–Doc    |
| `result/final_answer_accuracy/kg-table/`                    | Full all-source results for KG–Table  |
| `result/final_answer_accuracy/table-doc/`                   | Full all-source results for Table–Doc |
| `result/routing_accuracy/`                 | Routing-strategy evaluation           |
| `result/routing_accuracy/*/metrics.json`   | Per-source-pair routing metrics       |
| `result/routing_accuracy/*/per_query.csv`  | Per-query routing results             |
| `result/routing_accuracy/metrics_all.json` | Aggregated routing results            |

Private runtime configurations and credentials are not included.
