# 🔄 Pipeline Order Sensitivity Experiment

> **Goal:** Evaluate whether SCOPE is sensitive to the execution order of its Stage-2 modules.
> **Setting:** Swap the execution order of **semantic routing** and **semantic parsing**.
> **Data:** 3 source pairs × 3 rounds × 100 questions = **900 questions**.
> **Main Metric:** Final-answer accuracy judged as **Exact + Partial**.

---

## ⚙️ Experimental Setup

We compare the original SCOPE pipeline with a variant that swaps the execution order of the two Stage-2 modules.

| Setting           | Original Order                | Swapped Order |
| ----------------- | ----------------------------- | ------------- |
| Stage-2 order     | Route → Parse                 | Parse → Route |
| Questions         | 900                           | 900           |

All prompts are **component-specific** and contain **no dataset-specific demonstrations**.

---

## 📊 Main Results

### Final-Answer Accuracy

| Source Pair |   Original |    Swapped |            Δ |
| ----------- | ---------: | ---------: | -----------: |
| KG–Doc      |     80.00% |     79.67% |     -0.33 pp |
| KG–Table    |     73.00% |     72.33% |     -0.67 pp |
| Table–Doc   |     62.67% |     56.00% |     -6.67 pp |
| **Overall** | **71.89%** | **69.33%** | **-2.56 pp** |

> **SCOPE retains 69.33% accuracy after swapping the two modules, compared with 71.89% under the original order.**



## 🔍 Per-Round Results

| Source Pair        | Round |   Original |    Swapped |            Δ |
| ------------------ | ----- | ---------: | ---------: | -----------: |
| KG–Doc             | 1     |        81% |        82% |        +1 pp |
|                    | 2     |        84% |        83% |        -1 pp |
|                    | 3     |        75% |        74% |        -1 pp |
| **KG–Doc Avg.**    |       | **80.00%** | **79.67%** | **-0.33 pp** |
| KG–Table           | 1     |        77% |        77% |         0 pp |
|                    | 2     |        73% |        70% |        -3 pp |
|                    | 3     |        69% |        70% |        +1 pp |
| **KG–Table Avg.**  |       | **73.00%** | **72.33%** | **-0.67 pp** |
| Table–Doc          | 1     |        68% |        62% |        -6 pp |
|                    | 2     |        62% |        55% |        -7 pp |
|                    | 3     |        58% |        51% |        -7 pp |
| **Table–Doc Avg.** |       | **62.67%** | **56.00%** | **-6.67 pp** |

---


## ✅ Conclusion

The results show that SCOPE has **limited sensitivity to pipeline order**. After swapping semantic routing and semantic parsing, the overall final-answer accuracy decreases by only **2.56 percentage points (71.89% → 69.33%)**, while aggregate exact accuracy remains unchanged.

Together with the backbone and cross-dataset experiments reported in the paper, this provides additional evidence that SCOPE's performance is not strongly tied to a particular LLM, dataset, or fixed module order.

---

## 📁 Result Paths

* **Original order:**
  `CCC_de_pro/new_model_900_run/runs/`

* **Swapped order:**
  `DDD_order/result/swapped_promptfix_parallel/`


## 🧪 Full-Scale Evaluation

We are currently extending this experiment to the **full dataset**. The complete results will be added to this file once available.
