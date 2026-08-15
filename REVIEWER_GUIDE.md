# 🔍 Reviewer Guide

This document provides a quick guide to the released **SCOPE implementation, evaluation package, execution examples, and reviewer-requested experiments**.

---

## 🧩 1. Code & Reproducibility

The [`SCOPE/`](./SCOPE/) directory is the main entry point of the SCOPE framework.

To improve reproducibility and address the issues identified during review, we have completed the released artifact with the following materials:

| Resource          | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `SCOPE/`          | Main SCOPE framework and execution pipeline                |
| `SCOPE/eval/`     | Evaluation and configuration utilities                     |
| `SCOPE_wo_*`      | Entry points for the corresponding ablation settings       |
| `results/traces/` | Representative execution trajectories and running examples |

### Evaluation Files

The previously missing evaluation-related files have now been uploaded, including:

```text
SCOPE/eval/
├── analyze_routes.py
├── evaluate_answer_kg_doc.py
├── evaluate_answer_kg_table.py
├── evaluate_answer_table_doc.py
└── models_config.py
```

### Ablation Entry Points

The released implementation now also includes dedicated entry points for the major ablation settings, including:

```text
SCOPE_wo_consolidation
SCOPE_wo_decomposition
SCOPE_wo_operator_planning
SCOPE_wo_reflection
SCOPE_wo_semantic_directory_few-shot
SCOPE_wo_semantic_directory_summary
SCOPE_wo_semantic_guide_on_planning
```

Representative execution samples are provided in [`results/traces/`](./results/traces/) for inspecting the actual reasoning and execution process.

These updates mainly address **Reviewer #1 — W4/W5/W7** regarding the evaluation package, released artifact, and reproducibility.

---

## 🧪 2. Reviewer-Requested Experiments

The [`experiments/`](./experiments/) directory contains the additional experiments conducted in response to reviewer concerns.

Each experiment folder provides its own **README, experimental setup, results, and supporting evidence**.

| Experiment                                        | Reviewer Concern                | What It Addresses                                       |
| ------------------------------------------------- | ------------------------------- | ------------------------------------------------------- |
| `R1_All-Source_Availability_Experiment`           | **R1 — W1**                     | Dependence on pre-specified source pairs                |
| `R1_Adaptive_Decomposition_Experiment`            | **R1 — W1**                     | Dependence on fixed two-step decomposition              |
| `R1R2R3_Cross-Domain_Generalization_Experiment`   | **R1 — W2 / R2 — W4 / R3 — W2** | Cross-domain generalization and benchmark dependence    |
| `R1_Pipeline_Order_Sensitivity_Experiment`        | **R1 — W3**                     | Sensitivity to pipeline/module order                    |
| `R1_Reflection_Consolidation_Ablation_Experiment` | **R1 — W6**                     | Separate effects of Reflection and Consolidation        |
| `R2_AtomR_Reproduction_Fairness_Experiment`       | **R2 — W3**                     | Fairness and fidelity of the AtomR reproduction         |
| `R2_LLM_Judge_Reliability_Experiment`             | **R2 — W4**                     | Reliability of the LLM Judge and robustness under EM/F1 |
| `R3_Human_Quality_Verification`                   | **R3 — W2**                     | Independent human verification of CMQA quality          |

---

## 📌 Recommended Review Path

For convenient inspection, we recommend the following order:

**Framework & Code** → [`SCOPE/`](./SCOPE/)
**Evaluation Utilities** → [`SCOPE/eval/`](./SCOPE/eval/)
**Reviewer-Requested Evidence** → [`experiments/`](./experiments/)
**Execution Examples** → [`results/traces/`](./results/traces/)

The corresponding experiment README files provide the detailed settings, evidence, and results for each reviewer concern.

---

Thank you again for the constructive feedback. We hope these additional materials make the implementation and experimental evidence easier to inspect and reproduce.
