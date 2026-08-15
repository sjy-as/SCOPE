# 🔍 CMQA Quality Verification

This folder contains the independent quality verification of the full **CMQA benchmark (3,421 samples)**.

Three annotators independently assess each sample along four quality dimensions. We report both **dimension-level agreement** and the final **sample-level retain/reject agreement**.

---

## 📌 Verification Dimensions

| Dimension                  | What is checked                                                                  |
| -------------------------- | -------------------------------------------------------------------------------- |
| **Question Validity**      | Whether the question is clear, meaningful, and answerable                        |
| **Answer Correctness**     | Whether the provided gold answer is correct                                      |
| **Bridge Consistency**     | Whether the bridge entity correctly connects the two evidence sources            |
| **Cross-Source Necessity** | Whether answering the question genuinely requires evidence from multiple sources |

---

## 📊 Main Results

| Metric                           |     Result |
| -------------------------------- | ---------: |
| Samples verified                 |  **3,421** |
| Annotators                       |      **3** |
| Avg. binary agreement            | **95.77%** |
| Sample-level three-way agreement | **85.03%** |
| Overall Fleiss' κ                |  **0.426** |

The results show high agreement on the binary quality judgments and consistent sample-level retain/reject decisions across annotators.

### Dimension-Level Agreement

| Dimension              | Binary Agreement | Fleiss' κ |
| ---------------------- | ---------------: | --------: |
| Question Validity      |       **99.42%** |    -0.002 |
| Answer Correctness     |       **94.15%** |     0.427 |
| Bridge Consistency     |       **97.57%** |     0.127 |
| Cross-Source Necessity |       **91.93%** |     0.472 |
| **Macro Average**      |       **95.77%** | **0.256** |

> **Note:** κ can be low when one label strongly dominates, even when raw agreement is very high. We therefore report both agreement rates and chance-corrected statistics.

---

## 📁 Directory Structure

```text
R3_Human_Quality_Verification/
├── annotations/
│   ├── annotations_A.jsonl
│   ├── annotations_B.jsonl
│   └── annotations_C.jsonl
│
└── reports/
    ├── agreement_metrics.json
    ├── annotator_manifest.json
    ├── sample_inventory.jsonl
    └── validation_report.json
```

### `annotations/`

Contains the independent annotation results from annotators **A**, **B**, and **C**.

### `reports/`

| File                      | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| `agreement_metrics.json`  | Agreement statistics for all four dimensions and final retain/reject decisions |
| `annotator_manifest.json` | Annotation configuration and annotator metadata                                |
| `sample_inventory.jsonl`  | Inventory of all 3,421 CMQA samples used for verification                      |
| `validation_report.json`  | Completeness check for the annotation files                                    |

---

## ✅ Completeness Check

All three annotation files contain exactly **3,421 records**, with no missing samples detected.

```text
Annotator A: 3421 / 3421 ✓
Annotator B: 3421 / 3421 ✓
Annotator C: 3421 / 3421 ✓
```

This verification is used to assess the reliability and quality of the released CMQA benchmark.
