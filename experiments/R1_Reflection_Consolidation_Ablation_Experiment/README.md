# 🔁 Reflection & Consolidation Ablation

This experiment disentangles the roles of **Reflection** and **Consolidation** in SCOPE and addresses **Reviewer #1 — W6**.

## 🎯 Key Findings

| Setting               | Reflection | Consolidation |          Accuracy Effect | Main Role                            |
| --------------------- | :--------: | :-----------: | -----------------------: | ------------------------------------ |
| **Full SCOPE**        |      ✅     |       ✅       |                **0.721** | Accuracy + efficiency                |
| **w/o Consolidation** |      ✅     |       ❌       | **0.729** (~1 pp change) | Similar accuracy, higher cost        |
| **w/o Reflection**    |      ❌     |       ✅       |             **−17.8 pp** | Large accuracy degradation           |
| **w/o Both**          |      ❌     |       ❌       |             **−26.0 pp** | Matches the original ablation result |

**Conclusion.** Reflection is the primary **accuracy safeguard**, while Consolidation mainly improves **execution efficiency** by converting repeated reflection outcomes into reusable experience.

## ⚙️ Efficiency Impact of Consolidation

Without Consolidation, execution becomes more expensive:

| Metric               | Change w/o Consolidation |
| -------------------- | -----------------------: |
| 🤖 LLM calls         |               **+20.0%** |
| 🔎 Retrieval calls   |                **+5.2%** |
| 🔄 Re-route attempts |               **+29.9%** |

### Full-Model Cost Baseline

| Source Pair | Questions |  LLM Calls | Retrieval Calls |   LLM / Q | Retrieval / Q |
| ----------- | --------: | ---------: | --------------: | --------: | ------------: |
| KG–Doc      |     1,154 |     20,580 |           2,748 |     17.83 |          2.38 |
| KG–Table    |     1,147 |     18,680 |           2,547 |     16.29 |          2.22 |
| Table–Doc   |     1,120 |     15,994 |           2,609 |     14.28 |          2.33 |
| **Overall** | **3,421** | **55,254** |       **7,904** | **16.15** |      **2.31** |


## 🧠 What Consolidation Stores

Consolidation turns reflection outcomes into structured reusable experience:

```json
{
  "stage": "routing | semantic | planning",
  "when": "condition under which the experience applies",
  "action": "recommended routing / parsing / planning action",
  "lesson": "generalized lesson learned from previous executions",
  "evidence": "supporting record IDs"
}
```

### Example

| Stage       | Learned Experience                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🧭 Routing  | Narrative player-event or venue questions should prefer **Document**, while structured team relations can be resolved through **KG**.             |
| 📋 Routing  | Game records, scores, and playoff outcomes should prefer **Table** when the relevant structured fields are available.                             |
| 🧩 Planning | For questions asking which teams a player later played for, query the KG relation **`member_of_sports_team`** instead of aggregating game tables. |

## 🧪 Experimental Setup

| Item                        | Setting                       |
| --------------------------- | ----------------------------- |
| Model                       | `deepseek-chat`               |
| Routing mode                | `graph`                       |
| Prompt version              | `v2`                          |
| Semantic metadata injection | ✅ Enabled                     |
| Source pairs                | KG–Doc / KG–Table / Table–Doc |
| Total questions             | **3,421**                     |
| Failed runs                 | **0**                         |
| Consolidation interval      | **50** questions              |

## ✅ Takeaway

> **Reflection protects answer accuracy; Consolidation reduces repeated reasoning and execution cost.**
