# 🚫 Why We Do Not Evaluate SCOPE on AtomR's BlendQA Setup

We investigated whether SCOPE could be fairly evaluated on the **BlendQA setup used by AtomR**.

AtomR's original pipeline relies on three heterogeneous sources:

> **KB + Text + Web**

However, the original retrieval environment is no longer fully reproducible. In particular, the **KB service used by AtomR is currently unavailable**, while the Web source is only accessible through an external search API rather than as a fixed local collection.

---

## ✨ Key Findings

| Source   | Original Access           | Current Status   | Locally Inspectable |
| -------- | ------------------------- | ---------------- | :-----------------: |
| **KB**   | VisKoP online service     | ❌ Unavailable    |          ❌          |
| **Text** | Atlas / ColBERT Wikipedia | ✅ Available      |          ✅          |
| **Web**  | Google SerpAPI            | ✅ API accessible |          ❌          |

The main limitation is therefore not BlendQA itself, but the **availability of its original heterogeneous retrieval environment**.

---

## 1. ❌ KB Service Is Currently Unavailable

AtomR accesses its structured KB through the public VisKoP services:

```text
Semantic Parser:
https://viskop.xlore.cn/programApi

KoPL Engine:
https://viskop.xlore.cn/large
```

We tested both endpoints on **August 14, 2026**.

Both returned:

```text
HTTP/2 502 Bad Gateway
```

Specifically:

```text
POST https://viskop.xlore.cn/programApi
→ HTTP 502 Bad Gateway

POST https://viskop.xlore.cn/large
→ HTTP 502 Bad Gateway
```

The root domain also returned HTTP 502 when tested without a local proxy.

This indicates that the public gateway is reachable, but the backend services required by AtomR are currently unavailable.

> **Therefore, AtomR's original KB retrieval pipeline can no longer be directly reproduced using the released configuration.**

---

## 2. ✅ Text Source Is Reproducible

AtomR uses an **Atlas / ColBERT Wikipedia retrieval stack** for the Text source.

We recovered the required local resources, including:

* processed Wikipedia corpus;
* constructed ColBERT index;
* local retrieval resources.

Therefore, the Text component can still be reproduced locally:

```text
Wikipedia Corpus
      ↓
ColBERT Index
      ↓
Local Retrieval
```

The Text source itself does not prevent reproduction.

---

## 3. 🌐 Web Is API-Accessible but Not Locally Inspectable

AtomR retrieves Web information through **Google Search via SerpAPI**.

Queries can still be issued with a valid API key:

```text
Query
  ↓
Google SerpAPI
  ↓
Web Results
```

However, the Web source is not a fixed local collection.

Unlike the local Wikipedia corpus, we cannot enumerate the complete underlying Web source to obtain its:

* semantic concepts;
* coverage;
* source structure;
* source-level metadata.

This matters for SCOPE because its **Semantic Directory is constructed from source-level content and metadata before online reasoning**.

---

## 4. 🔍 Why This Prevents a Fair SCOPE Evaluation

SCOPE requires consistent access to heterogeneous sources in order to construct the **Semantic Directory**.

Under AtomR's original BlendQA setup:

| Source   | Usable by AtomR | Usable for SCOPE Directory |
| -------- | :-------------: | :------------------------: |
| **KB**   |        ❌        |              ❌             |
| **Text** |        ✅        |              ✅             |
| **Web**  |        ✅        |         ⚠️ Limited         |

Only the Text source is both available and locally inspectable.

This creates two major problems.

### Problem 1 — Semantic Directory Construction

For KB and Web, the original source-level content or metadata cannot be consistently accessed.

As a result, constructing SCOPE's Semantic Directory would require additional approximations or alternative sources, changing the original experimental setting.

### Problem 2 — AtomR Can No Longer Be Faithfully Re-run

The original AtomR KB service is unavailable.

Replacing it with:

* another Wikidata endpoint,
* a newly constructed local KG, or
* another retrieval backend

would change AtomR's original retrieval behavior.

Therefore, results obtained from such a modified setup would not be directly comparable with the BlendQA results reported in the AtomR paper.

---

## 📝 Conclusion

We therefore **do not evaluate SCOPE on AtomR's original BlendQA setup**.

The reason is that the original heterogeneous retrieval environment is no longer fully reproducible:

```text
KB   → ❌ Original service unavailable
Text → ✅ Locally reproducible
Web  → ⚠️ API accessible, but not locally enumerable
```

Consequently:

1. SCOPE cannot construct its Semantic Directory consistently over all three original sources; and
2. AtomR itself cannot be faithfully re-executed under its original KB retrieval environment.

For a fair comparison, we therefore retain evaluations in settings where the underlying heterogeneous sources are **consistently accessible to all compared methods**.
