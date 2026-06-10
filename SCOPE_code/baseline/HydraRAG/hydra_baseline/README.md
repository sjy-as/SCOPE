# HydraRAG baseline（本地 KG + Table / Doc 版）

把原版 HydraRAG 重构成与 `atomr` / `deepservice` 平级的 baseline，跑 `new_model`
的任务（NBA 领域 2-hop 多源问答），支持两类任务，用 `--mode` 切换：

- `--mode kg-table`（默认）：跑 `qa_bench/kg-table-160.jsonl`，外部补充头用 **Table BM25**。
- `--mode kg-doc`：跑 `qa_bench/kg-doc-160.jsonl`，外部补充头用 **Doc ColBERT 段落检索**。

- **知识库**：本地 NBA 知识图谱（`data_sources/KG/*.csv`），替换原版的 Freebase SPARQL / Wikidata RPC
- **外部补充**：Table BM25 检索服务（kg-table）或 Doc ColBERT 段落检索服务（kg-doc），
  替换原版的 Web / Wikipedia 检索
- 原版代码完整保留在 `../Hydra_run/`，本目录是全新干净实现，自包含、不依赖 new_model。

## HydraRAG 算法流程（保留原版灵魂）

```
Stage A 问题理解   LLM 分解出 Thinking CoT + 子问题，抽取 topic 实体
Stage B 实体定位   topic 实体链接到本地 NBA KG（精确 + 模糊匹配，拿 qid）
Stage C 多源检索   KG 头：从实体做子图 BFS，枚举 KG 路径
                   外部头：kg-table → BM25 检索相关表 → 用 LLM 把表行【转成 KG 边】
                          kg-doc   → ColBERT 检索相关段落 → 用 LLM 把段落【转成 KG 边】
Stage D 证据融合   多源 beam search 三步排序（相关性→源可信度→LLM 选 top 路径）
                   KG 边可信度最高，表/文档转出来的边次之
Stage E 答案+迭代  LLM 判 {Yes}/{No}；不足则重写检索 query / 预测桥接实体再检索，
                   最多 3 轮；最后强制综合出最终答案
```

「把检索到的外部证据改成 KG 边」是核心机制：kg-table 由 `table/table_to_kg.py`、
kg-doc 由 `doc/doc_to_kg.py` 完成。KG 路径与转出来的边共用统一表示
`{Head} -[relation]-> {Tail}`，从而能一起融合排序。两个外部头暴露同一套接口
（`retrieve` / `to_kg_edges` / `unit_id`），pipeline 与「外部源是表还是文档」无关。

## 目录结构

```
hydra_baseline/
├── config.py              全局配置（路径 / LLM / 超参）
├── llm.py                 OpenAI 兼容 LLM 客户端（重试 + 调用计数）
├── prompts.py             全部 prompt（由原版 cot_prompt_list.py 适配）
├── pipeline.py            HydraRAG 主流程（5 阶段 + 3 轮迭代）
├── run.py                 驱动：跑 benchmark，产出 predictions / traces
├── kg/
│   ├── localkg_index.py   本地 KG 索引（拷贝自 new_model）
│   ├── kg_retriever.py    KG 检索服务层（拷贝自 new_model，已自包含）
│   └── kg_explorer.py     KG 头：子图 BFS + 路径枚举【新写】
├── table/
│   ├── table_retriever.py Table BM25 客户端（拷贝自 new_model）
│   └── table_to_kg.py     Table 行 → KG 边【kg-table 外部头，核心】
├── doc/
│   ├── doc_retriever.py   Doc ColBERT 段落检索客户端（瘦 HTTP 客户端）
│   └── doc_to_kg.py       文档段落 → KG 边【kg-doc 外部头，核心】
├── fusion/
│   └── beam_search.py     多源 beam search 三步排序【新写】
├── eval/
│   └── evaluate.py        LLM-judge 评测（只评最终答案）
└── data_sources/          自包含的 KG CSV + Table SQL
```

## 运行

### 1. 前置：启动外部检索服务

- **kg-table** 依赖 Table BM25 HTTP 服务（默认 `http://127.0.0.1:1216/api/search`），
  启动脚本见 `new_model/step3_execute/service/Table/serve_table_bm25.py`。
- **kg-doc** 依赖 Doc ColBERT 段落检索 HTTP 服务（默认 `http://127.0.0.1:1215/api/search`），
  启动脚本见 `table_service/setup_service_nba_datalake.py`。

两个服务都与 atomr / deepservice / new_model 共用。服务不可达时对应的外部头会取空，
pipeline 仍能只用 KG 证据继续跑（结果会变差）。

### 2. 跑 baseline

kg-table（默认）：

```bash
cd /root/autodl-tmp/baseline/HydraRAG/hydra_baseline
python3 run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --output-dir result/run1 \
  --workers 8 \
  --api-key "你的API_KEY" \
  --llm-url "https://api.chatanywhere.tech/v1" \
  --llm-model "deepseek-chat"
```

kg-doc（加 `--mode kg-doc`）：

```bash
python3 run.py \
  --mode kg-doc \
  --input /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --output-dir result/run_doc \
  --workers 8 \
  --api-key "你的API_KEY" \
  --llm-url "https://api.chatanywhere.tech/v1" \
  --llm-model "deepseek-chat"
```

先小规模验证可加 `--max 5`；中断后加 `--resume` 续跑。

产物：
- `result/run1/predictions.jsonl`  每题预测（`final` 字段为最终答案）
- `result/run1/traces/idx_<i>.trace.json`  每题完整 trace（5 阶段 + 迭代 + 选中的 KG 边证据 + LLM 调用）
- `result/run1/summary.json`  运行汇总

### 3. 评测（LLM-judge，只评最终答案）

```bash
# kg-table
python3 eval/evaluate.py \
  --pred result/run1/predictions.jsonl \
  --gold /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --api-key "你的API_KEY"

# kg-doc
python3 eval/evaluate.py \
  --pred result/run_doc/predictions.jsonl \
  --gold /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --api-key "你的API_KEY"
```

主问题的标准答案取 gold 的 `q2` 答案（最后一跳），判定 exact / partial / miss，
报告写到 `result/run1/eval_report.json`。

## 关键超参（config.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `MAX_ITERATIONS` | 3 | 迭代轮数（证据不足自动再检索）|
| `KG_MAX_HOP` | 2 | KG 子图探索最大跳数 |
| `K_TABLE` | 5 | 每次 BM25 取回的候选表数（kg-table）|
| `K_DOC` | 5 | 每次 ColBERT 取回的候选段落数（kg-doc）|
| `TABLE_CONVERT_TOPN` | 6 | 每轮最多转成 KG 边的表/段落数（控制 LLM 调用量）|
| `BEAM_STEP1/2/3_TOPN` | 80/40/6 | beam search 三步各自保留的证据数 |
