"""HydraRAG baseline 全局配置。

run.py 会在启动时用命令行参数覆盖这里的 LLM 字段，因此 pipeline 里的所有
helper 直接读 config 模块即可拿到最新值。
"""
import os

# --------------------------------------------------------------------------
# 路径（self-contained：全部指向 hydra_baseline 目录内的拷贝）
# --------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
KG_DIR = os.path.join(_BASE, "data_sources", "KG")
WIKISQL_PATH = os.path.join(_BASE, "data_sources", "nba_wikisql.sql")

# --------------------------------------------------------------------------
# Table BM25 检索服务（与 atomr / deepservice / new_model 共用同一 HTTP 服务）
# --------------------------------------------------------------------------
TABLE_API_URL = os.environ.get("HYDRA_TABLE_API_URL", "http://127.0.0.1:1216/api/search")
K_TABLE = 5            # 每次 BM25 取回的候选表数量

# --------------------------------------------------------------------------
# Doc ColBERT 段落检索服务（kg-doc 任务用；与 atomr / deepservice / new_model 共用）
# --------------------------------------------------------------------------
DOC_API_URL = os.environ.get("HYDRA_DOC_API_URL", "http://127.0.0.1:1215/api/search")
K_DOC = 5              # 每次 ColBERT 取回的候选段落数量

# --------------------------------------------------------------------------
# LLM（OpenAI 兼容接口；run.py 用 --llm-* 覆盖）
# --------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
LLM_MODEL    = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")

# --------------------------------------------------------------------------
# HydraRAG 超参数
# --------------------------------------------------------------------------
MAX_ITERATIONS  = 3    # 最大迭代轮数（证据不足时自动再检索，保留原版 3 轮）
KG_MAX_HOP      = 2    # KG 子图探索最大跳数（benchmark 为 2-hop，KG 侧 1~2 跳足够）
KG_MAX_DEGREE   = 60   # 单节点扩展的关系/邻居上限，防爆图
TABLE_CONVERT_TOPN = 6 # 每轮最多把多少张表 / 多少段文档转成 KG 边（控制 LLM 调用量）
BEAM_RECALL_CAP = 80   # 送进 LLM 排序前的候选上限（仅控制上下文，不做打分排序）
BEAM_TOPN       = 6    # LLM 最终选出的证据条数

# 不同来源的可信度（HydraRAG 多源融合用）：KG 边最高，表/文档转边次之
SOURCE_RELIABILITY = {
    "kg":          1.00,   # 来自本地 KG 的真实边
    "table":       0.80,   # 由 Table 行经 LLM 转换得到的 KG 边
    "table_raw":   0.75,   # Table 原始行（未转边）
    "doc":         0.75,   # 由文档段落经 LLM 转换得到的 KG 边
    "doc_raw":     0.70,   # 文档原始段落（未转边）
    "text":        0.70,   # 兼容旧名
}
