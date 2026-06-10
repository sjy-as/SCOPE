import os
import requests
from typing import Dict, List, Tuple, Optional


"""MMQA Table query client (compat wrapper).

This wrapper queries the local Table BM25 service, then augments each retrieved
hit with full table content loaded from `data_sources/Table/nba_wikisql.sql`
using `table_id`.
"""


_FULL_TABLE_CACHE: Optional[Dict[str, Dict]] = None


def _default_table_sql_path() -> str:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(cur_dir, "..", ".."))

    # Prefer explicit override first.
    env_path = os.environ.get("TABLE_SQL_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # Common candidate locations in this project layout.
    candidates = [
        os.path.join(project_root, "data", "data_sources", "Table", "nba_wikisql.sql"),
        os.path.join(project_root, "data_sources", "Table", "nba_wikisql.sql"),
        os.path.join(project_root, "..", "atomr", "data_sources", "Table", "nba_wikisql.sql"),
    ]
    for p in candidates:
        p_abs = os.path.abspath(p)
        if os.path.exists(p_abs):
            return p_abs

    # Fallback to primary expected path even if missing (caller handles empty cache).
    return os.path.abspath(candidates[0])


def _pg_unescape_copy_value(s: str) -> str:
    # PostgreSQL COPY text format escaping
    return (
        s.replace("\\t", "\t")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
    )


def _parse_copy_header(line: str) -> Tuple[str, List[str]]:
    # Example:
    # COPY nba_wikisql.t_1_10015132_1 (player, "no.", nationality) FROM stdin;
    line = line.strip()
    left = line[len("COPY ") :]
    table_and_cols = left.split(" FROM stdin", 1)[0]

    table_name = table_and_cols.split("(", 1)[0].strip()
    cols_raw = table_and_cols.split("(", 1)[1].rsplit(")", 1)[0]

    cols = []
    cur = []
    in_quotes = False
    for ch in cols_raw:
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if ch == "," and not in_quotes:
            cols.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        cols.append("".join(cur).strip())

    return table_name, cols


def _canonical_table_id(table_id: str) -> str:
    s = str(table_id or "").strip().lower()
    if not s:
        return s
    s = s.replace("-", "_")
    if s.startswith("nba_wikisql."):
        s = s.split(".", 1)[1]
    if not s.startswith("t_"):
        s = f"t_{s}"
    return s


def _load_full_table_cache(sql_path: Optional[str] = None) -> Dict[str, Dict]:
    global _FULL_TABLE_CACHE
    if _FULL_TABLE_CACHE is not None:
        return _FULL_TABLE_CACHE

    path = sql_path or _default_table_sql_path()
    table_map: Dict[str, Dict] = {}

    if not os.path.exists(path):
        _FULL_TABLE_CACHE = table_map
        return _FULL_TABLE_CACHE

    current_table_id = None
    current_cols: List[str] = []
    current_rows: List[List[str]] = []
    in_copy = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if not in_copy:
                if line.startswith("COPY nba_wikisql.") and " FROM stdin;" in line:
                    full_table_name, cols = _parse_copy_header(line)
                    tbl = full_table_name.split(".", 1)[1] if "." in full_table_name else full_table_name
                    current_table_id = _canonical_table_id(tbl)
                    current_cols = cols
                    current_rows = []
                    in_copy = True
                continue

            # in_copy block
            if line == "\\.":
                if current_table_id:
                    table_map[current_table_id] = {
                        "table_id": current_table_id,
                        "header": current_cols,
                        "rows": current_rows,
                    }
                current_table_id = None
                current_cols = []
                current_rows = []
                in_copy = False
                continue

            parts = line.split("\t")
            row = []
            for v in parts:
                if v == "\\N":
                    row.append("")
                else:
                    row.append(_pg_unescape_copy_value(v))
            current_rows.append(row)

    _FULL_TABLE_CACHE = table_map
    return _FULL_TABLE_CACHE


def _format_full_table_text(header: List[str], rows: List[List[str]], max_rows: int = 200) -> str:
    header = header or []
    rows = rows or []
    lines = []
    if header:
        lines.append("Header: " + " | ".join(str(h) for h in header))

    if rows:
        lines.append("Rows:")
        for i, r in enumerate(rows[:max_rows], start=1):
            lines.append(f"[{i}] " + " | ".join(str(c) for c in (r or [])))

    return "\n".join(lines).strip()


def _get_full_table_by_id(table_id: str) -> Dict:
    if not table_id:
        return {"header": [], "rows": [], "table_text": ""}

    cache = _load_full_table_cache()
    canonical = _canonical_table_id(table_id)
    item = cache.get(canonical)
    if item is None:
        return {"header": [], "rows": [], "table_text": ""}

    header = item.get("header") or []
    rows = item.get("rows") or []
    table_text = _format_full_table_text(header, rows)
    return {"header": header, "rows": rows, "table_text": table_text}


def _normalize_table_hit(hit: Dict, rank: int) -> Dict:
    page = hit.get("page_title", "")
    section = hit.get("section_title", "")
    caption = hit.get("caption", "")
    text = hit.get("text", "")
    table_id = hit.get("table_id", "")
    header = hit.get("header") or []
    rows_preview = hit.get("rows_preview") or []

    full_table = _get_full_table_by_id(str(table_id))
    full_header = full_table.get("header") or header
    full_rows = full_table.get("rows") or []
    full_table_text = full_table.get("table_text", "")

    title = page or section or caption or f"table_{rank}"
    snippet_parts = []
    if section:
        snippet_parts.append(f"Section: {section}")
    if caption and caption != section:
        snippet_parts.append(f"Caption: {caption}")
    if table_id:
        snippet_parts.append(f"Table ID: {table_id}")
    if full_header:
        snippet_parts.append("Header: " + " | ".join(str(h) for h in full_header))
    if rows_preview:
        top_row = rows_preview[0] if isinstance(rows_preview[0], list) else []
        if top_row:
            snippet_parts.append("Top Row: " + " | ".join(str(c) for c in top_row))
    if text:
        snippet_parts.append(text)

    snippet = " ; ".join(snippet_parts).strip()

    return {
        "rank": rank,
        "title": title,
        "snippet": snippet,
        "source": "table",
        "link": "",
        # put full table into article_text so existing formatter can consume it
        "article_text": full_table_text,
        "table_id": table_id,
        "page_title": page,
        "section_title": section,
        "caption": caption,
        "header": full_header,
        "rows_preview": rows_preview,
        "rows": full_rows,
    }


def execute_table_query(query: str, table_api_url: str, k: int = 5) -> Tuple[List[str], Dict]:
    resp = requests.get(table_api_url, params={"query": query, "k": k}, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("topk", []) or []

    clean_title_list: List[str] = []
    normalized_hits: List[Dict] = []

    for i, hit in enumerate(hits[:k], start=1):
        normalized = _normalize_table_hit(hit, i)
        normalized_hits.append(normalized)
        if normalized["title"]:
            clean_title_list.append(normalized["title"])

    full_results = {"organic_results": normalized_hits}
    return clean_title_list, full_results
