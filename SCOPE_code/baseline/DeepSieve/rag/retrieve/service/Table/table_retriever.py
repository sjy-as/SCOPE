from typing import List, Optional
import re
import requests
import time

class TableRetriever:
    """Client wrapper for the BM25 table retrieval service."""

    WIKISQL_PATH = "/root/autodl-tmp/AAA_new_3.24日晚上/data_sources/Table/nba_wikisql.sql"

    def __init__(self, api_url: str = "http://127.0.0.1:1216/api/search", timeout: int = 60):
        self.api_url = api_url
        self.timeout = timeout
        
        # 💡 新增：内存缓存，防止重复扫描巨大的 SQL 文件
        self._rows_cache = {}
        self._header_cache = {}

    def retrieve_topk_tables(self, query: str, k: int = 5, max_retries: int = 3) -> Optional[List[dict]]:
        """Retrieve top-k tables with a robust retry mechanism."""
        params = {"query": query, "k": k}
        
        # 💡 新增：简单的重试机制，增强网络请求的健壮性
        for attempt in range(max_retries):
            try:
                resp = requests.get(url=self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json().get("topk")
            except requests.exceptions.RequestException as e:
                print(f"[TableRetriever] API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待 1 秒后重试
                else:
                    return None

    def is_alive(self) -> bool:
        try:
            resp = requests.get(self.api_url, params={"query": "test", "k": 1}, timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------ #
    #  SQL row loading helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _table_id_to_sql_name(table_id: str) -> str:
        if not table_id:
            return ""
        return "t_" + table_id.replace("-", "_")

    def _load_full_rows(self, table_id: str) -> List[List[str]]:
        if not table_id:
            return []
            
        # 💡 命中缓存直接返回，速度提升百倍
        if table_id in self._rows_cache:
            return self._rows_cache[table_id]

        target = self._table_id_to_sql_name(table_id)
        rows: List[List[str]] = []
        in_copy = False
        
        try:
            with open(self.WIKISQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not in_copy:
                        if line.startswith("COPY") and target in line:
                            in_copy = True
                        continue
                        
                    line = line.rstrip("\n")
                    if line == "\\.":
                        break
                    if not line:
                        continue
                    rows.append(line.split("\t"))
                    
            # 存入缓存
            self._rows_cache[table_id] = rows
        except FileNotFoundError:
            print(f"[TableRetriever] SQL file not found at {self.WIKISQL_PATH}")
        except Exception as e:
            print(f"[TableRetriever] _load_full_rows failed for {table_id}: {e}")
            
        return rows

    def _load_header_from_sql(self, table_id: str) -> List[str]:
        if not table_id:
            return []
            
        if table_id in self._header_cache:
            return self._header_cache[table_id]

        target = self._table_id_to_sql_name(table_id)
        header: List[str] = []
        in_create = False
        
        try:
            with open(self.WIKISQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if not in_create:
                        if "CREATE TABLE" in stripped and target in stripped:
                            in_create = True
                        continue
                        
                    if stripped.startswith(")"):
                        break
                        
                    # 💡 优化正则，适应更复杂的 SQL 定义格式
                    col_m = re.match(r'^"?([^"]+)"?\s+[a-zA-Z]+', stripped.rstrip(","))
                    if col_m:
                        header.append(col_m.group(1).strip())
                        
            self._header_cache[table_id] = header
        except Exception as e:
            print(f"[TableRetriever] _load_header_from_sql failed: {e}")
            
        return header

    # ------------------------------------------------------------------ #
    #  Table formatters                                                  #
    # ------------------------------------------------------------------ #

    def build_filter_context(self, evidence: list) -> dict:
        if not evidence:
            return {"page_title": "", "section_title": "", "header": [], "sample_row": {}}
        first = evidence[0]
        row_dict = first.get("matched_row", {})
        header = list(row_dict.keys()) if row_dict else []
        return {
            "page_title":    first.get("page_title", ""),
            "section_title": first.get("section_title", ""),
            "header":        header,
            "sample_row":    row_dict,
        }

    def format_table(self, table: dict) -> str:
        table_id      = table.get("table_id", "")
        page_title    = table.get("page_title", "")
        section_title = table.get("section_title", "")
        caption       = table.get("caption", "")
        header        = table.get("header") or []
        rows          = table.get("rows_preview") or []
        
        lines = []
        # 💡 核心修复：把 table_id 喂给大模型！
        if table_id:
            lines.append(f"Table ID: {table_id}")
        if page_title:
            lines.append(f"Page: {page_title}")
        if section_title:
            lines.append(f"Section: {section_title}")
        if caption and caption != section_title:
            lines.append(f"Caption: {caption}")
        if header:
            lines.append("Header: " + " | ".join(str(h) for h in header))
        for i, row in enumerate(rows[:5]):
            lines.append(f"Row {i+1}: " + " | ".join(str(c) for c in row))
            
        return "\n".join(lines)

    def format_table_full(self, table: dict) -> str:
        table_id      = table.get("table_id", "")
        page_title    = table.get("page_title", "")
        section_title = table.get("section_title", "")
        caption       = table.get("caption", "")
        header        = table.get("header") or []
        
        full_rows = self._load_full_rows(table_id) if table_id else []
        if not full_rows:
            full_rows = table.get("rows_preview") or []
            
        if not header and table_id:
            header = self._load_header_from_sql(table_id)
        if not header:
            matched_row = table.get("matched_row", {})
            if matched_row:
                header = list(matched_row.keys())
                
        lines = []
        # 💡 核心修复：把 table_id 喂给大模型！
        if table_id:
            lines.append(f"Table ID: {table_id}")
        if page_title:
            lines.append(f"Page: {page_title}")
        if section_title:
            lines.append(f"Section: {section_title}")
        if caption and caption != section_title:
            lines.append(f"Caption: {caption}")
        if header:
            lines.append("Header: " + " | ".join(str(h) for h in header))
        for i, row in enumerate(full_rows):
            lines.append(f"Row {i+1}: " + " | ".join(str(c) for c in row))
            
        return "\n".join(lines)

    def format_tables(self, tables: List[dict], full: bool = False) -> List[str]:
        fmt = self.format_table_full if full else self.format_table
        return [fmt(t) for t in tables]