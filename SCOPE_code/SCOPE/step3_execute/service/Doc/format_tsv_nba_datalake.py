import json
import os
import re


def _clean(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\t", " ")
    s = s.replace("\r", " ")
    s = s.replace("\n", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))

    input_path = os.path.join(repo_root, "datasets", "data_source", "text", "nba-datalake.wiki-documents.json")
    output_path = os.path.join(base_dir, "data_tsv", "nba_datalake_title_text.tsv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            raw = f.read()
            raw = re.sub(r",\s*(?=[\]}])", "", raw)
            raw = re.sub(r"([\]}])\s*(?=\")", r"\1,", raw)
            data = json.loads(raw)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("id\ttext\ttitle\n")

        pid = 1
        for entry in data:
            title = _clean(entry.get("wikipedia_title") or entry.get("title") or "")
            text_list = entry.get("text")
            if isinstance(text_list, list):
                text = " ".join(_clean(x) for x in text_list if x is not None)
            else:
                text = _clean(text_list)

            if not title and not text:
                continue

            out.write(f"{pid}\t{text}\t{title}\n")
            pid += 1

    print(f"Formatting completed. TSV file saved to '{output_path}'. Total passages: {pid-1}")


if __name__ == "__main__":
    main()
