"""
把原始 issue 的 title + body 整理成一段 bug_report，並存到 unlabeled.jsonl。

另外保留來源資料中原本可直接取得的欄位 direct_fields，方便之後：
1. 給 Code Llama 讀取 bug_report 做欄位抽取
2. 與來源資料欄位做部分對比
"""
import json
import os
import re

RAW_PATH = "dataset/raw/issues.json"
OUT_PATH = "dataset/processed/unlabeled.jsonl"

def clean_text(s: str) -> str:
    if not s:
        return ""

    # remove code blocks
    s = re.sub(r"```[\s\S]*?```", "", s)

    # remove inline code
    s = re.sub(r"`[^`]*`", "", s)

    # remove images
    s = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", s)

    # convert links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", s)

    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"找不到 {RAW_PATH}，請先跑 fetch_issue_fields.py 產生 issues.json")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        issues = json.load(f)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    kept = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, it in enumerate(issues):
            title = clean_text(it.get("title", ""))
            body  = clean_text(it.get("body", ""))
            source = it.get("source", "unknown")
            issue_id = str(it.get("issue_id", f"issue_{i}"))
            direct_fields = it.get("direct_fields", {}) or {}

            # 很多 issue body 可能是 None；過短的先跳過（太短不好標註/抽欄位）
            bug_report = (title + " " + body).strip()
            if len(bug_report) < 40:
                continue

            record = {
                "id": issue_id,
                "source": source,
                "bug_report": bug_report,
                "meta": {
                    "title": title,
                    "direct_fields": {
                        "product": direct_fields.get("product", ""),
                        "severity": direct_fields.get("severity", ""),
                        "bug_type": direct_fields.get("bug_type", ""),
                        "component": direct_fields.get("component", ""),
                        "os": direct_fields.get("os", ""),
                        "version": direct_fields.get("version", ""),
                        "priority": direct_fields.get("priority", ""),
                        "error_message": direct_fields.get("error_message", ""),
                    }
                },
                # 先留空：下一步人工標註 ground truth
                "json_ground_truth": None
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. wrote {kept} records -> {OUT_PATH}")

if __name__ == "__main__":
    main()
