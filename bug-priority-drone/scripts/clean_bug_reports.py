"""Clean raw Eclipse Bugzilla records before feature extraction.

輸入是 `fetch_eclipse_bugzilla.py` 產生的 raw CSV；輸出是後續 split、
REP- 權重訓練與 feature extraction 共用的 cleaned CSV。
"""

import re
import argparse
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "eclipse_bugzilla_raw.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "eclipse_bug_reports_clean.csv")

# Priority 是原始 Bugzilla 欄位；這裡只做「可讀性」標籤，不混用 severity 的語意。
PRIORITY_MAP = {
    "P1": "Highest Priority",
    "P2": "High Priority",
    "P3": "Medium Priority",
    "P4": "Low Priority",
    "P5": "Lowest Priority",
}

# priority_num 是給後續模型使用的 ordinal label，直接對齊 Ticket 等級：
# P1(Blocker) -> 1 ... P5(Trivial) -> 5。
# 這樣訓練與評估輸出的 class 1~5 就會等同於 P1~P5，避免結果解讀反向。
PRIORITY_NUM_MAP = {
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
    "P5": 5,
}

# ===== 清理規則設定 =====
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 5000
RARE_VALUE_THRESHOLD = 5

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
LONG_ALNUM_PATTERN = re.compile(r"\b[a-zA-Z0-9_./\\:-]{40,}\b")
REPEATED_SYMBOLS_PATTERN = re.compile(r"([=\-_*#])\1{3,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(x: object) -> str:
    """
    清理文字欄位，規則如下：
    1. 缺值轉為空字串。
    2. 統一小寫。
    3. 將 URL、Email、Hex Code 置換為統一標籤，避免特徵空間爆炸。
    4. 移除過長的 log / stack trace 殘影字串。
    5. 移除大量重複符號與多餘空白。
    """
    if pd.isna(x):
        return ""

    text = str(x)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.lower()
    text = URL_PATTERN.sub(" <URL> ", text)
    text = EMAIL_PATTERN.sub(" <EMAIL> ", text)
    text = HEX_PATTERN.sub(" <HEX> ", text)
    text = LONG_ALNUM_PATTERN.sub(" ", text)
    text = REPEATED_SYMBOLS_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def truncate_text(text: str, max_len: int = MAX_TEXT_LEN) -> str:
    """限制文字長度，避免極長 description 造成雜訊過高。"""
    if len(text) <= max_len:
        return text
    return text[:max_len].strip()


def replace_rare_categories(df: pd.DataFrame, col: str, min_count: int = RARE_VALUE_THRESHOLD) -> pd.DataFrame:
    """
    將低頻類別整併成 other，避免後續 One-Hot Encoding 維度過度膨脹。
    """
    if col not in df.columns:
        return df

    value_counts = df[col].value_counts(dropna=False)
    rare_values = value_counts[value_counts < min_count].index
    df[col] = df[col].replace(rare_values, "other")
    return df


# 清理規則對照：
# Rule 1  保留研究所需欄位                    -> keep_cols
# Rule 2  移除缺少原始 summary/description   -> drop empty report text
# Rule 3  清理 summary / description 文字    -> clean_text()
# Rule 4  保留有效 priority                  -> priority in PRIORITY_MAP
# Rule 5  建立 priority_label / priority_num -> map(PRIORITY_MAP / PRIORITY_NUM_MAP)
# Rule 6  補齊部分缺失值                     -> fillna("unknown")
# Rule 7  長尾類別整併                       -> replace_rare_categories()
# Rule 8  轉換 creation_time 並移除無效值    -> pd.to_datetime + dropna
# Rule 9  建立 text 並做長度控制             -> summary + description + truncate/filter
# Rule 10 依時間排序、去重後輸出             -> sort_values + drop_duplicates + to_csv
# 備註：description 來自 Eclipse Bugzilla 第一則 comment，不混入後續修復討論。


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw Bugzilla CSV data for priority prediction.")
    parser.add_argument("--input", default=RAW_PATH, help="Raw Bugzilla CSV input path.")
    parser.add_argument("--output", default=OUT_PATH, help="Cleaned CSV output path.")
    parser.add_argument("--exclude-ids-from", default=None, help="Optional CSV whose id column should be excluded from this output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Rule 1: 讀取原始資料，只保留目前研究需要的欄位。
    df = pd.read_csv(args.input)
    keep_cols = [
        "id",
        "summary",
        "description",
        "priority",
        "severity",
        "product",
        "component",
        "creator",
        "creation_time",
        "status",
        "resolution",
        "dupe_of",
        "fetch_group",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Rule 2: Eclipse description 來自第一則 comment；缺少 summary 或 description 的報告不納入訓練。
    df["summary"] = df["summary"].fillna("").astype(str)
    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)
    else:
        df["description"] = ""
    df = df[(df["summary"].str.strip() != "") & (df["description"].str.strip() != "")].copy()

    # Rule 3: 清理文字欄位，降低 URL / email / log 殘影等雜訊。
    df["summary"] = df["summary"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Rule 4: 移除 priority 缺值與非 P1~P5 的資料，確保標籤一致。
    df["priority"] = df["priority"].astype(str).str.strip()
    df = df[df["priority"].isin(PRIORITY_MAP.keys())].copy()

    # Rule 5: 建立可讀標籤 priority_label 與模型使用的 ordinal label priority_num。
    df["priority_label"] = df["priority"].map(PRIORITY_MAP)
    df["priority_num"] = df["priority"].map(PRIORITY_NUM_MAP)

    # Rule 6: 對常用結構化欄位補 unknown，避免後續特徵工程出現 NaN。
    for col in ["severity", "product", "component", "creator", "status", "resolution", "fetch_group"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    if "dupe_of" in df.columns:
        df["dupe_of"] = df["dupe_of"].fillna("").astype(str).str.strip()

    # Rule 7: 長尾類別整併，降低 creator / component / product 的稀疏程度。
    for col in ["creator", "component", "product"]:
        df = replace_rare_categories(df, col, min_count=RARE_VALUE_THRESHOLD)

    # Rule 8: 時間欄位轉型，並移除無法解析的紀錄。
    df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
    df = df.dropna(subset=["creation_time"]).copy()

    # Rule 9: 合併文字欄位，並做長度控制。
    df["text"] = (df["summary"] + " " + df["description"]).str.strip()
    df["text"] = df["text"].apply(truncate_text)
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= MIN_TEXT_LEN].copy()

    # Rule 10: 為避免後續 temporal feature 洩漏未來資訊，先依時間排序，再以 id 去重。
    df = df.sort_values("creation_time").drop_duplicates(subset=["id"]).reset_index(drop=True)

    if args.exclude_ids_from:
        exclude_ids = set(pd.read_csv(args.exclude_ids_from, usecols=["id"])["id"].astype(str))
        before_count = len(df)
        df = df[~df["id"].astype(str).isin(exclude_ids)].reset_index(drop=True)
        print(f"excluded overlapping ids: {before_count - len(df)}")

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved: {len(df)} rows -> {args.output}")
    preview_cols = [c for c in ["id", "priority", "priority_label", "priority_num", "fetch_group", "text_len"] if c in df.columns]
    print(df[preview_cols].head())


if __name__ == "__main__":
    main()
