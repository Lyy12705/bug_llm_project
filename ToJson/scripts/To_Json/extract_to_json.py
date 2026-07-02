"""
extract_to_json.py

用途：
讀取 dataset/processed/unlabeled.jsonl 中整理好的 bug_report，
依據 Code Llama 論文的 instruction-following 設定，
交給 Code Llama-Instruct 做零樣本欄位抽取，
並輸出到 dataset/predicted/predicted.jsonl。

輸出格式：
{
	  "id": "string",
	  "predicted_json": {
	    "ticket_id": "string",
	    "title": "string",
	    "description": "string",
	    "product": "string",
	    "severity": "string",
	    "bug_type": "string",
	    "component": "string",
	    "os": "string",
	    "version": "string",
	    "priority": "string",
	    "error_message": "string",
	    "steps_to_reproduce": ["string"],
	    "expected_behavior": "string",
	    "actual_behavior": "string",
	    "logs": "string",
	    "screenshots_text": "string"
	  }
	}
"""

import json
import os
import re
import argparse
import requests

INPUT = "dataset/processed/unlabeled.jsonl"
OUTPUT = "dataset/predicted/predicted.jsonl"

PAPER_REFERENCE = "Roziere et al., 2024, Code Llama: Open Foundation Models for Code"
MODEL = os.getenv("CODE_LLAMA_MODEL", "codellama:7b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Code Llama 論文強調 Instruct 版本具備 zero-shot instruction following，
# 並以 greedy decoding 回報 pass@1。資訊抽取任務需要穩定可重現輸出，
# 因此預設採用 temperature=0 的 deterministic 設定。
PROMPT_MODE = os.getenv("PROMPT_MODE", "zero_shot")  # zero_shot or few_shot
MAX_TITLE_CHARS = int(os.getenv("MAX_TITLE_CHARS", "500"))
MAX_REPORT_CHARS = int(os.getenv("MAX_REPORT_CHARS", "6000"))
NUM_CTX = int(os.getenv("CODE_LLAMA_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("CODE_LLAMA_NUM_PREDICT", "1024"))
SEED = int(os.getenv("CODE_LLAMA_SEED", "42"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
USE_OLLAMA_JSON_MODE = os.getenv("OLLAMA_JSON_MODE", "1") == "1"

SCHEMA = {
    "ticket_id": "",
    "title": "",
    "description": "",
    "product": "unknown",
    "severity": "unknown",
    "bug_type": "unknown",
    "component": "unknown",
    "os": "unknown",
    "version": "",
    "priority": "",
    "error_message": "",
    "steps_to_reproduce": [],
    "expected_behavior": "",
    "actual_behavior": "",
    "logs": "",
    "screenshots_text": "",
}

SCHEMA_KEYS = list(SCHEMA.keys())
LIST_FIELDS = {"steps_to_reproduce"}
CATEGORICAL_UNKNOWN_FIELDS = {"product", "severity", "bug_type", "component", "os"}

BUG_TYPE_RULES = """
bug_type 只能輸出以下其中一種：
- crash
- logic_error
- performance
- build_error
- compatibility
- ui_bug
- non_bug
- unknown

定義：
- crash：程式崩潰、AssertionError、RuntimeError、ModuleNotFoundError、ImportError 、exception等造成執行失敗
- logic_error：結果錯誤、setter/getter 行為錯誤、值過期、邏輯不一致
- performance：效能慢、overhead 過高、吞吐下降
- build_error：編譯失敗、build 失敗、nvcc/cmake 失敗
- compatibility：跨平台/版本/環境不相容
- ui_bug：介面問題，輸出截斷、顯示異常、可觀測性/UI 類問題
- non_bug：WIP、PR、feature request、doc fix、dependabot、重構、測試修改
- unknown：無法判斷
""".strip()

PRIORITY_RULES = """
priority 若可從原文或 issue tracker metadata 判斷，才能輸出 P1、P2、P3、P4、P5 其中之一；無法判斷時請輸出空字串 ""：
- P1 Blocker：系統無法運作/完全停擺，必須立刻修
- P2 Critical：嚴重影響功能且無合理替代方案
- P3 Major：有缺陷但有替代方案/功能缺失但未達Critical
- P4 MinoR：影響較小的錯誤/介面問題
- P5 Trivial：拼字/美觀/建議事項/非bug/很輕微問題
""".strip()

SEVERITY_RULES = """
severity 請優先輸出 issue tracker 或原文提到的嚴重程度，常見值包含：
- blocker
- critical
- major
- normal
- minor
- trivial
- enhancement
- unknown
""".strip()

FEW_SHOT_EXAMPLES = """
範例1
Title: Program crashes when loading dataset on Windows 11
Bug report: The application crashes every time I load a dataset on Windows 11. RuntimeError: DataLoader worker exited unexpectedly.
Output:
{"ticket_id":"","title":"Program crashes when loading dataset on Windows 11","description":"The application crashes every time I load a dataset on Windows 11.","product":"unknown","severity":"critical","bug_type":"crash","component":"DataLoader","os":"windows","version":"","priority":"P2","error_message":"RuntimeError: DataLoader worker exited unexpectedly.","steps_to_reproduce":["Load a dataset on Windows 11"],"expected_behavior":"","actual_behavior":"The application crashes.","logs":"RuntimeError: DataLoader worker exited unexpectedly.","screenshots_text":""}

範例2
Title: Build fails on Windows after 2026-02-27 nightly
Bug report: nvcc compiler can't compile files because they are not generated in the first place.
Output:
{"ticket_id":"","title":"Build fails on Windows after 2026-02-27 nightly","description":"nvcc compiler can't compile files because they are not generated in the first place.","product":"unknown","severity":"major","bug_type":"build_error","component":"build","os":"windows","version":"","priority":"P2","error_message":"nvcc compiler can't compile files because they are not generated in the first place.","steps_to_reproduce":[],"expected_behavior":"","actual_behavior":"Build fails on Windows nightly.","logs":"","screenshots_text":""}

範例3
Title: [WIP] tracing tests
Bug report: This is a WIP PR for tracing tests.
Output:
{"ticket_id":"","title":"[WIP] tracing tests","description":"This is a WIP PR for tracing tests.","product":"unknown","severity":"enhancement","bug_type":"non_bug","component":"unknown","os":"unknown","version":"","priority":"P5","error_message":"","steps_to_reproduce":[],"expected_behavior":"","actual_behavior":"","logs":"","screenshots_text":""}
""".strip()

FIELD_INSTRUCTION = f"""
任務：從非結構化 bug report 中抽取固定欄位，輸出單一 JSON 物件。

文獻方法對應：
- 使用 Code Llama-Instruct 的 instruction-following 能力處理自然語言與程式錯誤描述。
- 採用 zero-shot 指令式提示，讓模型直接遵循 schema。
- 保留較長 bug report 內容，以利用 Code Llama 對長上下文的支援。
- 使用 deterministic decoding，讓欄位抽取結果可重現。

輸出規則：
1. 只能輸出 JSON，不要加任何解釋、Markdown、程式碼區塊或前後文字。
2. key 必須且只能有：{", ".join(SCHEMA_KEYS)}。
3. product、severity、bug_type、component、os 若無法判斷，請填 "unknown"。
4. ticket_id、title、description、version、priority、error_message、expected_behavior、actual_behavior、logs、screenshots_text 若沒有值可填，請填 ""。
5. steps_to_reproduce 必須是 JSON array；沒有明確步驟時輸出 []。
6. {BUG_TYPE_RULES}
7. {PRIORITY_RULES}
8. {SEVERITY_RULES}
9. title 請優先複製 Title 區塊；description 請保留使用者回報的主要問題描述。
10. product 請填 issue tracker product、專案名稱或產品名稱；無法判斷就填 unknown。
11. os 請輸出作業系統或平台名稱；常見值包含 windows、linux、macos，也可保留 android、ios、freebsd、ubuntu、cuda 等明確平台。
12. component 請填受影響模組或元件名稱；無法判斷就填 unknown。
13. error_message 請填最具體的 exception/error 訊息；logs 可放較長的 log 摘要。
14. 若內容是 PR、WIP、文件修改、依賴更新、純測試調整、重構或 feature request，bug_type 請輸出 non_bug，priority 請輸出 P5。
15. 請優先根據 bug report 中明確提到的 exception、error、log、平台、版本、產品與模組名稱判斷。

輸出 schema：
{json.dumps(SCHEMA, ensure_ascii=False)}
""".strip()


def allowed_values_instruction(rec: dict) -> str:
    meta = rec.get("meta") or {}
    public_dataset = meta.get("public_dataset") or {}
    allowed_values = public_dataset.get("allowed_values") or {}
    if not isinstance(allowed_values, dict):
        return ""

    lines = []
    for field in ("product", "component", "severity", "os", "version", "priority"):
        values = allowed_values.get(field) or []
        if values:
            lines.append(f"- {field}: {', '.join(values)}")

    if not lines:
        return ""

    return (
        "候選值限制：\n"
        "若輸入中提供 issue tracker metadata 或候選清單，請優先做 metadata normalization，"
        "並讓下列欄位輸出完全符合候選值之一。\n"
        + "\n".join(lines)
    )


def build_prompt(title: str, bug_report: str, rec: dict | None = None) -> str:
    examples = ""
    if PROMPT_MODE == "few_shot":
        examples = f"\n\n參考範例：\n{FEW_SHOT_EXAMPLES}"

    candidate_instruction = ""
    if rec:
        candidate_instruction = allowed_values_instruction(rec)
        if candidate_instruction:
            candidate_instruction = f"\n\n{candidate_instruction}"

    user_instruction = f"""
{FIELD_INSTRUCTION}{candidate_instruction}{examples}

請根據以下輸入抽取欄位，並只輸出單一 JSON 物件。

Title:
{title}

Bug report:
{bug_report}
""".strip()

    return f"[INST] {user_instruction} [/INST]"


def clip_text(text: str, max_chars: int = 3000) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars]


def schema_defaults() -> dict:
    return {key: list(value) if isinstance(value, list) else value for key, value in SCHEMA.items()}


def normalize_source_fields(source_fields: dict) -> dict:
    out = schema_defaults()
    for key in out:
        if key in source_fields and key not in LIST_FIELDS:
            out[key] = str(source_fields.get(key) or "").strip()
    out["bug_type"] = canonical_bug_type(out["bug_type"])
    out["priority"] = canonical_priority(out["priority"]) if out["priority"] else ""
    out["os"] = normalize_os(out["os"])
    out["severity"] = canonical_severity(out["severity"])
    for key in CATEGORICAL_UNKNOWN_FIELDS:
        if not out[key]:
            out[key] = "unknown"
    return out


def source_defaults_from_record(rec: dict | None) -> dict:
    if not rec:
        return {}

    meta = rec.get("meta") or {}
    direct_fields = meta.get("direct_fields") or {}
    source_fields = meta.get("source_fields") or {}
    title = str(meta.get("title") or rec.get("title") or "").strip()
    bug_report = str(rec.get("bug_report") or "").strip()
    description = strip_title_prefix(bug_report, title)

    defaults = {
        "ticket_id": str(rec.get("id") or rec.get("ticket_id") or rec.get("issue_id") or "").strip(),
        "title": title,
        "description": description or bug_report,
    }

    for field in ("product", "severity", "component", "os", "version", "priority", "bug_type", "error_message"):
        value = source_fields.get(field, direct_fields.get(field, ""))
        if value:
            defaults[field] = str(value).strip()

    return {key: value for key, value in defaults.items() if value not in ("", None)}


def strip_title_prefix(text: str, title: str) -> str:
    if not text:
        return ""
    if title and text.lower().startswith(title.lower()):
        return text[len(title):].strip(" \n\t:-.。")
    return text


def normalize_steps(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = re.split(r"(?:\n+|\s*\d+\.\s+|\s*[-*]\s+)", raw)
        cleaned = [part.strip(" ;") for part in parts if part.strip(" ;")]
        return cleaned or [raw]
    return [str(value).strip()] if str(value).strip() else []


def get_with_aliases(obj: dict, key: str, default):
    aliases = {
        "ticket_id": ("id", "issue_id", "bug_id"),
        "title": ("summary", "subject"),
        "description": ("body", "content", "bug_report"),
        "product": ("project", "repository"),
        "os": ("platform", "op_sys", "environment"),
        "steps_to_reproduce": ("steps", "reproduction_steps", "steps_to_repro"),
        "expected_behavior": ("expected",),
        "actual_behavior": ("actual",),
        "logs": ("log", "stack_trace", "traceback"),
        "screenshots_text": ("screenshot_text", "image_text"),
    }
    if not isinstance(obj, dict):
        return default
    if key in obj:
        return obj[key]
    for alias in aliases.get(key, ()):
        if alias in obj:
            return obj[alias]
    return default


def canonical_bug_type(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "crash": "crash",
        "exception": "crash",
        "runtime_error": "crash",
        "assertionerror": "crash",
        "module not found": "crash",
        "logic": "logic_error",
        "logic_error": "logic_error",
        "wrong_result": "logic_error",
        "performance": "performance",
        "slow": "performance",
        "build": "build_error",
        "build_error": "build_error",
        "compile_error": "build_error",
        "compatibility": "compatibility",
        "ui": "ui_bug",
        "ui_bug": "ui_bug",
        "non_bug": "non_bug",
        "feature_request": "non_bug",
        "refactor": "non_bug",
        "unknown": "unknown",
    }
    return mapping.get(v, "unknown")


def canonical_priority(value: str) -> str:
    v = (value or "").strip().upper()
    if not v or v in {"UNKNOWN", "N/A", "NA", "NONE", "NULL", "---"}:
        return ""
    mapping = {
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "P4",
        "P5": "P5",
        "BLOCKER": "P1",
        "CRITICAL": "P2",
        "HIGH": "P2",
        "MAJOR": "P3",
        "NORMAL": "P3",
        "MEDIUM": "P3",
        "MINOR": "P4",
        "TRIVIAL": "P5",
        "LOW": "P5",
    }
    return mapping.get(v, "P3")


def canonical_severity(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "blocker": "blocker",
        "critical": "critical",
        "major": "major",
        "normal": "normal",
        "minor": "minor",
        "trivial": "trivial",
        "enhancement": "enhancement",
        "n/a": "unknown",
        "na": "unknown",
        "---": "unknown",
        "unknown": "unknown",
    }
    return mapping.get(v, v or "unknown")


def normalize_os(value: str) -> str:
    raw = (value or "").strip()
    v = raw.lower()
    if not v:
        return "unknown"
    if v in {"windows", "win", "win10", "win11"}:
        return "windows"
    if v in {"linux"}:
        return "linux"
    if v in {"mac", "macos", "osx", "mac os"}:
        return "macos"
    # 其他較具體平台名稱直接保留，例如 ubuntu、debian、android、ios、freebsd
    return raw


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "repeat_penalty": 1.0,
        },
    }
    if USE_OLLAMA_JSON_MODE:
        payload["format"] = "json"

    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return schema_defaults()


def canonical_allowed_value(value: str, allowed_values: list[str], default: str) -> str:
    raw = (value or "").strip()
    if not allowed_values:
        return raw or default

    by_lower = {str(item).strip().lower(): str(item).strip() for item in allowed_values}
    if raw.lower() in by_lower:
        return by_lower[raw.lower()]

    compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
    for item in allowed_values:
        item_compact = re.sub(r"[^a-z0-9]+", "", str(item).lower())
        if compact and compact == item_compact:
            return str(item).strip()

    return raw or default


def parse_tracker_metadata_value(bug_report: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*([^:]+?)(?:\s+[A-Z][A-Za-z ]{{1,30}}:| Component candidates:| User report title:| User report description:|$)"
    match = re.search(pattern, bug_report or "", flags=re.I)
    return match.group(1).strip() if match else ""


def public_eclipse_component_from_text(text: str, allowed_components: list[str]) -> str:
    lower = (text or "").lower()

    rules = [
        ("terminal", ["terminal", "terminals"]),
        ("cdt-lsp", [" lsp", "presentationreconcilercpp"]),
        ("cdt-parser", ["parser", "parsing", "lambda move capture"]),
        ("cdt-releng", ["releng", "deployable feature", "shared libraries"]),
        ("cdt-other", ["eclipse-rpm", "oprofile", "[rpm]", " rpm"]),
        ("cdt-refactoring", ["change function signature"]),
        (
            "cdt-build",
            [
                "build",
                "builder",
                "managedbuilder",
                "managed build",
                "managed make",
                "makefile",
                "subdir.mk",
                "scanner discovery",
                "toolchain",
                "compiler",
                "linker",
                "gmake",
                "make target",
            ],
        ),
    ]

    allowed = {item.lower(): item for item in allowed_components}
    for component, keywords in rules:
        if component in allowed and any(keyword in lower for keyword in keywords):
            return allowed[component]
    return ""


def postprocess_public_eclipse(pred: dict, rec: dict) -> dict:
    meta = rec.get("meta") or {}
    public_dataset = meta.get("public_dataset") or {}
    allowed_values = public_dataset.get("allowed_values") or {}
    if not isinstance(allowed_values, dict):
        return pred

    out = dict(pred)
    text = f"{(meta.get('title') or '')} {rec.get('bug_report') or ''}"

    for field in ("product", "component", "severity", "os", "version", "priority"):
        candidates = allowed_values.get(field) or []
        if candidates:
            out[field] = canonical_allowed_value(out.get(field, ""), candidates, out.get(field, ""))

    component_candidates = allowed_values.get("component") or []
    component_guess = public_eclipse_component_from_text(text, component_candidates)
    if component_guess and (not out.get("component") or out.get("component") == "unknown" or out.get("component") not in component_candidates):
        out["component"] = component_guess

    metadata_aliases = {
        "product": "Product",
        "severity": "Severity",
        "os": "Operating system",
        "version": "Version",
        "priority": "Priority",
    }
    for field, label in metadata_aliases.items():
        metadata_value = parse_tracker_metadata_value(rec.get("bug_report", ""), label)
        if metadata_value:
            candidates = allowed_values.get(field) or []
            if field == "os":
                metadata_value = normalize_os(metadata_value)
            if field == "priority":
                metadata_value = canonical_priority(metadata_value)
            if field == "severity":
                metadata_value = canonical_severity(metadata_value)
            out[field] = canonical_allowed_value(metadata_value, candidates, metadata_value)

    return out


def normalize_prediction(obj: dict, rec: dict | None = None, postprocess: str = "none") -> dict:
    out = schema_defaults()
    source_defaults = source_defaults_from_record(rec)

    for key in out:
        value = get_with_aliases(obj, key, source_defaults.get(key, out[key]))
        if value is None:
            value = source_defaults.get(key, out[key])
        if key in LIST_FIELDS:
            out[key] = normalize_steps(value)
        else:
            if not isinstance(value, str):
                value = str(value)
            out[key] = value.strip()

    # Source identifiers and original text fields are more reliable than model-generated copies.
    for key in ("ticket_id", "title"):
        if source_defaults.get(key):
            out[key] = source_defaults[key]
    if source_defaults.get("description") and not out["description"]:
        out["description"] = source_defaults["description"]
    for key in ("product", "severity"):
        if source_defaults.get(key) and out.get(key, "").lower() in {"", "unknown"}:
            out[key] = source_defaults[key]

    out["bug_type"] = canonical_bug_type(out["bug_type"])
    out["priority"] = canonical_priority(out["priority"])
    out["os"] = normalize_os(out["os"])
    out["severity"] = canonical_severity(out["severity"])

    for key in CATEGORICAL_UNKNOWN_FIELDS:
        if not out[key]:
            out[key] = "unknown"

    if rec is not None and postprocess == "public_eclipse":
        out = postprocess_public_eclipse(out, rec)

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract normalized JSON fields from bug_report records with Code Llama-Instruct."
    )
    parser.add_argument("--input", default=INPUT, help="Input JSONL with bug_report records.")
    parser.add_argument("--output", default=OUTPUT, help="Output JSONL for model predictions.")
    parser.add_argument(
        "--postprocess",
        choices=["none", "public_eclipse"],
        default="none",
        help="Optional deterministic post-processing for public benchmark normalization.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to process. Use 0 for all records.")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip ids already written.")
    return parser.parse_args()


def load_existing_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_id = rec.get("id")
            if rec_id:
                ids.add(str(rec_id))
    return ids


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]

    if args.limit:
        records = records[:args.limit]

    existing_ids = load_existing_ids(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"

    with open(args.output, mode, encoding="utf-8") as fout:
        for index, rec in enumerate(records, 1):
            rec_id = str(rec.get("id", ""))
            if rec_id in existing_ids:
                print(f"[{index}/{len(records)}] skip existing {rec_id}", flush=True)
                continue

            title = clip_text((rec.get("meta") or {}).get("title", ""), MAX_TITLE_CHARS)
            bug_report = clip_text(rec.get("bug_report", ""), MAX_REPORT_CHARS)

            prompt = build_prompt(title=title, bug_report=bug_report, rec=rec)
            raw_output = call_ollama(prompt)
            pred = normalize_prediction(extract_json(raw_output), rec=rec, postprocess=args.postprocess)

            result = {
                "id": rec_id,
                "predicted_json": pred,
                "extraction_meta": {
                    "reference": PAPER_REFERENCE,
                    "model": MODEL,
                    "prompt_style": "Code Llama-Instruct [INST] instruction",
                    "prompt_mode": PROMPT_MODE,
                    "decoding": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "seed": SEED,
                        "num_ctx": NUM_CTX,
                        "num_predict": NUM_PREDICT,
                    },
                    "max_title_chars": MAX_TITLE_CHARS,
                    "max_report_chars": MAX_REPORT_CHARS,
                    "postprocess": args.postprocess,
                },
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{index}/{len(records)}] wrote {rec_id}", flush=True)

    print(f"Wrote predictions -> {args.output}")


if __name__ == "__main__":
    main()
