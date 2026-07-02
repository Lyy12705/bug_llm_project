"""
evaluate_results.py

用途：
評估 Code Llama 從 bug_report 抽取 JSON 欄位的結果。

本程式會做兩種比較：

1. predicted_json vs json_ground_truth
   - 正式準確率評估
   - ground truth 來自人工標註

2. predicted_json vs source_fields
   - 與來源資料原生欄位做一致性比較
   - source_fields 來自 GitHub / Jira 可直接取得或初步偵測的欄位

輸出內容：
- matched records 數量
- 每個欄位的 accuracy
- overall field accuracy
- exact match accuracy
- 前幾筆 mismatch 範例
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

GOLD_PATH = "dataset/labeled/test.jsonl"
BACKUP_GOLD_PATH = "dataset/labeled/test_backup.jsonl"
PRED_PATH = "dataset/predicted/predicted.jsonl"

SCHEMA_FIELDS = [
    "ticket_id",
    "title",
    "description",
    "product",
    "severity",
    "bug_type",
    "component",
    "os",
    "version",
    "priority",
    "error_message",
    "steps_to_reproduce",
    "expected_behavior",
    "actual_behavior",
    "logs",
    "screenshots_text",
]
DEFAULT_FIELDS = ["bug_type", "component", "os", "version", "priority", "error_message"]
FIELDS = list(DEFAULT_FIELDS)


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON at {path}:{line_num} -> {e}")
    return data


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        value = json.dumps([str(item).strip() for item in value], ensure_ascii=False)
    if not isinstance(value, str):
        value = str(value)
    return value.strip().lower()


def build_gold_map(records, observable_only=False):
    gold_map = {}
    for rec in records:
        rec_id = rec.get("id")
        gt = rec.get("json_ground_truth") or {}
        if rec_id is None:
            continue
        if observable_only:
            observable_fields = (
                ((rec.get("meta") or {}).get("public_dataset") or {}).get("observable_ground_truth_fields")
                or []
            )
            gt = {field: value for field, value in gt.items() if field in set(observable_fields)}
        if not has_any_reference_value(gt):
            continue
        gold_map[rec_id] = gt
    return gold_map


def build_source_map(records):
    source_map = {}
    for rec in records:
        rec_id = rec.get("id")
        source_fields = (((rec.get("meta") or {}).get("direct_fields")) or {})
        if rec_id is None:
            continue
        source_map[rec_id] = source_fields
    return source_map


def build_pred_map(records):
    pred_map = {}
    for rec in records:
        rec_id = rec.get("id")
        pred = rec.get("predicted_json") or {}
        if rec_id is None:
            continue
        pred_map[rec_id] = pred
    return pred_map


def has_any_reference_value(reference):
    if not isinstance(reference, dict):
        return False
    return any(normalize_text(reference.get(field, "")) for field in FIELDS)


def count_annotated_records(records):
    return sum(1 for rec in records if has_any_reference_value(rec.get("json_ground_truth") or {}))


def print_dataset_status(gold_path, gold_records, pred_path, pred_records):
    print("=== Dataset Status ===")
    print(f"Ground truth file        : {gold_path}")
    print(f"Ground truth records     : {len(gold_records)}")
    print(f"Annotated JSON records   : {count_annotated_records(gold_records)}")
    print(f"Prediction file          : {pred_path}")
    print(f"Prediction records       : {len(pred_records)}")
    print(f"Evaluation fields        : {', '.join(FIELDS)}")
    print()


def evaluate_pair(
    reference_map,
    pred_map,
    label="Evaluation",
    skip_missing_reference=False,
    skip_empty_reference=False,
    no_reference_message=None,
):
    common_ids = sorted(set(reference_map.keys()) & set(pred_map.keys()))
    missing_pred = sorted(set(reference_map.keys()) - set(pred_map.keys()))
    extra_pred = sorted(set(pred_map.keys()) - set(reference_map.keys()))

    print(f"=== {label} ===")
    print(f"Reference records : {len(reference_map)}")
    print(f"Predicted records : {len(pred_map)}")
    print(f"Matched records   : {len(common_ids)}")
    print(f"Missing predicted : {len(missing_pred)}")
    print(f"Extra predicted   : {len(extra_pred)}")
    print()

    if len(reference_map) == 0:
        if no_reference_message:
            print(no_reference_message)
        else:
            print("No reference records found.")
        print()
        return

    if len(common_ids) == 0:
        print("No matched records found.")
        print()
        return

    field_correct = defaultdict(int)
    field_total = defaultdict(int)

    exact_match_count = 0
    exact_match_total = 0
    total_fields = 0
    correct_fields = 0
    mismatches = []

    for rec_id in common_ids:
        ref = reference_map[rec_id]
        pred = pred_map[rec_id]

        if not isinstance(ref, dict):
            ref = {}
        if not isinstance(pred, dict):
            pred = {}

        record_all_correct = True
        record_diff = {}
        compared_fields = 0

        for field in FIELDS:
            if skip_missing_reference and field not in ref:
                continue

            ref_val = normalize_text(ref.get(field, ""))
            pred_val = normalize_text(pred.get(field, ""))

            if skip_empty_reference and not ref_val:
                continue

            field_total[field] += 1
            total_fields += 1
            compared_fields += 1

            if ref_val == pred_val:
                field_correct[field] += 1
                correct_fields += 1
            else:
                record_all_correct = False
                record_diff[field] = {
                    "reference": ref.get(field, ""),
                    "pred": pred.get(field, "")
                }

        if compared_fields == 0:
            continue

        exact_match_total += 1
        if record_all_correct:
            exact_match_count += 1
        elif record_diff:
            mismatches.append({
                "id": rec_id,
                "diff": record_diff
            })

    if total_fields == 0:
        print("No comparable fields found.")
        print()
        return

    metric_name = "Field Consistency" if skip_empty_reference else "Field Accuracy"
    print(metric_name)
    for field in FIELDS:
        if field_total[field]:
            acc = field_correct[field] / field_total[field]
            print(f"{field:14s}: {field_correct[field]:3d}/{field_total[field]:3d} = {acc:.4f}")
        else:
            print(f"{field:14s}: N/A")

    overall_field_acc = correct_fields / total_fields if total_fields else 0
    exact_match_acc = exact_match_count / exact_match_total if exact_match_total else 0

    print()
    print("Overall Metrics")
    if skip_empty_reference:
        print(f"Overall field consistency : {correct_fields}/{total_fields} = {overall_field_acc:.4f}")
        print(f"Exact available-field match: {exact_match_count}/{exact_match_total} = {exact_match_acc:.4f}")
    else:
        print(f"Overall field accuracy : {correct_fields}/{total_fields} = {overall_field_acc:.4f}")
        print(f"Exact match accuracy   : {exact_match_count}/{exact_match_total} = {exact_match_acc:.4f}")

    print()
    print("Sample Mismatches (up to 10)")
    for item in mismatches[:10]:
        print(f"ID: {item['id']}")
        for field, diff in item["diff"].items():
            print(f"  - {field}")
            print(f"      reference: {diff['reference']}")
            print(f"      pred     : {diff['pred']}")
        print()

    print("-" * 60)
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Code Llama bug-report-to-JSON predictions."
    )
    parser.add_argument("--gold", default=GOLD_PATH, help="JSONL file with json_ground_truth.")
    parser.add_argument("--pred", default=PRED_PATH, help="JSONL file with predicted_json.")
    parser.add_argument(
        "--backup-gold",
        default=BACKUP_GOLD_PATH,
        help="Fallback labeled JSONL used when --gold has no filled json_ground_truth.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not fall back to dataset/labeled/test_backup.jsonl.",
    )
    parser.add_argument(
        "--skip-missing-ground-truth",
        action="store_true",
        help="For public datasets that label only some fields, do not evaluate fields missing from json_ground_truth.",
    )
    parser.add_argument(
        "--fields",
        default="",
        help=(
            "Comma-separated fields to evaluate. "
            f"Allowed fields: {', '.join(SCHEMA_FIELDS)}. "
            f"Default fields: {', '.join(DEFAULT_FIELDS)}."
        ),
    )
    parser.add_argument(
        "--no-source-consistency",
        action="store_true",
        help="Only print ground-truth accuracy; skip source field consistency output.",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Only evaluate ground-truth fields whose value appears in the model input bug_report.",
    )
    return parser.parse_args()


def parse_fields(value):
    if not value:
        return list(DEFAULT_FIELDS)

    selected = [field.strip() for field in value.split(",") if field.strip()]
    invalid = [field for field in selected if field not in SCHEMA_FIELDS]
    if invalid:
        allowed = ", ".join(SCHEMA_FIELDS)
        raise SystemExit(f"Invalid --fields value: {', '.join(invalid)}. Allowed fields: {allowed}")
    if not selected:
        raise SystemExit("--fields must include at least one field.")
    return selected


def main():
    global FIELDS
    args = parse_args()
    FIELDS = parse_fields(args.fields)

    gold_path = args.gold
    gold_records = load_jsonl(gold_path)
    pred_records = load_jsonl(args.pred)

    if count_annotated_records(gold_records) == 0 and not args.no_backup:
        backup_path = Path(args.backup_gold)
        if backup_path.exists():
            backup_records = load_jsonl(str(backup_path))
            if count_annotated_records(backup_records) > 0:
                print(
                    f"[INFO] {gold_path} has no filled json_ground_truth; "
                    f"using {backup_path} instead."
                )
                print()
                gold_path = str(backup_path)
                gold_records = backup_records

    print_dataset_status(gold_path, gold_records, args.pred, pred_records)

    gold_map = build_gold_map(gold_records, observable_only=args.observable_only)
    source_map = build_source_map(gold_records)
    pred_map = build_pred_map(pred_records)

    evaluate_pair(
        gold_map,
        pred_map,
        label="Ground Truth Evaluation",
        skip_missing_reference=args.skip_missing_ground_truth,
        no_reference_message=(
            "有效準確率：N/A\n"
            "原因：json_ground_truth 尚未填寫。Code Llama 論文沒有提供 "
            "bug report -> JSON 欄位抽取資料集；HumanEval/MBPP 是程式生成 benchmark。"
        ),
    )
    if not args.no_source_consistency:
        evaluate_pair(
            source_map,
            pred_map,
            label="Source Field Consistency Evaluation (not model accuracy)",
            skip_empty_reference=True,
            no_reference_message="No source fields found.",
        )


if __name__ == "__main__":
    main()
