from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PHASE1_ROOT,
    DEFAULT_RAW_PATH,
    build_metrics,
    clean,
    evaluate_current_triager,
    parse_date,
    read_jsonl,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
    prediction_csv_fields,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit assignment-time leakage and evaluate safer views.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-train-assignee-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE1_ROOT / "reports" / args.dataset / "assignment_time_leakage"
    data_dir = output_dir / "data"
    train_path = args.data_dir / f"{args.dataset}_history_train.jsonl"
    validation_path = args.data_dir / f"{args.dataset}_validation_set.jsonl"
    test_path = args.data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.data_dir / f"{args.dataset}_candidate_roster.json"

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    test_rows = read_jsonl(test_path)
    raw_by_ticket = {f"bmo_{row.get('id')}": row for row in read_jsonl(args.raw)}
    roster = read_roster(roster_path)
    test_start = min(parse_date(row["created_at"]) for row in test_rows if parse_date(row.get("created_at")))
    test_end = max(parse_date(row["created_at"]) for row in test_rows if parse_date(row.get("created_at")))
    if test_start is None or test_end is None:
        raise SystemExit("Test rows do not contain valid creation timestamps.")

    current_predictions, current_metrics = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=test_rows,
        roster=set(roster),
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    audit_rows = build_assignment_audit_rows(
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        raw_by_ticket=raw_by_ticket,
        test_start_iso=test_start.isoformat(),
        test_end_iso=test_end.isoformat(),
    )

    conservative_train = [
        row
        for row in train_rows
        if last_change_at_or_before(raw_by_ticket.get(row["ticket_id"], {}), test_start)
    ]
    conservative_counts = Counter(row["assignee"] for row in conservative_train)
    conservative_roster = {
        assignee for assignee, count in conservative_counts.items() if count >= args.min_train_assignee_count
    }
    conservative_test = [row for row in test_rows if row.get("assignee") in conservative_roster]
    conservative_validation = [row for row in validation_rows if row.get("assignee") in conservative_roster]

    conservative_train_path = data_dir / "conservative_history_train.jsonl"
    conservative_test_path = data_dir / "conservative_test_set.jsonl"
    write_jsonl(conservative_train_path, conservative_train)
    write_jsonl(data_dir / "conservative_validation_set.jsonl", conservative_validation)
    write_jsonl(conservative_test_path, conservative_test)
    write_json(data_dir / "conservative_candidate_roster.json", {"candidates": sorted(conservative_roster)})

    conservative_predictions, conservative_metrics = evaluate_current_triager(
        train_path=conservative_train_path,
        train_rows=conservative_train,
        test_rows=conservative_test,
        roster=conservative_roster,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    metrics = {
        "experiment_name": "assignment_time_leakage_audit",
        "dataset": args.dataset,
        "raw_path": str(args.raw),
        "assignment_history_available": False,
        "creation_time_assignment_semantics_fully_verifiable": False,
        "limitation": "creation-time assignment semantics not fully verifiable from current Bugzilla snapshot fields",
        "test_start_cutoff": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "views": {
            "current_snapshot": {
                "description": "Retrospective snapshot evaluation matching the existing closed-set benchmark.",
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "test_count": len(test_rows),
                "roster_size": len(roster),
                "excluded_count": 0,
                "exclusion_reason": "",
                "metrics": current_metrics,
            },
            "conservative": {
                "description": "History restricted to train tickets whose last_change_time is not after the test-start cutoff.",
                "train_count": len(conservative_train),
                "validation_count": len(conservative_validation),
                "test_count": len(conservative_test),
                "roster_size": len(conservative_roster),
                "excluded_count": len(test_rows) - len(conservative_test),
                "exclusion_reason": "expected assignee absent from conservative train-frequency roster",
                "metrics": conservative_metrics,
            },
            "strict": {
                "description": "Creation-time label-safe evaluation cannot be computed without assignment history.",
                "train_count": 0,
                "validation_count": 0,
                "test_count": 0,
                "roster_size": 0,
                "excluded_count": len(test_rows),
                "exclusion_reason": "assignment history unavailable; creation-time assignee cannot be verified",
                "metrics": None,
            },
        },
        "audit_counts": {
            "train_rows_with_last_change_after_test_start": sum(
                1 for row in train_rows if last_change_after(raw_by_ticket.get(row["ticket_id"], {}), test_start)
            ),
            "train_rows_with_last_change_after_test_end": sum(
                1 for row in train_rows if last_change_after(raw_by_ticket.get(row["ticket_id"], {}), test_end)
            ),
            "raw_rows_with_dupe_or_last_change_fields": sum(1 for row in raw_by_ticket.values() if "last_change_time" in row),
        },
        "notes": [
            "Bugzilla REST assigned_to is fetched as a snapshot field by the current fetch script; no assignment history is stored.",
            "Conservative view reduces future-state risk but still does not prove creation-time assignment semantics.",
            "Strict view intentionally reports metrics as null instead of pretending to be leakage-free.",
        ],
    }

    write_json(output_dir / "assignment_safe_metrics.json", metrics)
    write_csv(output_dir / "assignment_time_leakage_audit.csv", audit_rows, audit_fields())
    write_jsonl(output_dir / "current_snapshot_predictions.jsonl", current_predictions)
    write_jsonl(output_dir / "conservative_predictions.jsonl", conservative_predictions)
    write_csv(
        output_dir / "current_snapshot_predictions.csv",
        rows_to_prediction_csv(current_predictions),
        prediction_csv_fields(),
    )
    write_csv(
        output_dir / "conservative_predictions.csv",
        rows_to_prediction_csv(conservative_predictions),
        prediction_csv_fields(),
    )
    write_summary(output_dir / "assignment_safe_summary.md", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def read_roster(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted(str(candidate).strip().lower() for candidate in candidates if str(candidate).strip())


def build_assignment_audit_rows(
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    raw_by_ticket: dict[str, dict[str, Any]],
    test_start_iso: str,
    test_end_iso: str,
) -> list[dict[str, Any]]:
    test_start = parse_date(test_start_iso)
    test_end = parse_date(test_end_iso)
    output = []
    for split, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        for row in rows:
            raw = raw_by_ticket.get(row["ticket_id"], {})
            last_change = clean(raw.get("last_change_time"))
            output.append(
                {
                    "split": split,
                    "ticket_id": row.get("ticket_id"),
                    "created_at": row.get("created_at"),
                    "last_change_time": last_change,
                    "status": row.get("status"),
                    "resolution": row.get("resolution"),
                    "assigned_to_snapshot_present": bool(raw.get("assigned_to")),
                    "dupe_of": clean(raw.get("dupe_of")),
                    "last_change_after_test_start": last_change_after(raw, test_start),
                    "last_change_after_test_end": last_change_after(raw, test_end),
                    "conservative_history_eligible": split == "train" and last_change_at_or_before(raw, test_start),
                    "timing_semantics": "snapshot_assigned_to_not_creation_time_verified",
                }
            )
    return output


def last_change_after(raw: dict[str, Any], cutoff: datetime | None) -> bool:
    value = parse_date(raw.get("last_change_time"))
    return bool(value and cutoff and value > cutoff)


def last_change_at_or_before(raw: dict[str, Any], cutoff: datetime | None) -> bool:
    value = parse_date(raw.get("last_change_time"))
    return bool(value and cutoff and value <= cutoff)


def audit_fields() -> list[str]:
    return [
        "split",
        "ticket_id",
        "created_at",
        "last_change_time",
        "status",
        "resolution",
        "assigned_to_snapshot_present",
        "dupe_of",
        "last_change_after_test_start",
        "last_change_after_test_end",
        "conservative_history_eligible",
        "timing_semantics",
    ]


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Assignment-Time Leakage Audit",
        "",
        f"- Assignment history available: `{metrics['assignment_history_available']}`",
        f"- Creation-time assignment semantics fully verifiable: `{metrics['creation_time_assignment_semantics_fully_verifiable']}`",
        f"- Limitation: {metrics['limitation']}",
        f"- Test-start cutoff: `{metrics['test_start_cutoff']}`",
        "",
        "| View | Train | Validation | Test | Roster | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in metrics["views"].items():
        row = view["metrics"] or build_metrics([])
        lines.append(
            f"| {name} | {view['train_count']} | {view['validation_count']} | {view['test_count']} | "
            f"{view['roster_size']} | {row['top1_accuracy']:.6f} | {row['hit_at_3']:.6f} | "
            f"{row['hit_at_5']:.6f} | {row['hit_at_10']:.6f} | {row['mrr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Strict view metrics are null in JSON because the current dataset does not contain assignment history.",
            "The zero-valued markdown row is a display placeholder, not a valid strict score.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
