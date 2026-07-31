from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DECISION_FIELDS = (
    "ticket_id",
    "expected_assignee",
    "predicted_assignee",
    "ranked_candidates",
    "known_owner",
    "is_top1_correct",
    "is_top3_correct",
    "routing_status",
    "fallback_reason",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a registered rolling evaluation replay with its reference decisions."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-numeric-delta", type=float, default=0.0002)
    args = parser.parse_args()

    report = compare_predictions(
        read_jsonl(args.reference),
        read_jsonl(args.replay),
        maximum_numeric_delta=max(0.0, args.maximum_numeric_delta),
    )
    report["reference"] = {
        "path": str(args.reference.resolve()),
        "sha256": sha256_file(args.reference),
    }
    report["replay"] = {
        "path": str(args.replay.resolve()),
        "sha256": sha256_file(args.replay),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def compare_predictions(
    reference: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    *,
    maximum_numeric_delta: float,
) -> dict[str, Any]:
    row_count_matches = len(reference) == len(replay)
    decision_mismatches = []
    maximum_delta_by_field: dict[str, float] = {}
    compared_rows = min(len(reference), len(replay))
    for index, (left, right) in enumerate(zip(reference, replay)):
        differing_decisions = [
            field for field in DECISION_FIELDS if left.get(field) != right.get(field)
        ]
        if differing_decisions:
            decision_mismatches.append(
                {
                    "row": index,
                    "reference_ticket_id": left.get("ticket_id"),
                    "replay_ticket_id": right.get("ticket_id"),
                    "fields": differing_decisions,
                }
            )
        for field in set(left) & set(right):
            left_value = left[field]
            right_value = right[field]
            if (
                isinstance(left_value, (int, float))
                and not isinstance(left_value, bool)
                and isinstance(right_value, (int, float))
                and not isinstance(right_value, bool)
            ):
                delta = abs(float(left_value) - float(right_value))
                maximum_delta_by_field[field] = max(
                    maximum_delta_by_field.get(field, 0.0), delta
                )
    maximum_observed_delta = max(maximum_delta_by_field.values(), default=0.0)
    decisions_match = row_count_matches and not decision_mismatches
    numeric_tolerance_passed = maximum_observed_delta <= maximum_numeric_delta
    return {
        "schema_version": 1,
        "method": "rolling_routing_registered_replay_comparison_v1",
        "rows_reference": len(reference),
        "rows_replay": len(replay),
        "rows_compared": compared_rows,
        "row_count_matches": row_count_matches,
        "decision_fields": list(DECISION_FIELDS),
        "reference_decision_sha256": decision_digest(reference),
        "replay_decision_sha256": decision_digest(replay),
        "decision_mismatch_rows": len(decision_mismatches),
        "decision_mismatch_examples": decision_mismatches[:20],
        "maximum_numeric_delta": maximum_observed_delta,
        "maximum_numeric_delta_by_field": dict(sorted(maximum_delta_by_field.items())),
        "numeric_tolerance": maximum_numeric_delta,
        "checks": {
            "row_count_matches": row_count_matches,
            "routing_decisions_exactly_match": decisions_match,
            "numeric_deltas_within_tolerance": numeric_tolerance_passed,
        },
        "passed": decisions_match and numeric_tolerance_passed,
        "release_evidence": False,
    }


def decision_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        projected = {field: row.get(field) for field in DECISION_FIELDS}
        digest.update(
            json.dumps(
                projected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
