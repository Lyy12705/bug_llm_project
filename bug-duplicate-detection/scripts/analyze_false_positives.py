#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


METADATA_FIELDS = ("product", "component", "severity", "priority")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze false-positive duplicate decisions from tune-threshold decisions CSV.")
    parser.add_argument("--decisions-csv", required=True, help="CSV produced by tune-threshold --output-decisions-csv")
    parser.add_argument("--output-csv", default="reports/false_positive_decisions.csv")
    parser.add_argument("--summary-md", default="reports/false_positive_summary.md")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = read_decisions_csv(args.decisions_csv)
    false_positives = [row for row in rows if row.get("outcome") == "false_positive"]
    write_rows(false_positives, args.output_csv)
    summary = render_summary(rows, false_positives, top=args.top)
    output = Path(args.summary_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"output_csv={args.output_csv}")
    print(f"summary_md={args.summary_md}")
    return 0


def read_decisions_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "query_id",
        "actual_is_duplicate",
        "predicted_is_duplicate",
        "outcome",
        "best_candidate_id",
        "score",
        "threshold",
        "query_title",
        "best_title",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_summary(rows: list[dict[str, str]], false_positives: list[dict[str, str]], *, top: int) -> str:
    total = len(rows)
    fp_count = len(false_positives)
    threshold = first_present(false_positives, "threshold") or first_present(rows, "threshold") or "-"
    lines = [
        "# False Positive Analysis",
        "",
        f"- decisions={total}",
        f"- false_positives={fp_count}",
        f"- false_positive_rate={ratio(fp_count, non_duplicate_count(rows)):.4f}",
        f"- threshold={threshold}",
        "",
        "## Metadata Match Rates",
        "",
        "| Field | Match | Mismatch | Match rate |",
        "|---|---:|---:|---:|",
    ]
    for field in METADATA_FIELDS:
        matches = sum(1 for row in false_positives if row.get(f"{field}_match") == "true")
        mismatches = fp_count - matches
        lines.append(f"| {field} | {matches} | {mismatches} | {ratio(matches, fp_count):.4f} |")

    lines.extend(["", "## Common False Positive Buckets", ""])
    for title, key in (
        ("Query components", "query_component"),
        ("Candidate components", "best_component"),
        ("Component pairs", ("query_component", "best_component")),
        ("Query resolutions", "query_resolution"),
        ("Severity pairs", ("query_severity", "best_severity")),
        ("Priority pairs", ("query_priority", "best_priority")),
    ):
        lines.extend(render_counter_section(title, false_positives, key, top=top))

    lines.extend(["", "## Highest-score False Positives", "", "| Query | Candidate | Score | Query title | Candidate title |", "|---|---|---:|---|---|"])
    sorted_examples = sorted(false_positives, key=lambda row: float_or_zero(row.get("score")), reverse=True)
    for row in sorted_examples[:top]:
        lines.append(
            f"| {row.get('query_id', '')} | {row.get('best_candidate_id', '')} | {row.get('score', '')} | "
            f"{escape_cell(row.get('query_title', ''))} | {escape_cell(row.get('best_title', ''))} |"
        )

    return "\n".join(lines)


def render_counter_section(title: str, rows: list[dict[str, str]], key, *, top: int) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if isinstance(key, tuple):
            value = " -> ".join((row.get(part, "") or "-") for part in key)
        else:
            value = row.get(key, "") or "-"
        counter[value] += 1
    lines = [f"### {title}", "", "| Value | Count |", "|---|---:|"]
    for value, count in counter.most_common(top):
        lines.append(f"| {escape_cell(value)} | {count} |")
    lines.append("")
    return lines


def non_duplicate_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("actual_is_duplicate") == "false")


def first_present(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key, "")
        if value:
            return value
    return ""


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def float_or_zero(value: str | None) -> float:
    if not value:
        return 0.0
    return float(value)


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
