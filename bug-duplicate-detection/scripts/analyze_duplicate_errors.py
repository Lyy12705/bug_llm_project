#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector
from duplicate_ticket_detection.splits import train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Top-1 duplicate ranking errors for manual error analysis.")
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--model-dir", help="Saved SBERT model directory. Required for --method sbert")
    parser.add_argument("--combine", default="mean")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", default="reports/top1_error_analysis.csv")
    parser.add_argument("--summary-md", default="reports/top1_error_summary.md")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    detector = build_detector(args, split.train)
    rows = collect_top1_errors(detector, split.test, split.train, top_k=args.top_k)
    write_errors_csv(rows, args.output_csv)
    summary = render_summary(rows)
    summary_output = Path(args.summary_md)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(summary + "\n", encoding="utf-8")

    evaluable = count_evaluable_queries(split.test, split.train)
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    print(f"evaluable_queries={evaluable}")
    print(f"top1_errors={len(rows)}")
    print(f"top1_error_rate={0.0 if evaluable == 0 else len(rows) / evaluable:.4f}")
    print(f"output_csv={args.output_csv}")
    print(f"summary_md={args.summary_md}")
    return 0


def build_detector(args: argparse.Namespace, train: list[TicketRecord]):
    if args.method == "tfidf":
        return TfidfDuplicateDetector(combine=args.combine).fit(train)
    if not args.model_dir:
        raise SystemExit("--model-dir is required when --method sbert")
    return SbertDuplicateDetector.load(args.model_dir, combine=args.combine)


def collect_top1_errors(detector, queries: list[TicketRecord], candidates: list[TicketRecord], *, top_k: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    top_k = max(top_k, 1)
    candidate_by_id = {ticket.ticket_id: ticket for ticket in candidates}
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidates)
        if not relevant:
            continue
        ranking = detector.rank(query, candidates, top_k=top_k)
        best = ranking[0] if ranking else None
        best_ticket = None if best is None else candidate_by_id.get(best.ticket_id)
        if best and best.ticket_id in relevant:
            continue
        rows.append(
            {
                "query_id": query.ticket_id,
                "query_title": query.title,
                "query_product": field_value(query, "product"),
                "query_component": field_value(query, "component"),
                "query_severity": field_value(query, "severity"),
                "query_priority": field_value(query, "priority"),
                "query_category": ticket_category(query),
                "relevant_ids": " ".join(sorted(relevant)),
                "best_candidate_id": "" if best is None else best.ticket_id,
                "best_score": "" if best is None else f"{best.score:.6f}",
                "best_title_score": "" if best is None else f"{best.title_score:.6f}",
                "best_content_score": "" if best is None else f"{best.content_score:.6f}",
                "best_title": "" if best is None else best.title,
                "best_product": "" if best_ticket is None else field_value(best_ticket, "product"),
                "best_component": "" if best_ticket is None else field_value(best_ticket, "component"),
                "best_severity": "" if best_ticket is None else field_value(best_ticket, "severity"),
                "best_priority": "" if best_ticket is None else field_value(best_ticket, "priority"),
                "component_match": str(same_field(query, best_ticket, "component")).lower(),
                "product_match": str(same_field(query, best_ticket, "product")).lower(),
                "top_ranked_ids": " ".join(row.ticket_id for row in ranking),
            }
        )
    return rows


def write_errors_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_title",
        "query_product",
        "query_component",
        "query_severity",
        "query_priority",
        "query_category",
        "relevant_ids",
        "best_candidate_id",
        "best_score",
        "best_title_score",
        "best_content_score",
        "best_title",
        "best_product",
        "best_component",
        "best_severity",
        "best_priority",
        "component_match",
        "product_match",
        "top_ranked_ids",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_evaluable_queries(test: list[TicketRecord], train: list[TicketRecord]) -> int:
    return sum(1 for query in test if relevant_duplicate_ids(query, train))


def render_summary(rows: list[dict[str, str]], *, top: int = 10) -> str:
    lines = [
        "# Top-1 Error Analysis",
        "",
        f"- top1_errors={len(rows)}",
        f"- component_match_rate={ratio(count_matches(rows, 'component_match'), len(rows)):.4f}",
        f"- product_match_rate={ratio(count_matches(rows, 'product_match'), len(rows)):.4f}",
        "",
    ]
    for title, key in (
        ("Query categories", "query_category"),
        ("Query components", "query_component"),
        ("Best candidate components", "best_component"),
        ("Component pairs", ("query_component", "best_component")),
        ("Query products", "query_product"),
    ):
        lines.extend(render_counter_section(title, rows, key, top=top))
    return "\n".join(lines)


def render_counter_section(title: str, rows: list[dict[str, str]], key, *, top: int) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        if isinstance(key, tuple):
            value = " -> ".join((row.get(part, "") or "-") for part in key)
        else:
            value = row.get(key, "") or "-"
        counter[value] += 1
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    for value, count in counter.most_common(top):
        lines.append(f"| {escape_cell(value)} | {count} |")
    lines.append("")
    return lines


def field_value(ticket: TicketRecord | None, field: str) -> str:
    if ticket is None:
        return ""
    return str(ticket.fields.get(field, ""))


def same_field(left: TicketRecord | None, right: TicketRecord | None, field: str) -> bool:
    left_value = field_value(left, field).strip().lower()
    right_value = field_value(right, field).strip().lower()
    return bool(left_value and left_value == right_value)


def ticket_category(ticket: TicketRecord) -> str:
    text = f"{ticket.title} {ticket.content}".lower()
    categories = (
        ("crash", ("crash", "segfault", "hang", "freeze")),
        ("ui", ("ui", "button", "menu", "toolbar", "dialog", "window", "screen")),
        ("extension", ("extension", "add-on", "addon", "plugin")),
        ("tab/window", ("tab", "window", "popup")),
        ("bookmarks/history", ("bookmark", "history")),
        ("settings/preferences", ("preference", "preferences", "setting", "option")),
        ("website/compatibility", ("site", "webpage", "website", "html", "javascript")),
        ("performance", ("slow", "performance", "memory", "cpu")),
    )
    for category, keywords in categories:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def count_matches(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) == "true")


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
