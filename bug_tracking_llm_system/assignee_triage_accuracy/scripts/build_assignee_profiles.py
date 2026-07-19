from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]

TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+")
DATE_LIKE_RE = re.compile(r"\d{4}[-:.]\d{2}")
TEXT_FIELDS = (
    "title",
    "description",
    "product",
    "component",
    "bug_type",
)
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doesn",
    "don",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "info",
    "in",
    "into",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "our",
    "should",
    "task",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "was",
    "when",
    "will",
    "with",
    "would",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer assignee responsibility profiles from historical ticket ownership."
    )
    parser.add_argument("--dataset", default="controlled", help="Dataset prefix used in output filenames.")
    parser.add_argument("--history", type=Path, default=None, help="History/train JSONL path.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EVAL_ROOT / "data",
        help="Dataset directory used when --history is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_ROOT / "assignee_profiles",
        help="Directory for generated profile reports.",
    )
    parser.add_argument("--top-n", type=int, default=8, help="Number of top components/products/keywords to keep.")
    parser.add_argument("--examples-per-profile", type=int, default=5)
    parser.add_argument("--include-manual-triage", action="store_true")
    parser.add_argument(
        "--write-human-exports",
        action="store_true",
        help="Also write redundant CSV and Markdown views; JSON remains the canonical output.",
    )
    args = parser.parse_args()

    history_path = args.history or args.data_dir / f"{args.dataset}_history_train.jsonl"
    rows = read_jsonl(history_path)
    profiles = build_profiles(
        rows,
        top_n=args.top_n,
        examples_per_profile=args.examples_per_profile,
        include_manual_triage=args.include_manual_triage,
    )
    payload = {
        "metadata": {
            "dataset": args.dataset,
            "history_path": str(history_path),
            "source_rows": len(rows),
            "profile_count": len(profiles),
            "method": "extractive_profile_from_training_tickets",
            "note": (
                "Profiles are inferred from historical ticket ownership. They are not verified job titles "
                "or organization roles."
            ),
        },
        "profiles": profiles,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.dataset}_assignee_profiles.json"
    csv_path = args.output_dir / f"{args.dataset}_assignee_profiles.csv"
    md_path = args.output_dir / f"{args.dataset}_assignee_profiles.md"

    write_json(json_path, payload)
    outputs = {"json": str(json_path)}
    if args.write_human_exports:
        write_csv(csv_path, profiles)
        write_markdown(md_path, payload)
        outputs.update({"csv": str(csv_path), "markdown": str(md_path)})

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "history_path": str(history_path),
                "profiles": len(profiles),
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def build_profiles(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    examples_per_profile: int,
    include_manual_triage: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ticket_document_frequency: Counter[str] = Counter()

    for row in rows:
        assignee = normalize_assignee(row.get("assignee"))
        if not assignee:
            continue
        if assignee == "manual_triage" and not include_manual_triage:
            continue
        grouped[assignee].append(row)
        ticket_document_frequency.update(set(extract_terms(row_text(row))))

    total_tickets = sum(len(items) for items in grouped.values())
    idf = {
        term: math.log(1 + total_tickets / (1 + frequency)) + 1
        for term, frequency in ticket_document_frequency.items()
    }

    profiles = []
    for assignee, tickets in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        term_counts: Counter[str] = Counter()
        component_counts: Counter[str] = Counter()
        product_counts: Counter[str] = Counter()
        priority_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()

        for ticket in tickets:
            term_counts.update(extract_terms(row_text(ticket)))
            component_counts[normalize_label(ticket.get("component"))] += 1
            product = normalize_label(ticket.get("product"))
            if product != "unknown":
                product_counts[product] += 1
            priority = normalize_label(ticket.get("priority"))
            if priority != "unknown":
                priority_counts[priority] += 1
            severity = normalize_label(ticket.get("severity"))
            if severity != "unknown":
                severity_counts[severity] += 1
            status = normalize_label(ticket.get("status"))
            if status != "unknown":
                status_counts[status] += 1

        keyword_scores = {
            term: (1 + math.log(count)) * idf.get(term, 1.0)
            for term, count in term_counts.items()
            if count > 0
        }
        keywords = [
            {"term": term, "count": term_counts[term], "score": round(score, 4)}
            for term, score in sorted(keyword_scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        ]
        representative_tickets = pick_representative_tickets(
            tickets,
            keyword_scores,
            limit=examples_per_profile,
        )
        profile = {
            "assignee": assignee,
            "ticket_count": len(tickets),
            "profile_confidence": confidence_label(len(tickets)),
            "top_products": counter_items(product_counts, top_n),
            "top_components": counter_items(component_counts, top_n),
            "top_priorities": counter_items(priority_counts, top_n),
            "top_severities": counter_items(severity_counts, top_n),
            "top_statuses": counter_items(status_counts, top_n),
            "keywords": keywords,
            "representative_tickets": representative_tickets,
        }
        profile["inferred_responsibility"] = summarize_profile(profile)
        profiles.append(profile)

    return profiles


def pick_representative_tickets(
    tickets: list[dict[str, Any]],
    keyword_scores: dict[str, float],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    scored = []
    for ticket in tickets:
        terms = set(extract_terms(row_text(ticket)))
        score = sum(keyword_scores.get(term, 0.0) for term in terms)
        scored.append((score, str(ticket.get("created_at", "") or ""), str(ticket.get("ticket_id", "") or ""), ticket))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    representatives = []
    seen_components: set[str] = set()
    for _, _, _, ticket in scored:
        component = normalize_label(ticket.get("component"))
        if component in seen_components and len(seen_components) < min(limit, len({normalize_label(t.get("component")) for t in tickets})):
            continue
        representatives.append(ticket_summary(ticket))
        seen_components.add(component)
        if len(representatives) >= limit:
            return representatives

    for _, _, _, ticket in scored:
        summary = ticket_summary(ticket)
        if summary not in representatives:
            representatives.append(summary)
        if len(representatives) >= limit:
            break
    return representatives


def ticket_summary(ticket: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_id": ticket.get("ticket_id", ""),
        "title": ticket.get("title", ""),
        "product": ticket.get("product", ""),
        "component": ticket.get("component", ""),
        "priority": ticket.get("priority", ""),
        "created_at": ticket.get("created_at", ""),
    }


def summarize_profile(profile: dict[str, Any]) -> str:
    count = profile["ticket_count"]
    components = ", ".join(item["value"] for item in profile["top_components"][:3]) or "unknown components"
    products = ", ".join(item["value"] for item in profile["top_products"][:2])
    keywords = ", ".join(item["term"] for item in profile["keywords"][:5]) or "no strong keywords"
    product_part = f" in {products}" if products else ""
    return (
        f"Inferred from {count} historical tickets: likely focuses on {components}{product_part}. "
        f"Common work themes: {keywords}."
    )


def counter_items(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    total = sum(counter.values())
    if not total:
        return []
    return [
        {"value": value, "count": count, "ratio": round(count / total, 6)}
        for value, count in counter.most_common(limit)
    ]


def row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(field, "") or "") for field in TEXT_FIELDS)


def extract_terms(text: str) -> list[str]:
    tokens = [
        normalize_token(token)
        for token in TOKEN_RE.findall(text.lower())
    ]
    tokens = [token for token in tokens if is_useful_token(token)]
    bigrams = [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]
    return tokens + bigrams


def normalize_token(token: str) -> str:
    return token.strip("._-")


def is_useful_token(token: str) -> bool:
    if len(token) < 3:
        return False
    if token in STOPWORDS:
        return False
    if not any(character.isalpha() for character in token):
        return False
    if DATE_LIKE_RE.search(token):
        return False
    digits = sum(1 for character in token if character.isdigit())
    letters = sum(1 for character in token if character.isalpha())
    if digits > letters:
        return False
    return True


def confidence_label(ticket_count: int) -> str:
    if ticket_count >= 20:
        return "high"
    if ticket_count >= 5:
        return "medium"
    return "low"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_label(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, profiles: list[dict[str, Any]]) -> None:
    fields = [
        "assignee",
        "ticket_count",
        "profile_confidence",
        "top_products",
        "top_components",
        "top_priorities",
        "top_keywords",
        "representative_ticket_ids",
        "inferred_responsibility",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            writer.writerow(
                {
                    "assignee": profile["assignee"],
                    "ticket_count": profile["ticket_count"],
                    "profile_confidence": profile["profile_confidence"],
                    "top_products": join_counter_items(profile["top_products"]),
                    "top_components": join_counter_items(profile["top_components"]),
                    "top_priorities": join_counter_items(profile["top_priorities"]),
                    "top_keywords": "; ".join(item["term"] for item in profile["keywords"]),
                    "representative_ticket_ids": "; ".join(
                        str(item.get("ticket_id", "")) for item in profile["representative_tickets"]
                    ),
                    "inferred_responsibility": profile["inferred_responsibility"],
                }
            )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    profiles = payload["profiles"]
    lines = [
        f"# Assignee Responsibility Profiles: {metadata['dataset']}",
        "",
        "> These profiles are inferred from historical ticket ownership only. They are not verified job titles.",
        "",
        f"- History path: `{metadata['history_path']}`",
        f"- Source rows: `{metadata['source_rows']}`",
        f"- Profile count: `{metadata['profile_count']}`",
        "",
        "## Summary",
        "",
        "| Assignee | Tickets | Confidence | Top Components | Top Keywords |",
        "|---|---:|---|---|---|",
    ]
    for profile in profiles:
        components = join_counter_items(profile["top_components"][:3])
        keywords = ", ".join(item["term"] for item in profile["keywords"][:5])
        lines.append(
            f"| `{profile['assignee']}` | {profile['ticket_count']} | "
            f"{profile['profile_confidence']} | {components} | {keywords} |"
        )

    lines.extend(["", "## Details", ""])
    for profile in profiles:
        lines.extend(
            [
                f"### `{profile['assignee']}`",
                "",
                profile["inferred_responsibility"],
                "",
                f"- Ticket count: `{profile['ticket_count']}`",
                f"- Confidence: `{profile['profile_confidence']}`",
                f"- Top products: {join_counter_items(profile['top_products']) or 'n/a'}",
                f"- Top components: {join_counter_items(profile['top_components']) or 'n/a'}",
                f"- Top keywords: {', '.join(item['term'] for item in profile['keywords']) or 'n/a'}",
                "- Representative tickets:",
            ]
        )
        for ticket in profile["representative_tickets"]:
            ticket_id = ticket.get("ticket_id", "")
            title = str(ticket.get("title", "") or "").replace("\n", " ")
            component = ticket.get("component", "")
            lines.append(f"  - `{ticket_id}` [{component}] {title}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def join_counter_items(items: list[dict[str, Any]]) -> str:
    return "; ".join(f"{item['value']} ({item['count']})" for item in items)


if __name__ == "__main__":
    main()
