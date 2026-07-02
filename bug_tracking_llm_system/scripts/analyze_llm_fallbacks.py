#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FALLBACK_TYPES = (
    "schema invalid",
    "rank invalid",
    "timeout",
    "service unavailable",
    "echo-like output",
    "empty JSON",
    "other",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify optional LLM rerank fallback warnings.")
    parser.add_argument("--pred", required=True, help="Fault-localization prediction JSONL.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_predictions(read_jsonl(Path(args.pred)), source=str(args.pred))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def analyze_predictions(rows: list[dict[str, Any]], *, source: str = "") -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        ticket_id = str(row.get("ticket_id") or "")
        for warning in row.get("warnings") or []:
            message = str(warning)
            if "LLM reranking failed" not in message:
                continue
            fallback_type = classify_fallback(message)
            counts[fallback_type] += 1
            items.append(
                {
                    "ticket_id": ticket_id,
                    "repo": row.get("repo", ""),
                    "fallback_type": fallback_type,
                    "message": message,
                    "short_term_fix": short_term_fix(fallback_type),
                }
            )
    return {
        "source": source,
        "summary": {
            "prediction_rows": len(rows),
            "fallback_warnings": len(items),
            "counts": {fallback_type: counts.get(fallback_type, 0) for fallback_type in FALLBACK_TYPES},
        },
        "items": items,
    }


def classify_fallback(message: str) -> str:
    text = message.lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "operation not permitted" in text or "connection refused" in text or "urlopen error" in text or "service" in text:
        return "service unavailable"
    if "echo" in text or "do not echo" in text:
        return "echo-like output"
    if "empty json" in text or "raw_text" in text and "{}" in text:
        return "empty JSON"
    if (
        "does not include valid candidate ranks" in text
        or "does not include valid candidate ids" in text
        or "rank" in text
        and "valid" in text
        or "candidate id" in text
        and "valid" in text
    ):
        return "rank invalid"
    if "must include candidate ranking rows" in text or "schema" in text or "json object" in text:
        return "schema invalid"
    return "other"


def short_term_fix(fallback_type: str) -> str:
    if fallback_type == "schema invalid":
        return "Use stricter structured output and a shorter retry prompt."
    if fallback_type == "rank invalid":
        return "Force candidate_id-only output and validate rank IDs before blending."
    if fallback_type == "timeout":
        return "Shorten prompt, reduce candidate count, and keep response cache/checkpoint."
    if fallback_type == "service unavailable":
        return "Keep retrieval fallback; check local Ollama/network availability before demo."
    if fallback_type == "echo-like output":
        return "Keep echo guard; remove input-only fields from accepted output schema."
    if fallback_type == "empty JSON":
        return "Treat as invalid output and retry once with minimal JSON array instructions."
    return "Inspect the raw warning and add a targeted guard if it repeats."


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["summary"]["counts"]
    lines = [
        "# LLM Rerank Fallback Analysis",
        "",
        f"- Source: `{report.get('source', '')}`",
        f"- Prediction rows: {report['summary']['prediction_rows']}",
        f"- Fallback warnings: {report['summary']['fallback_warnings']}",
        "",
        "| Fallback type | Count | Likely cause | Short-term fix |",
        "|---|---:|---|---|",
    ]
    for fallback_type in FALLBACK_TYPES:
        lines.append(
            "| "
            + " | ".join(
                [
                    fallback_type,
                    str(counts.get(fallback_type, 0)),
                    likely_cause(fallback_type),
                    short_term_fix(fallback_type),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cases", "", "| Ticket | Repo | Type | Message |", "|---|---|---|---|"])
    for item in report["items"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(item.get("ticket_id", "")),
                    escape_md(item.get("repo", "")),
                    escape_md(item.get("fallback_type", "")),
                    escape_md(item.get("message", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def likely_cause(fallback_type: str) -> str:
    if fallback_type == "schema invalid":
        return "The model returned notes or prose instead of candidate rows."
    if fallback_type == "rank invalid":
        return "The model referenced files/ranks that did not match the provided candidate IDs."
    if fallback_type == "timeout":
        return "Local model response exceeded the configured timeout."
    if fallback_type == "service unavailable":
        return "The local LLM endpoint was unavailable or blocked."
    if fallback_type == "echo-like output":
        return "The model copied candidate input fields without a judgement."
    if fallback_type == "empty JSON":
        return "The parsed response did not contain usable JSON."
    return "Unclassified LLM rerank failure."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
