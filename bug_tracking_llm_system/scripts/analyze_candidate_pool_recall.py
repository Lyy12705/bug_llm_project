#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_fault_localization import read_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze candidate-pool Recall@K for fault localization predictions.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL with fixed files.")
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL to analyze.")
    parser.add_argument(
        "--focus-misses-from",
        default=None,
        help="Optional baseline prediction JSON/JSONL. Analyze only rows missed by this baseline.",
    )
    parser.add_argument("--focus-top-k", type=int, default=5)
    parser.add_argument("--ks", default="10,20,30", help="Comma-separated K values, e.g. 10,20,30.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ks = _parse_ks(args.ks)
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    focus_rows = read_records(args.focus_misses_from) if args.focus_misses_from else None
    report = analyze_candidate_pool_recall(
        gold_rows,
        pred_rows,
        ks=ks,
        focus_predictions=focus_rows,
        focus_top_k=args.focus_top_k,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")
    print(text)


def analyze_candidate_pool_recall(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    ks: list[int],
    focus_predictions: list[dict[str, Any]] | None = None,
    focus_top_k: int = 5,
) -> dict[str, Any]:
    pred_by_id = {_ticket_id(row): row for row in pred_rows if _ticket_id(row)}
    focus_ids = _focus_miss_ids(gold_rows, focus_predictions, focus_top_k=focus_top_k) if focus_predictions else None
    rows: list[dict[str, Any]] = []
    hits_at = {k: 0 for k in ks}
    reciprocal_rank_total = 0.0
    rank_counts: Counter[str] = Counter()
    missing_predictions: list[str] = []
    missing_ground_truth: list[str] = []
    per_repo_rows: dict[str, list[dict[str, Any]]] = {}

    for gold in gold_rows:
        ticket_id = _ticket_id(gold)
        if focus_ids is not None and ticket_id not in focus_ids:
            continue
        gold_files = _gold_files(gold)
        if not gold_files:
            missing_ground_truth.append(ticket_id)
            continue
        pred = pred_by_id.get(ticket_id)
        if pred is None:
            missing_predictions.append(ticket_id)
            candidate_files: list[str] = []
        else:
            candidate_files = _candidate_files(pred)
        rank = _first_relevant_rank(candidate_files, gold_files)
        if rank is not None:
            reciprocal_rank_total += 1.0 / rank
            rank_counts[str(rank)] += 1
            for k in ks:
                if rank <= k:
                    hits_at[k] += 1
        else:
            rank_counts["miss"] += 1
        row = {
            "ticket_id": ticket_id,
            "repo": _repo_value(gold) or _repo_value(pred or {}) or "unknown",
            "gold_files": gold_files,
            "rank": rank,
            "top_files": candidate_files[: max(ks)],
        }
        rows.append(row)
        per_repo_rows.setdefault(row["repo"], []).append(row)

    evaluated = len(rows)
    summary = {
        "rows_evaluated": evaluated,
        "focus_top_k": focus_top_k if focus_predictions else None,
        "ks": ks,
        "recall": {f"recall@{k}": _ratio(hits_at[k], evaluated) for k in ks},
        "mrr": _ratio(reciprocal_rank_total, evaluated),
        "rank_counts": dict(sorted(rank_counts.items(), key=lambda item: (item[0] == "miss", int(item[0]) if item[0].isdigit() else 10**9))),
        "missing_predictions": len(missing_predictions),
        "missing_ground_truth": len(missing_ground_truth),
    }
    return {
        "summary": summary,
        "per_repo": _per_repo_summary(per_repo_rows, ks),
        "misses": [row for row in rows if row["rank"] is None],
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    ks = summary["ks"]
    lines = [
        "# Candidate Pool Recall Report",
        "",
        f"- Rows evaluated: {summary['rows_evaluated']}",
        f"- Focus baseline Top-K: {summary['focus_top_k']}",
        f"- MRR: {summary['mrr']:.4f}",
        "",
        "## Recall",
        "",
        "| K | Recall |",
        "|---:|---:|",
    ]
    for k in ks:
        lines.append(f"| {k} | {summary['recall'][f'recall@{k}']:.4f} |")
    lines.extend(
        [
            "",
            "## Per Repo",
            "",
            "| Repo | Rows | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR |",
            "|---|---:|" + "|".join("---:" for _ in ks) + "|---:|",
        ]
    )
    for row in report["per_repo"]:
        values = [row["repo"], str(row["rows"])]
        values.extend(f"{row[f'recall@{k}']:.4f}" for k in ks)
        values.append(f"{row['mrr']:.4f}")
        lines.append("| " + " | ".join(_escape_md(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Misses",
            "",
            "| Ticket | Repo | Gold files | Top files preview |",
            "|---|---|---|---|",
        ]
    )
    for row in report["misses"][:50]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row["ticket_id"]),
                    _escape_md(row["repo"]),
                    _escape_md(", ".join(row["gold_files"])),
                    _escape_md(", ".join(row["top_files"][:5])),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _per_repo_summary(per_repo_rows: dict[str, list[dict[str, Any]]], ks: list[int]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for repo in sorted(per_repo_rows):
        rows = per_repo_rows[repo]
        hits_at = {k: 0 for k in ks}
        reciprocal_rank_total = 0.0
        for row in rows:
            rank = row["rank"]
            if rank is None:
                continue
            reciprocal_rank_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits_at[k] += 1
        repo_row = {"repo": repo, "rows": len(rows), "mrr": _ratio(reciprocal_rank_total, len(rows))}
        for k in ks:
            repo_row[f"recall@{k}"] = _ratio(hits_at[k], len(rows))
        summary.append(repo_row)
    return summary


def _focus_miss_ids(
    gold_rows: list[dict[str, Any]],
    focus_predictions: list[dict[str, Any]] | None,
    *,
    focus_top_k: int,
) -> set[str]:
    pred_by_id = {_ticket_id(row): row for row in focus_predictions or [] if _ticket_id(row)}
    ids: set[str] = set()
    for gold in gold_rows:
        ticket_id = _ticket_id(gold)
        pred = pred_by_id.get(ticket_id)
        if not ticket_id or pred is None:
            continue
        rank = _first_relevant_rank(_candidate_files(pred), _gold_files(gold))
        if rank is None or rank > focus_top_k:
            ids.add(ticket_id)
    return ids


def _candidate_files(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("localized_files")
    if not isinstance(candidates, list):
        return []
    values: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            path = _normalize_path(str(candidate.get("file_path") or candidate.get("file") or ""))
            if path and path not in values:
                values.append(path)
    return values


def _gold_files(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths", "file_path", "file"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return _unique_paths(values)


def _first_relevant_rank(candidate_files: list[str], gold_files: list[str]) -> int | None:
    for rank, candidate in enumerate(candidate_files, start=1):
        if any(_file_matches(candidate, gold) for gold in gold_files):
            return rank
    return None


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _repo_value(row: dict[str, Any]) -> str:
    repo = str(row.get("repo") or row.get("repository") or row.get("project") or "").strip()
    if repo:
        return repo
    ticket_id = _ticket_id(row)
    if "__" in ticket_id:
        owner, rest = ticket_id.split("__", maxsplit=1)
        project = rest.rsplit("-", maxsplit=1)[0]
        if owner and project:
            return f"{owner}/{project}"
    return ""


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _unique_paths(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = _normalize_path(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("./")


def _parse_ks(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--ks must include positive integers.")
    return values


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
