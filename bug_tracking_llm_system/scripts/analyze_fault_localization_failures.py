#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


WRAPPER_HINTS = {
    "connect.py",
    "dispatcher.py",
    "dispatch.py",
    "loader.py",
    "registry.py",
    "router.py",
}
WRAPPER_PATH_PARTS = {"registry"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze fault-localization misses against file-level gold data.")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL with fixed_files or modified_files.")
    parser.add_argument("--pred", required=True, help="Prediction JSON/JSONL with localized_candidates.")
    parser.add_argument("--index-dir", default=None, help="Optional code-index directory for gold-file coverage checks.")
    parser.add_argument("--output-json", default=None, help="Detailed JSON output path.")
    parser.add_argument("--output-csv", default=None, help="Miss details CSV output path.")
    parser.add_argument("--output-md", default=None, help="Markdown summary output path.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gold_rows = read_records(args.gold)
    pred_rows = read_records(args.pred)
    report = analyze_records(
        gold_rows,
        pred_rows,
        index_dir=Path(args.index_dir) if args.index_dir else None,
        top_k=args.top_k,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        write_misses_csv(Path(args.output_csv), report["misses"])
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def analyze_records(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    index_dir: Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    predictions = [row for row in pred_rows if _ticket_id(row)]
    misses: list[dict[str, Any]] = []
    hits_at = {1: 0, 3: 0, 5: 0}
    rank_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    repo_counts: Counter[str] = Counter()
    top_wrong_file_counts: Counter[str] = Counter()
    indexed_miss_count = 0
    unindexed_miss_count = 0

    for pred in predictions:
        ticket_id = _ticket_id(pred)
        gold = gold_by_id.get(ticket_id, {})
        gold_files = _file_values(gold)
        candidate_files = _candidate_files(pred, top_k=top_k)
        hit_rank = _first_hit_rank(candidate_files, gold_files)
        if hit_rank is not None:
            for k in hits_at:
                if hit_rank <= k:
                    hits_at[k] += 1
            rank_counts[str(hit_rank)] += 1
            continue

        gold_indexed = _gold_files_indexed(pred, gold_files, index_dir)
        if gold_indexed is True:
            indexed_miss_count += 1
        elif gold_indexed is False:
            unindexed_miss_count += 1
        reason = classify_miss(pred, gold_files, candidate_files, gold_indexed)
        pattern = classify_ranking_pattern(pred, gold_files, candidate_files, gold_indexed)
        reason_counts[reason] += 1
        pattern_counts[pattern] += 1
        repo_counts[str(pred.get("repo") or "unknown")] += 1
        if candidate_files:
            top_wrong_file_counts[candidate_files[0]] += 1
        misses.append(
            {
                "ticket_id": ticket_id,
                "repo": pred.get("repo", ""),
                "base_commit": pred.get("base_commit", ""),
                "gold_files": gold_files,
                "top_candidate_files": candidate_files,
                "gold_indexed": gold_indexed,
                "likely_reason": reason,
                "ranking_failure_pattern": pattern,
                "path_overlap_depth": _max_path_prefix_depth(candidate_files[:1], gold_files),
                "top_score_margin": _top_score_margin(pred),
                "top_candidate": _compact_candidate(_top_candidate(pred)),
                "bug_report_preview": str(pred.get("bug_report") or "")[:600],
            }
        )

    evaluated = len(predictions)
    summary = {
        "evaluated_predictions": evaluated,
        "miss_count": len(misses),
        "top_1_hits": hits_at[1],
        "top_3_hits": hits_at[3],
        "top_5_hits": hits_at[5],
        "top_1_accuracy": _ratio(hits_at[1], evaluated),
        "top_3_accuracy": _ratio(hits_at[3], evaluated),
        "top_5_accuracy": _ratio(hits_at[5], evaluated),
        "rank_counts": dict(rank_counts),
        "miss_reason_counts": dict(reason_counts),
        "ranking_failure_pattern_counts": dict(pattern_counts),
        "misses_by_repo": dict(repo_counts.most_common()),
        "top_wrong_files": dict(top_wrong_file_counts.most_common(12)),
        "indexed_misses": indexed_miss_count,
        "unindexed_misses": unindexed_miss_count,
    }
    return {"summary": summary, "misses": misses}


def classify_miss(
    pred: dict[str, Any],
    gold_files: list[str],
    candidate_files: list[str],
    gold_indexed: bool | None,
) -> str:
    if not candidate_files:
        return "no_candidates"
    if gold_indexed is False:
        return "gold_file_not_indexed"

    top = _top_candidate(pred)
    signals = top.get("scoring_signals") if isinstance(top, dict) else {}
    if not isinstance(signals, dict):
        signals = {}
    stack_score = float(signals.get("stack_trace_score") or 0.0)
    top_file = _normalize_path(str(top.get("file_path") or "")) if isinstance(top, dict) else ""
    if stack_score >= 0.55 and _looks_like_wrapper(top_file):
        return "stack_trace_wrapper_bias"

    report = str(pred.get("bug_report") or "").lower().replace("\\", "/")
    if any(_path_tokens_match_report(gold_file, report) for gold_file in gold_files):
        return "path_hint_not_ranked"

    top_score = float(top.get("score") or 0.0) if isinstance(top, dict) else 0.0
    if top_score < 0.35:
        return "weak_report_code_similarity"
    return "ranking_error_gold_indexed"


def classify_ranking_pattern(
    pred: dict[str, Any],
    gold_files: list[str],
    candidate_files: list[str],
    gold_indexed: bool | None,
) -> str:
    if not candidate_files:
        return "no_candidates"
    if gold_indexed is False:
        return "gold_file_not_indexed"

    top = _top_candidate(pred)
    signals = top.get("scoring_signals") if isinstance(top, dict) else {}
    if not isinstance(signals, dict):
        signals = {}
    top_file = _normalize_path(str(top.get("file_path") or "")) if isinstance(top, dict) else candidate_files[0]
    stack_score = _safe_float(signals.get("stack_trace_score"))
    path_hint_score = _safe_float(signals.get("path_hint_score"))
    identifier_score = _safe_float(signals.get("identifier_score"))
    top_score = _safe_float(top.get("score") if isinstance(top, dict) else 0.0)
    report = str(pred.get("bug_report") or "").lower().replace("\\", "/")

    if stack_score >= 0.55 and _looks_like_wrapper(top_file):
        return "stack_trace_wrapper_bias"
    if any(_path_tokens_match_report(gold_file, report) for gold_file in gold_files) and path_hint_score < 0.35:
        return "gold_path_hint_underweighted"
    if _any_generated_or_migration_noise(candidate_files):
        return "generated_or_migration_noise"

    overlap_depth = _max_path_prefix_depth(candidate_files[:1], gold_files)
    if overlap_depth >= 2:
        return "near_miss_same_subpackage"
    if overlap_depth == 1:
        return "same_top_level_package_confusion"
    if top_score < 0.50:
        return "weak_report_code_similarity"
    if top_score >= 0.75 and max(stack_score, path_hint_score, identifier_score) < 0.35:
        return "high_score_semantic_drift"
    return "general_ranking_error_gold_indexed"


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def write_misses_csv(path: Path, misses: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ticket_id",
        "repo",
        "base_commit",
        "gold_files",
        "top_candidate_files",
        "gold_indexed",
        "likely_reason",
        "ranking_failure_pattern",
        "path_overlap_depth",
        "top_score_margin",
        "top_file_path",
        "top_score",
        "top_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in misses:
            top = row.get("top_candidate", {})
            writer.writerow(
                {
                    "ticket_id": row.get("ticket_id", ""),
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "gold_files": ";".join(row.get("gold_files") or []),
                    "top_candidate_files": ";".join(row.get("top_candidate_files") or []),
                    "gold_indexed": row.get("gold_indexed"),
                    "likely_reason": row.get("likely_reason", ""),
                    "ranking_failure_pattern": row.get("ranking_failure_pattern", ""),
                    "path_overlap_depth": row.get("path_overlap_depth", ""),
                    "top_score_margin": row.get("top_score_margin", ""),
                    "top_file_path": top.get("file_path", "") if isinstance(top, dict) else "",
                    "top_score": top.get("score", "") if isinstance(top, dict) else "",
                    "top_reason": top.get("reason", "") if isinstance(top, dict) else "",
                }
            )


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# SWE-bench Lite 300 Top-5 Miss Pattern Analysis",
        "",
        f"- Evaluated predictions: {summary.get('evaluated_predictions', 0)}",
        f"- Top-5 misses: {summary.get('miss_count', 0)}",
        f"- Indexed misses: {summary.get('indexed_misses', 0)}",
        f"- Unindexed misses: {summary.get('unindexed_misses', 0)}",
        f"- Top-5 accuracy: {float(summary.get('top_5_accuracy', 0.0)):.4f}",
        "",
        "## Ranking Failure Patterns",
        "",
        "| Pattern | Count | Interpretation |",
        "|---|---:|---|",
    ]
    for pattern, count in summary.get("ranking_failure_pattern_counts", {}).items():
        lines.append(f"| {pattern} | {count} | {pattern_interpretation(pattern)} |")
    lines.extend(
        [
            "",
            "## Misses By Repository",
            "",
            "| Repo | Misses |",
            "|---|---:|",
        ]
    )
    for repo, count in summary.get("misses_by_repo", {}).items():
        lines.append(f"| {repo} | {count} |")
    lines.extend(
        [
            "",
            "## Frequent Top-1 Wrong Files",
            "",
            "| File | Count |",
            "|---|---:|",
        ]
    )
    for file_path, count in summary.get("top_wrong_files", {}).items():
        lines.append(f"| `{file_path}` | {count} |")
    lines.extend(
        [
            "",
            "## Miss Details",
            "",
            "| Ticket | Repo | Pattern | Gold files | Top-1 wrong file | Top score |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in report.get("misses", []):
        top = row.get("top_candidate") if isinstance(row.get("top_candidate"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row.get("ticket_id", "")),
                    _escape_md(row.get("repo", "")),
                    _escape_md(row.get("ranking_failure_pattern", "")),
                    _escape_md("; ".join(row.get("gold_files") or [])),
                    f"`{_escape_md(top.get('file_path', ''))}`",
                    str(top.get("score", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def pattern_interpretation(pattern: str) -> str:
    return {
        "stack_trace_wrapper_bias": "Top candidate follows a wrapper/dispatcher frame instead of the true implementation.",
        "gold_path_hint_underweighted": "The report contains a gold-path hint, but path evidence was not strong enough.",
        "generated_or_migration_noise": "Generated/migration-like files crowded the candidate list.",
        "near_miss_same_subpackage": "Top-1 is in the same subpackage as the gold file, suggesting local ranking confusion.",
        "same_top_level_package_confusion": "Top-1 is only broadly in the same top-level package.",
        "weak_report_code_similarity": "The report has weak lexical/semantic overlap with retrieved code.",
        "high_score_semantic_drift": "High retrieval score without direct stack/path/identifier evidence.",
        "general_ranking_error_gold_indexed": "Gold file is indexed, but current ranking signals did not lift it into Top-5.",
        "gold_file_not_indexed": "Gold file was absent from the code index.",
        "no_candidates": "No localization candidates were produced.",
    }.get(pattern, "Unclassified ranking pattern.")


def _gold_files_indexed(pred: dict[str, Any], gold_files: list[str], index_dir: Path | None) -> bool | None:
    if index_dir is None:
        return None
    repo = str(pred.get("repo") or "")
    commit = str(pred.get("base_commit") or "working-tree")
    if not repo:
        return None
    index_path = index_dir / f"{repo.replace('/', '__')}__{commit[:12]}.json"
    if not index_path.exists():
        return None
    try:
        value = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    indexed = {_normalize_path(str(chunk.get("file_path") or "")) for chunk in value.get("chunks", []) if isinstance(chunk, dict)}
    return any(_normalize_path(gold_file) in indexed for gold_file in gold_files)


def _candidate_files(row: dict[str, Any], *, top_k: int) -> list[str]:
    values: list[str] = []
    candidates = row.get("localized_files")
    if not isinstance(candidates, list):
        candidates = row.get("localized_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            file_path = _normalize_path(str(candidate.get("file_path") or ""))
            if file_path and file_path not in values:
                values.append(file_path)
            if len(values) >= top_k:
                break
    return values


def _top_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidates = row.get("localized_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
    return {}


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    signals = candidate.get("scoring_signals") if isinstance(candidate.get("scoring_signals"), dict) else {}
    return {
        "file_path": candidate.get("file_path", ""),
        "function_name": candidate.get("function_name") or candidate.get("class_name") or "",
        "start_line": candidate.get("start_line"),
        "end_line": candidate.get("end_line"),
        "score": candidate.get("score"),
        "reason": candidate.get("reason", ""),
        "scoring_signals": {
            "embedding_score": signals.get("embedding_score"),
            "stack_trace_score": signals.get("stack_trace_score"),
            "component_score": signals.get("component_score"),
            "keyword_score": signals.get("keyword_score"),
            "symbol_score": signals.get("symbol_score"),
            "path_hint_score": signals.get("path_hint_score"),
            "wrapper_penalty": signals.get("wrapper_penalty"),
            "final_score": signals.get("final_score"),
        },
    }


def _file_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths", "file_path", "file"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    unique: list[str] = []
    for value in values:
        normalized = _normalize_path(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _first_hit_rank(candidate_files: list[str], gold_files: list[str]) -> int | None:
    for rank, file_path in enumerate(candidate_files, start=1):
        if any(_file_matches(file_path, gold_file) for gold_file in gold_files):
            return rank
    return None


def _top_score_margin(row: dict[str, Any]) -> float:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list) or not candidates:
        return 0.0
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    second = candidates[1] if len(candidates) > 1 and isinstance(candidates[1], dict) else {}
    return round(max(0.0, _safe_float(first.get("score")) - _safe_float(second.get("score"))), 4)


def _max_path_prefix_depth(candidate_files: list[str], gold_files: list[str]) -> int:
    best = 0
    for candidate in candidate_files:
        candidate_parts = [part for part in Path(_normalize_path(candidate)).parts if part]
        for gold in gold_files:
            gold_parts = [part for part in Path(_normalize_path(gold)).parts if part]
            depth = 0
            for left, right in zip(candidate_parts, gold_parts):
                if left != right:
                    break
                depth += 1
            best = max(best, depth)
    return best


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _path_tokens_match_report(gold_file: str, report: str) -> bool:
    gold = _normalize_path(gold_file)
    stem = gold[:-3] if gold.endswith(".py") else gold
    dotted = stem.replace("/", ".")
    tail = ".".join(stem.split("/")[-2:])
    return dotted.lower() in report or tail.lower() in report


def _looks_like_wrapper(path: str) -> bool:
    parts = set(Path(path).parts)
    return Path(path).name in WRAPPER_HINTS or bool(parts & WRAPPER_PATH_PARTS)


def _any_generated_or_migration_noise(candidate_files: list[str]) -> bool:
    for path in candidate_files[:3]:
        normalized = _normalize_path(path)
        parts = set(Path(normalized).parts)
        name = Path(normalized).name
        if "migrations" in parts or re_like_migration_filename(name):
            return True
        if name.endswith("_pb2.py") or name.endswith("_generated.py"):
            return True
    return False


def re_like_migration_filename(name: str) -> bool:
    return bool(name.endswith(".py") and len(name) >= 8 and name[:4].isdigit() and name[4] == "_")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or row.get("query_id") or "")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    main()
