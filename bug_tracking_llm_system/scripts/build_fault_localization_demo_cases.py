#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small Markdown demo report for fault localization results.")
    parser.add_argument("--pred", required=True, help="Fault localization predictions JSONL.")
    parser.add_argument("--gold", required=True, help="Fault localization gold JSONL.")
    parser.add_argument("--metrics", default=None, help="Optional metrics JSON.")
    parser.add_argument("--failure-analysis", default=None, help="Optional failure analysis JSON.")
    parser.add_argument("--llm-pred", default=None, help="Optional LLM rerank prediction JSONL for a conservative rerank example.")
    parser.add_argument("--llm-cache", default=None, help="Optional LLM rerank cache JSONL.")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = read_records(Path(args.pred))
    gold_rows = read_records(Path(args.gold))
    metrics = read_json(Path(args.metrics)) if args.metrics else {}
    failure_analysis = read_json(Path(args.failure_analysis)) if args.failure_analysis else {}
    llm_predictions = read_records(Path(args.llm_pred)) if args.llm_pred else []
    llm_cache_rows = read_records(Path(args.llm_cache)) if args.llm_cache else []

    report = build_demo_report(
        predictions,
        gold_rows,
        metrics=metrics,
        failure_analysis=failure_analysis,
        llm_predictions=llm_predictions,
        llm_cache_rows=llm_cache_rows,
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output_md": str(output_md), "case_count": len(report["cases"])}, ensure_ascii=False))


def build_demo_report(
    predictions: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    metrics: dict[str, Any],
    failure_analysis: dict[str, Any],
    llm_predictions: list[dict[str, Any]],
    llm_cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    enriched = [_enrich_prediction(row, gold_by_id.get(_ticket_id(row), {})) for row in predictions]
    cases: list[dict[str, Any]] = []
    used: set[str] = set()

    add_case(
        cases,
        used,
        "Top-1 命中案例",
        enriched,
        lambda row: row["hit_rank"] == 1,
        "展示系統能直接把正解檔案排在第一名。",
    )
    add_case(
        cases,
        used,
        "Top-3/Top-5 命中案例",
        enriched,
        lambda row: row["hit_rank"] is not None and 2 <= int(row["hit_rank"]) <= 5,
        "展示即使第一名不是正解，Top-k 仍能提供有效候選給開發者。",
    )
    add_case(
        cases,
        used,
        "Top-5 未命中案例",
        enriched,
        lambda row: row["hit_rank"] is None,
        "展示目前限制：正解檔案在 index 中，但排序仍可能不夠準。",
    )
    add_case(
        cases,
        used,
        "Path/Stack Trace 訊號有幫助案例",
        enriched,
        _has_strong_trace_or_path_signal,
        "展示系統不是只靠語意相似度，也會利用 stack trace/path hint。",
    )

    llm_case = build_llm_case(llm_predictions, gold_by_id, llm_cache_rows)
    if llm_case is not None:
        cases.append(llm_case)

    return {
        "summary": {
            "source_prediction_rows": len(predictions),
            "metrics": metrics,
            "failure_summary": failure_analysis.get("summary", {}),
        },
        "cases": cases,
    }


def add_case(
    cases: list[dict[str, Any]],
    used: set[str],
    label: str,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    why_show: str,
) -> None:
    for row in rows:
        ticket_id = str(row.get("ticket_id") or "")
        if ticket_id in used:
            continue
        if predicate(row):
            cases.append({**row, "case_type": label, "why_show": why_show})
            used.add(ticket_id)
            return


def build_llm_case(
    llm_predictions: list[dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    llm_cache_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    cache_by_ticket: dict[str, dict[str, Any]] = {}
    for row in llm_cache_rows:
        ticket_id = str(row.get("ticket_id") or "")
        if ticket_id:
            cache_by_ticket[ticket_id] = row

    for prediction in llm_predictions:
        if not prediction.get("method", {}).get("llm_rerank"):
            continue
        ticket_id = _ticket_id(prediction)
        enriched = _enrich_prediction(prediction, gold_by_id.get(ticket_id, {}))
        cache_row = cache_by_ticket.get(ticket_id, {})
        selected = _cached_llm_selection(cache_row)
        if selected:
            enriched["llm_cached_selection"] = selected
        enriched["case_type"] = "LLM rerank 保守融合案例"
        enriched["why_show"] = (
            "展示 LLM rerank 是可選訊號；即使本機模型選到較弱候選，"
            "保守 blending 也不會輕易覆蓋 retrieval baseline。"
        )
        return enriched
    return None


def _enrich_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_rows(prediction)
    gold_files = _file_values(gold)
    candidate_files = [str(row.get("file_path") or row.get("file") or "") for row in candidates]
    hit_rank = _first_hit_rank(candidate_files, gold_files)
    return {
        "ticket_id": _ticket_id(prediction),
        "repo": prediction.get("repo", ""),
        "base_commit": prediction.get("base_commit", ""),
        "gold_files": gold_files,
        "hit_rank": hit_rank,
        "top_k_hit": hit_rank is not None,
        "bug_report_preview": _compact_text(str(prediction.get("bug_report") or ""), limit=650),
        "top_candidates": [_compact_candidate(row, gold_files) for row in candidates[:5]],
        "warnings": prediction.get("warnings", []),
    }


def _compact_candidate(candidate: dict[str, Any], gold_files: list[str]) -> dict[str, Any]:
    file_path = str(candidate.get("file_path") or candidate.get("file") or "")
    signals = candidate.get("scoring_signals") if isinstance(candidate.get("scoring_signals"), dict) else {}
    return {
        "rank": candidate.get("rank"),
        "file_path": file_path,
        "function_name": candidate.get("function_name") or candidate.get("class_name") or candidate.get("symbol_name") or "",
        "start_line": candidate.get("start_line"),
        "end_line": candidate.get("end_line"),
        "score": candidate.get("score"),
        "is_gold_match": any(_file_matches(file_path, gold_file) for gold_file in gold_files),
        "reason": _compact_text(str(candidate.get("reason") or ""), limit=260),
        "signals": {
            "stack_trace_score": signals.get("stack_trace_score"),
            "path_hint_score": signals.get("path_hint_score"),
            "identifier_score": signals.get("identifier_score"),
            "domain_path_score": signals.get("domain_path_score"),
            "llm_rerank_score": signals.get("llm_rerank_score"),
            "llm_rerank_blended_score": signals.get("llm_rerank_blended_score"),
            "llm_rerank_cache_hit": signals.get("llm_rerank_cache_hit"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fault Localization Demo Cases",
        "",
        "本文件整理可展示的錯誤定位案例。重點是說明目前系統採用 retrieval-first，LLM rerank 只作為小候選集的可選輔助，不直接讀完整 repository。",
        "",
        "## Overall Metrics",
        "",
    ]
    metrics = report.get("summary", {}).get("metrics", {})
    if metrics:
        lines.extend(
            [
                f"- Evaluated tickets: {metrics.get('rows', metrics.get('rows_with_ground_truth', ''))}",
                f"- Top-1 Accuracy: {_fmt(metrics.get('top_1_accuracy'))}",
                f"- Top-3 Accuracy: {_fmt(metrics.get('top_3_accuracy'))}",
                f"- Top-5 Accuracy: {_fmt(metrics.get('top_5_accuracy'))}",
                f"- MRR: {_fmt(metrics.get('mrr'))}",
                "",
            ]
        )
    failure_summary = report.get("summary", {}).get("failure_summary", {})
    if failure_summary:
        lines.extend(
            [
                "## Failure Analysis Summary",
                "",
                f"- Top-5 misses: {failure_summary.get('miss_count', '')}",
                f"- Indexed misses: {failure_summary.get('indexed_misses', '')}",
                f"- Unindexed misses: {failure_summary.get('unindexed_misses', '')}",
                f"- Main miss reason: {failure_summary.get('miss_reason_counts', {})}",
                "",
            ]
        )

    for index, case in enumerate(report.get("cases", []), start=1):
        lines.extend(render_case(index, case))
    return "\n".join(lines).rstrip() + "\n"


def render_case(index: int, case: dict[str, Any]) -> list[str]:
    lines = [
        f"## Case {index}: {case.get('case_type', '')}",
        "",
        f"- Ticket: `{case.get('ticket_id', '')}`",
        f"- Repo: `{case.get('repo', '')}`",
        f"- Gold file(s): `{', '.join(case.get('gold_files') or [])}`",
        f"- Hit rank: `{case.get('hit_rank')}`",
        f"- Why show this: {case.get('why_show', '')}",
        "",
        "### Bug Report Preview",
        "",
        "```text",
        str(case.get("bug_report_preview") or ""),
        "```",
        "",
        "### Top Candidates",
        "",
        "| Rank | File | Function/Class | Lines | Score | Gold? | Key Signals |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for candidate in case.get("top_candidates", []):
        signals = candidate.get("signals", {})
        key_signals = ", ".join(
            f"{key}={value}"
            for key, value in signals.items()
            if value not in (None, "", [])
        )
        lines.append(
            "| {rank} | `{file}` | `{func}` | {start}-{end} | {score} | {gold} | {signals} |".format(
                rank=candidate.get("rank"),
                file=candidate.get("file_path", ""),
                func=candidate.get("function_name", ""),
                start=candidate.get("start_line", ""),
                end=candidate.get("end_line", ""),
                score=candidate.get("score", ""),
                gold="yes" if candidate.get("is_gold_match") else "no",
                signals=key_signals,
            )
        )
    llm_selection = case.get("llm_cached_selection")
    if isinstance(llm_selection, dict):
        lines.extend(
            [
                "",
                "### LLM Cached Selection",
                "",
                f"- LLM selected original rank: `{llm_selection.get('rank', '')}`",
                f"- LLM selected file: `{llm_selection.get('file_path', '')}`",
                "- Note: conservative blending keeps retrieval evidence dominant when LLM output is weak.",
            ]
        )
    lines.append("")
    return lines


def _has_strong_trace_or_path_signal(row: dict[str, Any]) -> bool:
    if row.get("hit_rank") != 1:
        return False
    candidates = row.get("top_candidates") or []
    if not candidates:
        return False
    signals = candidates[0].get("signals", {})
    return float(signals.get("stack_trace_score") or 0.0) > 0.0 or float(signals.get("path_hint_score") or 0.0) > 0.0


def _cached_llm_selection(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    raw = payload.get("raw_payload") if isinstance(payload.get("raw_payload"), dict) else {}
    if raw:
        return {"rank": raw.get("rank"), "file_path": raw.get("file_path"), "score": raw.get("score")}
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    return {"rank": first.get("rank"), "file_path": first.get("file_path"), "score": first.get("score")} if first else {}


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        value = read_json(path)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict) and isinstance(value.get("records"), list):
            return [row for row in value["records"] if isinstance(row, dict)]
        return [value] if isinstance(value, dict) else []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _candidate_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("localized_candidates")
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    candidates = row.get("localized_files")
    return [candidate for candidate in candidates if isinstance(candidate, dict)] if isinstance(candidates, list) else []


def _file_values(row: dict[str, Any]) -> list[str]:
    values = row.get("fixed_files") or row.get("modified_files") or row.get("files") or []
    if isinstance(values, str):
        values = [values]
    return [_normalize_path(str(value)) for value in values if str(value).strip()]


def _first_hit_rank(candidate_files: list[str], gold_files: list[str]) -> int | None:
    for index, candidate_file in enumerate(candidate_files, start=1):
        if any(_file_matches(candidate_file, gold_file) for gold_file in gold_files):
            return index
    return None


def _file_matches(candidate_file: str, gold_file: str) -> bool:
    candidate = _normalize_path(candidate_file)
    gold = _normalize_path(gold_file)
    return candidate == gold or candidate.endswith(f"/{gold}") or gold.endswith(f"/{candidate}")


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("instance_id") or "")


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value or "")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
