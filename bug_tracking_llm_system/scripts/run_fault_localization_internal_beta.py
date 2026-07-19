#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_fault_localization import evaluate_records
from scripts.run_swebench_lite_fault_localization import BatchRunConfig, load_or_build_index, resolve_repository
from utils.fault_localization import (
    build_code_index,
    format_user_facing_localization_result,
    format_user_facing_validation_error,
    localize_ticket,
    validate_localization_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small internal beta check for user-facing fault-localization output."
    )
    parser.add_argument("--tickets-jsonl", default="demo/data/fault_localization_demo_cases.jsonl")
    parser.add_argument("--gold", default="demo/data/fault_localization_demo_gold.jsonl")
    parser.add_argument("--repo-path", default="demo/fault_localization_auth_repo")
    parser.add_argument("--output-dir", default="reports/fault_localization/internal_beta_demo")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-backend", default="tfidf")
    parser.add_argument("--skip-demo", action="store_true", help="Skip the local high/medium/low demo cases.")
    parser.add_argument("--skip-controlled", action="store_true", help="Skip the controlled SWE-bench Lite beta check.")
    parser.add_argument("--controlled-tickets", default="data/fault_localization/swebench_lite/test_tickets.jsonl")
    parser.add_argument("--controlled-gold", default="data/fault_localization/swebench_lite/test_gold.jsonl")
    parser.add_argument("--controlled-repo-cache-dir", default="data/fault_localization/swebench_lite/repos")
    parser.add_argument("--controlled-index-cache-dir", default="data/fault_localization/swebench_lite/indexes")
    parser.add_argument("--controlled-limit", type=int, default=20)
    parser.add_argument("--controlled-offset", type=int, default=0)
    parser.add_argument(
        "--controlled-embedding-backend",
        default=None,
        help="Retrieval backend for controlled beta cases. Defaults to --embedding-backend.",
    )
    parser.add_argument(
        "--controlled-predictions",
        default="reports/fault_localization/swebench_lite_llm_candidate_id_prompt_top5_pool_miss_20_ollama/test_predictions.jsonl",
        help="Optional existing controlled-evaluation predictions to summarize without rerunning LLM.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tickets_path = _resolve(args.tickets_jsonl)
    gold_path = _resolve(args.gold)
    repo_path = _resolve(args.repo_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_result = {}
    if not args.skip_demo:
        demo_result = run_demo_beta(
            tickets_path=tickets_path,
            gold_path=gold_path,
            repo_path=repo_path,
            output_dir=output_dir,
            top_k=args.top_k,
            embedding_backend=args.embedding_backend,
        )
    controlled_result = {}
    if not args.skip_controlled:
        controlled_result = run_controlled_beta(
            tickets_path=_resolve(args.controlled_tickets),
            gold_path=_resolve(args.controlled_gold),
            repo_cache_dir=_resolve(args.controlled_repo_cache_dir),
            index_cache_dir=_resolve(args.controlled_index_cache_dir),
            output_dir=output_dir,
            top_k=args.top_k,
            embedding_backend=args.controlled_embedding_backend or args.embedding_backend,
            limit=args.controlled_limit,
            offset=args.controlled_offset,
        )
    controlled_summary = summarize_existing_predictions(_resolve(args.controlled_predictions))
    summary = {
        "purpose": "fault_localization_internal_beta_user_facing_check",
        "output_dir": str(output_dir),
        "top_k": args.top_k,
        "embedding_backend": args.embedding_backend,
        "demo": demo_result,
        "controlled_beta": controlled_result,
        "controlled_existing_predictions_summary": controlled_summary,
        "deployment_precheck": build_deployment_precheck(demo_result, controlled_result, controlled_summary),
        "limitations": [
            "This beta check validates UX fields, confidence handling, fallback visibility, runtime, and demo consistency.",
            "It does not prove full-dataset accuracy improvement.",
            "LLM rerank remains optional and is not called by default.",
        ],
    }

    summary_json_path = output_dir / "internal_beta_evaluation.json"
    summary_md_path = output_dir / "internal_beta_evaluation.md"
    precheck_md_path = output_dir / "deployment_precheck_summary.md"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    precheck_md_path.write_text(render_deployment_precheck_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "deployment_precheck_md": str(precheck_md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_demo_beta(
    *,
    tickets_path: Path,
    gold_path: Path,
    repo_path: Path,
    output_dir: Path,
    top_k: int,
    embedding_backend: str,
) -> dict[str, Any]:
    tickets = read_records(tickets_path)
    gold_rows = read_records(gold_path)
    gold_by_id = {_ticket_id(row): row for row in gold_rows}
    code_index = build_code_index(repo_path)

    user_results: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for ticket in tickets:
        started = time.perf_counter()
        validation = validate_localization_request(ticket, repo_path=repo_path, code_index_supplied=True)
        raw_result: dict[str, Any] = {}
        if validation["errors"]:
            user_result = format_user_facing_validation_error(validation)
        else:
            raw_result = localize_ticket(
                ticket,
                code_index=code_index,
                top_k=top_k,
                embedding_backend=embedding_backend,
            )
            user_result = format_user_facing_localization_result(raw_result, validation=validation, top_k=top_k)
        rows.append(build_user_result_row(ticket, user_result, gold_by_id.get(_ticket_id(ticket), {}), time.perf_counter() - started))
        user_results.append(user_result)
        if raw_result:
            raw_results.append(raw_result)

    user_results_path = output_dir / "demo_user_facing_results.jsonl"
    raw_results_path = output_dir / "demo_raw_localization_results.jsonl"
    write_jsonl(user_results_path, user_results)
    write_jsonl(raw_results_path, raw_results)
    return {
        "tickets": str(tickets_path),
        "gold": str(gold_path),
        "repo_path": str(repo_path),
        "user_facing_output": str(user_results_path),
        "raw_output": str(raw_results_path),
        "cases": len(user_results),
        "summary": summarize_result_rows(rows),
        "rows": rows,
    }


def run_controlled_beta(
    *,
    tickets_path: Path,
    gold_path: Path,
    repo_cache_dir: Path,
    index_cache_dir: Path,
    output_dir: Path,
    top_k: int,
    embedding_backend: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    tickets = read_records(tickets_path)[offset : offset + limit]
    gold_rows_all = read_records(gold_path)
    gold_by_id = {_ticket_id(row): row for row in gold_rows_all}
    selected_ids = {_ticket_id(row) for row in tickets if _ticket_id(row)}
    selected_gold_rows = [row for row in gold_rows_all if _ticket_id(row) in selected_ids]
    config = BatchRunConfig(
        tickets_path=tickets_path,
        gold_path=gold_path,
        repo_cache_dir=repo_cache_dir,
        index_cache_dir=index_cache_dir,
        predictions_output=output_dir / "controlled_raw_predictions.jsonl",
        metrics_output=output_dir / "controlled_metrics.json",
        demo_cases_output=None,
        failures_output=output_dir / "controlled_failures.jsonl",
        checkout=False,
        top_k=top_k,
        embedding_backend=embedding_backend,
        progress_every=0,
        checkpoint_every=0,
    )
    raw_results: list[dict[str, Any]] = []
    user_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ticket in tickets:
        started = time.perf_counter()
        ticket_id = _ticket_id(ticket)
        try:
            repo_path = resolve_repository(ticket, config)
            index = load_or_build_index(ticket, repo_path, config)
            validation = validate_localization_request(ticket, repo_path=repo_path, code_index_supplied=True)
            raw_result: dict[str, Any] = {}
            if validation["errors"]:
                user_result = format_user_facing_validation_error(validation)
            else:
                raw_result = localize_ticket(
                    ticket,
                    code_index=index,
                    top_k=top_k,
                    embedding_backend=embedding_backend,
                    repository_proximity=config.repository_proximity,
                )
                raw_result["repo"] = str(ticket.get("repo") or "")
                raw_result["base_commit"] = str(ticket.get("base_commit") or "")
                user_result = format_user_facing_localization_result(raw_result, validation=validation, top_k=top_k)
            rows.append(build_user_result_row(ticket, user_result, gold_by_id.get(ticket_id, {}), time.perf_counter() - started))
            user_results.append(user_result)
            if raw_result:
                raw_results.append(raw_result)
        except Exception as exc:  # noqa: BLE001 - beta report should preserve per-ticket failure detail.
            failures.append(
                {
                    "ticket_id": ticket_id,
                    "repo": str(ticket.get("repo") or ""),
                    "error": str(exc),
                    "runtime_seconds": round(time.perf_counter() - started, 4),
                }
            )

    metrics = evaluate_records(selected_gold_rows, raw_results) if selected_gold_rows and raw_results else {}
    raw_output = output_dir / "controlled_raw_predictions.jsonl"
    user_output = output_dir / "controlled_user_facing_results.jsonl"
    failures_output = output_dir / "controlled_failures.jsonl"
    metrics_output = output_dir / "controlled_metrics.json"
    write_jsonl(raw_output, raw_results)
    write_jsonl(user_output, user_results)
    write_jsonl(failures_output, failures)
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "tickets": str(tickets_path),
        "gold": str(gold_path),
        "repo_cache_dir": str(repo_cache_dir),
        "index_cache_dir": str(index_cache_dir),
        "limit": limit,
        "offset": offset,
        "embedding_backend": embedding_backend,
        "user_facing_output": str(user_output),
        "raw_output": str(raw_output),
        "metrics_output": str(metrics_output),
        "failures_output": str(failures_output),
        "cases": len(user_results),
        "failures": len(failures),
        "summary": summarize_result_rows(rows),
        "metrics": metrics,
        "rows": rows,
        "failure_rows": failures,
    }


def build_user_result_row(
    ticket: dict[str, Any],
    user_result: dict[str, Any],
    gold: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    ticket_id = _ticket_id(ticket)
    summary = user_result.get("summary") if isinstance(user_result.get("summary"), dict) else {}
    validation = user_result.get("input_validation") if isinstance(user_result.get("input_validation"), dict) else {}
    gold_files = _gold_files(gold)
    predicted_files = [row.get("file_path", "") for row in user_result.get("top_k_suspicious_files", [])]
    warnings = list(user_result.get("warnings") or [])
    fallback_message = str(summary.get("fallback_message") or "")
    return {
        "ticket_id": ticket_id,
        "repo": str(ticket.get("repo") or gold.get("repo") or ""),
        "status": user_result.get("status", ""),
        "confidence_level": summary.get("confidence_level", ""),
        "manual_review": summary.get("should_manual_review", True),
        "patch_generation": summary.get("recommend_patch_generation", False),
        "patch_generation_policy": summary.get("patch_generation_policy", ""),
        "runtime_seconds": round(duration, 4),
        "validation_status": validation.get("status", ""),
        "validation_warnings": list(validation.get("warnings") or []),
        "warning_categories": classify_warnings(warnings, fallback_message),
        "warning_count": len(warnings),
        "fallback_used": bool(summary.get("fallback_used")),
        "fallback_message": fallback_message,
        "gold_files": gold_files,
        "top_files": predicted_files,
        "hit_rank": _first_relevant_rank(predicted_files, gold_files),
        "has_gold": bool(gold_files),
    }


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence = Counter(str(row.get("confidence_level") or "unknown") for row in rows)
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    validation_counts = Counter(str(row.get("validation_status") or "unknown") for row in rows)
    warning_categories = Counter(
        category for row in rows for category in list(row.get("warning_categories") or [])
    )
    rows_with_gold = sum(1 for row in rows if row.get("has_gold"))
    hit_at_top_k = sum(1 for row in rows if row.get("has_gold") and row.get("hit_rank") is not None)
    manual_review = sum(1 for row in rows if row.get("manual_review"))
    patch_generation = sum(1 for row in rows if row.get("patch_generation"))
    warning_rows = sum(1 for row in rows if row.get("warning_count"))
    fallback_rows = sum(1 for row in rows if row.get("fallback_used"))
    invalid_input_rows = sum(1 for row in rows if row.get("status") == "invalid_input")
    runtimes = [float(row.get("runtime_seconds") or 0.0) for row in rows]
    return {
        "cases": len(rows),
        "rows_with_gold": rows_with_gold,
        "confidence_counts": dict(confidence),
        "status_counts": dict(status_counts),
        "validation_status_counts": dict(validation_counts),
        "hit_at_top_k": hit_at_top_k,
        "hit_at_top_k_rate": round(hit_at_top_k / rows_with_gold, 4) if rows_with_gold else 0.0,
        "manual_review_cases": manual_review,
        "manual_review_rate": round(manual_review / len(rows), 4) if rows else 0.0,
        "patch_generation_recommended_cases": patch_generation,
        "patch_generation_recommended_rate": round(patch_generation / len(rows), 4) if rows else 0.0,
        "warning_rows": warning_rows,
        "fallback_rows": fallback_rows,
        "invalid_input_rows": invalid_input_rows,
        "warning_category_counts": dict(warning_categories),
        "average_runtime_seconds": round(sum(runtimes) / len(runtimes), 4) if runtimes else 0.0,
        "max_runtime_seconds": round(max(runtimes), 4) if runtimes else 0.0,
    }


def summarize_existing_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False}
    rows = read_records(path)
    confidence = Counter(str(row.get("confidence_level") or "unknown") for row in rows)
    fallback = sum(1 for row in rows if row.get("fallback_message") or row.get("warnings"))
    manual_review = sum(1 for row in rows if row.get("should_manual_review"))
    return {
        "path": str(path),
        "available": True,
        "rows": len(rows),
        "confidence_counts": dict(confidence),
        "rows_with_warning_or_fallback": fallback,
        "manual_review_cases": manual_review,
        "note": "Existing predictions are summarized only; this script does not rerun LLM rerank.",
    }


def render_markdown(summary: dict[str, Any]) -> str:
    demo = summary.get("demo") or {}
    controlled = summary.get("controlled_beta") or {}
    demo_summary = demo.get("summary") or {}
    controlled_summary = controlled.get("summary") or {}
    lines = [
        "# Fault Localization Internal Beta Evaluation",
        "",
        "This check focuses on user-facing readiness: stable output shape, confidence handling, warning/fallback visibility, runtime, and patch handoff safety.",
        "",
        "## Demo Summary",
        "",
        f"- Demo cases: {demo.get('cases', 0)}",
        f"- Hit@Top-k: {demo_summary.get('hit_at_top_k', 0)}/{demo_summary.get('rows_with_gold', 0)}",
        f"- Confidence counts: `{json.dumps(demo_summary['confidence_counts'], ensure_ascii=False)}`",
        f"- Manual-review cases: {demo_summary['manual_review_cases']}",
        f"- Patch-generation recommended cases: {demo_summary['patch_generation_recommended_cases']}",
        f"- Average runtime seconds: {demo_summary['average_runtime_seconds']}",
        "",
        "| Ticket | Confidence | Hit rank | Manual review | Patch generation | Runtime | Top files |",
        "|---|---|---:|---|---|---:|---|",
    ]
    lines.extend(result_row_table_lines(demo.get("rows") or []))
    lines.extend(
        [
            "",
            "## Controlled Beta Summary",
            "",
            f"- Controlled cases completed: {controlled.get('cases', 0)}",
            f"- Controlled failures: {controlled.get('failures', 0)}",
            f"- Hit@Top-k: {controlled_summary.get('hit_at_top_k', 0)}/{controlled_summary.get('rows_with_gold', 0)}",
            f"- Confidence counts: `{json.dumps(controlled_summary.get('confidence_counts', {}), ensure_ascii=False)}`",
            f"- Manual-review rate: {controlled_summary.get('manual_review_rate', 0.0):.4f}",
            f"- Patch-generation eligible rate: {controlled_summary.get('patch_generation_recommended_rate', 0.0):.4f}",
            f"- Warning rows: {controlled_summary.get('warning_rows', 0)}",
            f"- Fallback rows: {controlled_summary.get('fallback_rows', 0)}",
            f"- Warning categories: `{json.dumps(controlled_summary.get('warning_category_counts', {}), ensure_ascii=False)}`",
            f"- Average runtime seconds: {controlled_summary.get('average_runtime_seconds', 0.0)}",
            "",
            "| Ticket | Repo | Confidence | Hit rank | Manual review | Patch generation | Warnings | Runtime |",
            "|---|---|---|---:|---|---|---:|---:|",
        ]
    )
    for row in controlled.get("rows") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(row["ticket_id"]),
                    escape_md(row.get("repo", "")),
                    escape_md(row["confidence_level"]),
                    str(row["hit_rank"] or "miss"),
                    "yes" if row["manual_review"] else "no",
                    "yes" if row["patch_generation"] else "no",
                    str(row["warning_count"]),
                    str(row["runtime_seconds"]),
                ]
            )
            + " |"
        )
    existing = summary.get("controlled_existing_predictions_summary") or {}
    lines.extend(
        [
            "",
            "## Existing Controlled Prediction Summary",
            "",
            f"- Available: {existing.get('available', False)}",
            f"- Rows: {existing.get('rows', 0)}",
            f"- Confidence counts: `{json.dumps(existing.get('confidence_counts', {}), ensure_ascii=False)}`",
            f"- Rows with warning or fallback: {existing.get('rows_with_warning_or_fallback', 0)}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def render_deployment_precheck_markdown(summary: dict[str, Any]) -> str:
    demo_summary = ((summary.get("demo") or {}).get("summary") or {})
    controlled_summary = ((summary.get("controlled_beta") or {}).get("summary") or {})
    metrics = (summary.get("controlled_beta") or {}).get("metrics") or {}
    precheck = summary.get("deployment_precheck") or {}
    lines = [
        "# Fault Localization Deployment Precheck Summary",
        "",
        "| Check | Result | Notes |",
        "|---|---|---|",
        f"| Demo user-facing flow | {demo_summary.get('cases', 0)} cases | high/medium/low and invalid-input cases are reportable |",
        f"| Controlled beta flow | {controlled_summary.get('cases', 0)} cases | retrieval-first, no Ollama call by default |",
        f"| Controlled Hit@Top-k | {controlled_summary.get('hit_at_top_k', 0)}/{controlled_summary.get('rows_with_gold', 0)} | use as beta readiness signal, not full accuracy claim |",
        f"| Controlled Top-1 | {float(metrics.get('top_1_accuracy', 0.0)):.4f} | small subset only |",
        f"| Controlled Top-3 | {float(metrics.get('top_3_accuracy', 0.0)):.4f} | small subset only |",
        f"| Controlled Top-5 | {float(metrics.get('top_5_accuracy', 0.0)):.4f} | small subset only |",
        f"| Controlled MRR | {float(metrics.get('mrr', 0.0)):.4f} | small subset only |",
        f"| Manual review rate | {controlled_summary.get('manual_review_rate', 0.0):.4f} | lower confidence results should not auto-patch |",
        f"| Patch eligible rate | {controlled_summary.get('patch_generation_recommended_rate', 0.0):.4f} | high-confidence Top-1 only |",
        f"| Warning rows | {controlled_summary.get('warning_rows', 0)} | categories: `{json.dumps(controlled_summary.get('warning_category_counts', {}), ensure_ascii=False)}` |",
        f"| Deployment recommendation | {precheck.get('recommendation', 'unknown')} | {precheck.get('reason', '')} |",
        "",
        "## Do Not Overclaim",
        "",
        "- This is an internal beta readiness check, not a full production deployment evaluation.",
        "- The controlled beta run does not prove LLM rerank accuracy because Ollama is not called by default.",
        "- Patch generation remains suggestion-oriented and must respect manual-review flags.",
        "",
    ]
    return "\n".join(lines)


def result_row_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(row["ticket_id"]),
                    escape_md(row["confidence_level"]),
                    str(row["hit_rank"] or "miss" if row.get("has_gold") else "n/a"),
                    "yes" if row["manual_review"] else "no",
                    "yes" if row["patch_generation"] else "no",
                    str(row["runtime_seconds"]),
                    escape_md(", ".join(row["top_files"])),
                ]
            )
            + " |"
        )
    return lines


def classify_warnings(warnings: list[Any], fallback_message: str) -> list[str]:
    categories: list[str] = []
    if fallback_message:
        categories.append("fallback_message")
    for raw in warnings:
        warning = str(raw)
        lowered = warning.lower()
        if "input validation" in lowered or "stack trace" in lowered or "source path" in lowered:
            categories.append("input_validation")
        elif "llm reranking failed" in lowered or "optional llm" in lowered:
            categories.append("llm_fallback")
        elif "no localization candidates" in lowered:
            categories.append("no_candidates")
        else:
            categories.append("other_warning")
    return sorted(set(categories))


def build_deployment_precheck(
    demo: dict[str, Any],
    controlled: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    controlled_summary = controlled.get("summary") or {}
    controlled_cases = int(controlled.get("cases") or 0)
    controlled_failures = int(controlled.get("failures") or 0)
    warning_rows = int(controlled_summary.get("warning_rows") or 0)
    if controlled_cases >= 20 and controlled_failures == 0:
        recommendation = "internal_beta_ready"
        reason = "Controlled beta completed at least 20 retrieval-first cases without runner failures."
    elif controlled_cases > 0:
        recommendation = "internal_beta_limited"
        reason = "Controlled beta produced results but did not meet the 20-case no-failure target."
    else:
        recommendation = "not_ready"
        reason = "Controlled beta did not run."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "demo_cases": demo.get("cases", 0),
        "controlled_cases": controlled_cases,
        "controlled_failures": controlled_failures,
        "controlled_warning_rows": warning_rows,
        "existing_llm_prediction_rows": existing.get("rows", 0),
        "existing_llm_warning_or_fallback_rows": existing.get("rows_with_warning_or_fallback", 0),
    }


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return [row for row in value["records"] if isinstance(row, dict)]
    return [value] if isinstance(value, dict) else []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("id") or row.get("bug_id") or "")


def _gold_files(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "files", "file_path", "file"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return [_normalize_path(value) for value in values if _normalize_path(value)]


def _first_relevant_rank(ranked_files: list[str], gold_files: list[str]) -> int | None:
    for rank, file_path in enumerate(ranked_files, start=1):
        if any(_file_matches(file_path, gold) for gold in gold_files):
            return rank
    return None


def _file_matches(predicted: str, gold: str) -> bool:
    left = _normalize_path(predicted)
    right = _normalize_path(gold)
    return bool(left and right and (left == right or left.endswith("/" + right) or right.endswith("/" + left)))


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
