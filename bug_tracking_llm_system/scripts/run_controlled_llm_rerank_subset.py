#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.evaluate_fault_localization import evaluate_records, read_records
from scripts.run_swebench_lite_fault_localization import BatchRunConfig, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled LLM rerank experiment on a small subset of SWE-bench Lite "
            "Top-5 misses from a retrieval baseline."
        )
    )
    parser.add_argument(
        "--baseline-pred",
        default="reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/test_predictions.jsonl",
    )
    parser.add_argument("--dataset-dir", default="data/fault_localization/swebench_lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tickets", default=None)
    parser.add_argument("--gold", default=None)
    parser.add_argument("--repo-cache-dir", default=None)
    parser.add_argument("--index-cache-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="reports/fault_localization/swebench_lite_llm_rerank_controlled_top5_miss_20",
    )
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-backend", default="tfidf-sbert-rerank")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow-sbert-download", action="store_true")
    parser.add_argument("--sbert-cache-dir", default=None)
    parser.add_argument("--llm-candidate-k", type=int, default=10)
    parser.add_argument("--llm-cache-dir", default=None)
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the selected subset and command-ready summary; do not call the LLM.",
    )
    parser.add_argument(
        "--retrieval-probe-pred",
        default=None,
        help="Optional retrieval-only prediction JSONL, typically Top-10, for candidate-pool diagnostics.",
    )
    parser.add_argument(
        "--no-balance-by-repo",
        action="store_true",
        help="Select misses in baseline order instead of round-robin by repository.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    split = args.split
    baseline_pred_path = _resolve(args.baseline_pred)
    tickets_path = _resolve(args.tickets) if args.tickets else dataset_dir / f"{split}_tickets.jsonl"
    gold_path = _resolve(args.gold) if args.gold else dataset_dir / f"{split}_gold.jsonl"
    output_dir = _resolve(args.output_dir)
    repo_cache_dir = _resolve(args.repo_cache_dir) if args.repo_cache_dir else dataset_dir / "repos"
    index_cache_dir = _resolve(args.index_cache_dir) if args.index_cache_dir else dataset_dir / "indexes"
    sbert_cache_dir = _resolve(args.sbert_cache_dir) if args.sbert_cache_dir else dataset_dir / "embedding_cache"
    llm_cache_dir = _resolve(args.llm_cache_dir) if args.llm_cache_dir else output_dir / "llm_rerank_cache"

    gold_rows = read_records(gold_path)
    baseline_predictions = read_records(baseline_pred_path)
    tickets = read_records(tickets_path)
    selected = select_top5_misses(
        gold_rows,
        baseline_predictions,
        subset_size=args.subset_size,
        top_k=args.top_k,
        balance_by_repo=not args.no_balance_by_repo,
    )
    selected_ids = [row["ticket_id"] for row in selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tickets_path = output_dir / "selected_tickets.jsonl"
    selected_ids_path = output_dir / "selected_ticket_ids.json"
    predictions_output = output_dir / f"{split}_predictions.jsonl"
    metrics_output = output_dir / f"{split}_metrics.json"
    demo_cases_output = output_dir / f"{split}_demo_cases.json"
    failures_output = output_dir / f"{split}_failures.jsonl"
    summary_json = output_dir / "controlled_llm_rerank_summary.json"
    summary_md = output_dir / "controlled_llm_rerank_summary.md"

    ticket_by_id = {_ticket_id(row): row for row in tickets if _ticket_id(row)}
    selected_tickets = [ticket_by_id[ticket_id] for ticket_id in selected_ids if ticket_id in ticket_by_id]
    _write_jsonl(selected_tickets_path, selected_tickets)
    selected_ids_path.write_text(json.dumps(selected_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_summary: dict[str, Any] | None = None
    if not args.prepare_only:
        run_summary = run_batch(
            BatchRunConfig(
                tickets_path=tickets_path,
                gold_path=gold_path,
                repo_cache_dir=repo_cache_dir,
                index_cache_dir=index_cache_dir,
                predictions_output=predictions_output,
                metrics_output=metrics_output,
                demo_cases_output=demo_cases_output,
                failures_output=failures_output,
                ticket_ids=set(selected_ids),
                checkout=True,
                top_k=args.top_k,
                embedding_backend=args.embedding_backend,
                sbert_model=args.sbert_model,
                sbert_local_files_only=not args.allow_sbert_download,
                sbert_cache_dir=sbert_cache_dir,
                llm_rerank=True,
                llm_candidate_k=args.llm_candidate_k,
                llm_cache_dir=llm_cache_dir,
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url,
                ollama_timeout=args.ollama_timeout,
                demo_case_limit=5,
                resume=args.resume,
                progress_every=args.progress_every,
                checkpoint_every=args.checkpoint_every,
            )
        )

    llm_predictions = read_records(predictions_output) if predictions_output.exists() else []
    retrieval_probe_predictions = read_records(_resolve(args.retrieval_probe_pred)) if args.retrieval_probe_pred else []
    comparison = build_comparison(
        selected_ids=selected_ids,
        gold_rows=gold_rows,
        baseline_predictions=baseline_predictions,
        llm_predictions=llm_predictions,
        retrieval_probe_predictions=retrieval_probe_predictions,
        top_k=args.top_k,
    )
    summary = {
        "purpose": "controlled_llm_rerank_on_top5_misses",
        "baseline_predictions": str(baseline_pred_path),
        "gold": str(gold_path),
        "tickets": str(tickets_path),
        "output_dir": str(output_dir),
        "subset_size_requested": args.subset_size,
        "subset_size_selected": len(selected_ids),
        "selected_ticket_ids": selected_ids,
        "selected_tickets_output": str(selected_tickets_path),
        "prepare_only": args.prepare_only,
        "run_summary": run_summary,
        "comparison": comparison,
        "retrieval_probe_predictions": str(_resolve(args.retrieval_probe_pred)) if args.retrieval_probe_pred else "",
        "method": {
            "top_k": args.top_k,
            "embedding_backend": args.embedding_backend,
            "llm_candidate_k": args.llm_candidate_k,
            "llm_cache_dir": str(llm_cache_dir),
            "ollama_model": args.ollama_model,
            "ollama_timeout": args.ollama_timeout,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary, selected), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_json), "summary_md": str(summary_md)}, ensure_ascii=False, indent=2))


def select_top5_misses(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    subset_size: int,
    top_k: int,
    balance_by_repo: bool,
) -> list[dict[str, Any]]:
    pred_by_id = {_ticket_id(row): row for row in pred_rows if _ticket_id(row)}
    misses: list[dict[str, Any]] = []
    for gold in gold_rows:
        ticket_id = _ticket_id(gold)
        pred = pred_by_id.get(ticket_id)
        if not ticket_id or pred is None:
            continue
        gold_files = _gold_files(gold)
        rank = _first_relevant_rank(_candidate_files(pred), gold_files)
        if gold_files and (rank is None or rank > top_k):
            misses.append(
                {
                    "ticket_id": ticket_id,
                    "repo": _repo_value(gold) or _repo_value(pred) or "unknown",
                    "gold_files": gold_files,
                    "baseline_hit_rank": rank,
                    "baseline_top_files": _candidate_files(pred)[:top_k],
                }
            )
    if not balance_by_repo:
        return misses[:subset_size]

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in misses:
        by_repo.setdefault(str(row["repo"]), []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < subset_size and any(by_repo.values()):
        for repo in sorted(by_repo):
            rows = by_repo[repo]
            if rows:
                selected.append(rows.pop(0))
                if len(selected) >= subset_size:
                    break
    return selected


def build_comparison(
    *,
    selected_ids: list[str],
    gold_rows: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    llm_predictions: list[dict[str, Any]],
    retrieval_probe_predictions: list[dict[str, Any]] | None = None,
    top_k: int,
) -> dict[str, Any]:
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    baseline_by_id = {_ticket_id(row): row for row in baseline_predictions if _ticket_id(row)}
    llm_by_id = {_ticket_id(row): row for row in llm_predictions if _ticket_id(row)}
    probe_by_id = {_ticket_id(row): row for row in (retrieval_probe_predictions or []) if _ticket_id(row)}
    selected_gold = [gold_by_id[ticket_id] for ticket_id in selected_ids if ticket_id in gold_by_id]
    selected_baseline = [baseline_by_id[ticket_id] for ticket_id in selected_ids if ticket_id in baseline_by_id]
    selected_llm = [llm_by_id[ticket_id] for ticket_id in selected_ids if ticket_id in llm_by_id]
    selected_probe = [probe_by_id[ticket_id] for ticket_id in selected_ids if ticket_id in probe_by_id]
    baseline_metrics = evaluate_records(selected_gold, selected_baseline) if selected_baseline else {}
    llm_metrics = evaluate_records(selected_gold, selected_llm) if selected_llm else {}
    probe_metrics = evaluate_records(selected_gold, selected_probe) if selected_probe else {}

    rows: list[dict[str, Any]] = []
    improved = 0
    worsened = 0
    unchanged = 0
    for ticket_id in selected_ids:
        gold = gold_by_id.get(ticket_id, {})
        baseline = baseline_by_id.get(ticket_id, {})
        llm = llm_by_id.get(ticket_id, {})
        probe = probe_by_id.get(ticket_id, {})
        gold_files = _gold_files(gold)
        baseline_rank = _first_relevant_rank(_candidate_files(baseline), gold_files)
        llm_rank = _first_relevant_rank(_candidate_files(llm), gold_files)
        probe_rank = _first_relevant_rank(_candidate_files(probe), gold_files)
        if _rank_value(llm_rank, top_k) < _rank_value(baseline_rank, top_k):
            improved += 1
            change = "improved"
        elif _rank_value(llm_rank, top_k) > _rank_value(baseline_rank, top_k):
            worsened += 1
            change = "worsened"
        else:
            unchanged += 1
            change = "unchanged"
        rows.append(
            {
                "ticket_id": ticket_id,
                "repo": _repo_value(gold) or _repo_value(baseline) or _repo_value(llm) or _repo_value(probe) or "unknown",
                "gold_files": gold_files,
                "baseline_hit_rank": baseline_rank,
                "llm_hit_rank": llm_rank,
                "retrieval_probe_hit_rank": probe_rank,
                "change": change,
                "baseline_top1": (_candidate_files(baseline) or [""])[0],
                "llm_top1": (_candidate_files(llm) or [""])[0],
                "llm_top_files": _candidate_files(llm)[:top_k],
                "retrieval_probe_top_files": _candidate_files(probe),
            }
        )
    return {
        "baseline_metrics": baseline_metrics,
        "llm_metrics": llm_metrics,
        "retrieval_probe_metrics": probe_metrics,
        "tickets_compared": len(selected_llm),
        "retrieval_probe_tickets_compared": len(selected_probe),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    comparison = summary["comparison"]
    baseline_metrics = comparison.get("baseline_metrics") or {}
    llm_metrics = comparison.get("llm_metrics") or {}
    lines = [
        "# Controlled LLM Rerank Subset Report",
        "",
        "This report evaluates LLM reranking only on a small Top-5 miss subset. It does not run the LLM over the full 300-ticket split.",
        "",
        "## Setup",
        "",
        f"- Selected tickets: {summary['subset_size_selected']}",
        f"- Prepare only: {summary['prepare_only']}",
        f"- Embedding backend: {summary['method']['embedding_backend']}",
        f"- LLM candidate pool: Top-{summary['method']['llm_candidate_k']}",
        f"- LLM model: {summary['method']['ollama_model']}",
        f"- Cache dir: `{summary['method']['llm_cache_dir']}`",
        "",
        "## Metrics",
        "",
        "| Method | Rows | Top-1 | Top-3 | Top-5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        _metric_row("Retrieval baseline on selected misses", baseline_metrics),
        _metric_row("Controlled LLM rerank", llm_metrics),
        _metric_row("Retrieval-only candidate-pool probe", comparison.get("retrieval_probe_metrics") or {}),
        "",
        "## Change Summary",
        "",
        f"- Compared tickets: {comparison['tickets_compared']}",
        f"- Retrieval probe tickets: {comparison['retrieval_probe_tickets_compared']}",
        f"- Improved: {comparison['improved']}",
        f"- Worsened: {comparison['worsened']}",
        f"- Unchanged: {comparison['unchanged']}",
        "",
        "## Selected Tickets",
        "",
        "| Ticket | Repo | Gold files | Baseline Top-5 |",
        "|---|---|---|---|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row["ticket_id"]),
                    _escape_md(row["repo"]),
                    _escape_md(", ".join(row["gold_files"])),
                    _escape_md(", ".join(row["baseline_top_files"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-ticket Comparison",
            "",
            "| Ticket | Repo | Baseline rank | LLM rank | Probe rank | Change | LLM Top-1 |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in comparison["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row["ticket_id"]),
                    _escape_md(row["repo"]),
                    str(row["baseline_hit_rank"] or "miss"),
                    str(row["llm_hit_rank"] or "miss"),
                    str(row["retrieval_probe_hit_rank"] or "miss"),
                    _escape_md(row["change"]),
                    _escape_md(row["llm_top1"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_row(label: str, metrics: dict[str, Any]) -> str:
    if not metrics:
        return f"| {label} | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |"
    return (
        f"| {label} | {metrics.get('rows_with_file_ground_truth', 0)} | "
        f"{float(metrics.get('top_1_accuracy', 0.0)):.4f} | "
        f"{float(metrics.get('top_3_accuracy', 0.0)):.4f} | "
        f"{float(metrics.get('top_5_accuracy', 0.0)):.4f} | "
        f"{float(metrics.get('mrr', 0.0)):.4f} |"
    )


def _rank_value(rank: int | None, top_k: int) -> int:
    return rank if rank is not None else top_k + 1


def _candidate_files(row: dict[str, Any]) -> list[str]:
    candidates = row.get("localized_candidates")
    if not isinstance(candidates, list):
        candidates = row.get("localized_files")
    if not isinstance(candidates, list):
        return []
    files: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            file_path = _normalize_path(str(candidate.get("file_path") or candidate.get("file") or ""))
            if file_path and file_path not in files:
                files.append(file_path)
    return files


def _gold_files(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("fixed_files", "modified_files", "changed_files", "bug_fix_files", "files", "file_paths"):
        value = row.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return _unique_paths(values)


def _first_relevant_rank(ranked_files: list[str], gold_files: list[str]) -> int | None:
    for rank, file_path in enumerate(ranked_files, start=1):
        if any(_file_matches(file_path, gold) for gold in gold_files):
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


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
