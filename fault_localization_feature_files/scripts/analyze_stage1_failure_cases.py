#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.evaluate_fault_localization import (  # noqa: E402
    _file_matches,
    _gold_files,
    _repository_name,
    _stage1_candidate_files,
    _ticket_id,
    read_records,
)
from utils.fault_localization import CodeIndex, rank_code_chunks  # noqa: E402


FAILURE_STAGE_LABELS = {
    "index_coverage_failure": "正確檔案未進入Code Index",
    "initial_retrieval_miss": "正確檔案未進入TF-IDF前50名",
    "reranker_demotion": "正確檔案在TF-IDF前50名，但被SBERT排出前20名",
    "mixed_retrieval_and_rerank": "同一Ticket同時包含初步檢索與重新排序遺漏",
    "analysis_unavailable": "缺少Code Index或TF-IDF前50資料，暫時無法判定",
}

PRIMARY_CATEGORY_LABELS = {
    **FAILURE_STAGE_LABELS,
    "cross_file_partial_miss": "只找回部分正確檔案",
}


@dataclass(frozen=True)
class SplitSpec:
    label: str
    role: str
    tickets_path: Path
    gold_path: Path
    predictions_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Stage-1 partial and missed Top-20 predictions by index, "
            "TF-IDF Top-50, and SBERT reranking stage."
        )
    )
    parser.add_argument("--validation-tickets", required=True)
    parser.add_argument("--validation-gold", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--holdout-tickets")
    parser.add_argument("--holdout-gold")
    parser.add_argument("--holdout-predictions")
    parser.add_argument(
        "--index-cache-dir",
        action="append",
        required=True,
        help="Code-index directory. Repeat to add read-only fallback caches.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/fault_localization/stage1_next_iteration",
    )
    parser.add_argument(
        "--skip-tfidf-top50",
        action="store_true",
        help="Only inspect index coverage; leaves retrieval-stage cases unresolved.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--manual-review-sample", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive.")
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive.")
    if args.manual_review_sample <= 0:
        raise ValueError("manual-review-sample must be positive.")

    specs = [
        SplitSpec(
            label="validation",
            role="method_selection",
            tickets_path=Path(args.validation_tickets),
            gold_path=Path(args.validation_gold),
            predictions_path=Path(args.validation_predictions),
        )
    ]
    optional_holdout = (
        args.holdout_tickets,
        args.holdout_gold,
        args.holdout_predictions,
    )
    if any(optional_holdout) and not all(optional_holdout):
        raise ValueError(
            "holdout-tickets, holdout-gold, and holdout-predictions must be supplied together."
        )
    if all(optional_holdout):
        specs.append(
            SplitSpec(
                label="frozen_holdout_v1_posthoc",
                role="posthoc_description_only",
                tickets_path=Path(args.holdout_tickets),
                gold_path=Path(args.holdout_gold),
                predictions_path=Path(args.holdout_predictions),
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dirs = [Path(value) for value in args.index_cache_dir]
    for cache_dir in index_cache_dirs:
        if not cache_dir.is_dir():
            raise FileNotFoundError(cache_dir)

    all_cases: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    source_artifacts: dict[str, dict[str, str]] = {}
    for spec in specs:
        for path in (spec.tickets_path, spec.gold_path, spec.predictions_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        cache_path = output_dir / "tfidf_top50_cache" / f"{spec.label}.jsonl"
        cases, summary = analyze_split(
            spec,
            index_cache_dirs=index_cache_dirs,
            top50_cache_path=cache_path,
            compute_tfidf_top50=not args.skip_tfidf_top50,
            checkpoint_every=args.checkpoint_every,
            max_workers=args.max_workers,
        )
        all_cases.extend(cases)
        split_summaries[spec.label] = summary
        source_artifacts[spec.label] = {
            "role": spec.role,
            "tickets": str(spec.tickets_path),
            "gold": str(spec.gold_path),
            "predictions": str(spec.predictions_path),
            "tfidf_top50_cache": str(cache_path),
        }

    output = {
        "analysis_scope": "Stage-1 Top-20 partial and missed cases",
        "classification_method": (
            "Gold-file index coverage, TF-IDF Top-50 membership, then final SBERT Top-20 membership"
        ),
        "data_policy": {
            "validation": "May be used for v2 method comparison after settings are frozen.",
            "frozen_holdout_v1_posthoc": (
                "Post-hoc description of v1 only; do not tune v2 or claim a new independent v2 result on it."
            ),
        },
        "sources": source_artifacts,
        "split_summaries": split_summaries,
        "combined_summary": summarize_cases(all_cases),
        "cases": all_cases,
    }

    json_path = output_dir / "failure_analysis.json"
    csv_path = output_dir / "failure_cases.csv"
    markdown_path = output_dir / "failure_analysis_zh.md"
    review_path = output_dir / "manual_review_sample.csv"
    write_json(json_path, output)
    write_cases_csv(csv_path, all_cases)
    markdown_path.write_text(render_markdown(output), encoding="utf-8")
    write_cases_csv(
        review_path,
        select_manual_review_sample(all_cases, size=args.manual_review_sample),
    )
    print(
        json.dumps(
            {
                "failure_analysis": str(json_path),
                "failure_cases": str(csv_path),
                "failure_report": str(markdown_path),
                "manual_review_sample": str(review_path),
                "split_summaries": split_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def analyze_split(
    spec: SplitSpec,
    *,
    index_cache_dirs: list[Path],
    top50_cache_path: Path,
    compute_tfidf_top50: bool,
    checkpoint_every: int,
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tickets = read_records(spec.tickets_path)
    gold_rows = read_records(spec.gold_path)
    predictions = read_records(spec.predictions_path)
    tickets_by_id = {_ticket_id(row): row for row in tickets if _ticket_id(row)}
    predictions_by_id = {_ticket_id(row): row for row in predictions if _ticket_id(row)}
    gold_by_id = {_ticket_id(row): row for row in gold_rows if _ticket_id(row)}
    if set(tickets_by_id) != set(gold_by_id):
        raise ValueError(f"{spec.label}: ticket and gold ID sets do not match.")
    if not set(gold_by_id).issubset(predictions_by_id):
        missing = sorted(set(gold_by_id) - set(predictions_by_id))
        raise ValueError(f"{spec.label}: predictions missing IDs: {', '.join(missing[:5])}")

    failure_ids: list[str] = []
    preliminary: dict[str, tuple[str, list[str], list[str]]] = {}
    for current_id, gold in gold_by_id.items():
        gold_files = _gold_files(gold)
        final_files = _stage1_candidate_files(predictions_by_id[current_id])[:20]
        recovered = matching_gold_files(gold_files, final_files)
        if len(recovered) == len(gold_files):
            continue
        outcome = "partial_recall" if recovered else "miss"
        failure_ids.append(current_id)
        preliminary[current_id] = (outcome, gold_files, final_files)

    top50_cache = read_jsonl_by_id(top50_cache_path)
    ids_to_compute = [
        current_id
        for current_id in failure_ids
        if current_id not in top50_cache
        or (
            compute_tfidf_top50
            and not bool(top50_cache[current_id].get("tfidf_top50_computed"))
        )
    ]
    if ids_to_compute:
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    build_top50_cache_row,
                    tickets_by_id[current_id],
                    index_cache_dirs=index_cache_dirs,
                    compute_tfidf_top50=compute_tfidf_top50,
                ): current_id
                for current_id in ids_to_compute
            }
            for future in as_completed(futures):
                current_id = futures[future]
                top50_cache[current_id] = future.result()
                completed += 1
                print(
                    f"[{spec.label}] computed TF-IDF Top-50 "
                    f"{completed}/{len(ids_to_compute)} {current_id}",
                    flush=True,
                )
                if completed % checkpoint_every == 0:
                    write_jsonl(top50_cache_path, top50_cache.values())
        write_jsonl(top50_cache_path, top50_cache.values())

    cases: list[dict[str, Any]] = []
    for current_id in failure_ids:
        outcome, gold_files, final_files = preliminary[current_id]
        cases.append(
            classify_case(
                split_label=spec.label,
                split_role=spec.role,
                ticket=tickets_by_id[current_id],
                gold=gold_by_id[current_id],
                prediction=predictions_by_id[current_id],
                outcome=outcome,
                gold_files=gold_files,
                final_files=final_files,
                cache_row=top50_cache[current_id],
            )
        )
    summary = summarize_cases(cases)
    summary.update(
        {
            "role": spec.role,
            "tickets": len(gold_rows),
            "full_recall_rows": len(gold_rows) - len(cases),
        }
    )
    return cases, summary


def build_top50_cache_row(
    ticket: dict[str, Any],
    *,
    index_cache_dirs: list[Path],
    compute_tfidf_top50: bool,
) -> dict[str, Any]:
    current_id = _ticket_id(ticket)
    index_path = find_index_path(ticket, index_cache_dirs)
    if index_path is None:
        return {
            "ticket_id": current_id,
            "repo": _repository_name(ticket),
            "base_commit": str(ticket.get("base_commit") or ""),
            "index_path": "",
            "index_available": False,
            "indexed_files": [],
            "tfidf_top50": [],
            "tfidf_top50_computed": False,
        }

    index = CodeIndex.load(index_path)
    indexed_files = sorted({chunk.file_path for chunk in index.chunks})
    top50: list[dict[str, Any]] = []
    if compute_tfidf_top50:
        ranked, backend = rank_code_chunks(
            ticket,
            index.chunks,
            top_k=min(50, len(indexed_files)),
            embedding_backend="tfidf",
            semantic_candidate_k=50,
            file_aggregation=True,
            advanced_file_aggregation=False,
            generic_routing=False,
            domain_path_routing=False,
            repository_proximity=False,
        )
        top50 = [
            {
                "rank": rank,
                "file_path": candidate.chunk.file_path,
                "score": round(float(candidate.score), 6),
            }
            for rank, candidate in enumerate(ranked, start=1)
        ]
    else:
        backend = "not_computed"
    return {
        "ticket_id": current_id,
        "repo": _repository_name(ticket),
        "base_commit": str(ticket.get("base_commit") or ""),
        "index_path": str(index_path),
        "index_available": True,
        "indexed_file_count": len(indexed_files),
        "indexed_files": indexed_files,
        "tfidf_top50": top50,
        "tfidf_top50_computed": compute_tfidf_top50,
        "tfidf_backend": backend,
    }


def classify_case(
    *,
    split_label: str,
    split_role: str,
    ticket: dict[str, Any],
    gold: dict[str, Any],
    prediction: dict[str, Any],
    outcome: str,
    gold_files: list[str],
    final_files: list[str],
    cache_row: dict[str, Any],
) -> dict[str, Any]:
    indexed_files = list(cache_row.get("indexed_files") or [])
    top50_rows = list(cache_row.get("tfidf_top50") or [])
    top50_files = [str(row.get("file_path") or "") for row in top50_rows]
    top50_computed = bool(cache_row.get("tfidf_top50_computed"))
    evidence: list[dict[str, Any]] = []
    for gold_file in gold_files:
        final_rank = matching_rank(gold_file, final_files)
        index_present = any(_file_matches(indexed, gold_file) for indexed in indexed_files)
        tfidf_rank = matching_rank(gold_file, top50_files) if top50_computed else None
        if final_rank is not None:
            status = "recovered_final_top20"
        elif not bool(cache_row.get("index_available")):
            status = "analysis_unavailable"
        elif not index_present:
            status = "index_coverage_failure"
        elif not top50_computed:
            status = "analysis_unavailable"
        elif tfidf_rank is None:
            status = "initial_retrieval_miss"
        else:
            status = "reranker_demotion"
        evidence.append(
            {
                "gold_file": gold_file,
                "status": status,
                "index_present": index_present,
                "tfidf_top50_rank": tfidf_rank,
                "final_top20_rank": final_rank,
            }
        )

    missed_statuses = [
        row["status"] for row in evidence if row["status"] != "recovered_final_top20"
    ]
    failure_stage = choose_failure_stage(missed_statuses)
    primary_category = "cross_file_partial_miss" if outcome == "partial_recall" else failure_stage
    input_validation = prediction.get("input_validation") or {}
    signals_present = input_validation.get("signals_present") or {}
    sparse_clues = not any(bool(value) for value in signals_present.values())
    return {
        "split": split_label,
        "split_role": split_role,
        "ticket_id": _ticket_id(gold) or _ticket_id(ticket),
        "repo": _repository_name(gold) or _repository_name(ticket) or "unknown",
        "base_commit": str(ticket.get("base_commit") or gold.get("base_commit") or ""),
        "title": str(ticket.get("title") or ticket.get("summary") or ""),
        "outcome": outcome,
        "primary_category": primary_category,
        "primary_category_zh": PRIMARY_CATEGORY_LABELS[primary_category],
        "failure_stage": failure_stage,
        "failure_stage_zh": FAILURE_STAGE_LABELS[failure_stage],
        "suggested_next_step": suggested_next_step(failure_stage, outcome),
        "gold_file_count": len(gold_files),
        "recovered_gold_count": sum(
            row["status"] == "recovered_final_top20" for row in evidence
        ),
        "gold_files": gold_files,
        "missing_gold_files": [
            row["gold_file"]
            for row in evidence
            if row["status"] != "recovered_final_top20"
        ],
        "final_top20": final_files,
        "gold_file_evidence": evidence,
        "index_available": bool(cache_row.get("index_available")),
        "index_path": str(cache_row.get("index_path") or ""),
        "indexed_file_count": int(cache_row.get("indexed_file_count") or 0),
        "tfidf_top50_computed": top50_computed,
        "tfidf_top50": top50_rows,
        "ticket_signal_presence": {
            "stack_trace": bool(signals_present.get("stack_trace")),
            "path_hint": bool(signals_present.get("path_hint")),
            "identifier": bool(signals_present.get("identifier")),
        },
        "sparse_ticket_clues": sparse_clues,
        "manual_review_required": sparse_clues or failure_stage in {
            "mixed_retrieval_and_rerank",
            "analysis_unavailable",
        },
    }


def choose_failure_stage(statuses: Iterable[str]) -> str:
    unique = set(statuses)
    if not unique or "analysis_unavailable" in unique:
        return "analysis_unavailable"
    if "index_coverage_failure" in unique:
        return "index_coverage_failure"
    has_retrieval = "initial_retrieval_miss" in unique
    has_rerank = "reranker_demotion" in unique
    if has_retrieval and has_rerank:
        return "mixed_retrieval_and_rerank"
    if has_retrieval:
        return "initial_retrieval_miss"
    if has_rerank:
        return "reranker_demotion"
    return "analysis_unavailable"


def suggested_next_step(failure_stage: str, outcome: str) -> str:
    if failure_stage == "index_coverage_failure":
        return "檢查測試檔排除規則、支援副檔名與檔案大小限制。"
    if failure_stage == "initial_retrieval_miss":
        return "優先測試API查詢擴充、Import Graph與較大的TF-IDF候選池。"
    if failure_stage == "reranker_demotion":
        return "檢查SBERT重排與檔案分數整合，避免正確檔案被排出Top-20。"
    if failure_stage == "mixed_retrieval_and_rerank":
        return "分別處理初步檢索與SBERT重排，不能只調整單一階段。"
    if outcome == "partial_recall":
        return "人工確認遺漏檔案與已命中檔案的程式關係。"
    return "人工檢查Ticket線索、索引與候選排名。"


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    missed_file_statuses: Counter[str] = Counter()
    for case in cases:
        for row in case.get("gold_file_evidence") or []:
            status = str(row.get("status") or "")
            if status and status != "recovered_final_top20":
                missed_file_statuses[status] += 1
    return {
        "failure_cases": len(cases),
        "partial_recall_rows": sum(case["outcome"] == "partial_recall" for case in cases),
        "miss_rows": sum(case["outcome"] == "miss" for case in cases),
        "primary_category_counts": dict(
            sorted(Counter(case["primary_category"] for case in cases).items())
        ),
        "failure_stage_counts": dict(
            sorted(Counter(case["failure_stage"] for case in cases).items())
        ),
        "missed_gold_file_status_counts": dict(sorted(missed_file_statuses.items())),
        "repository_counts": dict(sorted(Counter(case["repo"] for case in cases).items())),
        "sparse_ticket_clue_rows": sum(bool(case["sparse_ticket_clues"]) for case in cases),
        "manual_review_required_rows": sum(
            bool(case["manual_review_required"]) for case in cases
        ),
    }


def matching_gold_files(gold_files: list[str], candidates: list[str]) -> list[str]:
    return [
        gold_file
        for gold_file in gold_files
        if any(_file_matches(candidate, gold_file) for candidate in candidates)
    ]


def matching_rank(gold_file: str, candidates: list[str]) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if _file_matches(candidate, gold_file):
            return rank
    return None


def find_index_path(ticket: dict[str, Any], cache_dirs: list[Path]) -> Path | None:
    repo = safe_name(str(ticket.get("repo") or "local"))
    commit = safe_name(str(ticket.get("base_commit") or "missing-commit"))
    names = (f"{repo}__{commit[:16]}.json", f"{repo}__{commit[:12]}.json")
    for cache_dir in cache_dirs:
        for name in names:
            candidate = cache_dir / name
            if candidate.is_file():
                return candidate
    return None


def safe_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in normalized)


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {_ticket_id(row): row for row in read_records(path) if _ticket_id(row)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row.get("ticket_id") or ""))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "split_role",
        "ticket_id",
        "repo",
        "base_commit",
        "title",
        "outcome",
        "primary_category",
        "primary_category_zh",
        "failure_stage",
        "failure_stage_zh",
        "suggested_next_step",
        "gold_file_count",
        "recovered_gold_count",
        "gold_files",
        "missing_gold_files",
        "index_available",
        "indexed_file_count",
        "sparse_ticket_clues",
        "manual_review_required",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key, "") for key in fieldnames}
            row["gold_files"] = " | ".join(case.get("gold_files") or [])
            row["missing_gold_files"] = " | ".join(case.get("missing_gold_files") or [])
            writer.writerow(row)
    temporary.replace(path)


def select_manual_review_sample(cases: list[dict[str, Any]], *, size: int) -> list[dict[str, Any]]:
    if len(cases) <= size:
        return list(cases)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        key = (str(case["split"]), str(case["failure_stage"]))
        groups.setdefault(key, []).append(case)
    selected: list[dict[str, Any]] = []
    while len(selected) < size and any(groups.values()):
        for key in sorted(groups):
            rows = groups[key]
            if rows and len(selected) < size:
                selected.append(rows.pop(0))
    return selected


def render_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# 第一階段未命中案例分析",
        "",
        "> 本報告只分析Top-20部分命中與完全未命中案例。舊Frozen Holdout只用於v1事後說明，不再作為v2獨立測試集。",
        "",
        "## 一、資料範圍",
        "",
        "| 資料組 | 用途 | 總Ticket | 完全找回 | 部分命中 | 完全未命中 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, summary in output["split_summaries"].items():
        lines.append(
            f"| {label} | {summary['role']} | {summary['tickets']} | "
            f"{summary['full_recall_rows']} | {summary['partial_recall_rows']} | "
            f"{summary['miss_rows']} |"
        )
    lines.extend(["", "## 二、失敗環節", ""])
    for label, summary in output["split_summaries"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                "| 失敗環節 | Ticket數 |",
                "|---|---:|",
            ]
        )
        for stage, count in summary["failure_stage_counts"].items():
            lines.append(f"| {FAILURE_STAGE_LABELS.get(stage, stage)} | {count} |")
        lines.append("")
    lines.extend(
        [
            "## 三、結果判讀方式",
            "",
            "- Index coverage failure：先修正索引範圍，排序模型無法找回不存在的候選。",
            "- Initial retrieval miss：優先改善TF-IDF候選池、查詢擴充或Repository大小調整。",
            "- Reranker demotion：優先檢查SBERT排序與檔案分數整合。",
            "- Cross-file partial miss：優先檢查Import／Call Graph是否能補回其他檔案。",
            "- Analysis unavailable：先補齊Code Index或TF-IDF前50資料，再做人工判讀。",
            "",
            "## 四、案例清單",
            "",
            "| 資料組 | Ticket | Repository | 結果 | 失敗環節 | 遺漏檔案 |",
            "|---|---|---|---|---|---|",
        ]
    )
    for case in output["cases"]:
        missing = "<br>".join(case["missing_gold_files"])
        lines.append(
            f"| {case['split']} | {case['ticket_id']} | {case['repo']} | "
            f"{case['primary_category_zh']} | {case['failure_stage_zh']} | {missing} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
