#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.analyze_stage1_failure_cases import (  # noqa: E402
    FAILURE_STAGE_LABELS,
    choose_failure_stage,
    safe_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a stratified Stage-1 failure sample against its raw ranking evidence."
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--repo-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mark-manual-inspection-complete",
        action="store_true",
        help="Record that the selected rows were also inspected manually after evidence validation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis_path = Path(args.analysis)
    sample_path = Path(args.sample)
    repo_cache_dir = Path(args.repo_cache_dir)
    output_dir = Path(args.output_dir)
    if not analysis_path.is_file():
        raise FileNotFoundError(analysis_path)
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    if not repo_cache_dir.is_dir():
        raise FileNotFoundError(repo_cache_dir)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    cases_by_id = {
        str(row.get("ticket_id") or ""): row
        for row in analysis.get("cases") or []
        if row.get("ticket_id")
    }
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    sample_ids = [str(row.get("ticket_id") or "") for row in sample_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manual review sample contains duplicate ticket IDs.")
    missing = [ticket_id for ticket_id in sample_ids if ticket_id not in cases_by_id]
    if missing:
        raise ValueError(f"Sample IDs missing from analysis: {', '.join(missing[:5])}")

    reviews = [
        review_case(
            cases_by_id[ticket_id],
            repo_cache_dir=repo_cache_dir,
            manual_inspection_complete=args.mark_manual_inspection_complete,
        )
        for ticket_id in sample_ids
    ]
    summary = summarize_reviews(reviews)
    output = {
        "review_scope": "Stratified 30-case Stage-1 failure sample",
        "analysis_path": str(analysis_path),
        "sample_path": str(sample_path),
        "repo_cache_dir": str(repo_cache_dir),
        "manual_inspection_complete": args.mark_manual_inspection_complete,
        "summary": summary,
        "reviews": reviews,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "manual_review_summary.json"
    csv_path = output_dir / "manual_review_completed.csv"
    markdown_path = output_dir / "manual_review_report_zh.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_csv(csv_path, reviews)
    markdown_path.write_text(render_markdown(output), encoding="utf-8")
    print(
        json.dumps(
            {
                "manual_review_summary": str(json_path),
                "manual_review_csv": str(csv_path),
                "manual_review_report": str(markdown_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def review_case(
    case: dict[str, Any],
    *,
    repo_cache_dir: Path,
    manual_inspection_complete: bool,
) -> dict[str, Any]:
    evidence = list(case.get("gold_file_evidence") or [])
    missed_statuses = [
        str(row.get("status") or "")
        for row in evidence
        if row.get("status") != "recovered_final_top20"
    ]
    expected_stage = choose_failure_stage(missed_statuses)
    expected_primary = (
        "cross_file_partial_miss"
        if case.get("outcome") == "partial_recall"
        else expected_stage
    )
    issues = evidence_consistency_issues(case)
    if case.get("failure_stage") != expected_stage:
        issues.append(
            f"failure_stage={case.get('failure_stage')} but evidence implies {expected_stage}"
        )
    if case.get("primary_category") != expected_primary:
        issues.append(
            f"primary_category={case.get('primary_category')} but expected {expected_primary}"
        )

    index_absence_details: list[dict[str, str]] = []
    for row in evidence:
        if row.get("status") != "index_coverage_failure":
            continue
        reason = index_absence_reason(
            repo_cache_dir=repo_cache_dir,
            repo=str(case.get("repo") or ""),
            base_commit=str(case.get("base_commit") or ""),
            file_path=str(row.get("gold_file") or ""),
        )
        index_absence_details.append(
            {"gold_file": str(row.get("gold_file") or ""), "reason": reason}
        )

    review_status = "confirmed" if not issues else "needs_correction"
    return {
        "split": str(case.get("split") or ""),
        "ticket_id": str(case.get("ticket_id") or ""),
        "repo": str(case.get("repo") or ""),
        "title": str(case.get("title") or ""),
        "outcome": str(case.get("outcome") or ""),
        "automatic_failure_stage": str(case.get("failure_stage") or ""),
        "reviewed_failure_stage": expected_stage,
        "reviewed_failure_stage_zh": FAILURE_STAGE_LABELS[expected_stage],
        "automatic_primary_category": str(case.get("primary_category") or ""),
        "review_status": review_status,
        "manual_inspection": "completed" if manual_inspection_complete else "pending",
        "evidence_issues": issues,
        "review_note": build_review_note(expected_stage, evidence, index_absence_details, issues),
        "index_absence_details": index_absence_details,
        "gold_files": list(case.get("gold_files") or []),
        "missing_gold_files": list(case.get("missing_gold_files") or []),
        "gold_file_evidence": evidence,
    }


def evidence_consistency_issues(case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for row in case.get("gold_file_evidence") or []:
        gold_file = str(row.get("gold_file") or "")
        status = str(row.get("status") or "")
        index_present = bool(row.get("index_present"))
        tfidf_rank = row.get("tfidf_top50_rank")
        final_rank = row.get("final_top20_rank")
        valid = {
            "recovered_final_top20": final_rank is not None,
            "index_coverage_failure": not index_present and final_rank is None,
            "initial_retrieval_miss": index_present and tfidf_rank is None and final_rank is None,
            "reranker_demotion": index_present and tfidf_rank is not None and final_rank is None,
            "analysis_unavailable": True,
        }.get(status, False)
        if not valid:
            issues.append(f"{gold_file}: inconsistent evidence for status {status}")
    return issues


def index_absence_reason(
    *,
    repo_cache_dir: Path,
    repo: str,
    base_commit: str,
    file_path: str,
) -> str:
    repo_path = repo_cache_dir / safe_name(repo)
    if not repo_path.is_dir():
        return "repository_unavailable"
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_check.returncode != 0:
        return "base_commit_unavailable"
    file_check = subprocess.run(
        ["git", "cat-file", "-e", f"{base_commit}:{file_path}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if file_check.returncode == 0:
        return "existing_file_excluded_by_index_rules"
    return "file_absent_at_base_commit"


def build_review_note(
    stage: str,
    evidence: list[dict[str, Any]],
    index_absence_details: list[dict[str, str]],
    issues: list[str],
) -> str:
    if issues:
        return "分類需要修正：" + "；".join(issues)
    if stage == "index_coverage_failure":
        reasons = Counter(row["reason"] for row in index_absence_details)
        detail = "、".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        return f"分類正確；至少一個正確檔案不在Code Index。原因：{detail}。"
    if stage == "initial_retrieval_miss":
        count = sum(row.get("status") == "initial_retrieval_miss" for row in evidence)
        return f"分類正確；{count}個遺漏檔案存在索引中，但未進入TF-IDF前50名。"
    if stage == "reranker_demotion":
        ranks = [
            str(row.get("tfidf_top50_rank"))
            for row in evidence
            if row.get("status") == "reranker_demotion"
        ]
        return f"分類正確；正確檔案原在TF-IDF第{', '.join(ranks)}名，重排後未進Top-20。"
    if stage == "mixed_retrieval_and_rerank":
        retrieval = sum(row.get("status") == "initial_retrieval_miss" for row in evidence)
        rerank = sum(row.get("status") == "reranker_demotion" for row in evidence)
        return f"分類正確；包含{retrieval}個初步檢索遺漏與{rerank}個SBERT重排遺漏。"
    return "證據完整，但仍需補充人工判讀。"


def summarize_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    absence_reasons = Counter(
        detail["reason"]
        for review in reviews
        for detail in review.get("index_absence_details") or []
    )
    return {
        "reviewed_rows": len(reviews),
        "review_status_counts": dict(
            sorted(Counter(review["review_status"] for review in reviews).items())
        ),
        "failure_stage_counts": dict(
            sorted(Counter(review["reviewed_failure_stage"] for review in reviews).items())
        ),
        "index_absence_reason_counts": dict(sorted(absence_reasons.items())),
        "manual_inspection_completed_rows": sum(
            review["manual_inspection"] == "completed" for review in reviews
        ),
    }


def write_review_csv(path: Path, reviews: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "ticket_id",
        "repo",
        "title",
        "outcome",
        "automatic_failure_stage",
        "reviewed_failure_stage",
        "reviewed_failure_stage_zh",
        "review_status",
        "manual_inspection",
        "review_note",
        "gold_files",
        "missing_gold_files",
        "index_absence_details",
        "evidence_issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            row = {key: review.get(key, "") for key in fieldnames}
            row["gold_files"] = " | ".join(review["gold_files"])
            row["missing_gold_files"] = " | ".join(review["missing_gold_files"])
            row["index_absence_details"] = json.dumps(
                review["index_absence_details"], ensure_ascii=False
            )
            row["evidence_issues"] = " | ".join(review["evidence_issues"])
            writer.writerow(row)


def render_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# 第一階段30筆失敗案例抽查報告",
        "",
        f"- 抽查案例：{summary['reviewed_rows']}筆",
        f"- 分類確認：{summary['review_status_counts'].get('confirmed', 0)}筆",
        f"- 需要修正：{summary['review_status_counts'].get('needs_correction', 0)}筆",
        f"- 已完成人工閱讀：{summary['manual_inspection_completed_rows']}筆",
        "",
        "## 索引缺漏原因",
        "",
        "| 原因 | 檔案數 |",
        "|---|---:|",
    ]
    reason_labels = {
        "file_absent_at_base_commit": "修正前版本尚不存在，可能是新增或改名檔案",
        "existing_file_excluded_by_index_rules": "修正前已存在，但被索引規則排除",
        "repository_unavailable": "本機Repository不存在",
        "base_commit_unavailable": "本機缺少base commit",
    }
    for reason, count in summary["index_absence_reason_counts"].items():
        lines.append(f"| {reason_labels.get(reason, reason)} | {count} |")
    lines.extend(
        [
            "",
            "## 抽查明細",
            "",
            "| # | Ticket | 自動分類 | 審查結果 | 備註 |",
            "|---:|---|---|---|---|",
        ]
    )
    for number, review in enumerate(output["reviews"], start=1):
        lines.append(
            f"| {number} | {review['ticket_id']} | {review['reviewed_failure_stage_zh']} | "
            f"{review['review_status']} | {review['review_note']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
