#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(SRC), str(ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.run_stage3_deterministic_g2 import (
    DEFAULT_CONFIG,
    _index_path,
    _project_path,
    _read_jsonl,
    _stage2_files,
    _ticket_id,
    _validate_config,
)
from utils.fault_localization import (
    build_symbol_candidate_pool,
    load_code_index,
    rank_symbol_candidates,
)
from utils.symbol_evaluation import SymbolEvaluationItemV1, match_symbol
from utils.merged_ground_truth import evaluation_policy, mapped_symbols


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the exact-symbol oracle ceiling inside Stage-2 Top-5 files "
            "and separate file, AST/policy, and Top-30 ranking losses."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--variant",
        default=None,
        help="Prediction variant to diagnose; defaults to summary.json selected_variant.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    tickets_path = _project_path(config["tickets"])
    gold_path = _project_path(config["gold"])
    stage2_path = _project_path(config["stage2_predictions"])
    index_dir = _project_path(config["index_dir"])
    output_dir = _project_path(config["output_dir"])
    if "holdout" in tickets_path.as_posix().casefold():
        raise SystemExit("Oracle analysis accepts development data only.")

    summary_path = output_dir / "summary.json"
    variant_id = args.variant
    if not variant_id:
        if not summary_path.exists():
            raise SystemExit("Missing G2 summary.json; pass --variant explicitly.")
        g2_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        variant_id = str(g2_summary["decision"]["selected_variant"])
    prediction_path = output_dir / f"{variant_id}_predictions.jsonl"
    for path in (tickets_path, gold_path, stage2_path, index_dir, prediction_path):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")

    tickets = {_ticket_id(row): row for row in _read_jsonl(tickets_path)}
    stage2 = {_ticket_id(row): row for row in _read_jsonl(stage2_path)}
    predictions = {_ticket_id(row): row for row in _read_jsonl(prediction_path)}
    gold_rows = _read_jsonl(gold_path)
    gold_policy = evaluation_policy(gold_rows, allow_legacy=config.get("allow_legacy_gold") is True)
    gold_by_ticket = _gold_by_ticket(gold_rows)
    missing = sorted(set(gold_by_ticket) - set(tickets) | set(gold_by_ticket) - set(stage2))
    if missing:
        raise SystemExit(f"Missing ticket or Stage-2 rows: {missing[:5]}")

    per_ticket: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for position, ticket_id in enumerate(sorted(gold_by_ticket), start=1):
        ticket = tickets[ticket_id]
        stage2_rows = _stage2_files(stage2[ticket_id])
        stage2_file_ranks = {
            SymbolEvaluationItemV1(
                file_path=str(row["file_path"]),
                qualified_name="<module>",
                symbol_kind="module",
            ).file_path: rank
            for rank, row in enumerate(stage2_rows, start=1)
        }
        stage2_paths = set(stage2_file_ranks)
        gold_symbols = gold_by_ticket[ticket_id]
        conditional = any(item.file_path in stage2_paths for item in gold_symbols)
        pool: list[SymbolEvaluationItemV1] = []
        if conditional:
            index_path = _index_path(ticket, index_dir)
            if not index_path.exists():
                raise SystemExit(f"Missing code index for {ticket_id}: {index_path}")
            index = load_code_index(index_path)
            candidate_pool = build_symbol_candidate_pool(index.chunks, stage2_paths)
            stage2_scores = {
                str(row["file_path"]): float(
                    row.get("score") or row.get("retrieval_score") or 0.0
                )
                for row in stage2_rows
            }
            full_ranking = rank_symbol_candidates(
                ticket,
                candidate_pool,
                stage2_file_scores=stage2_scores,
                mode="b1-structured",
                top_k=max(1, len(candidate_pool)),
                per_file_quota=0,
            )
            pool = _unique_symbols(
                SymbolEvaluationItemV1(
                    file_path=candidate.chunk.file_path,
                    qualified_name=candidate.chunk.symbol_qualified_name,
                    symbol_kind=candidate.chunk.symbol_kind,
                )
                for candidate in full_ranking
            )
            pool_scores = [float(candidate.score) for candidate in full_ranking]
        else:
            pool_scores = []
        top30 = _prediction_symbols(predictions.get(ticket_id, {}))
        ticket_result, ticket_failures = analyze_ticket(
            ticket_id=ticket_id,
            repository=str(ticket.get("repo") or ticket.get("repository") or ""),
            gold_symbols=gold_symbols,
            stage2_paths=stage2_paths,
            stage2_file_ranks=stage2_file_ranks,
            pool=pool,
            pool_scores=pool_scores,
            top30=top30,
        )
        per_ticket.append(ticket_result)
        failure_rows.extend(ticket_failures)
        print(
            f"[{position}/{len(gold_by_ticket)}] {ticket_id}: "
            f"conditional={conditional}, pool={len(pool)}",
            flush=True,
        )

    report = aggregate_oracle_results(
        per_ticket,
        failure_rows,
        variant_id=variant_id,
    )
    report["ground_truth"] = gold_policy
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_failure_csv(output_dir / "oracle_symbol_classification.csv", failure_rows)
    (output_dir / "ORACLE_REPORT_ZH.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], ensure_ascii=False), flush=True)


def analyze_ticket(
    *,
    ticket_id: str,
    repository: str,
    gold_symbols: list[SymbolEvaluationItemV1],
    stage2_paths: set[str],
    pool: list[SymbolEvaluationItemV1],
    top30: list[SymbolEvaluationItemV1],
    stage2_file_ranks: dict[str, int] | None = None,
    pool_scores: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditional = any(item.file_path in stage2_paths for item in gold_symbols)
    stage2_hits = 0
    pool_hits = 0
    top30_hits = 0
    classifications: Counter[str] = Counter()
    detail_rows: list[dict[str, Any]] = []
    file_ranks = stage2_file_ranks or {}
    scores = pool_scores or []
    for gold in gold_symbols:
        file_reached = gold.file_path in stage2_paths
        pool_exact = any(match_symbol(candidate, gold).exact for candidate in pool)
        top30_exact = any(match_symbol(candidate, gold).exact for candidate in top30)
        stage2_hits += int(file_reached)
        pool_hits += int(pool_exact)
        top30_hits += int(top30_exact)
        category = classify_gold_symbol(
            gold=gold,
            file_reached=file_reached,
            pool_exact=pool_exact,
            top30_exact=top30_exact,
            pool=pool,
        )
        classifications[category] += 1
        pool_rank = _first_exact_rank(pool, gold)
        within_file_rank = (
            sum(1 for candidate in pool[:pool_rank] if candidate.file_path == gold.file_path)
            if pool_rank is not None
            else None
        )
        detail_rows.append(
            {
                "ticket_id": ticket_id,
                "repo": repository,
                "conditional_eligible": conditional,
                "file_path": gold.file_path,
                "symbol_kind": gold.symbol_kind,
                "qualified_name": gold.qualified_name,
                "stage2_file_reached": file_reached,
                "stage2_file_rank": file_ranks.get(gold.file_path),
                "pool_exact": pool_exact,
                "pool_rank": pool_rank,
                "within_file_rank": within_file_rank,
                "pool_score": (
                    round(scores[pool_rank - 1], 6)
                    if pool_rank is not None and pool_rank <= len(scores)
                    else None
                ),
                "top30_exact": top30_exact,
                "classification": category,
            }
        )
    denominator = len(gold_symbols)
    return (
        {
            "ticket_id": ticket_id,
            "repo": repository,
            "conditional_eligible": conditional,
            "gold_symbol_count": denominator,
            "stage2_file_reached_count": stage2_hits,
            "pool_exact_count": pool_hits,
            "top30_exact_count": top30_hits,
            "stage2_file_oracle_recall": _ratio(stage2_hits, denominator),
            "ast_pool_oracle_recall": _ratio(pool_hits, denominator),
            "top30_recall": _ratio(top30_hits, denominator),
            "top30_recall_given_pool_reachable": _ratio(top30_hits, pool_hits),
            "classifications": dict(sorted(classifications.items())),
        },
        detail_rows,
    )


def classify_gold_symbol(
    *,
    gold: SymbolEvaluationItemV1,
    file_reached: bool,
    pool_exact: bool,
    top30_exact: bool,
    pool: list[SymbolEvaluationItemV1],
) -> str:
    if not file_reached:
        return "stage2_file_miss"
    if gold.symbol_kind == "module" and not pool_exact:
        return "module_excluded_by_candidate_policy"
    if not pool_exact:
        same_file_relaxed = any(
            candidate.file_path == gold.file_path
            and match_symbol(candidate, gold).relaxed
            for candidate in pool
        )
        return "pool_identity_mismatch" if same_file_relaxed else "ast_symbol_absent"
    if not top30_exact:
        return "top30_ranking_miss"
    return "top30_exact_hit"


def aggregate_oracle_results(
    per_ticket: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
    *,
    variant_id: str,
) -> dict[str, Any]:
    eligible = [row for row in per_ticket if row["conditional_eligible"]]
    eligible_details = [row for row in detail_rows if row["conditional_eligible"]]
    category_counts = Counter(row["classification"] for row in eligible_details)
    pool_reachable = sum(int(row["pool_exact"]) for row in eligible_details)
    top30_reached = sum(int(row["top30_exact"]) for row in eligible_details)
    metrics = {
        "formal_symbol_tickets": len(per_ticket),
        "conditional_eligible_tickets": len(eligible),
        "conditional_gold_symbols": len(eligible_details),
        "stage2_file_oracle_macro_recall": _mean(
            row["stage2_file_oracle_recall"] for row in eligible
        ),
        "ast_pool_oracle_macro_recall": _mean(
            row["ast_pool_oracle_recall"] for row in eligible
        ),
        "top30_macro_recall": _mean(row["top30_recall"] for row in eligible),
        "top30_macro_recall_given_pool_reachable": _mean(
            row["top30_recall_given_pool_reachable"]
            for row in eligible
            if row["pool_exact_count"]
        ),
        "pool_reachable_gold_symbols": pool_reachable,
        "top30_reached_gold_symbols": top30_reached,
        "top30_micro_recall_given_pool_reachable": _ratio(
            top30_reached,
            pool_reachable,
        ),
        "classification_counts": dict(sorted(category_counts.items())),
    }
    oracle = metrics["ast_pool_oracle_macro_recall"]
    top30 = metrics["top30_macro_recall"]
    return {
        "schema_version": "stage3-symbol-pool-oracle-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "development-only",
        "variant_id": variant_id,
        "metrics": metrics,
        "loss_decomposition": {
            "stage2_file_loss": 1.0 - metrics["stage2_file_oracle_macro_recall"],
            "ast_or_candidate_policy_loss_after_stage2": (
                metrics["stage2_file_oracle_macro_recall"] - oracle
            ),
            "top30_ranking_loss_after_pool": oracle - top30,
        },
        "decision": {
            "oracle_reaches_g2_threshold": oracle >= 0.90,
            "top30_reaches_g2_threshold": top30 >= 0.90,
            "primary_blocker": _primary_blocker(category_counts),
            "remain_in_work_package": "WP3",
            "llm_experiment_allowed": False,
        },
        "per_ticket": per_ticket,
    }


def _primary_blocker(counts: Counter[str]) -> str:
    blockers = {
        key: value for key, value in counts.items() if key != "top30_exact_hit"
    }
    return max(blockers, key=lambda key: (blockers[key], key)) if blockers else "none"


def _first_exact_rank(
    candidates: list[SymbolEvaluationItemV1],
    gold: SymbolEvaluationItemV1,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if match_symbol(candidate, gold).exact:
            return rank
    return None


def _gold_by_ticket(rows: list[dict[str, Any]]) -> dict[str, list[SymbolEvaluationItemV1]]:
    grouped: dict[str, list[SymbolEvaluationItemV1]] = {}
    for row in mapped_symbols(rows):
        if str(row.get("mapping_status") or "").strip().casefold() != "mapped":
            continue
        ticket_id = _ticket_id(row)
        item = SymbolEvaluationItemV1.from_mapping(row)
        if item not in grouped.setdefault(ticket_id, []):
            grouped[ticket_id].append(item)
    return grouped


def _prediction_symbols(row: dict[str, Any]) -> list[SymbolEvaluationItemV1]:
    values = row.get("stage3_candidate_symbols")
    if not isinstance(values, list):
        return []
    result: list[SymbolEvaluationItemV1] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            item = SymbolEvaluationItemV1.from_mapping(value)
        except ValueError:
            continue
        if item not in result:
            result.append(item)
    return result


def _unique_symbols(
    values: Iterable[SymbolEvaluationItemV1],
) -> list[SymbolEvaluationItemV1]:
    result: list[SymbolEvaluationItemV1] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        key = (value.file_path, value.symbol_kind, value.qualified_name)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def _write_failure_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ticket_id",
        "repo",
        "conditional_eligible",
        "file_path",
        "symbol_kind",
        "qualified_name",
        "stage2_file_reached",
        "stage2_file_rank",
        "pool_exact",
        "pool_rank",
        "within_file_rank",
        "pool_score",
        "top30_exact",
        "classification",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    losses = report["loss_decomposition"]
    decision = report["decision"]
    categories = metrics["classification_counts"]
    lines = [
        "# Stage-3 AST Symbol Pool Exact Oracle（Development v1）",
        "",
        f"- 診斷 variant：`{report['variant_id']}`",
        f"- Conditional eligible Tickets：{metrics['conditional_eligible_tickets']}",
        f"- Conditional gold symbols：{metrics['conditional_gold_symbols']}",
        f"- Stage-2 file oracle macro recall：{metrics['stage2_file_oracle_macro_recall']:.2%}",
        f"- AST pool exact oracle macro recall：{metrics['ast_pool_oracle_macro_recall']:.2%}",
        f"- 實際 Top-30 macro recall：{metrics['top30_macro_recall']:.2%}",
        f"- Pool 可達時 Top-30 micro recall：{metrics['top30_micro_recall_given_pool_reachable']:.2%}",
        "",
        "## Loss decomposition",
        "",
        "| Loss | 百分點 |",
        "|---|---:|",
        f"| Stage-2 未涵蓋全部 gold files | {losses['stage2_file_loss']:.2%} |",
        f"| AST／candidate policy 在已找到檔案後仍不可達 | {losses['ast_or_candidate_policy_loss_after_stage2']:.2%} |",
        f"| Pool 已可達但掉出 Top-30 | {losses['top30_ranking_loss_after_pool']:.2%} |",
        "",
        "## Gold symbol 分類（conditional rows）",
        "",
        "| 分類 | 數量 |",
        "|---|---:|",
    ]
    for category, count in categories.items():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(
        [
            "",
            "## 決策",
            "",
            f"主要 blocker：`{decision['primary_blocker']}`。",
            (
                "AST pool oracle 已達 90%，下一輪只需改善 Top-30 排序。"
                if decision["oracle_reaches_g2_threshold"]
                else "AST pool oracle 本身低於 90%；先修正 candidate policy／index identity，不調整 LLM。"
            ),
            "",
            "本分析只讀取 development artifacts 與 base-commit indexes，沒有呼叫 LLM。",
            "",
        ]
    )
    return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


if __name__ == "__main__":
    main()
