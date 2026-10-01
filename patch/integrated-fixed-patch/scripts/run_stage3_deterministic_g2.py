#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_fault_localization import evaluate_records
from utils.fault_localization import (
    LocalizationCandidate,
    SYMBOL_RETRIEVAL_MODES,
    SYMBOL_SELECTION_MODES,
    build_call_graph,
    build_symbol_candidate_pool,
    load_code_index,
    rank_symbol_candidates,
)


DEFAULT_CONFIG = ROOT / "configs/fault_localization/stage3_deterministic_g2_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the development-only deterministic Stage-3 G2 comparison."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive.")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)

    tickets_path = _project_path(config["tickets"])
    gold_path = _project_path(config["gold"])
    stage2_path = _project_path(config["stage2_predictions"])
    index_dir = _project_path(config["index_dir"])
    output_dir = _project_path(config["output_dir"])
    for path in (tickets_path, gold_path, stage2_path, index_dir):
        if not path.exists():
            raise SystemExit(f"Required input does not exist: {path}")
    if "holdout" in tickets_path.as_posix().casefold():
        raise SystemExit("G2 runner accepts development data only; holdout input was rejected.")
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets = _read_jsonl(tickets_path)
    gold = _read_jsonl(gold_path)
    stage2_by_ticket = {
        _ticket_id(row): row for row in _read_jsonl(stage2_path) if _ticket_id(row)
    }
    missing_stage2 = [
        _ticket_id(ticket)
        for ticket in tickets
        if _ticket_id(ticket) not in stage2_by_ticket
    ]
    if missing_stage2:
        raise SystemExit(
            f"Stage-2 source is missing {len(missing_stage2)} tickets: {missing_stage2[:5]}"
        )

    variants = config["variants"]
    predictions: dict[str, list[dict[str, Any]]] = {}
    completed: dict[str, set[str]] = {}
    for variant in variants:
        variant_id = variant["variant_id"]
        output_path = output_dir / f"{variant_id}_predictions.jsonl"
        rows = _read_jsonl(output_path) if args.resume and output_path.exists() else []
        predictions[variant_id] = rows
        completed[variant_id] = {_ticket_id(row) for row in rows if _ticket_id(row)}

    started = time.perf_counter()
    processed_since_checkpoint = 0
    for position, ticket in enumerate(tickets, start=1):
        ticket_id = _ticket_id(ticket)
        pending = [
            variant
            for variant in variants
            if ticket_id not in completed[variant["variant_id"]]
        ]
        if not pending:
            continue
        stage2 = stage2_by_ticket[ticket_id]
        stage2_files = _stage2_files(stage2)
        index_path = _index_path(ticket, index_dir)
        if not index_path.exists():
            raise SystemExit(f"Missing code index for {ticket_id}: {index_path}")
        index = load_code_index(index_path)
        wanted_paths = {row["file_path"] for row in stage2_files}
        symbol_pool = build_symbol_candidate_pool(index.chunks, wanted_paths)
        needs_cached_call_graph = any(
            str(variant.get("selection_mode") or "global") == "call-neighborhood-v2"
            for variant in pending
        )
        if needs_cached_call_graph and index.call_graph is None:
            index.call_graph = build_call_graph(index.chunks, index.symbol_definitions)
        stage2_scores = {
            row["file_path"]: float(row.get("score") or row.get("retrieval_score") or 0.0)
            for row in stage2_files
        }
        for variant in pending:
            ranked = rank_symbol_candidates(
                ticket,
                symbol_pool,
                stage2_file_scores=stage2_scores,
                mode=variant["mode"],
                top_k=int(config["candidate_k"]),
                per_file_quota=int(variant["per_file_quota"]),
                selection_mode=str(variant.get("selection_mode") or "global"),
                call_graph=index.call_graph,
            )
            row = _prediction_row(
                ticket=ticket,
                stage2=stage2,
                ranked=ranked,
                variant=variant,
                candidate_k=int(config["candidate_k"]),
                retrieval_top_k=int(config["retrieval_top_k"]),
                symbol_pool_count=len(symbol_pool),
            )
            predictions[variant["variant_id"]].append(row)
            completed[variant["variant_id"]].add(ticket_id)
        processed_since_checkpoint += 1
        print(
            f"[{position}/{len(tickets)}] {ticket_id}: "
            f"{len(symbol_pool)} symbols, {len(pending)} variants",
            flush=True,
        )
        if processed_since_checkpoint >= args.checkpoint_every:
            _write_predictions(output_dir, predictions)
            processed_since_checkpoint = 0

    _write_predictions(output_dir, predictions)
    summary = _build_summary(
        config=config,
        config_path=config_path,
        tickets=tickets,
        gold=gold,
        predictions=predictions,
        input_paths=(tickets_path, gold_path, stage2_path),
        runtime_seconds=time.perf_counter() - started,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT_ZH.md").write_text(
        _markdown_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["decision"], ensure_ascii=False), flush=True)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("data_scope") != "development-only":
        raise SystemExit("Config must declare data_scope=development-only.")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise SystemExit("Config must contain at least one variant.")
    ids = [str(row.get("variant_id") or "") for row in variants]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SystemExit("Every variant_id must be non-empty and unique.")
    if "b0_tfidf" not in ids:
        raise SystemExit("Config must include the b0_tfidf control.")
    if int(config.get("candidate_k") or 0) != 30:
        raise SystemExit("G2 v1 requires candidate_k=30.")
    for variant in variants:
        mode = str(variant.get("mode") or "")
        selection_mode = str(variant.get("selection_mode") or "global")
        quota = int(variant.get("per_file_quota") or 0)
        if mode not in SYMBOL_RETRIEVAL_MODES:
            raise SystemExit(f"Unsupported retrieval mode in {variant['variant_id']}: {mode}")
        if selection_mode not in SYMBOL_SELECTION_MODES:
            raise SystemExit(
                f"Unsupported selection mode in {variant['variant_id']}: {selection_mode}"
            )
        if selection_mode != "global" and quota:
            raise SystemExit(
                f"Coverage selector {variant['variant_id']} must use per_file_quota=0."
            )


def _prediction_row(
    *,
    ticket: dict[str, Any],
    stage2: dict[str, Any],
    ranked: list[LocalizationCandidate],
    variant: dict[str, Any],
    candidate_k: int,
    retrieval_top_k: int,
    symbol_pool_count: int,
) -> dict[str, Any]:
    candidate_rows = [_serialize_symbol(row, rank) for rank, row in enumerate(ranked, 1)]
    final_rows = deepcopy(candidate_rows[:retrieval_top_k])
    return {
        "ticket_id": _ticket_id(ticket),
        "repo": ticket.get("repo") or ticket.get("repository") or "",
        "base_commit": ticket.get("base_commit") or "",
        "method": {
            "name": "stage3_deterministic_symbol_retrieval",
            "variant_id": variant["variant_id"],
            "symbol_retrieval_mode": variant["mode"],
            "symbol_per_file_quota": int(variant["per_file_quota"]),
            "symbol_selection_mode": str(variant.get("selection_mode") or "global"),
            "symbol_candidate_k": candidate_k,
            "symbol_top_k": retrieval_top_k,
            "symbol_llm_rerank_requested": False,
        },
        "stage1_candidate_files": stage2.get("stage1_candidate_files", []),
        "localized_files": _stage2_files(stage2),
        "stage2_localized_files": _stage2_files(stage2),
        "stage3_candidate_symbols": candidate_rows,
        "stage3_retrieval_symbols": deepcopy(final_rows),
        "stage3_ranked_symbols": final_rows,
        "stage3_diagnostics": {
            "localization_requested": True,
            "requested": False,
            "eligible": bool(candidate_rows),
            "llm_attempted": False,
            "llm_valid": False,
            "llm_rerank_used": False,
            "fallback_used": False,
            "fallback_reason": "llm_not_requested" if candidate_rows else "no_symbol_candidates",
            "source": "symbol_retrieval" if candidate_rows else "not_run",
            "retrieval_mode": variant["mode"],
            "per_file_quota": int(variant["per_file_quota"]),
            "selection_mode": str(variant.get("selection_mode") or "global"),
            "stage2_file_count": len(_stage2_files(stage2)),
            "ast_symbol_pool_count": symbol_pool_count,
            "candidate_symbol_count": len(candidate_rows),
            "returned_symbol_count": len(final_rows),
        },
    }


def _serialize_symbol(candidate: LocalizationCandidate, rank: int) -> dict[str, Any]:
    chunk = candidate.chunk
    return {
        "rank": rank,
        "file_path": chunk.file_path,
        "symbol_qualified_name": chunk.symbol_qualified_name,
        "symbol_name": chunk.symbol_name,
        "symbol_kind": chunk.symbol_kind,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": round(float(candidate.score), 6),
        "retrieval_score": round(float(candidate.score), 6),
        "reason": candidate.reason,
        "chunk_id": chunk.chunk_id,
        "scoring_signals": _json_safe_signals(candidate.signals),
    }


def _json_safe_signals(signals: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in signals.items():
        if isinstance(value, float):
            result[key] = round(value, 6)
        elif isinstance(value, dict):
            result[key] = {
                str(inner_key): round(inner_value, 6)
                if isinstance(inner_value, float)
                else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            result[key] = value
    return result


def _build_summary(
    *,
    config: dict[str, Any],
    config_path: Path,
    tickets: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    input_paths: tuple[Path, ...],
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    compact: dict[str, dict[str, Any]] = {}
    variants_by_id = {row["variant_id"]: row for row in config["variants"]}
    for variant_id, rows in predictions.items():
        evaluated = evaluate_records(gold, rows)
        metrics[variant_id] = evaluated
        formal = evaluated["formal_symbol_evaluation"]
        candidate = formal["candidate_exact"]
        compact[variant_id] = {
            "mode": variants_by_id[variant_id]["mode"],
            "per_file_quota": variants_by_id[variant_id]["per_file_quota"],
            "selection_mode": str(
                variants_by_id[variant_id].get("selection_mode") or "global"
            ),
            "eligible_rows": candidate["eligible_rows"],
            "hit_at_10": candidate["hit_at_10"],
            "recall_at_10": candidate["recall_at_10"],
            "hit_at_30": candidate["hit_at_30"],
            "recall_at_30": candidate["recall_at_30"],
            "candidate_output_coverage": formal["coverage"]["candidate_output_coverage"],
        }
    selected = select_variant(compact)
    threshold = float(config["g2"]["threshold"])
    selected_recall = float(compact[selected]["recall_at_30"])
    return {
        "schema_version": "stage3-deterministic-g2-result-v1",
        "experiment_id": config["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": config["data_scope"],
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in input_paths
        },
        "ticket_rows": len(tickets),
        "mapped_gold_records": sum(
            1 for row in gold if str(row.get("mapping_status") or "").casefold() == "mapped"
        ),
        "runtime_seconds": round(runtime_seconds, 4),
        "variants": compact,
        "decision": {
            "selected_variant": selected,
            "selected_recall_at_30": selected_recall,
            "g2_threshold": threshold,
            "g2_passed": selected_recall >= threshold,
            "next_work_package": "WP4" if selected_recall >= threshold else "WP3",
        },
        "full_metrics": metrics,
    }


def select_variant(compact: dict[str, dict[str, Any]]) -> str:
    b0 = compact["b0_tfidf"]
    b1_ids = [variant_id for variant_id in compact if variant_id != "b0_tfidf"]
    best_b1 = max(
        b1_ids,
        key=lambda variant_id: (
            float(compact[variant_id]["recall_at_30"]),
            float(compact[variant_id]["recall_at_10"]),
            -int(compact[variant_id]["per_file_quota"] or 10**9),
            variant_id,
        ),
    )
    if float(compact[best_b1]["recall_at_30"]) > float(b0["recall_at_30"]):
        return best_b1
    return "b0_tfidf"


def _markdown_report(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Stage-3 Deterministic Candidate G2（Development v1）",
        "",
        f"- 狀態：{'PASS' if decision['g2_passed'] else 'FAIL'}",
        f"- 選定設定：`{decision['selected_variant']}`",
        f"- Conditional Exact Candidate Recall@30：{decision['selected_recall_at_30']:.2%}",
        f"- G2 門檻：{decision['g2_threshold']:.0%}",
        f"- Eligible tickets：{summary['variants'][decision['selected_variant']]['eligible_rows']}",
        f"- Development tickets：{summary['ticket_rows']}",
        "",
        "## 比較結果",
        "",
        "| Variant | Retrieval | Selection | Quota | Eligible | Hit@10 | Recall@10 | Hit@30 | Recall@30 | Coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, row in summary["variants"].items():
        lines.append(
            f"| `{variant_id}` | `{row['mode']}` | `{row['selection_mode']}` | {row['per_file_quota']} | "
            f"{row['eligible_rows']} | {row['hit_at_10']:.2%} | "
            f"{row['recall_at_10']:.2%} | {row['hit_at_30']:.2%} | "
            f"{row['recall_at_30']:.2%} | {row['candidate_output_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 決策",
            "",
            (
                "G2 已通過，可用選定 deterministic baseline 進入 WP4 Top-10 LLM 配對實驗。"
                if decision["g2_passed"]
                else "G2 未通過；維持 WP3，先改善 deterministic candidate retrieval，不調整 LLM prompt。"
            ),
            "",
            "本報告只使用 development data，且所有 variant 共用同一份已保存的 Stage-2 Top-5；沒有呼叫 LLM。",
            "",
        ]
    )
    return "\n".join(lines)


def _stage2_files(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("stage2_localized_files")
    if not isinstance(values, list):
        values = row.get("localized_files")
    return [item for item in values or [] if isinstance(item, dict) and item.get("file_path")]


def _index_path(ticket: dict[str, Any], index_dir: Path) -> Path:
    repository = str(ticket.get("repo") or ticket.get("repository") or "")
    commit = str(ticket.get("base_commit") or "")
    return index_dir / f"{_safe_name(repository)}__{_safe_name(commit[:16])}.json"


def _safe_name(value: str) -> str:
    normalized = value.replace("/", "__").replace("\\", "__").replace(":", "_")
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in normalized
    )


def _ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or row.get("id") or "")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _write_predictions(
    output_dir: Path,
    predictions: dict[str, list[dict[str, Any]]],
) -> None:
    for variant_id, rows in predictions.items():
        ordered = sorted(rows, key=lambda row: _ticket_id(row))
        path = output_dir / f"{variant_id}_predictions.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
