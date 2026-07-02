#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.decision import QueryDecision, tune_query_threshold
from duplicate_ticket_detection.rerank import RerankConfig, apply_metadata_rerank_scores, parse_field_weights
from duplicate_ticket_detection.splits import train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import combine_similarity_scores

from run_top1_improvement_experiments import (
    clear_torch_cache,
    group_mined_negative_ids,
    mine_wrong_top1_negatives,
    sbert_score_matrices,
    train_detector,
    write_dict_rows,
)
from duplicate_ticket_detection.cli import write_query_decisions_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune duplicate decision threshold with the Top-1-mined SBERT training flow."
    )
    parser.add_argument("--tickets", default="data/mozilla_firefox_decision_eval.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--combine", default="weighted:0.6,0.4")
    parser.add_argument("--mining-combine", default="weighted:0.6,0.4")
    parser.add_argument("--sbert-base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-triplets", type=int, default=2000)
    parser.add_argument("--mined-negatives-per-anchor", type=int, default=1)
    parser.add_argument("--negative-strategy", choices=("all", "random", "hard"), default="hard")
    parser.add_argument("--negatives-per-anchor", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", help="Torch device for SBERT training, e.g. cpu, mps, or cuda")
    parser.add_argument("--rerank-fields", nargs="*", default=["component:0.02"])
    parser.add_argument("--rerank-candidate-pool", type=int, default=50)
    parser.add_argument("--min-precision", type=float, help="Prefer thresholds with at least this precision")
    parser.add_argument("--min-recall", type=float, help="Prefer thresholds with at least this recall")
    parser.add_argument("--output-json", default="reports/top1_mined_decision_threshold_balanced.json")
    parser.add_argument("--output-decisions-csv", default="reports/top1_mined_decision_threshold_balanced_decisions.csv")
    parser.add_argument("--mined-negatives-csv", default="reports/top1_mined_threshold_hard_negatives.csv")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    rerank_config = RerankConfig(
        field_weights=parse_field_weights(args.rerank_fields),
        candidate_pool=args.rerank_candidate_pool,
    )

    print(f"tickets={len(tickets)}", flush=True)
    print(f"split={split.name}", flush=True)
    print(f"train={len(split.train)} validation={len(split.test)}", flush=True)
    print(f"combine={args.combine}", flush=True)
    print(f"rerank_fields={format_field_weights(rerank_config.field_weights)}", flush=True)

    print("[Top-1 threshold] training first-pass SBERT for mined negatives", flush=True)
    first_pass = train_detector(split.train, args, mined_negatives_by_anchor={})
    title_scores, content_scores = sbert_score_matrices(first_pass, split.train, split.train)
    mined_rows = mine_wrong_top1_negatives(
        split,
        split.train,
        title_scores,
        content_scores,
        combine=args.mining_combine,
        top_k=args.top_k,
    )
    mined_by_anchor = group_mined_negative_ids(mined_rows, limit=args.mined_negatives_per_anchor)
    write_dict_rows(mined_rows, args.mined_negatives_csv)
    del first_pass
    gc.collect()
    clear_torch_cache()
    print(f"mined_anchors={len(mined_by_anchor)} mined_negatives={len(mined_rows)}", flush=True)

    print("[Top-1 threshold] training final SBERT with mined hard negatives", flush=True)
    detector = train_detector(split.train, args, mined_negatives_by_anchor=mined_by_anchor)
    validation_title_scores, validation_content_scores = sbert_score_matrices(detector, split.test, split.train)
    scores = combine_similarity_scores(validation_title_scores, validation_content_scores, args.combine)
    scores = apply_metadata_rerank_scores(
        split.test,
        split.train,
        scores,
        validation_title_scores,
        validation_content_scores,
        rerank_config,
    )
    decisions = query_decisions_from_scores(
        split.test,
        split.train,
        scores,
        title_scores=validation_title_scores,
        content_scores=validation_content_scores,
    )
    metrics = tune_query_threshold(decisions, min_precision=args.min_precision, min_recall=args.min_recall)
    constraints_satisfied = passes_constraints(metrics, min_precision=args.min_precision, min_recall=args.min_recall)
    write_outputs(
        args,
        metrics,
        decisions,
        split.test,
        split.train,
        split_name=split.name,
        train_size=len(split.train),
        validation_size=len(split.test),
        constraints_satisfied=constraints_satisfied,
        rerank_fields=format_field_weights(rerank_config.field_weights),
    )
    print_decision_metrics(metrics, constraints_satisfied=constraints_satisfied)
    print(f"output_json={args.output_json}", flush=True)
    print(f"output_decisions_csv={args.output_decisions_csv}", flush=True)
    print(f"mined_negatives_csv={args.mined_negatives_csv}", flush=True)
    del detector
    gc.collect()
    clear_torch_cache()
    return 0


def query_decisions_from_scores(
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    scores: np.ndarray,
    *,
    title_scores: np.ndarray,
    content_scores: np.ndarray,
) -> list[QueryDecision]:
    decisions: list[QueryDecision] = []
    for query_index, query in enumerate(queries):
        relevant = relevant_duplicate_ids(query, candidates)
        query_scores = scores[query_index].copy()
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                query_scores[candidate_index] = -np.inf
        ranked_indices = np.argsort(-query_scores) if len(query_scores) else np.array([], dtype=int)
        best_index = int(ranked_indices[0]) if len(ranked_indices) else -1
        second_index = int(ranked_indices[1]) if len(ranked_indices) > 1 else -1
        best_candidate_id = "" if best_index < 0 else candidates[best_index].ticket_id
        best_score = float("-inf") if best_index < 0 else float(query_scores[best_index])
        second_score = float("-inf") if second_index < 0 else float(query_scores[second_index])
        decisions.append(
            QueryDecision(
                query_id=query.ticket_id,
                best_candidate_id=best_candidate_id,
                score=best_score,
                is_duplicate=bool(relevant),
                best_is_correct_duplicate=best_candidate_id in relevant,
                title_score=0.0 if best_index < 0 else float(title_scores[query_index, best_index]),
                content_score=0.0 if best_index < 0 else float(content_scores[query_index, best_index]),
                second_score=second_score,
            )
        )
    return decisions


def write_outputs(
    args: argparse.Namespace,
    metrics,
    decisions: list[QueryDecision],
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    *,
    split_name: str,
    train_size: int,
    validation_size: int,
    constraints_satisfied: bool,
    rerank_fields: str,
) -> None:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                **metrics.__dict__,
                "split": split_name,
                "method": "sbert_top1_mined",
                "combine": args.combine,
                "base_model": args.sbert_base_model,
                "epochs": args.epochs,
                "max_triplets": args.max_triplets,
                "negative_strategy": args.negative_strategy,
                "negatives_per_anchor": args.negatives_per_anchor,
                "mined_negatives_per_anchor": args.mined_negatives_per_anchor,
                "rerank_fields": rerank_fields,
                "train": train_size,
                "validation": validation_size,
                "min_precision": args.min_precision,
                "min_recall": args.min_recall,
                "constraints_satisfied": constraints_satisfied,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_query_decisions_csv(
        decisions,
        queries,
        candidates,
        threshold=metrics.threshold,
        output_path=args.output_decisions_csv,
    )


def passes_constraints(metrics, *, min_precision: float | None, min_recall: float | None) -> bool:
    if min_precision is not None and metrics.precision < min_precision:
        return False
    if min_recall is not None and metrics.recall < min_recall:
        return False
    return True


def print_decision_metrics(metrics, *, constraints_satisfied: bool) -> None:
    print("Duplicate Decision Metrics", flush=True)
    print(f"constraints_satisfied={str(constraints_satisfied).lower()}", flush=True)
    print(f"threshold={metrics.threshold:.6f}", flush=True)
    print(f"precision={metrics.precision:.4f}", flush=True)
    print(f"recall={metrics.recall:.4f}", flush=True)
    print(f"f1={metrics.f1:.4f}", flush=True)
    print(f"accuracy={metrics.accuracy:.4f}", flush=True)
    print(f"true_positives={metrics.true_positives}", flush=True)
    print(f"false_positives={metrics.false_positives}", flush=True)
    print(f"true_negatives={metrics.true_negatives}", flush=True)
    print(f"false_negatives={metrics.false_negatives}", flush=True)
    print(f"correct_duplicate_links={metrics.correct_duplicate_links}", flush=True)
    print("matrix_actual_by_predicted", flush=True)
    print("actual,predicted_non_duplicate,predicted_duplicate", flush=True)
    print(f"non_duplicate,{metrics.true_negatives},{metrics.false_positives}", flush=True)
    print(f"duplicate,{metrics.false_negatives},{metrics.true_positives}", flush=True)


def format_field_weights(field_weights) -> str:
    return ",".join(f"{field}:{weight:g}" for field, weight in field_weights)


if __name__ == "__main__":
    raise SystemExit(main())
