from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids, write_tickets_csv
from duplicate_ticket_detection.decision import (
    DecisionMetrics,
    QueryDecision,
    collect_query_decisions,
    tune_query_threshold,
)
from duplicate_ticket_detection.metrics import average_precision
from duplicate_ticket_detection.rerank import RerankConfig, apply_metadata_rerank_scores, parse_field_weights, rerank_ranked_tickets
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector, SbertTrainingConfig
from duplicate_ticket_detection.splits import DatasetSplit, kfold_ticket_splits, train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import EvaluationRow, TfidfDuplicateDetector, combine_similarity_scores
from duplicate_ticket_detection.triplets import build_triplets, write_triplets_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate ticket detection experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate duplicate ranking with MAP")
    _add_common_ticket_args(evaluate_parser)
    evaluate_parser.add_argument("--method", choices=("tfidf",), default="tfidf")
    evaluate_parser.add_argument("--combine", default="max")
    evaluate_parser.add_argument("--top-k", type=int)
    evaluate_parser.set_defaults(func=_evaluate)

    rank_parser = subparsers.add_parser("rank", help="Rank candidate duplicate tickets")
    _add_common_ticket_args(rank_parser)
    rank_parser.add_argument("--query-ticket-id")
    rank_parser.add_argument("--query-json", help="Path to a JSON object containing a new ticket")
    rank_parser.add_argument("--combine", default="max")
    rank_parser.add_argument("--threshold", type=float)
    rank_parser.add_argument("--top-k", type=int, default=10)
    _add_rerank_args(rank_parser)
    rank_parser.set_defaults(func=_rank)

    detect_parser = subparsers.add_parser("detect-duplicate", help="Decide whether a query ticket duplicates a historical ticket")
    _add_common_ticket_args(detect_parser)
    detect_parser.add_argument("--query-ticket-id")
    detect_parser.add_argument("--query-json", help="Path to a JSON object containing a new ticket")
    detect_parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    detect_parser.add_argument("--model-dir", help="Saved SBERT model directory. Required when --method sbert")
    detect_parser.add_argument("--combine", default="max")
    detect_parser.add_argument("--threshold", type=float, required=True)
    detect_parser.add_argument("--top-k", type=int, default=5)
    _add_rerank_args(detect_parser)
    detect_parser.set_defaults(func=_detect_duplicate)

    train_parser = subparsers.add_parser("train-tfidf", help="Fit and save the TF-IDF baseline detector")
    _add_common_ticket_args(train_parser)
    train_parser.add_argument("--combine", default="max")
    train_parser.add_argument("--output", required=True)
    train_parser.set_defaults(func=_train_tfidf)

    split_parser = subparsers.add_parser("split-data", help="Write train/test CSV files for duplicate retrieval")
    _add_common_ticket_args(split_parser)
    split_parser.add_argument("--test-size", type=float, default=0.2)
    split_parser.add_argument("--seed", type=int, default=13)
    split_parser.add_argument("--train-output", required=True)
    split_parser.add_argument("--test-output", required=True)
    split_parser.set_defaults(func=_split_data)

    split_eval_parser = subparsers.add_parser("evaluate-split", help="Train on a split and evaluate held-out duplicate queries")
    _add_common_ticket_args(split_eval_parser)
    _add_experiment_args(split_eval_parser)
    split_eval_parser.add_argument("--test-size", type=float, default=0.2)
    split_eval_parser.add_argument("--seed", type=int, default=13)
    split_eval_parser.add_argument("--details", action="store_true")
    split_eval_parser.set_defaults(func=_evaluate_split)

    cv_parser = subparsers.add_parser("cross-validate", help="Run duplicate retrieval cross-validation")
    _add_common_ticket_args(cv_parser)
    _add_experiment_args(cv_parser)
    cv_parser.add_argument("--folds", type=int, default=5)
    cv_parser.add_argument("--seed", type=int, default=13)
    cv_parser.add_argument("--details", action="store_true")
    cv_parser.set_defaults(func=_cross_validate)

    sbert_train_parser = subparsers.add_parser("train-sbert", help="Fine-tune title/content SBERT models with triplets")
    _add_common_ticket_args(sbert_train_parser)
    _add_sbert_args(sbert_train_parser)
    sbert_train_parser.add_argument("--combine", default="max")
    sbert_train_parser.add_argument("--output-dir", required=True)
    sbert_train_parser.set_defaults(func=_train_sbert)

    sbert_rank_parser = subparsers.add_parser("rank-sbert", help="Rank candidates with saved SBERT models")
    _add_common_ticket_args(sbert_rank_parser)
    sbert_rank_parser.add_argument("--model-dir", required=True)
    sbert_rank_parser.add_argument("--query-ticket-id")
    sbert_rank_parser.add_argument("--query-json")
    sbert_rank_parser.add_argument("--combine", default="max")
    sbert_rank_parser.add_argument("--threshold", type=float)
    sbert_rank_parser.add_argument("--top-k", type=int, default=10)
    _add_rerank_args(sbert_rank_parser)
    sbert_rank_parser.set_defaults(func=_rank_sbert)

    sbert_evaluate_parser = subparsers.add_parser("evaluate-sbert", help="Evaluate saved SBERT models with MAP and top-k hit rate")
    _add_common_ticket_args(sbert_evaluate_parser)
    sbert_evaluate_parser.add_argument("--model-dir", required=True)
    sbert_evaluate_parser.add_argument("--combine", default="max")
    sbert_evaluate_parser.add_argument("--top-k", type=int, default=3)
    sbert_evaluate_parser.set_defaults(func=_evaluate_sbert)

    threshold_parser = subparsers.add_parser("tune-threshold", help="Tune a similarity threshold for duplicate decisions")
    _add_common_ticket_args(threshold_parser)
    threshold_parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    threshold_parser.add_argument("--combine", default="max")
    threshold_parser.add_argument("--test-size", type=float, default=0.2)
    threshold_parser.add_argument("--seed", type=int, default=13)
    threshold_parser.add_argument("--output-json")
    threshold_parser.add_argument("--output-decisions-csv", help="Write per-query duplicate/non-duplicate decisions for error analysis")
    threshold_parser.add_argument("--min-precision", type=float, help="Prefer thresholds with at least this query-level precision")
    threshold_parser.add_argument("--min-recall", type=float, help="Prefer thresholds with at least this query-level recall")
    _add_rerank_args(threshold_parser)
    _add_sbert_args(threshold_parser)
    threshold_parser.set_defaults(func=_tune_threshold)

    triplet_parser = subparsers.add_parser("build-triplets", help="Build SBERT triplet fine-tuning data")
    _add_common_ticket_args(triplet_parser)
    triplet_parser.add_argument("--field", choices=("title", "content"), default="content")
    triplet_parser.add_argument("--max-triplets", type=int)
    triplet_parser.add_argument("--seed", type=int, default=13)
    triplet_parser.add_argument("--bidirectional", action="store_true")
    triplet_parser.add_argument("--negative-strategy", choices=("all", "random", "hard"), default="all")
    triplet_parser.add_argument("--negatives-per-anchor", type=int)
    triplet_parser.add_argument("--output", required=True)
    triplet_parser.set_defaults(func=_build_triplets)

    args = parser.parse_args(argv)
    return args.func(args)


def _add_common_ticket_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tickets", required=True, help="CSV or JSONL ticket dataset")
    parser.add_argument(
        "--content-columns",
        nargs="*",
        help="Columns to concatenate as ticket content. Defaults to common bug-report fields.",
    )


def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--combine", default="max")
    parser.add_argument("--top-k", type=int, default=10)
    _add_rerank_args(parser)
    _add_sbert_args(parser)


def _add_sbert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-triplets", type=int)
    parser.add_argument("--negative-strategy", choices=("all", "random", "hard"), default="all")
    parser.add_argument("--negatives-per-anchor", type=int)
    parser.add_argument("--local-files-only", action="store_true", help="Load the base model from the local Hugging Face cache only")
    parser.add_argument("--device", help="Torch device for SBERT training, e.g. cpu, mps, or cuda")


def _add_rerank_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rerank", action="store_true", help="Rerank the first-stage candidates with metadata match bonuses")
    parser.add_argument(
        "--rerank-fields",
        nargs="*",
        help="Metadata fields used by reranker. Each value can be field or field:weight. Defaults to component:0.02 product:0.01 severity:0.005 priority:0.005",
    )
    parser.add_argument("--rerank-field-weight", type=float, default=0.05, help="Default weight for --rerank-fields without :weight")
    parser.add_argument("--rerank-candidate-pool", type=int, default=50, help="How many first-stage candidates to rerank; <=0 means all")
    parser.add_argument("--rerank-title-weight", type=float, default=0.0, help="Extra title-score weight used only in reranking")
    parser.add_argument("--rerank-content-weight", type=float, default=0.0, help="Extra content-score weight used only in reranking")


def _evaluate(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = TfidfDuplicateDetector(combine=args.combine)
    mean_ap, rows = detector.evaluate(tickets, top_k=args.top_k)

    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k or 10)
    return 0


def _rank(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    query = _load_query(args, tickets)
    detector = TfidfDuplicateDetector(combine=args.combine).fit(tickets)
    ranking = detector.rank(query, tickets, top_k=_ranking_pool_size(args))
    ranking = _maybe_rerank(query, ranking, tickets, args, top_k=args.top_k)

    _print_threshold_decision(ranking, args.threshold)
    _print_ranking(ranking)
    return 0


def _detect_duplicate(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    query = _load_query(args, tickets)

    if args.method == "tfidf":
        detector = TfidfDuplicateDetector(combine=args.combine).fit(tickets)
    else:
        if not args.model_dir:
            raise SystemExit("--model-dir is required when --method sbert")
        detector = SbertDuplicateDetector.load(args.model_dir, combine=args.combine)

    top_k = max(args.top_k, 1)
    ranking = detector.rank(query, tickets, top_k=_ranking_pool_size(args))
    ranking = _maybe_rerank(query, ranking, tickets, args, top_k=top_k)
    _print_threshold_decision(ranking, args.threshold, include_best=True)
    _print_ranking(ranking)
    return 0


def _train_tfidf(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = TfidfDuplicateDetector(combine=args.combine).fit(tickets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output)
    print(f"saved={output}")
    return 0


def _split_data(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    write_tickets_csv(split.train, args.train_output)
    write_tickets_csv(split.test, args.test_output)
    print(f"tickets={len(tickets)}")
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    print(f"evaluable_test_queries={_count_evaluable_queries(split.test, split.train)}")
    print(f"train_output={args.train_output}")
    print(f"test_output={args.test_output}")
    return 0


def _evaluate_split(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    mean_ap, rows = _run_split(split, args)

    print(f"split={split.name}")
    print(f"train={len(split.train)}")
    print(f"test={len(split.test)}")
    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k, details=args.details)
    return 0


def _cross_validate(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    splits = kfold_ticket_splits(tickets, n_splits=args.folds, seed=args.seed)
    fold_summaries: list[dict[str, float]] = []

    print(f"tickets={len(tickets)}", flush=True)
    print(f"folds={len(splits)}", flush=True)
    print(f"method={args.method}", flush=True)
    print("fold,train,test,queries,MAP,top_1_accuracy,top_k_hit_rate,MRR", flush=True)
    for split in splits:
        print(f"{split.name}: training...", flush=True)
        mean_ap, rows = _run_split(split, args)
        print(f"{split.name}: evaluating done", flush=True)
        summary = _summarize_rows(mean_ap=mean_ap, rows=rows, top_k=args.top_k)
        fold_summaries.append(summary)
        print(
            f"{split.name},{len(split.train)},{len(split.test)},{len(rows)},"
            f"{summary['MAP']:.4f},{summary['top_1_accuracy']:.4f},"
            f"{summary['top_k_hit_rate']:.4f},{summary['MRR']:.4f}",
            flush=True,
        )
        if args.details:
            _print_query_rows(rows, top_k=args.top_k)

    if fold_summaries:
        evaluable_summaries = [summary for summary in fold_summaries if summary["queries"] > 0]
        averaged_summaries = evaluable_summaries or fold_summaries
        print("mean_over_evaluable_folds")
        for key in ("MAP", "top_1_accuracy", "top_k_hit_rate", "MRR"):
            print(f"{key}={sum(summary[key] for summary in averaged_summaries) / len(averaged_summaries):.4f}")
    return 0


def _train_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector, title_triplets, content_triplets = _train_sbert_detector(tickets, args)
    detector.save(args.output_dir)
    print(f"title_triplets={len(title_triplets)}")
    print(f"content_triplets={len(content_triplets)}")
    print(f"saved={args.output_dir}")
    return 0


def _rank_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    query = _load_query(args, tickets)
    detector = SbertDuplicateDetector.load(args.model_dir, combine=args.combine)
    ranking = detector.rank(query, tickets, top_k=_ranking_pool_size(args))
    ranking = _maybe_rerank(query, ranking, tickets, args, top_k=args.top_k)

    _print_threshold_decision(ranking, getattr(args, "threshold", None))
    _print_ranking(ranking)
    return 0


def _evaluate_sbert(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    detector = SbertDuplicateDetector.load(args.model_dir, combine=args.combine)
    mean_ap, rows = detector.evaluate(tickets)

    _print_evaluation(tickets_count=len(tickets), mean_ap=mean_ap, rows=rows, top_k=args.top_k, details=True)
    return 0


def _tune_threshold(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)

    if args.method == "tfidf":
        detector = TfidfDuplicateDetector(combine=args.combine).fit(split.train)
    else:
        detector, _, _ = _train_sbert_detector(split.train, args)

    print("threshold_eval=batch_scoring", flush=True)
    query_decisions = _collect_threshold_decisions(detector, split.test, split.train, args)
    try:
        metrics = tune_query_threshold(
            query_decisions,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"split={split.name}")
    print(f"method={args.method}")
    print(f"combine={args.combine}")
    print(f"train={len(split.train)}")
    print(f"validation={len(split.test)}")
    if args.min_precision is not None:
        print(f"min_precision={args.min_precision:.4f}")
    if args.min_recall is not None:
        print(f"min_recall={args.min_recall:.4f}")
    _print_decision_metrics(metrics)

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    **metrics.__dict__,
                    "split": split.name,
                    "method": args.method,
                    "combine": args.combine,
                    "train": len(split.train),
                    "validation": len(split.test),
                    "min_precision": args.min_precision,
                    "min_recall": args.min_recall,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"output_json={output}")
    if args.output_decisions_csv:
        write_query_decisions_csv(
            query_decisions,
            split.test,
            split.train,
            threshold=metrics.threshold,
            output_path=args.output_decisions_csv,
        )
        print(f"output_decisions_csv={args.output_decisions_csv}")
    return 0


def _build_triplets(args: argparse.Namespace) -> int:
    tickets = load_tickets(args.tickets, content_columns=args.content_columns)
    triplets = build_triplets(
        tickets,
        field=args.field,
        max_triplets=args.max_triplets,
        seed=args.seed,
        bidirectional=args.bidirectional,
        negative_strategy=args.negative_strategy,
        negatives_per_anchor=args.negatives_per_anchor,
    )
    write_triplets_csv(triplets, args.output)
    print(f"triplets={len(triplets)}")
    print(f"output={args.output}")
    return 0


def _load_query(args: argparse.Namespace, tickets: list[TicketRecord]) -> TicketRecord:
    if bool(args.query_ticket_id) == bool(args.query_json):
        raise SystemExit("Provide exactly one of --query-ticket-id or --query-json")

    if args.query_ticket_id:
        for ticket in tickets:
            if ticket.ticket_id == args.query_ticket_id:
                return ticket
        raise SystemExit(f"Ticket id not found: {args.query_ticket_id}")

    with open(args.query_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("--query-json must contain a JSON object")
    return TicketRecord(
        ticket_id=str(data.get("ticket_id", "__query__")),
        title=str(data.get("title", data.get("summary", ""))),
        content=str(data.get("description", data.get("content", data.get("body", "")))),
        fields={str(key): str(value) for key, value in data.items()},
    )


def _ranking_pool_size(args: argparse.Namespace) -> int | None:
    top_k = getattr(args, "top_k", None)
    if _rerank_config_from_args(args) is None:
        return top_k
    candidate_pool = getattr(args, "rerank_candidate_pool", 50)
    if candidate_pool <= 0:
        return None
    if top_k is None:
        return candidate_pool
    return max(top_k, candidate_pool)


def _maybe_rerank(query: TicketRecord, ranking, candidates: list[TicketRecord], args: argparse.Namespace, *, top_k: int | None):
    rerank_config = _rerank_config_from_args(args)
    if rerank_config is None:
        return ranking[:top_k] if top_k is not None else ranking
    return rerank_ranked_tickets(query, ranking, candidates, rerank_config, top_k=top_k)


def _rerank_config_from_args(args: argparse.Namespace) -> RerankConfig | None:
    if not getattr(args, "rerank", False):
        return None
    try:
        field_weights = parse_field_weights(
            getattr(args, "rerank_fields", None),
            default_weight=getattr(args, "rerank_field_weight", 0.05),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return RerankConfig(
        field_weights=field_weights,
        candidate_pool=getattr(args, "rerank_candidate_pool", 50),
        title_weight=getattr(args, "rerank_title_weight", 0.0),
        content_weight=getattr(args, "rerank_content_weight", 0.0),
    )


def _run_split(split: DatasetSplit, args: argparse.Namespace) -> tuple[float, list[EvaluationRow]]:
    if args.method == "tfidf":
        detector = TfidfDuplicateDetector(combine=args.combine).fit(split.train)
        return _evaluate_detector(detector, split.test, split.train, args)

    detector, _, _ = _train_sbert_detector(split.train, args)
    if _rerank_config_from_args(args) is None:
        return detector.evaluate(split.test, candidates=split.train)
    return _evaluate_detector(detector, split.test, split.train, args)


def _evaluate_detector(detector, queries: list[TicketRecord], candidates: list[TicketRecord], args: argparse.Namespace) -> tuple[float, list[EvaluationRow]]:
    rerank_config = _rerank_config_from_args(args)
    if rerank_config is None:
        return detector.evaluate(queries, candidates=candidates)

    rows: list[EvaluationRow] = []
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidates)
        if not relevant:
            continue
        ranking = detector.rank(query, candidates)
        ranking = rerank_ranked_tickets(query, ranking, candidates, rerank_config)
        ranked_ids = tuple(row.ticket_id for row in ranking)
        rows.append(
            EvaluationRow(
                query_id=query.ticket_id,
                average_precision=average_precision(ranked_ids, relevant),
                relevant_ids=tuple(sorted(relevant)),
                ranked_ids=ranked_ids,
            )
        )

    if not rows:
        return 0.0, []
    return sum(row.average_precision for row in rows) / len(rows), rows


def _train_sbert_detector(
    tickets: list[TicketRecord],
    args: argparse.Namespace,
) -> tuple[SbertDuplicateDetector, list, list]:
    seed = getattr(args, "seed", 13)
    negative_strategy = getattr(args, "negative_strategy", "all")
    negatives_per_anchor = getattr(args, "negatives_per_anchor", None)
    title_triplets = build_triplets(
        tickets,
        field="title",
        max_triplets=args.max_triplets,
        seed=seed,
        negative_strategy=negative_strategy,
        negatives_per_anchor=negatives_per_anchor,
    )
    content_triplets = build_triplets(
        tickets,
        field="content",
        max_triplets=args.max_triplets,
        seed=seed,
        negative_strategy=negative_strategy,
        negatives_per_anchor=negatives_per_anchor,
    )
    config = SbertTrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        warmup_steps=args.warmup_steps,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    detector = SbertDuplicateDetector(combine=args.combine, config=config)
    detector.fit(title_triplets=title_triplets, content_triplets=content_triplets)
    return detector, title_triplets, content_triplets


def _collect_threshold_decisions(detector, queries: list[TicketRecord], candidates: list[TicketRecord], args: argparse.Namespace) -> list[QueryDecision]:
    if isinstance(detector, TfidfDuplicateDetector):
        title_scores, content_scores = _tfidf_score_matrices(detector, queries, candidates)
    elif isinstance(detector, SbertDuplicateDetector):
        title_scores, content_scores = _sbert_score_matrices(detector, queries, candidates)
    else:
        return collect_query_decisions(
            detector,
            queries,
            candidates,
            rerank_config=_rerank_config_from_args(args),
        )

    scores = combine_similarity_scores(title_scores, content_scores, args.combine)
    rerank_config = _rerank_config_from_args(args)
    if rerank_config is not None:
        scores = apply_metadata_rerank_scores(
            queries,
            candidates,
            scores,
            title_scores,
            content_scores,
            rerank_config,
        )
    return _query_decisions_from_scores(queries, candidates, scores, title_scores=title_scores, content_scores=content_scores)


def _tfidf_score_matrices(detector: TfidfDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    query_titles = detector.title_vectorizer.transform(_safe_texts(ticket.title for ticket in queries))
    candidate_titles = detector.title_vectorizer.transform(_safe_texts(ticket.title for ticket in candidates))
    query_contents = detector.content_vectorizer.transform(_safe_texts(ticket.content for ticket in queries))
    candidate_contents = detector.content_vectorizer.transform(_safe_texts(ticket.content for ticket in candidates))
    return (query_titles @ candidate_titles.T).toarray(), (query_contents @ candidate_contents.T).toarray()


def _sbert_score_matrices(detector: SbertDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    detector._check_fitted()
    print(f"encoding_titles queries={len(queries)} candidates={len(candidates)}", flush=True)
    title_queries = detector.title_model.encode(
        [ticket.title for ticket in queries],
        batch_size=detector.config.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(queries) > 64,
    )
    title_candidates = detector.title_model.encode(
        [ticket.title for ticket in candidates],
        batch_size=detector.config.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(candidates) > 64,
    )
    print(f"encoding_contents queries={len(queries)} candidates={len(candidates)}", flush=True)
    content_queries = detector.content_model.encode(
        [ticket.content for ticket in queries],
        batch_size=detector.config.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(queries) > 64,
    )
    content_candidates = detector.content_model.encode(
        [ticket.content for ticket in candidates],
        batch_size=detector.config.batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(candidates) > 64,
    )
    return np.asarray(title_queries @ title_candidates.T), np.asarray(content_queries @ content_candidates.T)


def _query_decisions_from_scores(
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    scores: np.ndarray,
    *,
    title_scores: np.ndarray | None = None,
    content_scores: np.ndarray | None = None,
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
        best_candidate = candidates[best_index] if best_index >= 0 else None
        best_candidate_id = "" if best_candidate is None else best_candidate.ticket_id
        best_score = float("-inf") if best_index < 0 else float(query_scores[best_index])
        second_score = float("-inf") if second_index < 0 else float(query_scores[second_index])
        best_title_score = 0.0 if title_scores is None or best_index < 0 else float(title_scores[query_index, best_index])
        best_content_score = 0.0 if content_scores is None or best_index < 0 else float(content_scores[query_index, best_index])
        decisions.append(
            QueryDecision(
                query_id=query.ticket_id,
                best_candidate_id=best_candidate_id,
                score=best_score,
                is_duplicate=bool(relevant),
                best_is_correct_duplicate=best_candidate_id in relevant,
                title_score=best_title_score,
                content_score=best_content_score,
                second_score=second_score,
            )
        )
    return decisions


def write_query_decisions_csv(
    decisions: list[QueryDecision],
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    *,
    threshold: float,
    output_path: str | Path,
) -> None:
    query_by_id = {ticket.ticket_id: ticket for ticket in queries}
    candidate_by_id = {ticket.ticket_id: ticket for ticket in candidates}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "actual_is_duplicate",
        "predicted_is_duplicate",
        "outcome",
        "best_candidate_id",
        "score",
        "title_score",
        "content_score",
        "second_score",
        "score_margin",
        "threshold",
        "best_is_correct_duplicate",
        "query_title",
        "best_title",
        "query_resolution",
        "best_resolution",
        "query_product",
        "best_product",
        "product_match",
        "query_component",
        "best_component",
        "component_match",
        "query_severity",
        "best_severity",
        "severity_match",
        "query_priority",
        "best_priority",
        "priority_match",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision in decisions:
            query = query_by_id.get(decision.query_id)
            candidate = candidate_by_id.get(decision.best_candidate_id)
            predicted = decision.score >= threshold
            score_margin = decision.score - decision.second_score
            writer.writerow(
                {
                    "query_id": decision.query_id,
                    "actual_is_duplicate": str(decision.is_duplicate).lower(),
                    "predicted_is_duplicate": str(predicted).lower(),
                    "outcome": _decision_outcome(actual=decision.is_duplicate, predicted=predicted),
                    "best_candidate_id": decision.best_candidate_id,
                    "score": "" if not np.isfinite(decision.score) else f"{decision.score:.6f}",
                    "title_score": f"{decision.title_score:.6f}",
                    "content_score": f"{decision.content_score:.6f}",
                    "second_score": "" if not np.isfinite(decision.second_score) else f"{decision.second_score:.6f}",
                    "score_margin": "" if not np.isfinite(score_margin) else f"{score_margin:.6f}",
                    "threshold": f"{threshold:.6f}",
                    "best_is_correct_duplicate": str(decision.best_is_correct_duplicate).lower(),
                    "query_title": "" if query is None else query.title,
                    "best_title": "" if candidate is None else candidate.title,
                    "query_resolution": _field_value(query, "resolution"),
                    "best_resolution": _field_value(candidate, "resolution"),
                    "query_product": _field_value(query, "product"),
                    "best_product": _field_value(candidate, "product"),
                    "product_match": str(_same_field(query, candidate, "product")).lower(),
                    "query_component": _field_value(query, "component"),
                    "best_component": _field_value(candidate, "component"),
                    "component_match": str(_same_field(query, candidate, "component")).lower(),
                    "query_severity": _field_value(query, "severity"),
                    "best_severity": _field_value(candidate, "severity"),
                    "severity_match": str(_same_field(query, candidate, "severity")).lower(),
                    "query_priority": _field_value(query, "priority"),
                    "best_priority": _field_value(candidate, "priority"),
                    "priority_match": str(_same_field(query, candidate, "priority")).lower(),
                }
            )


def _decision_outcome(*, actual: bool, predicted: bool) -> str:
    if actual and predicted:
        return "true_positive"
    if actual and not predicted:
        return "false_negative"
    if not actual and predicted:
        return "false_positive"
    return "true_negative"


def _same_field(left: TicketRecord | None, right: TicketRecord | None, field: str) -> bool:
    left_value = _field_value(left, field).strip().lower()
    right_value = _field_value(right, field).strip().lower()
    return bool(left_value and left_value == right_value)


def _field_value(ticket: TicketRecord | None, field: str) -> str:
    if ticket is None or not ticket.fields:
        return ""
    return str(ticket.fields.get(field, ""))


def _safe_texts(texts) -> list[str]:
    values = [(text or "").strip() for text in texts]
    return [value if value else "__empty__" for value in values]


def _count_evaluable_queries(test: list[TicketRecord], train: list[TicketRecord]) -> int:
    return sum(1 for query in test if relevant_duplicate_ids(query, train))


def _print_evaluation(
    *,
    tickets_count: int,
    mean_ap: float,
    rows: list[EvaluationRow],
    top_k: int,
    details: bool = True,
) -> None:
    summary = _summarize_rows(mean_ap=mean_ap, rows=rows, top_k=top_k)

    print(f"tickets={tickets_count}")
    print(f"queries_with_duplicates={len(rows)}")
    print(f"MAP={summary['MAP']:.4f}")
    print(f"top_1_accuracy={summary['top_1_accuracy']:.4f}")
    print(f"top_{max(top_k, 1)}_hit_rate={summary['top_k_hit_rate']:.4f}")
    print(f"MRR={summary['MRR']:.4f}")
    if details:
        _print_query_rows(rows, top_k=top_k)


def _print_query_rows(rows: list[EvaluationRow], *, top_k: int) -> None:
    top_k = max(top_k, 1)
    print("query_id,AP,first_duplicate_rank,relevant_ids,top_ranked_ids")
    for row in rows:
        top_ranked = " ".join(row.ranked_ids[:top_k])
        relevant = " ".join(row.relevant_ids)
        print(f"{row.query_id},{row.average_precision:.4f},{_first_duplicate_rank(row)},{relevant},{top_ranked}")


def _print_ranking(ranking) -> None:
    print("rank,ticket_id,score,title_score,content_score,title")
    for index, row in enumerate(ranking, start=1):
        print(
            f"{index},{row.ticket_id},{row.score:.4f},{row.title_score:.4f},"
            f"{row.content_score:.4f},{_csv_cell(row.title)}"
        )


def _print_threshold_decision(ranking, threshold: float | None, *, include_best: bool = False) -> None:
    if threshold is None:
        return
    best = ranking[0] if ranking else None
    is_duplicate = bool(best and best.score >= threshold)
    print(f"is_duplicate={str(is_duplicate).lower()}")
    print(f"threshold={threshold:.4f}")
    if include_best and best:
        print(f"best_candidate_id={best.ticket_id}")
        print(f"best_score={best.score:.4f}")
        print(f"best_title_score={best.title_score:.4f}")
        print(f"best_content_score={best.content_score:.4f}")
        print(f"best_title={_csv_cell(best.title)}")


def _print_decision_metrics(metrics: DecisionMetrics) -> None:
    print("confusion_matrix=query_level")
    print(f"queries={metrics.queries}")
    print(f"positive_queries={metrics.positive_queries}")
    if metrics.queries and metrics.positive_queries == metrics.queries:
        print("warning=no_negative_queries; false_positives_cannot_be_estimated_from_this_split")
    elif metrics.positive_queries == 0:
        print("warning=no_positive_queries; recall_cannot_be_estimated_from_this_split")
    print(f"threshold={metrics.threshold:.6f}")
    print(f"precision={metrics.precision:.4f}")
    print(f"recall={metrics.recall:.4f}")
    print(f"f1={metrics.f1:.4f}")
    print(f"accuracy={metrics.accuracy:.4f}")
    print(f"true_positives={metrics.true_positives}")
    print(f"false_positives={metrics.false_positives}")
    print(f"true_negatives={metrics.true_negatives}")
    print(f"false_negatives={metrics.false_negatives}")
    print(f"correct_duplicate_links={metrics.correct_duplicate_links}")
    print("matrix_actual_by_predicted")
    print("actual,predicted_non_duplicate,predicted_duplicate")
    print(f"non_duplicate,{metrics.true_negatives},{metrics.false_positives}")
    print(f"duplicate,{metrics.false_negatives},{metrics.true_positives}")


def _summarize_rows(*, mean_ap: float, rows: list[EvaluationRow], top_k: int) -> dict[str, float]:
    top_k = max(top_k, 1)
    top_1_hits = sum(1 for row in rows if row.ranked_ids and row.ranked_ids[0] in row.relevant_ids)
    top_k_hits = sum(1 for row in rows if set(row.ranked_ids[:top_k]) & set(row.relevant_ids))
    reciprocal_ranks = [_reciprocal_rank(row) for row in rows]
    return {
        "queries": float(len(rows)),
        "MAP": mean_ap,
        "top_1_accuracy": _ratio(top_1_hits, len(rows)),
        "top_k_hit_rate": _ratio(top_k_hits, len(rows)),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }


def _first_duplicate_rank(row: EvaluationRow) -> int | str:
    relevant = set(row.relevant_ids)
    for rank, ticket_id in enumerate(row.ranked_ids, start=1):
        if ticket_id in relevant:
            return rank
    return ""


def _reciprocal_rank(row: EvaluationRow) -> float:
    first_rank = _first_duplicate_rank(row)
    if not isinstance(first_rank, int):
        return 0.0
    return 1 / first_rank


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _csv_cell(value: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([value])
    return output.getvalue().strip()


if __name__ == "__main__":
    raise SystemExit(main())
