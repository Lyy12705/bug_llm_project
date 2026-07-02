#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.metrics import average_precision
from duplicate_ticket_detection.rerank import RerankConfig, apply_metadata_rerank_scores, parse_field_weights
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector, SbertTrainingConfig
from duplicate_ticket_detection.splits import DatasetSplit, kfold_ticket_splits
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector, combine_similarity_scores
from duplicate_ticket_detection.triplets import build_triplets

DEFAULT_COMBINES = ("max", "title75", "content75", "mean", "weighted:0.4,0.6", "weighted:0.6,0.4")
DEFAULT_SBERT_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
)
DEFAULT_RERANK_GRID = (
    "component:0.01",
    "component:0.02",
    "component:0.05",
    "component:0.02,product:0.005",
    "component:0.05,product:0.02",
    "component:0.02,product:0.01,severity:0.005,priority:0.005",
    "component:0.02,severity:0.005,priority:0.005",
    "product:0.02",
)

@dataclass(frozen=True)
class Metrics:
    queries: int
    mean_average_precision: float
    top_1_accuracy: float
    top_k_hit_rate: float
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float

@dataclass(frozen=True)
class ResultRow:
    experiment: str
    method: str
    combine: str
    fold: str
    train_size: int
    test_size: int
    queries: int
    mean_average_precision: float
    top_1_accuracy: float
    top_k_hit_rate: float
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    base_model: str = ""
    epochs: int = 0
    max_triplets: int = 0
    seconds: float = 0.0

def main() -> int:
    parser = argparse.ArgumentParser(description="Run duplicate-ticket experiment comparison.")
    parser.add_argument("--tickets", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--combines", nargs="+", default=list(DEFAULT_COMBINES))
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--sbert-base-models", nargs="+", default=list(DEFAULT_SBERT_MODELS))
    parser.add_argument("--sbert-epochs", nargs="+", type=int, default=[2])
    parser.add_argument("--max-triplets", type=int, default=2000)
    parser.add_argument("--negative-strategy", choices=("all", "random", "hard"), default="all")
    parser.add_argument("--negatives-per-anchor", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", help="Torch device for SBERT training, e.g. cpu, mps, or cuda")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later SBERT folds if one fold fails")
    parser.add_argument("--rerank", action="store_true", help="Also evaluate metadata reranking for each method/combine")
    parser.add_argument("--rerank-grid", action="store_true", help="Evaluate a small grid of metadata rerank weights")
    parser.add_argument(
        "--rerank-configs",
        nargs="*",
        help="Explicit rerank configs, e.g. component:0.02 product:0.02 or component:0.02,product:0.005",
    )
    parser.add_argument(
        "--rerank-fields",
        nargs="*",
        help="Metadata fields used by reranker. Each value can be field or field:weight. Defaults to component:0.02 product:0.01 severity:0.005 priority:0.005",
    )
    parser.add_argument("--rerank-field-weight", type=float, default=0.05)
    parser.add_argument("--rerank-candidate-pool", type=int, default=50)
    parser.add_argument("--rerank-title-weight", type=float, default=0.0)
    parser.add_argument("--rerank-content-weight", type=float, default=0.0)
    parser.add_argument("--output-csv", default="reports/duplicate_experiment_results.csv")
    parser.add_argument("--output-md", default="reports/duplicate_experiment_results.md")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    splits = kfold_ticket_splits(tickets, n_splits=args.folds, seed=args.seed)
    rerank_configs = rerank_configs_from_args(args)
    print(f"tickets={len(tickets)} folds={len(splits)} top_k={args.top_k}", flush=True)
    print(f"combines={','.join(args.combines)}", flush=True)
    if rerank_configs:
        print(f"rerank_configs={';'.join(label for label, _ in rerank_configs)}", flush=True)
        print(f"rerank_candidate_pool={rerank_configs[0][1].candidate_pool}", flush=True)

    rows: list[ResultRow] = []
    if not args.skip_tfidf:
        rows.extend(run_tfidf_experiments(splits=splits, combines=args.combines, top_k=args.top_k, rerank_configs=rerank_configs))
        save_results_checkpoint(rows, args.output_csv, args.output_md)
    if not args.skip_sbert:
        rows.extend(run_sbert_experiments(
            splits=splits,
            combines=args.combines,
            top_k=args.top_k,
            base_models=args.sbert_base_models,
            epochs_values=args.sbert_epochs,
            max_triplets=args.max_triplets,
            negative_strategy=args.negative_strategy,
            negatives_per_anchor=args.negatives_per_anchor,
            batch_size=args.batch_size,
            margin=args.margin,
            warmup_steps=args.warmup_steps,
            local_files_only=args.local_files_only,
            device=args.device,
            continue_on_error=args.continue_on_error,
            rerank_configs=rerank_configs,
            checkpoint=lambda current_rows: save_results_checkpoint(rows + current_rows, args.output_csv, args.output_md),
        ))

    write_results_csv(rows, args.output_csv)
    write_results_markdown(rows, args.output_md)
    print_current_results(rows)
    print(f"csv={args.output_csv}", flush=True)
    print(f"markdown={args.output_md}", flush=True)
    return 0

def run_tfidf_experiments(*, splits: list[DatasetSplit], combines: list[str], top_k: int, rerank_configs: list[tuple[str, RerankConfig]] | None = None) -> list[ResultRow]:
    rows: list[ResultRow] = []
    fold_rows: list[ResultRow] = []
    for split in splits:
        print(f"[TF-IDF] {split.name}: fitting", flush=True)
        started = time.perf_counter()
        detector = TfidfDuplicateDetector(combine="max").fit(split.train)
        title_scores, content_scores = tfidf_score_matrices(detector, split.test, split.train)
        seconds = time.perf_counter() - started
        for combine in combines:
            scores = combine_similarity_scores(title_scores, content_scores, combine)
            for variant_name, variant_combine, variant_scores in score_variants(
                split.test, split.train, scores, title_scores, content_scores, combine=combine, rerank_configs=rerank_configs
            ):
                metrics = score_matrix_metrics(split.test, split.train, variant_scores, top_k=top_k)
                row = ResultRow(
                    experiment=f"tfidf_{variant_name}", method="tfidf", combine=variant_combine, fold=split.name,
                    train_size=len(split.train), test_size=len(split.test), queries=metrics.queries,
                    mean_average_precision=metrics.mean_average_precision,
                    top_1_accuracy=metrics.top_1_accuracy,
                    top_k_hit_rate=metrics.top_k_hit_rate,
                    precision_at_k=metrics.precision_at_k,
                    recall_at_k=metrics.recall_at_k,
                    mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                    seconds=seconds,
                )
                rows.append(row)
                fold_rows.append(row)
                print_result(row)
    rows.extend(mean_rows(fold_rows))
    return rows

def run_sbert_experiments(
    *, splits: list[DatasetSplit], combines: list[str], top_k: int, base_models: list[str],
    epochs_values: list[int], max_triplets: int, negative_strategy: str, negatives_per_anchor: int | None,
    batch_size: int, margin: float,
    warmup_steps: int, local_files_only: bool, device: str | None,
    continue_on_error: bool = False, rerank_configs: list[tuple[str, RerankConfig]] | None = None, checkpoint=None,
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for base_model in base_models:
        for epochs in epochs_values:
            experiment_prefix = f"sbert_{model_slug(base_model)}_e{epochs}"
            fold_rows: list[ResultRow] = []
            for split in splits:
                print(f"[SBERT] {experiment_prefix} {split.name}: building triplets", flush=True)
                title_triplets = build_triplets(
                    split.train,
                    field="title",
                    max_triplets=max_triplets,
                    negative_strategy=negative_strategy,
                    negatives_per_anchor=negatives_per_anchor,
                )
                content_triplets = build_triplets(
                    split.train,
                    field="content",
                    max_triplets=max_triplets,
                    negative_strategy=negative_strategy,
                    negatives_per_anchor=negatives_per_anchor,
                )
                print(f"[SBERT] {experiment_prefix} {split.name}: title_triplets={len(title_triplets)} content_triplets={len(content_triplets)}", flush=True)
                detector = None
                try:
                    config = SbertTrainingConfig(
                        base_model=base_model, epochs=epochs, batch_size=batch_size, margin=margin,
                        warmup_steps=warmup_steps, local_files_only=local_files_only, device=device,
                    )
                    detector = SbertDuplicateDetector(combine="max", config=config)
                    started = time.perf_counter()
                    detector.fit(title_triplets=title_triplets, content_triplets=content_triplets)
                    title_scores, content_scores = sbert_score_matrices(detector, split.test, split.train)
                    seconds = time.perf_counter() - started
                    for combine in combines:
                        scores = combine_similarity_scores(title_scores, content_scores, combine)
                        for variant_name, variant_combine, variant_scores in score_variants(
                            split.test, split.train, scores, title_scores, content_scores, combine=combine, rerank_configs=rerank_configs
                        ):
                            metrics = score_matrix_metrics(split.test, split.train, variant_scores, top_k=top_k)
                            row = ResultRow(
                                experiment=f"{experiment_prefix}_{variant_name}", method="sbert", combine=variant_combine,
                                fold=split.name, train_size=len(split.train), test_size=len(split.test),
                                queries=metrics.queries, mean_average_precision=metrics.mean_average_precision,
                                top_1_accuracy=metrics.top_1_accuracy, top_k_hit_rate=metrics.top_k_hit_rate,
                                precision_at_k=metrics.precision_at_k, recall_at_k=metrics.recall_at_k,
                                mean_reciprocal_rank=metrics.mean_reciprocal_rank, base_model=base_model,
                                epochs=epochs, max_triplets=max_triplets, seconds=seconds,
                            )
                            rows.append(row)
                            fold_rows.append(row)
                            print_result(row)
                    if checkpoint:
                        checkpoint(rows + mean_rows(fold_rows))
                except Exception as exc:
                    print(f"[SBERT] {experiment_prefix} {split.name}: failed: {type(exc).__name__}: {exc}", flush=True)
                    if checkpoint:
                        checkpoint(rows + mean_rows(fold_rows))
                    if not continue_on_error:
                        raise
                finally:
                    if detector is not None:
                        del detector
                    gc.collect()
                    clear_torch_cache()
            fold_mean_rows = mean_rows(fold_rows)
            rows.extend(fold_mean_rows)
            if checkpoint:
                checkpoint(rows)
    return rows

def tfidf_score_matrices(detector: TfidfDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    query_titles = detector.title_vectorizer.transform(safe_texts(ticket.title for ticket in queries))
    candidate_titles = detector.title_vectorizer.transform(safe_texts(ticket.title for ticket in candidates))
    query_contents = detector.content_vectorizer.transform(safe_texts(ticket.content for ticket in queries))
    candidate_contents = detector.content_vectorizer.transform(safe_texts(ticket.content for ticket in candidates))
    return (query_titles @ candidate_titles.T).toarray(), (query_contents @ candidate_contents.T).toarray()

def sbert_score_matrices(detector: SbertDuplicateDetector, queries: list[TicketRecord], candidates: list[TicketRecord]) -> tuple[np.ndarray, np.ndarray]:
    title_queries = detector.title_model.encode([ticket.title for ticket in queries], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    title_candidates = detector.title_model.encode([ticket.title for ticket in candidates], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    content_queries = detector.content_model.encode([ticket.content for ticket in queries], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    content_candidates = detector.content_model.encode([ticket.content for ticket in candidates], batch_size=detector.config.batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(title_queries @ title_candidates.T), np.asarray(content_queries @ content_candidates.T)

def score_variants(
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    base_scores: np.ndarray,
    title_scores: np.ndarray,
    content_scores: np.ndarray,
    *,
    combine: str,
    rerank_configs: list[tuple[str, RerankConfig]] | None,
) -> list[tuple[str, str, np.ndarray]]:
    variants = [(combine, combine, base_scores)]
    for label, rerank_config in rerank_configs or []:
        reranked_scores = apply_metadata_rerank_scores(
            queries,
            candidates,
            base_scores,
            title_scores,
            content_scores,
            rerank_config,
        )
        variants.append((f"{combine}_{label}", f"{combine}+{label}", reranked_scores))
    return variants

def score_matrix_metrics(queries: list[TicketRecord], candidates: list[TicketRecord], scores: np.ndarray, *, top_k: int) -> Metrics:
    candidate_ids = [ticket.ticket_id for ticket in candidates]
    ap_scores: list[float] = []
    precision_at_k_scores: list[float] = []
    recall_at_k_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    top_1_hits = 0
    top_k_hits = 0
    for query_index, query in enumerate(queries):
        relevant = relevant_duplicate_ids(query, candidates)
        if not relevant:
            continue
        query_scores = scores[query_index].copy()
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                query_scores[candidate_index] = -np.inf
        ranked_ids = [candidate_ids[index] for index in np.argsort(-query_scores)]
        ap_scores.append(average_precision(ranked_ids, relevant))
        top_k_ids = ranked_ids[:top_k]
        hits_at_k = len(set(top_k_ids) & set(relevant))
        precision_at_k_scores.append(ratio(hits_at_k, len(top_k_ids)))
        recall_at_k_scores.append(ratio(hits_at_k, len(relevant)))
        first_rank = first_relevant_rank(ranked_ids, relevant)
        reciprocal_ranks.append(0.0 if first_rank is None else 1 / first_rank)
        if ranked_ids and ranked_ids[0] in relevant:
            top_1_hits += 1
        if hits_at_k:
            top_k_hits += 1
    queries_count = len(ap_scores)
    return Metrics(
        queries=queries_count,
        mean_average_precision=mean(ap_scores),
        top_1_accuracy=ratio(top_1_hits, queries_count),
        top_k_hit_rate=ratio(top_k_hits, queries_count),
        precision_at_k=mean(precision_at_k_scores),
        recall_at_k=mean(recall_at_k_scores),
        mean_reciprocal_rank=mean(reciprocal_ranks),
    )

def mean_rows(rows: list[ResultRow]) -> list[ResultRow]:
    grouped: dict[tuple[str, str, str, str, int, int], list[ResultRow]] = {}
    for row in rows:
        if row.fold == "mean":
            continue
        key = (row.experiment, row.method, row.combine, row.base_model, row.epochs, row.max_triplets)
        grouped.setdefault(key, []).append(row)
    means: list[ResultRow] = []
    for (experiment, method, combine, base_model, epochs, max_triplets), group_rows in grouped.items():
        evaluable = [row for row in group_rows if row.queries > 0]
        source_rows = evaluable or group_rows
        means.append(ResultRow(
            experiment=experiment, method=method, combine=combine, fold="mean",
            train_size=0, test_size=0,
            queries=round(sum(row.queries for row in source_rows) / len(source_rows)),
            mean_average_precision=mean(row.mean_average_precision for row in source_rows),
            top_1_accuracy=mean(row.top_1_accuracy for row in source_rows),
            top_k_hit_rate=mean(row.top_k_hit_rate for row in source_rows),
            precision_at_k=mean(row.precision_at_k for row in source_rows),
            recall_at_k=mean(row.recall_at_k for row in source_rows),
            mean_reciprocal_rank=mean(row.mean_reciprocal_rank for row in source_rows),
            base_model=base_model, epochs=epochs, max_triplets=max_triplets,
            seconds=sum(row.seconds for row in group_rows),
        ))
    return means

def write_results_csv(rows: list[ResultRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ResultRow.__dataclass_fields__.keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})

def write_results_markdown(rows: list[ResultRow], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_results_markdown(rows) + "\n", encoding="utf-8")

def save_results_checkpoint(rows: list[ResultRow], output_csv: str | Path, output_md: str | Path) -> None:
    write_results_csv(rows, output_csv)
    write_results_markdown(rows, output_md)
    print(f"checkpoint_saved rows={len(rows)} csv={output_csv} markdown={output_md}", flush=True)

def render_results_markdown(rows: list[ResultRow]) -> str:
    mean_results = sorted([row for row in rows if row.fold == "mean"], key=lambda row: row.mean_average_precision, reverse=True)
    lines = [
        "# Duplicate Ticket Experiment Results",
        "",
        "| Rank | Experiment | Method | Combine | Base model | Epochs | MAP | Top-1 | Top-k hit | Recall | MRR | Seconds |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(mean_results, start=1):
        lines.append(
            f"| {index} | {row.experiment} | {row.method} | {row.combine} | {row.base_model or '-'} | "
            f"{row.epochs or '-'} | {row.mean_average_precision:.4f} | {row.top_1_accuracy:.4f} | "
            f"{row.top_k_hit_rate:.4f} | {row.recall_at_k:.4f} | "
            f"{row.mean_reciprocal_rank:.4f} | {row.seconds:.1f} |"
        )
    return "\n".join(lines)

def print_current_results(rows: list[ResultRow]) -> None:
    print("", flush=True)
    print(render_results_markdown(rows), flush=True)
    print("", flush=True)

def print_result(row: ResultRow) -> None:
    print(
        f"{row.fold},{row.experiment},queries={row.queries},MAP={row.mean_average_precision:.4f},"
        f"top1={row.top_1_accuracy:.4f},topk={row.top_k_hit_rate:.4f},"
        f"precision={row.precision_at_k:.4f},recall={row.recall_at_k:.4f},MRR={row.mean_reciprocal_rank:.4f}",
        flush=True,
    )

def first_relevant_rank(ranked_ids: list[str], relevant_ids: Iterable[str]) -> int | None:
    relevant = set(relevant_ids)
    for rank, ticket_id in enumerate(ranked_ids, start=1):
        if ticket_id in relevant:
            return rank
    return None

def safe_texts(texts: Iterable[str]) -> list[str]:
    values = [(text or "").strip() for text in texts]
    return [value if value else "__empty__" for value in values]

def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)

def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator

def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")

def clear_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

def rerank_configs_from_args(args: argparse.Namespace) -> list[tuple[str, RerankConfig]]:
    if not args.rerank and not args.rerank_grid and not args.rerank_configs:
        return []

    if args.rerank_configs:
        specs = args.rerank_configs
    elif args.rerank_grid:
        specs = list(DEFAULT_RERANK_GRID)
    else:
        field_weights = parse_field_weights(args.rerank_fields, default_weight=args.rerank_field_weight)
        config = RerankConfig(
            field_weights=field_weights,
            candidate_pool=args.rerank_candidate_pool,
            title_weight=args.rerank_title_weight,
            content_weight=args.rerank_content_weight,
        )
        return [(rerank_label(field_weights), config)]

    configs: list[tuple[str, RerankConfig]] = []
    for spec in specs:
        field_weights = parse_field_weights(spec.split(","), default_weight=args.rerank_field_weight)
        configs.append(
            (
                rerank_label(field_weights),
                RerankConfig(
                    field_weights=field_weights,
                    candidate_pool=args.rerank_candidate_pool,
                    title_weight=args.rerank_title_weight,
                    content_weight=args.rerank_content_weight,
                ),
            )
        )
    return configs

def format_field_weights(field_weights: Iterable[tuple[str, float]]) -> str:
    return ",".join(f"{field}:{weight:g}" for field, weight in field_weights)

def rerank_label(field_weights: Iterable[tuple[str, float]]) -> str:
    parts = []
    for field, weight in field_weights:
        safe_field = field.replace("-", "_").replace(".", "_")
        safe_weight = f"{weight:g}".replace(".", "p")
        parts.append(f"{safe_field}{safe_weight}")
    return "rerank_" + "_".join(parts)

if __name__ == "__main__":
    raise SystemExit(main())
