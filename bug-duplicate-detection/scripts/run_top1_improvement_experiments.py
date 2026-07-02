#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.rerank import RerankConfig, apply_metadata_rerank_scores, parse_field_weights
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector, SbertTrainingConfig
from duplicate_ticket_detection.splits import DatasetSplit, kfold_ticket_splits
from duplicate_ticket_detection.stable_rerank import StableRerankConfig, apply_stable_rerank_scores
from duplicate_ticket_detection.tfidf_detector import TfidfDuplicateDetector, combine_similarity_scores
from duplicate_ticket_detection.triplets import TripletRecord, build_triplets

from run_duplicate_experiments import (
    ResultRow,
    clear_torch_cache,
    mean_rows,
    print_result,
    score_matrix_metrics,
    sbert_score_matrices,
    tfidf_score_matrices,
    write_results_csv,
    write_results_markdown,
)


DEFAULT_COMBINES = ("weighted:0.6,0.4", "mean", "title75", "content75")
DEFAULT_RERANK_CONFIGS = (
    "component:0.02",
    "component:0.02,product:0.01",
    "component:0.02,product:0.01,severity:0.005",
    "component:0.02,product:0.01,severity:0.005,priority:0.005",
    "component:0.05,product:0.02",
    "component:0.05,product:0.02,severity:0.01,priority:0.01",
)
DEFAULT_STABLE_RERANK_CONFIGS = (
    "tfidf:0.20,metadata:0.05,agreement:0.05",
    "tfidf:0.30,metadata:0.08,agreement:0.08",
    "tfidf:0.40,metadata:0.06,agreement:0.10",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Improve Top-1 duplicate ranking by mining wrong Top-1 candidates as hard negatives."
    )
    parser.add_argument("--tickets", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--combines", nargs="+", default=list(DEFAULT_COMBINES))
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
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--rerank-configs",
        nargs="*",
        default=list(DEFAULT_RERANK_CONFIGS),
        help="Metadata rerank configs, e.g. component:0.02,product:0.01",
    )
    parser.add_argument("--rerank-candidate-pool", type=int, default=50)
    parser.add_argument("--stable-rerank", action="store_true", help="Also evaluate TF-IDF/metadata agreement stable reranking")
    parser.add_argument("--stable-rerank-grid", action="store_true", help="Evaluate a small grid of stable rerank weights")
    parser.add_argument(
        "--stable-rerank-configs",
        nargs="*",
        help="Stable rerank configs, e.g. tfidf:0.30,metadata:0.08,agreement:0.08",
    )
    parser.add_argument("--stable-candidate-pool", type=int, default=50)
    parser.add_argument("--output-csv", default="reports/top1_improvement_experiment_results.csv")
    parser.add_argument("--output-md", default="reports/top1_improvement_experiment_results.md")
    parser.add_argument("--mined-negatives-csv", default="reports/top1_mined_hard_negatives.csv")
    parser.add_argument("--error-analysis-csv", default="reports/top1_improvement_error_analysis.csv")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    splits = kfold_ticket_splits(tickets, n_splits=args.folds, seed=args.seed)
    rerank_configs = rerank_configs_from_specs(args.rerank_configs, candidate_pool=args.rerank_candidate_pool)
    stable_configs = stable_configs_from_args(args)
    print(f"tickets={len(tickets)} folds={len(splits)} top_k={args.top_k}", flush=True)
    print(f"combines={','.join(args.combines)}", flush=True)
    print(f"mining_combine={args.mining_combine}", flush=True)
    print(f"rerank_configs={';'.join(label for label, _ in rerank_configs)}", flush=True)
    if stable_configs:
        print(f"stable_rerank_configs={';'.join(label for label, _ in stable_configs)}", flush=True)

    rows: list[ResultRow] = []
    mined_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []

    for split in splits:
        try:
            fold_rows, fold_mined, fold_errors = run_fold(split, args, rerank_configs, stable_configs)
            rows.extend(fold_rows)
            mined_rows.extend(fold_mined)
            error_rows.extend(fold_errors)
            save_outputs(rows, mined_rows, error_rows, args)
        except Exception as exc:
            print(f"[Top-1 improvement] {split.name}: failed: {type(exc).__name__}: {exc}", flush=True)
            save_outputs(rows, mined_rows, error_rows, args)
            if not args.continue_on_error:
                raise

    rows.extend(mean_rows(rows))
    save_outputs(rows, mined_rows, error_rows, args)
    print_current_best(rows)
    print(f"csv={args.output_csv}", flush=True)
    print(f"markdown={args.output_md}", flush=True)
    print(f"mined_negatives_csv={args.mined_negatives_csv}", flush=True)
    print(f"error_analysis_csv={args.error_analysis_csv}", flush=True)
    return 0


def run_fold(
    split: DatasetSplit,
    args: argparse.Namespace,
    rerank_configs: list[tuple[str, RerankConfig]],
    stable_configs: list[tuple[str, StableRerankConfig]],
) -> tuple[list[ResultRow], list[dict[str, str]], list[dict[str, str]]]:
    print(f"[Top-1 improvement] {split.name}: training first-pass SBERT", flush=True)
    first_pass = train_detector(split.train, args, mined_negatives_by_anchor={})
    try:
        title_scores, content_scores = sbert_score_matrices(first_pass, split.train, split.train)
        mined_rows = mine_wrong_top1_negatives(
            split,
            split.train,
            title_scores,
            content_scores,
            combine=args.mining_combine,
            top_k=args.top_k,
        )
    finally:
        del first_pass
        gc.collect()
        clear_torch_cache()

    mined_by_anchor = group_mined_negative_ids(mined_rows, limit=args.mined_negatives_per_anchor)
    print(
        f"[Top-1 improvement] {split.name}: mined_anchors={len(mined_by_anchor)} "
        f"mined_negatives={len(mined_rows)}",
        flush=True,
    )

    print(f"[Top-1 improvement] {split.name}: training SBERT with mined hard negatives", flush=True)
    started = time.perf_counter()
    detector = train_detector(split.train, args, mined_negatives_by_anchor=mined_by_anchor)
    seconds = time.perf_counter() - started
    try:
        test_title_scores, test_content_scores = sbert_score_matrices(detector, split.test, split.train)
        tfidf_detector = TfidfDuplicateDetector(combine="max").fit(split.train)
        tfidf_title_scores, tfidf_content_scores = tfidf_score_matrices(tfidf_detector, split.test, split.train)
        fold_rows, error_rows = evaluate_score_variants(
            split,
            args,
            test_title_scores,
            test_content_scores,
            tfidf_title_scores,
            tfidf_content_scores,
            rerank_configs,
            stable_configs,
            seconds=seconds,
        )
    finally:
        del detector
        gc.collect()
        clear_torch_cache()

    return fold_rows, mined_rows, error_rows


def train_detector(
    train: list[TicketRecord],
    args: argparse.Namespace,
    *,
    mined_negatives_by_anchor: dict[str, list[str]],
) -> SbertDuplicateDetector:
    title_base = build_triplets(
        train,
        field="title",
        max_triplets=args.max_triplets,
        seed=args.seed,
        negative_strategy=args.negative_strategy,
        negatives_per_anchor=args.negatives_per_anchor,
    )
    content_base = build_triplets(
        train,
        field="content",
        max_triplets=args.max_triplets,
        seed=args.seed,
        negative_strategy=args.negative_strategy,
        negatives_per_anchor=args.negatives_per_anchor,
    )
    title_mined = build_mined_negative_triplets(train, field="title", mined_negatives_by_anchor=mined_negatives_by_anchor)
    content_mined = build_mined_negative_triplets(train, field="content", mined_negatives_by_anchor=mined_negatives_by_anchor)
    title_triplets = capped_triplets(title_mined, title_base, max_triplets=args.max_triplets, seed=args.seed)
    content_triplets = capped_triplets(content_mined, content_base, max_triplets=args.max_triplets, seed=args.seed)
    print(
        f"[SBERT] title_triplets={len(title_triplets)} content_triplets={len(content_triplets)} "
        f"mined_title={len(title_mined)} mined_content={len(content_mined)}",
        flush=True,
    )
    config = SbertTrainingConfig(
        base_model=args.sbert_base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        warmup_steps=args.warmup_steps,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    detector = SbertDuplicateDetector(combine=args.mining_combine, config=config)
    return detector.fit(title_triplets=title_triplets, content_triplets=content_triplets)


def build_mined_negative_triplets(
    tickets: list[TicketRecord],
    *,
    field: str,
    mined_negatives_by_anchor: dict[str, list[str]],
) -> list[TripletRecord]:
    by_id = {ticket.ticket_id: ticket for ticket in tickets}
    triplets: list[TripletRecord] = []
    for anchor_id, negative_ids in mined_negatives_by_anchor.items():
        anchor = by_id.get(anchor_id)
        if anchor is None:
            continue
        positives = [
            ticket
            for ticket in tickets
            if ticket.ticket_id != anchor.ticket_id and ticket.duplicate_group == anchor.duplicate_group
        ]
        for negative_id in negative_ids:
            negative = by_id.get(negative_id)
            if negative is None or negative.duplicate_group == anchor.duplicate_group:
                continue
            for positive in positives:
                triplets.append(
                    TripletRecord(
                        anchor_id=anchor.ticket_id,
                        positive_id=positive.ticket_id,
                        negative_id=negative.ticket_id,
                        anchor_text=anchor.text_for(field),
                        positive_text=positive.text_for(field),
                        negative_text=negative.text_for(field),
                        field=field,
                    )
                )
    return dedupe_triplets(triplets)


def capped_triplets(
    priority_triplets: list[TripletRecord],
    base_triplets: list[TripletRecord],
    *,
    max_triplets: int | None,
    seed: int,
) -> list[TripletRecord]:
    priority = dedupe_triplets(priority_triplets)
    priority_keys = {triplet_key(priority_triplet) for priority_triplet in priority}
    base = [
        triplet
        for triplet in dedupe_triplets(base_triplets)
        if triplet_key(triplet) not in priority_keys
    ]
    if max_triplets is None:
        return priority + base
    if len(priority) >= max_triplets:
        return priority[:max_triplets]
    remaining = max_triplets - len(priority)
    rng = random.Random(seed)
    if len(base) > remaining:
        base = rng.sample(base, remaining)
    return priority + base


def dedupe_triplets(triplets: Iterable[TripletRecord]) -> list[TripletRecord]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[TripletRecord] = []
    for triplet in triplets:
        key = triplet_key(triplet)
        if key in seen:
            continue
        seen.add(key)
        output.append(triplet)
    return output


def triplet_key(triplet: TripletRecord) -> tuple[str, str, str, str]:
    return (triplet.field, triplet.anchor_id, triplet.positive_id, triplet.negative_id)


def mine_wrong_top1_negatives(
    split: DatasetSplit,
    records: list[TicketRecord],
    title_scores: np.ndarray,
    content_scores: np.ndarray,
    *,
    combine: str,
    top_k: int,
) -> list[dict[str, str]]:
    scores = combine_similarity_scores(title_scores, content_scores, combine)
    candidate_ids = [ticket.ticket_id for ticket in records]
    candidate_by_id = {ticket.ticket_id: ticket for ticket in records}
    rows: list[dict[str, str]] = []
    for query_index, query in enumerate(records):
        relevant = relevant_duplicate_ids(query, records)
        if not relevant:
            continue
        query_scores = scores[query_index].copy()
        query_scores[query_index] = -np.inf
        ranked_indices = np.argsort(-query_scores)
        if not len(ranked_indices):
            continue
        best_index = int(ranked_indices[0])
        best_id = candidate_ids[best_index]
        if best_id in relevant:
            continue
        best_ticket = candidate_by_id[best_id]
        top_k_ids = [candidate_ids[index] for index in ranked_indices[:top_k]]
        rows.append(
            {
                "fold": split.name,
                "anchor_id": query.ticket_id,
                "positive_ids": " ".join(sorted(relevant)),
                "negative_id": best_id,
                "negative_score": f"{scores[query_index, best_index]:.6f}",
                "negative_title_score": f"{title_scores[query_index, best_index]:.6f}",
                "negative_content_score": f"{content_scores[query_index, best_index]:.6f}",
                "relevant_in_top_k": str(bool(set(top_k_ids) & set(relevant))).lower(),
                "anchor_product": field_value(query, "product"),
                "anchor_component": field_value(query, "component"),
                "negative_product": field_value(best_ticket, "product"),
                "negative_component": field_value(best_ticket, "component"),
                "product_match": str(same_field(query, best_ticket, "product")).lower(),
                "component_match": str(same_field(query, best_ticket, "component")).lower(),
                "anchor_title": query.title,
                "negative_title": best_ticket.title,
            }
        )
    return rows


def group_mined_negative_ids(rows: list[dict[str, str]], *, limit: int) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["anchor_id"]].append((float(row["negative_score"]), row["negative_id"]))
    output: dict[str, list[str]] = {}
    for anchor_id, scored_ids in grouped.items():
        seen: set[str] = set()
        output[anchor_id] = []
        for _, negative_id in sorted(scored_ids, reverse=True):
            if negative_id in seen:
                continue
            seen.add(negative_id)
            output[anchor_id].append(negative_id)
            if len(output[anchor_id]) >= limit:
                break
    return output


def evaluate_score_variants(
    split: DatasetSplit,
    args: argparse.Namespace,
    title_scores: np.ndarray,
    content_scores: np.ndarray,
    tfidf_title_scores: np.ndarray,
    tfidf_content_scores: np.ndarray,
    rerank_configs: list[tuple[str, RerankConfig]],
    stable_configs: list[tuple[str, StableRerankConfig]],
    *,
    seconds: float,
) -> tuple[list[ResultRow], list[dict[str, str]]]:
    fold_rows: list[ResultRow] = []
    error_rows: list[dict[str, str]] = []
    for combine in args.combines:
        base_scores = combine_similarity_scores(title_scores, content_scores, combine)
        tfidf_scores = combine_similarity_scores(tfidf_title_scores, tfidf_content_scores, combine)
        variants = [(combine, combine, base_scores)]
        for label, config in rerank_configs:
            reranked_scores = apply_metadata_rerank_scores(
                split.test,
                split.train,
                base_scores,
                title_scores,
                content_scores,
                config,
            )
            variants.append((f"{combine}_{label}", f"{combine}+{label}", reranked_scores))

        all_variants = list(variants)
        for variant_name, variant_combine, variant_scores in variants:
            for stable_label, stable_config in stable_configs:
                stable_scores = apply_stable_rerank_scores(
                    split.test,
                    split.train,
                    variant_scores,
                    tfidf_scores,
                    stable_config,
                )
                all_variants.append((f"{variant_name}_{stable_label}", f"{variant_combine}+{stable_label}", stable_scores))

        for variant_name, variant_combine, variant_scores in all_variants:
            metrics = score_matrix_metrics(split.test, split.train, variant_scores, top_k=args.top_k)
            row = ResultRow(
                experiment=f"sbert_top1_mined_{variant_name}",
                method="sbert",
                combine=variant_combine,
                fold=split.name,
                train_size=len(split.train),
                test_size=len(split.test),
                queries=metrics.queries,
                mean_average_precision=metrics.mean_average_precision,
                top_1_accuracy=metrics.top_1_accuracy,
                top_k_hit_rate=metrics.top_k_hit_rate,
                precision_at_k=metrics.precision_at_k,
                recall_at_k=metrics.recall_at_k,
                mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                base_model=args.sbert_base_model,
                epochs=args.epochs,
                max_triplets=args.max_triplets,
                seconds=seconds,
            )
            fold_rows.append(row)
            print_result(row)
            error_rows.extend(
                collect_top1_error_rows(
                    split,
                    variant_scores,
                    title_scores,
                    content_scores,
                    experiment=row.experiment,
                    combine=variant_combine,
                    top_k=args.top_k,
                )
            )
    return fold_rows, error_rows


def collect_top1_error_rows(
    split: DatasetSplit,
    scores: np.ndarray,
    title_scores: np.ndarray,
    content_scores: np.ndarray,
    *,
    experiment: str,
    combine: str,
    top_k: int,
) -> list[dict[str, str]]:
    candidate_ids = [ticket.ticket_id for ticket in split.train]
    candidate_by_id = {ticket.ticket_id: ticket for ticket in split.train}
    rows: list[dict[str, str]] = []
    for query_index, query in enumerate(split.test):
        relevant = relevant_duplicate_ids(query, split.train)
        if not relevant:
            continue
        query_scores = scores[query_index].copy()
        for candidate_index, candidate in enumerate(split.train):
            if candidate.ticket_id == query.ticket_id:
                query_scores[candidate_index] = -np.inf
        ranked_indices = np.argsort(-query_scores)
        if not len(ranked_indices):
            continue
        best_index = int(ranked_indices[0])
        best_id = candidate_ids[best_index]
        if best_id in relevant:
            continue
        best_ticket = candidate_by_id[best_id]
        top_k_ids = [candidate_ids[index] for index in ranked_indices[:top_k]]
        rows.append(
            {
                "fold": split.name,
                "experiment": experiment,
                "combine": combine,
                "query_id": query.ticket_id,
                "relevant_ids": " ".join(sorted(relevant)),
                "best_candidate_id": best_id,
                "best_score": f"{scores[query_index, best_index]:.6f}",
                "best_title_score": f"{title_scores[query_index, best_index]:.6f}",
                "best_content_score": f"{content_scores[query_index, best_index]:.6f}",
                "relevant_in_top_k": str(bool(set(top_k_ids) & set(relevant))).lower(),
                "query_product": field_value(query, "product"),
                "query_component": field_value(query, "component"),
                "best_product": field_value(best_ticket, "product"),
                "best_component": field_value(best_ticket, "component"),
                "product_match": str(same_field(query, best_ticket, "product")).lower(),
                "component_match": str(same_field(query, best_ticket, "component")).lower(),
                "query_title": query.title,
                "best_title": best_ticket.title,
                "top_ranked_ids": " ".join(top_k_ids),
            }
        )
    return rows


def rerank_configs_from_specs(specs: list[str], *, candidate_pool: int) -> list[tuple[str, RerankConfig]]:
    configs: list[tuple[str, RerankConfig]] = []
    for spec in specs:
        field_weights = parse_field_weights(spec.split(","))
        configs.append((rerank_label(field_weights), RerankConfig(field_weights=field_weights, candidate_pool=candidate_pool)))
    return configs


def stable_configs_from_args(args: argparse.Namespace) -> list[tuple[str, StableRerankConfig]]:
    if args.stable_rerank_configs:
        specs = args.stable_rerank_configs
    elif args.stable_rerank_grid:
        specs = list(DEFAULT_STABLE_RERANK_CONFIGS)
    elif args.stable_rerank:
        specs = ["tfidf:0.30,metadata:0.08,agreement:0.08"]
    else:
        return []

    configs: list[tuple[str, StableRerankConfig]] = []
    for spec in specs:
        config = parse_stable_config(spec, candidate_pool=args.stable_candidate_pool)
        configs.append((stable_label(config), config))
    return configs


def parse_stable_config(spec: str, *, candidate_pool: int) -> StableRerankConfig:
    values = {"tfidf": 0.30, "metadata": 0.08, "agreement": 0.08}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid stable rerank config: {spec!r}")
        name, value = part.split(":", maxsplit=1)
        name = name.strip().lower()
        if name not in values:
            raise ValueError(f"Unknown stable rerank weight: {name}")
        parsed = float(value)
        if parsed < 0:
            raise ValueError("Stable rerank weights must be non-negative")
        values[name] = parsed
    return StableRerankConfig(
        tfidf_weight=values["tfidf"],
        metadata_weight=values["metadata"],
        agreement_weight=values["agreement"],
        candidate_pool=candidate_pool,
    )


def stable_label(config: StableRerankConfig) -> str:
    return (
        "stable_"
        f"tfidf{format_weight(config.tfidf_weight)}_"
        f"metadata{format_weight(config.metadata_weight)}_"
        f"agreement{format_weight(config.agreement_weight)}"
    )


def rerank_label(field_weights: Iterable[tuple[str, float]]) -> str:
    return "rerank_" + "_".join(
        f"{field.replace('-', '_').replace('.', '_')}{weight:g}".replace(".", "p")
        for field, weight in field_weights
    )


def format_weight(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def save_outputs(
    rows: list[ResultRow],
    mined_rows: list[dict[str, str]],
    error_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    write_results_csv(rows, args.output_csv)
    write_results_markdown(rows, args.output_md)
    write_dict_rows(mined_rows, args.mined_negatives_csv)
    write_dict_rows(error_rows, args.error_analysis_csv)


def write_dict_rows(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with output.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_current_best(rows: list[ResultRow]) -> None:
    mean_rows_only = [row for row in rows if row.fold == "mean"]
    if not mean_rows_only:
        return
    best = max(mean_rows_only, key=lambda row: row.top_1_accuracy)
    print("", flush=True)
    print("Best Top-1 result", flush=True)
    print(
        f"experiment={best.experiment} combine={best.combine} "
        f"MAP={best.mean_average_precision:.4f} Top-1={best.top_1_accuracy:.4f} "
        f"Top-k={best.top_k_hit_rate:.4f} Recall={best.recall_at_k:.4f} MRR={best.mean_reciprocal_rank:.4f}",
        flush=True,
    )


def field_value(ticket: TicketRecord | None, field: str) -> str:
    if ticket is None:
        return ""
    return str((ticket.fields or {}).get(field, ""))


def same_field(left: TicketRecord | None, right: TicketRecord | None, field: str) -> bool:
    left_value = field_value(left, field).strip().lower()
    right_value = field_value(right, field).strip().lower()
    return bool(left_value and left_value == right_value)


if __name__ == "__main__":
    raise SystemExit(main())
