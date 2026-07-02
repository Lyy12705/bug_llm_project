#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from duplicate_ticket_detection.dataset import TicketRecord, load_tickets, relevant_duplicate_ids
from duplicate_ticket_detection.rerank import RerankConfig, parse_field_weights, rerank_ranked_tickets
from duplicate_ticket_detection.sbert_detector import SbertDuplicateDetector
from duplicate_ticket_detection.splits import train_test_split_tickets
from duplicate_ticket_detection.tfidf_detector import RankedTicket, TfidfDuplicateDetector


METADATA_FIELDS = ("product", "component", "severity", "priority")


@dataclass(frozen=True)
class StabilityFeature:
    model_rank: int | None = None
    model_score: float = 0.0
    tfidf_rank: int | None = None
    tfidf_score: float = 0.0
    stable_score: float = 0.0
    rank_agreement: str = "single_signal"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Top-k duplicate recommendations for engineer review instead of automatic classification."
    )
    parser.add_argument("--tickets", default="data/mozilla_firefox_duplicates.csv")
    parser.add_argument("--method", choices=("tfidf", "sbert"), default="tfidf")
    parser.add_argument("--model-dir", help="Saved SBERT model directory. Required when --method sbert")
    parser.add_argument("--combine", default="weighted:0.6,0.4")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-ticket-id", help="Export recommendations for one existing ticket")
    parser.add_argument("--query-json", help="Export recommendations for one new ticket JSON")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply metadata reranking. Use --no-rerank to disable.",
    )
    parser.add_argument("--rerank-fields", nargs="*", default=["component:0.02"])
    parser.add_argument("--rerank-candidate-pool", type=int, default=50)
    parser.add_argument(
        "--stable-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blend the primary model ranking with TF-IDF agreement and metadata signals for a steadier review order.",
    )
    parser.add_argument("--stable-candidate-pool", type=int, default=50)
    parser.add_argument("--stable-tfidf-weight", type=float, default=0.30)
    parser.add_argument("--stable-metadata-weight", type=float, default=0.08)
    parser.add_argument("--stable-agreement-weight", type=float, default=0.08)
    parser.add_argument("--output-csv", default="reports/duplicate_review_queue.csv")
    parser.add_argument("--summary-md", default="reports/duplicate_review_queue_summary.md")
    args = parser.parse_args()

    tickets = load_tickets(args.tickets)
    queries, candidates, detector, mode = prepare_inputs(args, tickets)
    rerank_config = build_rerank_config(args)
    stable_detector = build_stable_detector(args, candidates)
    rows = build_review_rows(
        detector,
        queries,
        candidates,
        top_k=args.top_k,
        rerank_config=rerank_config,
        stable_detector=stable_detector,
        stable_candidate_pool=args.stable_candidate_pool,
        stable_tfidf_weight=args.stable_tfidf_weight,
        stable_metadata_weight=args.stable_metadata_weight,
        stable_agreement_weight=args.stable_agreement_weight,
    )
    write_review_csv(rows, args.output_csv)
    write_summary(rows, args.summary_md, mode=mode, top_k=args.top_k)

    print(f"mode={mode}")
    print(f"queries={len(queries)}")
    print(f"candidates={len(candidates)}")
    print(f"recommendation_rows={len(rows)}")
    print(f"stable_rerank={str(stable_detector is not None).lower()}")
    print(f"output_csv={args.output_csv}")
    print(f"summary_md={args.summary_md}")
    return 0


def prepare_inputs(args: argparse.Namespace, tickets: list[TicketRecord]):
    if args.query_ticket_id or args.query_json:
        detector = build_detector(args, tickets)
        query = load_single_query(args, tickets)
        return [query], tickets, detector, "single_query"

    split = train_test_split_tickets(tickets, test_size=args.test_size, seed=args.seed)
    detector = build_detector(args, split.train)
    return split.test, split.train, detector, split.name


def build_detector(args: argparse.Namespace, train_tickets: list[TicketRecord]):
    if args.method == "tfidf":
        return TfidfDuplicateDetector(combine=args.combine).fit(train_tickets)
    if not args.model_dir:
        raise SystemExit("--model-dir is required when --method sbert")
    return SbertDuplicateDetector.load(args.model_dir, combine=args.combine)


def build_stable_detector(args: argparse.Namespace, candidates: list[TicketRecord]) -> TfidfDuplicateDetector | None:
    if not args.stable_rerank:
        return None
    return TfidfDuplicateDetector(combine=args.combine).fit(candidates)


def load_single_query(args: argparse.Namespace, tickets: list[TicketRecord]) -> TicketRecord:
    if args.query_ticket_id:
        for ticket in tickets:
            if ticket.ticket_id == args.query_ticket_id:
                return ticket
        raise SystemExit(f"query ticket not found: {args.query_ticket_id}")
    if args.query_json:
        return ticket_from_json(args.query_json)
    raise SystemExit("Provide --query-ticket-id or --query-json, or omit both to export a validation review queue.")


def ticket_from_json(path: str | Path) -> TicketRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("--query-json must contain a JSON object")
    title = str(data.get("title") or data.get("summary") or data.get("subject") or "")
    content = str(data.get("description") or data.get("content") or data.get("body") or "")
    ticket_id = str(data.get("ticket_id") or data.get("bug_id") or data.get("id") or "new_ticket")
    fields = {str(key): "" if value is None else str(value) for key, value in data.items()}
    return TicketRecord(
        ticket_id=ticket_id,
        title=title,
        content=content,
        duplicate_of=None,
        duplicate_group=None,
        fields=fields,
    )


def build_rerank_config(args: argparse.Namespace) -> RerankConfig | None:
    if not args.rerank:
        return None
    return RerankConfig(
        field_weights=parse_field_weights(args.rerank_fields),
        candidate_pool=args.rerank_candidate_pool,
    )


def build_review_rows(
    detector,
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    *,
    top_k: int,
    rerank_config: RerankConfig | None,
    stable_detector: TfidfDuplicateDetector | None = None,
    stable_candidate_pool: int = 50,
    stable_tfidf_weight: float = 0.30,
    stable_metadata_weight: float = 0.08,
    stable_agreement_weight: float = 0.08,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    top_k = max(1, top_k)
    candidate_by_id = {ticket.ticket_id: ticket for ticket in candidates}
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidates)
        pool_size = max(top_k, rerank_pool_size(rerank_config, top_k), stable_pool_size(stable_detector, stable_candidate_pool, top_k))
        ranking = detector.rank(query, candidates, top_k=pool_size)
        if rerank_config is not None:
            ranking = rerank_ranked_tickets(query, ranking, candidates, rerank_config, top_k=pool_size)
        features_by_id = model_features(ranking)
        if stable_detector is not None:
            ranking, features_by_id = stable_rerank_candidates(
                query,
                ranking,
                candidates,
                candidate_by_id,
                stable_detector,
                top_k=top_k,
                pool_size=pool_size,
                tfidf_weight=stable_tfidf_weight,
                metadata_weight=stable_metadata_weight,
                agreement_weight=stable_agreement_weight,
            )
        else:
            ranking = ranking[:top_k]
        rows.extend(review_rows_for_query(query, ranking, candidate_by_id, relevant, features_by_id=features_by_id))
    return rows


def stable_pool_size(stable_detector: TfidfDuplicateDetector | None, stable_candidate_pool: int, top_k: int) -> int:
    if stable_detector is None:
        return top_k
    return max(top_k, stable_candidate_pool)


def rerank_pool_size(rerank_config: RerankConfig | None, top_k: int) -> int:
    if rerank_config is None:
        return top_k
    if rerank_config.candidate_pool <= 0:
        return max(top_k, 50)
    return max(top_k, rerank_config.candidate_pool)


def stable_rerank_candidates(
    query: TicketRecord,
    model_ranking: list[RankedTicket],
    candidates: list[TicketRecord],
    candidate_by_id: dict[str, TicketRecord],
    stable_detector: TfidfDuplicateDetector,
    *,
    top_k: int,
    pool_size: int,
    tfidf_weight: float,
    metadata_weight: float,
    agreement_weight: float,
) -> tuple[list[RankedTicket], dict[str, StabilityFeature]]:
    tfidf_ranking = stable_detector.rank(query, candidates, top_k=pool_size)
    model_by_id = {row.ticket_id: row for row in model_ranking}
    tfidf_by_id = {row.ticket_id: row for row in tfidf_ranking}
    model_rank_by_id = {row.ticket_id: rank for rank, row in enumerate(model_ranking, start=1)}
    tfidf_rank_by_id = {row.ticket_id: rank for rank, row in enumerate(tfidf_ranking, start=1)}
    candidate_ids = list(dict.fromkeys([row.ticket_id for row in model_ranking] + [row.ticket_id for row in tfidf_ranking]))
    model_norm = normalized_score_map(model_by_id)
    tfidf_norm = normalized_score_map(tfidf_by_id)
    model_weight = max(0.0, 1.0 - tfidf_weight - metadata_weight - agreement_weight)

    ranked_rows: list[RankedTicket] = []
    features_by_id: dict[str, StabilityFeature] = {}
    for ticket_id in candidate_ids:
        candidate = candidate_by_id.get(ticket_id)
        if candidate is None:
            continue
        model_row = model_by_id.get(ticket_id)
        tfidf_row = tfidf_by_id.get(ticket_id)
        source_row = model_row or tfidf_row
        if source_row is None:
            continue
        model_rank = model_rank_by_id.get(ticket_id)
        tfidf_rank = tfidf_rank_by_id.get(ticket_id)
        metadata_fraction = metadata_match_fraction(query, candidate)
        agreement = rank_agreement_score(model_rank, tfidf_rank, pool_size=pool_size)
        stable_score = (
            model_weight * model_norm.get(ticket_id, 0.0)
            + tfidf_weight * tfidf_norm.get(ticket_id, 0.0)
            + metadata_weight * metadata_fraction
            + agreement_weight * agreement
        )
        features_by_id[ticket_id] = StabilityFeature(
            model_rank=model_rank,
            model_score=0.0 if model_row is None else model_row.score,
            tfidf_rank=tfidf_rank,
            tfidf_score=0.0 if tfidf_row is None else tfidf_row.score,
            stable_score=stable_score,
            rank_agreement=rank_agreement_label(model_rank, tfidf_rank),
        )
        ranked_rows.append(
            RankedTicket(
                ticket_id=ticket_id,
                score=stable_score,
                title_score=source_row.title_score,
                content_score=source_row.content_score,
                title=source_row.title,
            )
        )

    ranked_rows.sort(
        key=lambda row: (
            features_by_id[row.ticket_id].stable_score,
            -(features_by_id[row.ticket_id].model_rank or pool_size + 1),
            -(features_by_id[row.ticket_id].tfidf_rank or pool_size + 1),
        ),
        reverse=True,
    )
    return ranked_rows[:top_k], features_by_id


def model_features(ranking: list[RankedTicket]) -> dict[str, StabilityFeature]:
    return {
        row.ticket_id: StabilityFeature(
            model_rank=rank,
            model_score=row.score,
            stable_score=row.score,
            rank_agreement="primary_only",
        )
        for rank, row in enumerate(ranking, start=1)
    }


def normalized_score_map(rows_by_id: dict[str, RankedTicket]) -> dict[str, float]:
    if not rows_by_id:
        return {}
    scores = [row.score for row in rows_by_id.values()]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return {ticket_id: 1.0 for ticket_id in rows_by_id}
    return {ticket_id: (row.score - minimum) / (maximum - minimum) for ticket_id, row in rows_by_id.items()}


def rank_agreement_score(model_rank: int | None, tfidf_rank: int | None, *, pool_size: int) -> float:
    if model_rank is None or tfidf_rank is None:
        return 0.0
    distance = abs(model_rank - tfidf_rank)
    return max(0.0, 1.0 - distance / max(pool_size, 1))


def rank_agreement_label(model_rank: int | None, tfidf_rank: int | None) -> str:
    if model_rank is None:
        return "tfidf_only"
    if tfidf_rank is None:
        return "model_only"
    if model_rank <= 3 and tfidf_rank <= 3:
        return "both_top3"
    if model_rank <= 10 and tfidf_rank <= 10:
        return "both_top10"
    return "both_pool"


def review_rows_for_query(
    query: TicketRecord,
    ranking: list[RankedTicket],
    candidate_by_id: dict[str, TicketRecord],
    relevant: set[str],
    *,
    features_by_id: dict[str, StabilityFeature] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    first_score = ranking[0].score if ranking else 0.0
    second_score = ranking[1].score if len(ranking) > 1 else first_score
    top_margin = first_score - second_score
    relevant_in_top_k = bool(relevant and any(row.ticket_id in relevant for row in ranking))
    for rank, ranked in enumerate(ranking, start=1):
        candidate = candidate_by_id.get(ranked.ticket_id)
        matches = metadata_matches(query, candidate)
        features = (features_by_id or {}).get(ranked.ticket_id, StabilityFeature(model_rank=rank, model_score=ranked.score, stable_score=ranked.score))
        metadata_count = sum(matches.values())
        rows.append(
            {
                "query_id": query.ticket_id,
                "query_title": query.title,
                "query_product": field_value(query, "product"),
                "query_component": field_value(query, "component"),
                "rank": str(rank),
                "candidate_id": ranked.ticket_id,
                "candidate_title": ranked.title,
                "candidate_score": f"{ranked.score:.6f}",
                "stable_score": f"{features.stable_score:.6f}",
                "model_score": f"{features.model_score:.6f}",
                "model_rank": format_rank(features.model_rank),
                "tfidf_score": f"{features.tfidf_score:.6f}",
                "tfidf_rank": format_rank(features.tfidf_rank),
                "rank_agreement": features.rank_agreement,
                "title_score": f"{ranked.title_score:.6f}",
                "content_score": f"{ranked.content_score:.6f}",
                "top1_margin": f"{top_margin:.6f}",
                "candidate_product": field_value(candidate, "product"),
                "candidate_component": field_value(candidate, "component"),
                "product_match": str(matches["product"]).lower(),
                "component_match": str(matches["component"]).lower(),
                "severity_match": str(matches["severity"]).lower(),
                "priority_match": str(matches["priority"]).lower(),
                "metadata_match_count": str(metadata_count),
                "review_priority": review_priority(
                    rank=rank,
                    metadata_match_count=metadata_count,
                    top1_margin=top_margin,
                    rank_agreement=features.rank_agreement,
                ),
                "review_confidence": review_confidence(
                    rank=rank,
                    metadata_match_count=metadata_count,
                    top1_margin=top_margin,
                    rank_agreement=features.rank_agreement,
                ),
                "known_relevant_ids": " ".join(sorted(relevant)),
                "candidate_is_known_duplicate": str(ranked.ticket_id in relevant).lower(),
                "known_duplicate_in_top_k": str(relevant_in_top_k).lower(),
            }
        )
    return rows


def metadata_matches(query: TicketRecord, candidate: TicketRecord | None) -> dict[str, bool]:
    return {field: same_field(query, candidate, field) for field in METADATA_FIELDS}


def metadata_match_fraction(query: TicketRecord, candidate: TicketRecord | None) -> float:
    return ratio(sum(metadata_matches(query, candidate).values()), len(METADATA_FIELDS))


def review_priority(*, rank: int, metadata_match_count: int, top1_margin: float, rank_agreement: str = "") -> str:
    if rank == 1 and (metadata_match_count >= 2 or top1_margin >= 0.03 or rank_agreement in {"both_top3", "both_top10"}):
        return "high"
    if rank <= 3:
        return "medium"
    return "normal"


def review_confidence(*, rank: int, metadata_match_count: int, top1_margin: float, rank_agreement: str) -> str:
    if rank == 1 and rank_agreement == "both_top3" and metadata_match_count >= 2:
        return "strong"
    if rank <= 3 and rank_agreement in {"both_top3", "both_top10"}:
        return "moderate"
    if top1_margin < 0.02 and rank <= 3:
        return "needs_review"
    return "weak"


def write_review_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_title",
        "query_product",
        "query_component",
        "rank",
        "candidate_id",
        "candidate_title",
        "candidate_score",
        "stable_score",
        "model_score",
        "model_rank",
        "tfidf_score",
        "tfidf_rank",
        "rank_agreement",
        "title_score",
        "content_score",
        "top1_margin",
        "candidate_product",
        "candidate_component",
        "product_match",
        "component_match",
        "severity_match",
        "priority_match",
        "metadata_match_count",
        "review_priority",
        "review_confidence",
        "known_relevant_ids",
        "candidate_is_known_duplicate",
        "known_duplicate_in_top_k",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], output_path: str | Path, *, mode: str, top_k: int) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_summary(rows, mode=mode, top_k=top_k) + "\n", encoding="utf-8")


def render_summary(rows: list[dict[str, str]], *, mode: str, top_k: int) -> str:
    query_ids = sorted({row["query_id"] for row in rows})
    rank1_rows = [row for row in rows if row["rank"] == "1"]
    evaluable = [row for row in rank1_rows if row["known_relevant_ids"]]
    top1_hits = sum(1 for row in evaluable if row["candidate_is_known_duplicate"] == "true")
    topk_hits = sum(1 for row in evaluable if row["known_duplicate_in_top_k"] == "true")
    priority_counts = Counter(row["review_priority"] for row in rows)
    confidence_counts = Counter(row.get("review_confidence", "-") for row in rows)
    lines = [
        "# Duplicate Review Queue Summary",
        "",
        f"- mode={mode}",
        f"- queries={len(query_ids)}",
        f"- top_k={top_k}",
        f"- recommendation_rows={len(rows)}",
        f"- high_priority_rows={priority_counts.get('high', 0)}",
        f"- medium_priority_rows={priority_counts.get('medium', 0)}",
        f"- strong_confidence_rows={confidence_counts.get('strong', 0)}",
        f"- moderate_confidence_rows={confidence_counts.get('moderate', 0)}",
    ]
    if evaluable:
        lines.extend(
            [
                f"- evaluable_duplicate_queries={len(evaluable)}",
                f"- top1_hit_rate={ratio(top1_hits, len(evaluable)):.4f}",
                f"- topk_hit_rate={ratio(topk_hits, len(evaluable)):.4f}",
            ]
        )
    lines.extend(["", "## Review Priority", "", "| Priority | Count |", "|---|---:|"])
    for priority, count in priority_counts.most_common():
        lines.append(f"| {priority} | {count} |")
    lines.extend(["", "## Review Confidence", "", "| Confidence | Count |", "|---|---:|"])
    for confidence, count in confidence_counts.most_common():
        lines.append(f"| {confidence} | {count} |")
    return "\n".join(lines)


def field_value(ticket: TicketRecord | None, field: str) -> str:
    if ticket is None or not ticket.fields:
        return ""
    return str(ticket.fields.get(field, ""))


def same_field(left: TicketRecord | None, right: TicketRecord | None, field: str) -> bool:
    left_value = field_value(left, field).strip().lower()
    right_value = field_value(right, field).strip().lower()
    return bool(left_value and left_value == right_value)


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_rank(value: int | None) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
