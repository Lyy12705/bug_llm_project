from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from duplicate_ticket_detection.dataset import TicketRecord, relevant_duplicate_ids
from duplicate_ticket_detection.rerank import RerankConfig, rerank_ranked_tickets


@dataclass(frozen=True)
class PairScore:
    query_id: str
    candidate_id: str
    score: float
    is_duplicate: bool


@dataclass(frozen=True)
class QueryDecision:
    query_id: str
    best_candidate_id: str
    score: float
    is_duplicate: bool
    best_is_correct_duplicate: bool
    title_score: float = 0.0
    content_score: float = 0.0
    second_score: float = float("-inf")


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    pairs: int
    positive_pairs: int


@dataclass(frozen=True)
class DecisionMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    queries: int
    positive_queries: int
    correct_duplicate_links: int


def collect_pair_scores(
    detector,
    queries: Iterable[TicketRecord],
    candidates: Iterable[TicketRecord],
    *,
    rerank_config: RerankConfig | None = None,
) -> list[PairScore]:
    """Collect query-candidate similarity scores for threshold calibration."""

    candidate_records = list(candidates)
    pair_scores: list[PairScore] = []
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidate_records)
        ranking = detector.rank(query, candidate_records)
        if rerank_config is not None:
            ranking = rerank_ranked_tickets(query, ranking, candidate_records, rerank_config)
        for ranked in ranking:
            pair_scores.append(
                PairScore(
                    query_id=query.ticket_id,
                    candidate_id=ranked.ticket_id,
                    score=ranked.score,
                    is_duplicate=ranked.ticket_id in relevant,
                )
            )
    return pair_scores


def collect_query_decisions(
    detector,
    queries: Iterable[TicketRecord],
    candidates: Iterable[TicketRecord],
    *,
    rerank_config: RerankConfig | None = None,
) -> list[QueryDecision]:
    """Collect each query's best candidate score for duplicate/non-duplicate decisions."""

    candidate_records = list(candidates)
    decisions: list[QueryDecision] = []
    for query in queries:
        relevant = relevant_duplicate_ids(query, candidate_records)
        ranking = detector.rank(query, candidate_records)
        if rerank_config is not None:
            ranking = rerank_ranked_tickets(query, ranking, candidate_records, rerank_config)
        best = ranking[0] if ranking else None
        second = ranking[1] if len(ranking) > 1 else None
        best_candidate_id = "" if best is None else best.ticket_id
        best_score = float("-inf") if best is None else best.score
        decisions.append(
            QueryDecision(
                query_id=query.ticket_id,
                best_candidate_id=best_candidate_id,
                score=best_score,
                is_duplicate=bool(relevant),
                best_is_correct_duplicate=best_candidate_id in relevant,
                title_score=0.0 if best is None else best.title_score,
                content_score=0.0 if best is None else best.content_score,
                second_score=float("-inf") if second is None else second.score,
            )
        )
    return decisions


def tune_similarity_threshold(pair_scores: Iterable[PairScore]) -> ThresholdMetrics:
    """Select the similarity threshold that maximizes pair-level F1."""

    records = sorted(pair_scores, key=lambda pair: pair.score, reverse=True)
    pairs = len(records)
    positive_pairs = sum(1 for pair in records if pair.is_duplicate)
    negative_pairs = pairs - positive_pairs

    if not records:
        return _threshold_metrics(
            threshold=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            pairs=0,
            positive_pairs=0,
        )

    if positive_pairs == 0:
        return _threshold_metrics(
            threshold=records[0].score + 1e-12,
            true_positives=0,
            false_positives=0,
            true_negatives=negative_pairs,
            false_negatives=0,
            pairs=pairs,
            positive_pairs=0,
        )

    true_positives = 0
    false_positives = 0
    best: ThresholdMetrics | None = None
    index = 0
    while index < pairs:
        threshold = records[index].score
        while index < pairs and records[index].score == threshold:
            if records[index].is_duplicate:
                true_positives += 1
            else:
                false_positives += 1
            index += 1

        false_negatives = positive_pairs - true_positives
        true_negatives = negative_pairs - false_positives
        candidate = _threshold_metrics(
            threshold=threshold,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            pairs=pairs,
            positive_pairs=positive_pairs,
        )
        if best is None or _is_better_threshold(candidate, best):
            best = candidate

    return best or _threshold_metrics(
        threshold=records[0].score,
        true_positives=0,
        false_positives=0,
        true_negatives=negative_pairs,
        false_negatives=positive_pairs,
        pairs=pairs,
        positive_pairs=positive_pairs,
    )


def tune_query_threshold(
    decisions: Iterable[QueryDecision],
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> DecisionMetrics:
    """Select a threshold for query-level duplicate/non-duplicate decisions."""

    _validate_metric_floor("min_precision", min_precision)
    _validate_metric_floor("min_recall", min_recall)
    records = sorted(decisions, key=lambda decision: decision.score, reverse=True)
    queries = len(records)
    positive_queries = sum(1 for decision in records if decision.is_duplicate)
    negative_queries = queries - positive_queries

    if not records:
        return _decision_metrics(
            threshold=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            queries=0,
            positive_queries=0,
            correct_duplicate_links=0,
        )

    if positive_queries == 0:
        return _decision_metrics(
            threshold=records[0].score + 1e-12,
            true_positives=0,
            false_positives=0,
            true_negatives=negative_queries,
            false_negatives=0,
            queries=queries,
            positive_queries=0,
            correct_duplicate_links=0,
        )

    true_positives = 0
    false_positives = 0
    correct_duplicate_links = 0
    best_unconstrained: DecisionMetrics | None = None
    best_constrained: DecisionMetrics | None = None
    index = 0
    while index < queries:
        threshold = records[index].score
        while index < queries and records[index].score == threshold:
            if records[index].is_duplicate:
                true_positives += 1
                if records[index].best_is_correct_duplicate:
                    correct_duplicate_links += 1
            else:
                false_positives += 1
            index += 1

        false_negatives = positive_queries - true_positives
        true_negatives = negative_queries - false_positives
        candidate = _decision_metrics(
            threshold=threshold,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            queries=queries,
            positive_queries=positive_queries,
            correct_duplicate_links=correct_duplicate_links,
        )
        if best_unconstrained is None or _is_better_decision_threshold(candidate, best_unconstrained):
            best_unconstrained = candidate
        if _passes_metric_floors(candidate, min_precision=min_precision, min_recall=min_recall):
            if best_constrained is None or _is_better_decision_threshold(candidate, best_constrained):
                best_constrained = candidate

    if best_constrained is not None:
        return best_constrained
    if best_unconstrained is not None:
        return best_unconstrained
    return _decision_metrics(
        threshold=records[0].score,
        true_positives=0,
        false_positives=0,
        true_negatives=negative_queries,
        false_negatives=positive_queries,
        queries=queries,
        positive_queries=positive_queries,
        correct_duplicate_links=0,
    )


def _threshold_metrics(
    *,
    threshold: float,
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    pairs: int,
    positive_pairs: int,
) -> ThresholdMetrics:
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    accuracy = _ratio(true_positives + true_negatives, pairs)
    return ThresholdMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        pairs=pairs,
        positive_pairs=positive_pairs,
    )


def _decision_metrics(
    *,
    threshold: float,
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    queries: int,
    positive_queries: int,
    correct_duplicate_links: int,
) -> DecisionMetrics:
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    accuracy = _ratio(true_positives + true_negatives, queries)
    return DecisionMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        queries=queries,
        positive_queries=positive_queries,
        correct_duplicate_links=correct_duplicate_links,
    )


def _is_better_threshold(candidate: ThresholdMetrics, current: ThresholdMetrics) -> bool:
    candidate_key = (
        candidate.f1,
        candidate.precision,
        candidate.recall,
        candidate.accuracy,
        candidate.threshold,
    )
    current_key = (
        current.f1,
        current.precision,
        current.recall,
        current.accuracy,
        current.threshold,
    )
    return candidate_key > current_key


def _is_better_decision_threshold(candidate: DecisionMetrics, current: DecisionMetrics) -> bool:
    candidate_key = (
        candidate.f1,
        candidate.precision,
        candidate.recall,
        candidate.accuracy,
        candidate.correct_duplicate_links,
        candidate.threshold,
    )
    current_key = (
        current.f1,
        current.precision,
        current.recall,
        current.accuracy,
        current.correct_duplicate_links,
        current.threshold,
    )
    return candidate_key > current_key


def _passes_metric_floors(
    candidate: DecisionMetrics,
    *,
    min_precision: float | None,
    min_recall: float | None,
) -> bool:
    if min_precision is not None and candidate.precision < min_precision:
        return False
    if min_recall is not None and candidate.recall < min_recall:
        return False
    return True


def _validate_metric_floor(name: str, value: float | None) -> None:
    if value is None:
        return
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
