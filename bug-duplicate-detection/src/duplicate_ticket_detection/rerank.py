from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from duplicate_ticket_detection.dataset import TicketRecord
from duplicate_ticket_detection.tfidf_detector import RankedTicket


DEFAULT_RERANK_FIELDS = ("component:0.02", "product:0.01", "severity:0.005", "priority:0.005")
EMPTY_FIELD_VALUES = {"", "--", "-", "none", "null", "n/a", "na", "unknown"}


@dataclass(frozen=True)
class RerankConfig:
    """Metadata-aware second-stage ranking configuration.

    The first-stage detector still supplies the main similarity score. The
    reranker only adds small bonuses to candidates that match trusted metadata,
    such as the same Bugzilla component, so it can improve Top-1 without
    replacing the SBERT/TF-IDF retrieval model.
    """

    field_weights: tuple[tuple[str, float], ...] = (
        ("component", 0.02),
        ("product", 0.01),
        ("severity", 0.005),
        ("priority", 0.005),
    )
    candidate_pool: int = 50
    score_weight: float = 1.0
    title_weight: float = 0.0
    content_weight: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.field_weights or self.title_weight or self.content_weight)


def parse_field_weights(values: Iterable[str] | None, *, default_weight: float = 0.05) -> tuple[tuple[str, float], ...]:
    """Parse CLI field weights like ``component`` or ``component:0.08``."""

    parsed: list[tuple[str, float]] = []
    for raw_value in values or DEFAULT_RERANK_FIELDS:
        value = raw_value.strip()
        if not value:
            continue
        if ":" in value:
            field, weight_text = value.split(":", maxsplit=1)
            weight = float(weight_text)
        else:
            field = value
            weight = default_weight
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid rerank field: {raw_value!r}")
        if weight < 0:
            raise ValueError("Rerank field weights must be non-negative")
        parsed.append((field, weight))
    return tuple(parsed)


def apply_metadata_rerank_scores(
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    base_scores: np.ndarray,
    title_scores: np.ndarray,
    content_scores: np.ndarray,
    config: RerankConfig,
) -> np.ndarray:
    """Apply metadata reranking to a query-candidate score matrix."""

    if not config.enabled:
        return np.array(base_scores, copy=True)

    reranked = np.array(base_scores, copy=True)
    pool_size = _candidate_pool_size(config.candidate_pool, len(candidates))
    if pool_size == 0:
        return reranked

    for query_index, query in enumerate(queries):
        ordering_scores = np.array(base_scores[query_index], copy=True)
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                ordering_scores[candidate_index] = -np.inf

        for candidate_index in np.argsort(-ordering_scores)[:pool_size]:
            if not np.isfinite(ordering_scores[candidate_index]):
                continue
            candidate = candidates[candidate_index]
            reranked[query_index, candidate_index] = rerank_score(
                query,
                candidate,
                base_score=float(base_scores[query_index, candidate_index]),
                title_score=float(title_scores[query_index, candidate_index]),
                content_score=float(content_scores[query_index, candidate_index]),
                config=config,
            )

    return reranked


def rerank_ranked_tickets(
    query: TicketRecord,
    ranking: Iterable[RankedTicket],
    candidates: Iterable[TicketRecord],
    config: RerankConfig,
    *,
    top_k: int | None = None,
) -> list[RankedTicket]:
    """Rerank an already retrieved candidate list."""

    if not config.enabled:
        rows = list(ranking)
        return rows[:top_k] if top_k is not None else rows

    candidate_by_id = {ticket.ticket_id: ticket for ticket in candidates}
    pool_size = _candidate_pool_size(config.candidate_pool, len(candidate_by_id))
    rows = list(ranking)
    pool = rows[:pool_size]
    remainder = rows[pool_size:]
    reranked: list[RankedTicket] = []

    for ranked in pool:
        candidate = candidate_by_id.get(ranked.ticket_id)
        if candidate is None:
            continue
        score = rerank_score(
            query,
            candidate,
            base_score=ranked.score,
            title_score=ranked.title_score,
            content_score=ranked.content_score,
            config=config,
        )
        reranked.append(
            RankedTicket(
                ticket_id=ranked.ticket_id,
                score=score,
                title_score=ranked.title_score,
                content_score=ranked.content_score,
                title=ranked.title,
            )
        )

    reranked.sort(key=lambda row: row.score, reverse=True)
    output = reranked + remainder
    return output[:top_k] if top_k is not None else output


def rerank_score(
    query: TicketRecord,
    candidate: TicketRecord,
    *,
    base_score: float,
    title_score: float,
    content_score: float,
    config: RerankConfig,
) -> float:
    score = config.score_weight * base_score
    score += config.title_weight * title_score
    score += config.content_weight * content_score
    score += metadata_match_bonus(query, candidate, config.field_weights)
    return float(score)


def metadata_match_bonus(
    query: TicketRecord,
    candidate: TicketRecord,
    field_weights: Iterable[tuple[str, float]],
) -> float:
    bonus = 0.0
    for field, weight in field_weights:
        query_value = _normalized_field_value(query.fields, field)
        candidate_value = _normalized_field_value(candidate.fields, field)
        if query_value and query_value == candidate_value:
            bonus += weight
    return bonus


def _normalized_field_value(fields: Mapping[str, str] | None, field: str) -> str:
    if not fields:
        return ""
    value = str(fields.get(field, "")).strip().lower()
    return "" if value in EMPTY_FIELD_VALUES else value


def _candidate_pool_size(candidate_pool: int, candidates_count: int) -> int:
    if candidates_count <= 0:
        return 0
    if candidate_pool <= 0:
        return candidates_count
    return min(candidate_pool, candidates_count)
