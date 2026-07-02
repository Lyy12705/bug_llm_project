from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from duplicate_ticket_detection.dataset import TicketRecord
from duplicate_ticket_detection.rerank import EMPTY_FIELD_VALUES


STABLE_METADATA_FIELDS = ("product", "component", "severity", "priority")


@dataclass(frozen=True)
class StableRerankConfig:
    """Blend primary ranking, TF-IDF agreement, and metadata for review stability."""

    tfidf_weight: float = 0.30
    metadata_weight: float = 0.08
    agreement_weight: float = 0.08
    candidate_pool: int = 50

    @property
    def model_weight(self) -> float:
        return max(0.0, 1.0 - self.tfidf_weight - self.metadata_weight - self.agreement_weight)


def apply_stable_rerank_scores(
    queries: list[TicketRecord],
    candidates: list[TicketRecord],
    model_scores: np.ndarray,
    tfidf_scores: np.ndarray,
    config: StableRerankConfig,
) -> np.ndarray:
    """Return a score matrix that favors candidates supported by multiple signals."""

    if model_scores.shape != tfidf_scores.shape:
        raise ValueError("model_scores and tfidf_scores must have the same shape")
    if model_scores.shape != (len(queries), len(candidates)):
        raise ValueError("score matrix shape must match queries x candidates")

    output = np.zeros_like(model_scores, dtype=float)
    metadata_scores = metadata_match_matrix(queries, candidates)
    for query_index, query in enumerate(queries):
        model_row = np.array(model_scores[query_index], dtype=float, copy=True)
        tfidf_row = np.array(tfidf_scores[query_index], dtype=float, copy=True)
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                model_row[candidate_index] = -np.inf
                tfidf_row[candidate_index] = -np.inf

        model_norm = normalize_score_row(model_row)
        tfidf_norm = normalize_score_row(tfidf_row)
        agreement = rank_agreement_row(model_row, tfidf_row, candidate_pool=config.candidate_pool)
        stable_row = (
            config.model_weight * model_norm
            + config.tfidf_weight * tfidf_norm
            + config.metadata_weight * metadata_scores[query_index]
            + config.agreement_weight * agreement
        )
        for candidate_index, candidate in enumerate(candidates):
            if candidate.ticket_id == query.ticket_id:
                stable_row[candidate_index] = -np.inf
        output[query_index] = stable_row
    return output


def normalize_score_row(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    output = np.zeros_like(scores, dtype=float)
    if not finite.any():
        return output
    finite_scores = scores[finite]
    minimum = float(finite_scores.min())
    maximum = float(finite_scores.max())
    if maximum == minimum:
        output[finite] = 1.0
        return output
    output[finite] = (finite_scores - minimum) / (maximum - minimum)
    return output


def rank_agreement_row(model_scores: np.ndarray, tfidf_scores: np.ndarray, *, candidate_pool: int) -> np.ndarray:
    output = np.zeros_like(model_scores, dtype=float)
    finite = np.isfinite(model_scores) & np.isfinite(tfidf_scores)
    if not finite.any():
        return output

    model_order = [index for index in np.argsort(-model_scores) if finite[index]]
    tfidf_order = [index for index in np.argsort(-tfidf_scores) if finite[index]]
    pool_size = len(model_order) if candidate_pool <= 0 else min(candidate_pool, len(model_order), len(tfidf_order))
    model_rank = {index: rank for rank, index in enumerate(model_order[:pool_size], start=1)}
    tfidf_rank = {index: rank for rank, index in enumerate(tfidf_order[:pool_size], start=1)}
    for index in set(model_rank) & set(tfidf_rank):
        distance = abs(model_rank[index] - tfidf_rank[index])
        output[index] = max(0.0, 1.0 - distance / max(pool_size, 1))
    return output


def metadata_match_matrix(queries: list[TicketRecord], candidates: list[TicketRecord]) -> np.ndarray:
    matrix = np.zeros((len(queries), len(candidates)), dtype=float)
    for query_index, query in enumerate(queries):
        for candidate_index, candidate in enumerate(candidates):
            matrix[query_index, candidate_index] = metadata_match_fraction(query, candidate)
    return matrix


def metadata_match_fraction(query: TicketRecord, candidate: TicketRecord) -> float:
    matches = 0
    for field in STABLE_METADATA_FIELDS:
        query_value = normalized_field_value(query.fields, field)
        candidate_value = normalized_field_value(candidate.fields, field)
        if query_value and query_value == candidate_value:
            matches += 1
    return matches / len(STABLE_METADATA_FIELDS)


def normalized_field_value(fields: Mapping[str, str] | None, field: str) -> str:
    if not fields:
        return ""
    value = str(fields.get(field, "")).strip().lower()
    return "" if value in EMPTY_FIELD_VALUES else value
