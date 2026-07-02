from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from duplicate_ticket_detection.dataset import TicketRecord, relevant_duplicate_ids
from duplicate_ticket_detection.metrics import average_precision


@dataclass(frozen=True)
class RankedTicket:
    ticket_id: str
    score: float
    title_score: float
    content_score: float
    title: str


@dataclass(frozen=True)
class EvaluationRow:
    query_id: str
    average_precision: float
    relevant_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]


class TfidfDuplicateDetector:
    """Title/content TF-IDF duplicate detector used as a runnable baseline."""

    def __init__(
        self,
        *,
        combine: str = "max",
        analyzer: str = "char_wb",
        ngram_min: int = 3,
        ngram_max: int = 5,
        min_df: int | float = 1,
    ) -> None:
        self.combine = combine
        self.title_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            sublinear_tf=True,
            lowercase=True,
        )
        self.content_vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            sublinear_tf=True,
            lowercase=True,
        )
        self._is_fitted = False

    def fit(self, tickets: Iterable[TicketRecord]) -> "TfidfDuplicateDetector":
        records = list(tickets)
        self.title_vectorizer.fit(_safe_texts(ticket.title for ticket in records))
        self.content_vectorizer.fit(_safe_texts(ticket.content for ticket in records))
        self._is_fitted = True
        return self

    def rank(
        self,
        query: TicketRecord,
        candidates: Iterable[TicketRecord],
        *,
        top_k: int | None = None,
    ) -> list[RankedTicket]:
        self._check_fitted()
        candidate_records = [ticket for ticket in candidates if ticket.ticket_id != query.ticket_id]
        if not candidate_records:
            return []

        query_title = self.title_vectorizer.transform(_safe_texts([query.title]))
        query_content = self.content_vectorizer.transform(_safe_texts([query.content]))
        candidate_titles = self.title_vectorizer.transform(_safe_texts(ticket.title for ticket in candidate_records))
        candidate_contents = self.content_vectorizer.transform(_safe_texts(ticket.content for ticket in candidate_records))

        title_scores = (query_title @ candidate_titles.T).toarray().ravel()
        content_scores = (query_content @ candidate_contents.T).toarray().ravel()
        scores = combine_similarity_scores(title_scores, content_scores, self.combine)

        ranked_indices = np.argsort(-scores)
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [
            RankedTicket(
                ticket_id=candidate_records[index].ticket_id,
                score=float(scores[index]),
                title_score=float(title_scores[index]),
                content_score=float(content_scores[index]),
                title=candidate_records[index].title,
            )
            for index in ranked_indices
        ]

    def evaluate(
        self,
        tickets: Iterable[TicketRecord],
        *,
        candidates: Iterable[TicketRecord] | None = None,
        query_ids: Iterable[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[float, list[EvaluationRow]]:
        query_records = list(tickets)
        candidate_records = list(candidates) if candidates is not None else query_records
        if not self._is_fitted:
            self.fit(candidate_records)

        query_filter = set(query_ids) if query_ids else None
        rows: list[EvaluationRow] = []
        for query in query_records:
            if query_filter is not None and query.ticket_id not in query_filter:
                continue
            relevant = relevant_duplicate_ids(query, candidate_records)
            if not relevant:
                continue
            ranking = self.rank(query, candidate_records, top_k=top_k)
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

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfDuplicateDetector":
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Detector is not fitted yet. Call fit(tickets) first.")


def combine_similarity_scores(title_scores: np.ndarray, content_scores: np.ndarray, method: str) -> np.ndarray:
    method = method.strip().lower()
    if method in {"title", "title_only", "1.0_title"}:
        return title_scores
    if method in {"content", "content_only", "1.0_content"}:
        return content_scores
    if method in {"mean", "avg", "0.5_title_0.5_content"}:
        return 0.5 * title_scores + 0.5 * content_scores
    if method in {"title75", "title_0.75", "0.75_title_0.25_content"}:
        return 0.75 * title_scores + 0.25 * content_scores
    if method in {"content75", "content_0.75", "0.25_title_0.75_content"}:
        return 0.25 * title_scores + 0.75 * content_scores
    if method == "max":
        return np.maximum(title_scores, content_scores)
    if method == "min":
        return np.minimum(title_scores, content_scores)
    if method.startswith("weighted:"):
        title_weight, content_weight = _parse_weighted_method(method)
        return title_weight * title_scores + content_weight * content_scores
    raise ValueError(f"Unknown combine method: {method}")


def _parse_weighted_method(method: str) -> tuple[float, float]:
    _, weights = method.split(":", maxsplit=1)
    left, right = weights.split(",", maxsplit=1)
    title_weight = float(left)
    content_weight = float(right)
    if title_weight < 0 or content_weight < 0:
        raise ValueError("Similarity weights must be non-negative")
    total = title_weight + content_weight
    if total <= 0:
        raise ValueError("At least one similarity weight must be positive")
    return title_weight / total, content_weight / total


def _safe_texts(texts: Iterable[str]) -> list[str]:
    values = [(text or "").strip() for text in texts]
    return [value if value else "__empty__" for value in values]
