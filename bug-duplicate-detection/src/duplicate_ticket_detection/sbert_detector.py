from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from duplicate_ticket_detection.dataset import TicketRecord, relevant_duplicate_ids
from duplicate_ticket_detection.metrics import average_precision
from duplicate_ticket_detection.tfidf_detector import EvaluationRow, RankedTicket, combine_similarity_scores
from duplicate_ticket_detection.triplets import TripletRecord


@dataclass(frozen=True)
class SbertTrainingConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    epochs: int = 1
    batch_size: int = 16
    margin: float = 1.0
    warmup_steps: int = 100
    local_files_only: bool = False
    device: str | None = None


class SbertDuplicateDetector:
    """Optional SBERT implementation of the paper's title/content triplet setup."""

    def __init__(self, *, combine: str = "max", config: SbertTrainingConfig | None = None) -> None:
        self.combine = combine
        self.config = config or SbertTrainingConfig()
        self.title_model = None
        self.content_model = None

    def fit(
        self,
        *,
        title_triplets: Iterable[TripletRecord],
        content_triplets: Iterable[TripletRecord],
    ) -> "SbertDuplicateDetector":
        SentenceTransformer, InputExample, losses, torch = _load_sentence_transformers()
        model_kwargs = {"local_files_only": self.config.local_files_only}
        if self.config.device:
            model_kwargs["device"] = self.config.device

        self.title_model = SentenceTransformer(
            self.config.base_model,
            **model_kwargs,
        )
        self.content_model = SentenceTransformer(
            self.config.base_model,
            **model_kwargs,
        )

        title_examples = [_to_input_example(triplet, InputExample) for triplet in title_triplets]
        content_examples = [_to_input_example(triplet, InputExample) for triplet in content_triplets]

        if title_examples:
            title_loader = torch.utils.data.DataLoader(
                title_examples,
                shuffle=True,
                batch_size=self.config.batch_size,
            )
            title_loss = losses.TripletLoss(
                self.title_model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=self.config.margin,
            )
            self.title_model.fit(
                train_objectives=[(title_loader, title_loss)],
                epochs=self.config.epochs,
                warmup_steps=self.config.warmup_steps,
            )

        if content_examples:
            content_loader = torch.utils.data.DataLoader(
                content_examples,
                shuffle=True,
                batch_size=self.config.batch_size,
            )
            content_loss = losses.TripletLoss(
                self.content_model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=self.config.margin,
            )
            self.content_model.fit(
                train_objectives=[(content_loader, content_loss)],
                epochs=self.config.epochs,
                warmup_steps=self.config.warmup_steps,
            )

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

        title_query = self.title_model.encode([query.title], normalize_embeddings=True)
        title_candidates = self.title_model.encode(
            [ticket.title for ticket in candidate_records],
            normalize_embeddings=True,
        )
        content_query = self.content_model.encode([query.content], normalize_embeddings=True)
        content_candidates = self.content_model.encode(
            [ticket.content for ticket in candidate_records],
            normalize_embeddings=True,
        )

        title_scores = np.asarray(title_query @ title_candidates.T).ravel()
        content_scores = np.asarray(content_query @ content_candidates.T).ravel()
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
    ) -> tuple[float, list[EvaluationRow]]:
        """Evaluate duplicate ranking by encoding queries/candidates once.

        The single-query ``rank`` method is convenient for interactive use, but
        cross-validation would be painfully slow if every query re-encoded all
        candidates. This method batches embeddings for the whole fold.
        """

        self._check_fitted()
        query_records = list(tickets)
        candidate_records = list(candidates) if candidates is not None else query_records
        if not query_records or not candidate_records:
            return 0.0, []

        query_title_embeddings = self.title_model.encode(
            [ticket.title for ticket in query_records],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(query_records) > 64,
        )
        candidate_title_embeddings = self.title_model.encode(
            [ticket.title for ticket in candidate_records],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(candidate_records) > 64,
        )
        query_content_embeddings = self.content_model.encode(
            [ticket.content for ticket in query_records],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(query_records) > 64,
        )
        candidate_content_embeddings = self.content_model.encode(
            [ticket.content for ticket in candidate_records],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(candidate_records) > 64,
        )

        title_scores = np.asarray(query_title_embeddings @ candidate_title_embeddings.T)
        content_scores = np.asarray(query_content_embeddings @ candidate_content_embeddings.T)
        scores = combine_similarity_scores(title_scores, content_scores, self.combine)
        candidate_ids = [ticket.ticket_id for ticket in candidate_records]

        rows: list[EvaluationRow] = []
        for query_index, query in enumerate(query_records):
            relevant = relevant_duplicate_ids(query, candidate_records)
            if not relevant:
                continue
            query_scores = scores[query_index].copy()
            for candidate_index, candidate in enumerate(candidate_records):
                if candidate.ticket_id == query.ticket_id:
                    query_scores[candidate_index] = -np.inf
            ranked_ids = tuple(candidate_ids[index] for index in np.argsort(-query_scores))
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

    def save(self, output_dir: str | Path) -> None:
        self._check_fitted()
        output = Path(output_dir)
        self.title_model.save(str(output / "title_model"))
        self.content_model.save(str(output / "content_model"))

    @classmethod
    def load(cls, model_dir: str | Path, *, combine: str = "max") -> "SbertDuplicateDetector":
        SentenceTransformer, _, _, _ = _load_sentence_transformers()
        detector = cls(combine=combine)
        model_dir = Path(model_dir)
        detector.title_model = SentenceTransformer(str(model_dir / "title_model"))
        detector.content_model = SentenceTransformer(str(model_dir / "content_model"))
        return detector

    def _check_fitted(self) -> None:
        if self.title_model is None or self.content_model is None:
            raise RuntimeError("SBERT detector is not fitted or loaded yet.")


def _to_input_example(triplet: TripletRecord, input_example_type):
    return input_example_type(texts=[triplet.anchor_text, triplet.positive_text, triplet.negative_text])


def _load_sentence_transformers():
    try:
        from sentence_transformers import InputExample, SentenceTransformer
        from sentence_transformers.sentence_transformer import losses
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SBERT support requires optional dependencies. Install with: "
            "python3 -m pip install -e '.[sbert]'"
        ) from exc
    return SentenceTransformer, InputExample, losses, torch
