from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from duplicate_ticket_detection.dataset import TicketRecord


@dataclass(frozen=True)
class TripletRecord:
    anchor_id: str
    positive_id: str
    negative_id: str
    anchor_text: str
    positive_text: str
    negative_text: str
    field: str


def build_triplets(
    tickets: Iterable[TicketRecord],
    *,
    field: str = "content",
    max_triplets: int | None = None,
    seed: int = 13,
    bidirectional: bool = False,
    negative_strategy: str = "all",
    negatives_per_anchor: int | None = None,
) -> list[TripletRecord]:
    """Build anchor/positive/negative examples from duplicate groups."""

    records = list(tickets)
    negative_strategy = negative_strategy.strip().lower()
    if negative_strategy not in {"all", "random", "hard"}:
        raise ValueError("negative_strategy must be one of: all, random, hard")

    rng = random.Random(seed)
    hard_negatives = _hard_negative_candidates(records, field) if negative_strategy == "hard" else {}
    grouped: dict[str, list[TicketRecord]] = {}
    for ticket in records:
        if ticket.duplicate_group:
            grouped.setdefault(ticket.duplicate_group, []).append(ticket)

    grouped = {group: members for group, members in grouped.items() if len(members) >= 2}
    triplets: list[TripletRecord] = []

    for group, positives in grouped.items():
        negatives = [ticket for ticket in records if ticket.duplicate_group != group]
        if not negatives:
            continue

        for i, anchor in enumerate(positives):
            positive_range = range(len(positives)) if bidirectional else range(i + 1, len(positives))
            for j in positive_range:
                if i == j:
                    continue
                positive = positives[j]
                selected_negatives = _select_negatives(
                    anchor=anchor,
                    negatives=negatives,
                    hard_negatives=hard_negatives,
                    strategy=negative_strategy,
                    negatives_per_anchor=negatives_per_anchor,
                    rng=rng,
                )
                for negative in selected_negatives:
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

    if max_triplets is not None and len(triplets) > max_triplets:
        triplets = rng.sample(triplets, max_triplets)

    return triplets


def _select_negatives(
    *,
    anchor: TicketRecord,
    negatives: list[TicketRecord],
    hard_negatives: dict[str, list[TicketRecord]],
    strategy: str,
    negatives_per_anchor: int | None,
    rng: random.Random,
) -> list[TicketRecord]:
    if not negatives:
        return []
    limit = negatives_per_anchor if negatives_per_anchor is not None else len(negatives)
    limit = max(1, min(limit, len(negatives)))

    if strategy == "all":
        return negatives
    if strategy == "random":
        return rng.sample(negatives, limit)

    hard_pool = [
        candidate
        for candidate in hard_negatives.get(anchor.ticket_id, [])
        if candidate.duplicate_group != anchor.duplicate_group
    ]
    if hard_pool:
        return hard_pool[:limit]
    return rng.sample(negatives, limit)


def _hard_negative_candidates(records: list[TicketRecord], field: str) -> dict[str, list[TicketRecord]]:
    if len(records) < 2:
        return {}

    texts = [_safe_text(ticket.text_for(field)) for ticket in records]
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True).fit_transform(texts)
    similarities = vectors @ vectors.T
    candidates: dict[str, list[TicketRecord]] = {}

    for row_index, anchor in enumerate(records):
        scores = np.asarray(similarities[row_index].toarray()).ravel()
        ranked_indices = np.argsort(-scores)
        ranked_records = [
            records[index]
            for index in ranked_indices
            if records[index].ticket_id != anchor.ticket_id
            and records[index].duplicate_group != anchor.duplicate_group
        ]
        candidates[anchor.ticket_id] = ranked_records
    return candidates


def _safe_text(value: str) -> str:
    value = (value or "").strip()
    return value if value else "__empty__"


def write_triplets_csv(triplets: Iterable[TripletRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "field",
        "anchor_id",
        "positive_id",
        "negative_id",
        "anchor_text",
        "positive_text",
        "negative_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for triplet in triplets:
            writer.writerow({field: getattr(triplet, field) for field in fields})
