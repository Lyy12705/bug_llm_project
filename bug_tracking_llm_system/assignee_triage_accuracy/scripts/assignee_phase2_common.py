from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_PHASE1_ROOT,
    DEFAULT_RAW_PATH,
    SYSTEM_ROOT,
    build_history_counts,
    build_metrics,
    current_triager_candidates,
    evaluate_current_triager,
    load_normalized_raw_records,
    normalize_assignee,
    normalize_component,
    parse_date,
    ratio,
    ticket_text,
    tokens,
    write_jsonl,
)


DEFAULT_PHASE2_ROOT = DEFAULT_PHASE1_ROOT.parent / "phase2_temporal_description"


@dataclass(frozen=True)
class TemporalWindow:
    name: str
    kind: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def default_windows() -> list[TemporalWindow]:
    return [
        TemporalWindow("expanding_w1", "expanding", "2024-01-01", "2024-02-08", "2024-02-08", "2024-02-15", "2024-02-15", "2024-02-22"),
        TemporalWindow("expanding_w2", "expanding", "2024-01-01", "2024-02-15", "2024-02-15", "2024-02-22", "2024-02-22", "2024-03-01"),
        TemporalWindow("expanding_w3", "expanding", "2024-01-01", "2024-02-22", "2024-02-22", "2024-03-01", "2024-03-01", "2024-03-08"),
        TemporalWindow("expanding_w4", "expanding", "2024-01-01", "2024-03-01", "2024-03-01", "2024-03-08", "2024-03-08", "2024-03-16"),
        TemporalWindow("sliding_w1", "sliding", "2024-01-11", "2024-02-08", "2024-02-08", "2024-02-15", "2024-02-15", "2024-02-22"),
        TemporalWindow("sliding_w2", "sliding", "2024-01-18", "2024-02-15", "2024-02-15", "2024-02-22", "2024-02-22", "2024-03-01"),
        TemporalWindow("sliding_w3", "sliding", "2024-01-25", "2024-02-22", "2024-02-22", "2024-03-01", "2024-03-01", "2024-03-08"),
        TemporalWindow("sliding_w4", "sliding", "2024-02-01", "2024-03-01", "2024-03-01", "2024-03-08", "2024-03-08", "2024-03-16"),
    ]


def rows_in_range(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    start_dt = as_utc(parse_date(start))
    end_dt = as_utc(parse_date(end))
    if start_dt is None or end_dt is None:
        raise SystemExit(f"Invalid date range: {start} - {end}")
    output = []
    for row in rows:
        created = as_utc(parse_date(row.get("created_at")))
        if created and start_dt <= created < end_dt:
            output.append(row)
    return output


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def window_rows(records: list[dict[str, Any]], window: TemporalWindow) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        rows_in_range(records, window.train_start, window.train_end),
        rows_in_range(records, window.validation_start, window.validation_end),
        rows_in_range(records, window.test_start, window.test_end),
    )


def build_window_view(
    records: list[dict[str, Any]], window: TemporalWindow, *, min_train_assignee_count: int
) -> dict[str, Any]:
    train_raw, validation_raw, test_raw = window_rows(records, window)
    train_counts_all = Counter(normalize_assignee(row.get("assignee")) for row in train_raw)
    roster = {assignee for assignee, count in train_counts_all.items() if assignee and count >= min_train_assignee_count}
    train = [row for row in train_raw if normalize_assignee(row.get("assignee")) in roster]
    validation = [row for row in validation_raw if normalize_assignee(row.get("assignee")) in roster]
    test = [row for row in test_raw if normalize_assignee(row.get("assignee")) in roster]
    low_frequency_test = [
        row
        for row in test_raw
        if 0 < train_counts_all.get(normalize_assignee(row.get("assignee")), 0) < min_train_assignee_count
    ]
    unseen_test = [row for row in test_raw if train_counts_all.get(normalize_assignee(row.get("assignee")), 0) == 0]
    return {
        "train_raw": train_raw,
        "validation_raw": validation_raw,
        "test_raw": test_raw,
        "train": train,
        "validation": validation,
        "test": test,
        "roster": roster,
        "train_counts_all": train_counts_all,
        "low_frequency_test": low_frequency_test,
        "unseen_test": unseen_test,
    }


def evaluate_baselines_for_window(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: set[str],
    train_path: Path,
    baselines: list[str],
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = build_history_counts(train_rows)
    bm25 = BM25Index(train_rows)
    all_predictions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    current_predictions: list[dict[str, Any]] | None = None
    for baseline in baselines:
        if baseline == "current_assignee_triager":
            current_predictions, baseline_metrics = evaluate_current_triager(
                train_path=train_path,
                train_rows=train_rows,
                test_rows=test_rows,
                roster=roster,
                top_k=top_k,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            predictions = [
                {**row, "baseline": baseline}
                for row in current_predictions
            ]
        else:
            predictions = []
            for row in test_rows:
                candidates = baseline_candidates(baseline, row, counts, bm25, roster, top_k)
                expected = normalize_assignee(row.get("assignee"))
                rank = rank_of(expected, candidates)
                predictions.append(
                    {
                        "baseline": baseline,
                        "ticket_id": row.get("ticket_id"),
                        "title": row.get("title"),
                        "product": normalize_component(row.get("product")),
                        "component": normalize_component(row.get("component")),
                        "priority": row.get("priority", "P3"),
                        "expected_assignee": expected,
                        "predicted_assignee": candidates[0] if candidates else "",
                        "ranked_candidates": candidates,
                        "rank": rank,
                        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                        "is_top1_correct": bool(candidates and candidates[0] == expected),
                        "is_hit_at_3": expected in candidates[:3],
                        "is_hit_at_5": expected in candidates[:5],
                        "is_hit_at_10": expected in candidates[:10],
                    }
                )
            baseline_metrics = metric_with_macro(predictions, bootstrap_samples=bootstrap_samples, seed=seed)
        if baseline == "current_assignee_triager" and current_predictions is not None:
            baseline_metrics = metric_with_macro(predictions, bootstrap_samples=bootstrap_samples, seed=seed)
        metrics[baseline] = baseline_metrics
        all_predictions.extend(predictions)
    return all_predictions, metrics


def baseline_candidates(
    baseline: str,
    row: dict[str, Any],
    counts: dict[str, Any],
    bm25: "BM25Index",
    roster: set[str],
    top_k: int,
) -> list[str]:
    component = normalize_component(row.get("component"))
    product = normalize_component(row.get("product"))
    if baseline == "global_majority":
        return from_counters([counts["global_counts"]], roster, top_k)
    if baseline == "product_majority":
        return from_counters([counts["product_counts"].get(product, Counter()), counts["global_counts"]], roster, top_k)
    if baseline == "component_majority":
        return from_counters([counts["component_counts"].get(component, Counter()), counts["global_counts"]], roster, top_k)
    if baseline == "product_component_majority":
        return from_counters(
            [
                counts["product_component_counts"].get(f"{product}::{component}", Counter()),
                counts["component_counts"].get(component, Counter()),
                counts["product_counts"].get(product, Counter()),
                counts["global_counts"],
            ],
            roster,
            top_k,
        )
    if baseline == "bm25_text_knn":
        return bm25.rank_assignees(row, roster, counts["global_counts"], top_k)
    raise SystemExit(f"Unknown baseline: {baseline}")


def from_counters(counters: list[Counter[str]], roster: set[str], top_k: int) -> list[str]:
    candidates: list[str] = []
    for counter in counters:
        for assignee, _ in counter.most_common():
            if assignee in roster and assignee not in candidates:
                candidates.append(assignee)
            if len(candidates) >= top_k:
                return candidates
    return candidates[:top_k]


def metric_with_macro(predictions: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    metrics = build_metrics(predictions, bootstrap_samples=bootstrap_samples, seed=seed)
    by_assignee: dict[str, list[int]] = defaultdict(list)
    by_component: dict[str, list[int]] = defaultdict(list)
    for row in predictions:
        indicator = 1 if row.get("is_top1_correct") else 0
        by_assignee[row.get("expected_assignee", "")].append(indicator)
        by_component[row.get("component", "")].append(indicator)
    metrics["macro_assignee_top1"] = macro_average(by_assignee)
    metrics["macro_component_top1"] = macro_average(by_component)
    return metrics


def macro_average(groups: dict[str, list[int]]) -> float:
    if not groups:
        return 0.0
    return round(sum(sum(values) / len(values) for values in groups.values()) / len(groups), 6)


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def summarize_top1(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 6),
        "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]], *, neighbors: int = 25) -> None:
        self.rows = rows
        self.neighbors = neighbors
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        token_counts_by_doc = []
        document_frequencies: Counter[str] = Counter()
        for row in self.rows:
            counts = Counter(tokens(ticket_text(row)))
            token_counts_by_doc.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            document_frequencies.update(counts.keys())
        total_docs = len(token_counts_by_doc)
        self.avgdl = sum(self.doc_lengths) / total_docs if total_docs else 0.0
        for token, frequency in document_frequencies.items():
            self.idf[token] = math.log(1 + (total_docs - frequency + 0.5) / (frequency + 0.5))
        for doc_index, counts in enumerate(token_counts_by_doc):
            for token, frequency in counts.items():
                self.postings[token].append((doc_index, frequency))

    def rank_assignees(self, row: dict[str, Any], roster: set[str], global_counts: Counter[str], top_k: int) -> list[str]:
        query_tokens = Counter(tokens(ticket_text(row)))
        if not query_tokens:
            return from_counters([global_counts], roster, top_k)
        doc_scores: defaultdict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            if token not in self.postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_index, frequency in self.postings[token]:
                doc_len = self.doc_lengths[doc_index] or 1
                denominator = frequency + k1 * (1 - b + b * doc_len / (self.avgdl or 1.0))
                doc_scores[doc_index] += idf * ((frequency * (k1 + 1)) / denominator)
        if not doc_scores:
            return from_counters([global_counts], roster, top_k)
        assignee_scores: defaultdict[str, float] = defaultdict(float)
        for rank, (doc_index, score) in enumerate(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[: self.neighbors], start=1):
            assignee = normalize_assignee(self.rows[doc_index].get("assignee"))
            if assignee in roster:
                assignee_scores[assignee] += score / rank
        candidates = [assignee for assignee, _ in sorted(assignee_scores.items(), key=lambda item: item[1], reverse=True)]
        for assignee, _ in global_counts.most_common():
            if assignee in roster and assignee not in candidates:
                candidates.append(assignee)
            if len(candidates) >= top_k:
                break
        return candidates[:top_k]
