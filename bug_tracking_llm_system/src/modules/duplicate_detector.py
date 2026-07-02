from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PipelineConfig
from utils.embedding_utils import cosine_text_similarity, weighted_similarity


class DuplicateDetector:
    """Rank historical tickets and decide whether the query is a duplicate."""

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.threshold = threshold if threshold is not None else self.config.duplicate_threshold
        self.top_k = top_k if top_k is not None else self.config.duplicate_top_k

    def detect(self, ticket_json: dict[str, Any], historical_db: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        historical = historical_db if historical_db is not None else self._load_historical_tickets()
        ranked = sorted(
            (self._candidate_row(ticket_json, candidate) for candidate in historical),
            key=lambda row: row["similarity"],
            reverse=True,
        )
        top_candidates = ranked[: self.top_k]
        best = top_candidates[0] if top_candidates else None
        best_score = float(best["similarity"]) if best else 0.0
        is_duplicate = bool(best and best_score >= self.threshold)

        return {
            "is_duplicate": is_duplicate,
            "duplicate_of": best["ticket_id"] if is_duplicate and best else None,
            "similarity_score": best_score,
            "threshold": self.threshold,
            "needs_review": bool(best and not is_duplicate and best_score >= self.threshold - self.config.duplicate_review_margin),
            "top_k_candidates": top_candidates,
        }

    def _load_historical_tickets(self) -> list[dict[str, Any]]:
        path = self.config.historical_tickets_path
        if path is None or not Path(path).exists():
            return []
        tickets: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid historical ticket JSONL at line {line_number}: {exc}") from exc
                if isinstance(value, dict):
                    tickets.append(value)
        return tickets

    def _candidate_row(self, ticket: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        title_score = cosine_text_similarity(str(ticket.get("title", "")), str(candidate.get("title") or candidate.get("summary") or ""))
        body_score = weighted_similarity(
            [
                (str(ticket.get("description", "")), str(candidate.get("description") or candidate.get("content") or ""), 0.65),
                (str(ticket.get("error_message", "")), str(candidate.get("error_message") or candidate.get("logs") or ""), 0.35),
            ]
        )
        score = 0.40 * title_score + 0.55 * body_score
        if _same_nonempty(ticket.get("component"), candidate.get("component")):
            score += 0.05
        score = min(1.0, score)
        return {
            "ticket_id": str(candidate.get("ticket_id") or candidate.get("bug_id") or candidate.get("id") or ""),
            "title": str(candidate.get("title") or candidate.get("summary") or ""),
            "similarity": float(score),
            "title_similarity": float(title_score),
            "content_similarity": float(body_score),
            "component": candidate.get("component"),
            "priority": candidate.get("priority"),
            "assignee": candidate.get("assignee"),
        }


def _same_nonempty(left: Any, right: Any) -> bool:
    return bool(left and right and str(left).strip().lower() == str(right).strip().lower())
