from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import PipelineConfig


class AssigneeTriager:
    """Assign a likely owner using historical, metadata, and text signals."""

    def __init__(self, *, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._cached_history_path: Path | None = None
        self._cached_history_mtime: float | None = None
        self._cached_profile: dict[str, Any] | None = None

    def assign(self, ticket_json: dict[str, Any], priority_result: dict[str, Any]) -> dict[str, Any]:
        component = _normalize_component(ticket_json.get("component"))
        profile = self._history_profile()
        ranked = self._rank_candidates(ticket_json, profile)
        mapped_owner = self._mapped_owner(component)

        if ranked:
            assignee = ranked[0]["assignee"]
            confidence = self._confidence(ranked)
            reason = self._reason(component, priority_result, ranked[0])
        elif mapped_owner and mapped_owner != "manual_triage":
            assignee = mapped_owner
            confidence = 0.65
            priority = priority_result.get("predicted_priority", "P3")
            reason = f"Assigned from component-owner mapping for component '{component}' and priority {priority}."
        else:
            assignee = "manual_triage"
            confidence = 0.25
            reason = "No reliable owner signal was found; manual triage is required."

        ranked_candidates = [candidate["assignee"] for candidate in ranked[:5]]
        if assignee not in ranked_candidates and assignee != "manual_triage":
            ranked_candidates.insert(0, assignee)
        if not ranked_candidates:
            ranked_candidates = ["manual_triage"]

        return {
            "assignee": assignee,
            "confidence": confidence,
            "reason": reason,
            "ranked_candidates": ranked_candidates[:5],
            "candidate_scores": {candidate["assignee"]: round(candidate["score"], 6) for candidate in ranked[:5]},
        }

    def _rank_candidates(self, ticket_json: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        component = _normalize_component(ticket_json.get("component"))
        product = _normalize_component(ticket_json.get("product"))
        product_component = f"{product}::{component}"
        scores: defaultdict[str, float] = defaultdict(float)
        signals: dict[str, set[str]] = defaultdict(set)

        def add(assignee: Any, score: float, signal: str) -> None:
            owner = _normalize_assignee(assignee)
            if not owner or owner == "manual_triage" or score <= 0:
                return
            scores[owner] += score
            signals[owner].add(signal)

        if product != "unknown":
            for owner, count in profile["product_component_counts"].get(product_component, Counter()).items():
                add(owner, 4.0 * _count_ratio(count, profile["product_component_counts"][product_component]), "product_component_history")
        for owner, count in profile["component_counts"].get(component, Counter()).items():
            add(owner, 3.0 * _count_ratio(count, profile["component_counts"][component]), "component_history")
        if product != "unknown":
            for owner, count in profile["product_counts"].get(product, Counter()).items():
                add(owner, 0.75 * _count_ratio(count, profile["product_counts"][product]), "product_history")

        mapped_owner = self._mapped_owner(component)
        if mapped_owner and mapped_owner != "manual_triage":
            add(mapped_owner, 4.25, "component_owner_mapping")

        for owner, score in self._text_similarity_scores(ticket_json, profile).items():
            add(owner, score, "text_similarity")

        if not scores:
            return []

        ranked = [
            {"assignee": owner, "score": score, "signals": sorted(signals[owner])}
            for owner, score in scores.items()
            if signals[owner]
        ]
        ranked = [candidate for candidate in ranked if self._is_reliable_candidate(component, candidate)]
        ranked.sort(key=lambda item: (-item["score"], item["assignee"]))
        if not ranked:
            return []

        for owner, count in profile["global_counts"].most_common():
            if owner not in {candidate["assignee"] for candidate in ranked}:
                ranked.append(
                    {
                        "assignee": owner,
                        "score": 0.15 * _count_ratio(count, profile["global_counts"]),
                        "signals": ["global_prior"],
                    }
                )
            if len(ranked) >= 10:
                break
        return ranked

    def _is_reliable_candidate(self, component: str, candidate: dict[str, Any]) -> bool:
        signals = set(candidate.get("signals", []))
        strong_signals = {"component_history", "product_component_history", "component_owner_mapping"}
        if signals & strong_signals:
            return True
        if component in {"unknown", "documentation", "docs"}:
            return False
        return "text_similarity" in signals and candidate.get("score", 0.0) >= 1.25

    def _text_similarity_scores(self, ticket_json: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
        query_tokens = Counter(_tokens(_row_text(ticket_json)))
        if not query_tokens or not profile["documents"]:
            return {}

        doc_scores: defaultdict[int, float] = defaultdict(float)
        avgdl = profile["avgdl"] or 1.0
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            idf = profile["idf"].get(token)
            if idf is None:
                continue
            for doc_index, frequency in profile["postings"].get(token, []):
                doc_len = profile["doc_lengths"][doc_index] or 1
                denominator = frequency + k1 * (1 - b + b * doc_len / avgdl)
                doc_scores[doc_index] += idf * ((frequency * (k1 + 1)) / denominator)

        owner_scores: defaultdict[str, float] = defaultdict(float)
        for rank, (doc_index, score) in enumerate(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:10], start=1):
            owner = profile["documents"][doc_index]["assignee"]
            owner_scores[owner] += min(3.0, score) / rank
        return dict(owner_scores)

    def _confidence(self, ranked: list[dict[str, Any]]) -> float:
        if not ranked:
            return 0.25
        best = ranked[0]["score"]
        second = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = (best - second) / best if best else 0.0
        signal_bonus = min(0.20, 0.05 * len(ranked[0].get("signals", [])))
        return round(min(0.95, max(0.35, 0.55 + 0.30 * margin + signal_bonus)), 3)

    def _reason(self, component: str, priority_result: dict[str, Any], candidate: dict[str, Any]) -> str:
        priority = priority_result.get("predicted_priority", "P3")
        signals = ", ".join(candidate.get("signals", []))
        return (
            f"Assigned by hybrid ranking for component '{component}' and priority {priority}; "
            f"strongest signals: {signals}."
        )

    def _mapped_owner(self, component: str) -> str | None:
        mapping = {key.lower(): value for key, value in self.config.component_owner_mapping.items()}
        if component in mapping:
            return mapping[component]
        for key, value in mapping.items():
            if key != "unknown" and key in component:
                return value
        return mapping.get("unknown")

    def _historical_owner(self, component: str) -> str | None:
        counts = self._history_profile()["component_counts"].get(component, Counter())
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _history_profile(self) -> dict[str, Any]:
        path = self.config.assignee_dataset_path
        if path is None:
            return _empty_profile()
        path = Path(path)
        if not path.exists():
            return _empty_profile()

        mtime = path.stat().st_mtime
        if self._cached_profile is not None and self._cached_history_path == path and self._cached_history_mtime == mtime:
            return self._cached_profile

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                assignee = _normalize_assignee(row.get("assignee"))
                if not assignee or assignee == "manual_triage":
                    continue
                rows.append(
                    {
                        "assignee": assignee,
                        "component": _normalize_component(row.get("component")),
                        "product": _normalize_component(row.get("product")),
                        "title": str(row.get("title", "") or ""),
                        "description": str(row.get("description", "") or ""),
                        "text": _row_text(row),
                    }
                )

        profile = _build_profile(rows)
        self._cached_history_path = path
        self._cached_history_mtime = mtime
        self._cached_profile = profile
        return profile


TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "when",
    "with",
}


def _build_profile(rows: list[dict[str, str]]) -> dict[str, Any]:
    profile = _empty_profile()
    documents = profile["documents"]
    component_counts = profile["component_counts"]
    product_counts = profile["product_counts"]
    product_component_counts = profile["product_component_counts"]
    global_counts = profile["global_counts"]
    owner_component_counts = profile["owner_component_counts"]
    owner_examples = profile["owner_examples"]

    document_frequencies: Counter[str] = Counter()
    token_counts_by_doc: list[Counter[str]] = []
    for row in rows:
        owner = row["assignee"]
        component = row["component"]
        product = row["product"]
        product_component = f"{product}::{component}"
        documents.append(row)
        global_counts[owner] += 1
        component_counts[component][owner] += 1
        product_counts[product][owner] += 1
        product_component_counts[product_component][owner] += 1
        owner_component_counts[owner][component] += 1
        if len(owner_examples[owner]) < 6:
            owner_examples[owner].append(str(row.get("title", "") or row.get("text", ""))[:120])

        counts = Counter(_tokens(row["text"]))
        token_counts_by_doc.append(counts)
        profile["doc_lengths"].append(sum(counts.values()))
        document_frequencies.update(counts.keys())

    total_docs = len(documents)
    profile["avgdl"] = sum(profile["doc_lengths"]) / total_docs if total_docs else 0.0
    for token, frequency in document_frequencies.items():
        profile["idf"][token] = math.log(1 + (total_docs - frequency + 0.5) / (frequency + 0.5))
    for doc_index, counts in enumerate(token_counts_by_doc):
        for token, frequency in counts.items():
            profile["postings"][token].append((doc_index, frequency))
    return profile


def _empty_profile() -> dict[str, Any]:
    return {
        "documents": [],
        "global_counts": Counter(),
        "component_counts": defaultdict(Counter),
        "product_counts": defaultdict(Counter),
        "product_component_counts": defaultdict(Counter),
        "owner_component_counts": defaultdict(Counter),
        "owner_examples": defaultdict(list),
        "doc_lengths": [],
        "avgdl": 0.0,
        "idf": {},
        "postings": defaultdict(list),
    }


def _count_ratio(count: int, counter: Counter[str]) -> float:
    total = sum(counter.values())
    return count / total if total else 0.0


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, "") or "")
        for field in ("title", "description", "product", "component", "severity", "priority", "bug_type")
    )


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def _normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def _normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()
