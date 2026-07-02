from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAPER_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


DEFAULT_BASELINES = [
    "global_majority",
    "component_majority",
    "product_component_majority",
    "bm25_text_knn",
    "current_assignee_triager",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-grade assignee-triage baselines.")
    parser.add_argument("--dataset", default="bmo_paper", help="Processed dataset prefix.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PAPER_ROOT / "data" / "processed",
        help="Processed dataset directory.",
    )
    parser.add_argument("--train", type=Path, default=None, help="Train/history JSONL.")
    parser.add_argument("--test", type=Path, default=None, help="Test JSONL.")
    parser.add_argument("--roster", type=Path, default=None, help="Candidate roster JSON.")
    parser.add_argument("--output-dir", type=Path, default=PAPER_ROOT / "reports", help="Report directory.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-neighbors", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    args = parser.parse_args()

    train_path = args.train or args.data_dir / f"{args.dataset}_history_train.jsonl"
    test_path = args.test or args.data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.roster or args.data_dir / f"{args.dataset}_candidate_roster.json"

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)
    if not train_rows or not test_rows:
        raise SystemExit("Train and test files must both contain rows.")

    context = build_context(train_rows, roster, args.bm25_neighbors)
    all_predictions: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "dataset": args.dataset,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "top_k": args.top_k,
        "bm25_neighbors": args.bm25_neighbors,
        "bootstrap_samples": args.bootstrap_samples,
        "baselines": {},
    }

    baseline_indicators: dict[str, list[int]] = {}
    for baseline in args.baselines:
        predictions = predict_baseline(
            baseline=baseline,
            train_path=train_path,
            train_rows=train_rows,
            test_rows=test_rows,
            context=context,
            top_k=args.top_k,
        )
        all_predictions.extend(predictions)
        baseline_metrics = build_metrics(predictions, args.bootstrap_samples, args.seed)
        metrics["baselines"][baseline] = baseline_metrics
        baseline_indicators[baseline] = [1 if row["is_top1_correct"] else 0 for row in predictions]

    reference = "current_assignee_triager"
    if reference in baseline_indicators:
        for baseline, indicators in baseline_indicators.items():
            if baseline == reference:
                continue
            metrics["baselines"][baseline]["top1_delta_vs_current_triager"] = paired_delta_ci(
                baseline_indicators[reference], indicators, args.bootstrap_samples, args.seed
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.dataset}_metrics.json", metrics)
    write_jsonl(args.output_dir / f"{args.dataset}_predictions.jsonl", all_predictions)
    write_error_analysis(args.output_dir / f"{args.dataset}_error_analysis.csv", all_predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_roster(path: Path) -> list[str]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def build_context(train_rows: list[dict[str, Any]], roster: list[str], bm25_neighbors: int) -> dict[str, Any]:
    global_counts: Counter[str] = Counter()
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_component_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in train_rows:
        assignee = normalize_assignee(row.get("assignee"))
        component = normalize_component(row.get("component"))
        product = normalize_component(row.get("product"))
        key = product_component_key(product, component)
        global_counts[assignee] += 1
        component_counts[component][assignee] += 1
        product_counts[product][assignee] += 1
        product_component_counts[key][assignee] += 1

    return {
        "roster": set(roster),
        "global_counts": global_counts,
        "component_counts": dict(component_counts),
        "product_counts": dict(product_counts),
        "product_component_counts": dict(product_component_counts),
        "bm25": BM25Index(train_rows, neighbors=bm25_neighbors),
    }


def predict_baseline(
    *,
    baseline: str,
    train_path: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    context: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    predictions = []
    triager: AssigneeTriager | None = None
    if baseline == "current_assignee_triager":
        config = PipelineConfig(
            project_root=SYSTEM_ROOT,
            assignee_dataset_path=train_path,
            save_checkpoints=False,
        )
        triager = AssigneeTriager(config=config)

    for row in test_rows:
        expected = normalize_assignee(row.get("assignee"))
        if baseline == "global_majority":
            candidates = from_counters([context["global_counts"]], context["roster"], top_k)
        elif baseline == "component_majority":
            component = normalize_component(row.get("component"))
            candidates = from_counters(
                [context["component_counts"].get(component, Counter()), context["global_counts"]],
                context["roster"],
                top_k,
            )
        elif baseline == "product_component_majority":
            component = normalize_component(row.get("component"))
            product = normalize_component(row.get("product"))
            candidates = from_counters(
                [
                    context["product_component_counts"].get(product_component_key(product, component), Counter()),
                    context["component_counts"].get(component, Counter()),
                    context["product_counts"].get(product, Counter()),
                    context["global_counts"],
                ],
                context["roster"],
                top_k,
            )
        elif baseline == "bm25_text_knn":
            candidates = context["bm25"].rank_assignees(row, context["roster"], context["global_counts"], top_k)
        elif baseline == "current_assignee_triager":
            assert triager is not None
            result = triager.assign(row, {"predicted_priority": row.get("priority", "P3")})
            candidates = current_triager_candidates(result, row, context, top_k)
        else:
            raise SystemExit(f"Unknown baseline: {baseline}")

        predicted = candidates[0] if candidates else ""
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
                "predicted_assignee": predicted,
                "ranked_candidates": candidates,
                "rank": rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "is_top1_correct": expected == predicted,
                "is_hit_at_3": expected in candidates[:3],
                "is_hit_at_5": expected in candidates[:5],
                "is_hit_at_10": expected in candidates[:10],
                "error_type": error_type(expected, predicted, row, context),
            }
        )
    return predictions


def from_counters(counters: list[Counter[str]], roster: set[str], top_k: int) -> list[str]:
    candidates: list[str] = []
    for counter in counters:
        for assignee, _ in counter.most_common():
            if assignee in roster and assignee not in candidates:
                candidates.append(assignee)
            if len(candidates) >= top_k:
                return candidates
    return candidates[:top_k]


def current_triager_candidates(
    result: dict[str, Any], row: dict[str, Any], context: dict[str, Any], top_k: int
) -> list[str]:
    component = normalize_component(row.get("component"))
    candidates: list[str] = []

    def add(value: Any) -> None:
        assignee = normalize_assignee(value)
        if not assignee or assignee in candidates:
            return
        if assignee not in context["roster"] and assignee != "manual_triage":
            return
        candidates.append(assignee)

    for assignee in result.get("ranked_candidates", []):
        add(assignee)
    add(result.get("assignee"))
    for assignee, _ in context["component_counts"].get(component, Counter()).most_common():
        add(assignee)
    for assignee, _ in context["global_counts"].most_common():
        add(assignee)
    add("manual_triage")
    return candidates[:top_k]


def build_metrics(predictions: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    total = len(predictions)
    indicators = [1 if row["is_top1_correct"] else 0 for row in predictions]
    by_assignee: dict[str, list[int]] = defaultdict(list)
    by_component: dict[str, list[int]] = defaultdict(list)
    for row, indicator in zip(predictions, indicators, strict=True):
        by_assignee[row["expected_assignee"]].append(indicator)
        by_component[row["component"]].append(indicator)

    return {
        "rows": total,
        "top1_accuracy": ratio(sum(indicators), total),
        "hit_at_3": ratio(sum(1 for row in predictions if row["is_hit_at_3"]), total),
        "hit_at_5": ratio(sum(1 for row in predictions if row["is_hit_at_5"]), total),
        "hit_at_10": ratio(sum(1 for row in predictions if row["is_hit_at_10"]), total),
        "mrr": ratio(sum(float(row["reciprocal_rank"]) for row in predictions), total),
        "macro_top1_by_assignee": macro_average(by_assignee),
        "macro_top1_by_component": macro_average(by_component),
        "top1_bootstrap_95ci": bootstrap_ci(indicators, bootstrap_samples, seed),
        "error_breakdown": dict(sorted(Counter(row["error_type"] for row in predictions).items())),
    }


def macro_average(groups: dict[str, list[int]]) -> float:
    if not groups:
        return 0.0
    return round(sum(sum(values) / len(values) for values in groups.values()) / len(groups), 6)


def bootstrap_ci(indicators: list[int], samples: int, seed: int) -> dict[str, float]:
    if not indicators:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    estimates = []
    n = len(indicators)
    for _ in range(samples):
        draw = [indicators[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(draw) / n)
    estimates.sort()
    return {
        "low": round(percentile(estimates, 0.025), 6),
        "high": round(percentile(estimates, 0.975), 6),
    }


def paired_delta_ci(reference: list[int], candidate: list[int], samples: int, seed: int) -> dict[str, float]:
    if not reference or len(reference) != len(candidate):
        return {"delta": 0.0, "low": 0.0, "high": 0.0}
    deltas = [candidate_value - reference_value for reference_value, candidate_value in zip(reference, candidate)]
    rng = random.Random(seed)
    n = len(deltas)
    estimates = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(draw) / n)
    estimates.sort()
    return {
        "delta": round(sum(deltas) / n, 6),
        "low": round(percentile(estimates, 0.025), 6),
        "high": round(percentile(estimates, 0.975), 6),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def error_type(expected: str, predicted: str, row: dict[str, Any], context: dict[str, Any]) -> str:
    if expected == predicted:
        return "correct"
    component = normalize_component(row.get("component"))
    if predicted == "manual_triage":
        return "manual_triage_predicted"
    if component not in context["component_counts"]:
        return "cold_start_component"
    if context["component_counts"].get(component, Counter()).get(expected, 0) <= 1:
        return "long_tail_assignee"
    return "wrong_assignee"


def ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def product_component_key(product: str, component: str) -> str:
    return f"{product}::{component}"


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_error_analysis(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "baseline",
        "ticket_id",
        "product",
        "component",
        "priority",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "error_type",
        "ranked_candidates",
        "title",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            output = dict(row)
            output["ranked_candidates"] = "|".join(row.get("ranked_candidates", []))
            writer.writerow({field: output.get(field, "") for field in fields})


TOKEN_RE = re.compile(r"[a-z0-9_]+")


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]], *, neighbors: int) -> None:
        self.rows = rows
        self.neighbors = neighbors
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        document_frequencies: Counter[str] = Counter()
        documents = []
        for row in self.rows:
            tokens = tokenize(row_text(row))
            counts = Counter(tokens)
            documents.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            document_frequencies.update(counts.keys())

        total_docs = len(documents)
        self.avgdl = sum(self.doc_lengths) / total_docs if total_docs else 0.0
        for token, frequency in document_frequencies.items():
            self.idf[token] = math.log(1 + (total_docs - frequency + 0.5) / (frequency + 0.5))
        for doc_index, counts in enumerate(documents):
            for token, frequency in counts.items():
                self.postings[token].append((doc_index, frequency))

    def rank_assignees(
        self, query_row: dict[str, Any], roster: set[str], global_counts: Counter[str], top_k: int
    ) -> list[str]:
        scores: defaultdict[int, float] = defaultdict(float)
        query_tokens = Counter(tokenize(row_text(query_row)))
        if not query_tokens:
            return from_counters([global_counts], roster, top_k)

        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            if token not in self.postings:
                continue
            idf = self.idf.get(token, 0.0)
            for doc_index, frequency in self.postings[token]:
                doc_len = self.doc_lengths[doc_index] or 1
                denominator = frequency + k1 * (1 - b + b * doc_len / (self.avgdl or 1.0))
                scores[doc_index] += idf * ((frequency * (k1 + 1)) / denominator)

        if not scores:
            return from_counters([global_counts], roster, top_k)

        assignee_scores: defaultdict[str, float] = defaultdict(float)
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.neighbors]
        for rank, (doc_index, score) in enumerate(ranked_docs, start=1):
            assignee = normalize_assignee(self.rows[doc_index].get("assignee"))
            if assignee in roster:
                assignee_scores[assignee] += score / rank
        bm25_candidates = [assignee for assignee, _ in sorted(assignee_scores.items(), key=lambda item: item[1], reverse=True)]
        fallback = from_counters([global_counts], roster, top_k)
        candidates = []
        for assignee in bm25_candidates + fallback:
            if assignee not in candidates:
                candidates.append(assignee)
            if len(candidates) >= top_k:
                break
        return candidates


def row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, "") or "")
        for field in ("title", "description", "product", "component", "severity", "priority")
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


if __name__ == "__main__":
    main()
