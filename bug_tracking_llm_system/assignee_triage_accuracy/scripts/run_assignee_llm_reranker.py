from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
PAPER_ROOT = EVAL_ROOT / "paper_grade"
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "llm_assignee" / "reports"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank current assignee candidates with a constrained local LLM.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None, help="Override split file.")
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--evidence-tickets", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--acceptance-policy",
        choices=("safe", "raw"),
        default="safe",
        help="safe keeps the current ranking unless the LLM gives a strong, non-ambiguous reorder.",
    )
    parser.add_argument(
        "--only-current-errors",
        action="store_true",
        help="Diagnostic mode only. Do not use for final metrics.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    train_path = args.train or args.data_dir / f"{args.dataset}_history_train.jsonl"
    split_suffix = "validation_set" if args.split == "validation" else "test_set"
    test_path = args.test or args.data_dir / f"{args.dataset}_{split_suffix}.jsonl"
    roster_path = args.roster or args.data_dir / f"{args.dataset}_candidate_roster.json"
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.dataset / f"{args.split}_{safe_model_name(args.model)}_top{args.candidate_k}"
    cache_path = args.cache or output_dir / "ollama_cache.jsonl"

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)
    if args.max_rows is not None:
        test_rows = test_rows[: args.max_rows]

    context = build_context(train_rows, roster)
    profiles = build_profiles(train_rows, evidence_tickets=args.evidence_tickets)
    current_predictions = current_predictions_for_rows(
        rows=test_rows,
        train_path=train_path,
        context=context,
        top_k=args.candidate_k,
    )
    if args.only_current_errors:
        current_by_id = {row["ticket_id"]: row for row in current_predictions}
        test_rows = [row for row in test_rows if not current_by_id.get(row.get("ticket_id"), {}).get("is_top1_correct")]
        current_predictions = [current_by_id[row.get("ticket_id")] for row in test_rows if row.get("ticket_id") in current_by_id]

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    reranked_predictions: list[dict[str, Any]] = []
    started_at = time.time()

    for index, (row, current) in enumerate(zip(test_rows, current_predictions, strict=True), start=1):
        candidates = list(current["ranked_candidates"])
        prompt = build_rerank_prompt(row, candidates, profiles)
        cache_key = cache_key_for(args.model, prompt)
        cache_hit = cache_key in cache
        if cache_hit:
            llm_payload = cache[cache_key]
        else:
            llm_payload = call_ollama(
                url=args.ollama_url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
                temperature=args.temperature,
            )
            append_cache(cache_path, cache_key, llm_payload)
            cache[cache_key] = llm_payload

        parsed = parse_ranking(llm_payload.get("response", ""), candidates)
        final_candidates = accept_ranking(
            candidates,
            parsed,
            policy=args.acceptance_policy,
        )
        predicted = final_candidates[0] if final_candidates else ""
        expected = normalize_assignee(row.get("assignee"))
        rank = rank_of(expected, final_candidates)
        reranked_predictions.append(
            {
                "baseline": "hybrid_llm_reranker",
                "ticket_id": row.get("ticket_id"),
                "title": row.get("title"),
                "product": normalize_component(row.get("product")),
                "component": normalize_component(row.get("component")),
                "priority": row.get("priority", "P3"),
                "expected_assignee": expected,
                "predicted_assignee": predicted,
                "ranked_candidates": final_candidates,
                "rank": rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "is_top1_correct": expected == predicted,
                "is_hit_at_3": expected in final_candidates[:3],
                "is_hit_at_5": expected in final_candidates[:5],
                "is_hit_at_10": expected in final_candidates[:10],
                "error_type": error_type(expected, predicted, row, context),
                "llm_valid_output": parsed["valid"],
                "llm_cache_hit": cache_hit,
                "llm_confidence": parsed.get("confidence"),
                "llm_reason": parsed.get("reason", ""),
                "llm_total_duration_sec": duration_seconds(llm_payload.get("total_duration")),
                "llm_prompt_eval_duration_sec": duration_seconds(llm_payload.get("prompt_eval_duration")),
                "llm_eval_duration_sec": duration_seconds(llm_payload.get("eval_duration")),
                "current_ranked_candidates": candidates,
                "current_predicted_assignee": current["predicted_assignee"],
            }
        )
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(test_rows)):
            elapsed = time.time() - started_at
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(test_rows),
                        "elapsed_sec": round(elapsed, 1),
                        "current_top1": build_metrics(current_predictions[:index], 0, args.seed)["top1_accuracy"],
                        "rerank_top1": build_metrics(reranked_predictions, 0, args.seed)["top1_accuracy"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    current_metrics = build_metrics(current_predictions, args.bootstrap_samples, args.seed)
    rerank_metrics = build_metrics(reranked_predictions, args.bootstrap_samples, args.seed)
    comparison = {
        "top1_delta_vs_current_triager": paired_delta_ci(
            [1 if row["is_top1_correct"] else 0 for row in current_predictions],
            [1 if row["is_top1_correct"] else 0 for row in reranked_predictions],
            args.bootstrap_samples,
            args.seed,
        ),
        "hit_at_3_delta": round(rerank_metrics["hit_at_3"] - current_metrics["hit_at_3"], 6),
        "hit_at_5_delta": round(rerank_metrics["hit_at_5"] - current_metrics["hit_at_5"], 6),
        "hit_at_10_delta": round(rerank_metrics["hit_at_10"] - current_metrics["hit_at_10"], 6),
        "mrr_delta": round(rerank_metrics["mrr"] - current_metrics["mrr"], 6),
        "should_promote": rerank_metrics["top1_accuracy"] > current_metrics["top1_accuracy"],
    }
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "rows": len(test_rows),
        "candidate_roster_size": len(roster),
        "candidate_k": args.candidate_k,
        "model": args.model,
        "temperature": args.temperature,
        "acceptance_policy": args.acceptance_policy,
        "current_assignee_triager": current_metrics,
        "hybrid_llm_reranker": rerank_metrics,
        "comparison": comparison,
        "llm_runtime": {
            "valid_output_rate": ratio(sum(1 for row in reranked_predictions if row["llm_valid_output"]), len(reranked_predictions)),
            "cache_hit_rate": ratio(sum(1 for row in reranked_predictions if row["llm_cache_hit"]), len(reranked_predictions)),
            "mean_total_duration_sec": mean_present(row["llm_total_duration_sec"] for row in reranked_predictions),
            "mean_prompt_eval_duration_sec": mean_present(
                row["llm_prompt_eval_duration_sec"] for row in reranked_predictions
            ),
            "mean_generation_duration_sec": mean_present(row["llm_eval_duration_sec"] for row in reranked_predictions),
        },
        "note": "Production current_assignee_triager is not modified by this experiment.",
    }

    write_json(output_dir / f"{args.dataset}_{args.split}_metrics.json", metrics)
    write_jsonl(output_dir / f"{args.dataset}_{args.split}_current_predictions.jsonl", current_predictions)
    write_jsonl(output_dir / f"{args.dataset}_{args.split}_rerank_predictions.jsonl", reranked_predictions)
    write_error_analysis(output_dir / f"{args.dataset}_{args.split}_rerank_error_analysis.csv", reranked_predictions)
    write_summary(output_dir / f"{args.dataset}_{args.split}_comparison.md", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_rerank_prompt(row: dict[str, Any], candidates: list[str], profiles: dict[str, dict[str, Any]]) -> str:
    evidence_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        profile = profiles.get(candidate, {})
        evidence_rows.append(
            {
                "assignee": candidate,
                "current_rank": rank,
                "ticket_count": profile.get("ticket_count", 0),
                "top_components": profile.get("top_components", []),
                "top_products": profile.get("top_products", []),
                "keywords": profile.get("keywords", []),
                "representative_titles": profile.get("representative_titles", []),
            }
        )
    payload = {
        "production_prior": {
            "meaning": "Candidates are ordered by the current strongest deterministic baseline.",
            "instruction": "Keep the original order unless another candidate has clearly stronger evidence.",
            "current_top1": candidates[0] if candidates else "",
            "current_order": candidates,
        },
        "issue": {
            "title": clean(row.get("title")),
            "description": clean(row.get("description")) or "(empty)",
            "product": clean(row.get("product")) or "unknown",
            "component": clean(row.get("component")) or "unknown",
            "priority": clean(row.get("priority")) or "unknown",
            "severity": clean(row.get("severity")) or "unknown",
        },
        "candidate_assignees": evidence_rows,
    }
    return (
        "You are an expert bug triager.\n"
        "Rerank only the provided candidate assignee IDs for the issue.\n"
        "The candidate order is a strong production prior from the current best baseline.\n"
        "Keep the current order if the evidence is ambiguous.\n"
        "Only move a lower-ranked candidate to rank 1 when the evidence is clearly stronger than the current rank 1.\n"
        "Do not invent IDs. Do not output IDs outside the candidate list. Do not use a gold label.\n"
        "Return strict JSON only with this schema:\n"
        "{\"ranking\":[\"dev_0001\"],\"confidence\":0.0,\"manual_triage\":false,\"reason\":\"short reason\"}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def call_ollama(*, url: str, model: str, prompt: str, timeout: int, temperature: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Ollama request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("Ollama response was not a JSON object.")
    value.pop("context", None)
    return value


def parse_ranking(text: str, candidates: list[str]) -> dict[str, Any]:
    candidate_set = set(candidates)
    payload: Any | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        return {"valid": False, "ranking": candidates, "reason": "invalid_json"}
    raw_ranking = payload.get("ranking")
    if isinstance(raw_ranking, str):
        raw_values = [value.strip() for value in raw_ranking.split(",")]
    elif isinstance(raw_ranking, list):
        raw_values = [str(value).strip() for value in raw_ranking]
    else:
        raw_values = []

    ranking = []
    for value in raw_values:
        assignee = normalize_assignee(value)
        if assignee in candidate_set and assignee not in ranking:
            ranking.append(assignee)
    for candidate in candidates:
        if candidate not in ranking:
            ranking.append(candidate)
    return {
        "valid": bool(ranking) and ranking[0] in candidate_set,
        "ranking": ranking,
        "confidence": safe_float(payload.get("confidence")),
        "reason": str(payload.get("reason", ""))[:500],
        "manual_triage": bool(payload.get("manual_triage", False)),
    }


def accept_ranking(candidates: list[str], parsed: dict[str, Any], *, policy: str) -> list[str]:
    if not parsed.get("valid"):
        return candidates
    ranking = list(parsed.get("ranking") or candidates)
    if policy == "raw":
        return ranking
    if not candidates or not ranking:
        return candidates
    if ranking[0] == candidates[0]:
        return ranking

    confidence = parsed.get("confidence")
    current_top1 = candidates[0]
    llm_top3 = set(ranking[:3])
    # Local LLM confidence is not calibrated. In safe mode, a Top-1 change is
    # accepted only when the model is confident and explicitly pushes the
    # current Top-1 out of its own top-3, which avoids many plausible-but-harmful
    # swaps among adjacent experts.
    if confidence is None or confidence < 0.85:
        return candidates
    if current_top1 in llm_top3:
        return candidates
    return ranking


def current_predictions_for_rows(
    *,
    rows: list[dict[str, Any]],
    train_path: Path,
    context: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=train_path,
        assignee_top_k=top_k,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)
    predictions = []
    for row in rows:
        result = triager.assign(row, {"predicted_priority": row.get("priority", "P3")})
        candidates = current_triager_candidates(result, context, top_k)
        expected = normalize_assignee(row.get("assignee"))
        predicted = candidates[0] if candidates else ""
        rank = rank_of(expected, candidates)
        predictions.append(
            {
                "baseline": "current_assignee_triager",
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


def current_triager_candidates(result: dict[str, Any], context: dict[str, Any], top_k: int) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        assignee = normalize_assignee(value)
        if not assignee or assignee in candidates:
            return
        if assignee == "manual_triage" or assignee not in context["roster"]:
            return
        candidates.append(assignee)

    for assignee in result.get("ranked_candidates", []):
        add(assignee)
    return candidates[:top_k]


def build_context(train_rows: list[dict[str, Any]], roster: list[str]) -> dict[str, Any]:
    global_counts: Counter[str] = Counter()
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        assignee = normalize_assignee(row.get("assignee"))
        component = normalize_component(row.get("component"))
        if assignee:
            global_counts[assignee] += 1
            component_counts[component][assignee] += 1
    return {"roster": set(roster), "global_counts": global_counts, "component_counts": dict(component_counts)}


def build_profiles(train_rows: list[dict[str, Any]], *, evidence_tickets: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        assignee = normalize_assignee(row.get("assignee"))
        if assignee:
            grouped[assignee].append(row)

    profiles = {}
    for assignee, rows in grouped.items():
        component_counts = Counter(normalize_component(row.get("component")) for row in rows)
        product_counts = Counter(normalize_component(row.get("product")) for row in rows)
        keyword_counts = Counter()
        for row in rows:
            keyword_counts.update(tokens(row_text(row)))
        representatives = sorted(rows, key=lambda row: representative_score(row, keyword_counts), reverse=True)
        profiles[assignee] = {
            "ticket_count": len(rows),
            "top_components": [label for label, _ in component_counts.most_common(5)],
            "top_products": [label for label, _ in product_counts.most_common(3)],
            "keywords": [label for label, _ in keyword_counts.most_common(8)],
            "representative_titles": [clean(row.get("title"))[:160] for row in representatives[:evidence_tickets]],
        }
    return profiles


def representative_score(row: dict[str, Any], keyword_counts: Counter[str]) -> float:
    return sum(keyword_counts[token] for token in set(tokens(row_text(row))))


def build_metrics(predictions: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {
            "rows": 0,
            "top1_accuracy": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
            "mrr": 0.0,
            "macro_f1": 0.0,
            "macro_top1_by_assignee": 0.0,
            "macro_top1_by_component": 0.0,
            "top1_bootstrap_95ci": {"low": 0.0, "high": 0.0},
            "error_breakdown": {},
        }
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
        "macro_f1": macro_f1(
            [row["expected_assignee"] for row in predictions],
            [row["predicted_assignee"] for row in predictions],
        ),
        "macro_top1_by_assignee": macro_average(by_assignee),
        "macro_top1_by_component": macro_average(by_component),
        "top1_bootstrap_95ci": bootstrap_ci(indicators, bootstrap_samples, seed),
        "error_breakdown": dict(sorted(Counter(row["error_type"] for row in predictions).items())),
    }


def paired_delta_ci(reference: list[int], candidate: list[int], samples: int, seed: int) -> dict[str, float]:
    if not reference or len(reference) != len(candidate):
        return {"delta": 0.0, "low": 0.0, "high": 0.0}
    deltas = [candidate_value - reference_value for reference_value, candidate_value in zip(reference, candidate)]
    if samples <= 0:
        value = round(sum(deltas) / len(deltas), 6)
        return {"delta": value, "low": value, "high": value}
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


def bootstrap_ci(indicators: list[int], samples: int, seed: int) -> dict[str, float]:
    if not indicators:
        return {"low": 0.0, "high": 0.0}
    if samples <= 0:
        value = round(sum(indicators) / len(indicators), 6)
        return {"low": value, "high": value}
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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def macro_average(groups: dict[str, list[int]]) -> float:
    if not groups:
        return 0.0
    return round(sum(sum(values) / len(values) for values in groups.values()) / len(groups), 6)


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    labels = sorted({label for label in expected + predicted if label})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(expected, predicted, strict=True) if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 6)


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


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("cache_key")
            payload = row.get("payload")
            if isinstance(key, str) and isinstance(payload, dict):
                cache[key] = payload
    return cache


def append_cache(path: Path, cache_key: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cache_key": cache_key, "payload": payload}, ensure_ascii=False, sort_keys=True) + "\n")


def cache_key_for(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()


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
        raise SystemExit(f"Missing roster file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_error_analysis(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "ticket_id",
        "product",
        "component",
        "priority",
        "expected_assignee",
        "current_predicted_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "llm_valid_output",
        "llm_cache_hit",
        "llm_confidence",
        "error_type",
        "ranked_candidates",
        "current_ranked_candidates",
        "llm_reason",
        "title",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            output = dict(row)
            output["ranked_candidates"] = "|".join(row.get("ranked_candidates", []))
            output["current_ranked_candidates"] = "|".join(row.get("current_ranked_candidates", []))
            writer.writerow({field: output.get(field, "") for field in fields})


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    current = metrics["current_assignee_triager"]
    rerank = metrics["hybrid_llm_reranker"]
    comparison = metrics["comparison"]
    lines = [
        f"# Assignee LLM Reranker: {metrics['dataset']} {metrics['split']}",
        "",
        f"- Rows: `{metrics['rows']}`",
        f"- Model: `{metrics['model']}`",
        f"- Candidate K: `{metrics['candidate_k']}`",
        "",
        "| Model | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
        f"| current_assignee_triager | {current['top1_accuracy']:.6f} | {current['hit_at_3']:.6f} | {current['hit_at_5']:.6f} | {current['hit_at_10']:.6f} | {current['mrr']:.6f} |",
        f"| hybrid_llm_reranker | {rerank['top1_accuracy']:.6f} | {rerank['hit_at_3']:.6f} | {rerank['hit_at_5']:.6f} | {rerank['hit_at_10']:.6f} | {rerank['mrr']:.6f} |",
        "",
        f"- Top-1 delta vs current: `{comparison['top1_delta_vs_current_triager']['delta']}`",
        f"- Should promote: `{comparison['should_promote']}`",
        "",
        "Production `current_assignee_triager` was not modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, "") or "")
        for field in ("title", "description", "product", "component", "severity", "priority")
    )


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def duration_seconds(value: Any) -> float | None:
    number = safe_float(value)
    return round(number / 1_000_000_000, 6) if number is not None else None


def mean_present(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 6) if present else None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


if __name__ == "__main__":
    main()
