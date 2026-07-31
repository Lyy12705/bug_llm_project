from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "assignee_triage_accuracy" / "paper_grade" / "data" / "processed"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "assignee_triage_accuracy" / "phase6_candidate_ltr" / "reports" / "bmo_public_10k"
DEFAULT_CURRENT_REPORT_DIR = PROJECT_ROOT / "assignee_triage_accuracy" / "paper_grade" / "reports" / "bmo_public_10k"
FEATURE_NAMES = [
    "base_score",
    "global_log_count",
    "component_log_count",
    "product_component_log_count",
    "product_log_count",
    "component_share_smoothed",
    "product_component_share_smoothed",
    "product_share_smoothed",
    "recency_component_share",
    "recency_product_component_share",
    "recent_30d_component_share",
    "recent_90d_component_share",
    "owner_recency",
    "bm25_owner_score",
    "bm25_best_issue_score",
    "bm25_owner_issue_count",
    "component_support_log",
    "product_component_support_log",
    "component_token_share",
    "reporter_owner_share",
    "file_owner_share",
    "file_support_log",
    "source_product_component",
    "source_component",
    "source_bm25",
    "source_sbert",
    "source_recent_component",
    "source_product",
    "source_global_prior",
    "source_component_token",
    "source_reporter_history",
    "source_file_history",
]
CANDIDATE_SOURCE_ORDER = (
    "product_component",
    "component",
    "component_token",
    "reporter_history",
    "file_history",
    "bm25",
    "sbert",
    "recent_component",
    "product",
    "global_prior",
)
SOURCE_FAMILIES = {
    "product_component": "component_history",
    "component": "component_history",
    "component_token": "component_history",
    "recent_component": "recent_activity",
    "reporter_history": "reporter_history",
    "file_history": "file_ownership",
    "bm25": "lexical_retrieval",
    "sbert": "semantic_retrieval",
    "product": "product_history",
    "global_prior": "global_prior",
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STACK_LINE_RE = re.compile(
    r"^\s*(?:#?\d+\s+0x[0-9a-f]+|at\s+[\w.$<>]+\(|(?:traceback|thread\s+\d+|registers?):)",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"^(?:user agent|build id|steps to reproduce|actual results?|expected results?|additional information)\s*:\s*$",
    re.IGNORECASE,
)
GENERIC_ASSIGNEE_RE = re.compile(
    r"(nobody|unassigned|triage|inbox|default|bugzilla|bugs@|noreply|do-not-reply|disabled)",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "the", "to", "when", "with",
}


@dataclass
class Candidate:
    assignee: str
    base_score: float
    features: dict[str, float]
    sources: tuple[str, ...]


def ranked_counter_owners(counter: Counter[str]) -> list[str]:
    return [owner for owner, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def ranked_score_owners(scores: dict[str, float]) -> list[str]:
    return [owner for owner, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0.0]


def candidate_source_families_from_features(features: dict[str, float]) -> set[str]:
    return {
        family
        for source, family in SOURCE_FAMILIES.items()
        if bool(features.get(f"source_{source}", 0.0))
    }


def source_quotas(
    source_rankings: dict[str, list[str]],
    pool_size: int,
    *,
    candidate_generator_version: str = "v2",
) -> dict[str, int]:
    if candidate_generator_version == "v3":
        weights = {
            "product_component": 0.18,
            "component": 0.20,
            "component_token": 0.08,
            "reporter_history": 0.07,
            "file_history": 0.10,
            "bm25": 0.18,
            "sbert": 0.18,
            "recent_component": 0.08,
            "product": 0.05,
            "global_prior": 0.02,
        }
    elif candidate_generator_version == "v2":
        weights = {
            "product_component": 0.20,
            "component": 0.25,
            "component_token": 0.0,
            "reporter_history": 0.0,
            "file_history": 0.0,
            "bm25": 0.20,
            "sbert": 0.20,
            "recent_component": 0.08,
            "product": 0.05,
            "global_prior": 0.02,
        }
    else:
        raise ValueError("unsupported candidate generator version")
    available = [
        source
        for source in CANDIDATE_SOURCE_ORDER
        if source_rankings.get(source) and weights[source] > 0.0
    ]
    if not available or pool_size <= 0:
        return {}
    weight_total = sum(weights[source] for source in available)
    quotas = {
        source: max(1, int(math.floor(pool_size * weights[source] / weight_total)))
        for source in available
    }
    while sum(quotas.values()) > pool_size:
        source = max((name for name in available if quotas[name] > 1), key=quotas.get, default="")
        if not source:
            break
        quotas[source] -= 1
    for source in sorted(
        available,
        key=lambda name: (-(pool_size * weights[name] / weight_total - quotas[name]), CANDIDATE_SOURCE_ORDER.index(name)),
    ):
        if sum(quotas.values()) >= pool_size:
            break
        quotas[source] += 1
    return {source: quotas[source] for source in CANDIDATE_SOURCE_ORDER if source in quotas}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a lightweight temporal candidate-level assignee ranker."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl")
    parser.add_argument("--validation", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl")
    parser.add_argument("--test", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_test_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--candidate-pool-size", type=int, default=30)
    parser.add_argument("--output-k", type=int, default=10)
    parser.add_argument("--temporal-folds", type=int, default=4)
    parser.add_argument("--warmup-fraction", type=float, default=0.45)
    parser.add_argument("--half-life-days", type=float, default=90.0)
    parser.add_argument("--smoothing-alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--ranker",
        choices=(
            "auto",
            "logistic_regression",
            "pairwise_logistic_hard_negative",
            "hist_gradient_boosting_shallow",
            "hist_gradient_boosting",
        ),
        default="auto",
        help="Fix the ranker before a new holdout; auto selects on validation only.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("top1", "top3"),
        default="top1",
        help="Validation-only model selection objective fixed before the later holdout.",
    )
    parser.add_argument(
        "--artifact-name",
        default="",
        help="Explicit immutable model version. Defaults to the generated v3 name.",
    )
    parser.add_argument(
        "--current-validation-metrics",
        type=Path,
        default=DEFAULT_CURRENT_REPORT_DIR / "validation" / "bmo_public_10k_metrics.json",
    )
    parser.add_argument(
        "--current-test-metrics",
        type=Path,
        default=DEFAULT_CURRENT_REPORT_DIR / "bmo_public_10k_metrics.json",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("none", "sbert"),
        default="none",
        help="Optional local-only semantic feature. No model download is attempted.",
    )
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--new-holdout",
        action="store_true",
        help="Treat --test as a newly acquired untouched temporal holdout and include it in the promotion gate.",
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
        help="Optional source-assignee to anonymized-label map for later public holdouts.",
    )
    args = parser.parse_args()

    if args.new_holdout:
        raise SystemExit(
            "--new-holdout is disabled for ranker training. Freeze the ranker first, "
            "then use evaluate_assignee_open_set_holdout.py or "
            "evaluate_assignee_rolling_holdout.py for the one-time sealed evaluation."
        )
    if args.candidate_pool_size < max(30, args.output_k):
        raise SystemExit(
            "v3 requires --candidate-pool-size >= 30 and >= --output-k so recall@30 is auditable"
        )
    if not 0.2 <= args.warmup_fraction <= 0.8:
        raise SystemExit("--warmup-fraction must be between 0.2 and 0.8")

    label_map = read_label_map(args.assignee_label_map)
    train_rows = sorted(apply_label_map(canonicalize_source_rows(read_jsonl(args.train)), label_map), key=row_sort_key)
    validation_rows = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.validation)), label_map), key=row_sort_key
    )
    test_rows = sorted(apply_label_map(canonicalize_source_rows(read_jsonl(args.test)), label_map), key=row_sort_key)
    test_rows, cross_split_overlap_rows_excluded = exclude_cross_split_overlap(
        test_rows,
        [*train_rows, *validation_rows],
        strict_duplicate_families=True,
    )
    ensure_temporal_order(train_rows, validation_rows, test_rows)

    semantic = build_semantic_backend(args.embedding_backend, args.sbert_model)
    training_examples, fold_audit = build_temporal_training_examples(
        train_rows,
        folds=args.temporal_folds,
        warmup_fraction=args.warmup_fraction,
        pool_size=args.candidate_pool_size,
        half_life_days=args.half_life_days,
        smoothing_alpha=args.smoothing_alpha,
        semantic=semantic,
        candidate_generator_version="v3",
    )
    models = fit_rankers(training_examples, seed=args.seed)

    train_index = CandidateIndex(
        train_rows,
        half_life_days=args.half_life_days,
        smoothing_alpha=args.smoothing_alpha,
        semantic=semantic,
        candidate_generator_version="v3",
    )
    validation_by_model = {
        name: evaluate_split(validation_rows, train_index, model, args.candidate_pool_size, args.output_k)
        for name, model in models.items()
    }
    development_by_model = {
        name: evaluate_split(
            test_rows,
            train_index,
            candidate_model,
            args.candidate_pool_size,
            args.output_k,
        )
        for name, candidate_model in models.items()
    }
    selected_name = (
        select_model_across_windows(
            {
                "validation": validation_by_model,
                "later_development": development_by_model,
            },
            objective=args.selection_objective,
        )
        if args.ranker == "auto"
        else args.ranker
    )
    model = models[selected_name]
    validation_result = validation_by_model[selected_name]
    test_result = development_by_model[selected_name]
    current_validation = read_current_hybrid_metrics(args.current_validation_metrics)
    current_test = read_current_hybrid_metrics(args.current_test_metrics)

    development_gate = v3_ranker_development_gate(
        {
            "validation": validation_result,
            "later_development": test_result,
        }
    )
    promote = bool(development_gate["passed"])
    artifact = export_artifact(model, selected_name, args, validation_result, fold_audit, promote)
    artifact["development_gate"] = development_gate
    report = {
        "dataset": args.train.stem.replace("_history_train", ""),
        "method": "candidate_ltr_v3_multi_window",
        "deployment_status": "research_only",
        "validation_candidate": promote,
        "model": {
            "selected": selected_name,
            "selection_rule": (
                f"worst-window {args.selection_objective} improvement, then mean MRR and Macro-F1"
                if args.ranker == "auto"
                else "fixed_by_cli_before_holdout"
            ),
            "compared": sorted(models),
            "training_objectives": {
                name: str(
                    getattr(candidate_model, "training_objective_", "unknown")
                )
                for name, candidate_model in models.items()
            },
            "pairwise_training_audit": getattr(
                models.get("pairwise_logistic_hard_negative"),
                "pairwise_training_audit_",
                {},
            ),
            "long_tail_sample_weight": "1/sqrt(train_assignee_frequency)",
            "features": FEATURE_NAMES + semantic.feature_names,
            "embedding_backend": semantic.status,
        },
        "protocol": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
            "cross_split_overlap_rows_excluded": cross_split_overlap_rows_excluded,
            "candidate_pool_size": args.candidate_pool_size,
            "output_k": args.output_k,
            "temporal_folds": fold_audit,
            "test_interpretation": "later_development_only_previously_inspected",
            "sealed_holdout_labels_accessed": False,
            "file_path_feature_policy": "report_time_provenance_required",
        },
        "model_comparison_validation": {
            name: result["metrics"] for name, result in validation_by_model.items()
        },
        "model_comparison_development": {
            name: result["known_owner_metrics"] for name, result in development_by_model.items()
        },
        "current_hybrid_reference": {
            "validation": current_validation,
            "test_exploratory": current_test,
        },
        "validation": validation_result["metrics"],
        "validation_known_owner": validation_result["known_owner_metrics"],
        "test_exploratory": test_result["metrics"],
        "test_exploratory_known_owner": test_result["known_owner_metrics"],
        "test_exploratory_unseen_owner_rate": test_result["unseen_owner_rate"],
        "candidate_source_recall": candidate_source_recall_report(
            {
                "validation": validation_result,
                "later_development": test_result,
            }
        ),
        "candidate_source_ablation": candidate_source_ablation_report(
            {
                "validation": validation_result,
                "later_development": test_result,
            }
        ),
        "development_error_taxonomy": {
            "validation": prediction_error_taxonomy(validation_result["predictions"]),
            "later_development": prediction_error_taxonomy(test_result["predictions"]),
        },
        "v3_development_gate": development_gate,
        "promotion_gate": promotion_gate_payload(
            validation_result,
            current_validation,
            test_result,
            False,
            promote,
            development_gate,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "candidate_ltr_report.json", report)
    write_json(
        args.output_dir / "candidate_source_recall_report.json",
        report["candidate_source_recall"],
    )
    write_json(
        args.output_dir / "candidate_source_ablation_report.json",
        report["candidate_source_ablation"],
    )
    write_json(args.output_dir / "candidate_ltr_artifact.json", artifact)
    joblib.dump(model, args.output_dir / "candidate_ltr_model.joblib")
    write_jsonl(args.output_dir / "validation_predictions.jsonl", validation_result["predictions"])
    write_jsonl(args.output_dir / "test_exploratory_predictions.jsonl", test_result["predictions"])
    write_markdown(args.output_dir / "candidate_ltr_summary.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


class SemanticBackend:
    def __init__(self, *, status: str, model: Any | None = None) -> None:
        self.status = status
        self.model = model
        self.feature_names = ["sbert_owner_score"] if model is not None else []

    def build(self, history_texts: list[str]) -> Any | None:
        if self.model is None or not history_texts:
            return None
        return self.model.encode(history_texts, normalize_embeddings=True, show_progress_bar=False)

    def owner_scores(
        self,
        query_text: str,
        history_embeddings: Any | None,
        history_rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        if self.model is None or history_embeddings is None:
            return {}
        query = self.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray(history_embeddings) @ np.asarray(query)
        if not len(scores):
            return {}
        best_indices = np.argsort(scores)[-25:][::-1]
        owners: defaultdict[str, float] = defaultdict(float)
        for rank, index in enumerate(best_indices, start=1):
            owner = normalize_assignee(history_rows[int(index)].get("assignee"))
            if owner:
                owners[owner] += max(0.0, float(scores[int(index)])) / rank
        return dict(owners)


def build_semantic_backend(name: str, model_name: str) -> SemanticBackend:
    if name != "sbert":
        return SemanticBackend(status="disabled")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True)
        return SemanticBackend(status=f"sbert_local:{model_name}", model=model)
    except Exception as exc:
        return SemanticBackend(status=f"unavailable_local_only:{type(exc).__name__}")


class CandidateIndex:
    def __init__(
        self,
        history_rows: list[dict[str, Any]],
        *,
        half_life_days: float,
        smoothing_alpha: float,
        semantic: SemanticBackend,
        candidate_generator_version: str = "v2",
    ) -> None:
        self.rows = [normalize_row(row) for row in history_rows]
        self.half_life_days = max(1.0, half_life_days)
        self.smoothing_alpha = max(0.0, smoothing_alpha)
        self.semantic = semantic
        if candidate_generator_version not in {"v2", "v3"}:
            raise ValueError("unsupported candidate generator version")
        self.candidate_generator_version = candidate_generator_version
        self.global_counts: Counter[str] = Counter()
        self.component_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.product_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.pair_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.component_token_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.reporter_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.file_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.owner_timestamps: defaultdict[str, list[float]] = defaultdict(list)
        self.component_owner_timestamps: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        self.pair_owner_timestamps: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
        for row in self.rows:
            owner = row["assignee"]
            if not owner:
                continue
            component = row["component"]
            product = row["product"]
            pair = pair_key(product, component)
            self.global_counts[owner] += 1
            self.component_counts[component][owner] += 1
            self.product_counts[product][owner] += 1
            self.pair_counts[pair][owner] += 1
            for token in set(component_tokens(component)):
                self.component_token_counts[token][owner] += 1
            reporter = normalize_assignee(row.get("reporter") or row.get("creator"))
            if reporter:
                self.reporter_counts[reporter][owner] += 1
            for key in file_history_keys(row.get("file_paths")):
                self.file_counts[key][owner] += 1
            if row["timestamp"] is not None:
                self.owner_timestamps[owner].append(row["timestamp"])
                self.component_owner_timestamps[(component, owner)].append(row["timestamp"])
                self.pair_owner_timestamps[(pair, owner)].append(row["timestamp"])
        self.bm25 = BM25Index(self.rows)
        self.history_embeddings = semantic.build([row["text"] for row in self.rows])

    def candidates(self, query: dict[str, Any], pool_size: int, *, include_expected: bool = False) -> list[Candidate]:
        row = normalize_row(query)
        component = row["component"]
        product = row["product"]
        pair = pair_key(product, component)
        bm25_scores, bm25_best, bm25_hits = self.bm25.owner_scores(row["text"])
        semantic_scores = self.semantic.owner_scores(row["text"], self.history_embeddings, self.rows)
        query_ts = row["timestamp"] or self.latest_timestamp()
        recency_component = self.recency_counter(component, query_ts, self.component_owner_timestamps)
        recency_pair = self.recency_counter(pair, query_ts, self.pair_owner_timestamps)
        recent_30 = self.recent_counter(component, query_ts, self.component_owner_timestamps, 30)
        recent_90 = self.recent_counter(component, query_ts, self.component_owner_timestamps, 90)
        component_token_scores: Counter[str] = Counter()
        reporter_history: Counter[str] = Counter()
        file_history: Counter[str] = Counter()
        if self.candidate_generator_version == "v3":
            for token in set(component_tokens(component)):
                component_token_scores.update(self.component_token_counts[token])
            reporter = normalize_assignee(row.get("reporter") or row.get("creator"))
            reporter_history = self.reporter_counts[reporter]
            for key in file_history_keys(row.get("file_paths")):
                file_history.update(self.file_counts[key])

        source_rankings = {
            "product_component": ranked_counter_owners(self.pair_counts[pair]),
            "component": ranked_counter_owners(self.component_counts[component]),
            "component_token": ranked_counter_owners(component_token_scores),
            "reporter_history": ranked_counter_owners(reporter_history),
            "file_history": ranked_counter_owners(file_history),
            "bm25": ranked_score_owners(bm25_scores),
            "sbert": ranked_score_owners(semantic_scores),
            "recent_component": ranked_counter_owners(recent_90),
            "product": ranked_counter_owners(self.product_counts[product]),
            "global_prior": ranked_counter_owners(self.global_counts),
        }
        quotas = source_quotas(
            source_rankings,
            pool_size,
            candidate_generator_version=self.candidate_generator_version,
        )
        sources_by_owner: defaultdict[str, set[str]] = defaultdict(set)
        for source, owners in source_rankings.items():
            for owner in owners:
                sources_by_owner[owner].add(source)
        quota_sources_by_owner: defaultdict[str, set[str]] = defaultdict(set)
        for source, quota in quotas.items():
            for owner in source_rankings[source][:quota]:
                quota_sources_by_owner[owner].add(source)
        owners = set(sources_by_owner)
        expected = row["assignee"]
        if include_expected and expected in self.global_counts:
            owners.add(expected)
            quota_sources_by_owner[expected].add("training_positive")

        candidates = []
        for owner in owners:
            feature = self.features_for(
                owner,
                row,
                bm25_scores=bm25_scores,
                bm25_best=bm25_best,
                bm25_hits=bm25_hits,
                semantic_scores=semantic_scores,
                recency_component=recency_component,
                recency_pair=recency_pair,
                recent_30=recent_30,
                recent_90=recent_90,
                component_token_scores=component_token_scores,
                reporter_history=reporter_history,
                file_history=file_history,
                query_ts=query_ts,
                sources=quota_sources_by_owner[owner],
            )
            candidates.append(
                Candidate(owner, feature["base_score"], feature, tuple(sorted(quota_sources_by_owner[owner])))
            )
        candidates.sort(key=lambda item: (-item.base_score, item.assignee))
        by_owner = {candidate.assignee: candidate for candidate in candidates}
        selected: list[Candidate] = []
        selected_owners: set[str] = set()
        for source, quota in quotas.items():
            if len(selected) >= pool_size:
                break
            for owner in source_rankings[source]:
                if len(selected) >= pool_size:
                    break
                if owner in selected_owners:
                    continue
                selected.append(by_owner[owner])
                selected_owners.add(owner)
                if sum(source in candidate.sources for candidate in selected) >= quota:
                    break
        for candidate in candidates:
            if len(selected) >= pool_size:
                break
            if candidate.assignee not in selected_owners:
                selected.append(candidate)
                selected_owners.add(candidate.assignee)
        result = sorted(selected[:pool_size], key=lambda item: (-item.base_score, item.assignee))
        if include_expected and expected in self.global_counts and expected not in {item.assignee for item in result}:
            expected_item = next(item for item in candidates if item.assignee == expected)
            result = [*result, expected_item]
        return result

    def features_for(
        self,
        owner: str,
        row: dict[str, Any],
        *,
        bm25_scores: dict[str, float],
        bm25_best: dict[str, float],
        bm25_hits: Counter[str],
        semantic_scores: dict[str, float],
        recency_component: Counter[str],
        recency_pair: Counter[str],
        recent_30: Counter[str],
        recent_90: Counter[str],
        component_token_scores: Counter[str],
        reporter_history: Counter[str],
        file_history: Counter[str],
        query_ts: float | None,
        sources: set[str],
    ) -> dict[str, float]:
        component = row["component"]
        product = row["product"]
        pair = pair_key(product, component)
        global_count = self.global_counts[owner]
        component_count = self.component_counts[component][owner]
        pair_count = self.pair_counts[pair][owner]
        product_count = self.product_counts[product][owner]
        global_share = ratio(global_count, sum(self.global_counts.values()))
        component_share = smoothed_share(
            component_count, sum(self.component_counts[component].values()), global_share, self.smoothing_alpha
        )
        pair_share = smoothed_share(
            pair_count, sum(self.pair_counts[pair].values()), global_share, self.smoothing_alpha
        )
        product_share = smoothed_share(
            product_count, sum(self.product_counts[product].values()), global_share, self.smoothing_alpha
        )
        rec_component_share = ratio(recency_component[owner], sum(recency_component.values()))
        rec_pair_share = ratio(recency_pair[owner], sum(recency_pair.values()))
        recent_30_share = ratio(recent_30[owner], sum(recent_30.values()))
        recent_90_share = ratio(recent_90[owner], sum(recent_90.values()))
        component_token_share = ratio(
            component_token_scores[owner], sum(component_token_scores.values())
        )
        reporter_owner_share = ratio(
            reporter_history[owner], sum(reporter_history.values())
        )
        file_owner_share = ratio(file_history[owner], sum(file_history.values()))
        owner_recency = self.owner_recency(owner, query_ts)
        bm25_owner = bm25_scores.get(owner, 0.0)
        base_score = (
            4.0 * pair_share
            + 3.0 * component_share
            + 0.75 * product_share
            + bm25_owner
            + 1.25 * rec_component_share
            + 0.75 * rec_pair_share
            + 0.5 * recent_90_share
            + 0.75 * component_token_share
            + 1.0 * reporter_owner_share
            + 1.25 * file_owner_share
            + 1.25 * semantic_scores.get(owner, 0.0)
        )
        values = {
            "base_score": base_score,
            "global_log_count": math.log1p(global_count),
            "component_log_count": math.log1p(component_count),
            "product_component_log_count": math.log1p(pair_count),
            "product_log_count": math.log1p(product_count),
            "component_share_smoothed": component_share,
            "product_component_share_smoothed": pair_share,
            "product_share_smoothed": product_share,
            "recency_component_share": rec_component_share,
            "recency_product_component_share": rec_pair_share,
            "recent_30d_component_share": recent_30_share,
            "recent_90d_component_share": recent_90_share,
            "component_token_share": component_token_share,
            "reporter_owner_share": reporter_owner_share,
            "file_owner_share": file_owner_share,
            "file_support_log": math.log1p(sum(file_history.values())),
            "owner_recency": owner_recency,
            "bm25_owner_score": bm25_owner,
            "bm25_best_issue_score": bm25_best.get(owner, 0.0),
            "bm25_owner_issue_count": float(bm25_hits[owner]),
            "component_support_log": math.log1p(sum(self.component_counts[component].values())),
            "product_component_support_log": math.log1p(sum(self.pair_counts[pair].values())),
            "source_product_component": float("product_component" in sources),
            "source_component": float("component" in sources),
            "source_bm25": float("bm25" in sources),
            "source_sbert": float("sbert" in sources),
            "source_recent_component": float("recent_component" in sources),
            "source_product": float("product" in sources),
            "source_global_prior": float("global_prior" in sources),
            "source_component_token": float("component_token" in sources),
            "source_reporter_history": float("reporter_history" in sources),
            "source_file_history": float("file_history" in sources),
        }
        if self.semantic.model is not None:
            values["sbert_owner_score"] = semantic_scores.get(owner, 0.0)
        return values

    def latest_timestamp(self) -> float | None:
        values = [row["timestamp"] for row in self.rows if row["timestamp"] is not None]
        return max(values) if values else None

    def recency_counter(
        self,
        key: str,
        query_ts: float | None,
        source: dict[tuple[str, str], list[float]],
    ) -> Counter[str]:
        result: Counter[str] = Counter()
        if query_ts is None:
            return result
        decay = math.log(2.0) / self.half_life_days
        for (source_key, owner), timestamps in source.items():
            if source_key != key:
                continue
            result[owner] = sum(
                math.exp(-decay * max(0.0, (query_ts - timestamp) / 86400.0))
                for timestamp in timestamps
                if timestamp <= query_ts
            )
        return result

    @staticmethod
    def recent_counter(
        key: str,
        query_ts: float | None,
        source: dict[tuple[str, str], list[float]],
        days: int,
    ) -> Counter[str]:
        result: Counter[str] = Counter()
        if query_ts is None:
            return result
        cutoff = query_ts - days * 86400.0
        for (source_key, owner), timestamps in source.items():
            if source_key == key:
                result[owner] = sum(cutoff <= timestamp <= query_ts for timestamp in timestamps)
        return result

    def owner_recency(self, owner: str, query_ts: float | None) -> float:
        if query_ts is None or not self.owner_timestamps[owner]:
            return 0.0
        latest = max((value for value in self.owner_timestamps[owner] if value <= query_ts), default=None)
        if latest is None:
            return 0.0
        age_days = max(0.0, (query_ts - latest) / 86400.0)
        return math.exp(-math.log(2.0) * age_days / self.half_life_days)


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []
        self.idf: dict[str, float] = {}
        doc_frequency: Counter[str] = Counter()
        token_counts = []
        for row in rows:
            counts = Counter(tokens(row["text"]))
            token_counts.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            doc_frequency.update(counts.keys())
        self.avgdl = ratio(sum(self.doc_lengths), len(self.doc_lengths)) or 1.0
        total = len(rows)
        for token, frequency in doc_frequency.items():
            self.idf[token] = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
        for index, counts in enumerate(token_counts):
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))

    def owner_scores(self, text: str) -> tuple[dict[str, float], dict[str, float], Counter[str]]:
        scores: defaultdict[int, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for token in set(tokens(text)):
            for index, frequency in self.postings.get(token, []):
                denominator = frequency + k1 * (1 - b + b * self.doc_lengths[index] / self.avgdl)
                scores[index] += self.idf[token] * (frequency * (k1 + 1) / denominator)
        owners: defaultdict[str, float] = defaultdict(float)
        best: defaultdict[str, float] = defaultdict(float)
        hits: Counter[str] = Counter()
        for rank, (index, score) in enumerate(sorted(scores.items(), key=lambda item: -item[1])[:25], start=1):
            owner = self.rows[index]["assignee"]
            owners[owner] += min(3.0, score) / rank
            best[owner] = max(best[owner], score)
            hits[owner] += 1
        return dict(owners), dict(best), hits


def build_temporal_training_examples(
    rows: list[dict[str, Any]],
    *,
    folds: int,
    warmup_fraction: float,
    pool_size: int,
    half_life_days: float,
    smoothing_alpha: float,
    semantic: SemanticBackend,
    candidate_generator_version: str = "v2",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start = max(1, int(len(rows) * warmup_fraction))
    remaining = len(rows) - start
    width = max(1, math.ceil(remaining / max(1, folds)))
    examples = []
    audit = []
    for fold in range(max(1, folds)):
        query_start = start + fold * width
        query_end = min(len(rows), query_start + width)
        if query_start >= len(rows):
            break
        history = rows[:query_start]
        queries = rows[query_start:query_end]
        index = CandidateIndex(
            history,
            half_life_days=half_life_days,
            smoothing_alpha=smoothing_alpha,
            semantic=semantic,
            candidate_generator_version=candidate_generator_version,
        )
        history_counts = index.global_counts
        positives = 0
        example_start = len(examples)
        for query_index, query in enumerate(queries):
            expected = normalize_assignee(query.get("assignee"))
            if expected not in history_counts:
                continue
            candidates = index.candidates(query, pool_size, include_expected=True)
            tail_weight = 1.0 / math.sqrt(max(1, history_counts[expected]))
            query_id = str(query.get("ticket_id") or f"fold-{fold + 1}-{query_index}")
            for base_rank, candidate in enumerate(candidates, start=1):
                label = int(candidate.assignee == expected)
                positives += label
                examples.append(
                    {
                        "features": candidate.features,
                        "label": label,
                        "sample_weight": tail_weight,
                        "query_id": query_id,
                        "candidate_assignee": candidate.assignee,
                        "base_rank": base_rank,
                    }
                )
        audit.append(
            {
                "fold": fold + 1,
                "history_rows": len(history),
                "query_rows": len(queries),
                "candidate_examples": len(examples) - example_start,
                "positive_examples": positives,
                "history_end": timestamp_string(history[-1]) if history else "",
                "query_start": timestamp_string(queries[0]) if queries else "",
                "leakage_free": not history or not queries or row_sort_key(history[-1]) <= row_sort_key(queries[0]),
            }
        )
    if not examples or not any(item["label"] for item in examples):
        raise SystemExit("Temporal folds produced no usable candidate-level training examples.")
    return examples, audit


def pairwise_training_data(
    examples: list[dict[str, Any]],
    feature_names: list[str],
    *,
    maximum_hard_negatives_per_query: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if maximum_hard_negatives_per_query < 1:
        raise ValueError("maximum hard negatives must be positive")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        query_id = str(item.get("query_id") or "").strip()
        if query_id:
            grouped[query_id].append(item)
    pair_rows: list[list[float]] = []
    pair_labels: list[int] = []
    pair_weights: list[float] = []
    usable_queries = 0
    selected_negatives = 0
    for query_id in sorted(grouped):
        members = grouped[query_id]
        positives = [item for item in members if int(item["label"]) == 1]
        negatives = sorted(
            (item for item in members if int(item["label"]) == 0),
            key=lambda item: (
                int(item.get("base_rank") or 10**9),
                -float(item["features"].get("base_score", 0.0)),
                str(item.get("candidate_assignee") or ""),
            ),
        )[:maximum_hard_negatives_per_query]
        if not positives or not negatives:
            continue
        usable_queries += 1
        positive = positives[0]
        positive_vector = np.asarray(
            [positive["features"].get(name, 0.0) for name in feature_names],
            dtype=float,
        )
        for negative in negatives:
            negative_vector = np.asarray(
                [negative["features"].get(name, 0.0) for name in feature_names],
                dtype=float,
            )
            difference = positive_vector - negative_vector
            weight = float(positive.get("sample_weight", 1.0)) / math.sqrt(
                max(1, int(negative.get("base_rank") or 1))
            )
            pair_rows.extend([difference.tolist(), (-difference).tolist()])
            pair_labels.extend([1, 0])
            pair_weights.extend([weight, weight])
            selected_negatives += 1
    if not pair_rows:
        return (
            np.empty((0, len(feature_names))),
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
            {
                "usable_queries": 0,
                "selected_hard_negatives": 0,
                "pairwise_rows": 0,
                "maximum_hard_negatives_per_query": maximum_hard_negatives_per_query,
            },
        )
    return (
        np.asarray(pair_rows, dtype=float),
        np.asarray(pair_labels, dtype=int),
        np.asarray(pair_weights, dtype=float),
        {
            "usable_queries": usable_queries,
            "selected_hard_negatives": selected_negatives,
            "pairwise_rows": len(pair_rows),
            "maximum_hard_negatives_per_query": maximum_hard_negatives_per_query,
        },
    )


def fit_rankers(examples: list[dict[str, Any]], *, seed: int) -> dict[str, Pipeline]:
    feature_names = list(examples[0]["features"])
    matrix = np.asarray([[item["features"].get(name, 0.0) for name in feature_names] for item in examples])
    labels = np.asarray([item["label"] for item in examples])
    weights = np.asarray([item["sample_weight"] for item in examples])
    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=1000, random_state=seed)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("scale", "passthrough"),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        max_depth=5,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_shallow": Pipeline(
            [
                ("scale", "passthrough"),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        max_depth=3,
                        min_samples_leaf=40,
                        l2_regularization=1.5,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    for model in models.values():
        model.fit(matrix, labels, model__sample_weight=weights)
        model.feature_names_ = feature_names
        model.training_objective_ = "pointwise_binary_relevance"
    pair_matrix, pair_labels, pair_weights, pair_audit = pairwise_training_data(
        examples, feature_names
    )
    if len(pair_labels):
        pairwise = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=0.25, max_iter=1000, random_state=seed),
                ),
            ]
        )
        pairwise.fit(
            pair_matrix,
            pair_labels,
            model__sample_weight=pair_weights,
        )
        pairwise.feature_names_ = feature_names
        pairwise.training_objective_ = "pairwise_hard_negative_logistic"
        pairwise.pairwise_training_audit_ = pair_audit
        models["pairwise_logistic_hard_negative"] = pairwise
    return models


def select_model(results: dict[str, dict[str, Any]], *, objective: str = "top1") -> str:
    if objective not in {"top1", "top3"}:
        raise ValueError("objective must be top1 or top3")
    primary = "top1_accuracy" if objective == "top1" else "hit_at_3"
    return max(
        results,
        key=lambda name: (
            results[name]["metrics"][primary],
            results[name]["metrics"]["mrr"],
            results[name]["metrics"]["top1_accuracy"],
            results[name]["metrics"]["macro_f1"],
        ),
    )


def select_model_across_windows(
    windows: dict[str, dict[str, dict[str, Any]]], *, objective: str = "top1"
) -> str:
    """Select one ranker by its worst temporal window, never by one peak score."""

    if objective not in {"top1", "top3"}:
        raise ValueError("objective must be top1 or top3")
    if len(windows) < 2 or any(not results for results in windows.values()):
        raise ValueError("robust ranker selection requires at least two windows")
    model_names = set.intersection(*(set(results) for results in windows.values()))
    if not model_names:
        raise ValueError("ranker comparison windows have no model in common")
    primary = "top1_accuracy" if objective == "top1" else "hit_at_3"

    def score(name: str) -> tuple[float, ...]:
        metrics = [windows[window][name]["known_owner_metrics"] for window in windows]
        deltas = [
            float(row[primary])
            - float(row["base_top1_accuracy"] if objective == "top1" else row["hit_at_3"])
            for row in metrics
        ]
        if objective == "top3":
            # The deterministic base has no separately ranked Top-3 metric.
            # Do not invent a delta; use the observed worst-window Top-3 first.
            deltas = [float(row[primary]) for row in metrics]
        return (
            min(deltas),
            min(float(row[primary]) for row in metrics),
            float(np.mean([row[primary] for row in metrics])),
            float(np.mean([row["mrr"] for row in metrics])),
            min(float(row["macro_f1"]) for row in metrics),
        )

    return max(sorted(model_names), key=score)


def evaluate_split(
    rows: list[dict[str, Any]],
    index: CandidateIndex,
    model: Pipeline,
    pool_size: int,
    output_k: int,
) -> dict[str, Any]:
    predictions = []
    for row in rows:
        expected = normalize_assignee(row.get("assignee"))
        candidates = index.candidates(row, pool_size)
        base = [candidate.assignee for candidate in candidates]
        source_hits = {
            source: any(candidate.assignee == expected and source in candidate.sources for candidate in candidates)
            for source in CANDIDATE_SOURCE_ORDER
        }
        source_hits_at_k = {
            str(cutoff): {
                source: any(
                    candidate.assignee == expected and source in candidate.sources
                    for candidate in candidates[:cutoff]
                )
                for source in CANDIDATE_SOURCE_ORDER
            }
            for cutoff in (10, 30, 50)
        }
        scored = rank_with_model(candidates, model)
        ranked = [candidate.assignee for candidate, _ in scored]
        expected_history_count = int(index.global_counts.get(expected, 0))
        predictions.append(
            {
                "ticket_id": row.get("ticket_id"),
                "expected_assignee": expected,
                "known_owner": expected in index.global_counts,
                "base_ranked_candidates": base[:output_k],
                "ranked_candidates": ranked[:output_k],
                "candidate_pool": base,
                "candidate_sources": {
                    candidate.assignee: list(candidate.sources) for candidate in candidates
                },
                "candidate_pool_recall": expected in base,
                "candidate_pool_size": len(candidates),
                "candidate_source_hits": source_hits,
                "candidate_source_hits_at_k": source_hits_at_k,
                "expected_owner_history_count": expected_history_count,
                "base_rank": rank_of(expected, base),
                "rank": rank_of(expected, ranked),
                "base_top1_correct": bool(base and base[0] == expected),
                "top1_correct": bool(ranked and ranked[0] == expected),
                "product": row.get("product") or "unknown",
                "component": row.get("component") or "unknown",
                "created_at": row_timestamp(row),
                "description_missing": not bool(str(row.get("description") or "").strip()),
                "routing_safe_file_paths_missing": not bool(
                    routing_safe_file_paths(row)
                ),
                "file_paths_provenance": str(
                    row.get("file_paths_provenance") or ""
                ),
                "title": row.get("title", ""),
            }
        )
    known_predictions = [row for row in predictions if row["known_owner"]]
    return {
        "metrics": ranking_metrics(predictions, output_k),
        "known_owner_metrics": ranking_metrics(known_predictions, output_k),
        "unseen_owner_rate": round(ratio(len(predictions) - len(known_predictions), len(predictions)), 6),
        "predictions": predictions,
    }


def rank_with_model(candidates: list[Candidate], model: Pipeline) -> list[tuple[Candidate, float]]:
    if not candidates:
        return []
    names = model.feature_names_
    matrix = np.asarray([[candidate.features.get(name, 0.0) for name in names] for candidate in candidates])
    probabilities = model.predict_proba(matrix)[:, 1]
    return sorted(zip(candidates, probabilities, strict=True), key=lambda item: (-float(item[1]), item[0].assignee))


def ranking_metrics(predictions: list[dict[str, Any]], output_k: int) -> dict[str, Any]:
    total = len(predictions)
    expected = [row["expected_assignee"] for row in predictions]
    predicted = [row["ranked_candidates"][0] if row["ranked_candidates"] else "manual_triage" for row in predictions]
    base_predicted = [
        row["base_ranked_candidates"][0] if row["base_ranked_candidates"] else "manual_triage"
        for row in predictions
    ]
    metrics = {
        "rows": total,
        "candidate_recall_at_pool": ratio(sum(row["candidate_pool_recall"] for row in predictions), total),
        "mean_candidate_pool_size": ratio(sum(row["candidate_pool_size"] for row in predictions), total),
        "base_top1_accuracy": ratio(sum(row["base_top1_correct"] for row in predictions), total),
        "top1_accuracy": ratio(sum(row["top1_correct"] for row in predictions), total),
        "base_macro_f1": safe_macro_f1(expected, base_predicted),
        "macro_f1": safe_macro_f1(expected, predicted),
        "candidate_recall_by_source": {
            source: round(
                ratio(sum(row["candidate_source_hits"].get(source, False) for row in predictions), total), 6
            )
            for source in CANDIDATE_SOURCE_ORDER
        },
    }
    for cutoff in (10, 30, 50):
        metrics[f"candidate_recall_at_{cutoff}"] = ratio(
            sum(
                row["expected_assignee"]
                in row.get("candidate_pool", [])[:cutoff]
                for row in predictions
            ),
            total,
        )
        metrics[f"candidate_recall_by_source_at_{cutoff}"] = {
            source: round(
                ratio(
                    sum(
                        row.get("candidate_source_hits_at_k", {})
                        .get(str(cutoff), {})
                        .get(source, False)
                        for row in predictions
                    ),
                    total,
                ),
                6,
            )
            for source in CANDIDATE_SOURCE_ORDER
        }
    for k in (3, 5, output_k):
        metrics[f"hit_at_{k}"] = ratio(
            sum(row["expected_assignee"] in row["ranked_candidates"][:k] for row in predictions), total
        )
    metrics["mrr"] = ratio(
        sum(1.0 / row["rank"] for row in predictions if row["rank"] is not None), total
    )
    metrics["base_mrr"] = ratio(
        sum(1.0 / row["base_rank"] for row in predictions if row["base_rank"] is not None), total
    )
    metrics["top1_delta_vs_v2_base"] = metrics["top1_accuracy"] - metrics["base_top1_accuracy"]
    metrics["mrr_delta_vs_v2_base"] = metrics["mrr"] - metrics["base_mrr"]
    metrics["macro_f1_delta_vs_v2_base"] = metrics["macro_f1"] - metrics["base_macro_f1"]
    candidate_hits = sum(row["candidate_pool_recall"] for row in predictions)
    ranking_misses = sum(
        row["candidate_pool_recall"] and not row["top1_correct"] for row in predictions
    )
    base_ranking_misses = sum(
        row["candidate_pool_recall"] and not row["base_top1_correct"]
        for row in predictions
    )
    metrics["candidate_miss_rate"] = ratio(total - candidate_hits, total)
    metrics["ranking_miss_rate"] = ratio(ranking_misses, total)
    metrics["base_ranking_miss_rate"] = ratio(base_ranking_misses, total)
    metrics["ranking_miss_relative_reduction_vs_v2_base"] = ratio(
        base_ranking_misses - ranking_misses, base_ranking_misses
    )
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def candidate_source_recall_report(
    windows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "candidate_source_recall_and_miss_audit_v1",
        "intended_use": "development_only_not_sealed_holdout_selection",
        "targets": {
            "known_owner_candidate_recall_at_30": 0.95,
            "overall_candidate_recall_at_30_floor": 0.8295,
        },
        "windows": {
            name: {
                "overall": {
                    key: value
                    for key, value in result["metrics"].items()
                    if key.startswith("candidate_recall")
                    or key in {"rows", "candidate_miss_rate"}
                },
                "known_owner": {
                    key: value
                    for key, value in result["known_owner_metrics"].items()
                    if key.startswith("candidate_recall")
                    or key in {"rows", "candidate_miss_rate"}
                },
                "candidate_miss_clusters": prediction_error_taxonomy(
                    result["predictions"]
                )["candidate_miss_clusters"],
            }
            for name, result in windows.items()
        },
    }


def candidate_source_ablation_report(
    windows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "candidate_evidence_support_ablation_v1",
        "intended_use": "development_only_not_sealed_holdout_selection",
        "interpretation": (
            "Conservative evidence-support ablation. It removes one source or "
            "source family from the observed candidate evidence without filling "
            "the freed quota, so it does not overstate replacement-source gains."
        ),
        "windows": {
            name: {
                "overall": candidate_evidence_support_ablation(
                    result["predictions"]
                ),
                "known_owner": candidate_evidence_support_ablation(
                    [row for row in result["predictions"] if row["known_owner"]]
                ),
            }
            for name, result in windows.items()
        },
    }


def candidate_evidence_support_ablation(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(predictions), "cutoffs": {}}
    families = sorted(set(SOURCE_FAMILIES.values()))
    for cutoff in (10, 30, 50):
        baseline_hits = 0
        expected_evidence: list[tuple[bool, set[str]]] = []
        for row in predictions:
            expected = str(row.get("expected_assignee") or "")
            pool = list(row.get("candidate_pool") or [])
            hit = expected in pool[:cutoff]
            if hit:
                baseline_hits += 1
                sources = set(
                    row.get("candidate_sources", {}).get(expected, [])
                )
            else:
                sources = set()
            expected_evidence.append((hit, sources))
        baseline_recall = ratio(baseline_hits, len(predictions))
        source_rows = {}
        for source in CANDIDATE_SOURCE_ORDER:
            retained = sum(
                hit and (source not in sources or bool(sources - {source}))
                for hit, sources in expected_evidence
            )
            supported = sum(
                hit and source in sources for hit, sources in expected_evidence
            )
            unique = sum(
                hit and sources == {source} for hit, sources in expected_evidence
            )
            source_rows[source] = {
                "supported_hit_rows": supported,
                "unique_support_hit_rows": unique,
                "recall_without_source_lower_bound": round(
                    ratio(retained, len(predictions)), 6
                ),
                "recall_loss_upper_bound": round(
                    baseline_recall - ratio(retained, len(predictions)), 6
                ),
            }
        family_rows = {}
        for family in families:
            retained = 0
            supported = 0
            unique = 0
            for hit, sources in expected_evidence:
                source_families = {SOURCE_FAMILIES[source] for source in sources}
                supported += hit and family in source_families
                unique += hit and source_families == {family}
                retained += hit and (
                    family not in source_families
                    or bool(source_families - {family})
                )
            family_rows[family] = {
                "supported_hit_rows": supported,
                "unique_support_hit_rows": unique,
                "recall_without_family_lower_bound": round(
                    ratio(retained, len(predictions)), 6
                ),
                "recall_loss_upper_bound": round(
                    baseline_recall - ratio(retained, len(predictions)), 6
                ),
            }
        result["cutoffs"][str(cutoff)] = {
            "baseline_recall": round(baseline_recall, 6),
            "by_source": source_rows,
            "by_source_family": family_rows,
        }
    return result


def prediction_error_taxonomy(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    miss_components: Counter[str] = Counter()
    miss_frequency: Counter[str] = Counter()
    miss_months: Counter[str] = Counter()
    missing_data: Counter[str] = Counter()
    for row in predictions:
        if not row.get("known_owner"):
            outcome = "unseen_owner"
        elif not row.get("candidate_pool_recall"):
            outcome = "candidate_miss"
        elif row.get("top1_correct"):
            outcome = "correct_top1"
        else:
            outcome = "ranking_miss"
        outcomes[outcome] += 1
        if outcome != "candidate_miss":
            continue
        miss_components[str(row.get("component") or "unknown")] += 1
        history_count = int(row.get("expected_owner_history_count") or 0)
        bucket = (
            "1-4"
            if history_count < 5
            else "5-19"
            if history_count < 20
            else "20-99"
            if history_count < 100
            else "100+"
        )
        miss_frequency[bucket] += 1
        timestamp = str(row.get("created_at") or "")
        miss_months[timestamp[:7] if len(timestamp) >= 7 else "missing"] += 1
        if row.get("description_missing"):
            missing_data["description_missing"] += 1
        if row.get("routing_safe_file_paths_missing"):
            missing_data["routing_safe_file_paths_missing"] += 1
        if str(row.get("component") or "unknown") == "unknown":
            missing_data["component_missing"] += 1
    return {
        "rows": len(predictions),
        "outcomes": dict(sorted(outcomes.items())),
        "candidate_miss_clusters": {
            "component_top30": dict(miss_components.most_common(30)),
            "owner_history_frequency": dict(sorted(miss_frequency.items())),
            "query_month": dict(sorted(miss_months.items())),
            "missing_data": dict(sorted(missing_data.items())),
        },
    }


def v3_ranker_development_gate(
    windows: dict[str, dict[str, Any]],
    *,
    minimum_known_candidate_recall_at_30: float = 0.95,
    minimum_overall_candidate_recall_at_30: float = 0.8295,
    minimum_known_top1: float = 0.50,
    minimum_ranking_miss_reduction: float = 0.10,
) -> dict[str, Any]:
    checks: dict[str, dict[str, bool]] = {}
    observed: dict[str, Any] = {}
    for name, result in windows.items():
        overall = result["metrics"]
        known = result["known_owner_metrics"]
        checks[name] = {
            "known_candidate_recall_at_30": known["candidate_recall_at_30"]
            >= minimum_known_candidate_recall_at_30,
            "overall_candidate_recall_at_30": overall["candidate_recall_at_30"]
            >= minimum_overall_candidate_recall_at_30,
            "known_top1_target": known["top1_accuracy"] >= minimum_known_top1,
            "top1_improves_v2_base": known["top1_accuracy"]
            > known["base_top1_accuracy"],
            "macro_f1_non_decreasing": known["macro_f1"]
            >= known["base_macro_f1"],
            "ranking_miss_reduction": known[
                "ranking_miss_relative_reduction_vs_v2_base"
            ]
            >= minimum_ranking_miss_reduction,
        }
        observed[name] = {
            "overall": overall,
            "known_owner": known,
        }
    passed = bool(checks and all(all(values.values()) for values in checks.values()))
    return {
        "passed": passed,
        "requires_every_development_window": True,
        "checks": checks,
        "targets": {
            "minimum_known_candidate_recall_at_30": (
                minimum_known_candidate_recall_at_30
            ),
            "minimum_overall_candidate_recall_at_30": (
                minimum_overall_candidate_recall_at_30
            ),
            "minimum_known_top1": minimum_known_top1,
            "minimum_ranking_miss_reduction": minimum_ranking_miss_reduction,
        },
        "observed": observed,
        "blocker": (
            "new_untouched_holdout_required"
            if passed
            else "v3_multi_window_ranker_targets_not_met"
        ),
    }


def promotion_decision(
    result: dict[str, Any],
    current: dict[str, float],
    holdout: dict[str, Any],
    new_holdout: bool,
) -> bool:
    metrics = result["metrics"]
    validation_passed = bool(
        metrics["top1_accuracy"] - current.get("top1_accuracy", 0.0) >= 0.01
        and metrics["mrr"] > current.get("mrr", 0.0)
        and metrics["macro_f1"] >= current.get("macro_f1", 0.0)
        and metrics["candidate_recall_at_pool"] >= 0.90
    )
    if not new_holdout:
        return validation_passed
    known = holdout["known_owner_metrics"]
    holdout_passed = bool(
        known["top1_delta_vs_v2_base"] >= 0.01
        and known["mrr_delta_vs_v2_base"] > 0.0
        and known["macro_f1_delta_vs_v2_base"] >= 0.0
        and known["candidate_recall_at_pool"] >= 0.90
    )
    return validation_passed and holdout_passed


def promotion_gate_payload(
    result: dict[str, Any],
    current: dict[str, float],
    holdout: dict[str, Any],
    new_holdout: bool,
    promote: bool,
    development_gate: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "passed": promote,
        "requirements": {
            "every_development_window_required": True,
            "known_owner_candidate_recall_at_30_min": 0.95,
            "overall_candidate_recall_at_30_min": 0.8295,
            "known_owner_top1_min": 0.50,
            "top1_must_improve_v2_base": True,
            "macro_f1_must_not_decrease": True,
            "ranking_miss_relative_reduction_min": 0.10,
            "new_holdout_required_for_deployment": True,
        },
        "observed": {
            "development_window_gate": development_gate,
            "top1_delta_vs_current_hybrid": round(metrics["top1_accuracy"] - current.get("top1_accuracy", 0.0), 6),
            "mrr_delta_vs_current_hybrid": round(metrics["mrr"] - current.get("mrr", 0.0), 6),
            "macro_f1_delta_vs_current_hybrid": round(metrics["macro_f1"] - current.get("macro_f1", 0.0), 6),
            "validation_candidate_recall_at_30": metrics[
                "candidate_recall_at_30"
            ],
        },
        "new_holdout_evaluated": new_holdout,
        "new_holdout_known_owner_observed": holdout["known_owner_metrics"] if new_holdout else {},
        "deployment_blocker": (
            "explicit_operator_approval_and_confidence_recalibration_required"
            if promote and new_holdout
            else "new_untouched_temporal_holdout_required"
            if not new_holdout
            else "new_holdout_promotion_gate_failed"
        ),
    }


def export_artifact(
    pipeline: Pipeline,
    selected_name: str,
    args: argparse.Namespace,
    validation_result: dict[str, Any],
    fold_audit: list[dict[str, Any]],
    promote: bool,
) -> dict[str, Any]:
    artifact = {
        "schema_version": 1,
        "artifact_type": "assignee_candidate_ranker",
        "name": args.artifact_name or f"candidate_{selected_name}_ltr_v3",
        "deployment_status": "research_only",
        "validation_candidate": promote,
        "feature_names": list(pipeline.feature_names_),
        "model_file": "candidate_ltr_model.joblib",
        "parameters": {
            "candidate_pool_size": args.candidate_pool_size,
            "half_life_days": args.half_life_days,
            "smoothing_alpha": args.smoothing_alpha,
            "description_cleaner": "section_weighted_v1",
            "embedding_backend": args.embedding_backend,
            "candidate_generator_version": "v3",
            "file_path_feature_policy": "report_time_provenance_required",
            "selection_objective": args.selection_objective,
            "training_objective": str(
                getattr(pipeline, "training_objective_", "unknown")
            ),
        },
        "training_protocol": fold_audit,
        "validation_metrics": validation_result["metrics"],
        "approval_note": "Requires a new untouched temporal holdout and explicit approval before runtime use.",
    }
    if selected_name in {
        "logistic_regression",
        "pairwise_logistic_hard_negative",
    }:
        scaler = pipeline.named_steps["scale"]
        model = pipeline.named_steps["model"]
        artifact["portable_logistic"] = {
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "coefficients": [float(value) for value in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
        }
    if selected_name == "pairwise_logistic_hard_negative":
        artifact["pairwise_training_audit"] = getattr(
            pipeline, "pairwise_training_audit_", {}
        )
    return artifact


def read_current_hybrid_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("baselines", {}).get("current_assignee_triager", data)
    return {
        key: float(metrics.get(key, 0.0))
        for key in ("top1_accuracy", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "macro_f1")
    }


def read_label_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Assignee label map must be a JSON object: {path}")
    return {
        normalize_assignee(source): normalize_assignee(target)
        for source, target in value.items()
        if normalize_assignee(source) and normalize_assignee(target)
    }


def apply_label_map(rows: list[dict[str, Any]], label_map: dict[str, str]) -> list[dict[str, Any]]:
    mapped = []
    for row in rows:
        value = dict(row)
        assignee = normalize_assignee(value.get("assignee"))
        value["assignee"] = label_map.get(assignee, assignee)
        mapped.append(value)
    return mapped


def canonicalize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        if source.get("assigned_to") or source.get("summary") or source.get("creation_time"):
            bug_id = str(source.get("id") or "").strip()
            declared_label_source = str(
                source.get("label_availability_source") or ""
            ).strip()
            assignment_event_time = str(
                source.get("assignment_label_available_at")
                or source.get("assignment_changed_at")
                or source.get("assigned_at")
                or ""
            ).strip()
            generic_label_available_at = str(
                source.get("label_available_at") or ""
            ).strip()
            explicit_label_available_at = (
                ""
                if declared_label_source == "last_change_time_proxy"
                else assignment_event_time or generic_label_available_at
            )
            last_change_time = str(source.get("last_change_time") or "").strip()
            row = {
                "ticket_id": f"bmo_{bug_id}" if bug_id else "",
                "title": str(source.get("summary") or "").strip(),
                "description": str(source.get("description") or "").strip(),
                "product": normalize_component(source.get("product")),
                "component": normalize_component(source.get("component")),
                "priority": str(source.get("priority") or "P3").strip(),
                "severity": str(source.get("severity") or "").strip(),
                "assignee": normalize_assignee(source.get("assigned_to")),
                "created_at": str(source.get("creation_time") or "").strip(),
                "last_change_time": last_change_time,
                "label_available_at": (
                    assignment_event_time
                    or generic_label_available_at
                    or last_change_time
                ),
                "label_availability_source": (
                    "last_change_time_proxy"
                    if declared_label_source == "last_change_time_proxy"
                    else "explicit_assignment_event_time"
                    if explicit_label_available_at
                    else "last_change_time_proxy"
                    if last_change_time
                    else "missing"
                ),
                "duplicate_of": f"bmo_{source['dupe_of']}" if source.get("dupe_of") else "",
                "duplicate_family": f"bmo_{source['dupe_of']}" if source.get("dupe_of") else "",
                "owner_event_id": str(source.get("owner_event_id") or "").strip(),
                "reporter": normalize_assignee(source.get("creator")),
                "file_paths": routing_safe_file_paths(source),
                "file_paths_provenance": str(
                    source.get("file_paths_provenance") or ""
                ).strip(),
                "source": "bugzilla_mozilla",
            }
        else:
            row = dict(source)
            row["assignee"] = normalize_assignee(row.get("assignee"))
        if (
            not row.get("ticket_id")
            or not row.get("title")
            or not row.get("assignee")
            or parse_timestamp(row_timestamp(row)) is None
            or GENERIC_ASSIGNEE_RE.search(str(row.get("assignee")))
        ):
            continue
        result.append(row)
    return result


def exclude_cross_split_overlap(
    rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    *,
    strict_duplicate_families: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    history_ids = {str(row.get("ticket_id") or "") for row in history_rows}
    history_content = {content_key(row) for row in history_rows}
    history_duplicate_references = {
        reference
        for row in history_rows
        for reference in duplicate_family_references(row)
    }
    seen_ids = set(history_ids)
    seen_content = set(history_content)
    seen_duplicate_references = set(history_duplicate_references)
    output = []
    removed = 0
    for row in rows:
        ticket_id = str(row.get("ticket_id") or "")
        content = content_key(row)
        duplicate_references = duplicate_family_references(row)
        if (
            ticket_id in seen_ids
            or content in seen_content
            or (
                strict_duplicate_families
                and bool(duplicate_references & seen_duplicate_references)
            )
        ):
            removed += 1
            continue
        seen_ids.add(ticket_id)
        seen_content.add(content)
        if strict_duplicate_families:
            seen_duplicate_references.update(duplicate_references)
        output.append(row)
    return output, removed


def content_key(row: dict[str, Any]) -> str:
    return " ".join(tokens(f"{row.get('title', '')} {row.get('description', '')}"))


def duplicate_family_references(row: dict[str, Any]) -> set[str]:
    values = {
        normalize_ticket_reference(row.get("ticket_id")),
        normalize_ticket_reference(row.get("duplicate_family")),
        normalize_ticket_reference(row.get("duplicate_family_id")),
        normalize_ticket_reference(row.get("duplicate_of")),
        normalize_ticket_reference(row.get("dupe_of")),
    }
    return {value for value in values if value}


def normalize_ticket_reference(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("bmo_"):
        return text
    return f"bmo_{text}" if text.isdigit() else text


def clean_description(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    selected = []
    seen = set()
    stack_lines = 0
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or BOILERPLATE_RE.match(line):
            continue
        if STACK_LINE_RE.match(line):
            stack_lines += 1
            if stack_lines > 4:
                continue
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(line)
        if sum(len(item) + 1 for item in selected) >= max_chars:
            break
    return " ".join(selected)[:max_chars]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("title") or row.get("summary") or "").strip()
    description = clean_description(row.get("description") or row.get("body") or "")
    product = normalize_component(row.get("product"))
    component = normalize_component(row.get("component"))
    metadata = " ".join(
        str(row.get(field) or "") for field in ("severity", "priority", "bug_type")
    )
    # Title is intentionally weighted more heavily because raw first comments
    # reduced held-out BM25 quality in the existing description ablation.
    text = " ".join([title, title, title, description, product, component, metadata]).strip()
    return {
        **row,
        "assignee": normalize_assignee(row.get("assignee")),
        "product": product,
        "component": component,
        "reporter": normalize_assignee(row.get("reporter") or row.get("creator")),
        "file_paths": routing_safe_file_paths(row),
        "title": title,
        "description": description,
        "text": text,
        "timestamp": parse_timestamp(row_timestamp(row)),
    }


def smoothed_share(count: float, total: float, prior: float, alpha: float) -> float:
    return ratio(count + alpha * prior, total + alpha)


def tokens(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(value.lower()) if len(token) > 1 and token not in STOPWORDS]


def pair_key(product: str, component: str) -> str:
    return f"{product}::{component}"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def component_tokens(value: Any) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_component(value)) if len(token) >= 3]


def normalized_file_paths(row: Any) -> list[str]:
    if not isinstance(row, dict):
        values = row if isinstance(row, list) else [row]
    else:
        values = []
        for key in (
            "file_paths",
            "files",
            "file_path",
            "path",
            "paths",
            "module_paths",
        ):
            value = row.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value:
                values.append(value)
    normalized = {
        str(value).strip().lower().replace("\\", "/").lstrip("./")
        for value in values
        if str(value).strip()
    }
    return sorted(path for path in normalized if path)


def routing_safe_file_paths(row: dict[str, Any]) -> list[str]:
    report_time_paths = row.get("file_paths_at_report_time")
    if report_time_paths:
        return normalized_file_paths(report_time_paths)
    provenance = str(row.get("file_paths_provenance") or "").strip().lower()
    if provenance not in {
        "report_time",
        "reporter_supplied",
        "stack_trace",
        "reproduction_metadata",
        "ownership_snapshot",
    }:
        return []
    return normalized_file_paths(row)


def file_history_keys(value: Any) -> set[str]:
    paths = normalized_file_paths(value)
    keys: set[str] = set()
    for path in paths:
        parts = [part for part in path.split("/") if part]
        keys.add(f"file:{path}")
        if len(parts) >= 2:
            keys.add(f"tail:{'/'.join(parts[-2:])}")
            keys.add(f"dir:{'/'.join(parts[:-1])}")
    return keys


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rank_of(expected: str, candidates: list[str]) -> int | None:
    try:
        return candidates.index(expected) + 1
    except ValueError:
        return None


def safe_macro_f1(expected: list[str], predicted: list[str]) -> float:
    if not expected:
        return 0.0
    labels = sorted(set(expected) | set(predicted))
    return float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0))


def row_timestamp(row: dict[str, Any]) -> str:
    for key in ("created_at", "creation_time", "reported_at", "timestamp"):
        if row.get(key):
            return str(row[key])
    return ""


def parse_timestamp(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def row_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    timestamp = parse_timestamp(row_timestamp(row))
    return (timestamp if timestamp is not None else float("inf"), str(row.get("ticket_id") or ""))


def timestamp_string(row: dict[str, Any]) -> str:
    return row_timestamp(row)


def ensure_temporal_order(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]
) -> None:
    groups = [("train", train_rows), ("validation", validation_rows), ("test", test_rows)]
    for name, rows in groups:
        if not rows:
            raise SystemExit(f"{name} split is empty")
        if any(parse_timestamp(row_timestamp(row)) is None for row in rows):
            raise SystemExit(f"{name} contains missing or invalid timestamps")
    if row_sort_key(train_rows[-1]) > row_sort_key(validation_rows[0]):
        raise SystemExit("Temporal leakage: train overlaps validation")
    if row_sort_key(validation_rows[-1]) > row_sort_key(test_rows[0]):
        raise SystemExit("Temporal leakage: validation overlaps test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    validation = report["validation"]
    test = report["test_exploratory"]
    gate = report["promotion_gate"]
    lines = [
        "# Candidate LTR Experiment",
        "",
        f"- Deployment status: `{report['deployment_status']}`",
        f"- Validation candidate: `{report['validation_candidate']}`",
        f"- Selected model: `{report['model']['selected']}`",
        f"- Embedding backend: `{report['model']['embedding_backend']}`",
        f"- Promotion gate passed: `{gate['passed']}`",
        "",
        "## Validation",
        "",
        "| Metric | V2 base | Logistic LTR | Delta |",
        "|---|---:|---:|---:|",
        f"| Top-1 | {validation['base_top1_accuracy']} | {validation['top1_accuracy']} | {validation['top1_delta_vs_v2_base']} |",
        f"| MRR | {validation['base_mrr']} | {validation['mrr']} | {validation['mrr_delta_vs_v2_base']} |",
        f"| Macro-F1 | {validation['base_macro_f1']} | {validation['macro_f1']} | {validation['macro_f1_delta_vs_v2_base']} |",
        f"| Candidate recall | - | {validation['candidate_recall_at_pool']} | - |",
        "",
        "## Existing Test (Exploratory Only)",
        "",
        "The repository's prior frozen test has already been inspected, so these values are not a new final estimate.",
        "",
        f"- Top-1: `{test['top1_accuracy']}`",
        f"- Hit@3: `{test['hit_at_3']}`",
        f"- Hit@5: `{test['hit_at_5']}`",
        f"- MRR: `{test['mrr']}`",
        f"- Macro-F1: `{test['macro_f1']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
