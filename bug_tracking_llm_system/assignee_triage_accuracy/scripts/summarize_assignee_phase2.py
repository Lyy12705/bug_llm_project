from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from assignee_phase1_common import DEFAULT_PHASE1_ROOT, write_json
from assignee_phase2_common import DEFAULT_PHASE2_ROOT


DATASET = "bmo_paper_2024_3k"
ENRICHED_DATASET = "bmo_paper_2024_3k_description_enriched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 assignee-triage reliability artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE2_ROOT / "reports")
    args = parser.parse_args()

    paths = artifact_paths()
    phase1 = read_json(paths["phase1_summary"])
    closed_multi = read_json(paths["closed_multi"])
    open_multi = read_json(paths["open_multi"])
    description_audit = read_json(paths["description_audit"])
    ablation_summary = read_json(paths["ablation_summary"])
    ablation_metrics = read_csv(paths["ablation_metrics"])
    ablation_buckets = read_csv(paths["ablation_buckets"])

    analysis = build_analysis(
        phase1=phase1,
        closed_multi=closed_multi,
        open_multi=open_multi,
        description_audit=description_audit,
        ablation_summary=ablation_summary,
        ablation_metrics=ablation_metrics,
        ablation_buckets=ablation_buckets,
        paths=paths,
    )
    benchmark = future_llm_benchmark_definition(paths)
    payload = {
        "experiment_name": "assignee_triage_phase2_temporal_description_summary",
        "dataset": DATASET,
        "description_enriched_dataset": ENRICHED_DATASET,
        "analysis": analysis,
        "future_llm_fair_benchmark": benchmark,
        "artifact_paths": {key: str(value) for key, value in paths.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "phase2_assignee_triage_summary.json", payload)
    write_summary_markdown(args.output_dir / "phase2_assignee_triage_summary.md", payload)
    write_benchmark_markdown(args.output_dir / "future_llm_fair_benchmark.md", benchmark)
    print(json.dumps(analysis["final_judgment"], ensure_ascii=False, indent=2, sort_keys=True))


def artifact_paths() -> dict[str, Path]:
    phase1_root = DEFAULT_PHASE1_ROOT / "reports" / DATASET
    phase2_root = DEFAULT_PHASE2_ROOT / "reports"
    return {
        "phase1_summary": phase1_root / "phase1_reliability_summary.json",
        "closed_multi": phase2_root / DATASET / "multi_window_temporal" / "multi_window_metrics.json",
        "closed_multi_summary_csv": phase2_root / DATASET / "multi_window_temporal" / "multi_window_summary.csv",
        "open_multi": phase2_root / DATASET / "open_world_multi_window" / "open_world_multi_window_metrics.json",
        "open_multi_summary_csv": phase2_root / DATASET / "open_world_multi_window" / "open_world_multi_window_summary.csv",
        "description_audit": phase2_root / ENRICHED_DATASET / "description_enrichment_audit.json",
        "description_root_cause": phase2_root / ENRICHED_DATASET / "description_missing_root_cause.md",
        "ablation_summary": phase2_root / ENRICHED_DATASET / "description_ablation" / "description_ablation_summary.json",
        "ablation_metrics": phase2_root / ENRICHED_DATASET / "description_ablation" / "description_ablation_metrics.csv",
        "ablation_buckets": phase2_root / ENRICHED_DATASET / "description_ablation" / "description_ablation_frequency_buckets.csv",
        "ablation_predictions": phase2_root / ENRICHED_DATASET / "description_ablation" / "description_ablation_predictions.csv",
    }


def build_analysis(
    *,
    phase1: dict[str, Any],
    closed_multi: dict[str, Any],
    open_multi: dict[str, Any],
    description_audit: dict[str, Any],
    ablation_summary: dict[str, Any],
    ablation_metrics: list[dict[str, str]],
    ablation_buckets: list[dict[str, str]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    current_aggregate = closed_multi["aggregate"]["current_assignee_triager"]
    bm25_aggregate = closed_multi["aggregate"]["bm25_text_knn"]
    phase1_by_name = {row["experiment_name"]: row for row in phase1["experiments"]}
    closed_original = phase1_by_name["closed_set_current_repro"]
    open_original = phase1_by_name["prefilter_open_world_eval"]
    current_window_details = current_window_analysis(closed_multi)
    ablation_effects = ablation_summary["headline"]["description_effects"]
    current_title_metadata = metric_lookup(ablation_metrics, "current_assignee_triager", "title_plus_metadata")
    current_full = metric_lookup(ablation_metrics, "current_assignee_triager", "title_plus_description_plus_metadata")
    bm25_title_metadata = metric_lookup(ablation_metrics, "bm25_text_knn", "title_plus_metadata")
    bm25_full = metric_lookup(ablation_metrics, "bm25_text_knn", "title_plus_description_plus_metadata")
    long_tail = long_tail_analysis(ablation_buckets)

    closed_delta_original_vs_multi = round(
        float(closed_original["top1_accuracy"]) - float(current_aggregate["mean"]), 6
    )
    open_delta_original_vs_multi = round(
        float(open_original["top1_accuracy"]) - float(open_multi["aggregate"]["top1_before_abstention"]["mean"]), 6
    )

    final_judgment = {
        "A_current_assignee_triager_temporal_stability": {
            "judgment": "moderately_stable_closed_set_but_not_uniformly_best",
            "current_closed_set_mean_top1": current_aggregate["mean"],
            "std": current_aggregate["std"],
            "min": current_aggregate["min"],
            "max": current_aggregate["max"],
            "range": round(current_aggregate["max"] - current_aggregate["min"], 6),
            "strict_best_windows": current_window_details["strict_best_windows"],
            "tied_best_windows": current_window_details["tied_best_windows"],
            "lost_windows": current_window_details["lost_windows"],
        },
        "B_most_credible_deterministic_top1": {
            "closed_set_temporal_mean": current_aggregate["mean"],
            "open_world_temporal_mean": open_multi["aggregate"]["top1_before_abstention"]["mean"],
            "recommended_statement": "Report the multi-window closed-set mean and the more realistic open-world multi-window mean.",
        },
        "C_is_0_765625_optimistic": {
            "judgment": "yes_mildly_for_closed_set_strongly_for_open_world",
            "delta_vs_closed_multi_window_mean": closed_delta_original_vs_multi,
            "delta_vs_open_world_multi_window_mean": round(
                float(closed_original["top1_accuracy"]) - float(open_multi["aggregate"]["top1_before_abstention"]["mean"]),
                6,
            ),
        },
        "D_open_world_0_517857_cross_window_stability": {
            "judgment": "single_split_is_slightly_optimistic_relative_to_phase2_windows",
            "phase1_single_split_top1": open_original["top1_accuracy"],
            "multi_window_mean": open_multi["aggregate"]["top1_before_abstention"]["mean"],
            "multi_window_std": open_multi["aggregate"]["top1_before_abstention"]["std"],
            "multi_window_min": open_multi["aggregate"]["top1_before_abstention"]["min"],
            "multi_window_max": open_multi["aggregate"]["top1_before_abstention"]["max"],
            "delta_single_split_minus_multi_mean": open_delta_original_vs_multi,
        },
        "E_description_effect_on_bm25": {
            "title_plus_metadata_top1": float(bm25_title_metadata["top1_accuracy"]),
            "full_top1": float(bm25_full["top1_accuracy"]),
            "full_minus_title_plus_metadata_top1": ablation_effects["bm25_text_knn"]["all_minus_title_plus_metadata_top1"],
            "interpretation": "Raw first-comment descriptions hurt BM25 when metadata is present, although title+description improves over title-only in no-metadata stress tests.",
        },
        "F_description_effect_on_current_hybrid": {
            "title_plus_metadata_top1": float(current_title_metadata["top1_accuracy"]),
            "full_top1": float(current_full["top1_accuracy"]),
            "full_minus_title_plus_metadata_top1": ablation_effects["current_assignee_triager"]["all_minus_title_plus_metadata_top1"],
            "interpretation": "Description enrichment does not improve current hybrid as-is; metadata remains the dominant signal.",
        },
        "G_long_tail": long_tail,
        "H_best_input_for_llm": {
            "recommendation": "Use title + product/component/severity/priority + sanitized/truncated first-comment description as the LLM primary input, but keep title+metadata as a mandatory ablation/control.",
            "reason": "Deterministic methods peak at title+metadata, while LLM methods need richer text to test the research question; raw descriptions should be cleaned/truncated because they add noise for BM25/current hybrid.",
        },
        "I_ready_for_literature_llm_comparison": {
            "judgment": "ready_for_controlled_same_dataset_comparison_not_direct_literature_claim",
            "reason": "The benchmark now has versioned descriptions, temporal windows, rosters, predictions, and open-world diagnostics. It still lacks strict assignment-history semantics and is not directly comparable to literature datasets unless splits/candidates/input fields are matched.",
        },
        "J_next_step": {
            "recommendation": "confidence_calibration_first",
            "reason": "Open-world performance is the deployment bottleneck; validation-selected abstention thresholds produce low coverage and need calibration before trusting auto-assignment or LLM reranking. Literature LLM reproduction should follow on the frozen Phase 2 benchmark.",
        },
    }

    return {
        "closed_set_multi_window": {
            "current_assignee_triager": current_aggregate,
            "bm25_text_knn": bm25_aggregate,
            "current_vs_bm25_gap_by_window": current_window_details["current_minus_bm25_by_window"],
            "current_vs_bm25_gap_mean": current_window_details["current_minus_bm25_mean"],
            "head_tail_gap_current_mean": current_window_details["head_tail_gap_mean"],
        },
        "open_world_multi_window": open_multi["aggregate"],
        "description_audit": {
            "total_rows": description_audit["total_rows"],
            "nonempty_ratio": description_audit["description_nonempty_ratio"],
            "missing_ratio": description_audit["description_missing_ratio"],
            "mean_length": description_audit["description_length_mean"],
            "median_length": description_audit["description_length_median"],
            "mean_token_length": description_audit["description_token_length_mean"],
            "median_token_length": description_audit["description_token_length_median"],
            "html_content_ratio": description_audit["html_content_ratio"],
            "code_or_log_presence_ratio": description_audit["code_or_log_presence_ratio"],
            "fetch_error_count": description_audit["fetch_error_count"],
        },
        "description_ablation": {
            "current_title_plus_metadata": current_title_metadata,
            "current_full": current_full,
            "bm25_title_plus_metadata": bm25_title_metadata,
            "bm25_full": bm25_full,
            "effects": ablation_effects,
        },
        "compatibility": {
            "production_triager_modified": False,
            "phase1_artifacts_overwritten": False,
            "closed_set_benchmark_overwritten": False,
            "new_dataset_version": ENRICHED_DATASET,
            "phase2_outputs_root": str(DEFAULT_PHASE2_ROOT),
        },
        "final_judgment": final_judgment,
        "primary_artifacts": {
            "closed_multi_window": str(paths["closed_multi"]),
            "open_multi_window": str(paths["open_multi"]),
            "description_audit": str(paths["description_audit"]),
            "description_ablation": str(paths["ablation_metrics"]),
        },
    }


def current_window_analysis(closed_multi: dict[str, Any]) -> dict[str, Any]:
    strict_best = []
    tied_best = []
    lost = []
    gaps = {}
    head_tail_gaps = []
    for name, window in closed_multi["windows"].items():
        metric_map = {
            baseline: row["top1_accuracy"] for baseline, row in window["metrics"].items()
        }
        current = metric_map["current_assignee_triager"]
        best = max(metric_map.values())
        best_count = sum(1 for value in metric_map.values() if value == best)
        if current == best and best_count == 1:
            strict_best.append(name)
        elif current == best:
            tied_best.append(name)
        else:
            lost.append(name)
        gaps[name] = round(current - metric_map["bm25_text_knn"], 6)
        if window.get("head_tail_gap_current") is not None:
            head_tail_gaps.append(float(window["head_tail_gap_current"]))
    return {
        "strict_best_windows": strict_best,
        "tied_best_windows": tied_best,
        "lost_windows": lost,
        "current_minus_bm25_by_window": gaps,
        "current_minus_bm25_mean": round(statistics.mean(gaps.values()), 6),
        "current_minus_bm25_min": round(min(gaps.values()), 6),
        "current_minus_bm25_max": round(max(gaps.values()), 6),
        "head_tail_gap_mean": round(statistics.mean(head_tail_gaps), 6) if head_tail_gaps else None,
    }


def long_tail_analysis(bucket_rows: list[dict[str, str]]) -> dict[str, Any]:
    current_title_metadata = bucket_lookup(
        bucket_rows, "current_assignee_triager", "title_plus_metadata", "freq_10_19"
    )
    current_full = bucket_lookup(
        bucket_rows, "current_assignee_triager", "title_plus_description_plus_metadata", "freq_10_19"
    )
    bm25_title_metadata = bucket_lookup(bucket_rows, "bm25_text_knn", "title_plus_metadata", "freq_10_19")
    bm25_full = bucket_lookup(
        bucket_rows, "bm25_text_knn", "title_plus_description_plus_metadata", "freq_10_19"
    )
    return {
        "judgment": "not_fixed",
        "closed_set_tail_bucket": "freq_10_19",
        "current_title_plus_metadata_tail_top1": float(current_title_metadata["top1_accuracy"]),
        "current_full_tail_top1": float(current_full["top1_accuracy"]),
        "current_full_minus_title_plus_metadata_tail_top1": round(
            float(current_full["top1_accuracy"]) - float(current_title_metadata["top1_accuracy"]), 6
        ),
        "bm25_title_plus_metadata_tail_top1": float(bm25_title_metadata["top1_accuracy"]),
        "bm25_full_tail_top1": float(bm25_full["top1_accuracy"]),
        "bm25_full_minus_title_plus_metadata_tail_top1": round(
            float(bm25_full["top1_accuracy"]) - float(bm25_title_metadata["top1_accuracy"]), 6
        ),
        "reason": "Description-enriched full input reduces tail Top-1 for both current hybrid and BM25 in the closed-set tail bucket; Phase 1 open-world low-frequency seen Top-1 remained 0.228571.",
    }


def future_llm_benchmark_definition(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "dataset_version": ENRICHED_DATASET,
        "dataset_role": "Primary controlled dataset for future instruction-tuned LLM comparison; keep original BMO 3k as no-description baseline.",
        "temporal_windows": "Use the 8 Phase 2 windows for stability, plus the original frozen split only for backward comparability.",
        "train_validation_test": "Temporal only. Train history and roster are rebuilt inside each window. Validation is for thresholds, calibration, prompts, and checkpoint selection. Test is frozen.",
        "candidate_roster": {
            "closed_set": "Assignees with at least 10 train tickets in the same window.",
            "open_set": "All train assignees; low-frequency and unseen test labels remain visible for abstention/manual_triage analysis.",
            "rule": "No candidate may be introduced from validation/test labels.",
        },
        "closed_set_policy": "Primary accuracy leaderboard uses train-frequency roster only, with excluded low-frequency/unseen tickets reported separately.",
        "open_set_policy": "Evaluate all test tickets. Unseen assignees are not valid generated labels; success requires abstention/manual_triage under a validation-selected threshold.",
        "low_frequency_policy": "Report freq_1, freq_2_4, freq_5_9, freq_10_19, and freq_20_plus buckets where applicable.",
        "description_source": "first public Bugzilla comment, stored as description with description_source=first_comment, fetch timestamp, missing flag, and audit metrics.",
        "metadata_fields": ["product", "component", "severity", "priority", "status"],
        "input_precondition": "Tickets must already have passed the upstream duplicate-detection feature. Assignee triage does not classify duplicates; dataset preparation only checks exact cross-split overlap as a leakage guard.",
        "metrics": [
            "Top-1 Accuracy",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR",
            "macro assignee Top-1",
            "bootstrap 95% CI",
            "coverage/selective accuracy/manual-triage recall for open-set",
            "confidence calibration metrics such as ECE/Brier when confidence is produced",
        ],
        "per_ticket_prediction_format": [
            "ticket_id",
            "window_name",
            "model_name",
            "feature_config",
            "candidate_roster_version",
            "expected_assignee",
            "ranked_candidates",
            "candidate_scores_or_logprobs",
            "predicted_assignee",
            "rank",
            "confidence",
            "abstained/manual_triage",
            "reason",
        ],
        "confidence_abstention_policy": "Thresholds must be selected on validation only and then frozen for test.",
        "statistical_testing": "Use paired bootstrap over test tickets and window-level aggregate comparisons; apply correction when many model pairs are tested.",
        "models_to_compare": [
            "component_majority",
            "bm25_text_knn",
            "current_assignee_triager",
            "instruction_tuned_llm",
            "hybrid_current_topk_plus_llm_reranker",
        ],
        "llm_input_controls": [
            "title_plus_metadata",
            "title_plus_description_plus_metadata_sanitized_truncated",
        ],
        "artifact_requirements": {
            "multi_window_metrics": str(paths["closed_multi"]),
            "open_world_multi_window_metrics": str(paths["open_multi"]),
            "description_ablation_predictions": str(paths["ablation_predictions"]),
        },
    }


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    judgment = payload["analysis"]["final_judgment"]
    analysis = payload["analysis"]
    lines = [
        "# Assignee Triaging Phase 2 Summary",
        "",
        "Phase 2 was additive only. The production `current_assignee_triager` ranking logic and Phase 1 artifacts were not modified.",
        "",
        "## Key Results",
        "",
        f"- Closed-set current triager multi-window Top-1 mean/std/min/max: "
        f"`{judgment['A_current_assignee_triager_temporal_stability']['current_closed_set_mean_top1']}` / "
        f"`{judgment['A_current_assignee_triager_temporal_stability']['std']}` / "
        f"`{judgment['A_current_assignee_triager_temporal_stability']['min']}` / "
        f"`{judgment['A_current_assignee_triager_temporal_stability']['max']}`.",
        f"- Open-world current triager multi-window Top-1 mean/std/min/max: "
        f"`{judgment['D_open_world_0_517857_cross_window_stability']['multi_window_mean']}` / "
        f"`{judgment['D_open_world_0_517857_cross_window_stability']['multi_window_std']}` / "
        f"`{judgment['D_open_world_0_517857_cross_window_stability']['multi_window_min']}` / "
        f"`{judgment['D_open_world_0_517857_cross_window_stability']['multi_window_max']}`.",
        f"- Description nonempty ratio after enrichment: "
        f"`{analysis['description_audit']['nonempty_ratio']}`; missing ratio "
        f"`{analysis['description_audit']['missing_ratio']}`.",
        f"- BM25 full description effect vs title+metadata: "
        f"`{judgment['E_description_effect_on_bm25']['full_minus_title_plus_metadata_top1']}` Top-1.",
        f"- Current hybrid full description effect vs title+metadata: "
        f"`{judgment['F_description_effect_on_current_hybrid']['full_minus_title_plus_metadata_top1']}` Top-1.",
        "",
        "## Final Judgments",
        "",
    ]
    for key, value in judgment.items():
        lines.append(f"### {key}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"- {subkey}: `{subvalue}`")
        else:
            lines.append(f"- `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Primary Artifacts",
            "",
        ]
    )
    for key, value in analysis["primary_artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_markdown(path: Path, benchmark: dict[str, Any]) -> None:
    lines = [
        "# Future LLM Fair Benchmark Definition",
        "",
        f"- Dataset version: `{benchmark['dataset_version']}`",
        f"- Dataset role: {benchmark['dataset_role']}",
        f"- Temporal windows: {benchmark['temporal_windows']}",
        f"- Train/validation/test: {benchmark['train_validation_test']}",
        "",
        "## Candidate Policy",
        "",
    ]
    for key, value in benchmark["candidate_roster"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Evaluation Policy",
            "",
            f"- Closed set: {benchmark['closed_set_policy']}",
            f"- Open set: {benchmark['open_set_policy']}",
            f"- Low frequency: {benchmark['low_frequency_policy']}",
            f"- Description source: {benchmark['description_source']}",
            f"- Input precondition: {benchmark['input_precondition']}",
            f"- Confidence/abstention: {benchmark['confidence_abstention_policy']}",
            f"- Statistical testing: {benchmark['statistical_testing']}",
            "",
            "## Models",
            "",
        ]
    )
    for model in benchmark["models_to_compare"]:
        lines.append(f"- `{model}`")
    lines.extend(["", "## Metrics", ""])
    for metric in benchmark["metrics"]:
        lines.append(f"- {metric}")
    lines.extend(["", "## Per-Ticket Prediction Fields", ""])
    for field in benchmark["per_ticket_prediction_format"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## LLM Input Controls", ""])
    for config in benchmark["llm_input_controls"]:
        lines.append(f"- `{config}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_lookup(rows: list[dict[str, str]], baseline: str, feature_config: str) -> dict[str, str]:
    for row in rows:
        if row["baseline"] == baseline and row["feature_config"] == feature_config:
            return row
    raise SystemExit(f"Missing metric row for {baseline}/{feature_config}")


def bucket_lookup(
    rows: list[dict[str, str]], baseline: str, feature_config: str, bucket: str
) -> dict[str, str]:
    for row in rows:
        if (
            row["baseline"] == baseline
            and row["feature_config"] == feature_config
            and row["frequency_bucket"] == bucket
        ):
            return row
    raise SystemExit(f"Missing bucket row for {baseline}/{feature_config}/{bucket}")


if __name__ == "__main__":
    main()
