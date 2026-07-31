from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_eligibility import normalize_owner, normalize_scope  # noqa: E402

from assignee_open_set_common import wilson_interval, write_json  # noqa: E402
from train_assignee_ltr import DEFAULT_DATA_DIR, read_label_map  # noqa: E402
from train_assignee_rolling_open_set import (  # noqa: E402
    RAW_DIR,
    apply_label_availability,
    load_rows,
    prepare_rolling_windows,
    read_label_availability,
)


DEFAULT_WINDOWS = ("2021_q1", "2021_q2", "2021_q3")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a time-safe historical Product+Component experience proxy. "
            "This is exploratory evidence and is not a reviewed Eligible-owner label."
        )
    )
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=int,
        action="append",
        default=[],
        help="Minimum prior assignments in the same Product+Component; repeatable.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help="Policy-selection window to include; defaults to 2021_q1..2021_q3.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl",
    )
    parser.add_argument(
        "--base-raw",
        type=Path,
        default=RAW_DIR / "bmo_public_10k_raw.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl",
    )
    parser.add_argument(
        "--q3-2020",
        type=Path,
        default=RAW_DIR / "bmo_public_future_2020q3_raw.jsonl",
    )
    parser.add_argument(
        "--q4-2020",
        type=Path,
        default=RAW_DIR / "bmo_public_future_2020q4_raw.jsonl",
    )
    parser.add_argument(
        "--q1-2021",
        type=Path,
        default=RAW_DIR / "bmo_public_future_2021q1_raw.jsonl",
    )
    parser.add_argument(
        "--q2-2021",
        type=Path,
        default=RAW_DIR / "bmo_public_future_2021q2_raw.jsonl",
    )
    parser.add_argument(
        "--q3-2021",
        type=Path,
        default=RAW_DIR / "bmo_public_future_2021q3_raw.jsonl",
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    args = parser.parse_args()

    thresholds = sorted(set(args.threshold or [1, 3, 5]))
    if any(value < 1 for value in thresholds):
        raise SystemExit("--threshold must be a positive integer")
    selected_windows = tuple(args.window or DEFAULT_WINDOWS)

    label_map = read_label_map(args.assignee_label_map)
    availability = read_label_availability(args.base_raw)
    base_history = apply_label_availability(
        load_rows(args.train, label_map), availability
    )
    window_specs = [
        ("validation", args.validation),
        ("2020_q3", args.q3_2020),
        ("2020_q4", args.q4_2020),
        ("2021_q1", args.q1_2021),
        ("2021_q2", args.q2_2021),
        ("2021_q3", args.q3_2021),
    ]
    windows = prepare_rolling_windows(
        base_history,
        window_specs,
        label_map,
        availability,
    )
    windows_by_name = {window.name: window for window in windows}
    missing = sorted(set(selected_windows) - set(windows_by_name))
    if missing:
        raise SystemExit(f"unknown windows: {', '.join(missing)}")

    rows_by_window: dict[str, list[dict[str, Any]]] = {}
    experience_by_window: dict[str, Counter[tuple[str, str, str]]] = {}
    for name in selected_windows:
        prediction_path = args.predictions_dir / f"{name}_routing_predictions.jsonl"
        rows_by_window[name] = read_jsonl(prediction_path)
        experience_by_window[name] = product_component_experience(
            windows_by_name[name].history_before
        )

    report = {
        "schema_version": 1,
        "method": "historical_product_component_experience_proxy_v1",
        "formal_eligible_owner_result": False,
        "evaluation_scope": (
            "rows where the frozen Exact-owner Top-5 display policy set "
            "top5_assist_available=true"
        ),
        "proxy_definition": (
            "candidate has at least N assignments in the same normalized "
            "Product+Component before the policy-selection window"
        ),
        "label_availability_policy": (
            "history is filtered by the same time-safe rolling-window preparation"
        ),
        "threshold_results": {},
        "limitations": [
            "historical assignment is not maintainer-reviewed capability",
            "active employment and assignment permissions are not verified",
            "public data uses last_change_time as label-availability proxy",
            "results must be labeled exploratory domain-experience proxy",
        ],
    }
    for threshold in thresholds:
        per_window = {}
        pooled_rows = []
        for name in selected_windows:
            assisted = [
                row
                for row in rows_by_window[name]
                if row.get("top5_assist_available") is True
            ]
            labeled = annotate_proxy(
                assisted,
                experience_by_window[name],
                minimum_assignments=threshold,
            )
            per_window[name] = summarize(labeled)
            pooled_rows.extend(labeled)
        report["threshold_results"][str(threshold)] = {
            "minimum_prior_product_component_assignments": threshold,
            "per_window": per_window,
            "pooled": summarize(pooled_rows),
            "domain_macro": domain_macro_summary(pooled_rows),
            "domains": summarize_by_domain(pooled_rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def product_component_experience(
    history: list[dict[str, Any]],
) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in history:
        product = normalize_scope(row.get("product"))
        component = normalize_scope(row.get("component"))
        owner = normalize_owner(row.get("assignee") or row.get("assigned_to"))
        if product and component and owner:
            counts[(product, component, owner)] += 1
    return counts


def annotate_proxy(
    rows: list[dict[str, Any]],
    experience: Counter[tuple[str, str, str]],
    *,
    minimum_assignments: int,
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        product = normalize_scope(row.get("product"))
        component = normalize_scope(row.get("component"))
        ranked = [
            normalize_owner(owner)
            for owner in row.get("ranked_candidates") or []
            if normalize_owner(owner)
        ]
        top5 = ranked[:5]
        eligible = [
            owner
            for owner in top5
            if experience[(product, component, owner)] >= minimum_assignments
        ]
        row["_proxy_top1"] = bool(top5 and top5[0] in eligible)
        row["_proxy_top5"] = bool(eligible)
        row["_proxy_ranks2_to5_all"] = bool(
            len(top5) == 5
            and all(owner in eligible for owner in top5[1:5])
        )
        expected_owner = normalize_owner(row.get("expected_assignee"))
        row["_exact_owner_in_ranks2_to5"] = bool(
            expected_owner and expected_owner in top5[1:5]
        )
        row["_tiered_top5_success"] = bool(
            row.get("is_top1_correct")
            or row["_exact_owner_in_ranks2_to5"]
            or row["_proxy_ranks2_to5_all"]
        )
        row["_proxy_eligible_slots"] = len(eligible)
        row["_proxy_candidate_slots"] = len(top5)
        output.append(row)
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact_top1 = sum(bool(row.get("is_top1_correct")) for row in rows)
    exact_top5 = sum(bool(row.get("is_top5_correct")) for row in rows)
    proxy_top1 = sum(bool(row.get("_proxy_top1")) for row in rows)
    proxy_top5 = sum(bool(row.get("_proxy_top5")) for row in rows)
    tiered_top5_success = sum(
        bool(row.get("_tiered_top5_success"))
        for row in rows
    )
    eligible_slots = sum(int(row.get("_proxy_eligible_slots") or 0) for row in rows)
    candidate_slots = sum(int(row.get("_proxy_candidate_slots") or 0) for row in rows)
    return {
        "rows": total,
        "exact_owner_top1_correct_rows": exact_top1,
        "exact_owner_top1_accuracy": ratio(exact_top1, total),
        "exact_owner_top5_correct_rows": exact_top5,
        "exact_owner_top5_accuracy": ratio(exact_top5, total),
        "proxy_eligible_top1_correct_rows": proxy_top1,
        "proxy_eligible_top1_accuracy": ratio(proxy_top1, total),
        "proxy_eligible_top1_accuracy_ci95": wilson_interval(proxy_top1, total),
        "proxy_eligible_top5_correct_rows": proxy_top5,
        "proxy_eligible_top5_hit_rate": ratio(proxy_top5, total),
        "proxy_eligible_top5_hit_rate_ci95": wilson_interval(proxy_top5, total),
        "tiered_top5_success_rows": tiered_top5_success,
        "tiered_top5_accuracy": ratio(
            tiered_top5_success,
            total,
        ),
        "tiered_top5_accuracy_ci95": wilson_interval(
            tiered_top5_success,
            total,
        ),
        "top5_candidate_slots": candidate_slots,
        "proxy_eligible_candidate_slots": eligible_slots,
        "proxy_candidate_eligibility_precision": ratio(
            eligible_slots, candidate_slots
        ),
        "proxy_candidate_eligibility_precision_ci95": wilson_interval(
            eligible_slots, candidate_slots
        ),
    }


def summarize_by_domain(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            normalize_scope(row.get("product")),
            normalize_scope(row.get("component")),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (product, component), domain_rows in grouped.items():
        summary = summarize(domain_rows)
        output.append(
            {
                "product": product,
                "component": component,
                "domain": f"{product} / {component}",
                "sample_band": sample_band(len(domain_rows)),
                **summary,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -int(row["rows"]),
            str(row["product"]),
            str(row["component"]),
        ),
    )


def domain_macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = summarize_by_domain(rows)

    def average(field: str, minimum_rows: int = 1) -> float:
        values = [
            float(row[field])
            for row in domains
            if int(row["rows"]) >= minimum_rows
        ]
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "domains": len(domains),
        "domains_n_ge_20": sum(int(row["rows"]) >= 20 for row in domains),
        "domains_n_10_to_19": sum(10 <= int(row["rows"]) < 20 for row in domains),
        "domains_n_5_to_9": sum(5 <= int(row["rows"]) < 10 for row in domains),
        "domains_n_lt_5": sum(int(row["rows"]) < 5 for row in domains),
        "exact_owner_top1_macro_accuracy": average(
            "exact_owner_top1_accuracy"
        ),
        "exact_owner_top5_macro_accuracy": average(
            "exact_owner_top5_accuracy"
        ),
        "proxy_eligible_top1_macro_accuracy": average(
            "proxy_eligible_top1_accuracy"
        ),
        "proxy_eligible_top5_macro_hit_rate": average(
            "proxy_eligible_top5_hit_rate"
        ),
        "tiered_top5_macro_accuracy": average(
            "tiered_top5_accuracy"
        ),
        "proxy_candidate_eligibility_macro_precision": average(
            "proxy_candidate_eligibility_precision"
        ),
        "exact_owner_top5_macro_accuracy_n_ge_10": average(
            "exact_owner_top5_accuracy", minimum_rows=10
        ),
        "proxy_eligible_top5_macro_hit_rate_n_ge_10": average(
            "proxy_eligible_top5_hit_rate", minimum_rows=10
        ),
    }


def sample_band(rows: int) -> str:
    if rows >= 20:
        return "n>=20"
    if rows >= 10:
        return "n=10-19"
    if rows >= 5:
        return "n=5-9"
    return "n<5"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain objects: {path}")
    return rows


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


if __name__ == "__main__":
    main()
