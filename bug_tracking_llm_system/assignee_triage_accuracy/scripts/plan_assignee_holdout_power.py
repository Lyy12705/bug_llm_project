from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


Z_95_TWO_SIDED = 1.959963984540054


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan unseen-owner sample size so the two-sided Wilson 95% upper "
            "bound can fall below the release safety limit."
        )
    )
    parser.add_argument("--expected-unseen-auto-error-rate", type=float, required=True)
    parser.add_argument("--maximum-upper-bound", type=float, default=0.05)
    parser.add_argument("--expected-unseen-owner-fraction", type=float, default=0.07)
    parser.add_argument("--minimum-total-holdout-rows", type=int, default=2500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = holdout_sample_plan(
        expected_error_rate=args.expected_unseen_auto_error_rate,
        maximum_upper_bound=args.maximum_upper_bound,
        expected_unseen_fraction=args.expected_unseen_owner_fraction,
        minimum_total_rows=args.minimum_total_holdout_rows,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def holdout_sample_plan(
    *,
    expected_error_rate: float,
    maximum_upper_bound: float,
    expected_unseen_fraction: float,
    minimum_total_rows: int,
) -> dict[str, Any]:
    if not 0.0 <= expected_error_rate < maximum_upper_bound < 1.0:
        raise ValueError(
            "expected error must be non-negative and below the upper-bound target"
        )
    if not 0.0 < expected_unseen_fraction <= 1.0:
        raise ValueError("expected unseen-owner fraction must be in (0, 1]")
    if minimum_total_rows < 1:
        raise ValueError("minimum total holdout rows must be positive")
    unseen_rows = minimum_continuous_wilson_sample_size(
        expected_error_rate, maximum_upper_bound
    )
    total_for_unseen = math.ceil(unseen_rows / expected_unseen_fraction)
    recommended_total = max(minimum_total_rows, total_for_unseen)
    return {
        "schema_version": 1,
        "method": "two_sided_wilson_95_upper_bound_sample_planning_v1",
        "assumptions": {
            "expected_unseen_auto_error_rate": expected_error_rate,
            "maximum_unseen_rate_ci95_upper_bound": maximum_upper_bound,
            "expected_unseen_owner_fraction": expected_unseen_fraction,
            "z": Z_95_TWO_SIDED,
        },
        "minimum_unseen_owner_rows": unseen_rows,
        "minimum_total_rows_from_unseen_fraction": total_for_unseen,
        "minimum_total_holdout_rows_gate": minimum_total_rows,
        "recommended_total_holdout_rows": recommended_total,
        "expected_unseen_rows_at_recommended_total": math.floor(
            recommended_total * expected_unseen_fraction
        ),
        "limitations": [
            "This is deterministic event-rate planning, not a guarantee of the observed error count.",
            "Recalculate before sealing when the expected event rate or unseen-owner fraction changes.",
            "The formal evaluation must use integer observed errors and the same two-sided Wilson interval.",
        ],
    }


def minimum_continuous_wilson_sample_size(
    expected_rate: float, maximum_upper_bound: float
) -> int:
    for rows in range(1, 10_000_001):
        if wilson_upper_for_expected_rate(expected_rate, rows) < maximum_upper_bound:
            return rows
    raise ValueError("required unseen-owner sample exceeds ten million rows")


def wilson_upper_for_expected_rate(expected_rate: float, rows: int) -> float:
    if rows < 1 or not 0.0 <= expected_rate <= 1.0:
        raise ValueError("Wilson planning requires positive rows and a valid rate")
    z2 = Z_95_TWO_SIDED * Z_95_TWO_SIDED
    denominator = 1.0 + z2 / rows
    center = (expected_rate + z2 / (2.0 * rows)) / denominator
    margin = (
        Z_95_TWO_SIDED
        * math.sqrt(
            expected_rate * (1.0 - expected_rate) / rows
            + z2 / (4.0 * rows * rows)
        )
        / denominator
    )
    return min(1.0, center + margin)


if __name__ == "__main__":
    main()
