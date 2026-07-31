from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_feedback import read_assignee_feedback_with_audit  # noqa: E402
from modules.assignee_shadow import evaluate_shadow_feedback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate append-only assignee shadow feedback.")
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-auto-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-auto-coverage", type=float, default=0.10)
    parser.add_argument("--maximum-unseen-auto-rate", type=float, default=0.05)
    parser.add_argument("--minimum-rows", type=int, default=500)
    parser.add_argument("--minimum-observation-days", type=int, default=28)
    parser.add_argument("--minimum-auto-accuracy-lower-bound", type=float, default=0.80)
    parser.add_argument("--target-top3-accuracy", type=float, default=0.90)
    parser.add_argument("--target-top5-assist-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-top5-assist-coverage", type=float, default=0.10)
    parser.add_argument("--minimum-top5-assist-rows", type=int, default=100)
    parser.add_argument(
        "--minimum-top5-assist-accuracy-lower-bound", type=float, default=0.80
    )
    parser.add_argument("--required-consecutive-weeks", type=int, default=2)
    parser.add_argument("--maximum-prediction-latency-p95-ms", type=float, default=5000.0)
    parser.add_argument("--maximum-invalid-event-rate", type=float, default=0.01)
    args = parser.parse_args()

    feedback, input_audit = read_assignee_feedback_with_audit(args.feedback)
    report = evaluate_shadow_feedback(
        feedback,
        target_auto_accuracy=args.target_auto_accuracy,
        minimum_auto_coverage=args.minimum_auto_coverage,
        maximum_unseen_auto_rate=args.maximum_unseen_auto_rate,
        minimum_rows=max(1, args.minimum_rows),
        minimum_observation_days=max(1, args.minimum_observation_days),
        minimum_auto_accuracy_lower_bound=args.minimum_auto_accuracy_lower_bound,
        target_top3_accuracy=args.target_top3_accuracy,
        target_top5_assist_accuracy=args.target_top5_assist_accuracy,
        minimum_top5_assist_coverage=args.minimum_top5_assist_coverage,
        minimum_top5_assist_rows=max(1, args.minimum_top5_assist_rows),
        minimum_top5_assist_accuracy_lower_bound=(
            args.minimum_top5_assist_accuracy_lower_bound
        ),
        required_consecutive_weeks=max(1, args.required_consecutive_weeks),
        maximum_prediction_latency_p95_ms=max(0.0, args.maximum_prediction_latency_p95_ms),
        maximum_invalid_event_rate=max(0.0, args.maximum_invalid_event_rate),
        invalid_input_events=input_audit["invalid_input_events"],
    )
    report["input_audit"] = input_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
