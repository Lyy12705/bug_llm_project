from __future__ import annotations

import argparse

from common import read_records, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate patch generation success.")
    parser.add_argument("--pred", required=True, help="Predicted patch JSON or pipeline result JSON.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = read_records(args.pred)
    statuses = []
    for row in rows:
        patch = row.get("patch") if isinstance(row.get("patch"), dict) else row
        statuses.append(patch.get("patch_status") in {"generated", "provided"} and not patch.get("requires_manual_patch"))
    metrics = {"rows": len(rows), "patch_success_rate": sum(statuses) / len(statuses) if statuses else 0.0}
    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
