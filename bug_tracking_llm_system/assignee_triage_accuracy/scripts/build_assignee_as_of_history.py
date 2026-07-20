from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.assignee_deployment import GENERIC_ASSIGNEE_RE  # noqa: E402

from assignee_open_set_common import write_json, write_jsonl  # noqa: E402
from build_assignee_roster_snapshot import load_history, parse_datetime  # noqa: E402
from train_assignee_ltr import DEFAULT_DATA_DIR, parse_timestamp, row_sort_key  # noqa: E402
from train_assignee_rolling_open_set import RAW_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe ranker history whose labels were available by a cutoff."
    )
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--base-raw", type=Path, default=RAW_DIR / "bmo_public_10k_raw.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    cutoff = parse_datetime(args.as_of)
    if cutoff is None:
        raise SystemExit("--as-of must be a valid ISO-8601 timestamp")
    rows, source_audit = load_history(
        args.history,
        label_map_path=args.assignee_label_map,
        base_raw=args.base_raw,
    )
    eligible, audit = select_as_of_history(rows, cutoff)
    audit["sources"] = source_audit["sources"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, eligible)
    write_json(args.manifest, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def select_as_of_history(
    rows: list[dict[str, Any]], cutoff: datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cutoff_epoch = cutoff.astimezone(UTC).timestamp()
    eligible = []
    exclusions = Counter()
    for source in rows:
        row = dict(source)
        owner = str(row.get("assignee") or "").strip().lower()
        created_at = parse_timestamp(str(row.get("created_at") or ""))
        label_at = parse_timestamp(str(row.get("label_available_at") or ""))
        if not owner or GENERIC_ASSIGNEE_RE.search(owner):
            exclusions["missing_or_generic_owner"] += 1
        elif created_at is None:
            exclusions["missing_created_at"] += 1
        elif created_at >= cutoff_epoch:
            exclusions["created_at_or_after_cutoff"] += 1
        elif label_at is None:
            exclusions["missing_label_available_at"] += 1
        elif label_at > cutoff_epoch:
            exclusions["label_unavailable_at_cutoff"] += 1
        else:
            eligible.append(row)
    eligible.sort(key=row_sort_key)
    return eligible, {
        "schema_version": 1,
        "method": "label_availability_as_of_history_v1",
        "as_of": cutoff.astimezone(UTC).isoformat(),
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "owners": len({str(row.get("assignee") or "") for row in eligible}),
        "exclusions": dict(sorted(exclusions.items())),
        "temporal_order_valid": bool(eligible)
        and all(row_sort_key(row) < (cutoff_epoch, "") for row in eligible),
    }


if __name__ == "__main__":
    main()
