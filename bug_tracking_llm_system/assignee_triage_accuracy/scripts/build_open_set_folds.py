from __future__ import annotations

import argparse
import json
from pathlib import Path

from assignee_open_set_common import build_leave_assignee_out_folds, write_json, write_jsonl
from train_assignee_ltr import (
    DEFAULT_DATA_DIR,
    apply_label_map,
    canonicalize_source_rows,
    read_jsonl,
    read_label_map,
    row_sort_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "assignee_triage_accuracy" / "phase6_candidate_ltr" / "open_set_protocol"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-audited temporal leave-assignee-out folds."
    )
    parser.add_argument(
        "--history", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_history_train.jsonl"
    )
    parser.add_argument(
        "--queries", type=Path, default=DEFAULT_DATA_DIR / "bmo_public_10k_validation_set.jsonl"
    )
    parser.add_argument(
        "--assignee-label-map",
        type=Path,
        default=DEFAULT_DATA_DIR / "bmo_public_10k_assignee_label_map.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--hidden-query-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-history", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    label_map = read_label_map(args.assignee_label_map)
    history = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.history)), label_map),
        key=row_sort_key,
    )
    queries = sorted(
        apply_label_map(canonicalize_source_rows(read_jsonl(args.queries)), label_map),
        key=row_sort_key,
    )
    folds = build_leave_assignee_out_folds(
        history,
        queries,
        folds=args.folds,
        hidden_query_fraction=args.hidden_query_fraction,
        minimum_history=args.minimum_history,
        seed=args.seed,
    )

    manifest = {
        "schema_version": 1,
        "method": "temporal_leave_assignee_out_v1",
        "history_source": str(args.history),
        "query_source": str(args.queries),
        "parameters": {
            "folds": args.folds,
            "hidden_query_fraction": args.hidden_query_fraction,
            "minimum_history": args.minimum_history,
            "seed": args.seed,
        },
        "leakage_audit": [fold.audit for fold in folds],
        "all_folds_leakage_free": all(fold.audit["leakage_free"] for fold in folds),
    }
    for fold in folds:
        fold_dir = args.output_dir / f"fold_{fold.fold}"
        write_jsonl(fold_dir / "reduced_history.jsonl", fold.history_rows)
        write_jsonl(fold_dir / "queries.jsonl", fold.query_rows)
        write_json(
            fold_dir / "fold_manifest.json",
            {
                **fold.audit,
                "reduced_history": "reduced_history.jsonl",
                "queries": "queries.jsonl",
            },
        )
    write_json(args.output_dir / "leave_assignee_out_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
