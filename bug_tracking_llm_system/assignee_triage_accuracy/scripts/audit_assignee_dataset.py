from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PHASE1_ROOT,
    DEFAULT_RAW_PATH,
    combined_hash,
    evaluate_current_triager,
    file_sha256,
    read_jsonl,
    rows_to_prediction_csv,
    write_csv,
    write_json,
    write_jsonl,
    prediction_csv_fields,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and fingerprint the closed-set assignee benchmark.")
    parser.add_argument("--dataset", default="bmo_paper_2024_3k")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    output_dir = args.output_dir or DEFAULT_PHASE1_ROOT / "reports" / args.dataset / "closed_set_current_repro"
    train_path = args.data_dir / f"{args.dataset}_history_train.jsonl"
    validation_path = args.data_dir / f"{args.dataset}_validation_set.jsonl"
    test_path = args.data_dir / f"{args.dataset}_test_set.jsonl"
    roster_path = args.data_dir / f"{args.dataset}_candidate_roster.json"
    summary_path = args.data_dir / f"{args.dataset}_dataset_summary.json"

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    test_rows = read_jsonl(test_path)
    roster = read_roster(roster_path)

    predictions, current_metrics = evaluate_current_triager(
        train_path=train_path,
        train_rows=train_rows,
        test_rows=test_rows,
        roster=set(roster),
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    split_paths = [train_path, validation_path, test_path, roster_path]
    manifest = {
        "experiment_name": "closed_set_current_repro",
        "dataset": args.dataset,
        "task_setting": "closed_set_frequent_assignee",
        "raw_path": str(args.raw),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "roster_path": str(roster_path),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "roster_size": len(roster),
        "top_k": args.top_k,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "split_manifest_hash": combined_hash(split_paths),
        "dataset_hash": combined_hash(split_paths + ([summary_path] if summary_path.exists() else [])),
        "file_hashes": {str(path): file_sha256(path) for path in split_paths if path.exists()},
        "config_snapshot": {
            "model": "current_assignee_triager",
            "ranking_logic": "production AssigneeTriager unchanged",
            "assignee_dataset_path": str(train_path),
            "candidate_roster_source": "post-filter train labels with minimum train frequency threshold",
            "evaluation_roster_path": str(roster_path),
        },
        "metrics": current_metrics,
    }

    write_json(output_dir / "closed_set_current_repro_metrics.json", manifest)
    write_jsonl(output_dir / "closed_set_current_repro_predictions.jsonl", predictions)
    write_csv(
        output_dir / "closed_set_current_repro_predictions.csv",
        rows_to_prediction_csv(predictions),
        prediction_csv_fields(),
    )
    write_summary(output_dir / "closed_set_current_repro_summary.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def read_roster(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted(str(candidate).strip().lower() for candidate in candidates if str(candidate).strip())


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    metrics = manifest["metrics"]
    lines = [
        f"# {manifest['experiment_name']}",
        "",
        "- Production `current_assignee_triager` was not modified.",
        "- Existing frozen split and roster were not modified.",
        f"- Train / validation / test: `{manifest['train_count']}` / `{manifest['validation_count']}` / `{manifest['test_count']}`",
        f"- Candidate roster size: `{manifest['roster_size']}`",
        f"- Split manifest hash: `{manifest['split_manifest_hash']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Top-1 | {metrics['top1_accuracy']:.6f} |",
        f"| Hit@3 | {metrics['hit_at_3']:.6f} |",
        f"| Hit@5 | {metrics['hit_at_5']:.6f} |",
        f"| Hit@10 | {metrics['hit_at_10']:.6f} |",
        f"| MRR | {metrics['mrr']:.6f} |",
        f"| Top-1 95% CI | [{metrics['top1_bootstrap_95ci']['low']:.6f}, {metrics['top1_bootstrap_95ci']['high']:.6f}] |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
