#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (str(ROOT), str(SRC)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from scripts.evaluate_fault_localization import (
    discover_base_commit_existing_gold_files,
    evaluate_records,
)


METHOD_ARGUMENTS = {
    "tfidf": ["--embedding-backend", "tfidf", "--no-domain-path-routing"],
    "tfidf-routing": ["--embedding-backend", "tfidf", "--domain-path-routing"],
    "tfidf-sbert": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
    ],
    "e3-a-baseline": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "off",
    ],
    "e3-b-symbol-definitions": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions",
    ],
    "e3-b-symbol-definitions-strict": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-strict",
    ],
    "e3-b-symbol-definitions-guarded": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-guarded",
    ],
    "e3-c-api-implementation": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
    ],
    "e3-c-api-namespace": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api-namespace",
    ],
    "e4-b-call-outgoing": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
        "--call-graph-mode",
        "outgoing",
    ],
    "e4-c-call-outgoing-top3": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
        "--call-graph-mode",
        "outgoing-top3",
    ],
    "e5-b-size-aware-75": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--semantic-candidate-policy",
        "repository-size",
        "--medium-semantic-candidate-k",
        "75",
        "--large-semantic-candidate-k",
        "75",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
    ],
    "e5-c-size-aware-100": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--semantic-candidate-policy",
        "repository-size",
        "--medium-semantic-candidate-k",
        "100",
        "--large-semantic-candidate-k",
        "100",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
    ],
    "e7-a-e6-development-baseline": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
        "--call-graph-mode",
        "outgoing-top3",
    ],
    "e8-a-source-extension-contract": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
        "--call-graph-mode",
        "outgoing-top3",
    ],
    "e7-b-large-only-100": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--semantic-candidate-policy",
        "repository-size",
        "--medium-semantic-candidate-k",
        "50",
        "--large-semantic-candidate-k",
        "100",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--symbol-expansion-mode",
        "symbol-definitions-api",
        "--call-graph-mode",
        "outgoing-top3",
    ],
    "e1-b-supporting-chunks": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "supporting-chunks",
    ],
    "e1-c-supporting-symbols": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "supporting-symbols",
    ],
    "e1-d-supporting-symbols-package": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "supporting-symbols-package",
    ],
    "e2-b-import-outgoing": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--import-graph-mode",
        "outgoing",
    ],
    "e2-c-import-bidirectional": [
        "--embedding-backend",
        "tfidf-sbert-rerank",
        "--semantic-candidate-k",
        "50",
        "--no-domain-path-routing",
        "--file-aggregation-mode",
        "basic",
        "--import-graph-mode",
        "bidirectional",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Stage-1 method over a SWE-bench full split using repository-parallel shards."
    )
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--repo-cache-dir", required=True)
    parser.add_argument("--index-cache-dir", required=True)
    parser.add_argument(
        "--import-graph-cache-dir",
        default=None,
        help="Shared E2 graph cache; defaults to <index-cache-dir>/import_graphs.",
    )
    parser.add_argument(
        "--call-graph-cache-dir",
        default=None,
        help="Shared E4 graph cache; defaults to <index-cache-dir>/call_graphs.",
    )
    parser.add_argument("--fallback-index-cache-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_ARGUMENTS), required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--candidate-file-k", type=int, default=20)
    parser.add_argument(
        "--repository-size-groups",
        default="configs/fault_localization/e5_repository_size_groups.json",
        help="Frozen E5 repository-size group mapping passed to size-aware methods.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--allow-sbert-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--freeze-manifest",
        default=None,
        help="Required guard for a one-time Frozen Holdout run; validates method, data, and code hashes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers <= 0:
        raise ValueError("max-workers must be positive.")
    tickets_path = Path(args.tickets)
    gold_path = Path(args.gold)
    output_dir = Path(args.output_dir)
    if _uses_repository_size_policy(args.method) and not Path(
        args.repository_size_groups
    ).is_file():
        raise FileNotFoundError(args.repository_size_groups)
    freeze = (
        validate_freeze_manifest(
            Path(args.freeze_manifest),
            method_label=args.method,
            tickets_path=tickets_path,
            gold_path=gold_path,
            output_dir=output_dir,
        )
        if args.freeze_manifest
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tickets = read_jsonl(tickets_path)

    repositories: dict[str, list[dict[str, Any]]] = {}
    for ticket in tickets:
        repositories.setdefault(str(ticket.get("repo") or "unknown"), []).append(ticket)
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_specs: list[dict[str, Any]] = []
    for repository, rows in sorted(repositories.items()):
        safe_repo = safe_name(repository)
        shard_dir = shard_root / safe_repo
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_tickets = shard_dir / "tickets.jsonl"
        write_jsonl(shard_tickets, rows)
        shard_specs.append(
            {
                "repository": repository,
                "rows": len(rows),
                "tickets": shard_tickets,
                "output": shard_dir / "run",
                "log": shard_dir / "run.log",
            }
        )

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[experiment_started] method={args.method} tickets={len(tickets)} "
        f"repositories={len(shard_specs)} workers={args.max_workers}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_shard, spec, args): spec for spec in shard_specs}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            results.append(result)
            print(
                f"[repository_completed] method={args.method} repo={spec['repository']} "
                f"tickets={spec['rows']} predictions={result['predictions']} "
                f"failures={result['failures']} wall_seconds={result['wall_seconds']:.1f}",
                flush=True,
            )

    failures = [result for result in results if result["returncode"] != 0]
    if failures:
        details = ", ".join(f"{row['repository']} (exit {row['returncode']})" for row in failures)
        raise RuntimeError(f"Repository shard execution failed: {details}")

    predictions_by_id: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    for spec in shard_specs:
        run_dir = Path(spec["output"])
        for prediction in read_jsonl(run_dir / "test_predictions.jsonl"):
            predictions_by_id[ticket_id(prediction)] = prediction
        failure_rows.extend(read_jsonl(run_dir / "test_failures.jsonl"))
        shard_manifests.append(json.loads((run_dir / "test_run_manifest.json").read_text(encoding="utf-8")))
    predictions = [predictions_by_id[ticket_id(ticket)] for ticket in tickets if ticket_id(ticket) in predictions_by_id]
    prediction_output = output_dir / "predictions.jsonl"
    failure_output = output_dir / "failures.jsonl"
    metrics_output = output_dir / "metrics.json"
    manifest_output = output_dir / "run_manifest.json"
    write_jsonl(prediction_output, predictions)
    write_jsonl(failure_output, failure_rows)
    prediction_phase_completed_at = datetime.now(timezone.utc).isoformat()
    if len(predictions) != len(tickets) or failure_rows:
        raise ValueError(
            "Prediction phase must complete every frozen ticket before sealed gold is opened."
        )

    # The sealed gold file is intentionally opened only after all prediction
    # shards have completed and the combined predictions are durably written.
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {ticket_id(row): row for row in gold_rows}
    if {ticket_id(row) for row in tickets} != set(gold_by_id):
        raise ValueError("Ticket and gold ID sets do not match.")
    reachable_gold = discover_base_commit_existing_gold_files(
        gold_rows,
        Path(args.repo_cache_dir),
        ticket_rows=tickets,
    )
    metrics = evaluate_records(
        gold_rows,
        predictions,
        reachable_gold_files_by_ticket=reachable_gold,
    )
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    method_ids = {str(manifest.get("method_id") or "") for manifest in shard_manifests}
    method_specs = {
        json.dumps(manifest.get("method") or {}, ensure_ascii=False, sort_keys=True)
        for manifest in shard_manifests
    }
    if len(method_ids) != 1 or len(method_specs) != 1:
        raise ValueError("Repository shards did not use one identical method configuration.")
    selected_method = json.loads(next(iter(method_specs)))
    if freeze is not None:
        if next(iter(method_ids)) != freeze["selected_method_id"]:
            raise ValueError("Holdout method ID drifted from the frozen selection.")
        if selected_method != freeze["selected_method"]:
            raise ValueError("Holdout method settings drifted from the frozen selection.")
    manifest = {
        "protocol": "swebench-full-stage1-protocol-v1",
        "started_at_utc": started_at,
        "prediction_phase_completed_at_utc": prediction_phase_completed_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "method_label": args.method,
        "method_id": next(iter(method_ids)),
        "method": selected_method,
        "tickets_path": str(tickets_path),
        "gold_path": str(gold_path),
        "tickets": len(tickets),
        "predictions": len(predictions),
        "failures": len(failure_rows),
        "repositories": len(shard_specs),
        "max_workers": args.max_workers,
        "predictions_output": str(prediction_output),
        "metrics_output": str(metrics_output),
        "failures_output": str(failure_output),
        "repository_runs": sorted(results, key=lambda row: row["repository"]),
        "evaluation_order": {
            "prediction_only_shards": True,
            "predictions_completed_before_gold_access": True,
            "sealed_gold_accessed_during_ranking": False,
        },
        "metrics": metrics,
    }
    if args.freeze_manifest:
        manifest["freeze_manifest"] = artifact(Path(args.freeze_manifest))
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[experiment_completed] method={args.method} predictions={len(predictions)} "
        f"failures={len(failure_rows)} wall_seconds={manifest['wall_seconds']}",
        flush=True,
    )
    print(json.dumps({
        "method": args.method,
        "method_id": manifest["method_id"],
        "tickets": len(tickets),
        "predictions": len(predictions),
        "failures": len(failure_rows),
        "candidate_hit_at_20": metrics["candidate_hit_at_20"],
        "candidate_recall_at_20": metrics["candidate_recall_at_20"],
        "file_top_1_accuracy": metrics["file_top_1_accuracy"],
        "file_mrr": metrics["file_mrr"],
    }, ensure_ascii=False, indent=2))


def build_shard_command(spec: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_swebench_lite_fault_localization.py"),
        "--tickets",
        str(spec["tickets"]),
        "--prediction-only",
        "--repo-cache-dir",
        str(Path(args.repo_cache_dir)),
        "--snapshot-cache-dir",
        str(Path(args.output_dir) / "ephemeral_snapshots"),
        "--index-cache-dir",
        str(Path(args.index_cache_dir)),
        "--import-graph-cache-dir",
        str(
            Path(args.import_graph_cache_dir)
            if args.import_graph_cache_dir
            else Path(args.index_cache_dir) / "import_graphs"
        ),
        "--call-graph-cache-dir",
        str(
            Path(args.call_graph_cache_dir)
            if args.call_graph_cache_dir
            else Path(args.index_cache_dir) / "call_graphs"
        ),
        "--output-dir",
        str(spec["output"]),
        "--candidate-file-k",
        str(args.candidate_file_k),
        "--ephemeral-snapshots",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--progress",
        "none",
        *METHOD_ARGUMENTS[args.method],
    ]
    for cache_dir in args.fallback_index_cache_dir:
        command.extend(["--fallback-index-cache-dir", str(Path(cache_dir))])
    if not args.no_resume:
        command.append("--resume")
    if args.allow_sbert_download:
        command.append("--allow-sbert-download")
    if _uses_repository_size_policy(args.method):
        command.extend(
            ["--repository-size-groups", str(Path(args.repository_size_groups).resolve())]
        )
    return command


def _uses_repository_size_policy(method: str) -> bool:
    return "--semantic-candidate-policy" in METHOD_ARGUMENTS[method]


def run_shard(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = build_shard_command(spec, args)
    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    log_text = "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr
    Path(spec["log"]).write_text(log_text, encoding="utf-8")
    manifest_path = Path(spec["output"]) / "test_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "repository": spec["repository"],
        "tickets": spec["rows"],
        "returncode": completed.returncode,
        "predictions": int(manifest.get("predictions") or 0),
        "failures": int(manifest.get("failures") or 0),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "log": str(spec["log"]),
        "manifest": str(manifest_path),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def ticket_id(row: dict[str, Any]) -> str:
    return str(row.get("ticket_id") or row.get("instance_id") or row.get("id") or "")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def validate_freeze_manifest(
    path: Path,
    *,
    method_label: str,
    tickets_path: Path,
    gold_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("protocol") != "swebench-full-stage1-protocol-v1":
        raise ValueError("Freeze manifest uses the wrong protocol.")
    if freeze.get("status") != "frozen_before_holdout":
        raise ValueError("Freeze manifest is not in frozen_before_holdout state.")
    if freeze.get("selected_method_label") != method_label:
        raise ValueError(
            f"Requested method {method_label} does not match frozen method "
            f"{freeze.get('selected_method_label')}."
        )
    authorized_output = freeze.get("authorized_output_dir")
    if authorized_output:
        authorized_path = Path(str(authorized_output))
        if not authorized_path.is_absolute():
            authorized_path = ROOT / authorized_path
        if output_dir.resolve() != authorized_path.resolve():
            raise ValueError(
                "Holdout output directory differs from the frozen authorization: "
                f"{output_dir}"
            )
    final_record = freeze.get("final_record_path")
    if final_record:
        final_record_path = Path(str(final_record))
        if not final_record_path.is_absolute():
            final_record_path = ROOT / final_record_path
        if final_record_path.exists():
            raise FileExistsError(
                f"Final Holdout was already evaluated: {final_record_path}"
            )
    artifacts = freeze.get("frozen_holdout_artifacts") or {}
    if sha256_file(tickets_path) != (artifacts.get("tickets") or {}).get("sha256"):
        raise ValueError("Frozen Holdout ticket file hash does not match the freeze manifest.")
    if sha256_file(gold_path) != (artifacts.get("gold") or {}).get("sha256"):
        raise ValueError("Frozen Holdout gold file hash does not match the freeze manifest.")
    for relative, expected in (freeze.get("implementation") or {}).items():
        implementation_path = ROOT / relative
        if sha256_file(implementation_path) != expected.get("sha256"):
            raise ValueError(f"Implementation changed after freeze: {relative}")
    completed_manifest = output_dir / "run_manifest.json"
    if completed_manifest.exists():
        raise FileExistsError(
            f"Refusing to rerun a completed Frozen Holdout evaluation: {completed_manifest}"
        )
    return freeze


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
