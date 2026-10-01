"""Run the FL -> FIM repair pipeline over a whole ticket set and aggregate results.

``fim_patch.cli`` runs exactly one ticket against one already-checked-out
repository. This script is the batch layer around it: for every ticket it
materializes a detached snapshot at ``base_commit``, runs the same
``RepairPipeline``, and appends one flat record per ticket to a JSONL file so a
46-ticket run can be counted instead of read by hand.

Three properties this script is responsible for, all of them stated in
``STAGE4_PATCH_GENERATION_IMPLEMENTATION_PLAN_ZH.md``:

1. **Ground-truth isolation (plan section 4.1).** The developer ``patch``,
   ``test_patch``, ``fail_to_pass``, ``pass_to_pass`` and ``hints_text`` are
   stripped from every ticket before it reaches localization or generation.
   ``fim_patch.generator.evidence()`` already uses an allowlist; this is the
   second, independent barrier, so neither layer alone is load-bearing.
2. **One failure does not lose the run.** A ticket that raises is recorded with
   its error and the batch continues. ``--resume`` re-reads the JSONL and skips
   tickets already finished.
3. **No number in the summary is a repair rate.** ``patch_status=generated``
   counts generation, not repair. ``official_resolved`` stays ``None`` until the
   official SWE-bench Docker harness has judged the predictions file.

Usage::

    python scripts/run_stage4_batch.py --tickets TICKETS.jsonl --output runs/batch-001 \
        --target-policy first_supported_ast --local-context --max-body-lines 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fim_patch.backend import OllamaFIM  # noqa: E402
from fim_patch.generator import (  # noqa: E402
    RESEARCH_ONLY_TARGET_POLICIES,
    TARGET_POLICY_DEPTH,
    FIMPatchGenerator,
)
from fim_patch.pipeline import RepairPipeline, write_run  # noqa: E402
from utils.repo_snapshot import (  # noqa: E402
    RepoSnapshotError,
    materialize_commit_snapshot,
    resolve_repository_path,
)

#: Fields that would leak the answer into localization or the model prompt.
#: Mirrors ``scripts/generate_stage4_patches.py`` on the WP2 branch.
TICKET_GROUND_TRUTH_FIELDS = frozenset(
    {"patch", "test_patch", "gold_patch", "fail_to_pass", "pass_to_pass", "hints_text"}
)

#: Maps a target-resolution rejection message to a countable bucket. Order
#: matters: the first matching fragment wins.
REJECTION_BUCKETS = (
    ("Unsupported or ambiguous function symbol", "unsupported_symbol"),
    ("Localization coordinates disagree with AST", "coordinates_disagree"),
    ("max_body_lines", "body_too_large"),
    ("Inline function bodies", "inline_body"),
    ("Python source files only", "not_python_source"),
    ("Source is missing or outside repository", "source_missing"),
    ("Invalid source path", "source_missing"),
    ("Source hash differs", "source_mismatch"),
    ("Source differs from base_commit", "source_mismatch"),
    ("line endings", "line_endings"),
    ("Source exceeds", "source_too_large"),
    ("Stage-3", "no_stage3_symbols"),
)

DEFAULT_RECORDS_NAME = "stage4_batch_records.jsonl"
DEFAULT_SNAPSHOT_SUBDIR = "_stage4_snapshots"


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def load_ticket_id_allowlist(path: Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return {line.strip() for line in text.splitlines() if line.strip()}


def ticket_identifier(ticket: dict) -> str:
    return str(ticket.get("ticket_id") or ticket.get("instance_id") or "").strip()


def sanitize_ticket(ticket: dict) -> dict:
    """Return the ticket with every ground-truth field removed.

    Removal, not masking: a placeholder value would still be a value the
    localization stage could key on.
    """

    return {k: v for k, v in ticket.items() if k not in TICKET_GROUND_TRUTH_FIELDS}


def safe_dir_name(value: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in value)
    return cleaned or "unnamed"


# --------------------------------------------------------------------------
# per-ticket record
# --------------------------------------------------------------------------

def classify_rejection(reason: str) -> str:
    for fragment, bucket in REJECTION_BUCKETS:
        if fragment in reason:
            return bucket
    return "other"


def build_record(ticket: dict, *, result=None, location=None, error: str = "",
                 duration_s: float = 0.0, output_dir: str = "", skipped: str = "") -> dict:
    """Flatten one ticket's outcome into a single countable row."""

    record = {
        "ticket_id": ticket_identifier(ticket),
        "repo": ticket.get("repo", ""),
        "base_commit": ticket.get("base_commit", ""),
        "output_dir": output_dir,
        "duration_s": round(duration_s, 3),
        "error": error,
        "skipped": skipped,
        "patch_status": "",
        "validation_status": "",
        "target_policy": "",
        "target_rank": None,
        "target_fallback_used": None,
        "expand_truncated_end": None,
        "max_body_lines": None,
        "research_mode": None,
        "body_lines": None,
        "target_file": "",
        "target_symbol": "",
        "rejection_buckets": [],
        "rejection_reasons": [],
        "candidate_count": 0,
        "candidate_generated_count": 0,
        "patch_line_count": 0,
        "model": "",
        "confidence_level": "",
        "should_manual_review": None,
        "recommend_patch_generation": None,
        "production_gate_eligible": None,
        "rank1_symbol_kind": "",
        "rank1_line_span": None,
    }
    if location:
        record["confidence_level"] = location.get("confidence_level", "")
        record["should_manual_review"] = location.get("should_manual_review")
        record["recommend_patch_generation"] = location.get("recommend_patch_generation")
        # The generator's production gate is all three conditions at once.
        record["production_gate_eligible"] = (
            location.get("confidence_level") == "high"
            and location.get("should_manual_review") is False
            and location.get("recommend_patch_generation") is True)
        rows = location.get("stage3_ranked_symbols") or []
        if rows and isinstance(rows[0], dict):
            record["rank1_symbol_kind"] = str(rows[0].get("symbol_kind", ""))
            start, end = rows[0].get("start_line"), rows[0].get("end_line")
            if isinstance(start, int) and isinstance(end, int):
                record["rank1_line_span"] = end - start + 1
    if not result:
        return record

    record.update(
        patch_status=result.get("patch_status", ""),
        validation_status=result.get("validation_status", ""),
        target_policy=result.get("target_policy", ""),
        target_rank=result.get("target_rank"),
        target_fallback_used=result.get("target_fallback_used"),
        expand_truncated_end=result.get("expand_truncated_end"),
        max_body_lines=result.get("max_body_lines"),
        research_mode=result.get("research_mode"),
    )
    span = result.get("span") or {}
    record["body_lines"] = span.get("body_lines")
    bug_location = result.get("bug_location") or {}
    record["target_file"] = bug_location.get("file", "")
    record["target_symbol"] = bug_location.get("function", "")

    reasons = [str(entry.get("reason", "")) for entry in result.get("target_resolution") or []
               if entry.get("status") == "rejected" and entry.get("reason")]
    record["rejection_reasons"] = reasons
    record["rejection_buckets"] = [classify_rejection(reason) for reason in reasons]

    candidates = result.get("candidates") or []
    record["candidate_count"] = len(candidates)
    record["candidate_generated_count"] = sum(
        1 for c in candidates if c.get("status") == "generated")
    patch = result.get("patch") or ""
    record["patch_line_count"] = sum(
        1 for line in patch.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    model = result.get("model")
    if isinstance(model, dict):
        record["model"] = str(model.get("model", ""))
    elif model:
        record["model"] = str(model)
    if not record["error"] and result.get("explanation") and result.get("patch_status") != "generated":
        record["error"] = ""  # an explained block is not an error; keep them distinguishable
    record["explanation"] = str(result.get("explanation", ""))
    return record


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def summarize(records: list[dict]) -> dict:
    done = [r for r in records if not r.get("skipped")]
    errored = [r for r in done if r.get("error")]
    attempted = [r for r in done if not r.get("error")]

    status_counts: dict[str, int] = {}
    for record in attempted:
        status = record.get("patch_status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    bucket_counts: dict[str, int] = {}
    for record in attempted:
        for bucket in record.get("rejection_buckets") or []:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    rank_counts: dict[str, int] = {}
    for record in attempted:
        rank = record.get("target_rank")
        if rank is not None:
            rank_counts[str(rank)] = rank_counts.get(str(rank), 0) + 1

    confidence_counts: dict[str, int] = {}
    for record in attempted:
        level = record.get("confidence_level") or "unknown"
        confidence_counts[level] = confidence_counts.get(level, 0) + 1

    kind_counts: dict[str, int] = {}
    for record in attempted:
        kind = record.get("rank1_symbol_kind") or "unknown"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    spans = sorted(r["rank1_line_span"] for r in attempted if isinstance(r.get("rank1_line_span"), int))
    research_modes = sorted({bool(r.get("research_mode")) for r in attempted
                             if r.get("research_mode") is not None})

    generated = [r for r in attempted if r.get("patch_status") == "generated"]
    fallback = [r for r in generated if r.get("target_fallback_used")]
    body_lines = sorted(r["body_lines"] for r in attempted if isinstance(r.get("body_lines"), int))

    total_attempted = len(attempted)
    return {
        "total_tickets": len(records),
        "skipped": len(records) - len(done),
        "errored": len(errored),
        "attempted": total_attempted,
        "confidence_policy": "advisory_only",
        # True only when research_mode bypassed the generator's production gate.
        # Retained to distinguish historical research runs; confidence itself
        # is now advisory in both modes.
        "research_mode": research_modes[0] if len(research_modes) == 1 else research_modes,
        "confidence_level_counts": dict(sorted(confidence_counts.items())),
        "production_gate_eligible": sum(1 for r in attempted if r.get("production_gate_eligible")),
        "rank1_symbol_kind_counts": dict(sorted(kind_counts.items())),
        "rank1_line_span_median": spans[len(spans) // 2] if spans else None,
        "patch_status_counts": dict(sorted(status_counts.items())),
        "target_rejection_buckets": dict(sorted(bucket_counts.items())),
        "target_rank_counts": dict(sorted(rank_counts.items(), key=lambda kv: int(kv[0]))),
        # Generation rate. NOT a repair rate -- see the note below.
        "generated": len(generated),
        "generated_rate": round(len(generated) / total_attempted, 4) if total_attempted else 0.0,
        "generated_at_rank_1": len(generated) - len(fallback),
        "generated_via_fallback": len(fallback),
        "validation_plausible": sum(1 for r in generated if r.get("validation_status") == "plausible"),
        "body_lines_median": body_lines[len(body_lines) // 2] if body_lines else None,
        "body_lines_max": body_lines[-1] if body_lines else None,
        "official_resolved": None,
        "note": (
            "generated_rate counts patches that passed the static and apply gates. It is NOT a "
            "repair rate. validation_plausible counts supplied local tests only. A repair rate "
            "requires official_resolved from the official SWE-bench Docker harness, which this "
            "script does not run."
        ),
    }


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def make_generator(args) -> FIMPatchGenerator:
    return FIMPatchGenerator(
        OllamaFIM(model=args.model, url=args.url, num_ctx=args.num_ctx,
                  max_new_tokens=args.max_new_tokens),
        candidates=args.candidates, seed=args.seed,
        research_mode=args.research_mode, local_context=args.local_context,
        target_policy=args.target_policy,
        expand_truncated_end=args.expand_truncated_end,
        max_body_lines=args.max_body_lines,
    )


def run_one_ticket(ticket, *, pipeline, repo_cache_dir: Path, snapshot_cache_dir: Path,
                   ticket_output_dir: Path, clone_missing: bool, fetch_missing_commits: bool) -> dict:
    """Materialize the snapshot, run the pipeline, write the run, return a record."""

    started = time.monotonic()
    clean = sanitize_ticket(ticket)
    try:
        source_repo = resolve_repository_path(clean, repo_cache_dir=repo_cache_dir,
                                              clone_missing=clone_missing)
        snapshot = materialize_commit_snapshot(
            source_repo, repo=str(clean.get("repo") or ""),
            base_commit=str(clean.get("base_commit") or ""),
            snapshot_cache_dir=snapshot_cache_dir,
            fetch_missing_commits=fetch_missing_commits,
        )
        location, result = pipeline.run(clean, str(snapshot))
    except RepoSnapshotError as exc:
        return build_record(ticket, error=f"snapshot: {exc}", duration_s=time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 - one ticket must not end the batch
        return build_record(ticket, error=f"{type(exc).__name__}: {exc}",
                            duration_s=time.monotonic() - started)

    try:
        write_run(ticket_output_dir, clean, location, result)
    except Exception as exc:  # noqa: BLE001
        record = build_record(ticket, result=result, location=location,
                              duration_s=time.monotonic() - started)
        record["error"] = f"write_run: {type(exc).__name__}: {exc}"
        return record

    return build_record(ticket, result=result, location=location,
                        duration_s=time.monotonic() - started,
                        output_dir=str(ticket_output_dir))


def load_previous_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["ticket_id"]: r for r in load_jsonl(path) if r.get("ticket_id")}


def write_predictions(path: Path, records: list[dict], run_root: Path) -> int:
    """Merge each generated ticket's predictions.jsonl into one harness input."""

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.get("patch_status") != "generated" or not record.get("output_dir"):
                continue
            source = Path(record["output_dir"]) / "predictions.jsonl"
            if not source.exists():
                continue
            for row in load_jsonl(source):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickets", required=True, type=Path,
                        help="Ticket JSONL with ticket_id/repo/base_commit and a public description.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Run directory. Created if absent; reused with --resume.")
    parser.add_argument("--ticket-ids-from", type=Path, default=None,
                        help="Newline-separated ticket ids to restrict the run to.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N selected tickets.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tickets already present in the records file.")
    parser.add_argument("--repo-cache-dir", type=Path, default=Path("data/repositories"))
    parser.add_argument("--snapshot-cache-dir", type=Path, default=None,
                        help=f"Defaults to <repo-cache-dir>/{DEFAULT_SNAPSHOT_SUBDIR}")
    parser.add_argument("--clone-missing", action="store_true")
    parser.add_argument("--fetch-missing-commits", action="store_true")
    # generator passthrough
    parser.add_argument("--model", default="codellama:7b-instruct")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--research-mode", action="store_true")
    parser.add_argument("--local-context", action="store_true")
    parser.add_argument("--target-policy", choices=tuple(TARGET_POLICY_DEPTH), default="first_supported_ast")
    parser.add_argument("--expand-truncated-end", dest="expand_truncated_end",
                        action="store_true", default=None)
    parser.add_argument("--no-expand-truncated-end", dest="expand_truncated_end",
                        action="store_false")
    parser.add_argument("--max-body-lines", type=int, default=0)
    return parser


def select_tickets(rows: list[dict], *, allowlist: set[str] | None, limit: int | None) -> list[dict]:
    selected = [r for r in rows if ticket_identifier(r)]
    if allowlist is not None:
        selected = [r for r in selected if ticket_identifier(r) in allowlist]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_policy in RESEARCH_ONLY_TARGET_POLICIES and not args.research_mode:
        parser.error("Alternative target selection requires --research-mode")
    if args.max_body_lines < 0:
        parser.error("--max-body-lines must be zero or positive")

    run_root = args.output
    run_root.mkdir(parents=True, exist_ok=True)
    records_path = run_root / DEFAULT_RECORDS_NAME
    snapshot_cache_dir = args.snapshot_cache_dir or (args.repo_cache_dir / DEFAULT_SNAPSHOT_SUBDIR)

    rows = load_jsonl(args.tickets)
    allowlist = load_ticket_id_allowlist(args.ticket_ids_from) if args.ticket_ids_from else None
    tickets = select_tickets(rows, allowlist=allowlist, limit=args.limit)
    if not tickets:
        print("No tickets selected.", file=sys.stderr)
        return 1

    previous = load_previous_records(records_path) if args.resume else {}
    if previous:
        print(f"Resume: {len(previous)} ticket(s) already recorded in {records_path}")

    (run_root / "run_config.json").write_text(json.dumps({
        "tickets": str(args.tickets), "ticket_count": len(tickets),
        "target_policy": args.target_policy, "expand_truncated_end": args.expand_truncated_end,
        "max_body_lines": args.max_body_lines, "candidates": args.candidates, "seed": args.seed,
        "model": args.model, "num_ctx": args.num_ctx, "max_new_tokens": args.max_new_tokens,
        "local_context": args.local_context, "research_mode": args.research_mode,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    pipeline = RepairPipeline(make_generator(args))
    records: list[dict] = []
    handle = records_path.open("a" if previous else "w", encoding="utf-8")
    try:
        for position, ticket in enumerate(tickets, 1):
            ticket_id = ticket_identifier(ticket)
            if ticket_id in previous:
                records.append(previous[ticket_id])
                continue
            ticket_dir = run_root / "tickets" / safe_dir_name(ticket_id)
            if ticket_dir.exists():
                record = build_record(ticket, skipped="output directory already exists",
                                      output_dir=str(ticket_dir))
            else:
                try:
                    record = run_one_ticket(
                        ticket, pipeline=pipeline,
                        repo_cache_dir=args.repo_cache_dir,
                        snapshot_cache_dir=snapshot_cache_dir,
                        ticket_output_dir=ticket_dir,
                        clone_missing=args.clone_missing,
                        fetch_missing_commits=args.fetch_missing_commits,
                    )
                except KeyboardInterrupt:
                    print("\nInterrupted; records so far are already on disk.", file=sys.stderr)
                    raise
                except Exception:  # noqa: BLE001 - belt and braces around the per-ticket guard
                    record = build_record(ticket, error=traceback.format_exc(limit=3))
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{position}/{len(tickets)}] {ticket_id}: "
                  f"{record['patch_status'] or record['error'] or record['skipped']}"
                  f" (rank={record['target_rank']}, {record['duration_s']}s)")
    finally:
        handle.close()

    summary = summarize(records)
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    count = write_predictions(run_root / "predictions.jsonl", records, run_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRecords: {records_path}")
    print(f"Predictions for the official harness ({count} row(s)): {run_root / 'predictions.jsonl'}")
    return 0 if summary["attempted"] and not summary["errored"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
