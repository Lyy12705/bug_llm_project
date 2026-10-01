"""Stage-3 WP4 pilot: single-call, opaque-ID Code Llama symbol reranker.

Per STAGE3_SYMBOL_RERANKER_IMPLEMENTATION_PLAN_ZH.md section 6.3/6.4, WP4
must compare a deterministic Top-10 shortlist against a single Code Llama
call over that same shortlist, using opaque candidate identities so the
prompt carries no retrieval score, rank, Stage-2 file score, or gold.

This intentionally does NOT modify src/utils/fault_localization.py's
existing --symbol-llm-rerank path (batched-by-5, rank-as-id), which is a
known-imperfect prototype documented in the plan (section 0, item 5:
cross-batch scores are not calibrated). WP4 is a separate, stricter pilot
that reuses the already-frozen WP3 b1_coverage_aware_v1 Top-30 pool instead
of recomputing retrieval, and reuses the existing SymbolEvaluationItemV1 /
evaluate_ranked_symbols contract so results are directly comparable to the
WP3 report.

Formal decision recorded 2026-09-08 (see STAGE3 plan "目前進度"): the WP3 G2
gate (Conditional Exact Candidate Recall@30 >= 90%) is NOT met (68.64%).
Per the project's documented pragmatic-limitation policy (the same one used
for Stage-1's Final Holdout Recall@20 gap), this pilot proceeds anyway with
the explicit, reported caveat that end-to-end symbol accuracy is capped by
that candidate-pool ceiling. This is a WP4 pilot/smoke run, not a formal
adoption decision; G5 adoption still requires the full validation in
section 9.1 of the plan.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.llm_client import OllamaClient  # noqa: E402
from utils.merged_ground_truth import evaluation_policy, mapped_symbols
from utils.symbol_evaluation import (  # noqa: E402
    SymbolEvaluationItemV1,
    evaluate_ranked_symbols,
)

SCHEMA_VERSION = "stage3-wp4-pilot-v1"
DEFAULT_CANDIDATE_POOL = (
    "reports/fault_localization/stage3_deterministic_g2_dev_v1/"
    "b1_coverage_aware_v1_predictions.jsonl"
)
DEFAULT_SOURCE_TICKETS = (
    "data/fault_localization/swebench_full/stage3_symbol_gold/"
    "development_source_tickets.jsonl"
)
DEFAULT_GOLD = "data/fault_localization/swebench_full/stage3_symbol_gold/symbol_gold_v1.jsonl"
DEFAULT_INDEX_DIR = "data/fault_localization/swebench_full/indexes"
MAX_CODE_CHARS = 1200
MAX_BUG_REPORT_CHARS = 4000


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_path_for(repo: str, base_commit: str, index_dir: str | Path) -> Path:
    safe_repo = repo.replace("/", "__")
    return Path(index_dir) / f"{safe_repo}__{base_commit[:16]}.json"


def load_chunk_lookup(index_file: Path) -> dict[str, str]:
    with open(index_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    lookup: dict[str, str] = {}
    for chunk in data.get("chunks", []):
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            lookup[chunk_id] = chunk.get("code_text", "")
    return lookup


def gold_items_for_ticket(
    gold_rows: Iterable[dict[str, Any]], ticket_id: str
) -> list[SymbolEvaluationItemV1]:
    items: list[SymbolEvaluationItemV1] = []
    for row in mapped_symbols(gold_rows):
        if row.get("ticket_id") != ticket_id or row.get("mapping_status") != "mapped":
            continue
        items.append(
            SymbolEvaluationItemV1(
                file_path=row["file_path"],
                qualified_name=row["qualified_name"],
                symbol_kind=row["symbol_kind"],
                symbol_id=row.get("symbol_id", ""),
            )
        )
    return items


def candidate_to_item(candidate: dict[str, Any]) -> SymbolEvaluationItemV1:
    return SymbolEvaluationItemV1(
        file_path=candidate["file_path"],
        qualified_name=candidate.get("symbol_qualified_name")
        or candidate.get("symbol_name", ""),
        symbol_kind=candidate.get("symbol_kind", ""),
    )


def is_eligible(row: dict[str, Any], gold_items: Sequence[SymbolEvaluationItemV1]) -> bool:
    if not gold_items:
        return False
    stage2_files = {
        SymbolEvaluationItemV1(
            file_path=f["file_path"] if isinstance(f, dict) else f,
            qualified_name="<module>",
            symbol_kind="module",
        ).file_path
        for f in (row.get("stage2_localized_files") or [])
    }
    return any(item.file_path in stage2_files for item in gold_items)


@dataclass
class ShortlistEntry:
    opaque_id: str
    candidate: dict[str, Any]
    source_index: int


def build_shortlist(candidates: list[dict[str, Any]], k: int, seed: int) -> list[ShortlistEntry]:
    """Take the top-k deterministic candidates, assign SHUFFLED opaque IDs.

    Shuffling (not just relabeling in rank order) matters: the plan requires
    the prompt to carry no positional/rank signal, so opaque_id must not
    correlate with the candidate's deterministic rank.
    """

    top = list(candidates[:k])
    order = list(range(len(top)))
    random.Random(seed).shuffle(order)
    return [
        ShortlistEntry(opaque_id=f"C{position}", candidate=top[source_index], source_index=source_index)
        for position, source_index in enumerate(order, start=1)
    ]


def build_prompt(bug_report: str, entries: list[ShortlistEntry]) -> str:
    rows = []
    for entry in entries:
        candidate = entry.candidate
        rows.append(
            {
                "candidate_id": entry.opaque_id,
                "file_path": candidate["file_path"],
                "symbol_kind": candidate.get("symbol_kind", ""),
                "qualified_name": candidate.get("symbol_qualified_name")
                or candidate.get("symbol_name", ""),
                "start_line": candidate.get("start_line"),
                "end_line": candidate.get("end_line"),
                "code": (candidate.get("code_text") or "")[:MAX_CODE_CHARS],
            }
        )
    return (
        "You are ranking candidate code symbols (functions, methods, classes, "
        "or whole-module gaps) that could contain the root cause of a bug. "
        "You are given the bug report and a shuffled list of candidates, each "
        "with an opaque candidate_id. You do not know their original order, "
        "any retrieval score, or which file ranked higher upstream - judge "
        "each candidate only from the bug report and its code.\n\n"
        "Return JSON only in this shape: "
        '{"rankings":[{"candidate_id":"C1","score":0.0,"reason":"short reason"}]}. '
        "Return every supplied candidate_id exactly once, each score in [0.0, 1.0] "
        "reflecting how likely this candidate contains the bug's root cause, "
        "and a short specific reason grounded in the code.\n\n"
        f"Bug report:\n{bug_report[:MAX_BUG_REPORT_CHARS]}\n\n"
        f"Candidates:\n{json.dumps(rows, ensure_ascii=False, indent=2)}"
    )


def build_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": candidate_ids},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "score", "reason"],
                },
            }
        },
        "required": ["rankings"],
    }


def parse_llm_rankings(
    payload: Any, entries: list[ShortlistEntry]
) -> list[ShortlistEntry] | None:
    """Validate full, unique, finite, in-range coverage; None means fallback."""

    by_id = {entry.opaque_id: entry for entry in entries}
    rows = payload.get("rankings") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    scored: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        candidate_id = row.get("candidate_id")
        if candidate_id not in by_id or candidate_id in scored:
            return None
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            return None
        if score != score or score in (float("inf"), float("-inf")):
            return None
        if not (0.0 <= score <= 1.0):
            return None
        scored[candidate_id] = score
    if set(scored) != set(by_id):
        return None
    ranked_ids = sorted(scored, key=lambda cid: scored[cid], reverse=True)
    return [by_id[cid] for cid in ranked_ids]


@dataclass
class TicketOutcome:
    ticket_id: str
    eligible: bool
    llm_valid: bool
    fallback_reason: str
    baseline_top: list[dict[str, Any]] = field(default_factory=list)
    llm_top: list[dict[str, Any]] = field(default_factory=list)
    baseline_eval: dict[str, Any] | None = None
    llm_eval: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ticket_id": self.ticket_id,
            "eligible": self.eligible,
            "llm_valid": self.llm_valid,
            "fallback_reason": self.fallback_reason,
            "baseline_top": [
                {"file_path": c["file_path"], "qualified_name": c.get("symbol_qualified_name")}
                for c in self.baseline_top
            ],
            "llm_top": [
                {"file_path": c["file_path"], "qualified_name": c.get("symbol_qualified_name")}
                for c in self.llm_top
            ],
            "baseline_eval": self.baseline_eval,
            "llm_eval": self.llm_eval,
        }


def run_ticket(
    row: dict[str, Any],
    bug_report: str,
    gold_items: list[SymbolEvaluationItemV1],
    chunk_lookup: dict[str, str],
    llm_client: Any,
    *,
    shortlist_k: int,
    top_k: int,
    seed: int,
) -> TicketOutcome:
    candidates = list(row.get("stage3_candidate_symbols") or [])
    for candidate in candidates:
        if not candidate.get("code_text"):
            candidate["code_text"] = chunk_lookup.get(candidate.get("chunk_id", ""), "")

    eligible = is_eligible(row, gold_items)

    baseline_top = candidates[:top_k]
    baseline_eval = (
        evaluate_ranked_symbols([candidate_to_item(c) for c in baseline_top], gold_items)
        if gold_items
        else None
    )

    entries = build_shortlist(candidates, shortlist_k, seed)
    prompt = build_prompt(bug_report, entries)
    schema = build_schema([entry.opaque_id for entry in entries])

    llm_valid = False
    fallback_reason = ""
    # Fallback must be the deterministic baseline order, not the shuffled
    # prompt order used to strip positional signal from the LLM call.
    ranked_entries = sorted(entries, key=lambda entry: entry.source_index)
    payload = None
    try:
        payload = llm_client.generate_json_with_schema(prompt, schema)
    except Exception as exc:  # noqa: BLE001 - any transport/timeout failure is a fallback
        fallback_reason = f"llm_error:{exc}"

    if payload is not None:
        parsed = parse_llm_rankings(payload, entries)
        if parsed is None:
            fallback_reason = fallback_reason or "invalid_or_incomplete_output"
        else:
            ranked_entries = parsed
            llm_valid = True

    llm_top = [entry.candidate for entry in ranked_entries[:top_k]]
    llm_eval = (
        evaluate_ranked_symbols([candidate_to_item(c) for c in llm_top], gold_items)
        if gold_items
        else None
    )

    return TicketOutcome(
        ticket_id=row["ticket_id"],
        eligible=eligible,
        llm_valid=llm_valid,
        fallback_reason=fallback_reason,
        baseline_top=baseline_top,
        llm_top=llm_top,
        baseline_eval=baseline_eval,
        llm_eval=llm_eval,
    )


def summarize(outcomes: list[TicketOutcome]) -> dict[str, Any]:
    eligible = [o for o in outcomes if o.eligible]
    valid = [o for o in eligible if o.llm_valid]

    def avg_hit(items: list[TicketOutcome], field_name: str, k: str) -> float:
        values = [
            getattr(o, field_name)["exact"]["hit_at"][k]
            for o in items
            if getattr(o, field_name) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "ticket_count": len(outcomes),
        "eligible_count": len(eligible),
        "llm_valid_count": len(valid),
        "llm_fallback_count": len(eligible) - len(valid),
        "llm_fallback_rate": (
            (len(eligible) - len(valid)) / len(eligible) if eligible else 0.0
        ),
        "baseline_exact_hit_at": {
            k: avg_hit(eligible, "baseline_eval", k) for k in ("1", "3", "5")
        },
        "llm_exact_hit_at": {
            k: avg_hit(eligible, "llm_eval", k) for k in ("1", "3", "5")
        },
        "known_limitation": (
            "Deterministic candidate pool (b1_coverage_aware_v1) Conditional "
            "Exact Candidate Recall@30 is 68.64%, below the WP3 G2 gate "
            "(>=90%). End-to-end Hit@K here is capped by that pool ceiling; "
            "this run does not by itself demonstrate G2 or G5 adoption."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--source-tickets", default=DEFAULT_SOURCE_TICKETS)
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--allow-legacy-gold", action="store_true")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--ticket-ids", default="", help="Comma-separated ticket_id filter.")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit.")
    parser.add_argument("--eligible-only", action="store_true", default=True)
    parser.add_argument("--include-ineligible", dest="eligible_only", action="store_false")
    parser.add_argument("--shortlist-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    pool_rows = {row["ticket_id"]: row for row in load_jsonl(args.candidate_pool)}
    source_rows = {row["ticket_id"]: row for row in load_jsonl(args.source_tickets)}
    gold_rows = load_jsonl(args.gold)
    gold_policy = evaluation_policy(gold_rows, allow_legacy=args.allow_legacy_gold)

    ticket_ids = list(pool_rows.keys())
    if args.ticket_ids:
        wanted = {t.strip() for t in args.ticket_ids.split(",") if t.strip()}
        ticket_ids = [t for t in ticket_ids if t in wanted]

    llm_client = OllamaClient(
        url=args.ollama_url, model=args.ollama_model, timeout=args.ollama_timeout
    )

    chunk_cache: dict[Path, dict[str, str]] = {}
    outcomes: list[TicketOutcome] = []
    for ticket_id in ticket_ids:
        row = pool_rows[ticket_id]
        gold_items = gold_items_for_ticket(gold_rows, ticket_id)
        if args.eligible_only and not is_eligible(row, gold_items):
            continue

        source_row = source_rows.get(ticket_id, {})
        bug_report = str(source_row.get("bug_report") or source_row.get("description") or "")

        index_file = index_path_for(row["repo"], row["base_commit"], args.index_dir)
        if index_file not in chunk_cache:
            chunk_cache[index_file] = (
                load_chunk_lookup(index_file) if index_file.exists() else {}
            )

        outcome = run_ticket(
            row,
            bug_report,
            gold_items,
            chunk_cache[index_file],
            llm_client,
            shortlist_k=args.shortlist_k,
            top_k=args.top_k,
            seed=args.seed,
        )
        outcomes.append(outcome)
        print(
            f"[{len(outcomes)}] {ticket_id}: eligible={outcome.eligible} "
            f"llm_valid={outcome.llm_valid} fallback={outcome.fallback_reason or '-'}",
            file=sys.stderr,
        )
        if args.limit and len(outcomes) >= args.limit:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize(outcomes)
    summary["ground_truth"] = gold_policy
    summary["run_manifest"] = {
        "model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "ollama_timeout": args.ollama_timeout,
        "shortlist_k": args.shortlist_k,
        "top_k": args.top_k,
        "seed": args.seed,
        "candidate_pool": args.candidate_pool,
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
