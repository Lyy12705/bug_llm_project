# Fault Localization User Guide

This guide describes the current internal-beta fault localization workflow. The feature is designed to help engineers narrow the debugging area; it does not replace engineer review.

## Input

Use a JSON ticket with as much concrete evidence as possible:

| Field | Why it helps |
|---|---|
| `title` / `description` | Main retrieval query. |
| `logs` / stack trace | Strong evidence for files and line ranges. |
| `error_message` | Helps match exception behavior. |
| `steps_to_reproduce` | Adds workflow and domain terms. |
| `component` / `repo` | Helps disambiguate broad tickets. |
| `fail_to_pass` | Useful when failing test paths hint at source modules. |

Short or vague tickets still run only when the input passes validation, but they should be treated as low-confidence review guidance.

## CLI

Raw evaluation output:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --ticket demo/data/fault_localization_demo_ticket.json \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf
```

User-facing beta output:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --tickets-jsonl demo/data/fault_localization_demo_cases.jsonl \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --output-format user-facing \
  --index-cache-dir reports/fault_localization/index_cache \
  --progress text \
  --progress-file reports/fault_localization/demo_progress.jsonl \
  --output reports/fault_localization/demo_user_facing_localization.jsonl
```

Internal beta precheck:

```bash
PYTHONPATH=src python3 scripts/run_fault_localization_internal_beta.py \
  --output-dir reports/fault_localization/internal_beta_20
```

Pollable job mode for API or UI demos:

```bash
PYTHONPATH=src python3 scripts/fault_localization_job.py start \
  --ticket demo/data/fault_localization_demo_ticket.json \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --jobs-dir reports/fault_localization/jobs \
  --index-cache-dir reports/fault_localization/index_cache
```

Then inspect the job:

```bash
PYTHONPATH=src python3 scripts/fault_localization_job.py status \
  --job-dir reports/fault_localization/jobs/<job_id>
```

Use `--foreground` with `start` for smoke tests or classroom demos where the command should wait for completion while still writing `status.json`, `result.json`, and `progress.jsonl`.

## Output Fields

| Field | Meaning |
|---|---|
| `status` | User-facing result status such as `ready_for_engineer_review`, `needs_manual_review`, `low_confidence_manual_review`, or `invalid_input`. |
| `summary.confidence_level` | High / medium / low confidence gate. |
| `summary.should_manual_review` | Whether engineer review is required before patching. |
| `summary.recommend_patch_generation` | Whether patch suggestion may proceed. |
| `summary.fallback_message` | Explains retrieval-only or LLM fallback behavior when present. |
| `top_k_suspicious_files` | Ranked files with score, reason, key signals, and compact repository context. |
| `input_validation` | Ticket and repository quality checks. |
| `engineering_guidance` | Plain-language next action for engineers. |
| `runtime` | Index cache status and per-ticket runtime. |

## Confidence Levels

| Level | Intended use |
|---|---|
| High | Inspect Top-1 first; patch suggestion may be allowed, but still keep changes minimal. |
| Medium | Use Top-k as review guidance; do not generate a patch before manual confirmation. |
| Low | Manual review is required; do not hand directly to patch generation. |

## Patch Handoff Policy

Patch generation must respect the localization gate:

| Condition | Policy |
|---|---|
| High confidence | Patch suggestion may target the top-ranked file. |
| Medium confidence | Keep Top-k context, but require manual review first. |
| Low confidence | Block patch generation and ask for manual inspection. |
| LLM fallback or invalid output | Treat retrieval-first result as guidance only. |

## Current Limitations

- Top-k localization narrows the search area; it is not a root-cause guarantee.
- LLM rerank is optional and should not be required for the beta path.
- The internal beta run is a deployment-readiness check, not a full production evaluation.
- Public deployment still needs API authentication, job queue hardening, runtime limits, repo isolation, and stronger observability.
- Long-running repositories should use `--index-cache-dir` plus progress polling; a blocking synchronous UI is not recommended.
