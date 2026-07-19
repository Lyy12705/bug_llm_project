# Fault Localization Stage-5 Strengthening Summary

## Flow Diagram Text

Raw ticket
-> structured ticket extraction
-> repository code index
-> retrieval-first candidate ranking
-> file-level aggregation
-> lightweight repository context
-> optional LLM rerank with fallback
-> uncertainty gate
-> Top-k suspicious files
-> patch suggestion or manual review

## Current Completeness

The fault localization feature has moved from the earlier stage-4 integration
state toward a stage-5 demo/report-ready state. The core retrieval pipeline was
already runnable; this strengthening adds confidence labeling, safer patch
handoff, a consistent demo case, and fallback analysis artifacts.

Estimated completeness after this pass: about 86%. This is not a claim that the
model is production-ready. It means the feature is now easier to demonstrate,
explain, and evaluate in a project report.

## 300-Ticket Full-Split Development Result

Development result source:
`reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/test_metrics.json`

| Metric | Value |
|---|---:|
| Evaluated tickets | 300 |
| Runtime failures | 0 |
| Top-1 accuracy | 0.5500 |
| Top-3 accuracy | 0.7567 |
| Top-5 accuracy | 0.8133 |
| MRR | 0.6559 |

These are file-level metrics. Symbol-level ground truth is not available in the
prepared SWE-bench Lite split, so symbol metrics should not be emphasized.

## Retrieval-First Design Rationale

The system keeps retrieval as the primary architecture because it is bounded,
repeatable, and evaluable. It avoids sending an entire repository to an LLM and
instead builds a compact Top-k context from ranked code chunks, file-level
aggregation, import context, and symbol references.

This also follows the practical lesson from long-context work: more context is
not automatically better. A retrieval-first design reduces the chance that the
model misses important evidence in the middle of a long prompt.

## LLM Rerank Limitation

LLM rerank remains optional. The current evidence shows it can be useful as a
small reranking layer, but it is still unstable enough that it should not be a
hard dependency. The score-guarded Top-5 miss experiment contains 7 fallback
warnings across 10 prediction rows. Most are schema invalid responses, with one
timeout. This supports the report conclusion that the main improvement is
stability, controllability, and demo-readiness rather than a proven large LLM
accuracy gain.

## Uncertainty Gate Purpose

The uncertainty gate converts ranking evidence into a human-facing decision:

| Level | Meaning | Patch handoff |
|---|---|---|
| High | Top result has strong score plus direct path/stack evidence or clear margin. | Can be used as the first patch suggestion target. |
| Medium | Candidate is plausible but evidence is not strong enough for automatic patch generation. | Manual review before patch generation. |
| Low | Ranking evidence is weak, ambiguous, or missing. | Do not pass directly to patch generation. |

The gate uses existing signals only: Top-1 score, Top-1/Top-2 margin, stack
trace score, path hint score, identifier/domain/expansion signals, and whether
optional LLM rerank fell back to retrieval.

## Failure Cases And Limits

- The 300-ticket full-split development run still has 56 Top-5 misses.
- Failure analysis indicates the gold files were indexed but not ranked high
  enough, so remaining work is mainly ranking quality.
- LLM rerank can fail through invalid schema, invalid ranks, timeout, service
  availability, echo-like output, or empty JSON.
- Patch generation is now intentionally conservative: low-confidence
  localization produces manual review guidance rather than blind multi-file
  modification.

## Next Work

1. Analyze the 56 development-run Top-5 misses by repo and failure pattern.
2. Shorten and harden the LLM rerank prompt before running another controlled
   comparison.
3. Connect the demo output to a small report figure/table showing ticket,
   Top-k candidates, confidence, and patch handoff policy.
