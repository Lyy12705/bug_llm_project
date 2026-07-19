# Fault Localization Next-Step Implementation Summary

## Scope

This pass implements the two immediate next steps after the stage-5
strengthening work:

1. Analyze the 56 Top-5 misses from the 300-ticket development run.
2. Shorten the optional LLM rerank prompt and make its structured output
   candidate_id-only.

The retrieval-first architecture remains unchanged. LLM rerank is still
optional and guarded by retrieval fallback.

## 56 Top-5 Miss Analysis

Source prediction:
`reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/test_predictions.jsonl`

Generated artifacts:

- `reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/top5_miss_pattern_analysis.json`
- `reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/top5_miss_pattern_analysis.csv`
- `reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/top5_miss_pattern_analysis.md`

| Metric | Value |
|---|---:|
| Evaluated predictions | 300 |
| Top-5 misses | 56 |
| Indexed misses | 56 |
| Unindexed misses | 0 |
| Top-5 accuracy | 0.8133 |

## Main Miss Patterns

| Pattern | Count | Meaning |
|---|---:|---|
| same_top_level_package_confusion | 27 | The wrong Top-1 file is broadly in the same package family. |
| near_miss_same_subpackage | 24 | The wrong Top-1 file is very close to the gold file path. |
| generated_or_migration_noise | 2 | Migration/generated-like files crowd the candidate list. |
| general_ranking_error_gold_indexed | 3 | Gold file is indexed, but ranking signals still miss it. |

Interpretation: the remaining errors are mostly local ranking confusion, not
index coverage failures. The next accuracy work should therefore focus on
within-package reranking and stronger disambiguation between nearby files.

## LLM Rerank Prompt Change

The optional LLM rerank prompt is now shorter and asks for only candidate IDs:

```json
[{"candidate_id":"C1","confidence":0.0,"reason":"short reason"}]
```

The prompt explicitly forbids `rank`, `file_path`, `code_excerpt`,
`repository_context`, `retrieval_score`, and `retrieval_signals` in the output.

The parser now requires a valid `candidate_id` when candidate references are
available. Rank-only rows are rejected and fall back to retrieval. The cache key
was versioned to avoid mixing old rerank responses with the new output schema.

## Expected Effect

This change is primarily for stability and controllability. It should reduce
schema drift and echoed candidate rows, but it is not yet evidence of a formal
accuracy improvement. A new controlled LLM rerank experiment is still needed
before reporting LLM rerank as an accuracy gain.
