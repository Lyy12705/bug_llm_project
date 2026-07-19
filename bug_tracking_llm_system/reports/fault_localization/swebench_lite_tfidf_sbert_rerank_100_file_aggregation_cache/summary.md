# SWE-bench Lite Fault Localization 100-Ticket Summary

## Setup

- Dataset: SWE-bench Lite `test` split
- Subset size: first 100 tickets
- Repositories covered: `astropy/astropy`, `django/django`
- Ground truth: file-level fixed files parsed from developer patches
- Method: TF-IDF candidate pool + SBERT rerank + file-level aggregation
- SBERT cache: `data/fault_localization/swebench_lite/embedding_cache`
- LLM rerank: disabled

## Results

| Metric | Value |
|---|---:|
| Predictions | 100 |
| Failures | 0 |
| Top-1 Accuracy | 0.6000 |
| Top-3 Accuracy | 0.8000 |
| Top-5 Accuracy | 0.8700 |
| MRR | 0.7127 |
| Top-5 Misses | 13 |

## Cache Result

The run created a persistent SQLite SBERT embedding cache at:

`data/fault_localization/swebench_lite/embedding_cache/sentence-transformers_all-MiniLM-L6-v2.sqlite3`

After this run, the cache contains 26,821 embedding rows. This does not change the
ranking formula; it reduces repeated embedding cost for reruns and larger
experiments.

## Interpretation

The 100-ticket result is stable enough for a development experiment. Top-5
accuracy improves from the earlier 30-ticket TF-IDF file-aggregation baseline
(`0.7000`) to `0.8700`, showing that the SBERT rerank stage improves semantic
candidate recall. Top-1 accuracy is lower than the 30-ticket SBERT subset
(`0.6667` to `0.6000`), which suggests the larger sample contains harder ranking
cases, but Top-3 and Top-5 remain strong.

Failure analysis shows all 13 Top-5 misses are `ranking_error_gold_indexed`: the
gold files are present in the code index, but the ranking did not move them into
the first five file candidates. The next improvement should therefore focus on
ranking quality rather than repository indexing.

## Generated Files

- `test_predictions.jsonl`: 100 localization predictions
- `test_metrics.json`: Top-k/MRR metrics
- `test_demo_cases.json`: demo-ready example cases
- `test_failures.jsonl`: runtime failures, empty for this run
- `test_failure_analysis.json`: detailed miss analysis
- `test_failure_analysis.csv`: tabular miss analysis
