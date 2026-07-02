# BMO Paper Experiment Summary: bmo_paper_2024_3k

This report summarizes the paper-grade Mozilla Bugzilla/BMO assignee-triage
experiment generated on the local workspace.

## Dataset

- Source: Mozilla Bugzilla REST API
- Products: `Core`, `Firefox`
- Date range in collected raw data: `2024-01-01T00:34:44Z` to `2024-03-15T18:52:53Z`
- Raw fetch cap: `3000`
- Usable rows before assignee-frequency filtering: `2800`
- Minimum train assignee count: `10`
- Candidate roster size: `60`
- Assignees anonymized: `true`
- Text fields in this run: bug summary/title plus metadata. First comments were not included in this completed 3k run.

## Temporal Split

| Split | Rows | Purpose |
|---|---:|---|
| Train/history | 1569 | Build historical counts, BM25 index, and assignee responsibility profiles |
| Validation | 168 | Tuning / ablation check |
| Test | 192 | Final held-out evaluation |

## Test Results

| Baseline | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR | Macro Top-1 Assignee | Macro Top-1 Component | Top-1 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| global_majority | 0.385417 | 0.479167 | 0.489583 | 0.536458 | 0.434921 | 0.025000 | 0.280878 | [0.322917, 0.458333] |
| component_majority | 0.645833 | 0.869792 | 0.911458 | 0.927083 | 0.757378 | 0.461813 | 0.547991 | [0.578125, 0.708333] |
| product_component_majority | 0.645833 | 0.869792 | 0.906250 | 0.942708 | 0.758941 | 0.461813 | 0.547991 | [0.578125, 0.708333] |
| bm25_text_knn | 0.713542 | 0.869792 | 0.932292 | 0.963542 | 0.801893 | 0.507286 | 0.651414 | [0.645833, 0.781250] |
| current_assignee_triager | 0.765625 | 0.906250 | 0.937500 | 0.958333 | 0.838777 | 0.615539 | 0.704167 | [0.697917, 0.833333] |

## Validation Results

| Baseline | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR | Macro Top-1 Assignee | Macro Top-1 Component | Top-1 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| global_majority | 0.214286 | 0.303571 | 0.333333 | 0.392857 | 0.268185 | 0.023810 | 0.224537 | [0.154762, 0.273810] |
| component_majority | 0.541667 | 0.785714 | 0.815476 | 0.863095 | 0.668403 | 0.445448 | 0.500587 | [0.464286, 0.607143] |
| product_component_majority | 0.541667 | 0.785714 | 0.815476 | 0.922619 | 0.675765 | 0.445448 | 0.500587 | [0.464286, 0.607143] |
| bm25_text_knn | 0.601190 | 0.803571 | 0.875000 | 0.940476 | 0.715663 | 0.424718 | 0.547910 | [0.535714, 0.666667] |
| current_assignee_triager | 0.619048 | 0.833333 | 0.886905 | 0.904762 | 0.733241 | 0.490336 | 0.586383 | [0.547619, 0.684524] |

## Interpretation

The strongest held-out test result is `current_assignee_triager`, the deterministic hybrid ranker. It improves Top-1 accuracy over BM25 text KNN by `0.052083` and over component-majority baselines by `0.119792`.

Train-only assignee responsibility profiles remain useful as an analysis artifact, but they are not part of the selected main model.

Recommended main thesis model for this run: `current_assignee_triager`.
