# Paper-Grade Assignee Triage Benchmark

This folder upgrades the assignee-triage evaluation from a small functional
test into a reproducible benchmark suitable for a thesis or paper experiment.

The workflow follows common bug-triage papers:

- collect public Bugzilla/BMO issues with assignee labels;
- normalize bug summary/title and metadata as the bug-report text;
- sort by creation time and use temporal train/validation/test splits;
- build the candidate roster from training labels only;
- remove low-frequency developers using a minimum training count;
- optionally anonymize developer identifiers;
- compare the current project baseline against simple research baselines;
- report Top-1, Hit@K, MRR, macro accuracy, and bootstrap confidence intervals.

The separate `bug-duplicate-detection` feature remains responsible for online
duplicate classification before assignee triage. Any exact-link/content
handling in this benchmark is offline train/test leakage prevention only; it
is not called by `AssigneeTriager` and does not form part of routing.

## Folder Layout

```text
paper_grade/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── reports/
└── scripts/
    ├── evaluate_paper_grade_benchmarks.py
    ├── fetch_bmo_assignee_dataset.py
    ├── prepare_paper_grade_dataset.py
    └── run_bmo_paper_experiment.py
```

## One-Command Paper Experiment

Run the full pipeline with a larger BMO sample:

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/scripts/run_bmo_paper_experiment.py \
  --dataset bmo_paper_2024 \
  --products Core Firefox \
  --start-date 2024-01-01 \
  --max-bugs 3000 \
  --min-train-assignee-count 10 \
  --no-include-comments \
  --insecure
```

This performs:

- public BMO fetch;
- temporal train/validation/test split;
- train-only assignee responsibility profile construction;
- baseline evaluation;
- run manifest generation.

## 1. Fetch Public BMO Data

Start with a small smoke run:

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/scripts/fetch_bmo_assignee_dataset.py \
  --products Core Firefox \
  --start-date 2020-01-01 \
  --max-bugs 500 \
  --include-comments
```

If the local Python installation cannot verify the system CA bundle, add
`--insecure` for local data collection only. The default remains normal TLS
verification.

For the full experiment, increase `--max-bugs` or remove it, and choose a fixed
date window in the thesis text.

Output:

```text
bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/data/raw/bmo_bugs_raw.jsonl
bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/data/raw/bmo_fetch_manifest.json
```

## 2. Prepare Temporal Splits

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/scripts/prepare_paper_grade_dataset.py \
  --dataset bmo_paper \
  --min-train-assignee-count 10
```

Output:

```text
data/processed/bmo_paper_history_train.jsonl
data/processed/bmo_paper_validation_set.jsonl
data/processed/bmo_paper_test_set.jsonl
data/processed/bmo_paper_candidate_roster.json
data/processed/bmo_paper_dataset_summary.json
```

By default the prepared files anonymize assignees as `dev_0001`,
`dev_0002`, etc. Use `--keep-assignee-identifiers` only for private local
debugging.

## 3. Evaluate Baselines

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/scripts/evaluate_paper_grade_benchmarks.py \
  --dataset bmo_paper
```

Output:

```text
reports/bmo_paper_metrics.json
```

Add `--write-details` to also generate row-level predictions and CSV error
analysis. These detail files are reproducible and intentionally not kept in
the repository.

## Smoke Run Already Verified

The current folder includes a small BMO smoke sample fetched from public Mozilla
Bugzilla records:

```text
data/raw/bmo_bugs_raw.jsonl
data/raw/bmo_fetch_manifest.json
data/processed/bmo_smoke_*.jsonl
reports/bmo_smoke_metrics.json
```

This smoke sample is intentionally small and should not be reported as a final
paper result. It only verifies that fetching, temporal splitting, anonymization,
baseline evaluation, and report generation work end to end.

## Baselines

- `global_majority`: recommends the most frequent training assignees.
- `component_majority`: recommends assignees by historical component counts.
- `product_component_majority`: recommends by product+component, then
  component, product, and global fallback.
- `bm25_text_knn`: retrieves similar historical bug reports by BM25-like text
  matching and ranks their assignees.
- `current_assignee_triager`: calls the existing project implementation
  (`AssigneeTriager`) without modifying the production module.

The current project implementation is a hybrid deterministic ranker combining
metadata frequency signals, BM25-style text similarity, component-owner mapping,
and conservative manual-triage fallback.

## Reporting Notes

For a paper-grade result, report:

- dataset source and frozen date range;
- number of raw, filtered, train, validation, and test bugs;
- assignee filtering threshold;
- candidate roster size;
- Top-1, Hit@3, Hit@5, Hit@10, MRR;
- macro Top-1 by assignee and component;
- bootstrap 95% confidence intervals for Top-1;
- error analysis for cold-start components and long-tail assignees.
