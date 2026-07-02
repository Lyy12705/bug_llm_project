# Assignee Triage Accuracy Evaluation

This folder keeps the assignee-triage test data, evaluation scripts, and
generated reports in one place.

The evaluation follows the project plan and the paper
`Automated Bug Triaging using Instruction-Tuned Large Language Models`:

- treat assignee triage as `title + description + component -> assignee`;
- keep history/training rows separate from test rows;
- build the candidate roster from training labels only;
- report Top-1 accuracy, Hit@3, Hit@5, and MRR;
- write a row-level error analysis for follow-up improvements.

## Files

```text
assignee_triage_accuracy/
├── data/
│   ├── controlled_candidate_roster.json
│   ├── controlled_history_train.jsonl
│   └── controlled_test_set.jsonl
├── reports/
├── paper_grade/
├── scripts/
│   ├── build_assignee_profiles.py
│   ├── build_eclipse_sample_dataset.py
│   └── run_assignee_accuracy.py
└── README.md
```

The controlled dataset is designed to test the current baseline behavior:
historical owner lookup, component-owner mapping, partial component matching,
and manual triage fallback.

The current `AssigneeTriager` is a hybrid deterministic ranker. It combines:

- product+component and component historical assignee counts;
- product-level historical counts when product metadata is present;
- BM25-style text similarity over historical bug title/description;
- component-owner mapping as a strong fallback signal;
- manual triage when only weak or global-prior evidence is available.

The Eclipse sample dataset can be generated from:

```text
../../ToJson/dataset/raw/eclipse_issue_sample.csv
```

That CSV has only 100 assignee-labeled rows and is highly imbalanced, so it is
best treated as a small pilot benchmark rather than a final research result.

For a thesis/paper-style experiment, use `paper_grade/`. It contains a
reproducible Mozilla Bugzilla/BMO pipeline with temporal splitting, developer
frequency filtering, multiple baselines, and bootstrap confidence intervals.

## Build Assignee Responsibility Profiles

Public bug-triage datasets usually expose assignee identifiers but not verified
job titles or organization roles. As an alternative, this project can infer an
assignee responsibility profile from the historical tickets handled by each
assignee. These profiles should be described as inferred expertise/responsibility
areas, not real job positions.

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/build_assignee_profiles.py \
  --dataset controlled
```

Generated files:

```text
bug_tracking_llm_system/assignee_triage_accuracy/assignee_profiles/controlled_assignee_profiles.json
bug_tracking_llm_system/assignee_triage_accuracy/assignee_profiles/controlled_assignee_profiles.csv
bug_tracking_llm_system/assignee_triage_accuracy/assignee_profiles/controlled_assignee_profiles.md
```

For paper-grade processed data, pass an explicit train/history file:

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/build_assignee_profiles.py \
  --dataset bmo_smoke \
  --history bug_tracking_llm_system/assignee_triage_accuracy/paper_grade/data/processed/bmo_smoke_history_train.jsonl
```

## Run Controlled Evaluation

From the repository root:

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/run_assignee_accuracy.py \
  --dataset controlled
```

Generated files:

```text
bug_tracking_llm_system/assignee_triage_accuracy/reports/controlled_metrics.json
bug_tracking_llm_system/assignee_triage_accuracy/reports/controlled_predictions.jsonl
bug_tracking_llm_system/assignee_triage_accuracy/reports/controlled_error_analysis.csv
```

## Build And Run Eclipse Sample Evaluation

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/build_eclipse_sample_dataset.py

python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/run_assignee_accuracy.py \
  --dataset eclipse
```

Generated files:

```text
bug_tracking_llm_system/assignee_triage_accuracy/data/eclipse_history_train.jsonl
bug_tracking_llm_system/assignee_triage_accuracy/data/eclipse_test_set.jsonl
bug_tracking_llm_system/assignee_triage_accuracy/data/eclipse_candidate_roster.json
bug_tracking_llm_system/assignee_triage_accuracy/data/eclipse_dataset_summary.json
bug_tracking_llm_system/assignee_triage_accuracy/reports/eclipse_metrics.json
bug_tracking_llm_system/assignee_triage_accuracy/reports/eclipse_predictions.jsonl
bug_tracking_llm_system/assignee_triage_accuracy/reports/eclipse_error_analysis.csv
```
