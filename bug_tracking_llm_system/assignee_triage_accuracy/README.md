# Assignee Triage Accuracy Evaluation

This folder keeps the assignee-triage test data, evaluation scripts, and
generated reports in one place. JSON/JSONL is the source-of-truth format;
CSV and generated Markdown are optional human-readable exports.

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
├── phase1_reliability/ ... phase7_rolling_open_set/
├── scripts/
│   ├── build_assignee_profiles.py
│   ├── build_eclipse_sample_dataset.py
│   └── run_assignee_accuracy.py
└── README.md
```

Repository hygiene rules:

- keep source datasets, normalized datasets, metrics, manifests, reviewed
  deployment artifacts, and reproducible scripts;
- do not commit caches, row-level predictions, repeated temporal-window
  copies, or CSV/Markdown views that can be regenerated;
- `run_assignee_accuracy.py` and the paper-grade evaluator write compact
  metrics by default; pass `--write-details` only when row-level analysis is
  needed;
- `build_assignee_profiles.py` writes canonical JSON by default; pass
  `--write-human-exports` when CSV and Markdown views are needed temporarily.

## Feature Boundary

Duplicate detection is not part of assignee triage. The upstream
`bug-duplicate-detection` feature must process each incoming ticket first. If
it is a duplicate, the pipeline stops there; only non-duplicate tickets are
passed to `AssigneeTriager`. This folder therefore does not run a duplicate
classifier or use duplicate candidates when choosing an assignee.

Offline dataset preparation may still count exact ticket-ID or content overlap
between train/validation/test splits. That is a fail-closed leakage check, not
duplicate detection, and it never changes an online routing decision.

The controlled dataset is designed to test the current baseline behavior:
historical owner lookup, component-owner mapping, partial component matching,
and manual triage fallback.

The evaluation reports Top-1, Top-3, Top-5, MRR, and Macro-F1 over developer
recommendation rows only; `manual_triage` is not treated as a developer label.
Production abstention is evaluated separately with auto-assignment coverage,
auto-assignment accuracy, unable-to-decide rate, fallback trigger rate, and
manual-triage rate. Candidate metrics use exactly the production candidate
list and do not add history/global candidates after prediction. Class
imbalance and exact cross-split leakage checks are included in the same report.

The current `AssigneeTriager` is a hybrid deterministic ranker. It combines:

- product+component and component historical assignee counts;
- product-level historical counts when product metadata is present;
- BM25-style text similarity over historical bug title/description;
- component-owner mapping as a strong fallback signal;
- manual triage when only weak or global-prior evidence is available.

This is the production baseline by design. The reference instruction-tuned
LLM paper reports useful shortlists but very low exact Top-1 on Mozilla
(`0.013` Top-1 versus `0.753` Hit@10). Its DeepSeek-R1-Distill-Llama-8B + LoRA
method therefore remains an offline comparison in this project, not an
autonomous assignment dependency. CodeBERT is not added to assignee routing without file-level
training labels and a temporal offline gain.

Production output distinguishes automatic assignment from safe routing:
`routing_status`, `needs_manual_triage`, `fallback_used`, `fallback_reason`,
`suggested_assignee`, and `candidate_details` are emitted alongside the legacy
`assignee`, `confidence`, and `ranked_candidates` fields. Text-only matches are
kept as suggestions by default instead of being auto-assigned.

Production routing can now be constrained with JSON active rosters,
inactive/departed assignee lists, component ownership maps, file/module
ownership maps, calibration routing policy thresholds, and an optional
rule-based open-set risk gate. The corresponding CLI flags are available on
both `src/main.py` and `scripts/run_assignee_accuracy.py`.

No placeholder component owners are enabled by default. Ownership must come
from the target project's mapping files or explicit configuration. When an
active-roster path is configured but missing, empty, or invalid, routing fails
closed to `manual_triage`. Component and file rules use token/path boundaries
rather than arbitrary substring matching. Roster and ownership files are
cached by path and modification time.

Historical/text ranking remains available as Top-k recommendations, but it is
not auto-assigned by default until an approved calibration artifact is applied.
Explicit component/file ownership may still auto-assign because it is an
authoritative project policy rather than a learned probability. The
`--allow-uncalibrated-auto-assignment` evaluation flag exists only for offline
comparisons and should not be enabled in production.

Human triage outcomes can be recorded as JSONL `assignee_feedback` events via
`src/modules/assignee_feedback.py`. Confirmed final assignees can be converted
back into history rows for later owner-profile refresh without introducing a
new database dependency.

## Reviewed Runtime Artifacts And Feedback Refresh

Phase 3 now exports `assignee_confidence_calibrator.json`. It is a portable
isotonic mapping rather than a pickle, so the production ranker has no
scikit-learn runtime dependency. Phase 4 exports
`assignee_open_set_rule_based_novelty.json`; learned detector artifacts are not
exported for runtime because their frozen-test behavior was not stable enough.
Both artifacts are `research_only` by default. The runtime rejects them unless
their `deployment_status` is explicitly `approved` and their
`ranker_confidence_version` matches the current ranker.

```bash
python3 assignee_triage_accuracy/scripts/evaluate_assignee_calibration.py \
  --dataset bmo_paper_2024_3k --output-dir /tmp/assignee_phase3

python3 assignee_triage_accuracy/scripts/evaluate_assignee_open_set_detector.py \
  --dataset bmo_paper_2024_3k --output-dir /tmp/assignee_phase4
```

Only approve artifacts after repeating the temporal evaluation on the target
project. Supply approved artifacts with `--assignee-calibration-artifact` and
`--assignee-open-set --assignee-open-set-artifact`; responses expose raw versus
calibrated confidence plus the artifact status for auditability.

Confirmed human decisions can refresh a history snapshot without changing the
original source data. `--as-of` filters by feedback confirmation time and
`--exclude-tickets` keeps a held-out split out of the refreshed profile.

```bash
python3 assignee_triage_accuracy/scripts/merge_assignee_feedback.py \
  --history data/assignee_dataset.jsonl \
  --feedback data/assignee_feedback.jsonl \
  --exclude-tickets assignee_triage_accuracy/data/controlled_test_set.jsonl \
  --as-of 2026-07-10T00:00:00+00:00 \
  --output data/assignee_history_refreshed.jsonl
```

For online routing, `--assignee-feedback` can be used directly. When incoming
tickets include `created_at`/`creation_time`, feedback confirmed after that
timestamp is excluded to prevent temporal label leakage. Base history rows
with later timestamps are also excluded; rows without timestamps remain usable
but are counted in `profile_stats.history_rows_without_timestamp` so the
remaining leakage risk is visible.

The Eclipse sample dataset can be generated from:

```text
../../ToJson/dataset/raw/eclipse_issue_sample.csv
```

That CSV has only 100 assignee-labeled rows and is highly imbalanced, so it is
best treated as a small pilot benchmark rather than a final research result.

For a thesis/paper-style experiment, use `paper_grade/`. It contains a
reproducible Mozilla Bugzilla/BMO pipeline with temporal splitting, developer
frequency filtering, multiple baselines, and bootstrap confidence intervals.

## Train The Lightweight Candidate Ranker

`train_assignee_ltr.py` is the practical successor experiment to the fixed
Hybrid weights. It keeps a Top-30 source-quota candidate pool and learns only
the ordering of those candidates. Product-component, component, BM25, local
SBERT, recent-component, product, and global-prior sources receive explicit
candidate quotas. Training examples come from expanding temporal
folds inside the train split, so a query never uses future assignee history.

The compared models are scikit-learn logistic regression and histogram
gradient boosting. Features are inspectable: smoothed component ownership,
product-component ownership, BM25 evidence, 30/90-day activity, exponential
time decay, owner recency, and historical support. Long-tail queries receive
`1 / sqrt(assignee_frequency)` weight. The description cleaner removes repeated
boilerplate and limits stack-trace dominance while retaining error text.

```bash
python3 assignee_triage_accuracy/scripts/train_assignee_ltr.py \
  --temporal-folds 3
```

SBERT is an optional local-only ablation. The command never downloads a model;
if the configured model is unavailable locally, the report records that fact
and continues without semantic features.

```bash
python3 assignee_triage_accuracy/scripts/train_assignee_ltr.py \
  --temporal-folds 3 \
  --embedding-backend sbert \
  --output-dir assignee_triage_accuracy/phase6_candidate_ltr/reports/bmo_public_10k_sbert
```

Artifacts remain `research_only`, even when the validation promotion gate
passes. A new, later, untouched temporal holdout is required before the model
may replace the runtime Hybrid ranker. This prevents an already-inspected test
split from becoming an accidental tuning set.

After fixing a ranker on a development window, run post-hoc calibration. The
script splits validation chronologically into calibrator-fit and
threshold-selection portions, then reports holdout auto-assignment accuracy,
coverage, and unseen-owner false-auto-assignment rate. It never approves an
artifact automatically.

```bash
python3 assignee_triage_accuracy/scripts/calibrate_assignee_ltr.py
```

For open-world routing, build leakage-audited leave-assignee-out folds, train
the lightweight open-set detector on validation, and select policy thresholds
on Q3 development only:

```bash
python3 assignee_triage_accuracy/scripts/build_open_set_folds.py
python3 assignee_triage_accuracy/scripts/train_assignee_open_set.py
```

Evaluate a frozen bundle on a genuinely later holdout only once:

```bash
python3 assignee_triage_accuracy/scripts/evaluate_assignee_open_set_holdout.py \
  --holdout path/to/new_future_holdout.jsonl \
  --output-dir path/to/holdout_report \
  --confirm-untouched-holdout
```

The corrected shallow-ranker workflow was evaluated once on a new 2021 Q2
holdout. It failed the deployment gate because the frozen policy became
over-conservative and auto-assignment coverage fell to zero.
The artifact remains `research_only`; production routing continues to use the
existing fail-closed Hybrid path.

## Rolling History And Stable Open-World Routing

Phase 7 fixes the static-history limitation without introducing a large or
unverifiable model. It keeps the shallow candidate-level HistGradientBoosting
ranker and local SBERT feature from Phase 6, then adds:

- rolling owner profiles built only from earlier windows;
- conservative label availability based on `last_change_time`;
- portable logistic confidence calibration;
- leave-assignee-out training for a lightweight open-set detector;
- one policy that must pass every Q1-Q3 selection window;
- separate automatic-assignment and Top-3 confirmation thresholds;
- PSI score-distribution drift detection and fail-closed routing.

```bash
python3 assignee_triage_accuracy/scripts/train_assignee_rolling_open_set.py

python3 assignee_triage_accuracy/scripts/evaluate_assignee_rolling_holdout.py \
  --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl \
  --holdout-name 2021_q4 \
  --output-dir assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_public_2021q4_conservative_untouched_holdout \
  --confirm-untouched-holdout
```

The frozen Q4 holdout has 2,779 evaluated rows. Automatic assignment reached
`97.75%` accuracy at `24.04%` coverage, with zero false automatic assignments
among 141 unseen-owner rows. Another `21.05%` of rows were sent to Top-3
confirmation at `71.11%` Top-3 accuracy; `54.91%` remained manual. Maximum PSI
was `0.052`, below the `0.25` drift limit. Known-owner Top-1 was still only
`48.41%`, so this is a selective routing improvement rather than evidence that
all issues can be assigned automatically.

The bundle deliberately remains `research_only`. Public BMO snapshots do not
provide a reviewed active/departed roster, verified ownership policy, or exact
assignment-change timestamps. Runtime integration therefore still requires a
target-project roster, operator approval, and target-project temporal
validation. See `phase7_rolling_open_set/README.md` for the protocol and
limitations.

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
```

Add `--write-human-exports` to also generate CSV and Markdown views.

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
```

Add `--write-details` to also generate row-level predictions and CSV error
analysis.

## Build A Runnable Deployment Bundle

Use upstream-filtered project history containing `ticket_id`, `title`,
`assignee`, and `created_at`. `component`, `description`, and `file_paths` are
strongly recommended. The builder does not perform duplicate detection or
remove duplicate clusters; it only reports cross-split ID/content overlap and
refuses deployment approval when the input prerequisite is violated.

```bash
python3 assignee_triage_accuracy/scripts/prepare_assignee_deployment.py \
  --history data/project_issue_history.jsonl \
  --inactive-assignees data/inactive_assignees.json \
  --output-dir models/assignee_deployment \
  --minimum-assignee-history 5
```

The first run is intentionally `research_only`. Review
`active_assignees.json`, the leakage audit, frozen test metrics, and the two
`*.candidate.json` ownership files. A candidate ownership map is not loaded as
authoritative production ownership until a maintainer reviews it and adds its
path to `deployment_bundle.json`.

After roster review, request approval. Approval still fails closed unless the
calibration sample, target auto-assignment accuracy/coverage, chronological
split, and leakage gates all pass.

```bash
python3 assignee_triage_accuracy/scripts/prepare_assignee_deployment.py \
  --history data/project_issue_history.jsonl \
  --inactive-assignees data/inactive_assignees.json \
  --output-dir models/assignee_deployment \
  --minimum-assignee-history 5 \
  --target-auto-accuracy 0.80 \
  --minimum-auto-coverage 0.10 \
  --confirm-upstream-duplicate-filtered \
  --confirm-roster-reviewed \
  --approve
```

Run assignee recommendation by itself:

```bash
python3 assignee_triage_accuracy/scripts/recommend_assignee.py \
  --bundle models/assignee_deployment/deployment_bundle.json \
  --ticket data/new_ticket.json \
  --output reports/assignee_recommendation.json
```

Or load the same bundle in the integrated pipeline:

```bash
PYTHONPATH=src python3 src/main.py \
  --raw-ticket data/new_ticket.json \
  --repo-path /path/to/project/repository \
  --assignee-deployment-bundle models/assignee_deployment/deployment_bundle.json \
  --output reports/pipeline_result.json
```

Record a reviewed outcome. Online routing can consume this append-only file
immediately, while `merge_assignee_feedback.py` creates a frozen history
snapshot for the next reproducible evaluation cycle.

```bash
python3 assignee_triage_accuracy/scripts/record_assignee_feedback.py \
  --ticket data/new_ticket.json \
  --prediction reports/assignee_recommendation.json \
  --final-assignee developer@example.com \
  --feedback models/assignee_deployment/assignee_feedback.jsonl
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
```

Add `--write-details` when the Eclipse row-level predictions and CSV error
analysis are needed.
