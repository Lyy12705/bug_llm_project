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
`src/modules/assignee_feedback.py`. The Top-5 semi-automatic path exposes two
explicit assignment actions: choose candidate rank 1–5, or opt in to assigning
rank 1. `src/modules/assignee_selection.py` resolves both actions fail-closed;
the rank-1 shortcut is marked as user-authorized rather than autonomous. The
v4 feedback event preserves that distinction. Confirmed final assignees can be
converted back into history rows for later owner-profile refresh without
introducing a new database dependency.

The semi-automatic workflow now supports two correctness definitions side by
side. `is_top5_correct` means that the single historical recorded owner appears
in the list. `is_top5_eligible_correct` means that at least one candidate belongs
to a maintainer-reviewed, permission-confirmed, time-valid capability set. The
second definition never infers qualification from historical `components`.
Those values remain descriptive history only. Eligible-owner policy evaluation
also reports candidate eligibility precision, because one qualified person in a
five-person list is not the same as five safe choices.

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
  --intermediate-history 2021_q4=assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl \
  --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_raw.jsonl \
  --holdout-name 2022_q1 \
  --holdout-manifest assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_manifest.json \
  --output-dir assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_rolling_2022q1_holdout \
  --confirm-untouched-holdout
```

Q4 had already been inspected and is retained only as a consistency check. The
new SHA-256-sealed 2022 Q1 holdout has 2,777 evaluated rows. Automatic
assignment reached `99.17%` accuracy at `21.79%` coverage; its Wilson 95% lower
bound was `98.08%`. None of 178 unseen-owner rows was automatically assigned
(upper confidence bound `2.11%`). Maximum PSI was `0.0525`, below the `0.25`
drift limit, and every hard holdout gate passed. Top-3 confirmation accuracy was
only `67.23%`, below its 90% advisory target, so this remains selective routing
rather than evidence that all issues can be assigned automatically.

Production remains blocked. Public BMO snapshots do not provide a reviewed
active/departed roster, verified ownership policy, exact assignment-change
timestamps, or real shadow feedback. Runtime integration therefore still
requires a reviewed target-project roster, 500 reviewed shadow tickets or 28
observation days, and explicit operator approval. See
`phase7_rolling_open_set/README.md` for the protocol and limitations.

## V3 Safety And Improvement Workflow

The v3 workflow is implemented under `phase9_v3_protocol/`. It permanently
registers the inspected 2022 Q2 data as diagnosis-only, separates ranker,
calibrator, open-set, and policy-selection windows, and audits ticket IDs,
duplicate families, near-duplicate text, and owner events. Training commands
do not accept a sealed holdout as a ranker-selection input.

Candidate generation has an explicitly versioned v3 mode with component-family,
reporter-history, and routing-time-safe file-ownership retrieval. File paths are
used only when provenance says they were available at report time; paths learned
from a later fix are ignored. Duplicate families are strict split-exclusion keys,
not routing features. Existing frozen artifacts remain on v2 mode so registered
replays keep the original candidate semantics.

New LTR training compares pointwise logistic/gradient boosting with a bounded
hard-negative pairwise logistic ranker. Automatic selection still uses the worst
development window, so a pairwise model is retained only when it is actually
more stable. Reports include recall@10/30/50, conservative source and source-
family support ablations, pairwise-training audit, candidate/ranking miss
taxonomy, and a worst-window gate for the 95% known-owner recall@30 and 10%
ranking-miss reduction targets.

Rolling policy selection now defaults to the full strict gate in every
development window: at least two independent candidate-source families,
minimum auto rows, auto-accuracy Wilson lower bound, unseen-owner Wilson upper
bound, and component point/confidence floors. See
`phase9_v3_protocol/README.md` for protocol registration and holdout sample
planning commands.

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
  --active-assignees data/active_assignees.reviewed.json \
  --inactive-assignees data/inactive_assignees.json \
  --assignee-alias-map data/assignee_aliases.reviewed.json \
  --output-dir models/assignee_deployment \
  --minimum-assignee-history 5
```

The first run is intentionally `research_only`. It writes separate
`calibration_fit_set.jsonl`, `policy_selection_set.jsonl`, and `test_set.jsonl`
partitions plus `temporal_protocol_manifest.json`. Review `active_assignees.json`,
the leakage audit, frozen test metrics, and the two `*.candidate.json` ownership
files. Research-only bundles are not accepted by the runtime loader. A candidate
ownership map is not loaded as authoritative production ownership until a
maintainer reviews it and adds its path to the bundle configuration.

After roster review, request approval. Approval still fails closed unless the
calibration sample, target auto-assignment accuracy/coverage, chronological
split, and leakage gates all pass.

```bash
python3 assignee_triage_accuracy/scripts/prepare_assignee_deployment.py \
  --history data/project_issue_history.jsonl \
  --active-assignees data/active_assignees.reviewed.json \
  --inactive-assignees data/inactive_assignees.json \
  --assignee-alias-map data/assignee_aliases.reviewed.json \
  --output-dir models/assignee_deployment \
  --minimum-assignee-history 5 \
  --target-auto-accuracy 0.85 \
  --minimum-auto-coverage 0.10 \
  --maximum-unseen-auto-rate 0.05 \
  --minimum-test-rows 2500 \
  --minimum-auto-rows 250 \
  --minimum-auto-accuracy-lower-bound 0.80 \
  --confirm-upstream-duplicate-filtered \
  --confirm-roster-reviewed \
  --approve
```

Calibration fitting, routing-threshold selection, and frozen-test approval use
different chronological rows. An approved schema-v2 bundle also contains an
expiry time and SHA-256 manifest. Missing, expired, out-of-directory, or modified
artifacts cause loading to fail closed. The generated open-set detector file
remains research-only until it is evaluated with the same score definition used
at runtime; approved bundles currently pin `fallback_rule_based_v1`.

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
  --active-roster models/assignee_deployment/active_assignees.json \
  --feedback models/assignee_deployment/assignee_feedback.jsonl
```

Evaluate shadow operation without changing real assignments. The gate requires
at least 500 reviewed tickets or 28 observation days, complete known/unseen
labels, 85% auto accuracy, 10% auto coverage, unseen-owner auto rate and its
Wilson 95% upper bound below 5%, an 80% auto-accuracy Wilson lower bound, two
consecutive passing weeks, complete latency, and a p95 latency budget. Repeated feedback for one ticket uses
the latest reviewed event.

Run the frozen rolling LTR/open-set policy for one new ticket in shadow mode.
Repeat `--history` in chronological source order for every frozen and completed
rolling window. The command recovers label-availability timestamps, removes
cross-window overlap, filters future labels, checks the active roster, and never
changes the real assignment.

```bash
python3 assignee_triage_accuracy/scripts/recommend_assignee_rolling_shadow.py \
  --history assignee_triage_accuracy/paper_grade/data/processed/bmo_public_10k_history_train.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/processed/bmo_public_10k_validation_set.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2020q3_raw.jsonl \
  --active-roster data/active_assignees.reviewed.json \
  --ticket data/raw_tickets/assignee_shadow_ticket.example.json \
  --output reports/assignee_shadow_prediction.json
```

The same entry point also supports a fail-closed semi-automatic Top-5 mode.
The initial response leaves `assignee=manual_triage`; it exposes exactly five
ranked, active-roster candidates only when the frozen multi-window Top-5 policy
passes its 85% accuracy, coverage, sample-size, confidence-bound, open-set, and
candidate-source checks. A passing response offers `select_top5_candidate` and
`assign_top1`; both require an explicit user action. Otherwise `top5_candidates`
is empty and normal manual triage continues.

```bash
python3 assignee_triage_accuracy/scripts/recommend_assignee_rolling_shadow.py \
  --mode semi_auto_top5 \
  --routing-artifact reports/top5_assist/rolling_open_set_artifact.json \
  --model-dir models/assignee_ltr_v3 \
  --history data/assignee_history.jsonl \
  --active-roster data/active_assignees.reviewed.json \
  --ticket data/new_ticket.json \
  --output reports/assignee_top5_suggestion.json
```

Resolve the user's choice against the active roster again at decision time.
Choose rank 1–5 with `select_top5_candidate --selected-candidate-rank N`, or use
`assign_top1` for the explicit Top-1 shortcut. The command emits an authorized
decision for the issue-tracker adapter; it does not modify the external tracker
itself.

For an Eligible-owner policy, build the roster with a separate reviewed
capability attestation. Each eligible entry must have an explicit product,
component, or product/component scope and `permissions_confirmed=true`.
The review requires a named reviewer and a bounded validity interval. Missing,
expired, overlapping, or unreviewed data fails closed; it is not backfilled from
assignment history.

```bash
python3 assignee_triage_accuracy/scripts/build_assignee_roster_snapshot.py \
  --history data/assignee_history.jsonl \
  --as-of 2026-07-22T00:00:00Z \
  --reviewed-active data/active_assignees.reviewed.json \
  --reviewed-eligibility \
    assignee_triage_accuracy/phase8_operational_readiness/roster/assignee_eligibility_review.example.json \
  --reviewer engineering-lead@example.com \
  --confirm-organizational-review \
  --output data/active_assignees.with_eligibility.json \
  --report reports/assignee_roster_audit.json
```

Evaluate both label definitions without overwriting the exact-owner result:

```bash
python3 assignee_triage_accuracy/scripts/evaluate_assignee_eligible_owner.py \
  --predictions reports/rolling_predictions.jsonl \
  --eligibility-roster data/active_assignees.with_eligibility.json \
  --output-predictions reports/rolling_predictions.eligible.jsonl \
  --report reports/dual_owner_accuracy.json
```

To freeze a new multi-window Top-5 policy against the eligible set, pass
`--top5-correctness-field is_top5_eligible_correct` and repeat
`--eligibility-roster` for every non-overlapping snapshot covering the policy
selection windows. The same snapshots are mandatory on sealed holdout replay.
Runtime will only expose an eligible-policy Top-5 list when all five candidates
are currently active and eligible, then checks them again when the user makes a
selection. Consequently the displayed subset's eligible hit rate should be
100% by construction; coverage, qualification-label coverage, candidate
eligibility precision, and the same subset's secondary exact-owner accuracy
must remain visible so that this safety filter is not mistaken for ranking
quality. Feedback events preserve the eligibility verification evidence, and
the live-shadow gate fails if an eligible-policy candidate set or selected
assignee was not verified.

```bash
python3 assignee_triage_accuracy/scripts/resolve_assignee_top5_selection.py \
  --recommendation reports/assignee_top5_suggestion.json \
  --active-roster data/active_assignees.reviewed.json \
  --action assign_top1 \
  --reviewer triager@example.com \
  --output reports/assignee_top5_decision.json
```

Reviewed selections recorded with `record_assignee_feedback.py` include the
selection mode, chosen candidate rank, user action, and whether Top-1 was
explicitly authorized. `evaluate_assignee_shadow.py` reports a separate Top-5
assist gate and does not count user-authorized Top-1 as autonomous assignment.

```bash
python3 assignee_triage_accuracy/scripts/evaluate_assignee_shadow.py \
  --feedback models/assignee_deployment/assignee_feedback.jsonl \
  --output reports/assignee_shadow_report.json
```

After the sealed holdout, build the rolling-LTR candidate from ordered history.
The output is self-contained and hash-verified. Without both a reviewed roster
and a passed real shadow report it is deliberately emitted as `research_only`;
`--approve` cannot bypass a missing gate.

```bash
python3 assignee_triage_accuracy/scripts/prepare_assignee_rolling_deployment.py \
  --history assignee_triage_accuracy/paper_grade/data/processed/bmo_public_10k_history_train.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/processed/bmo_public_10k_validation_set.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2020q3_raw.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2020q4_raw.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q1_raw.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q2_raw.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q3_raw.jsonl \
  --history assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl \
  --active-roster data/active_assignees.reviewed.json \
  --shadow-report reports/assignee_shadow_report.json \
  --output-dir models/assignee_rolling_deployment \
  --approve
```

Only an approved, unexpired, integrity-checked rolling bundle is accepted by
the production entry point. Production also requires a short-lived operational
state bound to the bundle hash. A research-only, expired, modified, mismatched,
invalid-roster, stale-state, or unhealthy bundle writes `manual_triage`; it
never authorizes automatic assignment.

```bash
python3 assignee_triage_accuracy/scripts/evaluate_assignee_operational_guard.py \
  --bundle models/assignee_rolling_deployment/deployment_bundle.json \
  --shadow-report reports/assignee_shadow_report.json \
  --active-roster data/active_assignees.reviewed.json \
  --requested-rollout-percentage 5 \
  --output models/assignee_rolling_deployment/operational_state.json
```

```bash
python3 assignee_triage_accuracy/scripts/recommend_assignee_rolling.py \
  --bundle models/assignee_rolling_deployment/deployment_bundle.json \
  --operational-state models/assignee_rolling_deployment/operational_state.json \
  --ticket data/new_ticket.json \
  --output reports/assignee_rolling_decision.json
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
