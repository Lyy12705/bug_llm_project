# Phase 6 Candidate-Level LTR

## Scope

This phase tests a practical candidate-level learning-to-rank replacement for
the fixed Hybrid weights. It does not replace production routing. Existing
calibration, active-roster filtering, open-set abstention, and manual fallback
remain authoritative.

## Protocol

- Training source: BMO public 10k closed-set train split (6,507 rows).
- Ranker examples: three expanding temporal folds inside train.
- Candidate pool: Top-30 with per-source quotas; reported recommendation: Top-10.
- Compared models: logistic regression and histogram gradient boosting.
- Long-tail weighting: `1 / sqrt(train_assignee_frequency)`.
- Features: component/product ownership, smoothed shares, BM25, 30/90-day
  activity, exponential time decay, owner recency, support counts, and optional
  local-only SBERT similarity.
- New holdout: 3,000 BMO bugs fetched from 2020-07-01 onward; 2,740 remained
  after generic-assignee filtering and offline cross-split overlap exclusion.
  The overlap exclusion is a leakage control, not duplicate classification.
- New holdout unseen-owner rate: 15.58%.

## Results

Validation selected histogram gradient boosting. The new holdout comparison
below is limited to the 2,313 tickets whose assignee existed in train.

| Method | Top-1 | Hit@3 | Hit@5 | Hit@10 | MRR | Macro-F1 | Pool Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 candidate base | 0.4717 | - | - | - | 0.5950 | 0.2760 | 0.8962 |
| LTR without SBERT | 0.4712 | 0.6606 | 0.7492 | 0.8448 | 0.5922 | 0.3028 | 0.8962 |
| LTR with local SBERT | **0.4920** | **0.6913** | **0.7687** | **0.8526** | **0.6125** | **0.3283** | 0.8962 |

SBERT improved known-owner Top-1 by 2.03 percentage points over the same V2
candidate base, while MRR and Macro-F1 also improved. The non-SBERT learned
ranker did not improve Top-1 or MRR on the new holdout.

## Source-Quota And Q4 Follow-Up

The candidate generator was then changed from score-only truncation to a
Top-30 source-quota pool. Q3 became a development window and selected the
shallow histogram gradient boosting model before Q4 was fetched.

On the untouched Q4 holdout, 2,705 rows remained after filtering. There were
2,342 known-owner rows and 363 unseen-owner rows (13.42%).

| Q4 known-owner metric | Candidate base | Shallow LTR |
|---|---:|---:|
| Top-1 | 0.4466 | **0.4722** |
| MRR | 0.5730 | **0.5955** |
| Macro-F1 | 0.2297 | **0.2674** |
| Candidate pool recall | - | **0.9150** |

The source-quota pool therefore passed the predeclared 90% candidate-recall
gate and the fixed shallow LTR improved Top-1 by 2.56 percentage points.

Post-hoc calibration used 472 early validation rows for fitting and 316 later
validation rows for threshold selection. Multi-feature logistic calibration
was selected over isotonic regression. On Q4:

- Auto-assignment accuracy: 86.91%.
- Auto-assignment coverage: 24.29%.
- Unseen-owner auto-assignment rate: 7.16%.
- Maximum allowed unseen-owner auto-assignment rate: 5%.

The routing safety gate therefore failed even though selective accuracy and
coverage passed.

## Decision

The SBERT ranker and its calibrator remain `research_only`. They must not
auto-assign tickets because the unseen-owner safety requirement failed.

The next experiment should train an open-set detector with temporal
leave-assignee-out examples and validate it on another future window. Do not
lower the 5% safety limit or tune the detector against Q4 now that its labels
have been inspected.

## Open-Set V2 Implementation

The follow-up now has a reproducible implementation:

1. `build_open_set_folds.py` creates temporal leave-assignee-out folds and
   verifies that each hidden owner is absent from fold history.
2. `train_assignee_open_set.py` fits a regularized logistic detector without
   owner identity or future statistics. Validation is split chronologically:
   the first half fits Top-1 calibration and the second half fits the detector.
3. Q3 alone selects the open-set and confidence thresholds. Q4 is explicitly
   excluded from fitting and threshold selection.
4. `evaluate_assignee_open_set_holdout.py` requires an explicit untouched
   holdout acknowledgement and never changes the frozen artifact.

```bash
python3 assignee_triage_accuracy/scripts/build_open_set_folds.py \
  --output-dir assignee_triage_accuracy/phase6_candidate_ltr/open_set_protocol_v2

python3 assignee_triage_accuracy/scripts/train_assignee_open_set.py

python3 assignee_triage_accuracy/scripts/evaluate_assignee_open_set_holdout.py \
  --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q2_raw.jsonl \
  --output-dir assignee_triage_accuracy/phase6_candidate_ltr/reports/bmo_public_2021q2_open_set_shallow_v3_holdout \
  --confirm-untouched-holdout
```

The trainer now requires the exact ranker name
`candidate_hist_gradient_boosting_shallow_ltr_v1`. This prevents an adjacent
experimental model directory from being loaded silently. An earlier Q1 run
found this guard was needed; that run used the wrong deep-HGB artifact and is
invalidated for promotion purposes. Q1 was not reused after the correction.

With the intended shallow ranker, Q3 development reached 86.04%
auto-assignment accuracy, 23.80% coverage, and a 4.92% unseen-owner
auto-assignment rate. This passed the development gate but was close to the 5%
limit.

The corrected frozen policy then failed on the new 2021 Q2 holdout. It
produced no auto-assignments because every calibrated probability was below
the frozen high-confidence threshold, so coverage was 0%. It routed 19.24% to
Top-3 confirmation with 94.36% Top-3 accuracy and sent 80.76% to manual
triage. Known-owner candidate-pool recall remained 92.96%, but the detector's
holdout AUROC was only 0.6661. This is a calibration/feature drift failure,
not a successful production result. The artifact remains `research_only`,
and Q2 must not be reused to tune the next version.
