# Phase 7: Rolling Open-World Assignee Routing

## Goal

Phase 7 improves stability without replacing the existing candidate ranker
with an unvalidated large model. It combines a shallow candidate-level ranker,
local SBERT similarity, rolling historical ownership, portable logistic
calibration, open-set risk estimation, dual routing thresholds, and drift
detection.

This is an offline public-dataset experiment. The exported bundle is
`research_only` and is not automatically loaded by production routing.

## Data protocol

The temporal order is:

1. Fit calibration and open-set models on validation, 2020 Q3, and 2020 Q4.
2. Select one shared routing policy on 2021 Q1, Q2, and Q3.
3. Evaluate the frozen bundle once on 2021 Q4.

An issue is eligible for historical owner profiles only when its
`last_change_time` is at or before the next query window. This is conservative
and prevents the final-assignee snapshot from being exposed before it was
observable. Exact assignment-change history is unavailable in the public
snapshot, so `last_change_time` is a proxy rather than a perfect assignment
timestamp.

Input windows are expected to have passed upstream duplicate detection. Every
window is built only from earlier rows; exact ticket-ID/content overlap is
checked only as an offline leakage guard and is not part of assignee routing.

The first inspected 2021 Q3 holdout used creation-time history eligibility and
is invalid. It is retained only for audit and marked by
`reports/bmo_public_2021q3_untouched_holdout/INVALIDATED.md`. Q3 is now a
development policy-selection window and is not reported as test evidence.

## Model and routing

The fixed Phase 6 model is
`candidate_hist_gradient_boosting_shallow_ltr_v1`. It ranks a Top-30 candidate
pool using ownership shares, temporal activity/recency, BM25 evidence, local
SBERT similarity, and source agreement. A portable logistic calibrator predicts
whether the top-ranked owner is correct. A second balanced logistic model is
trained with leave-assignee-out folds to estimate unseen-owner risk.

The policy has two independent risk bands:

- `auto_assign`: high calibrated correctness and low unseen-owner risk;
- `top3_confirmation`: medium confidence and acceptable review risk;
- `manual_triage`: low confidence, high open-set risk, no viable candidate, or
  detected score drift.

The policy must satisfy every selection window, not just pooled averages:
automatic accuracy at least 85%, coverage at least 10%, and unseen-owner false
automatic assignment rate at most 5%. PSI above 0.25 disables automatic
assignment for the evaluated window.

## Reproduction

```bash
python3 assignee_triage_accuracy/scripts/train_assignee_rolling_open_set.py

python3 assignee_triage_accuracy/scripts/evaluate_assignee_rolling_holdout.py \
  --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl \
  --holdout-name 2021_q4 \
  --output-dir assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_public_2021q4_conservative_untouched_holdout \
  --confirm-untouched-holdout
```

The evaluator rejects missing confirmation, a mismatched ranker, a failed
development gate, and a history-window mismatch. It never refits models or
selects thresholds from the holdout.

## Results

Development policy selection:

| Window | Auto accuracy | Auto coverage | Unseen false-auto | Top-3 review rate | Manual rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 Q1 | 98.63% | 16.40% | 1.81% | 24.15% | 59.45% |
| 2021 Q2 | 99.54% | 16.25% | 0.00% | 19.67% | 64.08% |
| 2021 Q3 | 98.57% | 18.21% | 0.00% | 18.99% | 62.80% |

Untouched 2021 Q4 holdout:

| Metric | Result |
| --- | ---: |
| Evaluated rows | 2,779 |
| Known/unseen owners | 2,638 / 141 |
| Known-owner Top-1 / Top-3 / Top-5 | 48.41% / 64.33% / 70.47% |
| Known-owner MRR / Macro-F1 | 58.07% / 14.21% |
| Automatic accuracy / coverage | 97.75% / 24.04% |
| Unseen-owner false-auto rate | 0.00% |
| Top-3 confirmation rate / accuracy | 21.05% / 71.11% |
| Manual-triage rate | 54.91% |
| Brier / ECE | 0.1394 / 0.0419 |
| Open-set AUROC / AUPRC | 0.6963 / 0.0819 |
| Maximum PSI | 0.0518 |

These results support selective routing, not full automatic assignment. The
open-set detector has modest discrimination and rejects many known-owner rows;
its conservative threshold is useful for safety but reduces coverage.

## Remaining blockers

- No reviewed active/inactive/departed developer roster exists in public BMO
  snapshots.
- `last_change_time` is not the exact time at which the final assignee became
  known.
- File/module ownership is not available consistently enough for validation.
- Long-tail Macro-F1 remains low and candidate recall drops on future data.
- The holdout is capped at 3,000 fetched records and is not the complete Q4
  population.
- Production integration needs a target-project roster, authoritative
  ownership configuration, monitoring, and explicit operator approval.

The next practical improvement is owner/component candidate-recall work using
additional rolling windows, followed by a lightweight candidate-level logistic
or gradient-boosting ablation. An LLM reranker should remain offline unless it
beats this frozen baseline under the same temporal protocol and latency budget.
