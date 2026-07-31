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
3. Treat the already-inspected 2021 Q4 result as a consistency check, not as
   untouched evidence.
4. Update candidate history from Q4 without refitting any model or threshold,
   then evaluate the still-frozen bundle once on a SHA-256-sealed 2022 Q1
   holdout.

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

The current implementation also exports an independent
`top5_assist_policy`. It is a semi-automatic user-choice policy, not a third
automatic routing band. One threshold set must achieve at least 85% Top-5
accuracy in every policy-selection window, with minimum coverage, row count,
Wilson lower bound, two candidate evidence families, and five ranked
candidates. Runtime roster filtering is fail-closed: if any of the measured
Top-5 owners is inactive or unreviewed, no list is shown. The result always
keeps the real assignee unchanged.

The policy must satisfy every selection window, not just pooled averages:
automatic accuracy at least 85%, coverage at least 10%, and unseen-owner false
automatic assignment rate at most 5%. PSI above 0.25 disables automatic
assignment for the evaluated window.

## Reproduction

```bash
python3 assignee_triage_accuracy/scripts/train_assignee_rolling_open_set.py

python3 assignee_triage_accuracy/paper_grade/scripts/fetch_bmo_assignee_dataset.py \
  --start-date 2022-01-01 \
  --end-date 2022-03-31 \
  --max-bugs 3000 \
  --seal-output \
  --output assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_raw.jsonl \
  --manifest assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_manifest.json

python3 assignee_triage_accuracy/scripts/evaluate_assignee_rolling_holdout.py \
  --intermediate-history 2021_q4=assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2021q4_raw.jsonl \
  --holdout assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_raw.jsonl \
  --holdout-name 2022_q1 \
  --holdout-manifest assignee_triage_accuracy/paper_grade/data/raw/bmo_public_future_2022q1_manifest.json \
  --output-dir assignee_triage_accuracy/phase7_rolling_open_set/reports/bmo_rolling_2022q1_holdout \
  --confirm-untouched-holdout
```

The evaluator rejects missing confirmation or manifest, a SHA-256 mismatch, a
mismatched ranker, a failed development gate, duplicate window names, and a
frozen history-window mismatch. Intermediate windows update history only. They
never refit the ranker, calibrator, open-set detector, or routing policy. An
append-only registry claims the holdout before scoring and rejects any second
attempt to reuse the same SHA-256 as untouched evidence.

## Results

Development policy selection:

| Window | Auto accuracy | Auto coverage | Unseen false-auto | Top-3 review rate | Manual rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 Q1 | 98.63% | 16.40% | 1.81% | 24.15% | 59.45% |
| 2021 Q2 | 99.54% | 16.25% | 0.00% | 19.67% | 64.08% |
| 2021 Q3 | 98.57% | 18.21% | 0.00% | 18.99% | 62.80% |

Previously inspected 2021 Q4 consistency check:

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

Sealed 2022 Q1 holdout, evaluated once with Q4 as history-update-only:

| Metric | Result |
| --- | ---: |
| Evaluated rows | 2,777 |
| Known/unseen owners | 2,599 / 178 |
| Overall Top-1 / Top-3 / candidate recall | 44.29% / 60.61% / 83.58% |
| Known-owner Top-1 / Top-3 / candidate recall | 47.33% / 64.76% / 89.30% |
| Automatic rows | 605 |
| Automatic accuracy / coverage | 99.17% / 21.79% |
| Automatic-accuracy Wilson 95% lower bound | 98.08% |
| Unseen-owner false-auto rate / upper bound | 0.00% / 2.11% |
| Top-3 confirmation rate / accuracy | 21.32% / 67.23% |
| Manual-triage rate | 56.90% |
| Brier / ECE | 0.1424 / 0.0418 |
| Open-set AUROC / AUPRC | 0.6550 / 0.0960 |
| Maximum PSI | 0.0525 |
| Hard deployment gate | Passed |

The sealed file contains the earliest 3,000 matching rows in the Q1 fetch
window; 2,777 remained after normalization and cross-window overlap removal.
The manifest records the capped fetch, time range, immutable SHA-256, and seal
status. The result passes the model/policy hard gate but does not itself grant
production approval.

A registered reproducibility replay produced exactly the same 2,777 routing
decisions and decision digest
`a30103be4ed2b6484b2f235ac688a54c852d830b3cbea6eaa697e563ddf24d2b`.
The largest low-level numeric delta was `0.0001268`, below the declared `0.0002`
tolerance, and did not change any candidate order, routing status, or fallback
reason. The replay is marked `release_evidence=false`.

These results support selective routing, not full automatic assignment. The
open-set detector has modest discrimination and rejects many known-owner rows;
its conservative threshold is useful for safety but reduces coverage.

## Deployment boundary

`prepare_assignee_rolling_deployment.py` builds a self-contained history,
ranker, policy, roster, holdout, and shadow bundle. The loader verifies all
seven SHA-256 entries, path containment, expiry, cross-artifact versions, and
every approval gate. `recommend_assignee_rolling.py` accepts approved bundles
only. A research-only or invalid bundle returns a fail-closed manual-triage
decision and never authorizes automatic assignment. The production entry point
also requires a short-lived bundle-bound operational state, so stale monitoring
or an active kill switch cannot authorize auto. The current candidate is
correctly `research_only` because the stricter unseen confidence bound, frozen
candidate-source constraint, roster, and shadow gates are not yet available.

## Remaining blockers

- No reviewed active/inactive/departed developer roster exists in public BMO
  snapshots.
- `last_change_time` is not the exact time at which the final assignee became
  known.
- File/module ownership is not available consistently enough for validation.
- Long-tail Macro-F1 remains low and Top-3 confirmation quality is below its
  90% advisory target.
- The holdout is capped at 3,000 fetched records and is not the complete Q1
  population.
- Production integration still needs a reviewed target-project roster,
  authoritative ownership configuration, at least 500 reviewed shadow tickets
  or 28 observation days, and explicit operator approval.

The next practical improvement is owner/component candidate-recall work using
additional rolling windows, followed by a lightweight candidate-level logistic
or gradient-boosting ablation. An LLM reranker should remain offline unless it
beats this frozen baseline under the same temporal protocol and latency budget.
