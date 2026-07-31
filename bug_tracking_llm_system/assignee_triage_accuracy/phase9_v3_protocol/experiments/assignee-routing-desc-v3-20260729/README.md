# Description-enriched Top-5 traceable development run

Experiment ID: `assignee-routing-desc-v3-20260729`

This directory is the traceable development version for the Top-5
semi-automatic assignee recommendation deck. It replaces the old
no-description `3,258 rows / 114 domains` numbers. Do not combine metrics from
the two runs.

## Frozen scope

- Source: public Mozilla Bugzilla records.
- Description: first public comment, treated as the report-time description.
- Base raw history: 10,000 rows.
- Rolling raw windows: 2020 Q3, 2020 Q4, 2021 Q1, 2021 Q2, and 2021 Q3;
  3,000 rows per window.
- Combined raw Description coverage: 22,677 / 25,000 = 90.708%.
- Ranker: fixed shallow HistGradientBoosting candidate LTR.
- Candidate pool: Top-30.
- Text retrieval: BM25 and local `all-MiniLM-L6-v2` SBERT.
- Final recommendation list: Top-5.
- User action: candidate selection or user-authorized Top-1 only. Autonomous
  assignment remains disabled.

All important data and artifact SHA-256 values are recorded in
`run_manifest.json`. The stricter temporal/leakage audit is in
`protocol_manifest.json`.

## Time roles

| Role | Partition |
|---|---|
| Ranker fit | Enriched 10k history train |
| Ranker selection | Enriched 10k validation |
| Later ranker development check | Enriched 10k test |
| Confidence calibration fit | Enriched 2020 Q3 |
| Open-set fit | Enriched 2020 Q4 |
| Top-5 policy selection | Enriched 2021 Q1, Q2, and Q3 |

The three policy windows use one frozen ranker, calibrator, open-set detector,
and threshold set. Every reported Exact-owner and historical-domain metric
below uses the same displayed rows.

## Description coverage

| Window | Rows | Non-empty Description | Coverage | Fetch errors |
|---|---:|---:|---:|---:|
| Base 10k raw | 10,000 | 9,138 | 91.38% | 0 |
| 2020 Q3 | 3,000 | 2,706 | 90.20% | 0 |
| 2020 Q4 | 3,000 | 2,728 | 90.93% | 0 |
| 2021 Q1 | 3,000 | 2,726 | 90.87% | 0 |
| 2021 Q2 | 3,000 | 2,696 | 89.87% | 11 |
| 2021 Q3 | 3,000 | 2,683 | 89.43% | 10 |

An empty Description remains an explicit missing value. It is not imputed from
later discussion or fix information.

## Top-5 policy result

| Window | Evaluated rows | Displayed Top-5 | Coverage | Exact-owner Top-5 | Wilson 95% lower |
|---|---:|---:|---:|---:|---:|
| 2021 Q1 | 2,691 | 984 | 36.57% | 85.06% | 82.70% |
| 2021 Q2 | 2,683 | 1,201 | 44.76% | 86.34% | 84.29% |
| 2021 Q3 | 2,707 | 1,296 | 47.88% | 90.66% | 88.96% |

All three windows pass the numerical Top-5 policy thresholds. The policy still
fails the global provenance gate because the public data has no exact
assignment-event timestamp.

## Same-population accuracy table

Evaluation population: 3,481 displayed tickets across 137 normalized
Product x Component domains.

The historical-domain threshold is at least three prior assignments in the
same Product x Component before the evaluated window.

| Metric | Hits / denominator | Result | Evidence status |
|---|---:|---:|---|
| Top-1 is the historical exact owner | 2,371 / 3,481 | 68.11% | Historical exact label |
| Top-5 contains the historical exact owner | 3,049 / 3,481 | 87.59% | Historical exact label |
| Top-1 has historical same-domain experience | 3,353 / 3,481 | 96.32% | Exploratory proxy |
| Top-5 contains same-domain experience | 3,450 / 3,481 | 99.11% | Exploratory proxy |
| Top-1 correct, otherwise ranks 2-5 rescue | 3,107 / 3,481 | 89.26% | Mixed exact label and proxy |
| Candidate eligibility precision | 11,140 / 17,405 slots | 64.00% | Exploratory proxy |

The same-domain values are not formal Eligible-owner accuracy. Public data
does not provide maintainer-reviewed, time-versioned capability, permission,
and active-roster truth.

## Protocol blockers

This is a reproducible research/development run, not release evidence.

- Exact assignment-event timestamps are unavailable; `last_change_time` is
  only a conservative proxy.
- The strict near-duplicate audit flags 21,128 cross-partition text pairs.
  Exact ticket, exact content, duplicate-family, and owner-event overlaps are
  zero, but the near-duplicate text finding still requires review.
- The repository worktree was already dirty when the protocol was registered.
- No new untouched sealed holdout was evaluated.
- No reviewed active/eligibility roster exists for these historical windows.
- File-history evidence is unavailable and Reporter History is not complete.

The correct status is therefore `research_only`, with manual triage as the
fail-closed outcome.

## Primary artifacts

- `run_manifest.json`: human-curated run inventory, hashes, results, and
  evidence boundaries.
- `protocol_manifest.json`: automated v3 temporal and leakage audit.
- `model_frozen/`: frozen ranker artifact, model, evaluation report, and
  predictions.
- `rolling/`: calibrator, open-set detector, Top-5 policy, and per-window
  routing predictions.
- `domain_experience_proxy_report.json`: Exact-owner and domain-experience
  metrics on the identical displayed population.
- `audits/`: Description enrichment coverage and error reports.
