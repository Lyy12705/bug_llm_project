# Package Proximity Rerank Summary

## Scope

This pass adds a lightweight same-package / near-path rerank signal to the
retrieval-first fault localization pipeline. The goal is to address the
300-ticket development-run failure analysis where many Top-5 misses were local ranking
confusions rather than index coverage failures.

The change does not expand LLM usage. It only uses existing report terms,
identifier variants, source-path hints, and candidate file paths.

## Implementation

- Adds `package_proximity_score` and `package_proximity_boost`.
- Applies the signal after file-level aggregation, before final Top-k selection.
- Uses it as a tie-breaker when scores are saturated.
- Exposes the signal in raw `scoring_signals`, user-facing output, confidence
  reasoning, and candidate reasons.

## Controlled Check

Output directory:
`reports/fault_localization/swebench_lite_package_proximity_django_miss10`

Subset:
10 Django tickets selected from the 300-ticket development-run Top-5 miss list.

| Metric | Value |
|---|---:|
| Tickets | 10 |
| Failures | 0 |
| Top-1 | 0.2000 |
| Top-3 | 0.4000 |
| Top-5 | 0.6000 |
| MRR | 0.3167 |

These 10 tickets were Top-5 misses in the previous 300-ticket development prediction
file, so this is a useful local signal. It is not a replacement for a new
300-ticket full run.

## Full Top-5 Miss Subset Check

Output directory:
`reports/fault_localization/swebench_lite_package_proximity_top5_miss56`

Subset:
56 tickets selected from the 300-ticket development-run Top-5 miss list. This run used
the same retrieval-first configuration with `tfidf-sbert-rerank`, package
proximity enabled, and no LLM rerank. Existing index caches were reused, so no
repository checkout was needed during this comparison.

| Metric | Value |
|---|---:|
| Tickets | 56 |
| Failures | 0 |
| Cached index checkout skips | 56 |
| Top-1 | 0.2500 |
| Top-3 | 0.4464 |
| Top-5 | 0.5893 |
| MRR | 0.3646 |

Rank distribution:

| Rank bucket | Count |
|---|---:|
| Top-1 | 14 |
| Rank 2 | 6 |
| Rank 3 | 5 |
| Rank 4 | 3 |
| Rank 5 | 5 |
| Still miss | 23 |

Per-repository Top-5 recovery:

| Repository | Tickets | Top-5 hits |
|---|---:|---:|
| django/django | 13 | 9 |
| matplotlib/matplotlib | 4 | 2 |
| mwaskom/seaborn | 1 | 1 |
| pallets/flask | 1 | 1 |
| psf/requests | 1 | 1 |
| pydata/xarray | 1 | 1 |
| pylint-dev/pylint | 3 | 1 |
| pytest-dev/pytest | 6 | 3 |
| scikit-learn/scikit-learn | 2 | 1 |
| sphinx-doc/sphinx | 3 | 3 |
| sympy/sympy | 21 | 10 |

This shows the signal generalizes beyond the earlier Django-only sample, but it
does not solve all ranking failures. SymPy, Matplotlib, Pylint, and Pytest still
contain misses where broad same-package files or test/source naming overlap can
out-rank the gold file.

## Interpretation

The new signal can recover some same-package / near-path ranking misses without
requiring Code Llama or long-context repository prompting. The remaining misses
show that package proximity is not enough by itself; broad files with many
shared terms can still outrank the gold file.

The 56-ticket subset result should be reported as controlled recovery on prior
misses, not as a new full-split development accuracy number. A full 300-ticket rerun
is still needed before claiming overall Top-1/Top-3/Top-5/MRR improvement,
because prior Top-5 hits may still regress under the new weighting.

## Next Validation

Before updating the development benchmark, rerun the full 300-ticket split with
the same retrieval-first configuration and compare both recovered misses and
new regressions.
