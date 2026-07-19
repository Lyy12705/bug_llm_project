# Research protocol registry

`fault_localization_frozen_protocol_v1.json` records the deterministic
repository-disjoint split, source/output hashes, repository membership, and
prior-exposure status for the current 300-row SWE-bench Lite data.

The split is a prospective regression guardrail only. The method had already
been evaluated on the full 300 rows before this split was created, so its
holdout must not be described as an untouched final test. A paper-grade final
estimate requires newly collected tickets whose individual or aggregate
results have never been used during method selection.
