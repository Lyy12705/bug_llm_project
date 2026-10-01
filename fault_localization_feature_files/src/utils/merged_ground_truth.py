"""Merged-repair ground truth contract. Never infer repair success from a merge."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "merged-repair-ground-truth-v1"
SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def digest(value: str | bytes) -> str:
    raw = value.encode("utf-8", errors="surrogateescape") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def pull_identity(ticket: dict[str, Any]) -> tuple[str, int]:
    repo = str(ticket.get("repo") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid repository identity")
    numbers = set()
    if ticket.get("pull_number"):
        numbers.add(int(ticket["pull_number"]))
    if ticket.get("pr_url"):
        match = re.fullmatch(r"https://github.com/([^/]+/[^/]+)/pull/(\d+)/?", ticket["pr_url"])
        if not match or match[1].lower() != repo.lower():
            raise ValueError("PR URL does not match repository")
        numbers.add(int(match[2]))
    ticket_id = str(ticket.get("ticket_id") or ticket.get("instance_id") or "")
    prefix = repo.replace("/", "__") + "-"
    # SWE-bench IDs encode the repair PR, not the linked issue number.
    if ticket_id.startswith(prefix) and ticket_id[len(prefix):].isdigit():
        numbers.add(int(ticket_id[len(prefix):]))
    if len(numbers) != 1 or min(numbers) <= 0:
        raise ValueError("Missing or conflicting repair PR identity")
    return repo, numbers.pop()


def validate_pull(ticket: dict[str, Any], pull: dict[str, Any]) -> dict[str, Any]:
    repo, number = pull_identity(ticket)
    if pull.get("number") != number or str(pull.get("base", {}).get("repo", {}).get("full_name", "")).lower() != repo.lower():
        raise ValueError("GitHub response does not match requested repair PR")
    if pull.get("merged") is not True or not pull.get("merged_at") or pull.get("state") != "closed":
        raise ValueError("Repair PR is not actually merged")
    merge = str(pull.get("merge_commit_sha") or "")
    head = str(pull.get("head", {}).get("sha") or "")
    base = str(ticket.get("base_commit") or "")
    if not all(SHA.fullmatch(value) for value in (merge, head, base)):
        raise ValueError("Full base, head and merged commit SHAs are required")
    return {
        "schema_version": SCHEMA,
        "repo": repo,
        "pull_number": number,
        "pr_url": f"https://github.com/{repo}/pull/{number}",
        "merged": True,
        "merged_at": pull["merged_at"],
        "merge_commit_sha": merge,
        "pr_head_sha": head,
        "target_branch": pull["base"].get("ref"),
        "base_commit": base,
        "diff_scope": "base_commit_to_merged_commit_full_tree",
        "includes_tests_and_other_intervening_changes": True,
        "pull_response_sha256": digest(json.dumps(pull, sort_keys=True, ensure_ascii=False)),
    }


def test_names(ticket: dict[str, Any], key: str) -> list[str]:
    value = ticket.get(key.lower(), ticket.get(key, []))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Invalid {key} test names")
    return sorted(set(value))


def validate_repair_evidence(ticket: dict[str, Any], provenance: dict[str, Any], evidence: dict[str, Any], *, evidence_dir: Path | None) -> dict[str, Any]:
    """Import a trusted test runner receipt, binding both executions and logs to SHAs.

    CI green, dataset labels, and a bare 'resolved: true' are deliberately insufficient.
    The runner must execute the same regression tests on both snapshots.
    """
    ticket_id = str(ticket.get("ticket_id") or ticket.get("instance_id") or "")
    expected = {"ticket_id": ticket_id, "repo": provenance["repo"],
                "base_commit": provenance["base_commit"],
                "merge_commit_sha": provenance["merge_commit_sha"],
                "patch_sha256": provenance["patch_sha256"]}
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("Repair evidence identity/commit/patch mismatch")
    if evidence.get("schema_version") != "merged-repair-test-evidence-v1":
        raise ValueError("Unsupported repair evidence schema")
    for key in ("runner", "environment", "test_suite_sha256"):
        if not evidence.get(key):
            raise ValueError(f"Repair evidence missing {key}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence["test_suite_sha256"])):
        raise ValueError("Invalid test suite hash")
    failures, regressions = test_names(ticket, "FAIL_TO_PASS"), test_names(ticket, "PASS_TO_PASS")
    if not failures:
        raise ValueError("No FAIL_TO_PASS tests: successful repair cannot be established")
    if set(failures) & set(regressions):
        raise ValueError("FAIL_TO_PASS and PASS_TO_PASS overlap")
    if evidence.get("fail_to_pass") != failures or evidence.get("pass_to_pass") != regressions:
        raise ValueError("Repair evidence test inventory differs from dataset")
    for phase, sha in (("before", provenance["base_commit"]), ("after", provenance["merge_commit_sha"])):
        run = evidence.get(phase, {})
        if run.get("commit_sha") != sha or run.get("completed") is not True or not run.get("command"):
            raise ValueError(f"Incomplete {phase} test execution")
        results = run.get("tests", {})
        for name in failures + regressions:
            expected_status = "FAILED" if phase == "before" and name in failures else "PASSED"
            if results.get(name) != expected_status:
                raise ValueError(f"{phase}: test {name!r} must be {expected_status}")
        if not run.get("log_path") or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(run.get("log_sha256", ""))):
            raise ValueError(f"Missing {phase} log identity")
        if evidence_dir is not None:
            log = (evidence_dir / str(run.get("log_path") or "")).resolve()
            if not log.is_relative_to(evidence_dir.resolve()) or not log.is_file():
                raise ValueError(f"Missing or out-of-directory {phase} log")
            if digest(log.read_bytes()) != run.get("log_sha256"):
                raise ValueError(f"{phase} log hash mismatch")
    return {"status": "verified", "evidence": evidence,
            "evidence_sha256": digest(json.dumps(evidence, sort_keys=True, ensure_ascii=False))}


def validate_gold(row: dict[str, Any]) -> None:
    provenance = row.get("merged_fix", {})
    if row.get("ground_truth_schema") != SCHEMA or provenance.get("schema_version") != SCHEMA:
        raise ValueError("Merged-repair Gold required; rebuild legacy developer-patch Gold")
    if row.get("ground_truth_status") != "verified" or provenance.get("merged") is not True or not provenance.get("merged_at"):
        raise ValueError("Unverified merged-repair Gold cannot be scored")
    for key in ("repo", "base_commit"):
        if row.get(key) != provenance.get(key):
            raise ValueError(f"Gold {key} differs from merged repair provenance")
    for key in ("base_commit", "merge_commit_sha", "pr_head_sha"):
        if not SHA.fullmatch(str(provenance.get(key, ""))):
            raise ValueError(f"Invalid {key} in merged repair provenance")
    verification = provenance.get("repair_verification", {})
    evidence = verification.get("evidence", {})
    if verification.get("status") != "verified" or not evidence:
        raise ValueError("Successful repair evidence is required")
    if digest(json.dumps(evidence, sort_keys=True, ensure_ascii=False)) != verification.get("evidence_sha256"):
        raise ValueError("Repair evidence hash mismatch")
    for key in ("repo", "base_commit", "merge_commit_sha", "patch_sha256"):
        if evidence.get(key) != provenance.get(key):
            raise ValueError(f"Repair evidence {key} mismatch")
    if evidence.get("ticket_id") != row.get("ticket_id"):
        raise ValueError("Repair evidence ticket mismatch")
    validate_repair_evidence(row, provenance, evidence, evidence_dir=None)
    if "merged_patch" in row and digest(row["merged_patch"]) != provenance.get("patch_sha256"):
        raise ValueError("Merged patch hash mismatch")
    content = {key: row.get(key) for key in ("fixed_files", "fixed_symbol_records", "excluded_symbol_records")}
    if digest(json.dumps(content, sort_keys=True, ensure_ascii=False)) != row.get("gold_content_sha256"):
        raise ValueError("Gold file/symbol content hash mismatch")
    if not row.get("fixed_files"):
        raise ValueError("Merged Gold contains no changed files")
    for symbol in row.get("fixed_symbol_records", []) + row.get("excluded_symbol_records", []):
        if symbol.get("repo") != row["repo"] or symbol.get("base_commit") != row["base_commit"] or symbol.get("ticket_id") != row["ticket_id"]:
            raise ValueError("Symbol snapshot identity mismatch")
        if symbol.get("provenance", {}).get("patch_sha256") != provenance.get("patch_sha256"):
            raise ValueError("Symbol patch identity mismatch")


def evaluation_policy(rows: list[dict[str, Any]], *, allow_legacy: bool = False) -> dict[str, Any]:
    if not rows and not allow_legacy:
        raise ValueError("No verified merged Gold supplied")
    counts = {"verified_merged_rows": 0, "legacy_rows": 0}
    for row in rows:
        if "ground_truth_schema" in row or "merged_fix" in row:
            validate_gold(row)
            counts["verified_merged_rows"] += 1
        else:
            counts["legacy_rows"] += 1
    if counts["legacy_rows"] and (not allow_legacy or counts["verified_merged_rows"]):
        raise ValueError("Legacy/mixed Gold rejected. Rebuild merged Gold, or explicitly select legacy-only evaluation.")
    return {"policy": "legacy_developer_patch" if counts["legacy_rows"] else SCHEMA, **counts}


def mapped_symbols(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Read combined merged Gold or historical flat symbol rows without losing targets."""
    for row in rows:
        if "ground_truth_schema" in row or "merged_fix" in row:
            validate_gold(row)
            yield from (item for item in row.get("fixed_symbol_records", []) if item.get("mapping_status") == "mapped")
        elif row.get("mapping_status") == "mapped":
            yield row
