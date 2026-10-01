#!/usr/bin/env python3
"""Rebuild Gold from immutable merged Git trees; unresolved rows fail evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_symbol_gold import build_symbol_gold_records, read_jsonl, safe_name
from utils.merged_ground_truth import SCHEMA, digest, pull_identity, validate_pull, validate_repair_evidence
from utils.symbol_gold import parse_unified_diff


_MERGE_INDEX_LOCK = Lock()
_MERGE_INDEX: dict[str, dict[tuple[int, str], list[str]]] = {}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class GitHub:
    def __init__(self, cache: Path, *, offline: bool = False):
        self.cache, self.offline = cache, offline
        self.unavailable = ""

    def pull(self, repo: str, number: int) -> dict:
        path = self.cache / safe_name(repo) / f"pull-{number}.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("merged") is True:
                return cached
        if self.offline or self.unavailable:
            raise ValueError(self.unavailable or "No merged PR response in offline cache")
        headers = {"User-Agent": "fault-localization-merged-gold", "Accept": "application/vnd.github+json"}
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(f"https://api.github.com/repos/{repo}/pulls/{number}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 429):
                self.unavailable = f"GitHub access/rate limit HTTP {exc.code}; resume after access is available"
            raise ValueError(f"GitHub PR lookup HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            self.unavailable = "GitHub network unavailable; resume with network access"
            raise ValueError(self.unavailable) from exc
        write_json(path, result)
        return result


def git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, timeout=300)
    if result.returncode:
        raise ValueError(f"git {args[0]} failed: " + result.stderr.decode("utf-8", errors="replace")[:500])
    # Preserve non-UTF-8 source bytes losslessly for hashing and JSON round trips.
    return result.stdout.decode("utf-8", errors="surrogateescape")


def indexed_pr_merge_commits(repository: Path, pull_number: int, head: str) -> list[str]:
    """Find durable GitHub merge commits with one history scan per repository."""
    cache_key = str(repository.resolve())
    with _MERGE_INDEX_LOCK:
        index = _MERGE_INDEX.get(cache_key)
        if index is None:
            index = {}
            for line in git(repository, "log", "--all", "--format=%H%x09%P%x09%s").splitlines():
                fields = line.split("\t", 2)
                if len(fields) != 3:
                    continue
                match = re.match(r"Merge pull request #(\d+)\s", fields[2])
                if not match:
                    continue
                sha, parents = fields[0], fields[1].split()
                for parent in parents:
                    index.setdefault((int(match.group(1)), parent), []).append(sha)
            _MERGE_INDEX[cache_key] = index
        return list(index.get((pull_number, head), ()))


def repository_for(ticket: dict, cache: Path, *, fetch: bool) -> Path:
    repository = Path(ticket.get("local_repo_path") or cache / safe_name(ticket["repo"]))
    if not repository.exists():
        if not fetch:
            raise ValueError("Repository not cached; use --fetch to download Git objects")
        repository.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "clone", "--bare", "--filter=blob:none", "--no-tags",
                                 f"https://github.com/{ticket['repo']}.git", str(repository)],
                                capture_output=True, timeout=300)
        if result.returncode:
            raise ValueError("Repository clone failed: " + result.stderr.decode(errors="replace")[:500])
    git(repository, "rev-parse", "--git-dir")
    return repository


def merged_diff(repository: Path, provenance: dict, *, fetch: bool = False) -> str:
    head = provenance.get("pr_head_sha", provenance["merge_commit_sha"])
    for sha in (provenance["base_commit"],):
        try:
            resolved = git(repository, "rev-parse", "--verify", sha + "^{commit}").strip()
        except ValueError:
            if not fetch:
                raise
            git(repository, "fetch", "--no-tags", f"https://github.com/{provenance['repo']}.git", sha)
            resolved = git(repository, "rev-parse", "--verify", sha + "^{commit}").strip()
        if resolved != sha:
            raise ValueError("Resolved Git object differs from requested SHA")

    base = provenance["base_commit"]
    github_merge = provenance["merge_commit_sha"]
    pull_number = int(provenance.get("pull_number") or 0)
    # GitHub's merge_commit_sha for old PRs can name an expired synthetic merge
    # object.  Prefer the durable target-branch merge commit whose message names
    # the PR and whose parent list contains the recorded PR head.
    exact_merges = indexed_pr_merge_commits(repository, pull_number, head) if pull_number else []
    try:
        direct_fields = git(
            repository, "show", "-s", "--format=%H%x09%P%x09%s", github_merge,
        ).strip().split("\t", 2)
    except ValueError:
        direct_fields = []
    if (
        pull_number
        and len(direct_fields) == 3
        and head in direct_fields[1].split()
        and direct_fields[2].startswith(f"Merge pull request #{pull_number} ")
    ):
        exact_merges.append(direct_fields[0])
    if len(set(exact_merges)) > 1:
        merged_at = datetime.fromisoformat(str(provenance["merged_at"]).replace("Z", "+00:00"))
        exact_merges = [min(
            set(exact_merges),
            key=lambda sha: abs(
                datetime.fromisoformat(git(repository, "show", "-s", "--format=%cI", sha).strip())
                - merged_at
            ),
        )]

    if exact_merges:
        merge = exact_merges[0]
        resolution = "target_merge_commit_message_and_pr_head_parent"
    else:
        try:
            merge = git(repository, "rev-parse", "--verify", github_merge + "^{commit}").strip()
        except ValueError:
            if fetch:
                try:
                    git(repository, "fetch", "--no-tags", f"https://github.com/{provenance['repo']}.git", github_merge)
                    merge = git(repository, "rev-parse", "--verify", github_merge + "^{commit}").strip()
                except ValueError:
                    try:
                        git(repository, "rev-parse", "--verify", head + "^{commit}")
                    except ValueError:
                        git(repository, "fetch", "--no-tags", f"https://github.com/{provenance['repo']}.git", head)
                    merge = head
                    resolution = "pr_head_commit_reachable_from_target_history"
            else:
                git(repository, "rev-parse", "--verify", head + "^{commit}")
                merge = head
                resolution = "pr_head_commit_reachable_from_target_history"
        else:
            resolution = "github_merge_commit_sha"

    ref_scopes = (
        ("refs/remotes/origin", "refs/remotes/upstream", "refs/heads/main", "refs/heads/master", "refs/tags")
        if pull_number else ("refs/heads", "refs/remotes")
    )
    containing_refs = git(
        repository, "for-each-ref", "--contains", merge, "--format=%(refname)", *ref_scopes,
    ).splitlines()
    if not containing_refs and resolution == "github_merge_commit_sha":
        commit_time = datetime.fromisoformat(
            git(repository, "show", "-s", "--format=%cI", merge).strip()
        )
        merged_at = datetime.fromisoformat(str(provenance["merged_at"]).replace("Z", "+00:00"))
        if abs(commit_time - merged_at).total_seconds() <= 300:
            resolution = "github_merge_commit_sha_timestamp_attested"
    if not containing_refs and resolution not in {
        "target_merge_commit_message_and_pr_head_parent",
        "github_merge_commit_sha_timestamp_attested",
    }:
        raise ValueError("Resolved repair commit is not reachable from target repository history")
    provenance["github_merge_commit_sha"] = github_merge
    provenance["merge_commit_sha"] = merge
    provenance["merge_commit_resolution"] = resolution
    provenance["merge_commit_containing_refs"] = containing_refs
    merge_base = git(repository, "merge-base", base, merge).strip()
    provenance["base_is_ancestor_of_merge"] = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", base, merge],
        capture_output=True, timeout=300,
    ).returncode == 0
    provenance["merge_base_sha"] = merge_base
    provenance["merge_parents"] = git(repository, "show", "-s", "--format=%P", merge).strip().split()
    provenance["merge_topology"] = "merge_commit" if len(provenance["merge_parents"]) > 1 else "single_parent_squash_or_rebase"
    provenance["base_tree_sha"] = git(repository, "rev-parse", base + "^{tree}").strip()
    provenance["merged_tree_sha"] = git(repository, "rev-parse", merge + "^{tree}").strip()
    # Two endpoints, NOT merge^..merge and NOT the final PR head commit alone.
    # Disable external diff/textconv, rename heuristics and user color settings.
    patch = git(repository, "-c", "core.quotePath=true", "-c", "color.ui=false", "diff",
                "--no-ext-diff", "--no-textconv", "--no-renames", "--binary", "--full-index",
                "--src-prefix=a/", "--dst-prefix=b/", "--unified=3", base, merge, "--")
    if not patch:
        raise ValueError("Merged repair has no changes relative to base_commit")
    provenance["patch_sha256"] = digest(patch)
    return patch


def build_gold(ticket: dict, provenance: dict, patch: str, repository: Path, *, evidence: dict | None, evidence_dir: Path) -> dict:
    ticket_id = str(ticket.get("ticket_id") or ticket.get("instance_id") or "")
    hunks = parse_unified_diff(patch)
    files = list(dict.fromkeys(h.old_file_path or h.new_file_path for h in hunks))
    # Rename detection is disabled: old and new paths are separately visible.
    source = {"ticket_id": ticket_id, "repo": ticket["repo"], "base_commit": ticket["base_commit"],
              "local_repo_path": str(repository), "patch": patch}
    symbols, summary = build_symbol_gold_records([source], repo_cache_dir=repository.parent)
    symbol_rows = [symbol.to_dict() for symbol in symbols]
    for symbol in symbol_rows:
        symbol["provenance"]["source"] = "merged_git_tree_diff"
    row = {"ticket_id": ticket_id, "repo": ticket["repo"], "base_commit": ticket["base_commit"],
           "source_dataset": ticket.get("source_dataset", ""), "source_split": ticket.get("source_split", ""),
           "ground_truth_schema": SCHEMA, "ground_truth_source": "merged_git_tree_diff",
           "ground_truth_status": "unverified", "merged_fix": provenance,
           "merged_patch": patch, "fixed_files": files, "fixed_symbols": [],
           "fixed_symbol_records": [row for row in symbol_rows if row["mapping_status"] == "mapped"],
           "excluded_symbol_records": [row for row in symbol_rows if row["mapping_status"] != "mapped"],
           "symbol_mapping_summary": summary}
    row["fail_to_pass"] = ticket.get("fail_to_pass", ticket.get("FAIL_TO_PASS", []))
    row["pass_to_pass"] = ticket.get("pass_to_pass", ticket.get("PASS_TO_PASS", []))
    refresh_evidence(row, ticket, evidence=evidence, evidence_dir=evidence_dir)
    row["gold_content_sha256"] = digest(json.dumps({key: row[key] for key in
        ("fixed_files", "fixed_symbol_records", "excluded_symbol_records")}, sort_keys=True, ensure_ascii=False))
    return row


def refresh_evidence(row: dict, ticket: dict, *, evidence: dict | None, evidence_dir: Path) -> None:
    provenance = row["merged_fix"]
    row["ground_truth_status"] = "unverified"
    row.pop("ground_truth_error", None)
    provenance.pop("repair_verification", None)
    row["fail_to_pass"] = ticket.get("fail_to_pass", ticket.get("FAIL_TO_PASS", []))
    row["pass_to_pass"] = ticket.get("pass_to_pass", ticket.get("PASS_TO_PASS", []))
    if evidence is None:
        row["ground_truth_error"] = "Missing before/after repair test evidence for exact merged commit"
    else:
        try:
            provenance["repair_verification"] = validate_repair_evidence(ticket, provenance, evidence, evidence_dir=evidence_dir)
            row["ground_truth_status"] = "verified"
        except ValueError as exc:
            row["ground_truth_error"] = str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", required=True)
    parser.add_argument("--output-dir", required=True, help="New dataset directory; historical Gold is not overwritten")
    parser.add_argument("--repo-cache-dir", required=True)
    parser.add_argument("--evidence-dir", help="Trusted runner receipts <ticket_id>.json plus before/after logs")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse intact merged diff/mapping rows; always recheck current test receipts")
    parser.add_argument("--metadata-only", action="store_true", help="Inventory all merged PRs without deriving untested Git diffs")
    parser.add_argument("--workers", type=int, default=1, help="Independent ticket workers; fetches are serialized per repository")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="test", choices=("test", "dev", "development", "validation", "frozen_holdout"))
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be between 1 and 16")
    tickets = read_jsonl(Path(args.tickets))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tickets = tickets[:args.limit]
    ids = [str(row.get("ticket_id") or row.get("instance_id") or "") for row in tickets]
    if not tickets or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("Unique nonempty input ticket IDs required")
    if len({safe_name(value) for value in ids}) != len(ids):
        raise ValueError("Ticket IDs collide after cache path normalization")
    output = Path(args.output_dir)
    if output.resolve() == Path(args.tickets).resolve().parent:
        raise ValueError("Use a new output directory to preserve historical datasets")
    output.mkdir(parents=True, exist_ok=True)
    existing_gold = output / f"{args.split}_gold.jsonl"
    previous_rows = {}
    if existing_gold.is_file():
        with existing_gold.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    previous = json.loads(line)
                    if previous.get("ground_truth_schema") != SCHEMA:
                        raise ValueError("Refusing to overwrite historical Gold; choose a new output directory")
                    if args.resume:
                        previous_rows[previous["ticket_id"]] = previous
    client = GitHub(output / "github", offline=args.offline)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else output / "repair_evidence"
    repository_locks = {ticket["repo"]: Lock() for ticket in tickets}

    def process_ticket(pair):
        ticket, ticket_id = pair
        row = {"ticket_id": ticket_id, "repo": ticket.get("repo"), "base_commit": ticket.get("base_commit"),
               "ground_truth_schema": SCHEMA, "ground_truth_status": "unverified", "fixed_files": []}
        try:
            repo, number = pull_identity(ticket)
            provenance = validate_pull(ticket, client.pull(repo, number))
            row["merged_fix"] = provenance
            if args.metadata_only:
                raise ValueError("Metadata inventory only: Git diff and repair verification pending")
            path = evidence_dir / (safe_name(ticket_id) + ".json")
            evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            previous = previous_rows.get(ticket_id, {})
            checkpoint = output / "records" / (safe_name(ticket_id) + ".json")
            if args.resume and checkpoint.is_file():
                previous = json.loads(checkpoint.read_text(encoding="utf-8"))
            content = {key: previous.get(key) for key in ("fixed_files", "fixed_symbol_records", "excluded_symbol_records")}
            previous_fix = previous.get("merged_fix", {})
            previous_github_merge = previous_fix.get("github_merge_commit_sha", previous_fix.get("merge_commit_sha"))
            content_valid = (
                digest(json.dumps(content, sort_keys=True, ensure_ascii=False))
                == previous.get("gold_content_sha256")
            )
            reusable = (previous.get("merged_patch") and previous_fix.get("merge_commit_resolution") and
                        all(previous_fix.get(key) == provenance[key] for key in ("repo", "base_commit", "pr_head_sha")) and
                        previous_github_merge == provenance["merge_commit_sha"] and
                        digest(previous["merged_patch"]) == previous["merged_fix"].get("patch_sha256") and
                        content_valid)
            if reusable:
                row = previous
                refresh_evidence(row, ticket, evidence=evidence, evidence_dir=evidence_dir)
            else:
                if args.fetch:
                    # Fetch mutates shared Git object/ref state, so keep the whole
                    # resolve+diff operation serialized for that repository.
                    with repository_locks[ticket["repo"]]:
                        repository = repository_for(ticket, Path(args.repo_cache_dir), fetch=True)
                        patch = merged_diff(repository, provenance, fetch=True)
                else:
                    # Offline Git object reads are independent and safe to parallelize.
                    repository = repository_for(ticket, Path(args.repo_cache_dir), fetch=False)
                    patch = merged_diff(repository, provenance, fetch=args.fetch)
                if (
                    previous.get("ground_truth_schema") == SCHEMA
                    and previous.get("merged_patch") == patch
                    and content_valid
                ):
                    # The durable commit resolution changed only provenance; an
                    # identical endpoint patch has identical file/symbol Gold.
                    row = previous
                    row["merged_fix"] = provenance
                    row["merged_patch"] = patch
                    refresh_evidence(row, ticket, evidence=evidence, evidence_dir=evidence_dir)
                else:
                    row = build_gold(ticket, provenance, patch, repository, evidence=evidence, evidence_dir=evidence_dir)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            row["ground_truth_error"] = str(exc)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(process_ticket, zip(tickets, ids)), 1):
            rows.append(row)
            write_json(output / "records" / (safe_name(row["ticket_id"]) + ".json"), row)
            print(json.dumps({"progress": f"{index}/{len(tickets)}", "ticket_id": row["ticket_id"],
                              "status": row["ground_truth_status"], "error": row.get("ground_truth_error")}), flush=True)
    # One row per input, including failures. Never silently improve metrics by dropping unresolved rows.
    private_fields = {"patch", "developer_patch", "test_patch", "merged_patch", "merged_fix", "fixed_files",
                      "fixed_symbols", "fixed_symbol_records", "excluded_symbol_records", "ground_truth", "json_ground_truth"}
    public_tickets = [{key: value for key, value in ticket.items() if key not in private_fields} for ticket in tickets]
    for name, records in ((f"{args.split}_gold.jsonl", rows), (f"{args.split}_tickets.jsonl", public_tickets)):
        temporary = output / (name + ".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records), encoding="utf-8")
        temporary.replace(output / name)
    verified = sum(row["ground_truth_status"] == "verified" for row in rows)
    write_json(output / f"{args.split}_manifest.json", {"ground_truth_schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_tickets": str(Path(args.tickets).resolve()),
        "source_sha256": digest(Path(args.tickets).read_bytes()), "rows": len(rows), "verified_rows": verified,
        "merged_pr_rows": sum(row.get("merged_fix", {}).get("merged") is True for row in rows),
        "derived_diff_rows": sum(bool(row.get("merged_patch")) for row in rows),
        "mapped_symbol_records": sum(len(row.get("fixed_symbol_records", [])) for row in rows),
        "excluded_symbol_records": sum(len(row.get("excluded_symbol_records", [])) for row in rows),
        "unverified_rows": len(rows) - verified, "ready_for_evaluation": verified == len(rows),
        "gold_path": str(output / f"{args.split}_gold.jsonl"), "gold_sha256": digest((output / f"{args.split}_gold.jsonl").read_bytes()),
        "tickets_path": str(output / f"{args.split}_tickets.jsonl"), "metadata_only": args.metadata_only})
    return 0 if verified == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
