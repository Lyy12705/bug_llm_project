from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts.prepare_merged_fault_localization_gold import build_gold, git, main, merged_diff, write_json
from scripts.evaluate_fault_localization import evaluate_records
from utils.merged_ground_truth import digest, pull_identity, validate_gold, validate_pull


class MergedGoldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "Gold test")
        git(self.repo, "config", "user.email", "test@example.com")
        self.write("service.py", "def answer():\n    return 0\n")
        self.write("removed.py", "VALUE = 1\n")
        self.base = self.commit("base")
        self.ticket = {"ticket_id": "example__project-7", "repo": "example/project", "base_commit": self.base,
                       "fail_to_pass": ["test_answer"], "pass_to_pass": ["test_existing"], "local_repo_path": str(self.repo)}

    def write(self, name, content):
        (self.repo / name).write_text(content, encoding="utf-8")

    def commit(self, message):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-m", message)
        return git(self.repo, "rev-parse", "HEAD").strip()

    def merged(self, *, squash=False):
        git(self.repo, "checkout", "-b", "repair")
        self.write("service.py", "def answer():\n    return 1\n")
        self.commit("intermediate wrong fix")
        self.write("service.py", "def answer():\n    return 2\n")
        self.write("second.py", "NEW = True\n")
        self.head = self.commit("final repair")
        git(self.repo, "checkout", "main")
        if squash:
            git(self.repo, "merge", "--squash", "repair")
            merge = self.commit("squashed repair")
        else:
            git(self.repo, "merge", "--no-ff", "repair", "-m", "merged repair")
            merge = git(self.repo, "rev-parse", "HEAD").strip()
        self.pull = {"number": 7, "state": "closed", "merged": True, "merged_at": "2026-01-01T00:00:00Z",
                     "merge_commit_sha": merge, "head": {"sha": self.head},
                     "base": {"repo": {"full_name": "example/project"}, "ref": "main"}}
        provenance = validate_pull(self.ticket, self.pull)
        patch = merged_diff(self.repo, provenance)
        return provenance, patch

    def receipt(self, provenance):
        # Execute the same small regression suite against actual before/after Git source.
        evidence = {"schema_version": "merged-repair-test-evidence-v1", "ticket_id": self.ticket["ticket_id"],
                    "repo": self.ticket["repo"], "base_commit": self.base,
                    "merge_commit_sha": provenance["merge_commit_sha"], "patch_sha256": provenance["patch_sha256"],
                    "runner": "unittest-fixture-runner", "environment": sys.version,
                    "test_suite_sha256": digest("answer() == 2; existing test passes"),
                    "fail_to_pass": ["test_answer"], "pass_to_pass": ["test_existing"]}
        for phase, sha in (("before", self.base), ("after", provenance["merge_commit_sha"])):
            source = git(self.repo, "show", sha + ":service.py")
            result = subprocess.run([sys.executable, "-c", source + "\nassert answer() == 2\n"], capture_output=True)
            log = result.stdout + result.stderr + b"\ntest_existing PASSED\n"
            log_path = self.root / (phase + ".log")
            log_path.write_bytes(log)
            evidence[phase] = {"commit_sha": sha, "completed": True, "command": [sys.executable, "-c", "assert answer() == 2"],
                               "tests": {"test_answer": "PASSED" if result.returncode == 0 else "FAILED", "test_existing": "PASSED"},
                               "log_path": log_path.name, "log_sha256": digest(log)}
        return evidence

    def gold(self, *, squash=False):
        provenance, patch = self.merged(squash=squash)
        return build_gold(self.ticket, provenance, patch, self.repo, evidence=self.receipt(provenance), evidence_dir=self.root)

    def test_normal_multi_commit_uses_cumulative_merged_tree(self):
        row = self.gold()
        validate_gold(row)
        self.assertIn("-    return 0", row["merged_patch"])
        self.assertIn("+    return 2", row["merged_patch"])
        self.assertNotIn("return 1", row["merged_patch"])
        self.assertEqual(row["merged_fix"]["merge_topology"], "merge_commit")
        self.assertEqual(row["fixed_files"], ["second.py", "service.py"])
        self.assertTrue(any(s["qualified_name"] == "answer" for s in row["fixed_symbol_records"]))
        self.assertTrue(any(s["exclusion_reason"] == "new_file" for s in row["excluded_symbol_records"]))

    def test_squash_uses_merged_sha_not_final_head_sha(self):
        row = self.gold(squash=True)
        self.assertNotEqual(row["merged_fix"]["merge_commit_sha"], self.head)
        self.assertEqual(row["merged_fix"]["merge_topology"], "single_parent_squash_or_rebase")
        validate_gold(row)

    def test_durable_target_merge_resolves_stale_github_merge_sha(self):
        git(self.repo, "checkout", "-b", "repair")
        self.write("service.py", "def answer():\n    return 2\n")
        head = self.commit("final repair")
        git(self.repo, "checkout", "main")
        git(self.repo, "merge", "--no-ff", "repair", "-m", "Merge pull request #7 from example/repair")
        actual_merge = git(self.repo, "rev-parse", "HEAD").strip()
        provenance = {
            "repo": "example/project", "pull_number": 7, "base_commit": self.base,
            "merge_commit_sha": head, "pr_head_sha": head,
        }
        merged_diff(self.repo, provenance)
        self.assertEqual(provenance["github_merge_commit_sha"], head)
        self.assertEqual(provenance["merge_commit_sha"], actual_merge)
        self.assertEqual(provenance["merge_commit_resolution"], "target_merge_commit_message_and_pr_head_parent")

    def test_rebase_like_single_parent_tip_includes_all_commits(self):
        git(self.repo, "checkout", "-b", "repair")
        self.write("service.py", "def answer():\n    return 2\n")
        self.commit("first repair change")
        self.write("second.py", "VALUE = 3\n")
        merge = self.commit("second repair change")
        provenance = {"base_commit": self.base, "merge_commit_sha": merge}
        patch = merged_diff(self.repo, provenance)
        self.assertIn("service.py", patch)
        self.assertIn("second.py", patch)

    def test_unmerged_and_conflicting_identity_rejected(self):
        self.merged()
        self.pull["merged"] = False
        with self.assertRaisesRegex(ValueError, "not actually merged"):
            validate_pull(self.ticket, self.pull)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            pull_identity({**self.ticket, "pull_number": 8})

    def test_missing_test_evidence_never_scores(self):
        provenance, patch = self.merged()
        row = build_gold(self.ticket, provenance, patch, self.repo, evidence=None, evidence_dir=self.root)
        with self.assertRaisesRegex(ValueError, "Unverified"):
            evaluate_records([row], [])

    def test_failed_final_test_or_tampered_log_never_verifies(self):
        provenance, patch = self.merged()
        evidence = self.receipt(provenance)
        evidence["after"]["tests"]["test_answer"] = "FAILED"
        row = build_gold(self.ticket, provenance, patch, self.repo, evidence=evidence, evidence_dir=self.root)
        self.assertEqual(row["ground_truth_status"], "unverified")
        evidence = self.receipt(provenance)
        (self.root / "after.log").write_text("tampered")
        row = build_gold(self.ticket, provenance, patch, self.repo, evidence=evidence, evidence_dir=self.root)
        self.assertIn("hash mismatch", row["ground_truth_error"])

    def test_wrong_commit_receipt_and_no_fail_to_pass_rejected(self):
        provenance, patch = self.merged()
        evidence = self.receipt(provenance)
        evidence["merge_commit_sha"] = self.base
        row = build_gold(self.ticket, provenance, patch, self.repo, evidence=evidence, evidence_dir=self.root)
        self.assertEqual(row["ground_truth_status"], "unverified")
        evidence = self.receipt(provenance)
        row = build_gold({**self.ticket, "fail_to_pass": []}, provenance, patch, self.repo, evidence=evidence, evidence_dir=self.root)
        self.assertIn("No FAIL_TO_PASS", row["ground_truth_error"])

    def test_metrics_use_merged_file_and_symbol_gold_and_missing_predictions(self):
        row = self.gold()
        prediction = {"ticket_id": row["ticket_id"], "localized_files": [{"file_path": "service.py"}],
                      "stage1_candidate_files": ["service.py"], "stage3_ranked_symbols": row["fixed_symbol_records"]}
        metrics = evaluate_records([row], [prediction])
        self.assertEqual(metrics["file_mrr"], 1)
        self.assertEqual(metrics["candidate_recall_at_20"], .5)
        self.assertEqual(metrics["symbol_mrr"], 1)
        self.assertEqual(metrics["ground_truth"]["verified_merged_rows"], 1)
        self.assertEqual(evaluate_records([row], [])["file_mrr"], 0)
        altered = copy.deepcopy(row)
        altered["fixed_files"] = ["wrong.py"]
        with self.assertRaisesRegex(ValueError, "content hash"):
            evaluate_records([altered], [prediction])

    def test_legacy_requires_explicit_opt_in_and_cannot_mix(self):
        legacy = {"ticket_id": "old", "fixed_files": ["x.py"]}
        with self.assertRaisesRegex(ValueError, "Legacy/mixed"):
            evaluate_records([legacy], [])
        self.assertEqual(evaluate_records([legacy], [], allow_legacy_gold=True)["ground_truth"]["policy"], "legacy_developer_patch")
        with self.assertRaisesRegex(ValueError, "Legacy/mixed"):
            evaluate_records([self.gold(), legacy], [], allow_legacy_gold=True)

    def test_new_file_matching_is_exact_and_wrong_snapshot_is_rejected(self):
        row = self.gold()
        prediction = {"ticket_id": row["ticket_id"], "localized_files": [{"file_path": "other/service.py"}]}
        self.assertEqual(evaluate_records([row], [prediction])["file_mrr"], 0)
        prediction["base_commit"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "Prediction base_commit"):
            evaluate_records([row], [prediction])

    def test_existing_legacy_output_cannot_be_overwritten(self):
        tickets = self.root / "tickets.jsonl"
        tickets.write_text(json.dumps(self.ticket) + "\n", encoding="utf-8")
        output = self.root / "out"
        output.mkdir()
        original = '{"ticket_id":"old","fixed_files":["x.py"]}\n'
        (output / "test_gold.jsonl").write_text(original)
        with self.assertRaisesRegex(ValueError, "historical Gold"):
            main(["--tickets", str(tickets), "--output-dir", str(output), "--repo-cache-dir", str(self.root), "--offline"])
        self.assertEqual((output / "test_gold.jsonl").read_text(), original)

    def test_stage3_consumers_and_paired_outcomes_read_combined_gold(self):
        from scripts.run_stage3_wp4_symbol_llm_pilot import gold_items_for_ticket
        from scripts.analyze_stage3_symbol_pool_oracle import _gold_by_ticket
        from scripts.run_stage2_paired_comparison import _paired_rank_outcomes
        row = self.gold()
        items = gold_items_for_ticket([row], row["ticket_id"])
        self.assertTrue(items)
        self.assertEqual(items, _gold_by_ticket([row])[row["ticket_id"]])
        prediction = {"ticket_id": row["ticket_id"], "stage3_ranked_symbols": row["fixed_symbol_records"]}
        outcomes = _paired_rank_outcomes([row], [], [prediction], level="symbol")
        self.assertEqual(outcomes["improved"]["count"], 1)

    def test_resume_rechecks_changed_test_evidence(self):
        row = self.gold()
        output = self.root / "out"
        tickets = self.root / "tickets.jsonl"
        tickets.write_text(json.dumps(self.ticket) + "\n", encoding="utf-8")
        write_json(output / "github" / "example__project" / "pull-7.json", self.pull)
        write_json(output / "records" / (row["ticket_id"] + ".json"), row)
        # No receipt directory on resume: a previously verified checkpoint must not remain verified.
        result = main(["--tickets", str(tickets), "--output-dir", str(output), "--repo-cache-dir", str(self.root), "--offline", "--resume", "--workers", "2"])
        self.assertEqual(result, 2)
        restored = json.loads((output / "test_gold.jsonl").read_text())
        self.assertEqual(restored["ground_truth_status"], "unverified")
        self.assertNotIn("repair_verification", restored["merged_fix"])

    def test_offline_cli_preserves_unresolved_rows_and_exit_status(self):
        self.merged()
        tickets = self.root / "tickets.jsonl"
        tickets.write_text(json.dumps(self.ticket) + "\n", encoding="utf-8")
        output = self.root / "out"
        write_json(output / "github" / "example__project" / "pull-7.json", self.pull)
        result = main(["--tickets", str(tickets), "--output-dir", str(output), "--repo-cache-dir", str(self.root), "--offline"])
        self.assertEqual(result, 2)
        manifest = json.loads((output / "test_manifest.json").read_text())
        self.assertEqual(manifest["rows"], 1)
        self.assertFalse(manifest["ready_for_evaluation"])
        row = json.loads((output / "test_gold.jsonl").read_text())
        self.assertEqual(row["ground_truth_status"], "unverified")

    def test_deleted_binary_and_unicode_paths_are_retained(self):
        (self.repo / "removed.py").unlink()
        (self.repo / "asset.bin").write_bytes(b"\x00\x01\x02")
        self.write("含 空白.py", "VALUE = 2\n")
        merge = self.commit("paths")
        patch = merged_diff(self.repo, {"base_commit": self.base, "merge_commit_sha": merge})
        from utils.symbol_gold import parse_unified_diff
        paths = {h.old_file_path or h.new_file_path for h in parse_unified_diff(patch)}
        self.assertEqual(paths, {"removed.py", "asset.bin", "含 空白.py"})

    def test_non_utf8_source_diff_roundtrips_without_changing_hash(self):
        (self.repo / "legacy.txt").write_bytes(b"Gr\xfc\xdfe\n")
        merge = self.commit("legacy encoded source")
        provenance = {"base_commit": self.base, "merge_commit_sha": merge}
        patch = merged_diff(self.repo, provenance)
        path = self.root / "patch.json"
        write_json(path, {"patch": patch})
        restored = json.loads(path.read_text(encoding="utf-8"))["patch"]
        self.assertEqual(digest(restored), provenance["patch_sha256"])
        self.assertIn(b"Gr\xfc\xdfe", restored.encode("utf-8", errors="surrogateescape"))


if __name__ == "__main__":
    unittest.main()
