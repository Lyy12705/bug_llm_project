"""Real local-model smoke: synthetic source, nonempty suffix, actual tests."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fim_patch import FIMPatchGenerator
from fim_patch.generator import git
from fim_patch.validation import UnittestValidator
from fim_patch.pipeline import RepairPipeline, write_run


def main():
    repo = ROOT / "runs" / "smoke" / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    if (repo / ".git").exists():
        raise SystemExit("Smoke repo already exists; use the saved CLI inputs to rerun.")
    (repo / "username.py").write_text(
        'def normalize_username(value):\n    return value.strip().lower()\n\n\n'
        'def display_username(value):\n    return normalize_username(value)\n', encoding="utf-8", newline="\n")
    (repo / "test_username.py").write_text(
        'import unittest\nfrom username import normalize_username, display_username\n\n'
        'class Reproducer(unittest.TestCase):\n'
        '    def test_none(self):\n        self.assertEqual(normalize_username(None), "")\n\n'
        'class Regression(unittest.TestCase):\n'
        '    def test_string(self):\n        self.assertEqual(normalize_username(" Alice "), "alice")\n'
        '    def test_suffix(self):\n        self.assertEqual(display_username(" BOB "), "bob")\n',
        encoding="utf-8", newline="\n")
    git(repo, "init")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=FIM Smoke", "-c", "user.email=smoke@example.invalid", "commit", "-m", "Synthetic buggy base")
    base = git(repo, "rev-parse", "HEAD").decode().strip()
    ticket = {"ticket_id": "FIM-SMOKE-1", "base_commit": base,
              "title": "normalize_username crashes on None",
              "logs": 'File "username.py", line 2, in normalize_username: AttributeError',
              "description": 'normalize_username(None) must return "". For string inputs, strip whitespace and convert to lowercase.'}
    spec = {"reproducer": {"names": ["test_username.Reproducer"]},
            "regression": {"names": ["test_username.Regression"]}, "expected_failure": "AttributeError"}
    for name, obj in (("ticket", ticket), ("test_spec", spec)):
        (repo.parent / (name + ".json")).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    location, result = RepairPipeline(FIMPatchGenerator(validator=UnittestValidator(spec))).run(ticket, str(repo))
    (repo.parent / 'localization.json').write_text(json.dumps(location, ensure_ascii=False, indent=2), encoding='utf-8')
    write_run(repo.parent / 'pipeline_output', ticket, location, result)
    (repo.parent / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (repo.parent / "result.diff").write_bytes(result["patch"].encode("utf-8"))
    assert git(repo, "status", "--porcelain") == b"", "Source repository was modified"
    print(json.dumps({k: result[k] for k in ("patch_status", "validation_status", "modified_files")}))
    for c in result["candidates"]:
        print(c["candidate_id"], c["status"], c.get("reason", ""))
    raise SystemExit(0 if result["validation_status"] == "plausible" else 1)


if __name__ == "__main__":
    main()
