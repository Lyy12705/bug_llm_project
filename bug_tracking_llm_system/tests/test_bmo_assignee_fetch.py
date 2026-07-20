from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT_ROOT = PROJECT_ROOT / "assignee_triage_accuracy" / "paper_grade" / "scripts"
if str(FETCH_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(FETCH_SCRIPT_ROOT))

import fetch_bmo_assignee_dataset as fetcher  # noqa: E402


class BmoAssigneeFetchTests(unittest.TestCase):
    def test_failed_sealed_fetch_does_not_leave_an_empty_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "holdout.jsonl"
            manifest = root / "manifest.json"
            argv = [
                "fetch_bmo_assignee_dataset.py",
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--seal-output",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                fetcher, "search_bugs", side_effect=RuntimeError("network unavailable")
            ):
                with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                    fetcher.main()

            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual(list(root.glob("*.partial")), [])

    def test_capped_sealed_fetch_records_hash_and_termination_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "holdout.jsonl"
            manifest = root / "manifest.json"
            bug = {
                "id": 1,
                "summary": "example",
                "creation_time": "2022-01-01T00:00:00Z",
            }
            argv = [
                "fetch_bmo_assignee_dataset.py",
                "--output",
                str(output),
                "--manifest",
                str(manifest),
                "--max-bugs",
                "1",
                "--seal-output",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                fetcher, "search_bugs", return_value=[bug]
            ):
                with redirect_stdout(io.StringIO()):
                    fetcher.main()

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["termination_reason"], "max_bugs_reached")
            self.assertFalse(payload["complete"])
            self.assertTrue(payload["sealed_output"])
            self.assertEqual(len(payload["output_sha256"]), 64)
            self.assertEqual(output.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
