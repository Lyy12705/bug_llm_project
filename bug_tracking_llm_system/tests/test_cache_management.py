from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from manage_artifact_cache import CacheEntry, plan_cleanup, scan_cache_root


class ArtifactCacheTests(unittest.TestCase):
    def test_cleanup_prefers_expired_then_oldest_until_under_quota(self) -> None:
        entries = [
            CacheEntry(Path("expired"), 40, 10.0),
            CacheEntry(Path("old"), 50, 80.0),
            CacheEntry(Path("new"), 60, 95.0),
        ]

        selected = plan_cleanup(entries, max_age_seconds=50.0, max_total_bytes=60, now=100.0)

        self.assertEqual([entry.path.name for entry in selected], ["expired", "old"])

    def test_scan_does_not_follow_symlinked_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "owned.bin").write_bytes(b"1234")
            outside = root / "outside.bin"
            outside.write_bytes(b"secret")
            link = cache / "outside-link"
            try:
                os.symlink(outside, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable on this platform")

            entries = list(scan_cache_root(cache))

        self.assertEqual([entry.path.name for entry in entries], ["owned.bin"])
        self.assertEqual(entries[0].size_bytes, 4)


if __name__ == "__main__":
    unittest.main()
