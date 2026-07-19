#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    ROOT / "data" / "fault_localization" / "swebench_lite" / "indexes",
    ROOT / "data" / "fault_localization" / "swebench_lite" / "embedding_cache",
    ROOT / "data" / "fault_localization" / "swebench_lite" / "repos",
    ROOT / "reports" / "fault_localization" / "index_cache",
    ROOT / "reports" / "fault_localization" / "jobs",
)


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    size_bytes: int
    modified_at: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and prune generated fault-localization caches.")
    parser.add_argument("--root", action="append", dest="roots", help="Cache root under this project; repeatable.")
    parser.add_argument("--max-age-days", type=float, default=30.0)
    parser.add_argument("--max-total-gb", type=float, default=8.0)
    parser.add_argument("--apply", action="store_true", help="Delete planned entries. Default is a read-only dry run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = [Path(value).expanduser().resolve() for value in args.roots] if args.roots else list(DEFAULT_ROOTS)
    for root in roots:
        try:
            root.relative_to(ROOT)
        except ValueError as exc:
            raise SystemExit(f"cache root must stay inside project: {root}") from exc
    entries = [entry for root in roots for entry in scan_cache_root(root)]
    planned = plan_cleanup(
        entries,
        max_age_seconds=max(0.0, args.max_age_days) * 86400,
        max_total_bytes=max(0, round(args.max_total_gb * 1024**3)),
        now=time.time(),
    )
    removed_bytes = 0
    removed: list[str] = []
    if args.apply:
        for entry in planned:
            if entry.path.is_dir() and not entry.path.is_symlink():
                shutil.rmtree(entry.path)
            else:
                entry.path.unlink(missing_ok=True)
            removed_bytes += entry.size_bytes
            removed.append(str(entry.path))
    report = {
        "mode": "apply" if args.apply else "dry_run",
        "roots": [str(root) for root in roots],
        "entries": len(entries),
        "current_size_bytes": sum(entry.size_bytes for entry in entries),
        "planned_entries": len(planned),
        "planned_bytes": sum(entry.size_bytes for entry in planned),
        "removed_entries": removed,
        "removed_bytes": removed_bytes,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def scan_cache_root(root: Path) -> Iterable[CacheEntry]:
    if not root.exists() or root.is_symlink():
        return []
    entries: list[CacheEntry] = []
    for path in root.iterdir():
        if path.is_symlink():
            continue
        try:
            stat = path.stat()
            entries.append(CacheEntry(path=path, size_bytes=_entry_size(path), modified_at=stat.st_mtime))
        except OSError:
            continue
    return entries


def plan_cleanup(
    entries: list[CacheEntry],
    *,
    max_age_seconds: float,
    max_total_bytes: int,
    now: float,
) -> list[CacheEntry]:
    selected: dict[Path, CacheEntry] = {
        entry.path: entry for entry in entries if max_age_seconds == 0 or now - entry.modified_at > max_age_seconds
    }
    remaining_size = sum(entry.size_bytes for entry in entries if entry.path not in selected)
    if remaining_size > max_total_bytes:
        remaining = sorted(
            (entry for entry in entries if entry.path not in selected),
            key=lambda entry: (entry.modified_at, str(entry.path)),
        )
        for entry in remaining:
            if remaining_size <= max_total_bytes:
                break
            selected[entry.path] = entry
            remaining_size -= entry.size_bytes
    return sorted(selected.values(), key=lambda entry: (entry.modified_at, str(entry.path)))


def _entry_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink() or not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


if __name__ == "__main__":
    raise SystemExit(main())
