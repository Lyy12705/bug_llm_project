#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.fault_localization import build_code_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a code chunk index for fault localization.")
    parser.add_argument("--repo-path", required=True, help="Repository or source folder to index.")
    parser.add_argument("--output", required=True, help="Output JSON code index path.")
    parser.add_argument("--chunk-lines", type=int, default=80, help="Maximum lines per code chunk.")
    parser.add_argument("--overlap-lines", type=int, default=20, help="Overlap for fixed-size chunks.")
    parser.add_argument("--include-tests", action="store_true", help="Include test/spec folders in the index.")
    parser.add_argument("--max-file-bytes", type=int, default=500_000, help="Skip source files larger than this value.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    index = build_code_index(
        args.repo_path,
        chunk_lines=args.chunk_lines,
        overlap_lines=args.overlap_lines,
        include_tests=args.include_tests,
        max_file_bytes=args.max_file_bytes,
    )
    index.save(args.output)
    summary = {
        "repository_path": index.repository_path,
        "chunks": len(index.chunks),
        "output": str(Path(args.output)),
        "settings": index.settings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
