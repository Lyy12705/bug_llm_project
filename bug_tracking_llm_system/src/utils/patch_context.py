"""Prepare Stage-3 source context without silently truncating repair targets."""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


class PatchContextError(ValueError):
    pass


def build_stage3_context(location: dict[str, Any], repo_path: str) -> dict[str, Any]:
    candidates = location.get("stage3_ranked_symbols")
    if not isinstance(candidates, list) or not candidates:
        raise PatchContextError("Stage-3 returned no symbols; review the file candidates first.")
    root = Path(repo_path).resolve()
    base = str(location.get("base_commit") or "")
    if base and base != "working-tree":
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", base):
            raise PatchContextError("base_commit must be a commit SHA.")
        head = _git(root, "rev-parse", "HEAD").decode().strip()
        expected = _git(root, "rev-parse", "--verify", base + "^{commit}").decode().strip()
        if head != expected:
            raise PatchContextError("Repository HEAD does not match localization base_commit.")
        base = expected
    files: dict[str, dict[str, Any]] = {}
    symbols = []
    total_chars = 0
    fingerprints = location.get("source_file_sha256") or {}
    for rank, row in enumerate(candidates[:5], 1):
        if not isinstance(row, dict):
            raise PatchContextError("Invalid Stage-3 symbol record.")
        relative = str(row.get("file_path") or "").replace("\\", "/")
        parts = PurePosixPath(relative)
        if not relative or parts.is_absolute() or ".." in parts.parts or ":" in relative:
            raise PatchContextError("Invalid Stage-3 source path.")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise PatchContextError("Stage-3 source is missing or outside the repository.")
        if relative not in files:
            if path.stat().st_size > 500_000:
                raise PatchContextError("Source file exceeds the 500 KB context limit; manual selection required.")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if fingerprints.get(relative) and fingerprints[relative] != digest:
                raise PatchContextError("Source file changed after localization.")
            if base and base != "working-tree":
                original = _git(root, "show", f"{base}:{relative}")
                if original.replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
                    raise PatchContextError("Source file differs from localization base_commit.")
            try:
                code = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise PatchContextError("Source file is not UTF-8; explicit decoding is required.") from exc
            total_chars += len(code)
            if total_chars > 100_000:
                raise PatchContextError("Complete source exceeds the 100,000 character budget; manual selection required.")
            files[relative] = {"file_path": relative, "code_text": code, "sha256": digest,
                               "context_start_line": 1, "context_end_line": len(code.splitlines()),
                               "context_strategy": "complete_file", "truncated": False}
        try:
            start, end = int(row["start_line"]), int(row["end_line"])
        except (KeyError, ValueError, TypeError) as exc:
            raise PatchContextError("Stage-3 symbol has invalid line coordinates.") from exc
        if not 1 <= start <= end <= files[relative]["context_end_line"]:
            raise PatchContextError("Stage-3 symbol range is outside the source file.")
        symbols.append({"rank": rank, "file_path": relative,
                        "symbol_qualified_name": row.get("symbol_qualified_name", ""),
                        "symbol_kind": row.get("symbol_kind", ""),
                        "start_line": start, "end_line": end,
                        "score": row.get("score"), "reason": row.get("reason", "")})
    return {"schema_version": "stage3-patch-context-v1", "base_commit": base or "working-tree",
            "snapshot_verified": bool((base and base != "working-tree") or all(fingerprints.get(f) for f in files)),
            "symbols": symbols, "files": list(files.values())}


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise PatchContextError("Cannot verify the localization repository snapshot.") from exc
