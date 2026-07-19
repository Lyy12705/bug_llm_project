from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.embedding_utils import tokenize


CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs"}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "venv", ".venv", "data", "models", "reports"}


@dataclass(frozen=True)
class CodeCandidate:
    file: str
    score: float
    line_start: int = 1
    line_end: int = 1
    function: str = ""
    reason: str = ""


def retrieve_code_candidates(ticket: dict, repo_path: str | Path, *, top_k: int = 5) -> list[CodeCandidate]:
    root = Path(repo_path).resolve()
    if not root.exists():
        raise ValueError(f"Repository path does not exist: {root}")

    stack_candidates = _stack_trace_candidates(ticket, root)
    keyword_candidates = _keyword_candidates(ticket, root)
    combined: dict[str, CodeCandidate] = {}

    for candidate in [*stack_candidates, *keyword_candidates]:
        current = combined.get(candidate.file)
        if current is None or candidate.score > current.score:
            combined[candidate.file] = candidate

    return sorted(combined.values(), key=lambda item: item.score, reverse=True)[:top_k]


def read_context(
    repo_path: str | Path,
    relative_file: str,
    line_start: int,
    line_end: int,
    *,
    padding: int = 12,
    max_file_bytes: int = 500_000,
) -> str:
    root = Path(repo_path).resolve()
    path = (root / relative_file).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Context path escapes repository root: {relative_file}") from exc
    if not path.is_file():
        raise ValueError(f"Context path is not a file: {relative_file}")
    if path.stat().st_size > max_file_bytes:
        raise ValueError(f"Context file exceeds {max_file_bytes} byte limit: {relative_file}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(1, line_start - padding)
    end = min(len(lines), line_end + padding)
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def _iter_code_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if path.is_symlink():
            continue
        yield path


def _stack_trace_candidates(ticket: dict, root: Path) -> list[CodeCandidate]:
    text = "\n".join(str(ticket.get(key) or "") for key in ("error_message", "logs", "description"))
    candidates: list[CodeCandidate] = []
    patterns = [
        r"File \"(?P<file>[^\"]+)\", line (?P<line>\d+)",
        r"(?P<file>[A-Za-z0-9_./\\-]+\.(?:py|js|ts|tsx|jsx|java|c|cc|cpp|go|rs)):(?P<line>\d+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw_file = match.group("file").replace("\\", "/")
            line = int(match.group("line"))
            path = _resolve_stack_file(root, raw_file)
            if path is None:
                continue
            rel = path.relative_to(root).as_posix()
            function = _nearest_function(path, line)
            candidates.append(
                CodeCandidate(
                    file=rel,
                    score=1.0,
                    line_start=max(1, line - 3),
                    line_end=line + 3,
                    function=function,
                    reason="Matched stack trace or file:line reference.",
                )
            )
    return candidates


def _keyword_candidates(ticket: dict, root: Path) -> list[CodeCandidate]:
    terms = _query_terms(ticket)
    if not terms:
        return []
    candidates: list[CodeCandidate] = []
    for path in _iter_code_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lower = text.lower()
        matches = sum(lower.count(term) for term in terms)
        if matches <= 0:
            continue
        rel = path.relative_to(root).as_posix()
        first_line = _first_matching_line(text, terms)
        function = _nearest_function(path, first_line)
        score = min(0.95, 0.20 + matches / max(len(terms) * 4, 1))
        candidates.append(
            CodeCandidate(
                file=rel,
                score=float(score),
                line_start=max(1, first_line - 5),
                line_end=first_line + 5,
                function=function,
                reason="Matched ticket keywords in source code.",
            )
        )
    return candidates


def _query_terms(ticket: dict) -> list[str]:
    text = " ".join(str(ticket.get(key) or "") for key in ("component", "error_message", "title"))
    tokens = [token for token in tokenize(text) if len(token) >= 3]
    noisy = {"error", "exception", "typeerror", "runtimeerror", "bug", "fails", "failure", "when", "none", "null"}
    unique: list[str] = []
    for token in tokens:
        if token not in noisy and token not in unique:
            unique.append(token)
    return unique[:12]


def _resolve_stack_file(root: Path, raw_file: str) -> Path | None:
    path = Path(raw_file)
    if path.is_absolute() and path.exists():
        try:
            path.resolve().relative_to(root)
            return path.resolve()
        except ValueError:
            return None
    direct = root / raw_file
    if direct.exists():
        resolved = direct.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
        return resolved
    name = Path(raw_file).name
    matches = [candidate for candidate in _iter_code_files(root) if candidate.name == name]
    return matches[0].resolve() if matches else None


def _first_matching_line(text: str, terms: list[str]) -> int:
    for index, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(term in lower for term in terms):
            return index
    return 1


def _nearest_function(path: Path, line_number: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    for index in range(min(line_number, len(lines)), 0, -1):
        line = lines[index - 1].strip()
        match = re.match(r"(?:async\s+def|def|function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if match:
            return match.group(1)
    return ""
