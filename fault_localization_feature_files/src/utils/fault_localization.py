from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CODE_SUFFIXES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
}

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "data",
    "models",
    "reports",
}

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")

STOPWORDS = {
    "about",
    "actual",
    "after",
    "again",
    "also",
    "and",
    "are",
    "behavior",
    "bug",
    "but",
    "can",
    "cannot",
    "crash",
    "does",
    "error",
    "expected",
    "fail",
    "failed",
    "fails",
    "failure",
    "for",
    "from",
    "has",
    "have",
    "into",
    "issue",
    "log",
    "none",
    "not",
    "null",
    "only",
    "should",
    "step",
    "steps",
    "the",
    "this",
    "ticket",
    "typeerror",
    "value",
    "when",
    "with",
    "without",
}

SCORING_WEIGHTS = {
    "embedding_score": 0.72,
    "stack_trace_score": 1.00,
    "component_score": 0.08,
    "keyword_score": 0.12,
    "symbol_score": 0.08,
}


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    file_path: str
    language: str
    symbol_kind: str
    function_name: str
    class_name: str
    start_line: int
    end_line: int
    code_text: str

    @property
    def symbol_name(self) -> str:
        if self.function_name:
            return self.function_name
        if self.class_name:
            return self.class_name
        return ""

    @property
    def symbol_qualified_name(self) -> str:
        return self.symbol_name

    @property
    def evaluation_key(self) -> str:
        symbol = self.symbol_qualified_name or "<file>"
        return f"{self.file_path}::{symbol}:{self.start_line}-{self.end_line}"

    @property
    def search_text(self) -> str:
        return "\n".join(
            part
            for part in (
                self.file_path,
                self.language,
                self.symbol_kind,
                self.function_name,
                self.class_name,
                self.code_text,
            )
            if part
        )

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "symbol_kind": self.symbol_kind,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "symbol_name": self.symbol_name,
            "symbol_qualified_name": self.symbol_qualified_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evaluation_key": self.evaluation_key,
        }
        if include_code:
            row["code_text"] = self.code_text
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CodeChunk":
        return cls(
            chunk_id=str(row.get("chunk_id") or ""),
            file_path=str(row.get("file_path") or row.get("file") or ""),
            language=str(row.get("language") or ""),
            symbol_kind=str(row.get("symbol_kind") or row.get("kind") or "chunk"),
            function_name=str(row.get("function_name") or row.get("function") or ""),
            class_name=str(row.get("class_name") or ""),
            start_line=int(row.get("start_line") or row.get("line_start") or 1),
            end_line=int(row.get("end_line") or row.get("line_end") or 1),
            code_text=str(row.get("code_text") or row.get("snippet") or ""),
        )


@dataclass
class CodeIndex:
    repository_path: str
    chunks: list[CodeChunk]
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        return {
            "version": self.version,
            "repository_path": self.repository_path,
            "created_at_utc": self.created_at_utc,
            "settings": self.settings,
            "chunks": [chunk.to_dict(include_code=include_code) for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeIndex":
        chunks = [CodeChunk.from_dict(row) for row in payload.get("chunks", []) if isinstance(row, dict)]
        return cls(
            repository_path=str(payload.get("repository_path") or ""),
            chunks=chunks,
            created_at_utc=str(payload.get("created_at_utc") or ""),
            version=int(payload.get("version") or 1),
            settings=dict(payload.get("settings") or {}),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CodeIndex":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class LocalizationCandidate:
    chunk: CodeChunk
    score: float
    embedding_score: float
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, rank: int) -> dict[str, Any]:
        scoring_signals = _candidate_scoring_signals(self)
        return {
            "rank": rank,
            "file_path": self.chunk.file_path,
            "function_name": self.chunk.function_name,
            "class_name": self.chunk.class_name,
            "symbol_name": self.chunk.symbol_name,
            "symbol_qualified_name": self.chunk.symbol_qualified_name,
            "symbol_kind": self.chunk.symbol_kind,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "score": round(float(self.score), 4),
            "final_score": round(float(self.score), 4),
            "embedding_score": round(float(self.embedding_score), 4),
            "reason": self.reason,
            "code_text": self.chunk.code_text,
            "chunk_id": self.chunk.chunk_id,
            "evaluation_key": self.chunk.evaluation_key,
            "scoring_signals": scoring_signals,
            "signals": self.signals,
        }


class FaultLocalizer:
    """Retrieve likely faulty code chunks for a structured bug report."""

    def __init__(
        self,
        *,
        code_index: CodeIndex | None = None,
        repo_path: str | Path | None = None,
        top_k: int = 5,
        embedding_backend: str = "tfidf",
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_local_files_only: bool = True,
        llm_client: Any | None = None,
        llm_rerank: bool = False,
    ) -> None:
        _validate_top_k(top_k)
        self.code_index = code_index
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self.sbert_model = sbert_model
        self.sbert_local_files_only = sbert_local_files_only
        self.llm_client = llm_client
        self.llm_rerank = llm_rerank

    def localize(self, ticket_json: dict[str, Any]) -> dict[str, Any]:
        index = self.code_index
        if index is None:
            if self.repo_path is None:
                raise ValueError("FaultLocalizer requires either code_index or repo_path.")
            index = build_code_index(self.repo_path)

        bug_report = build_bug_report_text(ticket_json)
        candidates, backend_name = rank_code_chunks(
            ticket_json,
            index.chunks,
            top_k=self.top_k,
            embedding_backend=self.embedding_backend,
            sbert_model=self.sbert_model,
            sbert_local_files_only=self.sbert_local_files_only,
        )

        warnings: list[str] = []
        llm_rerank_used = False
        if self.llm_rerank and self.llm_client is not None and candidates:
            try:
                candidates = rerank_candidates_with_llm(ticket_json, candidates, self.llm_client, top_k=self.top_k)
                llm_rerank_used = True
            except Exception as exc:  # pragma: no cover - depends on external LLM service
                warnings.append(f"LLM reranking failed: {exc}")

        localized = [candidate.to_dict(rank) for rank, candidate in enumerate(candidates, start=1)]
        best = localized[0] if localized else None
        bug_location = _legacy_bug_location(best)
        context_preview = _line_numbered(best["code_text"], best["start_line"]) if best else ""
        return {
            "ticket_id": str(ticket_json.get("ticket_id") or ticket_json.get("id") or ticket_json.get("bug_id") or ""),
            "bug_report": bug_report,
            "method": {
                "name": "code_chunk_embedding_retrieval",
                "stages": ["retrieval", "optional_llm_rerank"],
                "embedding_backend": backend_name,
                "llm_rerank": llm_rerank_used,
                "top_k": self.top_k,
                "scoring_weights": SCORING_WEIGHTS,
            },
            "localized_candidates": localized,
            "bug_location": bug_location,
            "candidates": [_legacy_candidate(row) for row in localized],
            "context_preview": context_preview,
            "repository_path": index.repository_path,
            "evaluation_ready_fields": {
                "file_level": "localized_candidates[*].file_path",
                "symbol_level": "localized_candidates[*].symbol_qualified_name",
                "line_range": ["localized_candidates[*].start_line", "localized_candidates[*].end_line"],
            },
            "warnings": warnings,
        }


def build_code_index(
    repo_path: str | Path,
    *,
    chunk_lines: int = 80,
    overlap_lines: int = 20,
    include_tests: bool = False,
    max_file_bytes: int = 500_000,
) -> CodeIndex:
    root = Path(repo_path).resolve()
    if not root.exists():
        raise ValueError(f"Repository path does not exist: {root}")
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive.")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative.")
    if overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be smaller than chunk_lines.")

    chunks: list[CodeChunk] = []
    for path in _iter_source_files(root, include_tests=include_tests, max_file_bytes=max_file_bytes):
        chunks.extend(
            _chunks_for_file(
                path,
                root,
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
            )
        )

    return CodeIndex(
        repository_path=str(root),
        chunks=chunks,
        settings={
            "chunk_lines": chunk_lines,
            "overlap_lines": overlap_lines,
            "include_tests": include_tests,
            "max_file_bytes": max_file_bytes,
        },
    )


def load_code_index(path: str | Path) -> CodeIndex:
    return CodeIndex.load(path)


def localize_ticket(
    ticket_json: dict[str, Any],
    *,
    repo_path: str | Path | None = None,
    code_index: CodeIndex | None = None,
    top_k: int = 5,
    embedding_backend: str = "tfidf",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sbert_local_files_only: bool = True,
    llm_client: Any | None = None,
    llm_rerank: bool = False,
) -> dict[str, Any]:
    return FaultLocalizer(
        code_index=code_index,
        repo_path=repo_path,
        top_k=top_k,
        embedding_backend=embedding_backend,
        sbert_model=sbert_model,
        sbert_local_files_only=sbert_local_files_only,
        llm_client=llm_client,
        llm_rerank=llm_rerank,
    ).localize(ticket_json)


def rank_code_chunks(
    ticket_json: dict[str, Any],
    chunks: Iterable[CodeChunk],
    *,
    top_k: int = 5,
    embedding_backend: str = "tfidf",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sbert_local_files_only: bool = True,
) -> tuple[list[LocalizationCandidate], str]:
    _validate_top_k(top_k)
    chunk_list = list(chunks)
    if not chunk_list:
        return [], embedding_backend

    bug_report = build_bug_report_text(ticket_json)
    embedding_scores, backend_name = _embedding_scores(
        bug_report,
        [chunk.search_text for chunk in chunk_list],
        backend=embedding_backend,
        sbert_model=sbert_model,
        sbert_local_files_only=sbert_local_files_only,
    )
    stack_refs = _stack_trace_refs(ticket_json)
    query_terms = _important_terms(bug_report)
    component_terms = _important_terms(str(ticket_json.get("component") or ""))

    candidates: list[LocalizationCandidate] = []
    for chunk, embedding_score in zip(chunk_list, embedding_scores):
        stack_trace_score = _stack_signal(chunk, stack_refs)
        component_score = _term_overlap(component_terms, _tokenize(chunk.file_path + " " + chunk.symbol_name))
        keyword_score, matching_terms = _keyword_signal(query_terms, chunk)
        symbol_score = _symbol_signal(query_terms, chunk)
        score = min(
            1.0,
            SCORING_WEIGHTS["embedding_score"] * embedding_score
            + SCORING_WEIGHTS["stack_trace_score"] * stack_trace_score
            + SCORING_WEIGHTS["component_score"] * component_score
            + SCORING_WEIGHTS["keyword_score"] * keyword_score
            + SCORING_WEIGHTS["symbol_score"] * symbol_score,
        )
        signals = {
            "embedding_score": round(embedding_score, 4),
            "stack_trace_score": round(stack_trace_score, 4),
            "component_score": round(component_score, 4),
            "keyword_score": round(keyword_score, 4),
            "symbol_score": round(symbol_score, 4),
            "final_score": round(score, 4),
            "component_path": round(component_score, 4),
            "keyword_overlap": round(keyword_score, 4),
            "matching_terms": matching_terms[:8],
        }
        candidates.append(
            LocalizationCandidate(
                chunk=chunk,
                score=score,
                embedding_score=embedding_score,
                reason=_reason(chunk, embedding_score, signals),
                signals=signals,
            )
        )

    ranked = sorted(candidates, key=_ranking_key)
    return ranked[:top_k], backend_name


def build_bug_report_text(ticket_json: dict[str, Any]) -> str:
    parts: list[str] = []
    keys = (
        "ticket_id",
        "bug_report",
        "title",
        "summary",
        "body",
        "description",
        "bug_type",
        "product",
        "component",
        "severity",
        "priority",
        "error_message",
        "logs",
        "steps_to_reproduce",
        "expected_behavior",
        "actual_behavior",
        "screenshots_text",
        "environment",
        "os",
        "version",
    )
    for key in keys:
        value = ticket_json.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            text = " ".join(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            text = " ".join(f"{sub_key}: {sub_value}" for sub_key, sub_value in value.items())
        else:
            text = str(value)
        if text.strip():
            parts.append(f"{key}: {text.strip()}")
    return "\n".join(parts)


def rerank_candidates_with_llm(
    ticket_json: dict[str, Any],
    candidates: list[LocalizationCandidate],
    llm_client: Any,
    *,
    top_k: int = 5,
) -> list[LocalizationCandidate]:
    prompt = _llm_rerank_prompt(ticket_json, candidates)
    if hasattr(llm_client, "generate_json"):
        payload = llm_client.generate_json(prompt)
    else:
        from utils.json_schema import repair_json_object

        payload = repair_json_object(str(llm_client.generate(prompt)))

    rows = payload.get("candidates", []) if isinstance(payload, dict) else []
    by_rank = {index: candidate for index, candidate in enumerate(candidates, start=1)}
    reranked: list[LocalizationCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rank = int(row.get("rank") or row.get("original_rank") or 0)
        candidate = by_rank.get(rank)
        if candidate is None:
            continue
        score = _clamp(float(row.get("score", candidate.score)), 0.0, 1.0)
        reason = str(row.get("reason") or candidate.reason)
        signals = dict(candidate.signals)
        signals["llm_rerank_score"] = round(score, 4)
        signals["final_score"] = round(score, 4)
        reranked.append(replace(candidate, score=score, reason=reason, signals=signals))

    if not reranked:
        return candidates[:top_k]
    seen = {candidate.chunk.chunk_id for candidate in reranked}
    for candidate in candidates:
        if candidate.chunk.chunk_id not in seen:
            reranked.append(candidate)
    return sorted(reranked, key=_ranking_key)[:top_k]


def _iter_source_files(root: Path, *, include_tests: bool, max_file_bytes: int) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIRS for part in rel_parts):
            continue
        if not include_tests and any(part.lower() in TEST_DIR_NAMES for part in rel_parts):
            continue
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        yield path


def _chunks_for_file(path: Path, root: Path, *, chunk_lines: int, overlap_lines: int) -> list[CodeChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []
    rel = path.relative_to(root).as_posix()
    language = CODE_SUFFIXES.get(path.suffix.lower(), "text")
    symbol_ranges = _python_symbol_ranges(text) if language == "python" else _regex_symbol_ranges(lines, language)
    chunks: list[CodeChunk] = []
    for symbol in symbol_ranges:
        chunks.extend(_split_symbol(rel, language, lines, symbol, chunk_lines=chunk_lines, overlap_lines=overlap_lines))
    if not chunks:
        chunks.extend(
            _fixed_line_chunks(
                rel,
                language,
                lines,
                start_line=1,
                end_line=len(lines),
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
            )
        )
    return chunks


def _python_symbol_ranges(text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    ranges: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            end = int(getattr(node, "end_lineno", node.lineno))
            ranges.append(
                {
                    "symbol_kind": "class",
                    "function_name": "",
                    "class_name": node.name,
                    "start_line": node.lineno,
                    "end_line": end,
                }
            )
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self._visit_function(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
            self._visit_function(node, "async_function")

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
            class_name = self.class_stack[-1] if self.class_stack else ""
            name = f"{class_name}.{node.name}" if class_name else node.name
            if class_name:
                kind = "method" if kind == "function" else "async_method"
            end = int(getattr(node, "end_lineno", node.lineno))
            ranges.append(
                {
                    "symbol_kind": kind,
                    "function_name": name,
                    "class_name": class_name,
                    "start_line": node.lineno,
                    "end_line": end,
                }
            )
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(ranges, key=lambda row: (row["start_line"], row["end_line"], row["symbol_kind"] != "class"))


def _regex_symbol_ranges(lines: list[str], language: str) -> list[dict[str, Any]]:
    patterns = [
        re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\b(?:function|func|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"\b(?:def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?:async\s*)?\([^)]*\)\s*=>"),
        re.compile(r"\b(?:public|private|protected|static|final|async|\s)+[A-Za-z0-9_<>,\[\]?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ]
    ranges: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        for pattern in patterns:
            match = pattern.search(stripped)
            if not match:
                continue
            kind = "class" if stripped.startswith("class ") or " class " in stripped else "function"
            name = match.group(1)
            ranges.append(
                {
                    "symbol_kind": kind,
                    "function_name": "" if kind == "class" else name,
                    "class_name": name if kind == "class" else "",
                    "start_line": index,
                    "end_line": min(len(lines), index + 79),
                }
            )
            break
    return ranges


def _split_symbol(
    rel: str,
    language: str,
    lines: list[str],
    symbol: dict[str, Any],
    *,
    chunk_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    start_line = max(1, int(symbol["start_line"]))
    end_line = min(len(lines), int(symbol["end_line"]))
    if end_line - start_line + 1 <= chunk_lines:
        return [
            _make_chunk(
                rel,
                language,
                lines,
                symbol_kind=str(symbol["symbol_kind"]),
                function_name=str(symbol["function_name"]),
                class_name=str(symbol["class_name"]),
                start_line=start_line,
                end_line=end_line,
            )
        ]
    return _fixed_line_chunks(
        rel,
        language,
        lines,
        start_line=start_line,
        end_line=end_line,
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
        symbol_kind=str(symbol["symbol_kind"]),
        function_name=str(symbol["function_name"]),
        class_name=str(symbol["class_name"]),
    )


def _fixed_line_chunks(
    rel: str,
    language: str,
    lines: list[str],
    *,
    start_line: int,
    end_line: int,
    chunk_lines: int,
    overlap_lines: int,
    symbol_kind: str = "chunk",
    function_name: str = "",
    class_name: str = "",
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    step = max(1, chunk_lines - overlap_lines)
    current = start_line
    while current <= end_line:
        chunk_end = min(end_line, current + chunk_lines - 1)
        chunks.append(
            _make_chunk(
                rel,
                language,
                lines,
                symbol_kind=symbol_kind,
                function_name=function_name,
                class_name=class_name,
                start_line=current,
                end_line=chunk_end,
            )
        )
        if chunk_end >= end_line:
            break
        current += step
    return chunks


def _make_chunk(
    rel: str,
    language: str,
    lines: list[str],
    *,
    symbol_kind: str,
    function_name: str,
    class_name: str,
    start_line: int,
    end_line: int,
) -> CodeChunk:
    code_text = "\n".join(lines[start_line - 1 : end_line])
    symbol = function_name or class_name or symbol_kind
    chunk_id = f"{rel}:{start_line}-{end_line}:{symbol}"
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=rel,
        language=language,
        symbol_kind=symbol_kind,
        function_name=function_name,
        class_name=class_name,
        start_line=start_line,
        end_line=end_line,
        code_text=code_text,
    )


def _embedding_scores(
    query: str,
    documents: list[str],
    *,
    backend: str,
    sbert_model: str,
    sbert_local_files_only: bool,
) -> tuple[list[float], str]:
    normalized = backend.strip().lower()
    if normalized in {"auto", "sbert", "sentence-transformers", "sentence_transformers"}:
        try:
            return _sbert_scores(query, documents, sbert_model, local_files_only=sbert_local_files_only), f"sbert:{sbert_model}"
        except Exception:
            if normalized != "auto":
                raise
    return _tfidf_scores(query, documents), "tfidf"


def _sbert_scores(query: str, documents: list[str], model_name: str, *, local_files_only: bool) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    embeddings = model.encode([query, *documents], normalize_embeddings=True)
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]
    scores: list[float] = []
    for embedding in doc_embeddings:
        score = float(sum(float(left) * float(right) for left, right in zip(query_embedding, embedding)))
        scores.append(_clamp(score, 0.0, 1.0))
    return scores


def _tfidf_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [_tokenize(document) for document in documents]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0 for _ in documents]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))
    total_docs = max(1, len(tokenized_docs))
    idf = {term: math.log((total_docs + 1) / (df + 1)) + 1.0 for term, df in doc_freq.items()}
    query_vector = _tfidf_vector(query_tokens, idf)
    return [_cosine(query_vector, _tfidf_vector(tokens, idf)) for tokens in tokenized_docs]


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    vector: dict[str, float] = {}
    for token, count in counts.items():
        if token not in idf:
            continue
        vector[token] = (1.0 + math.log(count)) * idf[token]
    return vector


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        lowered = raw.lower()
        tokens.append(lowered)
        for piece in re.split(r"[_\W]+", raw):
            tokens.extend(part.lower() for part in CAMEL_RE.findall(piece) if len(part) > 1)
    return [token for token in tokens if len(token) > 1]


def _important_terms(text: str) -> list[str]:
    unique: list[str] = []
    for token in _tokenize(text):
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in unique:
            unique.append(token)
    return unique[:40]


def _stack_trace_refs(ticket_json: dict[str, Any]) -> list[tuple[str, int | None]]:
    text = "\n".join(
        str(ticket_json.get(key) or "")
        for key in ("error_message", "logs", "description", "body", "actual_behavior")
    )
    refs: list[tuple[str, int | None]] = []
    patterns = [
        r"File \"(?P<file>[^\"]+)\", line (?P<line>\d+)",
        r"(?P<file>[A-Za-z0-9_./\\-]+\.(?:py|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|go|rs)):(?P<line>\d+)",
        r"(?P<file>[A-Za-z0-9_./\\-]+\.(?:py|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|go|rs))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            line_text = match.groupdict().get("line")
            refs.append((match.group("file").replace("\\", "/"), int(line_text) if line_text else None))
    return refs


def _stack_signal(chunk: CodeChunk, refs: list[tuple[str, int | None]]) -> float:
    if not refs:
        return 0.0
    chunk_path = _normalize_path(chunk.file_path)
    chunk_name = Path(chunk_path).name
    best = 0.0
    for raw_file, line in refs:
        ref = _normalize_path(raw_file)
        ref_name = Path(ref).name
        file_matches = ref == chunk_path or ref.endswith("/" + chunk_path) or chunk_path.endswith("/" + ref) or ref_name == chunk_name
        if not file_matches:
            continue
        if line is not None and chunk.start_line <= line <= chunk.end_line:
            best = max(best, 0.75)
        elif line is not None:
            distance = min(abs(chunk.start_line - line), abs(chunk.end_line - line))
            best = max(best, 0.55 if distance <= 25 else 0.35)
        else:
            best = max(best, 0.35)
    return best


def _keyword_signal(query_terms: list[str], chunk: CodeChunk) -> tuple[float, list[str]]:
    if not query_terms:
        return 0.0, []
    chunk_terms = set(_tokenize(chunk.search_text))
    matching = [term for term in query_terms if term in chunk_terms]
    return min(1.0, len(matching) / max(4, min(len(query_terms), 12))), matching


def _symbol_signal(query_terms: list[str], chunk: CodeChunk) -> float:
    symbol_terms = set(_tokenize(" ".join([chunk.file_path, chunk.function_name, chunk.class_name])))
    if not symbol_terms:
        return 0.0
    return _term_overlap(query_terms, symbol_terms)


def _term_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def _candidate_scoring_signals(candidate: LocalizationCandidate) -> dict[str, Any]:
    signals = dict(candidate.signals)
    row = {
        "embedding_score": round(float(signals.get("embedding_score", candidate.embedding_score)), 4),
        "stack_trace_score": round(float(signals.get("stack_trace_score", 0.0)), 4),
        "component_score": round(float(signals.get("component_score", 0.0)), 4),
        "keyword_score": round(float(signals.get("keyword_score", 0.0)), 4),
        "symbol_score": round(float(signals.get("symbol_score", 0.0)), 4),
        "final_score": round(float(signals.get("final_score", candidate.score)), 4),
        "weights": SCORING_WEIGHTS,
    }
    if "llm_rerank_score" in signals:
        row["llm_rerank_score"] = round(float(signals["llm_rerank_score"]), 4)
    return row


def _reason(chunk: CodeChunk, embedding_score: float, signals: dict[str, Any]) -> str:
    pieces: list[str] = []
    if float(signals.get("stack_trace_score", 0.0)) >= 0.55:
        pieces.append("The ticket contains a stack trace or file:line reference that points to this chunk.")
    elif float(signals.get("stack_trace_score", 0.0)) > 0:
        pieces.append("The ticket references this file.")
    if float(signals.get("component_score", 0.0)) > 0:
        pieces.append("The component terms match the file path or symbol name.")
    matching_terms = signals.get("matching_terms") or []
    if matching_terms:
        pieces.append(f"Shared report/code terms include: {', '.join(matching_terms[:5])}.")
    if embedding_score > 0:
        pieces.append("The bug report vector is similar to the code chunk text.")
    if not pieces:
        pieces.append("This chunk is one of the closest available code vectors for the report.")
    location = chunk.function_name or chunk.class_name or chunk.file_path
    return f"{location}: " + " ".join(pieces)


def _ranking_key(candidate: LocalizationCandidate) -> tuple[float, int, int, str]:
    chunk = candidate.chunk
    return (
        -candidate.score,
        _symbol_rank(chunk.symbol_kind),
        chunk.end_line - chunk.start_line,
        chunk.chunk_id,
    )


def _symbol_rank(symbol_kind: str) -> int:
    normalized = symbol_kind.lower()
    if "method" in normalized or "function" in normalized:
        return 0
    if normalized == "class":
        return 1
    return 2


def _llm_rerank_prompt(ticket_json: dict[str, Any], candidates: list[LocalizationCandidate]) -> str:
    candidate_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate_rows.append(
            {
                "rank": rank,
                "file_path": candidate.chunk.file_path,
                "function_name": candidate.chunk.function_name,
                "class_name": candidate.chunk.class_name,
                "start_line": candidate.chunk.start_line,
                "end_line": candidate.chunk.end_line,
                "retrieval_score": round(candidate.score, 4),
                "code_text": candidate.chunk.code_text[:1600],
            }
        )
    return (
        "You are reranking fault localization candidates. "
        "Given the bug report and candidate code chunks, return JSON only in this shape: "
        '{"candidates":[{"rank":1,"score":0.0,"reason":"short reason"}]}. '
        "Use score 0.0-1.0 and keep the original rank value.\n\n"
        f"Bug report:\n{json.dumps(ticket_json, ensure_ascii=False, indent=2)}\n\n"
        f"Candidates:\n{json.dumps(candidate_rows, ensure_ascii=False, indent=2)}"
    )


def _legacy_bug_location(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {}
    return {
        "file": best.get("file_path", ""),
        "function": best.get("function_name") or best.get("class_name") or best.get("symbol_name") or "",
        "line_start": best.get("start_line", 1),
        "line_end": best.get("end_line", 1),
        "reason": best.get("reason", ""),
    }


def _legacy_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": row.get("file_path", ""),
        "score": row.get("score", 0.0),
        "function": row.get("function_name") or row.get("class_name") or row.get("symbol_name") or "",
        "line_start": row.get("start_line", 1),
        "line_end": row.get("end_line", 1),
        "reason": row.get("reason", ""),
    }


def _line_numbered(code_text: str, start_line: int) -> str:
    return "\n".join(f"{line_no}: {line}" for line_no, line in enumerate(code_text.splitlines(), start=start_line))


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
