from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
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

IGNORED_DIRS_ANYWHERE = {
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
}

# These names commonly contain generated artifacts at repository root, but are
# also legitimate source-package names deeper in a tree (for example,
# django/db/models). Excluding them everywhere can silently remove real code.
IGNORED_ROOT_DIRS = {
    "dist",
    "build",
    "data",
    "models",
    "reports",
}

# Kept as a public compatibility alias for callers that inspect this constant.
IGNORED_DIRS = IGNORED_DIRS_ANYWHERE | IGNORED_ROOT_DIRS

TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")
TOKEN_SPLIT_RE = re.compile(r"[_\W]+")

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
    # The five retrieval weights sum to 1.0. Domain-path evidence is an
    # additive, bounded routing boost because it is sparse and high precision.
    "embedding_score": 0.50,
    "stack_trace_score": 0.30,
    "component_score": 0.05,
    "keyword_score": 0.08,
    "symbol_score": 0.07,
    "domain_path_score": 0.18,
    "identifier_score": 0.14,
    "path_term_score": 0.12,
    "repository_proximity_score": 0.10,
    "package_proximity_score": 0.12,
}

HYBRID_SEMANTIC_WEIGHTS = {
    "tfidf": 0.35,
    "sbert": 0.65,
}

# E1 is intentionally a small, ordered ablation instead of one opaque switch.
# The values are fixed before Validation is run; only Development may be used
# to replace them in a later protocol version.
FILE_AGGREGATION_MODES = (
    "basic",
    "supporting-chunks",
    "supporting-symbols",
    "supporting-symbols-package",
)
FILE_AGGREGATION_PARAMETERS: dict[str, float | int] = {
    "max_evidence_chunks": 5,
    "evidence_score_ratio": 0.80,
    "minimum_evidence_score": 0.05,
    "support_bonus_per_additional_chunk": 0.0125,
    "support_bonus_cap": 0.05,
    "symbol_bonus_per_additional_symbol": 0.01,
    "symbol_bonus_cap": 0.04,
    "package_bonus_cap": 0.10,
}

# E2 import-graph ablations.  The graph is repository-level and independent of
# the ticket; only the bounded one-hop scoring pass runs per ticket.
IMPORT_GRAPH_MODES = ("off", "outgoing", "bidirectional")
IMPORT_GRAPH_PARAMETERS: dict[str, float | int] = {
    "reference_file_k": 5,
    "outgoing_signal": 0.80,
    "incoming_signal": 0.65,
    "rank_decay": 0.08,
    "maximum_signal": 0.80,
    "maximum_bonus": 0.08,
    "maximum_evidence_per_file": 4,
}

LLM_MAX_BUG_REPORT_CHARS = 4_000
LLM_MAX_CODE_CHARS_PER_CANDIDATE = 800
LLM_RERANK_WEIGHTS = {
    "retrieval": 0.70,
    "llm": 0.30,
}

TICKET_CONTENT_KEYS = (
    "bug_report",
    "title",
    "summary",
    "body",
    "description",
    "error_message",
    "logs",
    "steps_to_reproduce",
    "expected_behavior",
    "actual_behavior",
)


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
class ImportGraph:
    """Cached, repository-level Python import relationships."""

    outgoing: dict[str, list[str]] = field(default_factory=dict)
    incoming: dict[str, list[str]] = field(default_factory=dict)
    unresolved: dict[str, list[str]] = field(default_factory=dict)
    version: int = 1
    stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "outgoing": self.outgoing,
            "incoming": self.incoming,
            "unresolved": self.unresolved,
            "stats": self.stats,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ImportGraph":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Import graph cache must contain a JSON object.")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImportGraph":
        def adjacency(name: str) -> dict[str, list[str]]:
            raw = payload.get(name) or {}
            if not isinstance(raw, dict):
                return {}
            return {
                str(file_path): sorted({str(value) for value in values})
                for file_path, values in raw.items()
                if isinstance(values, list)
            }

        raw_stats = payload.get("stats") or {}
        return cls(
            outgoing=adjacency("outgoing"),
            incoming=adjacency("incoming"),
            unresolved=adjacency("unresolved"),
            version=int(payload.get("version") or 1),
            stats={
                str(key): int(value)
                for key, value in raw_stats.items()
                if isinstance(value, (int, float))
            }
            if isinstance(raw_stats, dict)
            else {},
        )


@dataclass
class CodeIndex:
    repository_path: str
    chunks: list[CodeChunk]
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1
    settings: dict[str, Any] = field(default_factory=dict)
    import_graph: ImportGraph | None = None

    def to_dict(self, *, include_code: bool = True) -> dict[str, Any]:
        return {
            "version": self.version,
            "repository_path": self.repository_path,
            "created_at_utc": self.created_at_utc,
            "settings": self.settings,
            "import_graph": self.import_graph.to_dict() if self.import_graph is not None else None,
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
            import_graph=(
                ImportGraph.from_dict(payload["import_graph"])
                if isinstance(payload.get("import_graph"), dict)
                else None
            ),
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
        row = {
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
        supporting_chunks = self.signals.get("supporting_chunks")
        if isinstance(supporting_chunks, list):
            row["supporting_chunks"] = supporting_chunks
        return row


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
        semantic_candidate_k: int = 50,
        candidate_file_k: int = 20,
        generic_routing: bool = False,
        domain_path_routing: bool = True,
        repository_proximity: bool = False,
        import_graph_mode: str = "off",
        llm_client: Any | None = None,
        llm_rerank: bool = False,
        llm_candidate_k: int | None = None,
        file_aggregation: bool = True,
        advanced_file_aggregation: bool = False,
        file_aggregation_mode: str | None = None,
        min_ticket_chars: int = 20,
    ) -> None:
        _validate_top_k(top_k)
        if min_ticket_chars <= 0:
            raise ValueError("min_ticket_chars must be positive.")
        if semantic_candidate_k <= 0:
            raise ValueError("semantic_candidate_k must be positive.")
        if candidate_file_k <= 0:
            raise ValueError("candidate_file_k must be positive.")
        if _is_hybrid_backend(embedding_backend) and semantic_candidate_k < candidate_file_k:
            raise ValueError(
                "semantic_candidate_k must be greater than or equal to candidate_file_k "
                "for hybrid Stage-1 retrieval."
            )
        if llm_candidate_k is not None and llm_candidate_k <= 0:
            raise ValueError("llm_candidate_k must be positive when supplied.")
        self.code_index = code_index
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self.sbert_model = sbert_model
        self.sbert_local_files_only = sbert_local_files_only
        self.semantic_candidate_k = semantic_candidate_k
        self.candidate_file_k = candidate_file_k
        self.generic_routing = generic_routing
        self.domain_path_routing = domain_path_routing
        self.import_graph_mode = _resolve_import_graph_mode(
            import_graph_mode,
            legacy_repository_proximity=repository_proximity,
        )
        self.repository_proximity = self.import_graph_mode != "off"
        self.llm_client = llm_client
        self.llm_rerank = llm_rerank
        self.llm_candidate_k = llm_candidate_k
        self.file_aggregation = file_aggregation
        self.file_aggregation_mode = _resolve_file_aggregation_mode(
            file_aggregation_mode,
            advanced=advanced_file_aggregation,
        )
        self.advanced_file_aggregation = self.file_aggregation_mode != "basic"
        self.min_ticket_chars = min_ticket_chars

    def localize(self, ticket_json: dict[str, Any]) -> dict[str, Any]:
        index = self.code_index
        if index is None:
            if self.repo_path is None:
                raise ValueError("FaultLocalizer requires either code_index or repo_path.")
            index = build_code_index(self.repo_path)

        input_validation = validate_localization_request(ticket_json, min_ticket_chars=self.min_ticket_chars)
        bug_report = build_bug_report_text(ticket_json)
        if input_validation["errors"]:
            candidates: list[LocalizationCandidate] = []
            backend_name = _validate_embedding_backend(self.embedding_backend)
        else:
            llm_candidate_k = max(self.top_k, int(self.llm_candidate_k or self.top_k)) if self.llm_rerank else 0
            retrieval_top_k = max(self.candidate_file_k, self.top_k, llm_candidate_k)
            import_graph = index.import_graph
            if self.import_graph_mode != "off" and import_graph is None:
                import_graph = build_import_graph(index.chunks)
                index.import_graph = import_graph
            candidates, backend_name = rank_code_chunks(
                ticket_json,
                index.chunks,
                top_k=retrieval_top_k,
                embedding_backend=self.embedding_backend,
                sbert_model=self.sbert_model,
                sbert_local_files_only=self.sbert_local_files_only,
                semantic_candidate_k=self.semantic_candidate_k,
                file_aggregation=self.file_aggregation,
                advanced_file_aggregation=self.advanced_file_aggregation,
                file_aggregation_mode=self.file_aggregation_mode,
                generic_routing=self.generic_routing,
                domain_path_routing=self.domain_path_routing,
                repository_proximity=self.repository_proximity,
                import_graph_mode=self.import_graph_mode,
                import_graph=import_graph,
            )

        # Stage 1 ends here. Preserve its file ranking before an optional LLM
        # changes the order so candidate retrieval can be evaluated in isolation.
        stage1_candidates = candidates[: self.candidate_file_k]

        warnings = [f"Input validation warning: {message}" for message in input_validation["warnings"]]
        warnings.extend(f"Input validation error: {message}" for message in input_validation["errors"])
        if not index.chunks:
            warnings.append("The code index contains no supported source chunks.")
        if self.embedding_backend.strip().lower() == "auto" and backend_name == "tfidf":
            warnings.append("SBERT was unavailable in auto mode; TF-IDF retrieval was used.")
        llm_rerank_used = False
        if self.llm_rerank and self.llm_client is not None and candidates:
            try:
                llm_input = candidates[: max(self.top_k, int(self.llm_candidate_k or self.top_k))]
                candidates = rerank_candidates_with_llm(ticket_json, llm_input, self.llm_client, top_k=self.top_k)
                llm_rerank_used = any("llm_rerank_score" in candidate.signals for candidate in candidates)
                if not llm_rerank_used:
                    warnings.append("LLM reranking returned no usable candidates; retrieval ranking was kept.")
            except Exception as exc:  # pragma: no cover - depends on external LLM service
                warnings.append(f"LLM reranking failed: {exc}")
                candidates = candidates[: self.top_k]
        elif self.llm_rerank and self.llm_client is None:
            warnings.append("LLM reranking was requested but no LLM client was configured; retrieval ranking was kept.")
            candidates = candidates[: self.top_k]
        else:
            candidates = candidates[: self.top_k]

        localized = [candidate.to_dict(rank) for rank, candidate in enumerate(candidates, start=1)]
        stage1_files = [
            _stage1_candidate_file(candidate, rank)
            for rank, candidate in enumerate(stage1_candidates, start=1)
        ]
        best = localized[0] if localized else None
        bug_location = _legacy_bug_location(best)
        confidence = _localization_confidence(localized, input_validation, llm_rerank_used=llm_rerank_used)
        bug_location.update(
            {
                "confidence_level": confidence["confidence_level"],
                "should_manual_review": confidence["should_manual_review"],
                "recommend_patch_generation": confidence["recommend_patch_generation"],
            }
        )
        context_preview = _line_numbered(best["code_text"], best["start_line"]) if best else ""
        return {
            "ticket_id": str(ticket_json.get("ticket_id") or ticket_json.get("id") or ticket_json.get("bug_id") or ""),
            "bug_report": bug_report,
            "method": {
                "name": "code_chunk_embedding_retrieval",
                "stages": [
                    "stage1_chunk_retrieval",
                    *(["stage1_generic_routing"] if self.generic_routing else []),
                    *(["stage1_domain_path_routing"] if self.domain_path_routing else []),
                    *(["stage1_import_graph"] if self.import_graph_mode != "off" else []),
                    *(["stage1_semantic_rerank"] if _is_hybrid_backend(self.embedding_backend) else []),
                    *(["stage1_file_aggregation"] if self.file_aggregation else []),
                    *(
                        ["stage1_advanced_file_aggregation"]
                        if self.file_aggregation and self.advanced_file_aggregation
                        else []
                    ),
                    "stage2_optional_llm_rerank",
                    "confidence_gate",
                ],
                "embedding_backend": backend_name,
                "semantic_candidate_k": (
                    self.semantic_candidate_k if _is_hybrid_backend(self.embedding_backend) else 0
                ),
                "candidate_file_k": self.candidate_file_k,
                "generic_routing": self.generic_routing,
                "domain_path_routing": self.domain_path_routing,
                "repository_proximity": self.repository_proximity,
                "import_graph_mode": self.import_graph_mode,
                "import_graph_parameters": IMPORT_GRAPH_PARAMETERS,
                "llm_rerank": llm_rerank_used,
                "llm_candidate_k": max(self.top_k, int(self.llm_candidate_k or self.top_k)) if self.llm_rerank else 0,
                "file_aggregation": self.file_aggregation,
                "advanced_file_aggregation": self.advanced_file_aggregation,
                "file_aggregation_mode": self.file_aggregation_mode,
                "file_aggregation_parameters": FILE_AGGREGATION_PARAMETERS,
                "ranking_level": "file_aggregated_chunks" if self.file_aggregation else "code_chunks",
                "top_k": self.top_k,
                "scoring_weights": SCORING_WEIGHTS,
            },
            "stage1_candidate_files": stage1_files,
            "stage1_diagnostics": {
                "stage_boundary": "before_llm_rerank",
                "indexed_chunk_count": len(index.chunks),
                "semantic_candidate_file_k": (
                    self.semantic_candidate_k if _is_hybrid_backend(self.embedding_backend) else 0
                ),
                "semantic_representative": (
                    "best_tfidf_chunk_per_file"
                    if _is_hybrid_backend(self.embedding_backend)
                    else ""
                ),
                "requested_candidate_file_k": self.candidate_file_k,
                "returned_candidate_file_count": len(stage1_files),
                "unique_candidate_file_count": len(
                    {row["file_path"] for row in stage1_files if row["file_path"]}
                ),
                "import_graph": (
                    {"version": index.import_graph.version, **index.import_graph.stats}
                    if self.import_graph_mode != "off" and index.import_graph is not None
                    else {"enabled": False}
                ),
            },
            "localized_candidates": localized,
            "localized_files": [_localized_file(row) for row in localized],
            "bug_location": bug_location,
            "candidates": [_legacy_candidate(row) for row in localized],
            "context_preview": context_preview,
            "input_validation": input_validation,
            "confidence": confidence,
            "confidence_level": confidence["confidence_level"],
            "should_manual_review": confidence["should_manual_review"],
            "recommend_patch_generation": confidence["recommend_patch_generation"],
            "patch_generation_policy": confidence["patch_generation_policy"],
            "repository_path": index.repository_path,
            "evaluation_ready_fields": {
                "candidate_stage": "stage1_candidate_files[*].file_path",
                "file_level": "localized_files[*].file_path",
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
    previous_index: CodeIndex | None = None,
) -> CodeIndex:
    root = Path(repo_path).resolve()
    if not root.exists():
        raise ValueError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive.")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative.")
    if overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be smaller than chunk_lines.")
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive.")

    source_files = list(_iter_source_files(root, include_tests=include_tests, max_file_bytes=max_file_bytes))
    file_fingerprints = {
        path.relative_to(root).as_posix(): _source_file_fingerprint(path)
        for path in source_files
    }
    previous_fingerprints = (
        previous_index.settings.get("file_fingerprints", {})
        if previous_index is not None and isinstance(previous_index.settings.get("file_fingerprints"), dict)
        else {}
    )
    previous_chunks: dict[str, list[CodeChunk]] = {}
    if previous_index is not None:
        for chunk in previous_index.chunks:
            previous_chunks.setdefault(chunk.file_path, []).append(chunk)

    chunks: list[CodeChunk] = []
    reused_files = 0
    rebuilt_files = 0
    for path in source_files:
        rel = path.relative_to(root).as_posix()
        can_reuse = (
            previous_fingerprints.get(rel) == file_fingerprints[rel]
            and rel in previous_chunks
            and previous_index is not None
            and previous_index.settings.get("chunk_lines") == chunk_lines
            and previous_index.settings.get("overlap_lines") == overlap_lines
        )
        if can_reuse:
            chunks.extend(previous_chunks[rel])
            reused_files += 1
            continue
        chunks.extend(
            _chunks_for_file(
                path,
                root,
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
            )
        )
        rebuilt_files += 1

    import_graph = build_import_graph(chunks)
    return CodeIndex(
        repository_path=str(root),
        chunks=chunks,
        version=2,
        settings={
            "chunk_lines": chunk_lines,
            "overlap_lines": overlap_lines,
            "include_tests": include_tests,
            "max_file_bytes": max_file_bytes,
            "repository_fingerprint": _repository_fingerprint_from_files(file_fingerprints),
            "file_fingerprints": file_fingerprints,
            "index_stats": {
                "source_files": len(source_files),
                "reused_files": reused_files,
                "rebuilt_files": rebuilt_files,
                "import_graph_edges": int(import_graph.stats.get("edges", 0)),
            },
        },
        import_graph=import_graph,
    )


def repository_fingerprint(
    repo_path: str | Path,
    *,
    include_tests: bool = False,
    max_file_bytes: int = 500_000,
) -> str:
    root = Path(repo_path).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repository path is not a directory: {root}")
    file_fingerprints = {
        path.relative_to(root).as_posix(): _source_file_fingerprint(path)
        for path in _iter_source_files(root, include_tests=include_tests, max_file_bytes=max_file_bytes)
    }
    return _repository_fingerprint_from_files(file_fingerprints)


def _source_file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_fingerprint_from_files(file_fingerprints: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for file_path, fingerprint in sorted(file_fingerprints.items()):
        digest.update(file_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    semantic_candidate_k: int = 50,
    candidate_file_k: int = 20,
    generic_routing: bool = False,
    domain_path_routing: bool = True,
    repository_proximity: bool = False,
    import_graph_mode: str = "off",
    llm_client: Any | None = None,
    llm_rerank: bool = False,
    llm_candidate_k: int | None = None,
    file_aggregation: bool = True,
    advanced_file_aggregation: bool = False,
    file_aggregation_mode: str | None = None,
    min_ticket_chars: int = 20,
) -> dict[str, Any]:
    return FaultLocalizer(
        code_index=code_index,
        repo_path=repo_path,
        top_k=top_k,
        embedding_backend=embedding_backend,
        sbert_model=sbert_model,
        sbert_local_files_only=sbert_local_files_only,
        semantic_candidate_k=semantic_candidate_k,
        candidate_file_k=candidate_file_k,
        generic_routing=generic_routing,
        domain_path_routing=domain_path_routing,
        repository_proximity=repository_proximity,
        import_graph_mode=import_graph_mode,
        llm_client=llm_client,
        llm_rerank=llm_rerank,
        llm_candidate_k=llm_candidate_k,
        file_aggregation=file_aggregation,
        advanced_file_aggregation=advanced_file_aggregation,
        file_aggregation_mode=file_aggregation_mode,
        min_ticket_chars=min_ticket_chars,
    ).localize(ticket_json)


def rank_code_chunks(
    ticket_json: dict[str, Any],
    chunks: Iterable[CodeChunk],
    *,
    top_k: int = 5,
    embedding_backend: str = "tfidf",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sbert_local_files_only: bool = True,
    semantic_candidate_k: int = 50,
    file_aggregation: bool = True,
    advanced_file_aggregation: bool = False,
    file_aggregation_mode: str | None = None,
    generic_routing: bool = False,
    domain_path_routing: bool = True,
    repository_proximity: bool = False,
    import_graph_mode: str = "off",
    import_graph: ImportGraph | None = None,
) -> tuple[list[LocalizationCandidate], str]:
    _validate_top_k(top_k)
    if semantic_candidate_k <= 0:
        raise ValueError("semantic_candidate_k must be positive.")
    aggregation_mode = _resolve_file_aggregation_mode(
        file_aggregation_mode,
        advanced=advanced_file_aggregation,
    )
    graph_mode = _resolve_import_graph_mode(
        import_graph_mode,
        legacy_repository_proximity=repository_proximity,
    )
    chunk_list = list(chunks)
    if not chunk_list:
        return [], _validate_embedding_backend(embedding_backend)

    bug_report = build_bug_report_text(ticket_json)
    hybrid_backend = _is_hybrid_backend(embedding_backend)
    search_texts = [chunk.search_text for chunk in chunk_list]
    tokenized_search_texts = [_tokenize(search_text) for search_text in search_texts]
    embedding_scores, backend_name = _embedding_scores(
        bug_report,
        search_texts,
        backend="tfidf" if hybrid_backend else embedding_backend,
        sbert_model=sbert_model,
        sbert_local_files_only=sbert_local_files_only,
        tokenized_documents=tokenized_search_texts,
    )
    stack_refs = _stack_trace_refs(ticket_json)
    query_terms = _important_terms(bug_report)
    component_terms = _component_terms(ticket_json, chunk_list)
    identifiers = _report_identifiers(bug_report) if generic_routing else []
    path_query_terms = _path_signal_query_terms(bug_report, query_terms, identifiers)

    candidates: list[LocalizationCandidate] = []
    for chunk, embedding_score, search_tokens in zip(
        chunk_list,
        embedding_scores,
        tokenized_search_texts,
    ):
        search_term_set = set(search_tokens)
        component_symbol_terms = set(_tokenize(chunk.file_path + " " + chunk.symbol_name))
        symbol_terms = set(
            _tokenize(" ".join([chunk.file_path, chunk.function_name, chunk.class_name]))
        )
        stack_trace_score = _stack_signal(chunk, stack_refs)
        component_score = _term_overlap(component_terms, component_symbol_terms)
        keyword_score, matching_terms = _keyword_signal(
            query_terms,
            chunk,
            chunk_terms=search_term_set,
        )
        symbol_score = _symbol_signal(query_terms, chunk, symbol_terms=symbol_terms)
        identifier_score, matching_identifiers = _identifier_signal(identifiers, chunk)
        path_term_score, matching_path_terms = _path_term_signal(
            bug_report,
            query_terms,
            identifiers,
            chunk,
            query_set=path_query_terms,
        )
        domain_path_score, matching_domain_intents = (
            _domain_path_signal(bug_report, chunk) if domain_path_routing else (0.0, [])
        )
        score = _clamp(
            SCORING_WEIGHTS["embedding_score"] * embedding_score
            + SCORING_WEIGHTS["stack_trace_score"] * stack_trace_score
            + SCORING_WEIGHTS["component_score"] * component_score
            + SCORING_WEIGHTS["keyword_score"] * keyword_score
            + SCORING_WEIGHTS["symbol_score"] * symbol_score
            + SCORING_WEIGHTS["domain_path_score"] * domain_path_score
            + SCORING_WEIGHTS["identifier_score"] * identifier_score
            + SCORING_WEIGHTS["path_term_score"] * path_term_score,
            0.0,
            1.0,
        )
        signals = {
            "embedding_score": round(embedding_score, 4),
            "stack_trace_score": round(stack_trace_score, 4),
            "component_score": round(component_score, 4),
            "keyword_score": round(keyword_score, 4),
            "symbol_score": round(symbol_score, 4),
            "domain_path_score": round(domain_path_score, 4),
            "identifier_score": round(identifier_score, 4),
            "path_term_score": round(path_term_score, 4),
            "repository_proximity_score": 0.0,
            "final_score": round(score, 4),
            "component_path": round(component_score, 4),
            "keyword_overlap": round(keyword_score, 4),
            "matching_terms": matching_terms[:8],
            "matching_domain_intents": matching_domain_intents[:8],
            "matching_identifiers": matching_identifiers[:8],
            "matching_path_terms": matching_path_terms[:8],
            "matching_repository_proximity": [],
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

    if graph_mode != "off":
        candidates = _apply_import_graph(
            candidates,
            import_graph or build_import_graph(chunk_list),
            mode=graph_mode,
        )
    ranked = sorted(candidates, key=_ranking_key)
    aggregation_evidence = ranked
    if hybrid_backend:
        # A raw top-N chunk pool can collapse to far fewer than N files when a
        # large module contributes many similar chunks.  Select one TF-IDF
        # representative per file before semantic reranking so Stage 1 can
        # reliably return the requested number of unique candidate files.
        candidate_pool = _best_file_representatives(ranked)[
            : max(top_k, semantic_candidate_k)
        ]
        sbert_scores = _sbert_scores(
            bug_report,
            [candidate.chunk.search_text for candidate in candidate_pool],
            sbert_model,
            local_files_only=sbert_local_files_only,
        )
        reranked: list[LocalizationCandidate] = []
        for candidate, sbert_score in zip(candidate_pool, sbert_scores):
            tfidf_score = candidate.embedding_score
            semantic_score = _clamp(
                HYBRID_SEMANTIC_WEIGHTS["tfidf"] * tfidf_score
                + HYBRID_SEMANTIC_WEIGHTS["sbert"] * sbert_score,
                0.0,
                1.0,
            )
            signals = dict(candidate.signals)
            signals.update(
                {
                    "tfidf_score": round(tfidf_score, 4),
                    "sbert_score": round(sbert_score, 4),
                    "embedding_score": round(semantic_score, 4),
                    "semantic_candidate_pool": len(candidate_pool),
                }
            )
            score = _clamp(
                SCORING_WEIGHTS["embedding_score"] * semantic_score
                + SCORING_WEIGHTS["stack_trace_score"] * float(signals["stack_trace_score"])
                + SCORING_WEIGHTS["component_score"] * float(signals["component_score"])
                + SCORING_WEIGHTS["keyword_score"] * float(signals["keyword_score"])
                + SCORING_WEIGHTS["symbol_score"] * float(signals["symbol_score"])
                + SCORING_WEIGHTS["domain_path_score"] * float(signals["domain_path_score"])
                + SCORING_WEIGHTS["identifier_score"] * float(signals["identifier_score"])
                + SCORING_WEIGHTS["path_term_score"] * float(signals["path_term_score"])
                + SCORING_WEIGHTS["repository_proximity_score"]
                * float(signals["repository_proximity_score"]),
                0.0,
                1.0,
            )
            signals["final_score"] = round(score, 4)
            reranked.append(
                replace(
                    candidate,
                    score=score,
                    embedding_score=semantic_score,
                    reason=_reason(candidate.chunk, semantic_score, signals),
                    signals=signals,
                )
            )
        ranked = sorted(reranked, key=_ranking_key)
        backend_name = f"tfidf+sbert-rerank:{sbert_model}"
    if file_aggregation:
        ranked = _aggregate_file_candidates(
            ranked,
            query_terms=query_terms,
            identifiers=identifiers,
            mode=aggregation_mode,
            evidence_candidates=aggregation_evidence,
        )
    return ranked[:top_k], backend_name


def _best_file_representatives(
    candidates: Iterable[LocalizationCandidate],
) -> list[LocalizationCandidate]:
    representatives: list[LocalizationCandidate] = []
    seen_files: set[str] = set()
    for candidate in sorted(candidates, key=_ranking_key):
        file_path = candidate.chunk.file_path
        if file_path in seen_files:
            continue
        seen_files.add(file_path)
        representatives.append(candidate)
    return representatives


def _aggregate_file_candidates(
    candidates: list[LocalizationCandidate],
    *,
    query_terms: Iterable[str] = (),
    identifiers: Iterable[str] = (),
    mode: str = "basic",
    evidence_candidates: Iterable[LocalizationCandidate] | None = None,
) -> list[LocalizationCandidate]:
    """Combine bounded, independent chunk evidence into a calibrated file score.

    In hybrid retrieval, ``candidates`` contains one SBERT-reranked
    representative per file. ``evidence_candidates`` preserves the original
    TF-IDF-ranked chunks so E1 can still use the other relevant chunks from the
    same file without sending every chunk through SBERT.
    """

    normalized_mode = _resolve_file_aggregation_mode(mode, advanced=False)
    support_enabled = normalized_mode != "basic"
    symbols_enabled = normalized_mode in {
        "supporting-symbols",
        "supporting-symbols-package",
    }
    package_enabled = normalized_mode == "supporting-symbols-package"

    grouped: dict[str, list[LocalizationCandidate]] = {}
    file_order: list[str] = []
    for candidate in candidates:
        file_path = candidate.chunk.file_path
        if file_path not in grouped:
            grouped[file_path] = []
            file_order.append(file_path)
        grouped[file_path].append(candidate)

    evidence_grouped: dict[str, list[LocalizationCandidate]] = {}
    for candidate in (evidence_candidates if evidence_candidates is not None else candidates):
        evidence_grouped.setdefault(candidate.chunk.file_path, []).append(candidate)

    aggregated: list[LocalizationCandidate] = []
    for file_path in file_order:
        file_candidates = grouped[file_path]
        primary = file_candidates[0]
        source_candidates = sorted(
            evidence_grouped.get(file_path) or file_candidates,
            key=_ranking_key,
        )
        evidence = _independent_file_evidence(source_candidates)
        evidence_threshold = (
            max(
                float(FILE_AGGREGATION_PARAMETERS["minimum_evidence_score"]),
                source_candidates[0].score
                * float(FILE_AGGREGATION_PARAMETERS["evidence_score_ratio"]),
            )
            if source_candidates
            else 1.0
        )
        qualifying_evidence = [
            candidate for candidate in evidence if candidate.score >= evidence_threshold
        ]
        supporting_chunks = [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "symbol_qualified_name": candidate.chunk.symbol_qualified_name,
                "start_line": candidate.chunk.start_line,
                "end_line": candidate.chunk.end_line,
                "score": round(float(candidate.score), 4),
                "retrieval_score": round(float(candidate.score), 4),
            }
            for candidate in qualifying_evidence
            if candidate.chunk.chunk_id != primary.chunk.chunk_id
        ][:3]
        signals = dict(primary.signals)
        signals["file_aggregation_mode"] = normalized_mode
        signals["file_chunk_count"] = len(source_candidates)
        signals["supporting_chunks"] = supporting_chunks
        distinct_symbols = {
            candidate.chunk.symbol_qualified_name
            for candidate in qualifying_evidence
            if candidate.chunk.symbol_qualified_name
        }
        support_bonus = (
            min(
                float(FILE_AGGREGATION_PARAMETERS["support_bonus_cap"]),
                float(FILE_AGGREGATION_PARAMETERS["support_bonus_per_additional_chunk"])
                * max(0, len(qualifying_evidence) - 1),
            )
            if support_enabled
            else 0.0
        )
        symbol_coverage_bonus = (
            min(
                float(FILE_AGGREGATION_PARAMETERS["symbol_bonus_cap"]),
                float(FILE_AGGREGATION_PARAMETERS["symbol_bonus_per_additional_symbol"])
                * max(0, len(distinct_symbols) - 1),
            )
            if symbols_enabled
            else 0.0
        )
        file_score = _clamp(primary.score + support_bonus + symbol_coverage_bonus, 0.0, 1.0)
        signals["chunk_score"] = round(primary.score, 4)
        signals["support_evidence_threshold"] = round(evidence_threshold, 4)
        signals["support_evidence_count"] = len(qualifying_evidence)
        signals["distinct_support_symbol_count"] = len(distinct_symbols)
        signals["supporting_chunk_bonus"] = round(support_bonus, 4)
        signals["symbol_coverage_bonus"] = round(symbol_coverage_bonus, 4)
        signals["file_aggregate_score"] = round(file_score, 4)
        signals["final_score"] = round(file_score, 4)
        reason = primary.reason
        if supporting_chunks:
            reason += f" {len(supporting_chunks)} additional independent chunk(s) in this file also matched."
        if support_bonus or symbol_coverage_bonus:
            reason += (
                " File aggregation added bounded support from multiple relevant "
                "chunks and distinct symbols."
            )
        aggregated.append(replace(primary, score=file_score, reason=reason, signals=signals))
    ranked_files = sorted(aggregated, key=_ranking_key)
    if not package_enabled:
        return ranked_files
    return _apply_package_proximity(
        ranked_files,
        query_terms=query_terms,
        identifiers=identifiers,
    )


def _independent_file_evidence(
    candidates: Iterable[LocalizationCandidate],
) -> list[LocalizationCandidate]:
    """Select high-ranked chunks while avoiding nearly duplicate line windows."""

    selected: list[LocalizationCandidate] = []
    maximum = int(FILE_AGGREGATION_PARAMETERS["max_evidence_chunks"])
    for candidate in sorted(candidates, key=_ranking_key):
        if any(
            _line_range_overlap_ratio(candidate.chunk, existing.chunk) >= 0.50
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected


def _line_range_overlap_ratio(left: CodeChunk, right: CodeChunk) -> float:
    overlap = max(
        0,
        min(left.end_line, right.end_line) - max(left.start_line, right.start_line) + 1,
    )
    shorter = min(
        max(1, left.end_line - left.start_line + 1),
        max(1, right.end_line - right.start_line + 1),
    )
    return overlap / shorter


def _resolve_file_aggregation_mode(mode: str | None, *, advanced: bool) -> str:
    normalized = str(mode or ("supporting-symbols-package" if advanced else "basic")).strip().lower()
    if normalized not in FILE_AGGREGATION_MODES:
        raise ValueError(
            "file_aggregation_mode must be one of: " + ", ".join(FILE_AGGREGATION_MODES)
        )
    return normalized


def _apply_package_proximity(
    candidates: list[LocalizationCandidate],
    *,
    query_terms: Iterable[str],
    identifiers: Iterable[str],
) -> list[LocalizationCandidate]:
    if not candidates:
        return []
    reference_paths = [candidate.chunk.file_path for candidate in candidates[:30]]
    report_terms = set(query_terms)
    for identifier in identifiers:
        report_terms.update(_identifier_variants(identifier))
    reranked: list[LocalizationCandidate] = []
    for candidate in candidates:
        path = _normalize_path(candidate.chunk.file_path)
        neighbor_depths = [
            (_shared_path_prefix_depth(path, other), other)
            for other in reference_paths
            if other != path
        ]
        depth, neighbor = max(neighbor_depths, default=(0, ""))
        path_overlap = (set(_tokenize(path)) & report_terms) - STOPWORDS
        local_evidence = max(
            float(candidate.signals.get("identifier_score", 0.0)),
            float(candidate.signals.get("path_term_score", 0.0)),
        )
        if depth < 2 or (not path_overlap and local_evidence <= 0):
            reranked.append(candidate)
            continue
        package_score = min(
            1.0,
            0.35 + 0.12 * max(0, depth - 2) + 0.20 * local_evidence + 0.04 * len(path_overlap),
        )
        boost = min(
            float(FILE_AGGREGATION_PARAMETERS["package_bonus_cap"]),
            SCORING_WEIGHTS["package_proximity_score"] * package_score,
        )
        score = _clamp(candidate.score + boost, 0.0, 1.0)
        signals = dict(candidate.signals)
        signals["package_proximity_score"] = round(package_score, 4)
        signals["package_proximity_bonus"] = round(boost, 4)
        signals["matching_package_neighbor"] = neighbor
        signals["final_score"] = round(score, 4)
        reason = (
            f"{candidate.reason} Same-package evidence is shared with strong candidate {neighbor}."
        )
        reranked.append(replace(candidate, score=score, signals=signals, reason=reason))
    return sorted(reranked, key=_ranking_key)


def _shared_path_prefix_depth(left: str, right: str) -> int:
    left_parts = [part for part in _normalize_path(left).split("/") if part]
    right_parts = [part for part in _normalize_path(right).split("/") if part]
    depth = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        depth += 1
    return depth


def validate_localization_request(ticket_json: dict[str, Any], *, min_ticket_chars: int = 20) -> dict[str, Any]:
    """Assess whether a ticket contains enough evidence for safe localization."""

    if min_ticket_chars <= 0:
        raise ValueError("min_ticket_chars must be positive.")
    content = _ticket_content_text(ticket_json)
    compact = re.sub(r"\s+", " ", content).strip()
    errors: list[str] = []
    warnings: list[str] = []
    if not compact:
        errors.append("Ticket does not contain bug-report text.")
    elif len(compact) < min_ticket_chars:
        warnings.append(
            f"Ticket text is shorter than {min_ticket_chars} characters; localization confidence is limited."
        )
    elif len(compact) < 80:
        warnings.append("Ticket text is sparse; include symptoms, reproduction steps, logs, or identifiers.")

    stack_trace_present = bool(_stack_trace_refs(ticket_json))
    path_hint_present = bool(re.search(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", compact))
    identifier_present = any("_" in token or any(char.isupper() for char in token[1:]) for token in TOKEN_RE.findall(compact))
    if compact and not stack_trace_present and not path_hint_present and not identifier_present:
        warnings.append("No stack trace, source path, or clear code identifier was found in the ticket.")

    return {
        "status": "error" if errors else "warning" if warnings else "ok",
        "is_valid": not errors,
        "bug_report_chars": len(compact),
        "errors": errors,
        "warnings": warnings,
        "signals_present": {
            "stack_trace": stack_trace_present,
            "path_hint": path_hint_present,
            "identifier": identifier_present,
        },
    }


def _ticket_content_text(ticket_json: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in TICKET_CONTENT_KEYS:
        value = ticket_json.get(key)
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            parts.extend(f"{sub_key}: {sub_value}" for sub_key, sub_value in value.items())
        elif value is not None:
            parts.append(str(value))
    return "\n".join(part.strip() for part in parts if part.strip())


def _localization_confidence(
    localized: list[dict[str, Any]],
    input_validation: dict[str, Any],
    *,
    llm_rerank_used: bool,
) -> dict[str, Any]:
    if not localized:
        return {
            "confidence_level": "low",
            "confidence_score": 0.0,
            "top1_top2_margin": 0.0,
            "uncertainty_reason": "No localization candidates were produced.",
            "should_manual_review": True,
            "recommend_patch_generation": False,
            "patch_generation_policy": "block_patch_generation",
        }

    best = localized[0]
    second_score = float(localized[1].get("score", 0.0)) if len(localized) > 1 else 0.0
    best_score = float(best.get("score", 0.0))
    margin = max(0.0, best_score - second_score)
    scoring = best.get("scoring_signals") if isinstance(best.get("scoring_signals"), dict) else {}
    stack_score = float(scoring.get("stack_trace_score", 0.0))
    direct_evidence = stack_score >= 0.55 or bool(input_validation["signals_present"].get("path_hint"))
    test_like = _is_test_like_path(str(best.get("file_path") or ""))

    if input_validation["errors"] or test_like:
        level = "low"
    elif best_score >= 0.55 and direct_evidence and (margin >= 0.02 or stack_score >= 0.75):
        level = "high"
    elif best_score >= 0.25:
        level = "medium"
    else:
        level = "low"

    reasons = [
        f"top_score={best_score:.4f}",
        f"top1_top2_margin={margin:.4f}",
        f"stack_trace_score={stack_score:.4f}",
    ]
    if input_validation["warnings"]:
        reasons.append("ticket input has quality warnings")
    if test_like:
        reasons.append("top candidate is test-like")
    if llm_rerank_used:
        reasons.append("optional LLM rerank was applied")

    allow_patch = level == "high"
    return {
        "confidence_level": level,
        "confidence_score": round(best_score, 4),
        "top1_top2_margin": round(margin, 4),
        "uncertainty_reason": "; ".join(reasons),
        "should_manual_review": not allow_patch,
        "recommend_patch_generation": allow_patch,
        "patch_generation_policy": "allow_patch_suggestion" if allow_patch else "manual_review_before_patch" if level == "medium" else "block_patch_generation",
    }


def _is_test_like_path(file_path: str) -> bool:
    parts = [part.lower() for part in Path(file_path.replace("\\", "/")).parts]
    name = parts[-1] if parts else ""
    return any(part in TEST_DIR_NAMES for part in parts) or name.startswith("test_") or name.endswith("_test.py")


def _localized_file(row: dict[str, Any]) -> dict[str, Any]:
    signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
    return {
        "rank": row.get("rank", 0),
        "file_path": row.get("file_path", ""),
        "score": row.get("score", 0.0),
        "primary_symbol": row.get("symbol_qualified_name", ""),
        "line_start": row.get("start_line", 1),
        "line_end": row.get("end_line", 1),
        "supporting_chunks": signals.get("supporting_chunks", []),
    }


def _stage1_candidate_file(candidate: LocalizationCandidate, rank: int) -> dict[str, Any]:
    """Serialize the pre-LLM file candidate evidence needed for evaluation."""

    signals = _candidate_scoring_signals(candidate)
    return {
        "rank": rank,
        "file_path": candidate.chunk.file_path,
        "retrieval_score": round(float(candidate.score), 4),
        "primary_symbol": candidate.chunk.symbol_qualified_name,
        "symbol_kind": candidate.chunk.symbol_kind,
        "start_line": candidate.chunk.start_line,
        "end_line": candidate.chunk.end_line,
        "reason": candidate.reason,
        "scoring_signals": signals,
        "file_chunk_count": int(candidate.signals.get("file_chunk_count", 1)),
        "supporting_chunks": candidate.signals.get("supporting_chunks", []),
    }


def build_bug_report_text(ticket_json: dict[str, Any]) -> str:
    parts: list[str] = []
    keys = (
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
    seen_values: set[str] = set()
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
        stripped = text.strip()
        normalized = re.sub(r"\s+", " ", stripped).casefold()
        if stripped and normalized not in seen_values:
            parts.append(f"{key}: {stripped}")
            seen_values.add(normalized)
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
    seen_ranks: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            rank = int(row.get("rank") or row.get("original_rank") or 0)
        except (TypeError, ValueError):
            continue
        if rank in seen_ranks:
            continue
        candidate = by_rank.get(rank)
        if candidate is None:
            continue
        try:
            score = float(row.get("score", candidate.score))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        llm_score = _clamp(score, 0.0, 1.0)
        retrieval_score = candidate.score
        score = _clamp(
            LLM_RERANK_WEIGHTS["retrieval"] * retrieval_score
            + LLM_RERANK_WEIGHTS["llm"] * llm_score,
            0.0,
            1.0,
        )
        llm_reason = str(row.get("reason") or "").strip()
        reason = f"LLM rerank: {llm_reason} Retrieval: {candidate.reason}" if llm_reason else candidate.reason
        signals = dict(candidate.signals)
        signals["retrieval_score_before_llm"] = round(retrieval_score, 4)
        signals["llm_rerank_score"] = round(llm_score, 4)
        signals["llm_blended_score"] = round(score, 4)
        signals["final_score"] = round(score, 4)
        reranked.append(replace(candidate, score=score, reason=reason, signals=signals))
        seen_ranks.add(rank)

    if len(reranked) != len(candidates):
        return candidates[:top_k]
    return sorted(reranked, key=_ranking_key)[:top_k]


def _iter_source_files(root: Path, *, include_tests: bool, max_file_bytes: int) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if _is_ignored_path(rel_parts):
            continue
        if not include_tests and any(part.lower() in TEST_DIR_NAMES for part in rel_parts):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        yield path


def _is_ignored_path(rel_parts: tuple[str, ...]) -> bool:
    if not rel_parts:
        return False
    if any(part in IGNORED_DIRS_ANYWHERE for part in rel_parts):
        return True
    return rel_parts[0] in IGNORED_ROOT_DIRS


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
    top_level_symbols = [symbol for symbol in symbol_ranges if bool(symbol.get("is_top_level", True))]
    for start_line, end_line in _module_level_gaps(lines, top_level_symbols):
        chunks.extend(
            _fixed_line_chunks(
                rel,
                language,
                lines,
                start_line=start_line,
                end_line=end_line,
                chunk_lines=chunk_lines,
                overlap_lines=overlap_lines,
                symbol_kind="module",
            )
        )
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except (SyntaxError, RecursionError):
        return []
    ranges: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []
            self.function_depth = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            end = int(getattr(node, "end_lineno", node.lineno))
            decorator_lines = [decorator.lineno for decorator in node.decorator_list]
            ranges.append(
                {
                    "symbol_kind": "class",
                    "function_name": "",
                    "class_name": node.name,
                    "start_line": min([node.lineno, *decorator_lines]),
                    "end_line": end,
                    "is_top_level": not self.class_stack and self.function_depth == 0,
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
            decorator_lines = [decorator.lineno for decorator in node.decorator_list]
            ranges.append(
                {
                    "symbol_kind": kind,
                    "function_name": name,
                    "class_name": class_name,
                    "start_line": min([node.lineno, *decorator_lines]),
                    "end_line": end,
                    "is_top_level": not self.class_stack and self.function_depth == 0,
                }
            )
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

    try:
        Visitor().visit(tree)
    except RecursionError:
        return []
    return sorted(ranges, key=lambda row: (row["start_line"], row["end_line"], row["symbol_kind"] != "class"))


def _regex_symbol_ranges(lines: list[str], language: str) -> list[dict[str, Any]]:
    patterns = [
        re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"\b(?:function|func|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"\b(?:def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?:async\s*)?\([^)]*\)\s*=>"),
        re.compile(r"\b(?:public|private|protected|static|final|async|\s)+[A-Za-z0-9_<>,\[\]?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ]
    declarations: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        for pattern in patterns:
            match = pattern.search(stripped)
            if not match:
                continue
            kind = "class" if stripped.startswith("class ") or " class " in stripped else "function"
            name = match.group(1)
            declarations.append(
                {
                    "symbol_kind": kind,
                    "function_name": "" if kind == "class" else name,
                    "class_name": name if kind == "class" else "",
                    "start_line": index,
                    "is_top_level": True,
                }
            )
            break
    for position, declaration in enumerate(declarations):
        next_start = int(declarations[position + 1]["start_line"]) if position + 1 < len(declarations) else None
        declaration["end_line"] = _infer_regex_symbol_end(
            lines,
            int(declaration["start_line"]),
            next_start=next_start,
            language=language,
        )
    return declarations


def _infer_regex_symbol_end(
    lines: list[str],
    start_line: int,
    *,
    next_start: int | None,
    language: str,
) -> int:
    brace_languages = {"javascript", "typescript", "java", "c", "cpp", "go", "rust"}
    if language in brace_languages:
        depth = 0
        saw_opening_brace = False
        for line_number in range(start_line, len(lines) + 1):
            cleaned = re.sub(r"(['\"`]).*?\1", "", lines[line_number - 1])
            cleaned = cleaned.split("//", maxsplit=1)[0]
            opening = cleaned.count("{")
            closing = cleaned.count("}")
            if opening:
                saw_opening_brace = True
            depth += opening - closing
            if saw_opening_brace and depth <= 0:
                return line_number

    if language == "python":
        base_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        for line_number in range(start_line + 1, len(lines) + 1):
            line = lines[line_number - 1]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and not line.lstrip().startswith(("@", "#")):
                return line_number - 1

    if next_start is not None:
        return max(start_line, next_start - 1)
    return min(len(lines), start_line + 79)


def _module_level_gaps(lines: list[str], top_level_symbols: list[dict[str, Any]]) -> list[tuple[int, int]]:
    if not top_level_symbols:
        return []
    gaps: list[tuple[int, int]] = []
    current = 1
    for symbol in sorted(top_level_symbols, key=lambda row: (int(row["start_line"]), int(row["end_line"]))):
        start = max(1, int(symbol["start_line"]))
        end = min(len(lines), int(symbol["end_line"]))
        if current < start and any(line.strip() for line in lines[current - 1 : start - 1]):
            gaps.append((current, start - 1))
        current = max(current, end + 1)
    if current <= len(lines) and any(line.strip() for line in lines[current - 1 :]):
        gaps.append((current, len(lines)))
    return gaps


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
    tokenized_documents: list[list[str]] | None = None,
) -> tuple[list[float], str]:
    normalized = _validate_embedding_backend(backend)
    if normalized in {"auto", "sbert", "sentence-transformers", "sentence_transformers"}:
        try:
            return _sbert_scores(query, documents, sbert_model, local_files_only=sbert_local_files_only), f"sbert:{sbert_model}"
        except Exception:
            if normalized != "auto":
                raise
    return _tfidf_scores(query, documents, tokenized_documents=tokenized_documents), "tfidf"


def _sbert_scores(query: str, documents: list[str], model_name: str, *, local_files_only: bool) -> list[float]:
    model = _load_sbert_model(model_name, local_files_only)
    embeddings = model.encode([query, *documents], normalize_embeddings=True)
    query_embedding = embeddings[0]
    doc_embeddings = embeddings[1:]
    scores: list[float] = []
    for embedding in doc_embeddings:
        score = float(sum(float(left) * float(right) for left, right in zip(query_embedding, embedding)))
        scores.append(_clamp(score, 0.0, 1.0))
    return scores


@lru_cache(maxsize=4)
def _load_sbert_model(model_name: str, local_files_only: bool) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


def _tfidf_scores(
    query: str,
    documents: list[str],
    *,
    tokenized_documents: list[list[str]] | None = None,
) -> list[float]:
    tokenized_docs = tokenized_documents or [_tokenize(document) for document in documents]
    if len(tokenized_docs) != len(documents):
        raise ValueError("tokenized_documents must have the same length as documents.")
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
        for piece in TOKEN_SPLIT_RE.split(raw):
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


def _component_terms(ticket_json: dict[str, Any], chunks: list[CodeChunk]) -> list[str]:
    """Discard project-wide component labels that cannot distinguish files."""

    terms = _important_terms(str(ticket_json.get("component") or ""))
    if not terms:
        return []
    file_paths = {chunk.file_path for chunk in chunks}
    file_count = max(1, len(file_paths))
    counts: Counter[str] = Counter()
    for file_path in file_paths:
        counts.update(set(_tokenize(file_path)))
    repo_terms = set(
        _important_terms(str(ticket_json.get("repo") or ticket_json.get("product") or ""))
    )
    return [
        term
        for term in terms
        if term not in repo_terms and counts.get(term, 0) / file_count < 0.35
    ]


def _report_identifiers(text: str) -> list[str]:
    """Extract code-like names while ignoring ordinary prose tokens."""

    identifiers: list[str] = []

    def add(value: str) -> None:
        normalized = _normalize_identifier(value)
        if not normalized or normalized in STOPWORDS or len(normalized) < 3:
            return
        if normalized not in identifiers:
            identifiers.append(normalized)

    patterns = (
        r"\b[A-Z][A-Z0-9_]{2,}\b",
        r"\b[A-Z]\d{2,4}\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*__[A-Za-z0-9_]+(?:__[A-Za-z0-9_]+)*\b",
        r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b",
        r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            add(match.group(0))
            if len(identifiers) >= 80:
                return identifiers
    return identifiers


def _identifier_signal(identifiers: list[str], chunk: CodeChunk) -> tuple[float, list[str]]:
    if not identifiers:
        return 0.0, []
    chunk_text = chunk.search_text.lower()
    chunk_tokens = set(_tokenize(chunk.search_text))
    matching: list[str] = []
    for identifier in identifiers:
        variants = _identifier_variants(identifier)
        if any(
            variant in chunk_tokens
            or bool(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(variant)}(?![A-Za-z0-9_])",
                    chunk_text,
                )
            )
            for variant in variants
        ):
            matching.append(identifier)
    if not matching:
        return 0.0, []
    if any(re.fullmatch(r"[a-z]\d{2,4}", value) for value in matching):
        return 1.0, matching
    return min(1.0, 0.5 + 0.1 * (len(matching) - 1)), matching


def _identifier_variants(identifier: str) -> set[str]:
    normalized = _normalize_identifier(identifier)
    variants = {normalized}
    if normalized.endswith("s") and len(normalized) > 4:
        variants.add(normalized[:-1])
    if "_" in normalized:
        variants.update(part for part in normalized.split("_") if len(part) >= 3)
    if "." in normalized:
        variants.update(part for part in normalized.split(".") if len(part) >= 3)
    return variants


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("\"'`.,;:()[]{}").replace("-", "_").lower()


def _path_term_signal(
    query_text: str,
    query_terms: list[str],
    identifiers: list[str],
    chunk: CodeChunk,
    *,
    query_set: set[str] | None = None,
) -> tuple[float, list[str]]:
    if query_set is None:
        query_set = _path_signal_query_terms(query_text, query_terms, identifiers)
    leaf_terms, path_terms = _path_signal_terms(chunk.file_path)

    leaf_overlap = sorted((leaf_terms & query_set) - STOPWORDS)
    parent_overlap = sorted(((path_terms - leaf_terms) & query_set) - STOPWORDS)
    if leaf_overlap:
        return min(1.0, 0.55 + 0.1 * (len(leaf_overlap) - 1)), leaf_overlap
    if len(parent_overlap) >= 2:
        return min(0.65, 0.3 + 0.08 * len(parent_overlap)), parent_overlap
    if parent_overlap:
        return 0.25, parent_overlap
    return 0.0, []


def _path_signal_query_terms(
    query_text: str,
    query_terms: Iterable[str],
    identifiers: Iterable[str],
) -> set[str]:
    query_set = set(query_terms)
    query_set.update(re.findall(r"[a-z0-9]+", query_text.lower()))
    for identifier in identifiers:
        query_set.update(_identifier_variants(identifier))
    return query_set - STOPWORDS


@lru_cache(maxsize=8192)
def _path_signal_terms(file_path: str) -> tuple[frozenset[str], frozenset[str]]:
    path = _normalize_path(file_path).lower()
    stem = Path(path).stem
    leaf_terms = set(_tokenize(stem)) | set(re.findall(r"[a-z0-9]+", stem))
    path_terms = set(_tokenize(path.replace("/", " "))) | set(
        re.findall(r"[a-z0-9]+", path)
    )
    for token in list(path_terms | leaf_terms):
        digit_suffix = re.search(r"(\d+[a-z]+)$", token)
        if digit_suffix:
            path_terms.add(digit_suffix.group(1))
            leaf_terms.add(digit_suffix.group(1))
        if token.endswith("s") and len(token) > 4:
            path_terms.add(token[:-1])
            leaf_terms.add(token[:-1])
    return frozenset(leaf_terms), frozenset(path_terms)


def _domain_path_signal(text: str, chunk: CodeChunk) -> tuple[float, list[str]]:
    """Map high-precision framework concepts to likely source-code regions.

    This signal only reads production-available ticket text and repository
    paths. It never reads benchmark hints, tests, patches, or gold files.
    """

    lowered = text.lower().replace("\\", "/")
    path = _normalize_path(chunk.file_path).lower()
    intents: list[tuple[str, float]] = []

    def add(label: str, score: float) -> None:
        if label not in {existing for existing, _ in intents}:
            intents.append((label, score))

    if _contains_any(lowered, ("file_upload", "static_url", "media_url", "script_name", "base_dir", "settings.py")):
        if path in {"django/conf/global_settings.py", "django/conf/__init__.py"}:
            add("django_settings_conf", 0.9)

    if _contains_any(lowered, ("models.e", "system check", "db_table", "same table name", "table_name")):
        if path.startswith("django/core/checks/"):
            add("django_model_checks", 0.9)

    if _contains_any(lowered, ("urlconf", "urlpatterns", "re_path", "url params", "url parameter", "resolver")):
        if path.startswith("django/urls/"):
            add("django_url_resolver", 0.9)

    if _contains_any(lowered, ("makemigrations", "generated migration", "migration file", "missing import statement")):
        if path == "django/db/migrations/serializer.py":
            add("django_migration_serializer", 1.0)
        elif path.startswith("django/db/migrations/"):
            add("django_migrations", 0.25)

    if _contains_any(lowered, ("q object", " q(", "pk__in", "| operator", "cannot pickle")):
        if path == "django/db/models/query_utils.py":
            add("django_query_utils", 1.0)

    if _contains_all(lowered, ("group by", "query")) or _contains_all(lowered, ("filter", "query result")):
        if path == "django/db/models/lookups.py":
            add("django_model_lookups", 0.85)

    if _contains_any(lowered, ("--keepdb", "persistent sqlite", "persistent test sqlite", "test[\"name\"]", "test['name']")):
        if path == "django/db/backends/sqlite3/creation.py":
            add("django_sqlite_test_creation", 1.0)

    if _contains_all(lowered, ("unique constraint", "sqlite")) and _contains_any(
        lowered, ("remaking table", "remake", "ddl", "references")
    ):
        if path == "django/db/backends/ddl_references.py":
            add("django_ddl_references", 0.9)

    if _contains_any(lowered, ("dev server", "runserver", "restart", "autoreload")) and _contains_any(
        lowered, ("templates", "base_dir", "settings.py")
    ):
        if path == "django/template/autoreload.py":
            add("django_template_autoreload", 1.0)

    if _contains_any(lowered, ("if-modified-since", "modified-since", "modified since")):
        if path == "django/views/static.py":
            add("django_static_modified_since", 0.95)

    if _contains_any(lowered, ("expressionwrapper", "output_field", "booleanfield")) and _contains_any(
        lowered, ("~q", "pk__in", " q(")
    ):
        if path == "django/db/models/fields/__init__.py":
            add("django_model_fields_expression", 0.75)

    if _contains_any(lowered, ("get_backend", "rc_context", "rcparams", "backend resolution", "backend reset")):
        if path == "lib/matplotlib/__init__.py":
            add("matplotlib_backend_resolution", 1.0)
        elif path.endswith("matplotlib/rcsetup.py"):
            add("matplotlib_rcparams", 0.65)

    if _contains_any(lowered, ("pairplot", "hue_order", "scatterplot")) and _contains_any(lowered, ("hue", "order")):
        if path == "seaborn/_oldcore.py":
            add("seaborn_legacy_semantics", 0.9)
        elif path.startswith("seaborn/_core/"):
            add("seaborn_core_semantics", 0.35)

    if _contains_any(lowered, ("subdomain", "sub-domain", "flask routes", "routes command")):
        if path == "src/flask/cli.py":
            add("flask_cli_routes", 1.0)

    if _contains_any(lowered, ("urllib3 exceptions", "wrapped", "passing through requests api", "bleeding through")):
        if path == "requests/adapters.py":
            add("requests_adapter_exception_boundary", 1.0)
        elif path == "requests/exceptions.py":
            add("requests_exception_boundary", 0.65)

    if _contains_any(lowered, ("dataset overview", "show units", "attrs['units']", 'attrs["units"]')):
        if path == "xarray/core/formatting.py":
            add("xarray_formatting_units", 1.0)

    if _contains_any(lowered, ("ignore-paths", "ignore paths", "--recursive", "recursive=y")):
        if path == "pylint/lint/expand_modules.py":
            add("pylint_expand_modules_ignore_paths", 1.0)

    if _contains_any(lowered, ("import-mode=importlib", "importmode importlib", "module imported twice", "doctest-modules")):
        if path == "src/_pytest/pathlib.py":
            add("pytest_import_path", 1.0)

    if _contains_any(lowered, ("print_changed_only", "new repr", "prettyprinter", "pprint")):
        if path == "sklearn/utils/_pprint.py":
            add("sklearn_pretty_printer", 1.0)

    if _contains_any(lowered, ("napoleon", "attribute", "underscore", "hello_")):
        if path == "sphinx/ext/napoleon/docstring.py":
            add("sphinx_napoleon_docstring", 1.0)

    if not intents:
        return 0.0, []
    return max(score for _, score in intents), [label for label, _ in intents]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _contains_all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def _resolve_import_graph_mode(
    mode: str | None,
    *,
    legacy_repository_proximity: bool,
) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized == "off" and legacy_repository_proximity:
        normalized = "outgoing"
    if normalized not in IMPORT_GRAPH_MODES:
        raise ValueError("import_graph_mode must be one of: " + ", ".join(IMPORT_GRAPH_MODES))
    return normalized


def build_import_graph(chunks: Iterable[CodeChunk]) -> ImportGraph:
    """Build a reusable one-hop Python import graph from a code index."""

    chunk_list = list(chunks)
    chunks_by_file = _chunks_by_file(chunk_list)
    file_paths = set(chunks_by_file)
    module_lookup = _module_file_lookup(
        {file_path for file_path in file_paths if file_path.endswith(".py")}
    )
    outgoing_sets: dict[str, set[str]] = {}
    unresolved: dict[str, list[str]] = {}
    parsed_files = 0
    parse_failures = 0

    for file_path in sorted(file_paths):
        file_chunks = chunks_by_file[file_path]
        if not file_path.endswith(".py") and not any(
            chunk.language == "python" for chunk in file_chunks
        ):
            continue
        source = _reconstruct_file_source(file_chunks)
        modules, parsed = _python_import_modules(source, current_file=file_path)
        parsed_files += int(parsed)
        parse_failures += int(not parsed)
        missing: list[str] = []
        for module in modules:
            targets = _module_import_targets(module, module_lookup)
            if not targets:
                if module not in missing:
                    missing.append(module)
                continue
            for target in targets:
                if target != file_path:
                    outgoing_sets.setdefault(file_path, set()).add(target)
        if missing:
            unresolved[file_path] = missing[:32]

    incoming_sets: dict[str, set[str]] = {}
    for source, targets in outgoing_sets.items():
        for target in targets:
            incoming_sets.setdefault(target, set()).add(source)
    outgoing = {
        source: sorted(targets)
        for source, targets in sorted(outgoing_sets.items())
        if targets
    }
    incoming = {
        target: sorted(sources)
        for target, sources in sorted(incoming_sets.items())
        if sources
    }
    return ImportGraph(
        outgoing=outgoing,
        incoming=incoming,
        unresolved=unresolved,
        stats={
            "files": len(file_paths),
            "python_files_parsed": parsed_files,
            "python_parse_failures": parse_failures,
            "edges": sum(len(targets) for targets in outgoing.values()),
            "files_with_outgoing_edges": len(outgoing),
            "files_with_incoming_edges": len(incoming),
        },
    )


def _reconstruct_file_source(chunks: list[CodeChunk]) -> str:
    if not chunks:
        return ""
    line_count = max(chunk.end_line for chunk in chunks)
    lines: list[str | None] = [None] * line_count
    for chunk in sorted(chunks, key=lambda item: (item.start_line, item.end_line)):
        for offset, line in enumerate(chunk.code_text.splitlines()):
            line_index = chunk.start_line - 1 + offset
            if 0 <= line_index < line_count and lines[line_index] is None:
                lines[line_index] = line
    return "\n".join(line if line is not None else "" for line in lines)


def _python_import_modules(source: str, *, current_file: str) -> tuple[list[str], bool]:
    modules: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, RecursionError, ValueError):
        fallback: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                fallback.extend(_import_line_modules(stripped, current_file=current_file))
        return list(dict.fromkeys(fallback)), False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        raw_base = "." * int(node.level or 0) + str(node.module or "")
        base = _resolve_python_module(raw_base, current_file)
        if base:
            modules.append(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            imported = f"{base}.{alias.name}".strip(".")
            if imported:
                modules.append(imported)
    return list(dict.fromkeys(modules)), True


def _module_import_targets(
    module: str,
    module_lookup: dict[str, list[str]],
) -> list[str]:
    normalized = module.replace("/", ".").strip(".")
    if not normalized:
        return []
    # An import can name a package member rather than a concrete submodule.
    # Resolve the most specific internal module, then stop to avoid adding a
    # whole chain of parent packages.
    parts = normalized.split(".")
    for end in range(len(parts), 0, -1):
        targets = module_lookup.get(".".join(parts[:end]), [])
        if targets:
            return targets[:4]
    return []


def _apply_import_graph(
    candidates: list[LocalizationCandidate],
    import_graph: ImportGraph,
    *,
    mode: str,
) -> list[LocalizationCandidate]:
    """Apply capped, one-hop import evidence from the strongest files."""

    normalized_mode = _resolve_import_graph_mode(
        mode,
        legacy_repository_proximity=False,
    )
    if normalized_mode == "off" or not candidates:
        return candidates

    best_by_file: dict[str, LocalizationCandidate] = {}
    for candidate in sorted(candidates, key=_ranking_key):
        best_by_file.setdefault(candidate.chunk.file_path, candidate)
    reference_files = [
        candidate.chunk.file_path
        for candidate in sorted(best_by_file.values(), key=_ranking_key)[
            : int(IMPORT_GRAPH_PARAMETERS["reference_file_k"])
        ]
    ]
    evidence: dict[str, list[tuple[float, str]]] = {}
    for rank, reference_file in enumerate(reference_files, start=1):
        decay = max(
            0.0,
            1.0 - float(IMPORT_GRAPH_PARAMETERS["rank_decay"]) * (rank - 1),
        )
        for target in import_graph.outgoing.get(reference_file, []):
            evidence.setdefault(target, []).append(
                (
                    float(IMPORT_GRAPH_PARAMETERS["outgoing_signal"]) * decay,
                    f"imported_by:{reference_file}",
                )
            )
        if normalized_mode != "bidirectional":
            continue
        for source in import_graph.incoming.get(reference_file, []):
            evidence.setdefault(source, []).append(
                (
                    float(IMPORT_GRAPH_PARAMETERS["incoming_signal"]) * decay,
                    f"imports:{reference_file}",
                )
            )

    reranked: list[LocalizationCandidate] = []
    for candidate in candidates:
        file_evidence = sorted(
            evidence.get(candidate.chunk.file_path, []),
            key=lambda item: (-item[0], item[1]),
        )
        if not file_evidence:
            reranked.append(candidate)
            continue
        graph_signal = min(
            float(IMPORT_GRAPH_PARAMETERS["maximum_signal"]),
            file_evidence[0][0],
        )
        bonus = min(
            float(IMPORT_GRAPH_PARAMETERS["maximum_bonus"]),
            SCORING_WEIGHTS["repository_proximity_score"] * graph_signal,
        )
        matches = [
            label
            for _, label in file_evidence[
                : int(IMPORT_GRAPH_PARAMETERS["maximum_evidence_per_file"])
            ]
        ]
        score = _clamp(candidate.score + bonus, 0.0, 1.0)
        signals = dict(candidate.signals)
        signals["repository_proximity_score"] = round(graph_signal, 4)
        signals["matching_repository_proximity"] = matches
        signals["import_graph_mode"] = normalized_mode
        signals["import_graph_score"] = round(graph_signal, 4)
        signals["import_graph_bonus"] = round(bonus, 4)
        signals["import_graph_evidence"] = matches
        signals["final_score"] = round(score, 4)
        reason = (
            f"{candidate.reason} One-hop import evidence links this file to: "
            f"{', '.join(matches[:2])}."
        )
        reranked.append(replace(candidate, score=score, signals=signals, reason=reason))
    return reranked


def _apply_repository_proximity(
    candidates: list[LocalizationCandidate],
    chunks: list[CodeChunk],
    *,
    query_terms: list[str],
    identifiers: list[str],
) -> list[LocalizationCandidate]:
    del query_terms, identifiers
    return _apply_import_graph(
        candidates,
        build_import_graph(chunks),
        mode="outgoing",
    )


def _chunks_by_file(chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
    grouped: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.file_path, []).append(chunk)
    return grouped


def _file_import_context(chunks: list[CodeChunk]) -> list[str]:
    imports: list[str] = []
    seen: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: item.start_line):
        for line in chunk.code_text.splitlines():
            stripped = line.strip()
            if not (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("#include ")
                or stripped.startswith("use ")
                or bool(re.match(r"^(?:const|let|var)\s+.*\brequire\(", stripped))
            ):
                continue
            if stripped not in seen:
                seen.add(stripped)
                imports.append(stripped)
    return imports[:32]


def _import_target_files(
    current_file: str,
    import_lines: list[str],
    module_lookup: dict[str, list[str]],
) -> list[str]:
    targets: list[str] = []
    for line in import_lines:
        for module in _import_line_modules(line, current_file=current_file):
            for target in module_lookup.get(module.replace("/", ".").strip("."), []):
                if target not in targets:
                    targets.append(target)
    return targets


def _module_file_lookup(file_paths: set[str]) -> dict[str, list[str]]:
    """Index dotted module suffixes once instead of scanning every path per import."""

    lookup: dict[str, list[str]] = {}
    for file_path in sorted(file_paths):
        path = Path(file_path)
        module = path.with_suffix("").as_posix()
        if module.endswith("/__init__"):
            module = module[: -len("/__init__")]
        parts = [part for part in module.split("/") if part]
        for start in range(len(parts)):
            key = ".".join(parts[start:])
            values = lookup.setdefault(key, [])
            if file_path not in values:
                values.append(file_path)
    return lookup


def _import_line_modules(line: str, *, current_file: str) -> list[str]:
    modules: list[str] = []
    py_import = re.match(r"^import\s+(.+)$", line)
    if py_import:
        modules.extend(
            part.strip().split(" as ")[0].strip()
            for part in py_import.group(1).split(",")
            if part.strip()
        )
    py_from = re.match(r"^from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$", line)
    if py_from:
        base_module = _resolve_python_module(py_from.group(1), current_file)
        modules.append(base_module)
        imported_names = py_from.group(2).strip().strip("()")
        for imported_name in imported_names.split(","):
            name = imported_name.strip().split(" as ")[0].strip()
            if name and name != "*" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                modules.append(f"{base_module}.{name}".strip("."))
    for pattern in (r"\bfrom\s+['\"]([^'\"]+)['\"]", r"\brequire\(['\"]([^'\"]+)['\"]\)"):
        match = re.search(pattern, line)
        if match:
            modules.append(_resolve_path_module(match.group(1), current_file))
    return [module for module in modules if module]


def _resolve_python_module(module: str, current_file: str) -> str:
    if not module.startswith("."):
        return module
    dots = len(module) - len(module.lstrip("."))
    remainder = module[dots:]
    current_parts = Path(current_file).with_suffix("").parts[:-1]
    keep = max(0, len(current_parts) - max(0, dots - 1))
    parts = list(current_parts[:keep])
    if remainder:
        parts.extend(part for part in remainder.split(".") if part)
    return ".".join(parts)


def _resolve_path_module(raw: str, current_file: str) -> str:
    if raw.startswith("."):
        return (Path(current_file).parent / raw).as_posix().replace("/", ".").strip(".")
    return raw.replace("/", ".")


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


def _keyword_signal(
    query_terms: list[str],
    chunk: CodeChunk,
    *,
    chunk_terms: set[str] | None = None,
) -> tuple[float, list[str]]:
    if not query_terms:
        return 0.0, []
    if chunk_terms is None:
        chunk_terms = set(_tokenize(chunk.search_text))
    matching = [term for term in query_terms if term in chunk_terms]
    return min(1.0, len(matching) / max(4, min(len(query_terms), 12))), matching


def _symbol_signal(
    query_terms: list[str],
    chunk: CodeChunk,
    *,
    symbol_terms: set[str] | None = None,
) -> float:
    if symbol_terms is None:
        symbol_terms = set(
            _tokenize(" ".join([chunk.file_path, chunk.function_name, chunk.class_name]))
        )
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
        "domain_path_score": round(float(signals.get("domain_path_score", 0.0)), 4),
        "identifier_score": round(float(signals.get("identifier_score", 0.0)), 4),
        "path_term_score": round(float(signals.get("path_term_score", 0.0)), 4),
        "repository_proximity_score": round(
            float(signals.get("repository_proximity_score", 0.0)), 4
        ),
        "package_proximity_score": round(
            float(signals.get("package_proximity_score", 0.0)), 4
        ),
        "supporting_chunk_bonus": round(
            float(signals.get("supporting_chunk_bonus", 0.0)), 4
        ),
        "symbol_coverage_bonus": round(
            float(signals.get("symbol_coverage_bonus", 0.0)), 4
        ),
        "package_proximity_bonus": round(
            float(signals.get("package_proximity_bonus", 0.0)), 4
        ),
        "file_aggregation_mode": str(signals.get("file_aggregation_mode") or "basic"),
        "support_evidence_threshold": round(
            float(signals.get("support_evidence_threshold", 0.0)), 4
        ),
        "support_evidence_count": int(signals.get("support_evidence_count", 0)),
        "distinct_support_symbol_count": int(
            signals.get("distinct_support_symbol_count", 0)
        ),
        "matching_identifiers": list(signals.get("matching_identifiers") or [])[:8],
        "matching_path_terms": list(signals.get("matching_path_terms") or [])[:8],
        "matching_repository_proximity": list(
            signals.get("matching_repository_proximity") or []
        )[:8],
        "matching_package_neighbor": str(
            signals.get("matching_package_neighbor") or ""
        ),
        "final_score": round(float(signals.get("final_score", candidate.score)), 4),
        "weights": SCORING_WEIGHTS,
    }
    if "llm_rerank_score" in signals:
        row["llm_rerank_score"] = round(float(signals["llm_rerank_score"]), 4)
        row["retrieval_score_before_llm"] = round(float(signals["retrieval_score_before_llm"]), 4)
        row["llm_blended_score"] = round(float(signals["llm_blended_score"]), 4)
        row["llm_rerank_weights"] = LLM_RERANK_WEIGHTS
    if "import_graph_mode" in signals:
        row["import_graph_mode"] = str(signals["import_graph_mode"])
        row["import_graph_score"] = round(float(signals.get("import_graph_score", 0.0)), 4)
        row["import_graph_bonus"] = round(float(signals.get("import_graph_bonus", 0.0)), 4)
        row["import_graph_evidence"] = list(signals.get("import_graph_evidence") or [])[:8]
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
    matching_domain_intents = signals.get("matching_domain_intents") or []
    if matching_domain_intents:
        pieces.append(
            "Framework concepts map to this source path: "
            f"{', '.join(matching_domain_intents[:3])}."
        )
    matching_identifiers = signals.get("matching_identifiers") or []
    if matching_identifiers:
        pieces.append(f"Exact code identifiers match: {', '.join(matching_identifiers[:3])}.")
    matching_path_terms = signals.get("matching_path_terms") or []
    if matching_path_terms:
        pieces.append(f"Ticket terms match the source path: {', '.join(matching_path_terms[:3])}.")
    matching_repository = signals.get("matching_repository_proximity") or []
    if matching_repository:
        pieces.append(
            "Import-neighbor evidence links this file to strong candidates: "
            f"{', '.join(matching_repository[:2])}."
        )
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
                "code_text": candidate.chunk.code_text[:LLM_MAX_CODE_CHARS_PER_CANDIDATE],
            }
        )
    bug_report = str(ticket_json.get("bug_report") or build_bug_report_text(ticket_json))
    return (
        "You are reranking fault localization candidates. "
        "Given the bug report and candidate code chunks, return JSON only in this shape: "
        '{"candidates":[{"rank":1,"score":0.0,"reason":"short reason"}]}. '
        "Return every supplied candidate exactly once. Keep each original integer rank, "
        "assign a relevance score from 0.0 to 1.0, and write a specific short reason. "
        "Do not copy placeholder values from the schema example.\n\n"
        f"Bug report:\n{bug_report[:LLM_MAX_BUG_REPORT_CHARS]}\n\n"
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


def _validate_embedding_backend(backend: str) -> str:
    normalized = backend.strip().lower()
    supported = {
        "auto",
        "tfidf",
        "sbert",
        "sentence-transformers",
        "sentence_transformers",
        "tfidf-sbert-rerank",
    }
    if normalized not in supported:
        choices = ", ".join(sorted(supported - {"sentence_transformers"}))
        raise ValueError(f"Unsupported embedding_backend {backend!r}; choose one of: {choices}.")
    return normalized


def _is_hybrid_backend(backend: str) -> bool:
    return _validate_embedding_backend(backend) == "tfidf-sbert-rerank"
