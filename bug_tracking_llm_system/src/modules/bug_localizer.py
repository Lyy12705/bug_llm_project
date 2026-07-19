from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.fault_localization import CODE_SUFFIXES, CodeIndex, build_code_index, load_code_index, localize_ticket


class BugLocalizer:
    """Locate likely faulty files/functions from ticket text and repository code.

    The default implementation is a runnable retrieval baseline: it builds code
    chunks from the repository, ranks them against the bug report with vector
    similarity, and optionally leaves room for an LLM reranker.
    """

    def __init__(
        self,
        *,
        code_index_path: str | Path | None = None,
        top_k: int = 5,
        embedding_backend: str = "tfidf",
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_local_files_only: bool = True,
        sbert_cache_dir: str | Path | None = None,
        llm_client: Any | None = None,
        llm_rerank: bool = False,
        llm_candidate_k: int | None = None,
        llm_cache_dir: str | Path | None = None,
    ) -> None:
        self.code_index_path = Path(code_index_path) if code_index_path else None
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self.sbert_model = sbert_model
        self.sbert_local_files_only = sbert_local_files_only
        self.sbert_cache_dir = Path(sbert_cache_dir) if sbert_cache_dir else None
        self.llm_client = llm_client
        self.llm_rerank = llm_rerank
        self.llm_candidate_k = llm_candidate_k
        self.llm_cache_dir = Path(llm_cache_dir) if llm_cache_dir else None
        self._cached_index_key: tuple[str, str] | None = None
        self._cached_index: CodeIndex | None = None

    def localize(self, ticket_json: dict[str, Any], repo_path: str) -> dict[str, Any]:
        code_index: CodeIndex | None = None
        if self.code_index_path is not None and self.code_index_path.exists():
            key = (str(self.code_index_path.resolve()), str(self.code_index_path.stat().st_mtime_ns))
            if self._cached_index_key != key:
                self._cached_index = load_code_index(self.code_index_path)
                self._cached_index_key = key
            code_index = self._cached_index
        else:
            root = Path(repo_path).resolve()
            key = (str(root), _source_tree_token(root))
            if self._cached_index_key != key:
                self._cached_index = build_code_index(root)
                self._cached_index_key = key
            code_index = self._cached_index

        result = localize_ticket(
            ticket_json,
            repo_path=repo_path if code_index is None else None,
            code_index=code_index,
            top_k=self.top_k,
            embedding_backend=self.embedding_backend,
            sbert_model=self.sbert_model,
            sbert_local_files_only=self.sbert_local_files_only,
            sbert_cache_dir=self.sbert_cache_dir,
            llm_client=self.llm_client,
            llm_rerank=self.llm_rerank,
            llm_candidate_k=self.llm_candidate_k,
            llm_cache_dir=self.llm_cache_dir,
        )
        if not result.get("localized_candidates"):
            raise ValueError("No bug localization candidates found in the repository code index.")
        if not result.get("repository_path"):
            result["repository_path"] = str(Path(repo_path).resolve())
        return result


def _source_tree_token(root: Path) -> str:
    count = 0
    total_size = 0
    latest_mtime_ns = 0
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv"}
    for path in root.rglob("*"):
        if any(part in ignored for part in path.parts) or path.is_symlink():
            continue
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
            stat = resolved.stat()
        except (OSError, ValueError):
            continue
        count += 1
        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return f"count={count}:size={total_size}:mtime={latest_mtime_ns}"
