# Fault Localization Method Comparison

## Dataset

- Dataset: SWE-bench Lite `test` split
- Development subset sizes: 30 and 100 tickets
- Full-split development benchmark: 300 tickets
- Ground truth: file-level fixed files parsed from developer patches
- Symbol-level ground truth: not available in this prepared split

## Compared Methods

| Method | Description |
|---|---|
| Original TF-IDF | Initial chunk retrieval baseline |
| Index fix | TF-IDF after fixing source index coverage for nested `models/` packages |
| File aggregation | Index fix plus one ranked file candidate per `file_path` |
| TF-IDF + SBERT rerank | TF-IDF candidate pool plus SBERT reranking and file aggregation |
| Domain-aware TF-IDF + SBERT rerank | Adds identifier, path, wrapper-frame, and framework-domain ranking signals |

## Results

| Method | Top-1 | Top-3 | Top-5 | MRR | Misses |
|---|---:|---:|---:|---:|---:|
| Original TF-IDF | 0.4000 | 0.4667 | 0.4667 | 0.4278 | 16 |
| Index fix | 0.6000 | 0.6667 | 0.6667 | 0.6278 | 10 |
| File aggregation | 0.6000 | 0.6667 | 0.7000 | 0.6361 | 9 |
| TF-IDF + SBERT rerank | 0.6667 | 0.8000 | 0.8333 | 0.7361 | 5 |

## 100-Ticket Validation

The best method was rerun on the first 100 SWE-bench Lite test tickets with a
persistent SBERT embedding cache.

| Method | Tickets | Top-1 | Top-3 | Top-5 | MRR | Misses | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + SBERT rerank + file aggregation + cache | 100 | 0.6000 | 0.8000 | 0.8700 | 0.7127 | 13 | 0 |
| Domain-aware TF-IDF + SBERT rerank + file aggregation + cache | 100 | 0.6600 | 0.8600 | 0.9400 | 0.7702 | 6 | 0 |

The first 100-ticket run produced stable Top-3/Top-5 behavior on a larger
subset. The lower Top-1 score compared with the 30-ticket subset indicates that
the larger sample contains harder first-rank ordering cases.

The domain-aware ranking layer is now the best 100-ticket result. It improves
Top-1 by 6 percentage points and Top-5 by 7 percentage points over the previous
100-ticket best run. Remaining misses are still indexed, so the next improvement
should focus on deeper reranking rather than source indexing.

## 300-Ticket Full-Split Development Evaluation

The domain-aware TF-IDF + SBERT rerank method was then run on all 300
SWE-bench Lite test tickets with repository/index/embedding caches and resumable
checkpointing enabled.

| Method | Tickets | Top-1 | Top-3 | Top-5 | MRR | Misses | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Domain-aware TF-IDF + SBERT rerank + file aggregation + cache | 300 | 0.5500 | 0.7567 | 0.8133 | 0.6559 | 56 | 0 |

This 300-ticket result is the strongest development benchmark so far. Because
the first 30/100 rows of the same public `test` split were used for method
selection, it is not an untouched final-test estimate and must be labeled as a
development result. It is lower than the first 100-ticket subset because the full split includes harder repos
and issue reports, especially ranking-heavy cases in SymPy, pytest, Sphinx, and
Pylint. The important practical observation is that the run completed on real
multi-repository data with no failed tickets after adding checkpoint/resume,
warning suppression, and AST recursion fallback.

## Interpretation

The largest gain came from fixing code-index coverage: valid Django source files
under nested `django/db/models/` paths were no longer excluded. File-level
aggregation then improved Top-5 by preventing repeated chunks from the same file
from crowding out other files.

The practical design is retrieval-first. Full SBERT retrieval over all chunks
was too slow for repeated evaluation on large repositories, and asking an LLM to
read an entire repository is not realistic. The implemented backend uses TF-IDF
to build a compact candidate pool and applies SBERT only to that pool. The
domain-aware ranking layer then adds exact identifier, path, wrapper-frame, and
framework-domain signals. On the 300-ticket full split, the resulting Top-5
accuracy is `0.8133`.

## LLM Rerank Note

A 30-ticket Code Llama rerank run was attempted on top of the TF-IDF + SBERT
candidate pool, but synchronous local reranking was too slow for this evaluation
run and was interrupted. LLM reranking should remain optional and should only be
used on a small Top-10 or Top-20 candidate set with timeout, response cache,
checkpointed partial output, and retrieval fallback.

The implementation now supports that practical setup: `--llm-candidate-k`
controls the Top-10/Top-20 candidate pool, `--llm-cache-dir` stores JSONL rerank
responses, `--checkpoint-every` protects long batch runs, and invalid or failed
LLM responses fall back to the retrieval ranking. This keeps LLM reranking as a
bounded comparison layer rather than a dependency for the core system. LLM
scores are blended with retrieval scores so local model mistakes cannot easily
override strong retrieval evidence.

## Remaining Errors

After domain-aware TF-IDF + SBERT rerank on the 300-ticket full split, 56
tickets remain missed in Top-5. Failure analysis classified all 56 misses as
`ranking_error_gold_indexed`: the gold files were present in the code index, but
the ranker did not place them in the Top-5. The next accuracy work should focus
on deeper candidate reranking rather than source indexing.
