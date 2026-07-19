# Candidate-ID Prompt Controlled Rerank Result

## Purpose

Run a small controlled experiment on 10 development-run Top-5 miss cases using the newer
candidate_id-only LLM rerank prompt. The goal is to check whether fallback
decreases and whether Top-k localization improves.

## Runs

| Run | Output directory | Meaning |
|---|---|---|
| Sandbox run | `reports/fault_localization/swebench_lite_llm_candidate_id_prompt_top5_miss_10` | Python could not connect to local Ollama. This run is not a valid LLM rerank result. |
| Real Ollama run | `reports/fault_localization/swebench_lite_llm_candidate_id_prompt_top5_miss_10_ollama` | Re-run with local Ollama access allowed. This is the run to cite. |

## Real Ollama Result

| Metric | Value |
|---|---:|
| Selected Top-5 miss tickets | 10 |
| Pipeline failures | 0 |
| LLM rerank accepted | 0 |
| LLM rerank fallback | 10 |
| Top-1 | 0.7000 |
| Top-3 | 0.9000 |
| Top-5 | 0.9000 |
| MRR | 0.7833 |

Important interpretation: the Top-k metrics improved compared with the selected
baseline misses, but the prediction rows all show `method.llm_rerank=false`.
Therefore the improvement should be attributed to the rerun retrieval/candidate
pool behavior, not to successful LLM reranking.

## Fallback Comparison

| Experiment | Rows | Schema invalid | Rank/Candidate ID invalid | Timeout | Service unavailable | Total fallback |
|---|---:|---:|---:|---:|---:|---:|
| Previous scoreguard prompt | 10 | 6 | 0 | 1 | 0 | 7 |
| Candidate-ID prompt, sandbox-blocked | 10 | 0 | 0 | 0 | 10 | 10 |
| Candidate-ID prompt, real Ollama | 10 | 7 | 2 | 1 | 0 | 10 |

## Interpretation

The candidate_id-only prompt successfully removed the sandbox/service issue
when rerun with local Ollama access, but it did not reduce model-output fallback
on this subset. Code Llama still often returned no candidate-ranking JSON, or
returned IDs that did not pass the strict candidate_id validator.

This is useful evidence for the report: LLM rerank remains optional and
experimental. The system's retrieval-first fallback is working, but the LLM
reranker is not yet stable enough to claim as a formal accuracy improvement.

## Next Practical Fix

Before another LLM run, add invalid-response logging for failed rerank attempts
and reduce the LLM candidate pool from Top-10 to Top-5. Without raw invalid
responses, it is difficult to know whether Code Llama is returning prose,
malformed JSON, wrong candidate IDs, or truncated output.
