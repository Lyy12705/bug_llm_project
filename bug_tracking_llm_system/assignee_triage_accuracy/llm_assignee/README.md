# LLM Assignee Experiments

This folder keeps additive experiments for assignee triage. The production
`current_assignee_triager` remains unchanged and is treated as the strongest
baseline.

## Goals

- Export the frozen paper-grade split into conversational JSONL for SFT.
- Evaluate LLM-based assignee prediction on the same train/validation/test
  split, roster, and metrics as the deterministic baselines.
- Test a safer hybrid design where `current_assignee_triager` proposes Top-K
  candidates and an LLM reranks only those candidates.

## Important Guardrails

- Do not use validation/test assignees to build the roster.
- Do not put the gold assignee into inference prompts.
- Do not let the LLM invent assignees. Reranker output is constrained to the
  provided candidate IDs and falls back to the original ranking on invalid
  output.
- Do not replace `current_assignee_triager` unless a held-out test comparison
  shows a clear improvement.

## Typical Flow

Prepare conversational JSONL:

```bash
python3 bug_tracking_llm_system/assignee_triage_accuracy/scripts/prepare_llm_assignee_data.py \
  --dataset bmo_paper_2024_3k
```

Run the hybrid reranker on validation first:

```bash
PYTHONPATH=bug_tracking_llm_system/src python3 \
  bug_tracking_llm_system/assignee_triage_accuracy/scripts/run_assignee_llm_reranker.py \
  --dataset bmo_paper_2024_3k \
  --split validation \
  --model llama3.1:8b \
  --candidate-k 10
```

Only run the held-out test split after the validation design is frozen.

