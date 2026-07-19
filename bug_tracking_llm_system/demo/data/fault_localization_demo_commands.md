# Fault Localization Demo Commands

Run localization only:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --ticket demo/data/fault_localization_demo_ticket.json \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --output reports/fault_localization/demo_stage5_fault_localization_result.json
```

Run the beta localization demo cases:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --tickets-jsonl demo/data/fault_localization_demo_cases.jsonl \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --output reports/fault_localization/demo_stage5_beta_case_localization.jsonl
```

Run the beta user-facing localization demo cases:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --tickets-jsonl demo/data/fault_localization_demo_cases.jsonl \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --output-format user-facing \
  --index-cache-dir reports/fault_localization/index_cache \
  --progress text \
  --progress-file reports/fault_localization/demo_progress.jsonl \
  --output reports/fault_localization/demo_user_facing_localization.jsonl
```

Run the integrated pipeline:

```bash
PYTHONPATH=src python3 src/main.py \
  --raw-ticket demo/data/fault_localization_demo_ticket.json \
  --repo-path demo/fault_localization_auth_repo \
  --fault-top-k 3 \
  --fault-embedding-backend tfidf \
  --output reports/fault_localization/demo_stage5_pipeline_result.json \
  --no-checkpoints
```

Run the internal beta readiness check:

```bash
PYTHONPATH=src python3 scripts/run_fault_localization_internal_beta.py \
  --output-dir reports/fault_localization/internal_beta_20
```

The readiness check runs the local demo cases plus 20 retrieval-first controlled
cases by default. It does not call Ollama unless the fault localization CLI is
run separately with `--llm-rerank`.

Start a pollable job for API/UI-style demos:

```bash
PYTHONPATH=src python3 scripts/fault_localization_job.py start \
  --ticket demo/data/fault_localization_demo_ticket.json \
  --repo-path demo/fault_localization_auth_repo \
  --top-k 3 \
  --embedding-backend tfidf \
  --jobs-dir reports/fault_localization/jobs \
  --index-cache-dir reports/fault_localization/index_cache
```

Check job status:

```bash
PYTHONPATH=src python3 scripts/fault_localization_job.py status \
  --job-dir reports/fault_localization/jobs/<job_id>
```
