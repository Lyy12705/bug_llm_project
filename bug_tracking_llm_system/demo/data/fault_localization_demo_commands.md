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
