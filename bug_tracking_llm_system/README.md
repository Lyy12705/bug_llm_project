# Integrated Bug Tracking LLM System

This directory implements the integration layer described in
`../bug_tracking_llm_system_code_flow_plan.md`.

The existing research subprojects remain reusable:

- `../ToJson`: ticket-to-JSON extraction experiments.
- `../bug-duplicate-detection`: duplicate ranking experiments.
- `../bug-priority-drone`: DRONE/GRAY priority prediction experiments.

This package adds the missing orchestration contract: every module exposes the
class interface from the plan, returns stable JSON, and can be called by a single
`PipelineOrchestrator`.

## Pipeline

```text
raw ticket
  -> TicketExtractor
  -> DuplicateDetector
  -> PriorityClassifier
  -> AssigneeTriager
  -> BugLocalizer
  -> PatchGenerator
  -> TestGenerator
  -> RegressionTester
  -> CommitMessageGenerator
  -> final_pipeline_result.json
```

The default implementation is intentionally runnable without network access or
large model downloads. It uses deterministic baselines and explicit fallback
statuses so integration can proceed safely:

- duplicate detection uses a lightweight text similarity baseline;
- priority classification uses DRONE/GRAY-style factor scores;
- assignee triage uses the Hybrid triager with historical BMO train/history data
  when available, plus component-owner mapping as a fallback;
- bug localization uses code chunk indexing plus embedding-style retrieval, with
  stack trace/path boosts and optional LLM reranking;
- patch generation accepts a provided unified diff or reports
  `needs_manual_patch`;
- regression testing validates patches in a temporary copy of the repository.

LLM and trained-model adapters can replace the default modules later without
changing the orchestrator interface.

## Run

From this directory:

```bash
PYTHONPATH=src python3 src/main.py \
  --raw-ticket data/raw_tickets/raw_ticket.example.json \
  --repo-path ../bug-duplicate-detection \
  --fault-top-k 5 \
  --fault-embedding-backend tfidf \
  --output final_pipeline_result.json
```

By default, the integrated pipeline uses the paper-grade BMO assignee history if
this file exists:

```text
assignee_triage_accuracy/paper_grade/data/processed/bmo_paper_2024_3k_history_train.jsonl
```

For another project or a private bug tracker export, pass a compatible JSONL file:

```bash
PYTHONPATH=src python3 src/main.py \
  --raw-ticket data/raw_tickets/raw_ticket.example.json \
  --repo-path ../bug-duplicate-detection \
  --assignee-dataset path/to/assignee_history.jsonl \
  --output final_pipeline_result.json
```

Each assignee-history row should include at least `assignee`, `component`, and
`title`; `product`, `description`, `severity`, and `priority` are used when
present.

Step-level JSON checkpoints are written under:

```text
data/processed_tickets/<ticket_id>/
```

Patch validation reports `patch_applied_unverified`, `tests_passed`, or
`completed_verified`. The final status is only `completed_verified` when an
executable reproduction test fails before the patch, passes after it, and the
configured regression suite also passes. Ticket-supplied commands are blocked
by default; use `--allow-ticket-test-commands` only for trusted ticket inputs
and repositories. Each command is limited by `--test-timeout` (300 seconds by
default).

Fault localization consumes the structured ticket JSON produced by
`TicketExtractor`. The most useful fields are `title`, `description`/`body`/
`bug_report`, `component`, `product`, `bug_type`, `error_message`, `logs` or
`stack_trace`, and reproduction fields. Use `--fault-code-index` to reuse a
prebuilt index, `--fault-top-k` to change candidate count, `--fault-embedding-backend`
to switch between `tfidf`, `auto`, and `sbert`, and `--fault-llm-rerank` to add
an optional Ollama/Code Llama reranking step.

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run a lightweight full-system health check:

```bash
PYTHONPATH=src python3 scripts/health_check.py
```

Audit generated repository/index/embedding caches without deleting anything:

```bash
python3 scripts/manage_artifact_cache.py --max-age-days 30 --max-total-gb 8
```

Review the dry-run JSON before adding `--apply`. The command is restricted to
cache roots inside this project.

The health check runs only local smoke tests. It verifies the integrated
pipeline, retrieval-first fault localization, pollable localization job,
duplicate detector, and priority classifier using source-controlled fixtures.
It intentionally does not call Ollama, download embedding models, rerun the full
SWE-bench Lite experiment, or retrain classifiers. The latest JSON report is
written to:

```text
reports/health_check/health_check_latest.json
```

## Fault Localization Baseline

The fault-localization implementation is intentionally retrieval-first. It does
not ask an LLM to read an entire repository or guess a file from memory. Instead,
it scans a repository into code chunks, keeps metadata such as `file_path`,
`function_name`, `class_name`, `symbol_name`, `start_line`, `end_line`, and
`code_text`, retrieves a compact set of file/function/class candidates, and only
then optionally reranks those candidates with an instruction-tuned code LLM.

The default backend is a local TF-IDF vector baseline so it works without model
downloads. If `sentence-transformers` and a cached model are available, use
`--embedding-backend sbert` for full SBERT-style semantic retrieval. For larger
repositories, prefer `--embedding-backend tfidf-sbert-rerank`: it uses TF-IDF to
build a compact candidate pool, then applies SBERT only to that pool. Use
`--sbert-cache-dir data/fault_localization/swebench_lite/embedding_cache` to
persist SBERT embeddings across tickets and reruns. This keeps the method usable
on real repositories where sending all source code to an LLM would be too slow,
too expensive, and too unreliable.

The current best development configuration is the domain-aware
`tfidf-sbert-rerank` pipeline with file-level aggregation. On the full 300-ticket
SWE-bench Lite split, it achieved development file-level Top-1 `0.5500`, Top-3
`0.7567`, Top-5 `0.8133`, and MRR `0.6559`. The same split was used for earlier
30/100-ticket method selection, so these are not untouched final-test numbers.
They should be presented as file-level development results; symbol-level ground
truth is not available in the prepared split yet.

Create a repository-disjoint frozen holdout before further tuning:

```bash
PYTHONPATH=src python3 scripts/create_fault_localization_frozen_split.py \
  --tickets data/fault_localization/swebench_lite/test_tickets.jsonl \
  --gold data/fault_localization/swebench_lite/test_gold.jsonl \
  --output-dir data/fault_localization/swebench_lite/frozen_protocol_v1 \
  --prior-exposure previously-evaluated
```

Because the current method was already evaluated on all 300 rows, this
repository-disjoint split is a prospective guardrail for future changes, not an
untouched final test. Use only `development_*` during further feature and
threshold work. A paper-grade final claim still requires newly collected,
time-separated or repository-separated tickets that have never appeared in
method selection or prior aggregate results.

Optional LLM reranking is implemented as a small, practical second-stage
reranker. It does not replace retrieval and does not send a full repository to
the model. When `--llm-rerank` is enabled, the system first retrieves a limited
candidate pool such as Top-10 or Top-20 files, sends only those candidates to
the LLM, caches the JSON response, and falls back to retrieval ranking if the
LLM call fails or returns invalid JSON. LLM scores are blended conservatively
with retrieval scores so a weak or poorly formatted LLM response cannot easily
override strong retrieval evidence.

Each candidate includes explainable scoring fields:

- `embedding_score`
- `stack_trace_score`
- `component_score`
- `keyword_score`
- `symbol_score`
- `final_score`

Build a reusable code index:

```bash
PYTHONPATH=src python3 scripts/build_code_index.py \
  --repo-path ../bug-duplicate-detection \
  --output data/code_index/bug_duplicate_detection.json
```

Localize a single ticket:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --ticket data/raw_tickets/raw_ticket.example.json \
  --code-index data/code_index/bug_duplicate_detection.json \
  --output data/processed_tickets/RAW-001/fault_localization_result.json \
  --embedding-backend tfidf-sbert-rerank \
  --sbert-cache-dir data/fault_localization/embedding_cache \
  --top-k 5
```

Run directly without a prebuilt index:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --ticket data/raw_tickets/raw_ticket.example.json \
  --repo-path ../bug-duplicate-detection
```

Run JSONL batch localization:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --tickets-jsonl data/historical_tickets.jsonl \
  --repo-path ../bug-duplicate-detection \
  --output data/evaluation_results/fault_localization_predictions.jsonl
```

Optional Code Llama/Ollama reranking:

```bash
PYTHONPATH=src python3 scripts/fault_localization.py \
  --ticket data/raw_tickets/raw_ticket.example.json \
  --repo-path ../bug-duplicate-detection \
  --llm-rerank \
  --ollama-model codellama:7b-instruct
```

Evaluate predictions when ground truth fixed files are available:

```bash
PYTHONPATH=src python3 scripts/prepare_fault_localization_gold.py \
  --input path/to/bug_fix_records.jsonl \
  --output data/evaluation_results/fault_localization_gold.jsonl \
  --skip-empty

PYTHONPATH=src python3 scripts/evaluate_fault_localization.py \
  --gold data/evaluation_results/fault_localization_gold.jsonl \
  --pred data/evaluation_results/fault_localization_predictions.jsonl \
  --output data/evaluation_results/fault_localization_metrics.json
```

Gold rows can use fields such as `fixed_files`, `modified_files`,
`changed_files`, `file_path`, or `file` for file-level evaluation, plus
`fixed_symbols`, `function_name`, `class_name`, or `symbol_qualified_name` for
symbol-level evaluation. The evaluator reports file-level and symbol-level
Top-1, Top-3, Top-5 accuracy and MRR. If the dataset does not yet include
bug-fixing files or commit-derived changed files, rows without ground truth are
skipped and the metrics output explains what needs to be added.

### Public Evaluation Dataset

This project now includes a legal public dataset preparation path for
SWE-bench Lite. SWE-bench Lite contains real GitHub issues and developer patches
from open-source Python repositories. The prepared fault-localization view uses:

- `problem_statement` as the bug report text.
- files changed in the developer `patch` as file-level ground truth.
- `repo` and `base_commit` as the source snapshot to check out before running
  localization.

Prepare the dataset:

```bash
PYTHONPATH=src python3 scripts/prepare_swebench_lite_fault_localization.py \
  --split test \
  --output-dir data/fault_localization/swebench_lite
```

Generated files:

```text
data/fault_localization/swebench_lite/test.parquet
data/fault_localization/swebench_lite/test_tickets.jsonl
data/fault_localization/swebench_lite/test_gold.jsonl
data/fault_localization/swebench_lite/test_repos.jsonl
data/fault_localization/swebench_lite/test_manifest.json
```

The prepared `test_gold.jsonl` can be passed directly to
`scripts/evaluate_fault_localization.py` after you produce predictions. Because
each SWE-bench Lite instance may have a different `repo` and `base_commit`, run
localization against the matching checked-out repository snapshot for each
ticket before aggregating predictions.

Example for one instance:

```bash
# Inspect the first prepared ticket and gold row.
python3 -m json.tool data/fault_localization/swebench_lite/test_manifest.json
head -n 1 data/fault_localization/swebench_lite/test_tickets.jsonl
head -n 1 data/fault_localization/swebench_lite/test_gold.jsonl

# Then clone the referenced repo and checkout the row's base_commit.
# Example from the first SWE-bench Lite test row:
git clone https://github.com/astropy/astropy data/fault_localization/swebench_lite/repos/astropy__astropy
cd data/fault_localization/swebench_lite/repos/astropy__astropy
git checkout d16bfe05a744909de4b27f5875fe0d4ed41ce607
```

After checkout, build a code index for that repo snapshot, run localization for
the matching ticket, append the prediction to a JSONL file, and evaluate against
`test_gold.jsonl`. The prepared data gives legal file-level ground truth; it does
not include function/class-level ground truth unless you add symbol annotations.

For repeatable subset or full-split evaluation, use the batch runner. It reads
the prepared tickets/gold files, resolves each row's repository snapshot, writes
predictions, computes Top-k/MRR metrics, and saves a small demo-cases JSON file:

```bash
PYTHONPATH=src python3 scripts/run_swebench_lite_fault_localization.py \
  --dataset-dir data/fault_localization/swebench_lite \
  --split test \
  --repo-cache-dir data/fault_localization/swebench_lite/repos \
  --output-dir reports/fault_localization/swebench_lite \
  --embedding-backend tfidf-sbert-rerank \
  --sbert-cache-dir data/fault_localization/swebench_lite/embedding_cache \
  --resume \
  --progress-every 25 \
  --limit 30 \
  --top-k 5
```

If the repositories are not cached yet, add `--clone-missing`. For an already
checked-out local snapshot or a small demo row, add `local_repo_path` to the
ticket JSONL and pass `--no-checkout`. For 100/300-ticket experiments, keep
`--resume` enabled so interrupted runs continue from existing successful
predictions instead of restarting from the first row.

Run a small cached LLM rerank experiment on top of the stable retrieval baseline:

```bash
PYTHONPATH=src python3 scripts/run_swebench_lite_fault_localization.py \
  --dataset-dir data/fault_localization/swebench_lite \
  --split test \
  --repo-cache-dir data/fault_localization/swebench_lite/repos \
  --index-cache-dir data/fault_localization/swebench_lite/indexes \
  --output-dir reports/fault_localization/swebench_lite_llm_rerank_30 \
  --embedding-backend tfidf-sbert-rerank \
  --sbert-cache-dir data/fault_localization/swebench_lite/embedding_cache \
  --llm-rerank \
  --llm-candidate-k 10 \
  --llm-cache-dir reports/fault_localization/llm_rerank_cache \
  --ollama-model codellama:7b-instruct \
  --ollama-timeout 180 \
  --resume \
  --progress-every 1 \
  --checkpoint-every 1 \
  --limit 30 \
  --top-k 5
```

Use this mode for small comparison runs first. The retrieval-only 300-ticket
result remains the stable baseline.

For a more controlled LLM comparison, run the reranker only on Top-5 misses from
the retrieval baseline instead of reranking the full 300-ticket split:

```bash
PYTHONPATH=src python3 scripts/run_controlled_llm_rerank_subset.py \
  --baseline-pred reports/fault_localization/swebench_lite_tfidf_sbert_domain_rerank_300/test_predictions.jsonl \
  --dataset-dir data/fault_localization/swebench_lite \
  --output-dir reports/fault_localization/swebench_lite_llm_rerank_controlled_top5_miss_10 \
  --subset-size 10 \
  --llm-candidate-k 10 \
  --ollama-model codellama:7b-instruct \
  --resume \
  --progress-every 1 \
  --checkpoint-every 1 \
  --top-k 5
```

The evaluator now also includes a `per_repo` section in the metrics JSON, so a
presentation can show repository-level results such as Django, SymPy, pytest,
and matplotlib rather than only the aggregate score. For a stable live demo that
does not require cloning repositories or rerunning the full split, use:
`reports/fault_localization/demo_commands.md`.

## Demo Interface

The `demo/` directory provides a local web interface for presenting this project
as an AI assistant that can be attached to an existing bug tracker.

From the repository root:

```bash
python3 bug_tracking_llm_system/demo/demo_app.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The web demo currently covers duplicate review, priority, and assignee triage:

- a duplicate-ticket case that stops after duplicate recommendation;
- a non-duplicate case that continues through priority and assignee triage.

Fault localization and patch-handoff demonstrations are separate CLI workflows;
see `reports/fault_localization/demo_commands.md` and
`demo/data/fault_localization_demo_commands.md`.

## Module Files

The implementation follows the plan's paths:

- `src/modules/ticket_extractor.py`
- `src/modules/duplicate_detector.py`
- `src/modules/priority_classifier.py`
- `src/modules/assignee_triager.py`
- `src/modules/bug_localizer.py`
- `src/modules/patch_generator.py`
- `src/modules/test_generator.py`
- `src/modules/regression_tester.py`
- `src/modules/commit_message_generator.py`
- `src/pipeline/orchestrator.py`
- `src/config.py`
- `src/utils/fault_localization.py`
- `scripts/build_code_index.py`
- `scripts/fault_localization.py`
- `scripts/evaluate_fault_localization.py`
- `scripts/prepare_fault_localization_gold.py`
- `scripts/prepare_swebench_lite_fault_localization.py`
- `scripts/run_swebench_lite_fault_localization.py`

The `experiments/evaluate_*.py` scripts provide lightweight JSON-level metrics
for each module output, so pipeline checkpoints can be evaluated without
rerunning the model experiments.

## Replacing Baselines With Models

Keep each replacement behind the same method names:

- `TicketExtractor.extract(raw_ticket) -> structured_ticket`
- `DuplicateDetector.detect(ticket_json, historical_db=None) -> duplicate_result`
- `PriorityClassifier.predict(ticket_json, duplicate_candidates) -> priority_result`
- `AssigneeTriager.assign(ticket_json, priority_result) -> assignee_result`
- `BugLocalizer.localize(ticket_json, repo_path) -> bug_location_result`
- `PatchGenerator.generate(ticket_json, bug_location, repo_path) -> patch_result`
- `TestGenerator.generate_tests(ticket_json, patch) -> generated_tests_result`
- `RegressionTester.run(patch, repo_path, reproduction_tests) -> regression_test_result`
- `CommitMessageGenerator.generate(ticket_json, patch, test_result) -> commit_message_result`
