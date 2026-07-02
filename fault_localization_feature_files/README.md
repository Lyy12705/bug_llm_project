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

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Fault Localization Baseline

The first fault-localization implementation follows the literature-friendly
two-stage shape used by modern LLM bug-fixing systems: retrieve a compact set of
file/function/class candidates first, then optionally rerank those candidates
with an instruction-tuned code LLM. It scans a repository into code chunks, keeps
metadata such as `file_path`, `function_name`, `class_name`, `symbol_name`,
`start_line`, `end_line`, and `code_text`, then ranks chunks against the ticket
text. The default backend is a local TF-IDF vector baseline so it works without
downloads. If `sentence-transformers` and a cached model are available, use
`--embedding-backend sbert` for SBERT-style semantic retrieval.

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

The demo includes two cases:

- a duplicate-ticket case that stops after duplicate recommendation;
- a non-duplicate case that continues through priority, assignee, localization,
  patch, tests, regression, and commit message.

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
- `RegressionTester.run(patch, repo_path) -> regression_test_result`
- `CommitMessageGenerator.generate(ticket_json, patch, test_result) -> commit_message_result`
