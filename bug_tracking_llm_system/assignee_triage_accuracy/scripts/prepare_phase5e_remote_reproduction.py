from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assignee_phase1_common import EVAL_ROOT, file_sha256, normalize_assignee, read_jsonl, write_json, write_jsonl


DEFAULT_DATASET = "bmo_paper_2024_3k"
DEFAULT_PHASE5_ROOT = EVAL_ROOT / "phase5_llm_reproduction"
DEFAULT_PHASE5AD_REPORT = DEFAULT_PHASE5_ROOT / "reports" / DEFAULT_DATASET
DEFAULT_PHASE5E_ROOT = DEFAULT_PHASE5_ROOT / "phase5e_prep"
PRIMARY_VARIANT = "sanitized_title_plus_description"
ADAPTED_METADATA_VARIANT = "sanitized_title_plus_description_plus_metadata"
SEED = 3407

ENVIRONMENT_PINS = {
    "python": "3.10.14",
    "cuda": "12.1",
    "pytorch": "2.4.1+cu121",
    "transformers": "4.46.3",
    "peft": "0.13.2",
    "trl": "0.12.2",
    "bitsandbytes": "0.44.1",
    "accelerate": "1.0.1",
    "datasets": "3.1.0",
    "safetensors": "0.4.5",
    "tokenizers": "0.20.3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 5E remote CUDA QLoRA reproduction package.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--phase5-report-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-train-size", type=int, default=64)
    parser.add_argument("--smoke-validation-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    phase5_report_dir = args.phase5_report_dir or DEFAULT_PHASE5_ROOT / "reports" / args.dataset
    output_dir = args.output_dir or DEFAULT_PHASE5E_ROOT / "reports" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(phase5_report_dir / "llm_dataset_manifest.json")
    training_config = read_json(phase5_report_dir / "llm_training_config.json")
    hardware = read_json(phase5_report_dir / "phase5_hardware_feasibility.json")
    primary_train_path = Path(manifest["generated_files"][PRIMARY_VARIANT]["train_jsonl"])
    primary_validation_path = Path(manifest["generated_files"][PRIMARY_VARIANT]["validation_jsonl"])
    train_records = read_jsonl(primary_train_path)
    validation_records = read_jsonl(primary_validation_path)
    test_rows = read_jsonl(Path(manifest["source_paths"]["test"]))
    roster = list(manifest["candidate_roster"]["candidates"])

    write_environment_files(output_dir)
    write_json(output_dir / "phase5e_execution_config.json", build_execution_config(args.dataset, manifest, training_config, output_dir))
    audit_rows, leakage_audit = audit_prompts(phase5_report_dir, manifest)
    write_csv(output_dir / "prompt_integrity_audit.csv", audit_rows, list(audit_rows[0].keys()) if audit_rows else [])
    write_json(output_dir / "label_leakage_audit.json", leakage_audit)
    tokenization_audit = build_tokenization_audit(training_config, hardware)
    write_csv(output_dir / "tokenization_audit.csv", tokenization_audit, list(tokenization_audit[0].keys()))

    smoke_train = select_diverse_subset(train_records, args.smoke_train_size, args.seed)
    smoke_validation = select_diverse_subset(validation_records, args.smoke_validation_size, args.seed + 1)
    write_jsonl(output_dir / "smoke_train.jsonl", smoke_train)
    write_jsonl(output_dir / "smoke_validation.jsonl", smoke_validation)
    smoke_manifest = build_smoke_manifest(
        smoke_train=smoke_train,
        smoke_validation=smoke_validation,
        train_source=primary_train_path,
        validation_source=primary_validation_path,
        seed=args.seed,
        output_dir=output_dir,
    )
    write_json(output_dir / "smoke_manifest.json", smoke_manifest)

    write_smoke_training_placeholders(output_dir, hardware)
    inference_rows, inference_schema, inference_report = build_dry_run_inference_smoke(test_rows, roster, args.seed)
    write_csv(output_dir / "smoke_llm_predictions.csv", inference_rows, list(inference_rows[0].keys()) if inference_rows else [])
    write_json(output_dir / "inference_schema.json", inference_schema)
    (output_dir / "inference_smoke_report.md").write_text(inference_report, encoding="utf-8")

    runtime_estimates = build_runtime_estimates(training_config, train_records)
    write_csv(output_dir / "runtime_cost_estimate.csv", runtime_estimates, list(runtime_estimates[0].keys()))
    write_json(output_dir / "runtime_cost_estimate.json", {"estimates": runtime_estimates, "precision": "rough_estimate"})

    gate = build_full_training_gate(
        leakage_audit=leakage_audit,
        tokenization_audit=tokenization_audit,
        hardware=hardware,
        inference_schema=inference_schema,
    )
    write_json(output_dir / "full_training_gate.json", gate)
    (output_dir / "environment_report.md").write_text(build_environment_report(), encoding="utf-8")
    (output_dir / "remote_training_runbook.md").write_text(build_runbook(args.dataset), encoding="utf-8")

    summary = build_summary(
        output_dir=output_dir,
        manifest=manifest,
        leakage_audit=leakage_audit,
        tokenization_audit=tokenization_audit,
        gate=gate,
        runtime_estimates=runtime_estimates,
    )
    write_json(output_dir / "phase5e_prep_summary.json", summary)
    (output_dir / "phase5e_prep_summary.md").write_text(build_summary_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "output_dir": str(output_dir),
                "prompt_integrity_passed": leakage_audit["passed"],
                "dry_run_inference_rows": len(inference_rows),
                "full_training_gate_passed": gate["passed"],
                "failed_checks": gate["failed_checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_environment_files(output_dir: Path) -> None:
    requirements = f"""--extra-index-url https://download.pytorch.org/whl/cu121
torch=={ENVIRONMENT_PINS['pytorch']}
transformers=={ENVIRONMENT_PINS['transformers']}
peft=={ENVIRONMENT_PINS['peft']}
trl=={ENVIRONMENT_PINS['trl']}
bitsandbytes=={ENVIRONMENT_PINS['bitsandbytes']}
accelerate=={ENVIRONMENT_PINS['accelerate']}
datasets=={ENVIRONMENT_PINS['datasets']}
safetensors=={ENVIRONMENT_PINS['safetensors']}
tokenizers=={ENVIRONMENT_PINS['tokenizers']}
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
tqdm==4.66.6
"""
    env_yml = f"""name: phase5-llm-qlora
channels:
  - nvidia
  - pytorch
  - conda-forge
dependencies:
  - python={ENVIRONMENT_PINS['python']}
  - pip
  - cuda-toolkit={ENVIRONMENT_PINS['cuda']}
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/cu121
      - torch=={ENVIRONMENT_PINS['pytorch']}
      - transformers=={ENVIRONMENT_PINS['transformers']}
      - peft=={ENVIRONMENT_PINS['peft']}
      - trl=={ENVIRONMENT_PINS['trl']}
      - bitsandbytes=={ENVIRONMENT_PINS['bitsandbytes']}
      - accelerate=={ENVIRONMENT_PINS['accelerate']}
      - datasets=={ENVIRONMENT_PINS['datasets']}
      - safetensors=={ENVIRONMENT_PINS['safetensors']}
      - tokenizers=={ENVIRONMENT_PINS['tokenizers']}
      - scikit-learn==1.5.2
      - pandas==2.2.3
      - numpy==1.26.4
      - tqdm==4.66.6
"""
    (output_dir / "requirements_phase5_llm.txt").write_text(requirements, encoding="utf-8")
    (output_dir / "environment_phase5_llm.yml").write_text(env_yml, encoding="utf-8")


def build_execution_config(dataset: str, manifest: dict[str, Any], training_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    primary_paths = manifest["generated_files"][PRIMARY_VARIANT]
    adapted_paths = manifest["generated_files"][ADAPTED_METADATA_VARIANT]
    config = dict(training_config)
    config["experiment_roles"] = {
        "literature_aligned_primary_reproduction": {
            "input_variant": PRIMARY_VARIANT,
            "reason": "Literature core input is title plus description/body; this run excludes benchmark metadata from the prompt.",
            "train_jsonl": primary_paths["train_jsonl"],
            "validation_jsonl": primary_paths["validation_jsonl"],
        },
        "metadata_enriched_adapted_reproduction": {
            "input_variant": ADAPTED_METADATA_VARIANT,
            "reason": "Adds product/component/priority/severity/status metadata; useful controlled adaptation but not exact literature input.",
            "train_jsonl": adapted_paths["train_jsonl"],
            "validation_jsonl": adapted_paths["validation_jsonl"],
        },
    }
    config["sft"] = dict(config["sft"])
    config["sft"]["primary_input_variant"] = PRIMARY_VARIANT
    config["sft"]["train_jsonl"] = str(output_dir / "smoke_train.jsonl")
    config["sft"]["validation_jsonl"] = str(output_dir / "smoke_validation.jsonl")
    config["reproduction_label"] = "literature_aligned_smoke_gate_not_exact_completed"
    config["metadata_variant_policy"] = "Report metadata-inclusive results only as adapted reproduction."
    return config


def audit_prompts(phase5_report_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roster = set(manifest["candidate_roster"]["candidates"])
    train_ticket_ids_by_variant: dict[str, set[str]] = defaultdict(set)
    test_ticket_ids = {str(row.get("ticket_id")) for row in read_jsonl(Path(manifest["source_paths"]["test"]))}
    test_labels = {normalize_assignee(row.get("assignee")) for row in read_jsonl(Path(manifest["source_paths"]["test"]))}
    train_labels = set(manifest["leakage_guard"]["long_tail_train_bucket_counts"])  # sentinel to catch manifest shape changes
    rows = []
    errors = []
    warnings = []
    totals = Counter()
    for variant, paths in manifest["generated_files"].items():
        for split_name, path_key in (("train", "train_jsonl"), ("validation", "validation_jsonl")):
            records = read_jsonl(Path(paths[path_key]))
            for record in records:
                row = audit_record(record, variant=variant, split_name=split_name)
                rows.append(row)
                totals[row["status"]] += 1
                if row["status"] == "fail":
                    errors.append(row["failure_reason"])
                if split_name == "train":
                    train_ticket_ids_by_variant[variant].add(str(record.get("ticket_id")))
                    if str(record.get("ticket_id")) in test_ticket_ids:
                        errors.append(f"test ticket appears in train JSONL: {record.get('ticket_id')}")
    roster_has_test_only = sorted(test_labels - roster)
    if roster_has_test_only:
        errors.append(f"candidate roster contains test-only labels: {roster_has_test_only[:10]}")
    if "freq_20_plus" not in train_labels:
        warnings.append("manifest long-tail summary shape changed; verify train label frequency metadata manually")
    leakage_audit = {
        "passed": not errors and totals.get("fail", 0) == 0,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "records_audited": len(rows),
        "status_counts": dict(totals),
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "candidate_roster_size": len(roster),
        "candidate_roster_train_derived_only": manifest["leakage_guard"]["candidate_roster_policy"][
            "test_label_not_used_to_expand_roster"
        ],
        "test_ticket_overlap_with_train_by_variant": {
            variant: len(ids & test_ticket_ids) for variant, ids in sorted(train_ticket_ids_by_variant.items())
        },
        "test_only_assignees_in_roster": roster_has_test_only,
        "prompt_policy": {
            "gold_assignee_allowed_locations": ["assistant.content", "metadata.expected_assignee"],
            "gold_assignee_forbidden_locations": ["messages[system].content", "messages[user].content"],
            "prompt_must_end_with": "### Assignee:",
        },
        "source_phase5_report_dir": str(phase5_report_dir),
    }
    return rows, leakage_audit


def audit_record(record: dict[str, Any], *, variant: str, split_name: str) -> dict[str, Any]:
    messages = record.get("messages", [])
    system = messages[0].get("content", "") if len(messages) > 0 and messages[0].get("role") == "system" else ""
    user = messages[1].get("content", "") if len(messages) > 1 and messages[1].get("role") == "user" else ""
    assistant = messages[2].get("content", "") if len(messages) > 2 and messages[2].get("role") == "assistant" else ""
    expected = normalize_assignee(record.get("metadata", {}).get("expected_assignee"))
    failures = []
    if len(messages) != 3:
        failures.append("expected exactly system/user/assistant messages")
    if not system:
        failures.append("missing system message")
    if not user:
        failures.append("missing user message")
    if not assistant:
        failures.append("missing assistant target")
    if expected and assistant != expected:
        failures.append("assistant target does not match metadata expected_assignee")
    if expected and expected in user:
        failures.append("gold assignee appears in user prompt")
    if expected and expected in system:
        failures.append("gold assignee appears in system prompt")
    if not user.rstrip().endswith("### Assignee:"):
        failures.append("user prompt does not end with inference-compatible assignee marker")
    strict_text = str(record.get("strict_sft_text", ""))
    strict_expected_count = strict_text.count(expected) if expected else 0
    if expected and strict_expected_count != 1:
        failures.append("strict_sft_text should contain gold exactly once as target")
    return {
        "split": split_name,
        "input_variant": variant,
        "ticket_id": record.get("ticket_id", ""),
        "expected_assignee": expected,
        "message_count": len(messages),
        "has_system": bool(system),
        "has_user": bool(user),
        "has_assistant": bool(assistant),
        "assistant_matches_expected": assistant == expected,
        "gold_in_system": bool(expected and expected in system),
        "gold_in_user": bool(expected and expected in user),
        "prompt_ends_with_assignee_marker": user.rstrip().endswith("### Assignee:"),
        "strict_target_count": strict_expected_count,
        "status": "fail" if failures else "pass",
        "failure_reason": "; ".join(failures),
    }


def build_tokenization_audit(training_config: dict[str, Any], hardware: dict[str, Any]) -> list[dict[str, Any]]:
    local_can_tokenize = bool(hardware.get("torch", {}).get("cuda_available")) and hardware["packages"]["transformers"]["installed"]
    return [
        {
            "check": "tokenizer_chat_template",
            "status": "deferred_remote" if not local_can_tokenize else "ready_remote_or_local",
            "required": True,
            "finding": "DeepSeek tokenizer must be loaded on remote GPU environment; use tokenizer.apply_chat_template when available.",
            "blocking_for_full_training": not local_can_tokenize,
        },
        {
            "check": "eos_token",
            "status": "configured_in_training_script",
            "required": True,
            "finding": "Training script appends tokenizer.eos_token to formatted examples when available.",
            "blocking_for_full_training": False,
        },
        {
            "check": "pad_token",
            "status": "configured_in_training_script",
            "required": True,
            "finding": "Training script sets pad_token to eos_token when tokenizer has no pad token.",
            "blocking_for_full_training": False,
        },
        {
            "check": "padding_side",
            "status": "configured_in_training_script",
            "required": True,
            "finding": "Training script uses right padding for SFT training.",
            "blocking_for_full_training": False,
        },
        {
            "check": "assistant_only_loss_masking",
            "status": "required_and_configured",
            "required": True,
            "finding": "Training script masks prompt tokens and computes loss only after the '### Assignee:' marker.",
            "blocking_for_full_training": False,
        },
        {
            "check": "max_seq_length",
            "status": "configured",
            "required": True,
            "finding": f"Configured max_seq_length={training_config['sft']['max_seq_length']}; remote tokenizer audit must report truncation.",
            "blocking_for_full_training": False,
        },
    ]


def select_diverse_subset(records: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[normalize_assignee(record.get("metadata", {}).get("expected_assignee"))].append(record)
    for rows in grouped.values():
        rng.shuffle(rows)
    selected = []
    labels = sorted(grouped)
    cursor = 0
    while len(selected) < min(size, len(records)) and labels:
        label = labels[cursor % len(labels)]
        if grouped[label]:
            selected.append(grouped[label].pop())
        labels = [item for item in labels if grouped[item]]
        cursor += 1
    selected.sort(key=lambda row: str(row.get("ticket_id")))
    return selected


def build_smoke_manifest(
    *,
    smoke_train: list[dict[str, Any]],
    smoke_validation: list[dict[str, Any]],
    train_source: Path,
    validation_source: Path,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "input_variant": PRIMARY_VARIANT,
        "experiment_role": "literature_aligned_primary_reproduction",
        "train_source": str(train_source),
        "validation_source": str(validation_source),
        "smoke_train_path": str(output_dir / "smoke_train.jsonl"),
        "smoke_validation_path": str(output_dir / "smoke_validation.jsonl"),
        "smoke_train_rows": len(smoke_train),
        "smoke_validation_rows": len(smoke_validation),
        "smoke_train_unique_assignees": len({row["metadata"]["expected_assignee"] for row in smoke_train}),
        "smoke_validation_unique_assignees": len({row["metadata"]["expected_assignee"] for row in smoke_validation}),
        "smoke_train_ticket_ids": [row.get("ticket_id") for row in smoke_train],
        "smoke_validation_ticket_ids": [row.get("ticket_id") for row in smoke_validation],
        "test_rows_included": False,
        "train_sha256": file_sha256(output_dir / "smoke_train.jsonl") if (output_dir / "smoke_train.jsonl").exists() else "",
        "validation_sha256": file_sha256(output_dir / "smoke_validation.jsonl") if (output_dir / "smoke_validation.jsonl").exists() else "",
    }


def write_smoke_training_placeholders(output_dir: Path, hardware: dict[str, Any]) -> None:
    rows = [
        {
            "step": 0,
            "check": "local_smoke_training",
            "status": "skipped",
            "loss": "",
            "grad_norm": "",
            "reason": "Local environment has no CUDA/NF4 bitsandbytes path; run train_phase5_llm_qlora.py --smoke on remote CUDA.",
        }
    ]
    write_csv(output_dir / "smoke_training_metrics.csv", rows, list(rows[0].keys()))
    report = [
        "# Phase 5E Smoke Training Report",
        "",
        "Smoke training was not executed locally.",
        "",
        f"- CUDA available locally: `{hardware.get('torch', {}).get('cuda_available')}`",
        f"- Local exact 8B QLoRA feasible: `{hardware.get('decision', {}).get('can_exact_8b_qlora_locally')}`",
        "- Required remote command is documented in `remote_training_runbook.md`.",
        "",
        "Full training remains blocked until remote smoke training proves model load, NF4 load, LoRA attach, finite loss, checkpoint save/reload, adapter reload, and validation forward.",
    ]
    (output_dir / "smoke_test_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_dry_run_inference_smoke(
    test_rows: list[dict[str, Any]], roster: list[str], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows = []
    for row in test_rows:
        scored = []
        for candidate in roster:
            value = int(hashlib.sha256(f"{seed}|{row.get('ticket_id')}|{candidate}".encode("utf-8")).hexdigest()[:12], 16)
            score = value / float(0xFFFFFFFFFFFF)
            scored.append((candidate, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        top10 = scored[:10]
        ranked_candidates = [candidate for candidate, _ in top10]
        rows.append(
            {
                "model_name": "dry_run_schema_only_not_llm",
                "inference_mode": "candidate_scoring_schema_smoke",
                "ticket_id": row.get("ticket_id", ""),
                "candidate_roster_size": len(roster),
                "top1": top10[0][0],
                "top3": "|".join(candidate for candidate, _ in top10[:3]),
                "top5": "|".join(candidate for candidate, _ in top10[:5]),
                "top10": "|".join(candidate for candidate, _ in top10),
                "ranked_candidates": "|".join(ranked_candidates),
                "candidate_ranks": json.dumps({candidate: index + 1 for index, candidate in enumerate(ranked_candidates)}, sort_keys=True),
                "candidate_scores": json.dumps({candidate: round(score, 8) for candidate, score in top10}, sort_keys=True),
                "score_type": "deterministic_schema_hash_not_model_logprob",
                "scores_are_model_logprobs": False,
                "topk_unique": len({candidate for candidate, _ in top10}) == len(top10),
                "score_sortable": all(top10[index][1] >= top10[index + 1][1] for index in range(len(top10) - 1)),
                "invalid_output": False,
                "invalid_output_rate_applicable": False,
                "note": "Dry-run validates schema only; not an LLM prediction and not an accuracy result.",
            }
        )
    schema = {
        "status": "schema_smoke_only",
        "rows": len(rows),
        "candidate_roster_size": len(roster),
        "top_k": [1, 3, 5, 10],
        "fields": list(rows[0].keys()) if rows else [],
        "required_for_real_inference": [
            "scores_are_model_logprobs must be true",
            "model_name must identify base/adapted checkpoint",
            "invalid_output_rate must be computed for free/post-hoc modes",
            "candidate roster must remain train-derived only",
        ],
        "evaluator_compatibility": {
            "has_ticket_id": True,
            "has_ranked_candidates": bool(rows and "ranked_candidates" in rows[0]),
            "has_candidate_scores": True,
            "topk_unique": all(str(row["topk_unique"]) == "True" or row["topk_unique"] is True for row in rows),
            "invalid_output_rate": 0.0,
        },
    }
    report = [
        "# Phase 5F Inference Smoke Report",
        "",
        "This is a schema-only dry run. It does not use a base model or adapter.",
        "",
        f"- Test rows loaded: `{len(test_rows)}`",
        f"- Candidate roster size: `{len(roster)}`",
        f"- Every row has Top-1/3/5/10: `{all(row['top10'] for row in rows)}`",
        f"- Top-K unique: `{all(row['topk_unique'] for row in rows)}`",
        f"- Scores sortable: `{all(row['score_sortable'] for row in rows)}`",
        "",
        "Real Phase 5F must run `infer_phase5_llm.py --mode candidate_scoring` on remote/model-capable hardware and replace dry-run scores with model log-probabilities.",
    ]
    return rows, schema, "\n".join(report) + "\n"


def build_full_training_gate(
    *,
    leakage_audit: dict[str, Any],
    tokenization_audit: list[dict[str, Any]],
    hardware: dict[str, Any],
    inference_schema: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "environment_ok": False,
        "nf4_load_ok": False,
        "lora_attach_ok": False,
        "prompt_integrity_ok": leakage_audit["passed"],
        "label_leakage_ok": leakage_audit["passed"],
        "tokenization_ok": not any(row["blocking_for_full_training"] for row in tokenization_audit),
        "smoke_loss_finite": False,
        "checkpoint_reload_ok": False,
        "adapter_reload_ok": False,
        "inference_smoke_ok": inference_schema["status"] != "schema_smoke_only",
        "evaluator_compatibility_ok": True,
    }
    warnings = [
        "Local dry-run inference validates schema only; it is not model inference.",
        "Remote CUDA smoke training is required before full training.",
    ]
    if not hardware.get("decision", {}).get("can_exact_8b_qlora_locally"):
        warnings.extend(hardware.get("decision", {}).get("blocking_reasons", []))
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "warnings": warnings,
        "policy": "Full training is forbidden until all checks pass on remote CUDA.",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def build_runtime_estimates(training_config: dict[str, Any], train_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch = int(training_config["sft"]["per_device_train_batch_size"])
    grad_accum = int(training_config["sft"]["gradient_accumulation_steps"])
    epochs = int(training_config["sft"]["num_train_epochs"])
    steps_per_epoch = max(1, (len(train_records) + (batch * grad_accum) - 1) // (batch * grad_accum))
    total_steps = steps_per_epoch * epochs
    return [
        estimate_row("16GB consumer GPU", True, "6-10 hours", "medium-high", "batch=1, grad_accum=16, seq_len<=3072 preferred", total_steps),
        estimate_row("24GB consumer GPU", True, "3-6 hours", "medium", "batch=1, grad_accum=8-16, seq_len=4096 feasible", total_steps),
        estimate_row("A100 40GB", True, "1-3 hours", "low", "batch=2-4, grad_accum tuned; recommended for exact reproduction", total_steps),
        estimate_row("A100 80GB", True, "<1-2 hours", "very low", "most stable; useful for multi-variant/window runs", total_steps),
    ]


def estimate_row(
    gpu: str, feasible: bool, runtime: str, oom_risk: str, recommended_config: str, total_steps: int
) -> dict[str, Any]:
    return {
        "gpu": gpu,
        "feasible_estimate": feasible,
        "expected_runtime_estimate": runtime,
        "oom_risk_estimate": oom_risk,
        "recommended_configuration": recommended_config,
        "estimated_train_steps": total_steps,
        "precision": "rough_estimate_not_benchmark",
    }


def build_environment_report() -> str:
    return f"""# Phase 5E Remote Environment Report

This environment is intended for Linux CUDA remote execution, not local Darwin/arm64 exact reproduction.

## Experiment Roles

- Literature-aligned primary reproduction: `{PRIMARY_VARIANT}`
- Metadata-enriched adapted reproduction: `{ADAPTED_METADATA_VARIANT}`
- Exact reproduction completed: `false` until the target training and candidate-scoring inference path actually runs.

## Version Pins

- Python: `{ENVIRONMENT_PINS['python']}`
- Expected CUDA: `{ENVIRONMENT_PINS['cuda']}`
- PyTorch: `{ENVIRONMENT_PINS['pytorch']}`
- transformers: `{ENVIRONMENT_PINS['transformers']}`
- peft: `{ENVIRONMENT_PINS['peft']}`
- trl: `{ENVIRONMENT_PINS['trl']}`
- bitsandbytes: `{ENVIRONMENT_PINS['bitsandbytes']}`
- accelerate: `{ENVIRONMENT_PINS['accelerate']}`
- datasets: `{ENVIRONMENT_PINS['datasets']}`
- safetensors: `{ENVIRONMENT_PINS['safetensors']}`

## GPU Requirement

- Minimum VRAM: `16GB`
- Recommended VRAM: `24GB+`
- Preferred GPU for clean reproduction: `A100 40GB`
- Best for repeated variants/windows: `A100 80GB`

## Reproducibility Notes

- Use the generated smoke dataset first.
- Do not start full training unless `full_training_gate.json` passes after remote smoke training/inference.
- Keep candidate roster train-derived only.
"""


def build_runbook(dataset: str) -> str:
    report_dir = f"bug_tracking_llm_system/assignee_triage_accuracy/phase5_llm_reproduction/phase5e_prep/reports/{dataset}"
    return f"""# Phase 5E Remote Training Runbook

Primary smoke role: literature-aligned reproduction using `{PRIMARY_VARIANT}`.
Metadata-inclusive `{ADAPTED_METADATA_VARIANT}` is an adapted reproduction and must not be reported as the literature-exact input.

## 1. Create Environment

```bash
conda env create -f {report_dir}/environment_phase5_llm.yml
conda activate phase5-llm-qlora
```

## 2. Verify Environment

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import transformers, peft, trl, bitsandbytes, accelerate; print('ok')"
```

## 3. Smoke Training Only

```bash
python bug_tracking_llm_system/assignee_triage_accuracy/scripts/run_phase5e_remote_smoke_gate.py \\
  --execution-config {report_dir}/phase5e_execution_config.json \\
  --phase5-report-dir bug_tracking_llm_system/assignee_triage_accuracy/phase5_llm_reproduction/reports/{dataset} \\
  --phase5e-report-dir {report_dir} \\
  --max-steps 20
```

## 4. Optional Standalone Inference Smoke

```bash
python bug_tracking_llm_system/assignee_triage_accuracy/scripts/infer_phase5_llm.py \\
  --config {report_dir}/phase5e_execution_config.json \\
  --adapter-dir {report_dir}/smoke_checkpoint/final_adapter \\
  --input-jsonl bug_tracking_llm_system/assignee_triage_accuracy/phase5_llm_reproduction/reports/{dataset}/datasets/{PRIMARY_VARIANT}/validation.jsonl \\
  --roster-json bug_tracking_llm_system/assignee_triage_accuracy/phase5_llm_reproduction/reports/{dataset}/llm_dataset_manifest.json \\
  --output-csv {report_dir}/smoke_llm_predictions.csv \\
  --mode candidate_scoring
```

## 5. Full Training Gate

Update `full_training_gate.json` only after remote smoke confirms NF4 load, LoRA attach, finite loss, checkpoint reload, adapter reload, validation forward, and candidate-scoring inference.
"""


def build_summary(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    leakage_audit: dict[str, Any],
    tokenization_audit: list[dict[str, Any]],
    gate: dict[str, Any],
    runtime_estimates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": "phase5e_prep",
        "status": "remote_reproduction_package_prepared_no_full_training",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "local_platform": {
            "python": sys.version,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dataset": manifest["dataset"],
        "dataset_version": manifest["dataset_version"],
        "train_rows": manifest["split_counts"]["train"],
        "validation_rows": manifest["split_counts"]["validation"],
        "test_rows": manifest["split_counts"]["test"],
        "candidate_roster_size": manifest["candidate_roster"]["size"],
        "primary_input_variant": PRIMARY_VARIANT,
        "adapted_metadata_input_variant": ADAPTED_METADATA_VARIANT,
        "experiment_roles": {
            "literature_aligned_primary_reproduction": PRIMARY_VARIANT,
            "metadata_enriched_adapted_reproduction": ADAPTED_METADATA_VARIANT,
        },
        "prompt_integrity_passed": leakage_audit["passed"],
        "tokenization_blocking_checks": [
            row["check"] for row in tokenization_audit if row.get("blocking_for_full_training")
        ],
        "full_training_gate": gate,
        "runtime_estimates": runtime_estimates,
        "recommendation": {
            "A_safe_to_full_train_now": False,
            "B_recommended_gpu": "A100 40GB; 24GB consumer GPU acceptable for single smoke/full run with careful seq length",
            "C_16gb_enough": "possibly, but OOM risk is medium-high",
            "D_24gb_more_stable": True,
            "E_A100_recommended": True,
            "F_first_input_variant": PRIMARY_VARIANT,
            "G_metadata_inclusive_variant": f"Run {ADAPTED_METADATA_VARIANT} only as metadata-enriched adapted reproduction.",
            "H_constrained_inference": "candidate_scoring first; true constrained decoding optional later",
            "I_reproduction_label_now": "literature_aligned_preparation_not_exact_training",
            "J_blocking_issue": "Remote CUDA smoke training and real candidate-scoring inference have not passed.",
        },
        "artifacts_root": str(output_dir),
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    rec = summary["recommendation"]
    return f"""# Phase 5E-Prep Summary

Remote CUDA reproduction package is prepared. Full training was not run.

## Status

- Dataset: `{summary['dataset']}`
- Primary input variant: `{summary['primary_input_variant']}`
- Adapted metadata variant: `{summary['adapted_metadata_input_variant']}`
- Candidate roster size: `{summary['candidate_roster_size']}`
- Prompt integrity passed: `{summary['prompt_integrity_passed']}`
- Full training gate passed: `{summary['full_training_gate']['passed']}`
- Failed checks: `{summary['full_training_gate']['failed_checks']}`

## Recommendation

- Safe to full train now: `{rec['A_safe_to_full_train_now']}`
- Recommended GPU: `{rec['B_recommended_gpu']}`
- 16GB enough: `{rec['C_16gb_enough']}`
- 24GB more stable: `{rec['D_24gb_more_stable']}`
- A100 recommended: `{rec['E_A100_recommended']}`
- First input variant: `{rec['F_first_input_variant']}`
- Inference method: `{rec['H_constrained_inference']}`
- Current reproduction label: `{rec['I_reproduction_label_now']}`
- Blocking issue: `{rec['J_blocking_issue']}`

## Important Note

`smoke_llm_predictions.csv` is a dry-run schema artifact only until `infer_phase5_llm.py --mode candidate_scoring` is run with a real base model or adapter.
"""


if __name__ == "__main__":
    main()
