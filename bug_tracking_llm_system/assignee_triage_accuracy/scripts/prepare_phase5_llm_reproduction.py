from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assignee_phase1_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PHASE1_ROOT,
    EVAL_ROOT,
    clean,
    file_sha256,
    normalize_assignee,
    parse_date,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET = "bmo_paper_2024_3k"
DEFAULT_PHASE2_ROOT = EVAL_ROOT / "phase2_temporal_description"
DEFAULT_PHASE3_ROOT = EVAL_ROOT / "phase3_confidence_calibration"
DEFAULT_PHASE4_ROOT = EVAL_ROOT / "phase4_open_set_detection"
DEFAULT_PHASE5_ROOT = EVAL_ROOT / "phase5_llm_reproduction"
SYSTEM_PROMPT = "You are an expert bug triager."
STRICT_TEMPLATE = """Below is an issue. Suggest the single best developer to resolve it.

### Issue:
{ticket_text}

### Assignee:
{gold_assignee}"""

INPUT_VARIANTS = {
    "title_only": {
        "description": "Title only; lowest-information control.",
        "fields": ("title",),
        "include_description": False,
        "include_metadata": False,
    },
    "title_plus_metadata": {
        "description": "Title plus stable metadata; deterministic strongest-control input.",
        "fields": ("title", "product", "component", "priority", "severity", "status"),
        "include_description": False,
        "include_metadata": True,
    },
    "sanitized_title_plus_description": {
        "description": "Title plus sanitized/truncated first-comment description; tests text understanding.",
        "fields": ("title", "description"),
        "include_description": True,
        "include_metadata": False,
    },
    "sanitized_title_plus_description_plus_metadata": {
        "description": "Primary LLM input: title, sanitized description, and stable metadata.",
        "fields": ("title", "description", "product", "component", "priority", "severity", "status"),
        "include_description": True,
        "include_metadata": True,
    },
}

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
LOG_LINE_RE = re.compile(
    r"(error|exception|fail|fatal|traceback|assert|crash|segfault|stack|test-|warning|TypeError|ReferenceError|"
    r"SyntaxError|Timeout|INFO - TEST|UNEXPECTED)",
    re.IGNORECASE,
)
QUOTE_LINE_RE = re.compile(r"^\s*(>|&gt;|On .+ wrote:|From:|Sent:|Subject:)", re.IGNORECASE)
ORIGINAL_MESSAGE_RE = re.compile(r"^\s*(-{2,}\s*Original Message\s*-{2,}|_{5,})\s*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 5 literature-style LLM assignee reproduction data and feasibility reports."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-description-chars", type=int, default=4000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.output_dir or DEFAULT_PHASE5_ROOT / "reports" / args.dataset
    dataset_dir = report_dir / "datasets"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_source_paths(args.dataset, args.data_dir)
    train_rows = read_jsonl(source["train"])
    validation_rows = read_jsonl(source["validation"])
    test_rows = read_jsonl(source["test"])
    roster = read_roster(source["roster"])

    guard = leakage_guard(train_rows, validation_rows, test_rows, roster)
    if guard["errors"]:
        raise SystemExit("Phase 5 dataset guard failed: " + "; ".join(guard["errors"]))

    generated = {}
    variant_stats = {}
    for variant_name, variant_spec in INPUT_VARIANTS.items():
        variant_dir = dataset_dir / variant_name
        train_records, train_stats = build_records(
            train_rows,
            variant_name=variant_name,
            variant_spec=variant_spec,
            split="train",
            include_answer=True,
            max_description_chars=args.max_description_chars,
        )
        validation_records, validation_stats = build_records(
            validation_rows,
            variant_name=variant_name,
            variant_spec=variant_spec,
            split="validation",
            include_answer=True,
            max_description_chars=args.max_description_chars,
        )
        write_jsonl(variant_dir / "train.jsonl", train_records)
        write_jsonl(variant_dir / "validation.jsonl", validation_records)
        generated[variant_name] = {
            "train_jsonl": str(variant_dir / "train.jsonl"),
            "validation_jsonl": str(variant_dir / "validation.jsonl"),
            "train_sha256": file_sha256(variant_dir / "train.jsonl"),
            "validation_sha256": file_sha256(variant_dir / "validation.jsonl"),
        }
        variant_stats[variant_name] = {
            "train": train_stats,
            "validation": validation_stats,
            "input_fields": list(variant_spec["fields"]),
        }

    hardware = inspect_hardware()
    training_config = build_training_config(args.dataset, args.seed, hardware, generated)
    manifest = build_manifest(
        dataset=args.dataset,
        seed=args.seed,
        source=source,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        roster=roster,
        guard=guard,
        generated=generated,
        variant_stats=variant_stats,
        max_description_chars=args.max_description_chars,
    )
    gap_markdown, gap_json = build_gap_report(hardware)
    summary_markdown, summary_json = build_summary(manifest, hardware, training_config, gap_json)

    write_json(report_dir / "llm_dataset_manifest.json", manifest)
    write_json(report_dir / "phase5_hardware_feasibility.json", hardware)
    write_json(report_dir / "llm_training_config.json", training_config)
    write_json(report_dir / "phase5_llm_reproduction_summary.json", summary_json)
    (report_dir / "literature_method_gap.md").write_text(gap_markdown, encoding="utf-8")
    (report_dir / "phase5_llm_reproduction_summary.md").write_text(summary_markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "output_dir": str(report_dir),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows_not_exported_for_training": len(test_rows),
                "candidate_roster_size": len(roster),
                "exact_8b_qlora_feasible": hardware["decision"]["can_exact_8b_qlora_locally"],
                "recommended_mode": hardware["decision"]["recommended_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def resolve_source_paths(dataset: str, data_dir: Path) -> dict[str, Path]:
    enriched_root = (
        DEFAULT_PHASE2_ROOT
        / "reports"
        / f"{dataset}_description_enriched"
        / "description_ablation"
        / "feature_sliced_data"
        / "title_plus_description_plus_metadata"
    )
    if enriched_root.exists():
        return {
            "dataset_version": Path(f"{dataset}_description_enriched"),
            "train": enriched_root / "history_train.jsonl",
            "validation": enriched_root / "validation_set.jsonl",
            "test": enriched_root / "test_set.jsonl",
            "roster": enriched_root / "candidate_roster.json",
            "source_kind": Path("phase2_description_enriched_primary"),
        }
    return {
        "dataset_version": Path(dataset),
        "train": data_dir / f"{dataset}_history_train.jsonl",
        "validation": data_dir / f"{dataset}_validation_set.jsonl",
        "test": data_dir / f"{dataset}_test_set.jsonl",
        "roster": data_dir / f"{dataset}_candidate_roster.json",
        "source_kind": Path("paper_grade_processed_fallback"),
    }


def read_roster(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing roster file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates", []) if isinstance(value, dict) else value
    return sorted({normalize_assignee(candidate) for candidate in candidates if normalize_assignee(candidate)})


def leakage_guard(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: list[str],
) -> dict[str, Any]:
    train_ids = {clean(row.get("ticket_id")) for row in train_rows}
    validation_ids = {clean(row.get("ticket_id")) for row in validation_rows}
    test_ids = {clean(row.get("ticket_id")) for row in test_rows}
    train_labels = {normalize_assignee(row.get("assignee")) for row in train_rows if normalize_assignee(row.get("assignee"))}
    validation_labels = {
        normalize_assignee(row.get("assignee")) for row in validation_rows if normalize_assignee(row.get("assignee"))
    }
    test_labels = {normalize_assignee(row.get("assignee")) for row in test_rows if normalize_assignee(row.get("assignee"))}
    roster_set = set(roster)
    errors = []
    if train_ids & validation_ids:
        errors.append("train/validation ticket overlap")
    if train_ids & test_ids:
        errors.append("train/test ticket overlap")
    if validation_ids & test_ids:
        errors.append("validation/test ticket overlap")
    if not roster_set <= train_labels:
        errors.append("candidate roster contains labels absent from train")
    if not validation_labels <= roster_set:
        errors.append("closed-set validation contains labels absent from train roster")
    if not test_labels <= roster_set:
        errors.append("closed-set test contains labels absent from train roster")

    train_max = max_date(train_rows)
    validation_min = min_date(validation_rows)
    validation_max = max_date(validation_rows)
    test_min = min_date(test_rows)
    temporal_ok = bool(train_max and validation_min and validation_max and test_min and train_max <= validation_min <= validation_max <= test_min)
    if not temporal_ok:
        errors.append("temporal split order could not be verified")

    train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
    return {
        "errors": errors,
        "ticket_overlap": {
            "train_validation": len(train_ids & validation_ids),
            "train_test": len(train_ids & test_ids),
            "validation_test": len(validation_ids & test_ids),
        },
        "candidate_roster_policy": {
            "roster_size": len(roster_set),
            "train_label_count": len(train_labels),
            "roster_subset_of_train_labels": roster_set <= train_labels,
            "validation_labels_subset_of_roster": validation_labels <= roster_set,
            "test_labels_subset_of_roster": test_labels <= roster_set,
            "test_label_not_used_to_expand_roster": roster_set <= train_labels,
        },
        "temporal_order": {
            "train_max_created_at": train_max.isoformat() if train_max else "",
            "validation_min_created_at": validation_min.isoformat() if validation_min else "",
            "validation_max_created_at": validation_max.isoformat() if validation_max else "",
            "test_min_created_at": test_min.isoformat() if test_min else "",
            "temporal_order_verified": temporal_ok,
        },
        "long_tail_train_bucket_counts": {
            "freq_1": sum(1 for count in train_counts.values() if count == 1),
            "freq_2_4": sum(1 for count in train_counts.values() if 2 <= count <= 4),
            "freq_5_9": sum(1 for count in train_counts.values() if 5 <= count <= 9),
            "freq_10_19": sum(1 for count in train_counts.values() if 10 <= count <= 19),
            "freq_20_plus": sum(1 for count in train_counts.values() if count >= 20),
        },
    }


def min_date(rows: list[dict[str, Any]]) -> datetime | None:
    values = [as_utc(parse_date(row.get("created_at"))) for row in rows]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def max_date(rows: list[dict[str, Any]]) -> datetime | None:
    values = [as_utc(parse_date(row.get("created_at"))) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def build_records(
    rows: list[dict[str, Any]],
    *,
    variant_name: str,
    variant_spec: dict[str, Any],
    split: str,
    include_answer: bool,
    max_description_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    sanitization_rows = []
    for row in rows:
        ticket_text, sanitization = build_ticket_text(row, variant_spec, max_description_chars=max_description_chars)
        assignee = normalize_assignee(row.get("assignee"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ticket_text)},
        ]
        if include_answer:
            messages.append({"role": "assistant", "content": assignee})
        records.append(
            {
                "id": f"{split}:{variant_name}:{clean(row.get('ticket_id'))}",
                "ticket_id": clean(row.get("ticket_id")),
                "split": split,
                "input_variant": variant_name,
                "messages": messages,
                "strict_sft_text": STRICT_TEMPLATE.format(ticket_text=ticket_text, gold_assignee=assignee),
                "metadata": {
                    "expected_assignee": assignee,
                    "product": clean(row.get("product")).lower() or "unknown",
                    "component": clean(row.get("component")).lower() or "unknown",
                    "priority": clean(row.get("priority")) or "unknown",
                    "severity": clean(row.get("severity")) or "unknown",
                    "status": clean(row.get("status")) or "unknown",
                    "created_at": clean(row.get("created_at")),
                    "sanitization": sanitization,
                },
            }
        )
        sanitization_rows.append(sanitization)
    return records, summarize_sanitization(sanitization_rows)


def build_ticket_text(row: dict[str, Any], variant_spec: dict[str, Any], *, max_description_chars: int) -> tuple[str, dict[str, Any]]:
    title = clean(row.get("title"))
    description_raw = clean(row.get("description"))
    description_clean, sanitization = sanitize_description(description_raw, max_chars=max_description_chars)
    lines = [f"Title: {title or '(empty)'}"]
    if variant_spec["include_description"]:
        lines.append(f"Description: {description_clean or '(empty)'}")
    if variant_spec["include_metadata"]:
        lines.append("Metadata:")
        lines.append(f"- Product: {clean(row.get('product')).lower() or 'unknown'}")
        lines.append(f"- Component: {clean(row.get('component')).lower() or 'unknown'}")
        lines.append(f"- Priority: {clean(row.get('priority')) or 'unknown'}")
        lines.append(f"- Severity: {clean(row.get('severity')) or 'unknown'}")
        lines.append(f"- Status: {clean(row.get('status')) or 'unknown'}")
    sanitization["used_description"] = bool(variant_spec["include_description"])
    sanitization["used_metadata"] = bool(variant_spec["include_metadata"])
    return "\n".join(lines), sanitization


def build_user_prompt(ticket_text: str) -> str:
    return (
        "Below is an issue. Suggest the single best developer to resolve it.\n\n"
        f"### Issue:\n{ticket_text}\n\n"
        "### Assignee:"
    )


def sanitize_description(value: str, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    raw = value or ""
    text = html.unescape(raw)
    had_html = bool(HTML_TAG_RE.search(text))
    text = HTML_TAG_RE.sub(" ", text)
    output_lines = []
    quoted_removed = 0
    original_message_removed = False
    for line in text.splitlines():
        if ORIGINAL_MESSAGE_RE.search(line):
            original_message_removed = True
            break
        if QUOTE_LINE_RE.search(line):
            quoted_removed += 1
            continue
        output_lines.append(line)
    text = "\n".join(output_lines)
    text = WHITESPACE_RE.sub(" ", text).strip()
    collapsed_chars = len(text)
    truncated = False
    if len(text) > max_chars:
        truncated = True
        text = truncate_preserving_errors(text, max_chars=max_chars)
    return text, {
        "raw_description_chars": len(raw),
        "sanitized_description_chars": len(text),
        "collapsed_description_chars_before_truncation": collapsed_chars,
        "had_html": had_html,
        "quoted_lines_removed": quoted_removed,
        "original_message_removed": original_message_removed,
        "description_was_truncated": truncated,
        "max_description_chars": max_chars,
    }


def truncate_preserving_errors(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    salient = []
    for sentence in re.split(r"(?<=[.!?])\s+|\s{2,}", text):
        if LOG_LINE_RE.search(sentence):
            salient.append(sentence.strip())
    prefix_budget = max(1000, max_chars // 2)
    prefix = text[:prefix_budget].strip()
    suffix_budget = max_chars - len(prefix) - 80
    salient_text = " ".join(salient)
    if suffix_budget > 0 and salient_text:
        suffix = salient_text[:suffix_budget].strip()
        return f"{prefix}\n\n[TRUNCATED; preserved salient error/log lines]\n{suffix}".strip()
    return f"{text[: max_chars - 40].strip()}\n\n[TRUNCATED]".strip()


def summarize_sanitization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {}
    return {
        "rows": total,
        "used_description_rows": sum(1 for row in rows if row["used_description"]),
        "used_metadata_rows": sum(1 for row in rows if row["used_metadata"]),
        "nonempty_raw_description_rows": sum(1 for row in rows if row["raw_description_chars"] > 0),
        "html_rows": sum(1 for row in rows if row["had_html"]),
        "truncated_rows": sum(1 for row in rows if row["description_was_truncated"]),
        "quoted_lines_removed_total": sum(int(row["quoted_lines_removed"]) for row in rows),
        "mean_raw_description_chars": round(sum(row["raw_description_chars"] for row in rows) / total, 3),
        "mean_sanitized_description_chars": round(sum(row["sanitized_description_chars"] for row in rows) / total, 3),
    }


def build_manifest(
    *,
    dataset: str,
    seed: int,
    source: dict[str, Path],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: list[str],
    guard: dict[str, Any],
    generated: dict[str, Any],
    variant_stats: dict[str, Any],
    max_description_chars: int,
) -> dict[str, Any]:
    source_files = [source["train"], source["validation"], source["test"], source["roster"]]
    train_counts = Counter(normalize_assignee(row.get("assignee")) for row in train_rows)
    return {
        "phase": "phase5_llm_reproduction",
        "status": "phase5a_to_5d_pretraining_artifacts_only",
        "dataset": dataset,
        "dataset_version": str(source["dataset_version"]),
        "source_kind": str(source["source_kind"]),
        "seed": seed,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_paths": {key: str(value) for key, value in source.items()},
        "source_hashes": {str(path): file_sha256(path) for path in source_files},
        "dataset_hash": combined_content_hash(source_files, generated),
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "note": "test rows are referenced for fairness/leakage checks only; no test JSONL is exported for SFT.",
        },
        "candidate_roster": {
            "size": len(roster),
            "source": str(source["roster"]),
            "policy": "train_labels_only; no validation/test label expansion",
            "candidates": roster,
        },
        "input_variants": {
            name: {**spec, "fields": list(spec["fields"])} for name, spec in INPUT_VARIANTS.items()
        },
        "description_sanitization": {
            "rules": [
                "HTML tags removed and entities unescaped",
                "quoted reply lines removed",
                "original-message blocks removed",
                "repeated whitespace collapsed",
                "extreme logs truncated while preserving error/log lines",
            ],
            "max_description_chars": max_description_chars,
            "stats": variant_stats,
        },
        "leakage_guard": guard,
        "long_tail_buckets": {
            "ticket_frequency_definition": "count of expected assignee in train split",
            "bucket_rules": {"1": "freq_1", "2-4": "freq_2_4", "5-9": "freq_5_9", "10-19": "freq_10_19", "20+": "freq_20_plus"},
            "train_assignee_frequency_distribution": dict(sorted(train_counts.items())),
        },
        "benchmark_views": benchmark_views(dataset),
        "generated_files": generated,
        "forbidden_usage": [
            "Do not train on validation/test rows.",
            "Do not add validation/test-only assignees to closed-set roster.",
            "Do not select checkpoints, prompts, thresholds, or candidate policy on test labels.",
            "Do not compare LLM to deterministic baselines on different splits or rosters.",
        ],
    }


def benchmark_views(dataset: str) -> dict[str, Any]:
    phase1 = DEFAULT_PHASE1_ROOT / "reports" / dataset
    phase2 = DEFAULT_PHASE2_ROOT / "reports" / dataset
    enriched = DEFAULT_PHASE2_ROOT / "reports" / f"{dataset}_description_enriched"
    phase4 = DEFAULT_PHASE4_ROOT / "reports" / dataset
    return {
        "frozen_closed_set": {
            "train": str(DEFAULT_DATA_DIR / f"{dataset}_history_train.jsonl"),
            "validation": str(DEFAULT_DATA_DIR / f"{dataset}_validation_set.jsonl"),
            "test": str(DEFAULT_DATA_DIR / f"{dataset}_test_set.jsonl"),
            "roster": str(DEFAULT_DATA_DIR / f"{dataset}_candidate_roster.json"),
        },
        "multi_window_temporal": {
            "manifest": str(phase2 / "multi_window_temporal" / "multi_window_manifest.json"),
            "window_data_root": str(phase2 / "multi_window_temporal" / "window_data"),
            "summary": str(phase2 / "multi_window_temporal" / "multi_window_summary.csv"),
        },
        "open_world": {
            "train": str(phase1 / "prefilter_open_world_eval" / "data" / "prefilter_history_train.jsonl"),
            "validation": str(phase1 / "prefilter_open_world_eval" / "data" / "prefilter_validation_set.jsonl"),
            "test": str(phase1 / "prefilter_open_world_eval" / "data" / "prefilter_test_set.jsonl"),
            "roster": str(phase1 / "prefilter_open_world_eval" / "data" / "prefilter_candidate_roster.json"),
            "phase4_shared_routing": str(phase4 / "routing_policy_open_set_comparison.csv"),
        },
        "long_tail": {
            "phase1_open_world_bucket_metrics": str(phase1 / "prefilter_open_world_eval" / "long_tail_bucket_metrics.csv"),
            "phase2_description_ablation_root": str(enriched / "description_ablation"),
        },
    }


def combined_content_hash(source_files: list[Path], generated: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    for variant in sorted(generated):
        digest.update(variant.encode("utf-8"))
        digest.update(generated[variant]["train_sha256"].encode("ascii"))
        digest.update(generated[variant]["validation_sha256"].encode("ascii"))
    return digest.hexdigest()


def inspect_hardware() -> dict[str, Any]:
    packages = {name: package_version(name) for name in PACKAGE_NAMES}
    ram_bytes = total_memory_bytes()
    disk = shutil.disk_usage(EVAL_ROOT)
    torch_info = inspect_torch()
    nvidia_smi = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    nvcc = run_command(["nvcc", "--version"])
    max_cuda_vram_gb = max((gpu.get("total_memory_gb", 0.0) for gpu in torch_info.get("cuda_devices", [])), default=0.0)
    has_cuda = bool(torch_info.get("cuda_available"))
    has_required_libs = all(packages.get(name, {}).get("installed") for name in ("torch", "transformers", "peft", "trl", "bitsandbytes"))
    exact_feasible = bool(has_cuda and max_cuda_vram_gb >= 14.0 and has_required_libs)
    reasons = []
    if not has_cuda:
        reasons.append("CUDA GPU not available to Python torch runtime.")
    if max_cuda_vram_gb and max_cuda_vram_gb < 14.0:
        reasons.append(f"Detected CUDA VRAM {max_cuda_vram_gb:.2f} GB is below a conservative 14 GB 8B QLoRA floor.")
    for name in ("transformers", "peft", "trl", "bitsandbytes"):
        if not packages.get(name, {}).get("installed"):
            reasons.append(f"{name} is not installed.")
    if torch_info.get("mps_available") and not has_cuda:
        reasons.append("Apple MPS may run smaller LoRA/adapters, but it is not an exact 4-bit NF4 bitsandbytes QLoRA reproduction.")
    return {
        "platform": {
            "python": sys.version,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "memory": {
            "ram_bytes": ram_bytes,
            "ram_gb": round(ram_bytes / (1024**3), 3) if ram_bytes else None,
        },
        "disk": {
            "path": str(EVAL_ROOT),
            "total_gb": round(disk.total / (1024**3), 3),
            "free_gb": round(disk.free / (1024**3), 3),
        },
        "torch": torch_info,
        "commands": {
            "nvidia_smi": nvidia_smi,
            "nvcc": nvcc,
        },
        "packages": packages,
        "decision": {
            "can_exact_8b_qlora_locally": exact_feasible,
            "requires_remote_gpu": not exact_feasible,
            "recommended_mode": "exact_literature_reproduction" if exact_feasible else "resource_constrained_pretraining_only_until_remote_gpu",
            "minimum_remote_gpu_recommendation": "CUDA GPU with >=16 GB VRAM; >=24 GB preferred for DeepSeek-R1-Distill-Llama-8B QLoRA SFT.",
            "acceptable_adaptation_if_remote_unavailable": "smaller instruction model or CPU/MPS candidate-scoring smoke test, clearly labeled resource-constrained adaptation.",
            "blocking_reasons": reasons,
        },
    }


def package_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return int(page_size * pages)
        except (ValueError, OSError, AttributeError):
            pass
    sysctl = run_command(["sysctl", "-n", "hw.memsize"])
    if sysctl["returncode"] == 0:
        try:
            return int(sysctl["stdout"].strip())
        except ValueError:
            return None
    return None


def inspect_torch() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"installed": False, "import_error": repr(exc), "cuda_available": False, "mps_available": False}
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {
        "installed": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": cuda_devices,
        "mps_available": mps_available,
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
    }


def run_command(command: list[str]) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "returncode": None, "stdout": "", "stderr": "command not found"}
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"available": True, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_training_config(dataset: str, seed: int, hardware: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    primary_variant = "sanitized_title_plus_description_plus_metadata"
    return {
        "status": "not_trained_yet",
        "dataset": dataset,
        "seed": seed,
        "reproduction_mode": (
            "exact_literature_reproduction_target"
            if hardware["decision"]["can_exact_8b_qlora_locally"]
            else "exact_config_recorded_but_training_requires_remote_gpu_or_adaptation"
        ),
        "base_model": "DeepSeek-R1-Distill-Llama-8B",
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "sft": {
            "objective": "next_token_prediction",
            "format": "conversational_jsonl_with_strict_sft_text",
            "primary_input_variant": primary_variant,
            "train_jsonl": generated[primary_variant]["train_jsonl"],
            "validation_jsonl": generated[primary_variant]["validation_jsonl"],
            "max_seq_length": 4096,
            "num_train_epochs": 3,
            "learning_rate": 0.0002,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "checkpoint_selection": "validation_loss_only_no_test_selection",
        },
        "inference_plan": {
            "candidate_roster_policy": "train_labels_only",
            "top_k": [1, 3, 5, 10],
            "preferred_method": "candidate_scoring_or_true_constrained_decoding; never free-form hallucinated assignee",
            "invalid_assignee_prevention": "decode only valid candidate token sequences when feasible; otherwise score every roster candidate and sort",
        },
    }


def build_gap_report(hardware: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "item": "Base model",
            "literature_target": "DeepSeek-R1-Distill-Llama-8B",
            "phase5_status": "cannot reproduce",
            "reason": "No local checkpoint/training run exists yet; config records the exact target for a future run.",
        },
        {
            "item": "Quantization",
            "literature_target": "4-bit NF4",
            "phase5_status": "cannot reproduce" if not hardware["decision"]["can_exact_8b_qlora_locally"] else "exact reproduction",
            "reason": "Exact NF4 QLoRA requires CUDA bitsandbytes; current feasibility check decides whether local execution is possible.",
        },
        {
            "item": "LoRA modules",
            "literature_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "phase5_status": "exact reproduction",
            "reason": "The Phase 5 training config preserves the target module list exactly.",
        },
        {
            "item": "LoRA hyperparameters",
            "literature_target": "r=16, alpha=16, dropout=0.0",
            "phase5_status": "exact reproduction",
            "reason": "The Phase 5 training config preserves these values exactly.",
        },
        {
            "item": "SFT",
            "literature_target": "next-token objective over conversational JSONL",
            "phase5_status": "adapted reproduction",
            "reason": "Conversational JSONL and strict text template are prepared; actual SFT is intentionally not run in Phase 5A-D.",
        },
        {
            "item": "Prompt",
            "literature_target": "system/user/assistant",
            "phase5_status": "adapted reproduction",
            "reason": "Role structure is reproduced; user content is adapted to fixed BMO fields and controlled input variants.",
        },
        {
            "item": "Input",
            "literature_target": "title + description",
            "phase5_status": "adapted reproduction",
            "reason": "Adds mandatory controls: title_only, title_plus_metadata, sanitized_title_plus_description, and sanitized_title_plus_description_plus_metadata.",
        },
        {
            "item": "Candidate-constrained decoding",
            "literature_target": "restrict output to valid assignees",
            "phase5_status": "cannot reproduce",
            "reason": "No inference code is run yet; Phase 5F must implement true constrained decoding or candidate scoring and report which one is used.",
        },
        {
            "item": "Top-1",
            "literature_target": "single best assignee",
            "phase5_status": "cannot reproduce",
            "reason": "No trained adapter or constrained inference output exists yet.",
        },
        {
            "item": "Top-K",
            "literature_target": "ranked assignee candidates",
            "phase5_status": "cannot reproduce",
            "reason": "Top-K candidate scoring/inference is planned for Phase 5F, not Phase 5A-D.",
        },
    ]
    lines = [
        "# Phase 5A Literature Method Gap Check",
        "",
        "This report is additive and does not modify `current_assignee_triager`.",
        "",
        "| Item | Literature target | Phase 5 status | Reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['item']} | {row['literature_target']} | {row['phase5_status']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Deviation Policy",
            "",
            "- `exact reproduction`: config and execution path match the literature item.",
            "- `adapted reproduction`: benchmark-controlled adaptation is required for same split/roster/input fairness.",
            "- `cannot reproduce`: not executed yet or blocked by local hardware/runtime.",
            "- `intentionally changed`: would be used only when the literature design is unsafe or unfair for this benchmark.",
        ]
    )
    return "\n".join(lines) + "\n", rows


def build_summary(
    manifest: dict[str, Any],
    hardware: dict[str, Any],
    training_config: dict[str, Any],
    gap_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    exact_possible = hardware["decision"]["can_exact_8b_qlora_locally"]
    deferred_artifacts = [
        "training_metrics.csv",
        "llm_predictions.csv",
        "llm_closed_set_metrics.json",
        "llm_long_tail_metrics.csv",
        "llm_multi_window_summary.csv",
        "llm_open_world_metrics.json",
        "llm_vs_baselines.csv",
        "statistical_comparison.csv",
    ]
    summary = {
        "phase": "phase5_llm_reproduction",
        "status": "phase5a_to_5d_complete_no_training_run",
        "dataset": manifest["dataset"],
        "split_counts": manifest["split_counts"],
        "candidate_roster_size": manifest["candidate_roster"]["size"],
        "dataset_hash": manifest["dataset_hash"],
        "leakage_guard_errors": manifest["leakage_guard"]["errors"],
        "exact_8b_qlora_feasible_locally": exact_possible,
        "recommended_mode": hardware["decision"]["recommended_mode"],
        "blocking_reasons": hardware["decision"]["blocking_reasons"],
        "primary_training_variant": training_config["sft"]["primary_input_variant"],
        "gap_status_counts": dict(Counter(row["phase5_status"] for row in gap_rows)),
        "deferred_artifacts_until_training_and_inference": deferred_artifacts,
        "next_gate": (
            "Phase 5E can run locally if model weights are available."
            if exact_possible
            else "Use remote CUDA GPU for exact reproduction, or run a clearly labeled resource-constrained adaptation."
        ),
    }
    lines = [
        "# Phase 5 LLM Reproduction Summary",
        "",
        "Phase 5A-D artifacts were prepared. No LLM training or inference was run.",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Dataset version: `{manifest['dataset_version']}`",
        f"- Train / validation / test: `{manifest['split_counts']['train']}` / `{manifest['split_counts']['validation']}` / `{manifest['split_counts']['test']}`",
        f"- Candidate roster size: `{manifest['candidate_roster']['size']}`",
        f"- Dataset hash: `{manifest['dataset_hash']}`",
        f"- Leakage guard errors: `{manifest['leakage_guard']['errors']}`",
        "",
        "## Hardware",
        "",
        f"- Exact 8B QLoRA feasible locally: `{exact_possible}`",
        f"- Recommended mode: `{hardware['decision']['recommended_mode']}`",
        f"- Blocking reasons: `{hardware['decision']['blocking_reasons']}`",
        "",
        "## Deferred Artifacts",
        "",
        "These were not generated because Phase 5E-F training/inference was not run:",
        "",
        *[f"- `{artifact}`" for artifact in deferred_artifacts],
        "",
        "## Next Gate",
        "",
        summary["next_gate"],
    ]
    return "\n".join(lines) + "\n", summary


if __name__ == "__main__":
    main()
