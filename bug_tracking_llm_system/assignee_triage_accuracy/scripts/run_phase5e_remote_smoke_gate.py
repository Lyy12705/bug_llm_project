from __future__ import annotations

import argparse
import csv
import gc
import html
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PRIMARY_VARIANT = "sanitized_title_plus_description"
ADAPTED_METADATA_VARIANT = "sanitized_title_plus_description_plus_metadata"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GATE_CHECKS = (
    "environment_ok",
    "nf4_load_ok",
    "lora_attach_ok",
    "prompt_integrity_ok",
    "label_leakage_ok",
    "tokenization_ok",
    "smoke_loss_finite",
    "checkpoint_reload_ok",
    "adapter_reload_ok",
    "inference_smoke_ok",
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
QUOTE_LINE_RE = re.compile(r"^\s*(>|&gt;|On .+ wrote:|From:|Sent:|Subject:)", re.IGNORECASE)
ORIGINAL_MESSAGE_RE = re.compile(r"^\s*(-{2,}\s*Original Message\s*-{2,}|_{5,})\s*$", re.IGNORECASE)
LOG_LINE_RE = re.compile(
    r"(error|exception|fail|fatal|traceback|assert|crash|segfault|stack|test-|warning|TypeError|ReferenceError|"
    r"SyntaxError|Timeout|INFO - TEST|UNEXPECTED)",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5E remote CUDA smoke execution gate.")
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--phase5-report-dir", type=Path, required=True)
    parser.add_argument("--phase5e-report-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--inference-limit", type=int, default=192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reload-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    args.phase5e_report_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(args.execution_config)
    manifest = read_json(args.phase5_report_dir / "llm_dataset_manifest.json")
    label_audit = read_json(args.phase5e_report_dir / "label_leakage_audit.json")
    smoke_train = read_jsonl(args.phase5e_report_dir / "smoke_train.jsonl")
    smoke_validation = read_jsonl(args.phase5e_report_dir / "smoke_validation.jsonl")
    test_rows = read_jsonl(Path(manifest["source_paths"]["test"]))
    roster = list(manifest["candidate_roster"]["candidates"])

    gate_state = {name: False for name in GATE_CHECKS}
    gate_state["prompt_integrity_ok"] = bool(label_audit.get("passed"))
    gate_state["label_leakage_ok"] = bool(label_audit.get("passed"))

    env_report, runtime = verify_remote_environment()
    write_json(args.phase5e_report_dir / "remote_environment_runtime.json", env_report)
    (args.phase5e_report_dir / "remote_environment_report.md").write_text(
        render_remote_environment_report(env_report), encoding="utf-8"
    )
    gate_state["environment_ok"] = bool(env_report["checks"]["environment_ok"])
    if not gate_state["environment_ok"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, env_report["failed_checks"])
        print_gate(args.phase5e_report_dir, gate_state, env_report["failed_checks"])
        return

    torch = runtime["torch"]
    transformers = runtime["transformers"]
    peft = runtime["peft"]
    set_seed(args.seed, torch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, model_report = load_nf4_model(cfg, torch, transformers, args.phase5e_report_dir)
    write_json(args.phase5e_report_dir / "model_load_report.json", model_report)
    gate_state["nf4_load_ok"] = bool(model_report["checks"]["nf4_load_ok"])
    if not gate_state["nf4_load_ok"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, ["nf4 model load failed"])
        print_gate(args.phase5e_report_dir, gate_state, ["nf4 model load failed"])
        return

    model, lora_report = attach_lora(model, cfg, peft)
    write_json(args.phase5e_report_dir / "lora_attach_report.json", lora_report)
    gate_state["lora_attach_ok"] = bool(lora_report["checks"]["lora_attach_ok"])
    if not gate_state["lora_attach_ok"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, ["LoRA attach failed"])
        print_gate(args.phase5e_report_dir, gate_state, ["LoRA attach failed"])
        return

    token_rows, mask_report = runtime_tokenization_audit(smoke_train + smoke_validation, tokenizer, max_length=cfg["sft"]["max_seq_length"])
    write_csv(args.phase5e_report_dir / "runtime_tokenization_audit.csv", token_rows, list(token_rows[0].keys()))
    write_json(args.phase5e_report_dir / "label_mask_audit.json", mask_report)
    gate_state["tokenization_ok"] = bool(mask_report["passed"])
    if not gate_state["tokenization_ok"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, mask_report["errors"])
        print_gate(args.phase5e_report_dir, gate_state, mask_report["errors"])
        return

    train_result = smoke_train_model(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        cfg=cfg,
        train_records=smoke_train,
        validation_records=smoke_validation,
        output_dir=args.phase5e_report_dir,
        max_steps=args.max_steps,
    )
    gate_state["smoke_loss_finite"] = bool(train_result["passed"])
    if not gate_state["smoke_loss_finite"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, train_result["errors"])
        print_gate(args.phase5e_report_dir, gate_state, train_result["errors"])
        return

    fixed_prompt = render_prompt(smoke_validation[0]["messages"][:2], tokenizer)
    reference_scores = score_candidates(fixed_prompt, roster[:10], model, tokenizer, torch)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint_report, _checkpoint_model = reload_and_compare_from_scores(
        cfg=cfg,
        adapter_dir=args.phase5e_report_dir / "smoke_checkpoint" / "checkpoint-smoke",
        prompt=fixed_prompt,
        roster=roster[:10],
        reference_scores=reference_scores,
        tokenizer=tokenizer,
        torch=torch,
        transformers=transformers,
        peft=peft,
        tolerance=args.reload_tolerance,
        keep_model=False,
    )
    write_json(args.phase5e_report_dir / "checkpoint_reload_report.json", checkpoint_report)
    gate_state["checkpoint_reload_ok"] = bool(checkpoint_report["passed"])

    adapter_report, model = reload_and_compare_from_scores(
        cfg=cfg,
        adapter_dir=args.phase5e_report_dir / "smoke_checkpoint" / "final_adapter",
        prompt=fixed_prompt,
        roster=roster[:10],
        reference_scores=reference_scores,
        tokenizer=tokenizer,
        torch=torch,
        transformers=transformers,
        peft=peft,
        tolerance=args.reload_tolerance,
        keep_model=True,
    )
    write_json(args.phase5e_report_dir / "adapter_reload_report.json", adapter_report)
    gate_state["adapter_reload_ok"] = bool(adapter_report["passed"])
    if not gate_state["checkpoint_reload_ok"] or not gate_state["adapter_reload_ok"]:
        write_blocked_artifacts(args.phase5e_report_dir, gate_state, ["checkpoint or adapter reload failed"])
        print_gate(args.phase5e_report_dir, gate_state, ["checkpoint or adapter reload failed"])
        return

    inference_result = real_candidate_scoring_smoke(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        records=test_rows[: args.inference_limit],
        roster=roster,
        output_dir=args.phase5e_report_dir,
        seed=args.seed,
    )
    gate_state["inference_smoke_ok"] = bool(inference_result["passed"])
    write_full_gate(args.phase5e_report_dir, gate_state, inference_result.get("warnings", []))
    write_remote_summary(args.phase5e_report_dir, gate_state, model_report, lora_report, mask_report, train_result, inference_result)
    print_gate(args.phase5e_report_dir, gate_state, inference_result.get("warnings", []))


def verify_remote_environment() -> tuple[dict[str, Any], dict[str, Any]]:
    packages = {}
    runtime: dict[str, Any] = {}
    failed = []
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "safetensors"):
        try:
            packages[name] = {"installed": True, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
            failed.append(f"{name} not installed")
    try:
        import torch
        import transformers
        import peft
        import bitsandbytes as bnb
    except Exception as exc:
        failed.append(f"runtime import failed: {exc}")
        return base_environment_report(packages, failed), runtime

    runtime.update({"torch": torch, "transformers": transformers, "peft": peft, "bitsandbytes": bnb})
    cuda_tensor_ok = False
    bnb_cuda_ok = False
    cuda_devices = []
    if torch.cuda.is_available():
        try:
            x = torch.ones(4, device="cuda")
            cuda_tensor_ok = bool(float((x + 1).sum().detach().cpu()) == 8.0)
        except Exception as exc:
            failed.append(f"CUDA tensor operation failed: {exc}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        try:
            layer = bnb.nn.Linear4bit(4, 4, bias=False, quant_type="nf4", compute_dtype=torch.float16).to("cuda")
            y = layer(torch.randn(2, 4, device="cuda", dtype=torch.float16))
            bnb_cuda_ok = bool(torch.isfinite(y).all().detach().cpu())
        except Exception as exc:
            failed.append(f"bitsandbytes CUDA backend check failed: {exc}")
    else:
        failed.append("torch.cuda.is_available() is false")

    nvidia_smi = run_command(["nvidia-smi"])
    nvcc = run_command(["nvcc", "--version"])
    checks = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_tensor_operation_ok": cuda_tensor_ok,
        "bitsandbytes_cuda_backend_ok": bnb_cuda_ok,
        "environment_ok": torch.cuda.is_available() and cuda_tensor_ok and bnb_cuda_ok and all(packages[p]["installed"] for p in packages),
    }
    if not checks["environment_ok"] and not failed:
        failed.append("one or more environment checks failed")
    return (
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "os": platform.platform(),
            "python": sys.version,
            "packages": packages,
            "torch": {
                "version": getattr(torch, "__version__", ""),
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "cuda_available": torch.cuda.is_available(),
                "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpus": cuda_devices,
            },
            "nvidia_smi": nvidia_smi,
            "nvcc": nvcc,
            "checks": checks,
            "failed_checks": failed,
        },
        runtime,
    )


def base_environment_report(packages: dict[str, Any], failed: list[str]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "torch": {"cuda_available": False, "gpu_count": 0, "gpus": []},
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "checks": {
            "torch_cuda_available": False,
            "cuda_tensor_operation_ok": False,
            "bitsandbytes_cuda_backend_ok": False,
            "environment_ok": False,
        },
        "failed_checks": failed,
    }


def load_nf4_model(cfg: dict[str, Any], torch: Any, transformers: Any, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    before = gpu_memory(torch)
    quant_cfg = cfg["quantization"]
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    after = gpu_memory(torch)
    quant_effective = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_quantized", False))
    class_counts = Counter(type(param).__name__ for _, param in model.named_parameters())
    device_counts = Counter(str(param.device) for _, param in model.named_parameters())
    footprint = int(model.get_memory_footprint()) if hasattr(model, "get_memory_footprint") else None
    report = {
        "model": cfg["base_model"],
        "memory_before": before,
        "memory_after": after,
        "memory_footprint_bytes": footprint,
        "effective_quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_use_double_quant": bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            "bnb_4bit_compute_dtype": str(compute_dtype),
        },
        "parameter_class_counts": dict(class_counts),
        "parameter_device_counts": dict(device_counts),
        "checks": {
            "model_loaded": True,
            "quantization_config_present": hasattr(model, "quantization_config"),
            "effective_4bit_or_quantized_flag": quant_effective,
            "reasonable_memory_footprint": footprint is None or footprint < 10 * 1024**3,
            "device_placement_has_cuda": any("cuda" in device for device in device_counts),
        },
    }
    report["checks"]["nf4_load_ok"] = all(report["checks"].values())
    return model, report


def attach_lora(model: Any, cfg: dict[str, Any], peft: Any) -> tuple[Any, dict[str, Any]]:
    matched = Counter()
    for name, _module in model.named_modules():
        suffix = name.split(".")[-1]
        if suffix in TARGET_MODULES:
            matched[suffix] += 1
    if hasattr(peft, "prepare_model_for_kbit_training"):
        model = peft.prepare_model_for_kbit_training(model)
    lora_cfg = cfg["lora"]
    peft_config = peft.LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = peft.get_peft_model(model, peft_config)
    total_params = 0
    trainable_params = 0
    non_lora_trainable = []
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
            if "lora_" not in name:
                non_lora_trainable.append(name)
    checks = {
        "all_target_module_types_matched": all(matched[target] > 0 for target in TARGET_MODULES),
        "no_zero_matched_modules": sum(matched.values()) > 0,
        "lora_params_trainable": trainable_params > 0,
        "base_weights_frozen": not non_lora_trainable,
    }
    checks["lora_attach_ok"] = all(checks.values())
    return (
        model,
        {
            "target_modules": list(TARGET_MODULES),
            "matched_module_counts": dict(matched),
            "trainable_param_summary": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percentage": round(100.0 * trainable_params / total_params, 6) if total_params else 0.0,
                "non_lora_trainable_params_sample": non_lora_trainable[:20],
            },
            "checks": checks,
        },
    )


def runtime_tokenization_audit(records: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    errors = []
    for record in records:
        example, detail = build_assistant_only_example(record, tokenizer, max_length=max_length, return_detail=True)
        labels = example["labels"]
        supervised = [label for label in labels if label != -100]
        pad_loss_count = sum(1 for token_id, label in zip(example["input_ids"], labels) if token_id == tokenizer.pad_token_id and label != -100)
        row = {
            "ticket_id": record.get("ticket_id", ""),
            "split": record.get("split", ""),
            "input_variant": record.get("input_variant", ""),
            "prompt_tokens": detail["prompt_tokens"],
            "target_tokens": detail["target_tokens"],
            "input_tokens": len(example["input_ids"]),
            "supervised_tokens": len(supervised),
            "ignored_tokens": sum(1 for label in labels if label == -100),
            "all_labels_ignored": len(supervised) == 0,
            "prompt_tokens_masked": all(label == -100 for label in labels[: detail["prompt_tokens_after_truncation"]]),
            "assistant_target_kept": detail["target_tokens_after_truncation"] > 0,
            "target_truncated": detail["target_truncated"],
            "prompt_truncated": detail["prompt_truncated"],
            "pad_loss_count": pad_loss_count,
            "status": "pass",
            "failure_reason": "",
        }
        failures = []
        if row["all_labels_ignored"]:
            failures.append("all labels are -100")
        if not row["prompt_tokens_masked"]:
            failures.append("prompt token included in assistant-only loss")
        if not row["assistant_target_kept"]:
            failures.append("assistant target missing after truncation")
        if row["target_truncated"]:
            failures.append("assistant target truncated")
        if pad_loss_count:
            failures.append("pad token participates in loss")
        if failures:
            row["status"] = "fail"
            row["failure_reason"] = "; ".join(failures)
            errors.extend(failures)
        rows.append(row)
    report = {
        "passed": not errors,
        "loss_policy": "assistant_only_loss",
        "relation_to_literature": "Uses standard next-token SFT objective but masks prompt tokens to ensure only assistant assignee target contributes loss.",
        "records_audited": len(rows),
        "failures": sum(1 for row in rows if row["status"] == "fail"),
        "errors": sorted(set(errors)),
        "tokenizer": {
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "max_sequence_length": max_length,
        },
    }
    return rows, report


def smoke_train_model(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    cfg: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    output_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    max_length = int(cfg["sft"]["max_seq_length"])
    train_examples = [build_assistant_only_example(record, tokenizer, max_length=max_length) for record in train_records]
    val_examples = [build_assistant_only_example(record, tokenizer, max_length=max_length) for record in validation_records]
    collator = AssistantOnlyCollator(tokenizer, torch)
    loader = DataLoader(train_examples, batch_size=1, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=float(cfg["sft"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(1, max_steps))
    metrics = []
    errors = []
    model.train()
    iterator = iter(loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            errors.append(f"non-finite loss at step {step}")
            break
        loss.backward()
        grad_norm = grad_norm_value(model, torch)
        if not math.isfinite(grad_norm):
            errors.append(f"non-finite gradient norm at step {step}")
            break
        optimizer.step()
        scheduler.step()
        val_loss = validation_loss(model, val_examples[: min(4, len(val_examples))], collator, torch)
        memory = gpu_memory(torch)
        metrics.append(
            {
                "step": step,
                "train_loss": round(float(loss.detach().cpu()), 8),
                "validation_loss": round(val_loss, 8),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": round(grad_norm, 8),
                "gpu_allocated_gb": memory["allocated_gb"],
                "gpu_reserved_gb": memory["reserved_gb"],
                "gpu_peak_allocated_gb": memory["max_allocated_gb"],
            }
        )
    write_csv(output_dir / "smoke_training_metrics.csv", metrics, list(metrics[0].keys()) if metrics else ["step"])
    checkpoint_dir = output_dir / "smoke_checkpoint" / "checkpoint-smoke"
    final_dir = output_dir / "smoke_checkpoint" / "final_adapter"
    model.save_pretrained(str(checkpoint_dir))
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    report = {
        "passed": not errors and bool(metrics),
        "errors": errors,
        "steps_completed": len(metrics),
        "loss_finite": not errors and all(math.isfinite(float(row["train_loss"])) for row in metrics),
        "validation_forward_ok": not errors and all(math.isfinite(float(row["validation_loss"])) for row in metrics),
        "checkpoint_dir": str(checkpoint_dir),
        "final_adapter_dir": str(final_dir),
    }
    (output_dir / "smoke_test_report.md").write_text(render_smoke_report(report), encoding="utf-8")
    return report


def reload_and_compare_from_scores(
    *,
    cfg: dict[str, Any],
    adapter_dir: Path,
    prompt: str,
    roster: list[str],
    reference_scores: dict[str, dict[str, float]],
    tokenizer: Any,
    torch: Any,
    transformers: Any,
    peft: Any,
    tolerance: float,
    keep_model: bool,
) -> tuple[dict[str, Any], Any | None]:
    deltas = {}
    kept_model = None
    try:
        reloaded_base, _report = load_nf4_model(cfg, torch, transformers, adapter_dir)
        reloaded = peft.PeftModel.from_pretrained(reloaded_base, str(adapter_dir))
        reloaded.eval()
        actual = score_candidates(prompt, roster, reloaded, tokenizer, torch)
        for candidate in roster:
            deltas[candidate] = abs(reference_scores[candidate]["mean_logprob"] - actual[candidate]["mean_logprob"])
        max_delta = max(deltas.values()) if deltas else math.inf
        passed = max_delta <= tolerance
        error = ""
        if keep_model:
            kept_model = reloaded
        else:
            del reloaded
            del reloaded_base
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        max_delta = math.inf
        passed = False
        error = repr(exc)
    return (
        {
            "passed": passed,
            "adapter_dir": str(adapter_dir),
            "tolerance": tolerance,
            "max_abs_mean_logprob_delta": max_delta,
            "candidate_deltas": deltas,
            "error": error,
        },
        kept_model,
    )


def real_candidate_scoring_smoke(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    records: list[dict[str, Any]],
    roster: list[str],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    token_rows = []
    for candidate in roster:
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        ids_eos = tokenizer(candidate + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        token_rows.append(
            {
                "candidate": candidate,
                "token_count": len(ids),
                "token_count_with_eos": len(ids_eos),
                "token_ids": " ".join(str(item) for item in ids),
            }
        )
    write_csv(output_dir / "candidate_token_length_audit.csv", token_rows, list(token_rows[0].keys()))
    length_by_candidate = {row["candidate"]: int(row["token_count_with_eos"]) for row in token_rows}
    prediction_rows = []
    agreement = 0
    all_sum_scores = []
    all_mean_scores = []
    all_lengths = []
    warnings = []
    for record in records:
        prompt = build_test_prompt(record, tokenizer)
        scores = score_candidates(prompt, roster, model, tokenizer, torch)
        sorted_sum = sorted(scores.items(), key=lambda item: (-item[1]["sum_logprob"], item[0]))
        sorted_mean = sorted(scores.items(), key=lambda item: (-item[1]["mean_logprob"], item[0]))
        if sorted_sum[0][0] == sorted_mean[0][0]:
            agreement += 1
        for candidate, values in scores.items():
            all_sum_scores.append(values["sum_logprob"])
            all_mean_scores.append(values["mean_logprob"])
            all_lengths.append(length_by_candidate[candidate])
        for method, ranked in (("sum_logprob", sorted_sum), ("mean_token_logprob", sorted_mean)):
            top10 = ranked[:10]
            prediction_rows.append(
                {
                    "ticket_id": record.get("ticket_id", ""),
                    "scoring_method": method,
                    "top1": top10[0][0],
                    "top3": "|".join(candidate for candidate, _score in top10[:3]),
                    "top5": "|".join(candidate for candidate, _score in top10[:5]),
                    "top10": "|".join(candidate for candidate, _score in top10),
                    "ranked_candidates": "|".join(candidate for candidate, _score in top10),
                    "candidate_scores": json.dumps(
                        {candidate: round(values["sum_logprob" if method == "sum_logprob" else "mean_logprob"], 8) for candidate, values in ranked},
                        sort_keys=True,
                    ),
                    "candidate_score_count": len(ranked),
                    "topk_unique": len({candidate for candidate, _score in top10}) == len(top10),
                    "score_finite": all(math.isfinite(values["sum_logprob"]) and math.isfinite(values["mean_logprob"]) for _candidate, values in ranked),
                    "score_sortable": True,
                    "gold_label_used_in_scoring": False,
                    "test_only_assignee_used": False,
                }
            )
    write_csv(output_dir / "real_smoke_llm_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    comparison = [
        {
            "metric": "top1_agreement_sum_vs_mean",
            "value": round(agreement / len(records), 6) if records else 0.0,
            "note": "Agreement between sum logprob and mean-token logprob top1.",
        },
        {
            "metric": "sum_logprob_length_pearson",
            "value": round(pearson(all_sum_scores, all_lengths), 6),
            "note": "Negative correlation indicates longer tokenized candidates are penalized by sum logprob.",
        },
        {
            "metric": "mean_logprob_length_pearson",
            "value": round(pearson(all_mean_scores, all_lengths), 6),
            "note": "Closer to zero is preferred for length-normalized scoring.",
        },
    ]
    write_csv(output_dir / "candidate_scoring_method_comparison.csv", comparison, list(comparison[0].keys()))
    passed = (
        len(records) == 192
        and len(roster) == 60
        and all(row["candidate_score_count"] == 60 for row in prediction_rows)
        and all(row["topk_unique"] for row in prediction_rows)
        and all(row["score_finite"] for row in prediction_rows)
    )
    if len(records) != 192:
        warnings.append(f"scored {len(records)} records; expected 192")
    return {
        "passed": passed,
        "records_scored": len(records),
        "candidate_roster_size": len(roster),
        "recommended_scoring_method": "mean_token_logprob",
        "reason": "Mean token log probability reduces candidate identifier length bias relative to raw sum log probability.",
        "warnings": warnings,
    }


def score_candidates(prompt: str, roster: list[str], model: Any, tokenizer: Any, torch: Any) -> dict[str, dict[str, float]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]
    output = {}
    with torch.no_grad():
        for candidate in roster:
            candidate_ids = tokenizer(candidate + (tokenizer.eos_token or ""), add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
            logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[:, :-1, :]
            labels = input_ids[:, 1:]
            start = max(0, prompt_len - 1)
            log_probs = torch.nn.functional.log_softmax(logits[:, start:, :], dim=-1)
            target_labels = labels[:, start:]
            token_log_probs = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
            output[candidate] = {
                "sum_logprob": float(token_log_probs.sum().detach().cpu()),
                "mean_logprob": float(token_log_probs.mean().detach().cpu()),
                "token_count": int(token_log_probs.numel()),
            }
    return output


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any, torch: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            attention.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": self.torch.tensor(input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(attention, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def build_assistant_only_example(
    record: dict[str, Any], tokenizer: Any, *, max_length: int, return_detail: bool = False
) -> Any:
    prompt = render_prompt(record["messages"][:2], tokenizer)
    completion = record["messages"][2]["content"].strip() + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    original_prompt_len = len(prompt_ids)
    original_target_len = len(completion_ids)
    target_truncated = False
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1] + [tokenizer.eos_token_id]
        prompt_ids = []
        target_truncated = True
    elif len(prompt_ids) + len(completion_ids) > max_length:
        prompt_ids = prompt_ids[-(max_length - len(completion_ids)) :]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    example = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
    if not return_detail:
        return example
    return example, {
        "prompt_tokens": original_prompt_len,
        "target_tokens": original_target_len,
        "prompt_tokens_after_truncation": len(prompt_ids),
        "target_tokens_after_truncation": len(completion_ids),
        "prompt_truncated": len(prompt_ids) < original_prompt_len,
        "target_truncated": target_truncated,
    }


def render_prompt(prompt_messages: list[dict[str, str]], tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return f"{prompt_messages[0]['content'].strip()}\n\n{prompt_messages[1]['content'].strip()}\n"


def build_test_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    title = str(row.get("title") or "").strip()
    description = sanitize_description(str(row.get("description") or ""), max_chars=4000)
    messages = [
        {"role": "system", "content": "You are an expert bug triager."},
        {
            "role": "user",
            "content": (
                "Below is an issue. Suggest the single best developer to resolve it.\n\n"
                f"### Issue:\nTitle: {title or '(empty)'}\nDescription: {description or '(empty)'}\n\n### Assignee:"
            ),
        },
    ]
    return render_prompt(messages, tokenizer)


def sanitize_description(value: str, *, max_chars: int) -> str:
    text = html.unescape(value or "")
    text = HTML_TAG_RE.sub(" ", text)
    output_lines = []
    for line in text.splitlines():
        if ORIGINAL_MESSAGE_RE.search(line):
            break
        if QUOTE_LINE_RE.search(line):
            continue
        output_lines.append(line)
    text = WHITESPACE_RE.sub(" ", "\n".join(output_lines)).strip()
    if len(text) <= max_chars:
        return text
    salient = []
    for sentence in re.split(r"(?<=[.!?])\s+|\s{2,}", text):
        if LOG_LINE_RE.search(sentence):
            salient.append(sentence.strip())
    prefix = text[: max(1000, max_chars // 2)].strip()
    suffix_budget = max_chars - len(prefix) - 80
    salient_text = " ".join(salient)
    if suffix_budget > 0 and salient_text:
        return f"{prefix}\n\n[TRUNCATED; preserved salient error/log lines]\n{salient_text[:suffix_budget].strip()}".strip()
    return f"{text[: max_chars - 40].strip()}\n\n[TRUNCATED]".strip()


def validation_loss(model: Any, examples: list[dict[str, Any]], collator: AssistantOnlyCollator, torch: Any) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for example in examples:
            batch = {key: value.to(model.device) for key, value in collator([example]).items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else math.inf


def grad_norm_value(model: Any, torch: Any) -> float:
    total = torch.zeros([], device=model.device)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().norm(2) ** 2
    return float(torch.sqrt(total).detach().cpu())


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_memory(torch: Any) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0, "max_reserved_gb": 0.0}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 6),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 6),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 6),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 6),
    }


def write_blocked_artifacts(output_dir: Path, gate_state: dict[str, bool], reasons: list[str]) -> None:
    write_json(output_dir / "model_load_report.json", {"checks": {"nf4_load_ok": False}, "status": "blocked", "reasons": reasons})
    write_json(output_dir / "lora_attach_report.json", {"checks": {"lora_attach_ok": False}, "status": "blocked", "reasons": reasons})
    write_csv(output_dir / "runtime_tokenization_audit.csv", [], [])
    write_json(output_dir / "label_mask_audit.json", {"passed": False, "status": "blocked", "errors": reasons})
    write_json(output_dir / "checkpoint_reload_report.json", {"passed": False, "status": "blocked", "errors": reasons})
    write_json(output_dir / "adapter_reload_report.json", {"passed": False, "status": "blocked", "errors": reasons})
    write_csv(output_dir / "candidate_token_length_audit.csv", [], [])
    write_csv(output_dir / "candidate_scoring_method_comparison.csv", [], [])
    write_csv(output_dir / "real_smoke_llm_predictions.csv", [], [])
    write_full_gate(output_dir, gate_state, reasons)
    write_remote_summary(output_dir, gate_state, {}, {}, {}, {"passed": False, "errors": reasons}, {"passed": False, "warnings": reasons})


def write_full_gate(output_dir: Path, gate_state: dict[str, bool], warnings: list[str]) -> None:
    failed = [name for name in GATE_CHECKS if not gate_state.get(name)]
    write_json(
        output_dir / "full_training_gate.json",
        {
            "passed": not failed,
            "checks": {name: bool(gate_state.get(name)) for name in GATE_CHECKS},
            "failed_checks": failed,
            "warnings": warnings,
            "policy": "Full training is forbidden until every gate check passes on remote CUDA.",
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def write_remote_summary(
    output_dir: Path,
    gate_state: dict[str, bool],
    model_report: dict[str, Any],
    lora_report: dict[str, Any],
    mask_report: dict[str, Any],
    train_result: dict[str, Any],
    inference_result: dict[str, Any],
) -> None:
    summary = {
        "phase": "phase5e_remote_smoke_execution_gate",
        "reproduction_role": "literature_aligned_primary_reproduction",
        "primary_input_variant": PRIMARY_VARIANT,
        "metadata_enriched_adapted_variant": ADAPTED_METADATA_VARIANT,
        "exact_reproduction_completed": False,
        "gate_checks": gate_state,
        "full_training_gate_passed": all(gate_state.get(name) for name in GATE_CHECKS),
        "safe_to_start_full_training": all(gate_state.get(name) for name in GATE_CHECKS),
        "nf4_effective": model_report.get("checks", {}).get("nf4_load_ok", False),
        "lora_attached": lora_report.get("checks", {}).get("lora_attach_ok", False),
        "assistant_only_loss_mask_ok": mask_report.get("passed", False),
        "smoke_loss_finite": train_result.get("loss_finite", False),
        "candidate_scoring_completed": inference_result.get("passed", False),
        "recommended_candidate_scoring": inference_result.get("recommended_scoring_method", "mean_token_logprob"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output_dir / "remote_smoke_execution_summary.json", summary)
    lines = [
        "# Phase 5E Remote Smoke Execution Summary",
        "",
        f"- Primary run role: `{summary['reproduction_role']}`",
        f"- Primary input variant: `{PRIMARY_VARIANT}`",
        f"- Metadata adapted variant: `{ADAPTED_METADATA_VARIANT}`",
        f"- Full training gate passed: `{summary['full_training_gate_passed']}`",
        f"- Safe to start full training: `{summary['safe_to_start_full_training']}`",
        f"- Recommended scoring method: `{summary['recommended_candidate_scoring']}`",
        "",
        "Exact reproduction is still `false` until full target training and inference are completed.",
    ]
    (output_dir / "remote_smoke_execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_gate(output_dir: Path, gate_state: dict[str, bool], warnings: list[str]) -> None:
    payload = {
        "output_dir": str(output_dir),
        "passed": all(gate_state.get(name) for name in GATE_CHECKS),
        "checks": gate_state,
        "failed_checks": [name for name in GATE_CHECKS if not gate_state.get(name)],
        "warnings": warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def render_remote_environment_report(report: dict[str, Any]) -> str:
    gpus = report.get("torch", {}).get("gpus", [])
    return "\n".join(
        [
            "# Remote Environment Runtime Report",
            "",
            f"- OS: `{report.get('os')}`",
            f"- Python: `{report.get('python')}`",
            f"- CUDA available: `{report.get('torch', {}).get('cuda_available')}`",
            f"- GPU count: `{report.get('torch', {}).get('gpu_count')}`",
            f"- GPUs: `{gpus}`",
            f"- Environment OK: `{report.get('checks', {}).get('environment_ok')}`",
            f"- Failed checks: `{report.get('failed_checks')}`",
        ]
    ) + "\n"


def render_smoke_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Smoke Training Report",
            "",
            f"- Passed: `{report['passed']}`",
            f"- Steps completed: `{report['steps_completed']}`",
            f"- Loss finite: `{report['loss_finite']}`",
            f"- Validation forward OK: `{report['validation_forward_ok']}`",
            f"- Errors: `{report['errors']}`",
        ]
    ) + "\n"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else 0.0


def run_command(command: list[str]) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "returncode": None, "stdout": "", "stderr": "command not found"}
    completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=15)
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:8000],
        "stderr": completed.stderr.strip()[:8000],
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


if __name__ == "__main__":
    main()
