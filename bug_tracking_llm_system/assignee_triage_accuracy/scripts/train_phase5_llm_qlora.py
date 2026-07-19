from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote CUDA Phase 5 DeepSeek QLoRA SFT training.")
    parser.add_argument("--config", type=Path, required=True, help="Path to Phase 5 llm_training_config.json.")
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--validation-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    cfg = read_json(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config_snapshot.json", {"args": vars_to_json(args), "config": cfg})

    train_jsonl = args.train_jsonl or Path(cfg["sft"]["train_jsonl"])
    validation_jsonl = args.validation_jsonl or Path(cfg["sft"]["validation_jsonl"])
    model_name = args.model_name or cfg["base_model"]

    torch, transformers, peft = import_training_libraries()
    set_seed(int(cfg["seed"]), torch)
    metadata = collect_runtime_metadata(torch)
    write_json(output_dir / "runtime_metadata.json", metadata)
    enforce_cuda_for_exact_qlora(torch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = cfg["quantization"]
    compute_dtype = choose_compute_dtype(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(quant_cfg["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = cfg["lora"]
    if hasattr(peft, "prepare_model_for_kbit_training"):
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    peft_config = peft.LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_length = int(cfg["sft"]["max_seq_length"])
    train_records = read_jsonl(train_jsonl)
    validation_records = read_jsonl(validation_jsonl)
    train_dataset = AssistantOnlyDataset(train_records, tokenizer, max_length=max_length)
    validation_dataset = AssistantOnlyDataset(validation_records, tokenizer, max_length=max_length)
    collator = AssistantOnlyCollator(tokenizer, torch)

    max_steps = args.max_steps
    if args.smoke and max_steps is None:
        max_steps = 20
    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["seed"]),
        "num_train_epochs": int(cfg["sft"]["num_train_epochs"]),
        "max_steps": max_steps if max_steps is not None else -1,
        "learning_rate": float(cfg["sft"]["learning_rate"]),
        "per_device_train_batch_size": int(cfg["sft"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(cfg["sft"]["gradient_accumulation_steps"]),
        "logging_steps": 1 if args.smoke else 10,
        "eval_steps": 5 if args.smoke else None,
        "save_strategy": "steps" if args.smoke else cfg["sft"].get("save_strategy", "epoch"),
        "save_steps": 10 if args.smoke else 100,
        "save_total_limit": 2,
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    strategy_name = "eval_strategy" if "eval_strategy" in inspect.signature(transformers.TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_name] = "steps" if args.smoke else cfg["sft"].get("evaluation_strategy", "epoch")
    training_args = transformers.TrainingArguments(**training_kwargs)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    eval_metrics = trainer.evaluate()
    write_json(output_dir / "train_result.json", train_result.metrics)
    write_json(output_dir / "validation_metrics.json", eval_metrics)
    write_training_metrics_csv(output_dir / "training_metrics.csv", trainer.state.log_history)
    write_json(
        output_dir / "smoke_success_report.json",
        {
            "smoke": args.smoke,
            "loss_finite": all_finite_losses(trainer.state.log_history),
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "final_adapter": str(output_dir / "final_adapter"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    )


class AssistantOnlyDataset:
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, *, max_length: int) -> None:
        self.examples = [build_assistant_only_example(record, tokenizer, max_length=max_length) for record in records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.torch = torch_module

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad)
            batch_attention.append(feature["attention_mask"] + [0] * pad)
            batch_labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": self.torch.tensor(batch_input_ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(batch_attention, dtype=self.torch.long),
            "labels": self.torch.tensor(batch_labels, dtype=self.torch.long),
        }


def build_assistant_only_example(record: dict[str, Any], tokenizer: Any, *, max_length: int) -> dict[str, list[int]]:
    messages = record["messages"]
    prompt_messages = [messages[0], messages[1]]
    assistant = messages[2]["content"].strip()
    prompt = render_prompt(prompt_messages, tokenizer)
    completion = assistant + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1] + [tokenizer.eos_token_id]
        prompt_ids = []
    elif len(prompt_ids) + len(completion_ids) > max_length:
        prompt_budget = max_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def render_prompt(prompt_messages: list[dict[str, str]], tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    system = prompt_messages[0]["content"].strip()
    user = prompt_messages[1]["content"].strip()
    return f"{system}\n\n{user}\n"


def import_training_libraries() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        import peft
    except Exception as exc:  # pragma: no cover - remote environment check
        raise SystemExit(f"Missing training dependency: {exc}") from exc
    return torch, transformers, peft


def enforce_cuda_for_exact_qlora(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for exact DeepSeek 8B NF4 QLoRA reproduction.")


def choose_compute_dtype(torch: Any, configured: str) -> Any:
    if configured.lower() == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict[str, Any]:
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


def vars_to_json(args: argparse.Namespace) -> dict[str, Any]:
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def collect_runtime_metadata(torch: Any) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "python": sys.version,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "gpu": gpu,
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }


def write_training_metrics_csv(path: Path, log_history: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in log_history for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in log_history:
            writer.writerow({field: row.get(field, "") for field in fields})


def all_finite_losses(log_history: list[dict[str, Any]]) -> bool:
    for row in log_history:
        for key in ("loss", "eval_loss"):
            if key in row:
                value = float(row[key])
                if value != value or value in (float("inf"), float("-inf")):
                    return False
    return True


if __name__ == "__main__":
    main()
