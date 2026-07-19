from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 LLM assignee inference with explicit candidate scoring.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--roster-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--mode", choices=["candidate_scoring", "free_generation", "posthoc_filtered_generation"], default="candidate_scoring")
    parser.add_argument("--no-4bit", action="store_true", help="Disable config-driven 4-bit loading for debugging only.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    cfg = read_json(args.config)
    records = read_jsonl(args.input_jsonl)
    roster = read_roster(args.roster_json)
    if not roster:
        raise SystemExit("Candidate roster is empty.")
    random.seed(args.seed)

    torch, transformers, peft = import_inference_libraries(args.adapter_dir)
    model_name = args.base_model or cfg["base_model"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if not args.no_4bit and cfg.get("quantization", {}).get("load_in_4bit"):
        quant_cfg = cfg["quantization"]
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=choose_inference_dtype(torch),
        )
    else:
        model_kwargs["torch_dtype"] = choose_inference_dtype(torch)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if args.adapter_dir:
        model = peft.PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    rows = []
    for record in records:
        prompt = render_prompt(record["messages"][:2], tokenizer)
        if args.mode == "candidate_scoring":
            scored = score_candidates(prompt, roster, model, tokenizer, torch)
            invalid_output = False
            raw_generation = ""
        else:
            raw_generation = generate_text(prompt, model, tokenizer, torch, max_new_tokens=args.max_new_tokens)
            scored, invalid_output = generation_to_ranked_candidates(raw_generation, roster, args.mode)
        top = scored[: args.top_k]
        expected = record.get("metadata", {}).get("expected_assignee", "")
        rows.append(
            {
                "model_name": model_name,
                "adapter_dir": str(args.adapter_dir or ""),
                "inference_mode": args.mode,
                "ticket_id": record.get("ticket_id", ""),
                "expected_assignee": expected,
                "predicted_assignee": top[0]["candidate"] if top else "",
                "ranked_candidates": "|".join(item["candidate"] for item in top),
                "top1": top[0]["candidate"] if top else "",
                "top3": "|".join(item["candidate"] for item in top[:3]),
                "top5": "|".join(item["candidate"] for item in top[:5]),
                "top10": "|".join(item["candidate"] for item in top[:10]),
                "candidate_scores": json.dumps({item["candidate"]: item["score"] for item in top}, sort_keys=True),
                "candidate_normalized_scores": json.dumps(
                    {item["candidate"]: item.get("normalized_score", item["score"]) for item in top},
                    sort_keys=True,
                ),
                "invalid_output": invalid_output,
                "raw_generation": raw_generation,
                "candidate_roster_size": len(roster),
                "topk_unique": len({item["candidate"] for item in top}) == len(top),
                "score_sortable": scores_are_sorted(top),
                "scores_are_model_logprobs": args.mode == "candidate_scoring",
            }
        )
    write_csv(args.output_csv, rows)


def score_candidates(prompt: str, roster: list[str], model: Any, tokenizer: Any, torch: Any) -> list[dict[str, Any]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]
    scored = []
    with torch.no_grad():
        for candidate in roster:
            candidate_text = candidate + (tokenizer.eos_token or "")
            candidate_ids = tokenizer(candidate_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
            attention = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            start = max(0, prompt_len - 1)
            end = labels.shape[1]
            candidate_labels = labels[:, start:end]
            candidate_log_probs = log_probs[:, start:end, :].gather(2, candidate_labels.unsqueeze(-1)).squeeze(-1)
            score = float(candidate_log_probs.sum().detach().cpu())
            normalized = float(candidate_log_probs.mean().detach().cpu()) if candidate_log_probs.numel() else -math.inf
            scored.append({"candidate": candidate, "score": round(score, 8), "normalized_score": round(normalized, 8)})
    scored.sort(key=lambda item: (-item["normalized_score"], -item["score"], item["candidate"]))
    return scored


def generate_text(prompt: str, model: Any, tokenizer: Any, torch: Any, *, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generation_to_ranked_candidates(text: str, roster: list[str], mode: str) -> tuple[list[dict[str, Any]], bool]:
    normalized = text.strip().lower()
    matched = [candidate for candidate in roster if candidate.lower() == normalized]
    invalid = not matched
    ranked = matched + [candidate for candidate in roster if candidate not in matched]
    scored = [
        {"candidate": candidate, "score": round(1.0 / (index + 1), 8), "normalized_score": round(1.0 / (index + 1), 8)}
        for index, candidate in enumerate(ranked)
    ]
    if mode == "free_generation":
        return scored[:1], invalid
    return scored, invalid


def render_prompt(prompt_messages: list[dict[str, str]], tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    return f"{prompt_messages[0]['content'].strip()}\n\n{prompt_messages[1]['content'].strip()}\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_roster(path: Path) -> list[str]:
    payload = read_json(path)
    if "candidate_roster" in payload:
        values = payload["candidate_roster"]["candidates"]
    elif "candidates" in payload:
        values = payload["candidates"]
    else:
        values = payload
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_inference_libraries(adapter_dir: Path | None) -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - remote environment check
        raise SystemExit(f"Missing inference dependency: {exc}") from exc
    peft = None
    if adapter_dir:
        try:
            import peft as peft_module
        except Exception as exc:  # pragma: no cover - remote environment check
            raise SystemExit(f"Adapter inference requires peft: {exc}") from exc
        peft = peft_module
    return torch, transformers, peft


def choose_inference_dtype(torch: Any) -> Any:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def scores_are_sorted(rows: list[dict[str, Any]]) -> bool:
    return all(
        (rows[index].get("normalized_score", rows[index]["score"]), rows[index]["score"])
        >= (rows[index + 1].get("normalized_score", rows[index + 1]["score"]), rows[index + 1]["score"])
        for index in range(len(rows) - 1)
    )


if __name__ == "__main__":
    main()
