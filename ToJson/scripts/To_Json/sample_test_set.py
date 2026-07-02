"""
從 dataset/processed/unlabeled.jsonl 建立 test.jsonl，
供人工標註使用。
"""
import random
import json

INPUT = "dataset/processed/unlabeled.jsonl"
OUTPUT = "dataset/labeled/test.jsonl"

sample_size = None
random.seed(42)

with open(INPUT, "r", encoding="utf-8") as f:
    data = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print("Skipping invalid JSON line")

samples = data
sample_size = len(samples)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for s in samples:
        record = dict(s)
        record["json_ground_truth"] = {
            "ticket_id": "",
            "title": "",
            "description": "",
            "product": "",
            "severity": "",
            "bug_type": "",
            "component": "",
            "os": "",
            "version": "",
            "priority": "",
            "error_message": "",
            "steps_to_reproduce": [],
            "expected_behavior": "",
            "actual_behavior": "",
            "logs": "",
            "screenshots_text": "",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Prepared {sample_size} bug reports -> test.jsonl (json_ground_truth initialized as empty manual-annotation template)")
