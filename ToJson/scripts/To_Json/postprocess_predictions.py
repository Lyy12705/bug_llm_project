"""
Apply deterministic post-processing to JSON extraction predictions.

This is useful for public issue-tracker datasets where the model may output
natural-language component names such as "managedBuilder", while the official
ground truth uses fixed taxonomy labels such as "cdt-build".
"""

import argparse
import importlib.util
import json
from pathlib import Path


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_extract_module():
    module_path = Path(__file__).with_name("extract_to_json.py")
    spec = importlib.util.spec_from_file_location("extract_to_json_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process predicted_json records.")
    parser.add_argument("--input-records", required=True, help="Input JSONL records used by the model.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL to post-process.")
    parser.add_argument("--output", required=True, help="Output post-processed prediction JSONL.")
    parser.add_argument(
        "--mode",
        choices=["public_eclipse"],
        default="public_eclipse",
        help="Post-processing rule set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_module = load_extract_module()

    records = {rec.get("id"): rec for rec in load_jsonl(args.input_records)}
    predictions = load_jsonl(args.pred)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for pred_record in predictions:
            rec_id = pred_record.get("id")
            source_record = records.get(rec_id)
            if not source_record:
                continue

            updated = dict(pred_record)
            updated["predicted_json"] = extract_module.normalize_prediction(
                pred_record.get("predicted_json") or {},
                rec=source_record,
                postprocess=args.mode,
            )
            updated.setdefault("postprocess_meta", {})
            updated["postprocess_meta"] = {
                **updated["postprocess_meta"],
                "mode": args.mode,
                "input_records": args.input_records,
            }
            out.write(json.dumps(updated, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote post-processed predictions: {written} -> {output_path}")


if __name__ == "__main__":
    main()
