"""
Prepare a public bug-report-to-JSON evaluation set.

Default source:
Eclipse issue report dataset sample from Zenodo
https://zenodo.org/records/15348468

The public dataset already contains structured issue-tracker fields. This script
uses Summary + Description as the unstructured bug_report input, and uses
official tracker fields as json_ground_truth where they match this project:

- Product -> product
- Severity -> severity
- Component -> component
- Op sys / Platform -> os
- Version -> version
- Priority -> priority

Fields not provided as official labels, such as this project's bug_type and
error_message, are omitted from json_ground_truth. Evaluate with
evaluate_results.py --skip-missing-ground-truth.
"""

import argparse
import csv
import json
import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_SOURCE_URL = "https://zenodo.org/records/15348468/files/sample_data.csv?download=1"
DEFAULT_RAW_CSV = "dataset/raw/eclipse_issue_sample.csv"
DEFAULT_PROCESSED_OUT = "dataset/processed/public_bug_reports.jsonl"
DEFAULT_LABELED_OUT = "dataset/labeled/public_test.jsonl"

TARGET_FIELDS = ["product", "severity", "bug_type", "component", "os", "version", "priority", "error_message"]


def clean_text(value: Optional[str]) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_priority(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    upper = raw.upper()
    mapping = {
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "P4",
        "P5": "P5",
        "BLOCKER": "P1",
        "CRITICAL": "P2",
        "MAJOR": "P3",
        "NORMAL": "P3",
        "MINOR": "P4",
        "TRIVIAL": "P5",
    }
    return mapping.get(upper, raw)


def normalize_os(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    lower = raw.lower()
    if lower in {"---", "unspecified", "unknown"}:
        return ""
    if lower in {"all", "all/all"}:
        return "all"
    if "windows" in lower or lower.startswith("win"):
        return "windows"
    if "linux" in lower:
        return "linux"
    if "mac" in lower or "os x" in lower:
        return "macos"
    return raw


def first_value(row: Dict[str, str], *names: str) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def build_allowed_values(rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    values = {
        "product": set(),
        "severity": set(),
        "component": set(),
        "os": set(),
        "version": set(),
        "priority": set(),
    }
    for row in rows:
        product = first_value(row, "Product", "product")
        severity = first_value(row, "Severity", "severity")
        component = first_value(row, "Component", "component")
        version = first_value(row, "Version", "version")
        os_value = normalize_os(first_value(row, "Op sys", "OS", "Platform", "platform"))
        priority = normalize_priority(first_value(row, "Priority", "priority"))
        if product:
            values["product"].add(product)
        if severity:
            values["severity"].add(severity)
        if component:
            values["component"].add(component)
        if version:
            values["version"].add(version)
        if os_value:
            values["os"].add(os_value)
        if priority:
            values["priority"].add(priority)
    return {key: sorted(value_set) for key, value_set in values.items()}


def metadata_lines(row: Dict[str, str], allowed_values: Dict[str, List[str]], include_component: bool) -> List[str]:
    fields = [
        ("Classification", first_value(row, "Classification")),
        ("Product", first_value(row, "Product", "product")),
        ("Platform", first_value(row, "Platform", "platform")),
        ("Operating system", first_value(row, "Op sys", "OS")),
        ("Version", first_value(row, "Version", "version")),
        ("Severity", first_value(row, "Severity", "severity")),
        ("Priority", first_value(row, "Priority", "priority")),
        ("Status", first_value(row, "Status", "status")),
        ("Resolution", first_value(row, "Resolution", "resolution")),
    ]
    if include_component:
        fields.insert(2, ("Component", first_value(row, "Component", "component")))

    lines = [f"{label}: {value}" for label, value in fields if value]
    component_candidates = ", ".join(allowed_values.get("component", []))
    if component_candidates:
        lines.append(f"Component candidates: {component_candidates}")
    return lines


def build_bug_report(
    row: Dict[str, str],
    include_comments: bool,
    input_mode: str,
    allowed_values: Dict[str, List[str]],
    include_component_metadata: bool,
) -> str:
    title = first_value(row, "Summary", "summary", "Title", "title")
    description = first_value(row, "Description", "description", "Body", "body")

    parts = [
        title,
        description,
    ]
    if include_comments:
        parts.append(first_value(row, "Comments", "comments"))

    report_text = clean_text(" ".join(part for part in parts if part))
    if input_mode == "text_only":
        return report_text

    metadata = "\n".join(metadata_lines(row, allowed_values, include_component_metadata))
    return clean_text(
        f"Tracker metadata:\n{metadata}\n\nUser report title:\n{title}\n\nUser report description:\n{description}"
    )


def build_ground_truth(row: Dict[str, str]) -> Dict[str, str]:
    product = first_value(row, "Product", "product")
    severity = first_value(row, "Severity", "severity")
    component = first_value(row, "Component", "component")
    version = first_value(row, "Version", "version")
    os_value = normalize_os(first_value(row, "Op sys", "OS", "Platform", "platform"))
    priority = normalize_priority(first_value(row, "Priority", "priority"))

    ground_truth: Dict[str, str] = {}
    if product:
        ground_truth["product"] = product
    if severity:
        ground_truth["severity"] = severity
    if component:
        ground_truth["component"] = component
    if os_value:
        ground_truth["os"] = os_value
    if version:
        ground_truth["version"] = version
    if priority:
        ground_truth["priority"] = priority
    return ground_truth


def direct_fields_from_ground_truth(ground_truth: Dict[str, str]) -> Dict[str, str]:
    return {field: ground_truth.get(field, "") for field in TARGET_FIELDS}


def normalized_contains(text: str, value: str) -> bool:
    text_norm = clean_text(text).lower()
    value_norm = clean_text(value).lower()
    if not value_norm:
        return False
    return value_norm in text_norm


def observable_ground_truth_fields(bug_report: str, ground_truth: Dict[str, str]) -> List[str]:
    return [
        field
        for field, value in ground_truth.items()
        if value and normalized_contains(bug_report, value)
    ]


def download_if_needed(path: str, url: str, overwrite: bool) -> None:
    target = Path(path)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public dataset sample -> {target}")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        print(f"[WARN] Verified HTTPS download failed: {exc}")
        print("[WARN] Retrying with certificate verification disabled for this public dataset download.")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response:
            target.write_bytes(response.read())


def iter_rows(path: str) -> Iterable[Dict[str, str]]:
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {str(k): (v if v is not None else "") for k, v in row.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a public issue-tracker CSV into this project's JSON evaluation format."
    )
    parser.add_argument("--input-csv", default="", help="Existing CSV to convert. If omitted, downloads the Eclipse sample.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="Download URL for the default public sample.")
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV, help="Where to save the downloaded CSV sample.")
    parser.add_argument("--processed-out", default=DEFAULT_PROCESSED_OUT, help="Output JSONL without labels.")
    parser.add_argument("--labeled-out", default=DEFAULT_LABELED_OUT, help="Output JSONL with json_ground_truth.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum records to keep. Use 0 for no limit.")
    parser.add_argument("--min-chars", type=int, default=40, help="Minimum bug_report length.")
    parser.add_argument("--include-comments", action="store_true", help="Append Comments to the bug_report input.")
    parser.add_argument(
        "--input-mode",
        choices=["text_only", "metadata"],
        default="text_only",
        help="text_only uses Summary+Description; metadata also prepends issue tracker metadata.",
    )
    parser.add_argument(
        "--include-component-metadata",
        action="store_true",
        help="Also include the official Component metadata in bug_report. This is an oracle normalization setting.",
    )
    parser.add_argument("--overwrite-download", action="store_true", help="Redownload the default CSV sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv or args.raw_csv
    if not args.input_csv:
        download_if_needed(input_csv, args.source_url, args.overwrite_download)

    processed_path = Path(args.processed_out)
    labeled_path = Path(args.labeled_out)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_rows(input_csv))
    allowed_values = build_allowed_values(rows)

    kept = 0
    skipped = 0
    comparable_fields = {field: 0 for field in TARGET_FIELDS}

    with open(processed_path, "w", encoding="utf-8") as processed, open(
        labeled_path, "w", encoding="utf-8"
    ) as labeled:
        for index, row in enumerate(rows, 1):
            if args.limit and kept >= args.limit:
                break

            bug_report = build_bug_report(
                row,
                include_comments=args.include_comments,
                input_mode=args.input_mode,
                allowed_values=allowed_values,
                include_component_metadata=args.include_component_metadata,
            )
            ground_truth = build_ground_truth(row)
            issue_id = first_value(row, "ID", "Issue ID", "Issue id", "bug_id") or f"row_{index}"

            if len(bug_report) < args.min_chars or not ground_truth:
                skipped += 1
                continue

            for field in ground_truth:
                comparable_fields[field] += 1

            title = clean_text(first_value(row, "Summary", "summary", "Title", "title"))
            record = {
                "id": f"eclipse_{issue_id}",
                "source": "eclipse_bugzilla_zenodo",
                "bug_report": bug_report,
                "meta": {
                    "title": title,
                    "direct_fields": direct_fields_from_ground_truth(ground_truth),
                    "public_dataset": {
                        "name": "Eclipse issue report dataset",
                        "url": "https://zenodo.org/records/15348468",
                        "ground_truth_fields": sorted(ground_truth.keys()),
                        "input_mode": args.input_mode,
                        "allowed_values": allowed_values,
                        "observable_ground_truth_fields": observable_ground_truth_fields(bug_report, ground_truth),
                        "note": "bug_type and error_message are not official labels in this dataset.",
                    },
                    "source_fields": {
                        "classification": first_value(row, "Classification"),
                        "platform": first_value(row, "Platform", "platform"),
                        "product": first_value(row, "Product", "product"),
                        "severity": first_value(row, "Severity", "severity"),
                        "status": first_value(row, "Status", "status"),
                        "resolution": first_value(row, "Resolution", "resolution"),
                    },
                },
                "json_ground_truth": ground_truth,
            }

            processed_record = dict(record)
            processed_record["json_ground_truth"] = None

            processed.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
            labeled.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Input CSV       : {input_csv}")
    print(f"Processed JSONL : {processed_path}")
    print(f"Labeled JSONL   : {labeled_path}")
    print(f"Kept records    : {kept}")
    print(f"Skipped records : {skipped}")
    print("Comparable official fields:")
    for field, count in comparable_fields.items():
        if count:
            print(f"  - {field}: {count}")


if __name__ == "__main__":
    main()
