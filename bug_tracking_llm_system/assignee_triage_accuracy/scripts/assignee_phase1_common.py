from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = EVAL_ROOT.parent
SRC_ROOT = SYSTEM_ROOT / "src"
PAPER_ROOT = EVAL_ROOT / "paper_grade"
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "processed"
DEFAULT_RAW_PATH = PAPER_ROOT / "data" / "raw" / "bmo_paper_2024_3k_raw.jsonl"
DEFAULT_PHASE1_ROOT = EVAL_ROOT / "phase1_reliability"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402


GENERIC_ASSIGNEE_RE = re.compile(
    r"(nobody|unassigned|triage|inbox|default|bugzilla|bugs@|noreply|do-not-reply|disabled)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_component(value: Any) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def normalize_assignee(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", clean(value).lower())).strip()


def parse_date(value: Any) -> datetime | None:
    text = clean(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_raw_record(row: dict[str, Any]) -> dict[str, Any] | None:
    bug_id = clean(row.get("id"))
    title = clean(row.get("summary"))
    assignee = clean(row.get("assigned_to")).lower()
    created_at = clean(row.get("creation_time"))
    if not bug_id or not title or not assignee or not created_at or parse_date(created_at) is None:
        return None
    return {
        "ticket_id": f"bmo_{bug_id}",
        "title": title,
        "description": clean(row.get("description")),
        "product": clean(row.get("product")).lower() or "unknown",
        "component": clean(row.get("component")).lower() or "unknown",
        "priority": clean(row.get("priority")) or "P3",
        "severity": clean(row.get("severity")),
        "status": clean(row.get("status")),
        "resolution": clean(row.get("resolution")),
        "assignee": assignee,
        "created_at": created_at,
        "last_change_time": clean(row.get("last_change_time")),
        "dupe_of": clean(row.get("dupe_of")),
        "source": "bugzilla_mozilla",
    }


def load_normalized_raw_records(raw_path: Path, *, keep_generic_assignees: bool = False) -> list[dict[str, Any]]:
    records = [record for record in (normalize_raw_record(row) for row in read_jsonl(raw_path)) if record]
    if not keep_generic_assignees:
        records = [row for row in records if not GENERIC_ASSIGNEE_RE.search(row["assignee"])]
    records.sort(key=lambda row: (row["created_at"], row["ticket_id"]))
    return records


def temporal_split(
    records: list[dict[str, Any]], train_ratio: float = 0.80, validation_ratio: float = 0.10
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 3:
        raise SystemExit(f"Need at least 3 records for temporal split, got {len(records)}")
    train_end = max(1, min(len(records) - 2, int(len(records) * train_ratio)))
    validation_end = max(train_end + 1, min(len(records) - 1, train_end + int(len(records) * validation_ratio)))
    return records[:train_end], records[train_end:validation_end], records[validation_end:]


def build_phase1_label_map(rows_by_split: list[list[dict[str, Any]]], train_rows: list[dict[str, Any]]) -> dict[str, str]:
    train_assignees = sorted({normalize_assignee(row.get("assignee")) for row in train_rows if normalize_assignee(row.get("assignee"))})
    label_map = {assignee: f"dev_{index:04d}" for index, assignee in enumerate(train_assignees, start=1)}
    non_train = sorted(
        {
            normalize_assignee(row.get("assignee"))
            for rows in rows_by_split
            for row in rows
            if normalize_assignee(row.get("assignee")) and normalize_assignee(row.get("assignee")) not in label_map
        }
    )
    for index, assignee in enumerate(non_train, start=1):
        label_map[assignee] = f"unseen_{index:04d}"
    return label_map


def anonymize_rows(rows: list[dict[str, Any]], label_map: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copy = dict(row)
        copy["raw_assignee_hash"] = sha256_text(normalize_assignee(copy.get("assignee")))[:12]
        copy["assignee"] = label_map[normalize_assignee(copy.get("assignee"))]
        output.append(copy)
    return output


def build_prefilter_views(raw_path: Path, min_train_assignee_count: int = 10) -> dict[str, Any]:
    records = load_normalized_raw_records(raw_path)
    pre_train_raw, pre_validation_raw, pre_test_raw = temporal_split(records)
    train_counts_raw = Counter(normalize_assignee(row.get("assignee")) for row in pre_train_raw)
    frequent_train_raw = {
        assignee for assignee, count in train_counts_raw.items() if assignee and count >= min_train_assignee_count
    }
    label_map = build_phase1_label_map([pre_train_raw, pre_validation_raw, pre_test_raw], pre_train_raw)
    pre_train = anonymize_rows(pre_train_raw, label_map)
    pre_validation = anonymize_rows(pre_validation_raw, label_map)
    pre_test = anonymize_rows(pre_test_raw, label_map)
    train_counts = Counter(row["assignee"] for row in pre_train)
    frequent_train = {label_map[assignee] for assignee in frequent_train_raw}
    return {
        "records": anonymize_rows(records, label_map),
        "pre_train": pre_train,
        "pre_validation": pre_validation,
        "pre_test": pre_test,
        "pre_train_raw": pre_train_raw,
        "pre_validation_raw": pre_validation_raw,
        "pre_test_raw": pre_test_raw,
        "train_counts": train_counts,
        "train_counts_raw": train_counts_raw,
        "train_assignees": set(train_counts),
        "frequent_train_assignees": frequent_train,
        "label_map": label_map,
        "min_train_assignee_count": min_train_assignee_count,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def build_history_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_counts: dict[str, Counter[str]] = defaultdict(Counter)
    product_component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in rows:
        assignee = normalize_assignee(row.get("assignee"))
        if not assignee or assignee == "manual_triage":
            continue
        component = normalize_component(row.get("component"))
        product = normalize_component(row.get("product"))
        component_counts[component][assignee] += 1
        product_counts[product][assignee] += 1
        product_component_counts[f"{product}::{component}"][assignee] += 1
        global_counts[assignee] += 1
    return {
        "component_counts": dict(component_counts),
        "product_counts": dict(product_counts),
        "product_component_counts": dict(product_component_counts),
        "global_counts": global_counts,
    }


def current_triager_candidates(
    result: dict[str, Any],
    row: dict[str, Any],
    counts: dict[str, Any],
    roster: set[str],
    top_k: int,
) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        assignee = normalize_assignee(value)
        if not assignee or assignee in candidates:
            return
        if assignee == "manual_triage" or (roster and assignee not in roster):
            return
        candidates.append(assignee)

    for assignee in result.get("ranked_candidates", []):
        add(assignee)
    return candidates[:top_k]


def prediction_margin(candidate_scores: dict[str, Any]) -> float | None:
    values = sorted((float(value) for value in candidate_scores.values()), reverse=True)
    if not values:
        return None
    if len(values) == 1:
        return 1.0
    best = values[0]
    if best <= 0:
        return 0.0
    return round((best - values[1]) / best, 6)


def evaluate_current_triager(
    *,
    train_path: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    roster: set[str],
    top_k: int = 10,
    bootstrap_samples: int = 500,
    seed: int = 3407,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = PipelineConfig(
        project_root=SYSTEM_ROOT,
        assignee_dataset_path=train_path,
        assignee_top_k=top_k,
        save_checkpoints=False,
    )
    triager = AssigneeTriager(config=config)
    counts = build_history_counts(train_rows)
    predictions = []
    for row in test_rows:
        expected = normalize_assignee(row.get("assignee"))
        result = triager.assign(row, {"predicted_priority": row.get("priority", "P3")})
        candidates = current_triager_candidates(result, row, counts, roster, top_k)
        predicted = candidates[0] if candidates else ""
        rank = rank_of(expected, candidates)
        predictions.append(
            {
                "ticket_id": row.get("ticket_id"),
                "title": row.get("title"),
                "product": normalize_component(row.get("product")),
                "component": normalize_component(row.get("component")),
                "priority": row.get("priority", "P3"),
                "expected_assignee": expected,
                "predicted_assignee": predicted,
                "raw_assignee_hash": row.get("raw_assignee_hash", ""),
                "ranked_candidates": candidates,
                "rank": rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "is_top1_correct": expected == predicted,
                "is_hit_at_3": expected in candidates[:3],
                "is_hit_at_5": expected in candidates[:5],
                "is_hit_at_10": expected in candidates[:10],
                "confidence": result.get("confidence"),
                "margin": prediction_margin(result.get("candidate_scores", {})),
                "reason": result.get("reason"),
                "candidate_scores": result.get("candidate_scores", {}),
            }
        )
    return predictions, build_metrics(predictions, bootstrap_samples=bootstrap_samples, seed=seed)


def rank_of(expected: str, candidates: list[str]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate == expected:
            return index
    return None


def build_metrics(predictions: list[dict[str, Any]], *, bootstrap_samples: int = 500, seed: int = 3407) -> dict[str, Any]:
    total = len(predictions)
    indicators = [1 if row.get("is_top1_correct") else 0 for row in predictions]
    return {
        "rows": total,
        "top1_accuracy": ratio(sum(indicators), total),
        "hit_at_3": ratio(sum(1 for row in predictions if row.get("is_hit_at_3")), total),
        "hit_at_5": ratio(sum(1 for row in predictions if row.get("is_hit_at_5")), total),
        "hit_at_10": ratio(sum(1 for row in predictions if row.get("is_hit_at_10")), total),
        "mrr": ratio(sum(float(row.get("reciprocal_rank", 0.0)) for row in predictions), total),
        "top1_bootstrap_95ci": bootstrap_ci(indicators, bootstrap_samples, seed),
    }


def grouped_metrics(
    predictions: list[dict[str, Any]], group_key: str, *, bootstrap_samples: int = 500, seed: int = 3407
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(group_key) or "unknown")].append(row)
    return {
        group: build_metrics(rows, bootstrap_samples=bootstrap_samples, seed=seed)
        for group, rows in sorted(grouped.items())
    }


def bootstrap_ci(indicators: list[int], samples: int, seed: int) -> dict[str, float]:
    if not indicators:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    estimates = []
    n = len(indicators)
    for _ in range(samples):
        draw = [indicators[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(draw) / n)
    estimates.sort()
    return {
        "low": round(percentile(estimates, 0.025), 6),
        "high": round(percentile(estimates, 0.975), 6),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def ratio(numerator: float, denominator: int) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def rows_to_prediction_csv(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in predictions:
        copy = dict(row)
        copy["ranked_candidates"] = "|".join(row.get("ranked_candidates", []))
        copy["candidate_scores"] = json.dumps(row.get("candidate_scores", {}), ensure_ascii=False, sort_keys=True)
        output.append(copy)
    return output


def prediction_csv_fields(extra: list[str] | None = None) -> list[str]:
    fields = [
        "ticket_id",
        "product",
        "component",
        "priority",
        "expected_assignee",
        "predicted_assignee",
        "rank",
        "reciprocal_rank",
        "is_top1_correct",
        "is_hit_at_3",
        "is_hit_at_5",
        "is_hit_at_10",
        "confidence",
        "margin",
        "ranked_candidates",
        "candidate_scores",
        "reason",
        "title",
    ]
    return fields + [field for field in (extra or []) if field not in fields]


def frequency_bucket(count: int) -> str:
    if count <= 0:
        return "unseen"
    if count == 1:
        return "freq_1"
    if 2 <= count <= 4:
        return "freq_2_4"
    if 5 <= count <= 9:
        return "freq_5_9"
    if 10 <= count <= 19:
        return "freq_10_19"
    return "freq_20_plus"


def ticket_text(row: dict[str, Any]) -> str:
    return " ".join(
        clean(row.get(field))
        for field in ("title", "description", "product", "component", "severity", "priority")
        if clean(row.get(field))
    )


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
