"""Fetch Eclipse Bugzilla reports for P1-P5 priority prediction.

此程式依 priority 分批抓取 Eclipse Bugzilla metadata，並額外抓第一則
comment 作為 description。只使用第一則 comment 是為了避免後續修復討論造成資料洩漏。
"""

import argparse
import os
import time
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://bugs.eclipse.org/bugs/rest/bug"
COMMENT_URL_TEMPLATE = "https://bugs.eclipse.org/bugs/rest/bug/{bug_id}/comment"
XML_URL = "https://bugs.eclipse.org/bugs/show_bug.cgi"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "eclipse_bugzilla_raw.csv")

PRIORITIES = ["P1", "P2", "P3", "P4", "P5"]
INCLUDE_FIELDS = [
    "id",
    "summary",
    "priority",
    "severity",
    "product",
    "component",
    "creator",
    "creation_time",
    "status",
    "resolution",
    "dupe_of",
]


def ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def get_with_retries(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 2,
    retry_sleep: float = 1.0,
) -> requests.Response:
    # Bugzilla 偶爾會回 5xx；retry 可避免長時間抓資料中途失敗。
    last_error: requests.RequestException | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code >= 500 and attempt < retries:
                time.sleep(retry_sleep)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("unreachable retry state")


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 2,
    retry_sleep: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request_params = params.copy() if params else None
        if attempt > 0 and request_params is not None:
            request_params["_retry"] = f"{attempt}-{int(time.time() * 1000)}"
        try:
            response = get_with_retries(session, url, params=request_params, retries=0, retry_sleep=retry_sleep)
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("unreachable JSON retry state")


def fetch_first_comment(
    session: requests.Session,
    bug_id: int,
    retries: int = 2,
    retry_sleep: float = 1.0,
) -> tuple[str, str, str]:
    # 優先使用 REST comment API；若失敗，再用 XML endpoint 抓第一則描述。
    rest_error = ""
    try:
        data = request_json(
            session,
            COMMENT_URL_TEMPLATE.format(bug_id=bug_id),
            retries=retries,
            retry_sleep=retry_sleep,
        )
        comments = data.get("bugs", {}).get(str(bug_id), {}).get("comments", [])
        if comments:
            first_comment = min(comments, key=lambda item: item.get("count", 0))
            text = first_comment.get("raw_text") or first_comment.get("text") or ""
            return text, "rest", ""
    except (requests.RequestException, ValueError) as exc:
        rest_error = f"{type(exc).__name__}: {exc}"

    # Eclipse currently returns an empty body for /rest/bug/<id>/comment on some
    # public requests. The XML endpoint is a stable Bugzilla fallback and exposes
    # the original report text as the long_desc with comment_count=0.
    try:
        response = get_with_retries(
            session,
            XML_URL,
            params={"ctype": "xml", "id": bug_id},
            retries=retries,
            retry_sleep=retry_sleep,
        )
        root = ET.fromstring(response.content)
        first_long_desc = root.find(".//long_desc[comment_count='0']/thetext")
        if first_long_desc is None:
            first_long_desc = root.find(".//long_desc/thetext")
        text = first_long_desc.text if first_long_desc is not None and first_long_desc.text else ""
        return text, "xml", ""
    except (requests.RequestException, ET.ParseError) as exc:
        xml_error = f"{type(exc).__name__}: {exc}"
        error = xml_error if not rest_error else f"REST={rest_error} | XML={xml_error}"
        return "", "missing", error


def build_search_params(args: argparse.Namespace, priority: str, offset: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "classification": args.classification,
        "priority": priority,
        "include_fields": ",".join(INCLUDE_FIELDS),
        "limit": args.limit,
        "offset": offset,
    }
    if args.product:
        params["product"] = args.product
    if args.component:
        params["component"] = args.component
    if args.status:
        params["status"] = args.status
    if args.resolution:
        params["resolution"] = args.resolution
    return params


def fetch_priority_records(session: requests.Session, args: argparse.Namespace, priority: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = args.offset

    with tqdm(total=args.target_per_priority, desc=f"Eclipse {priority}") as pbar:
        while len(records) < args.target_per_priority:
            try:
                data = request_json(
                    session,
                    BASE_URL,
                    params=build_search_params(args, priority, offset),
                    retries=args.search_retries,
                    retry_sleep=args.retry_sleep,
                )
            except (requests.RequestException, ValueError) as exc:
                tqdm.write(f"warning: stopping {priority} at offset {offset}; search request failed: {exc}")
                break
            bugs = data.get("bugs", [])
            if not bugs:
                break

            remaining = args.target_per_priority - len(records)
            for bug in bugs[:remaining]:
                bug = bug.copy()
                if args.skip_comments:
                    description, description_source, description_error = "", "skipped", ""
                else:
                    description, description_source, description_error = fetch_first_comment(
                        session,
                        int(bug["id"]),
                        retries=args.comment_retries,
                        retry_sleep=args.retry_sleep,
                    )
                bug["description"] = description
                bug["description_source"] = description_source
                bug["description_error"] = description_error
                if description_error and args.log_description_errors:
                    tqdm.write(f"warning: bug {bug['id']} description unavailable: {description_error}")
                bug["fetch_group"] = f"quota_{priority}"
                bug["fetch_rank"] = PRIORITIES.index(priority) + 1
                records.append(bug)
                pbar.update(1)
                if args.sleep > 0 and not args.skip_comments:
                    time.sleep(args.sleep)

            offset += len(bugs)
            if len(bugs) < args.limit:
                break
            if args.page_sleep > 0:
                time.sleep(args.page_sleep)

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Eclipse Bugzilla reports for DRONE/GRAY priority prediction.")
    parser.add_argument("--classification", default="Eclipse Project", help="Bugzilla classification to fetch.")
    parser.add_argument("--product", action="append", default=None, help="Optional product filter. Can be passed more than once.")
    parser.add_argument("--component", action="append", default=None, help="Optional component filter. Can be passed more than once.")
    parser.add_argument("--status", action="append", default=None, help="Optional status filter. Defaults to Bugzilla search default.")
    parser.add_argument("--resolution", action="append", default=None, help="Optional resolution filter.")
    parser.add_argument("--target-per-priority", type=int, default=2000, help="Target records per P1~P5 priority.")
    parser.add_argument("--limit", type=int, default=100, help="Bugzilla REST page size.")
    parser.add_argument("--offset", type=int, default=0, help="Starting offset for each priority query.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay after each comment request.")
    parser.add_argument("--page-sleep", type=float, default=0.25, help="Delay after each search page request.")
    parser.add_argument("--comment-retries", type=int, default=2, help="Retries for each description request.")
    parser.add_argument("--search-retries", type=int, default=3, help="Retries for each Bugzilla search page request.")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds between retry attempts.")
    parser.add_argument("--log-description-errors", action="store_true", help="Print per-bug description fetch failures.")
    parser.add_argument("--skip-comments", action="store_true", help="Fetch only metadata and leave description blank.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Raw CSV output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "Cache-Control": "no-cache"})

    all_records: list[dict[str, Any]] = []
    for priority in PRIORITIES:
        all_records.extend(fetch_priority_records(session, args, priority))

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values(["priority", "creation_time", "id"], na_position="last")
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    ensure_parent_dir(args.output)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"saved: {len(df)} unique rows -> {args.output}")
    if "priority" in df.columns:
        print("\nPriority distribution:")
        print(df["priority"].value_counts(dropna=False).sort_index().to_string())
    if "description" in df.columns:
        missing_description = int((df["description"].fillna("").astype(str).str.strip() == "").sum())
        print(f"\nMissing first-comment descriptions: {missing_description}")
    if "description_source" in df.columns:
        print("\nDescription source distribution:")
        print(df["description_source"].value_counts(dropna=False).sort_index().to_string())
    if "description_error" in df.columns:
        error_count = int((df["description_error"].fillna("").astype(str).str.strip() != "").sum())
        print(f"\nDescription fetch errors: {error_count}")


if __name__ == "__main__":
    main()
