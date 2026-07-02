"""
fetch_issue_fields.py

用途：
從多個 issue / bug report 來源抓取原始資料，統一輸出成後續流程可使用的格式。

輸出格式：
{
  "source": "string",
  "issue_id": "string",
  "title": "string",
  "body": "string",
  "direct_fields": {
    "product": "string",
    "severity": "string",
    "bug_type": "string",
    "component": "string",
    "os": "string",
    "version": "string",
    "priority": "string",
    "error_message": "string"
  }
}

規則：
- `title` / `body` 保留原始文字，提供後續 transform_issues.py 組成 bug_report
- `direct_fields` 只填來源本身能直接取得或簡單偵測到的欄位
- 沒有的欄位一律留空字串 ""
- 目前支援：GitHub Issues、Jira Search API

注意：
這支程式只負責「抓來源資料並做初步映射」，不負責：
- transform 成 bug_report
- LLM 抽取 JSON
- evaluation
"""
import json
import os
import re
from typing import Any, Dict, List

import requests

# direct_fields 中保留下游流程會用到的目標欄位。
TARGET_FIELDS = [
    "product",
    "severity",
    "bug_type",
    "component",
    "os",
    "version",
    "priority",
    "error_message",
]

# 可加入多個 GitHub repo
GITHUB_REPOS = [
    "pytorch/pytorch",
    # "tensorflow/tensorflow",
    # "huggingface/transformers",
]

# 可加入多個 Jira source
JIRA_SOURCES = [
    {
        "base_url": "https://issues.apache.org/jira/rest/api/2/search",
        "jql": "project = HADOOP ORDER BY created DESC",
        "maxResults": 30,
    },
]

OS_PATTERNS = {
    "windows": re.compile(r"\bwindows\b|\bwin11\b|\bwin10\b", re.I),
    "linux": re.compile(r"\blinux\b|\bubuntu\b|\bdebian\b|\bredhat\b", re.I),
    "macos": re.compile(r"\bmacos\b|\bmac os\b|\bosx\b|\bmac\b", re.I),
}

ERROR_PATTERN = re.compile(
    r"(AssertionError|RuntimeError|ValueError|TypeError|ModuleNotFoundError|ImportError|Exception|Error)[^\n]{0,160}",
    re.I,
)


def empty_record() -> Dict[str, str]:
    return {key: "" for key in TARGET_FIELDS}


def empty_raw_record() -> Dict[str, Any]:
    return {
        "source": "",
        "issue_id": "",
        "title": "",
        "body": "",
        "direct_fields": empty_record(),
    }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def detect_os(text: str) -> str:
    for name, pattern in OS_PATTERNS.items():
        if pattern.search(text or ""):
            return name
    return ""


def detect_error_message(text: str) -> str:
    if not text:
        return ""
    match = ERROR_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def extract_description(desc: Any) -> str:
    if desc is None:
        return ""
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, dict):
        parts: List[str] = []
        for key in ("text", "content"):
            value = desc.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                for item in value:
                    parts.append(extract_description(item))
        return " ".join(p for p in parts if p).strip()
    if isinstance(desc, list):
        return " ".join(extract_description(item) for item in desc if item).strip()
    return str(desc).strip()


# 將 GitHub issue 映射成統一原始格式

def map_github_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    record = empty_raw_record()

    title = normalize_text(issue.get("title"))
    body = normalize_text(issue.get("body"))
    text = f"{title} {body}".strip()

    labels = issue.get("labels", []) or []
    label_names = []
    for label in labels:
        if isinstance(label, dict):
            label_names.append(normalize_text(label.get("name")))
        else:
            label_names.append(normalize_text(label))

    milestone = issue.get("milestone") or {}
    direct_fields = empty_record()
    repository_url = normalize_text(issue.get("repository_url"))
    if "/repos/" in repository_url:
        direct_fields["product"] = repository_url.rsplit("/repos/", 1)[-1]

    component_candidates = [
        name for name in label_names
        if name and not name.lower().startswith(("priority", "severity", "module:"))
    ]
    direct_fields["component"] = component_candidates[0] if component_candidates else ""

    if isinstance(milestone, dict):
        direct_fields["version"] = normalize_text(milestone.get("title"))

    for name in label_names:
        lower = name.lower()
        if not direct_fields["priority"] and lower in {"p1", "p2", "p3", "p4", "p5", "critical", "major", "minor", "trivial", "low"}:
            direct_fields["priority"] = name
        if lower.startswith("severity:"):
            direct_fields["severity"] = name.split(":", 1)[-1].strip()
        elif lower in {"blocker", "critical", "major", "normal", "minor", "trivial", "enhancement"}:
            direct_fields["severity"] = name

    direct_fields["os"] = detect_os(text)
    direct_fields["error_message"] = detect_error_message(text)
    direct_fields["bug_type"] = ""

    record["source"] = "github"
    record["issue_id"] = normalize_text(issue.get("number")) or normalize_text(issue.get("id"))
    record["title"] = title
    record["body"] = body
    record["direct_fields"] = direct_fields

    return record


# 將 Jira issue 映射成統一原始格式

def map_jira_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    record = empty_raw_record()
    fields = issue.get("fields", {}) or {}

    summary = normalize_text(fields.get("summary"))
    description = extract_description(fields.get("description"))
    environment = normalize_text(fields.get("environment"))
    text = f"{summary} {description} {environment}".strip()

    issue_type = fields.get("issuetype") or {}
    priority = fields.get("priority") or {}
    project = fields.get("project") or {}
    components = fields.get("components") or []
    versions = fields.get("versions") or []
    fix_versions = fields.get("fixVersions") or []

    issue_type_name = normalize_text(issue_type.get("name")) if isinstance(issue_type, dict) else ""
    priority_name = normalize_text(priority.get("name")) if isinstance(priority, dict) else ""

    direct_fields = empty_record()
    direct_fields["bug_type"] = issue_type_name
    direct_fields["priority"] = priority_name
    if isinstance(project, dict):
        direct_fields["product"] = normalize_text(project.get("name")) or normalize_text(project.get("key"))

    if components:
        first_component = components[0]
        if isinstance(first_component, dict):
            direct_fields["component"] = normalize_text(first_component.get("name"))
        else:
            direct_fields["component"] = normalize_text(first_component)

    version_names = []
    for item in versions + fix_versions:
        if isinstance(item, dict):
            version_names.append(normalize_text(item.get("name")))
        else:
            version_names.append(normalize_text(item))
    direct_fields["version"] = ", ".join([v for v in version_names if v])

    direct_fields["os"] = detect_os(text)
    direct_fields["error_message"] = detect_error_message(text)

    record["source"] = "jira"
    record["issue_id"] = normalize_text(issue.get("key")) or normalize_text(issue.get("id"))
    record["title"] = summary
    record["body"] = description
    record["direct_fields"] = direct_fields

    return record


def fetch_github_repo(repo: str, per_page: int = 30) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/issues"
    response = requests.get(url, params={"per_page": per_page}, timeout=30)
    response.raise_for_status()
    issues = response.json()

    dataset: List[Dict[str, Any]] = []
    for issue in issues:
        dataset.append(map_github_issue(issue))
    return dataset


def fetch_jira_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    response = requests.get(
        source["base_url"],
        params={
            "jql": source["jql"],
            "maxResults": source.get("maxResults", 30),
            "fields": "summary,description,environment,issuetype,priority,project,components,versions,fixVersions",
        },
        timeout=30,
    )
    response.raise_for_status()
    response_json = response.json()
    issues = response_json.get("issues", [])

    dataset: List[Dict[str, Any]] = []
    for issue in issues:
        dataset.append(map_jira_issue(issue))
    return dataset


def main() -> None:
    dataset: List[Dict[str, Any]] = []

    for repo in GITHUB_REPOS:
        try:
            dataset.extend(fetch_github_repo(repo))
            print(f"Fetched GitHub repo: {repo}")
        except Exception as e:
            print(f"[WARN] Failed to fetch GitHub repo {repo}: {e}")

    for source in JIRA_SOURCES:
        try:
            dataset.extend(fetch_jira_source(source))
            print(f"Fetched Jira source: {source['jql']}")
        except Exception as e:
            print(f"[WARN] Failed to fetch Jira source {source['jql']}: {e}")

    os.makedirs("dataset/raw", exist_ok=True)
    with open("dataset/raw/issues.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"dataset saved: {len(dataset)} records")


if __name__ == "__main__":
    main()
