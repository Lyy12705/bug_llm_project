from __future__ import annotations

import json
import re
from typing import Any


PRIORITIES = {"P1", "P2", "P3", "P4", "P5"}
BUG_TYPES = {
    "runtime_error",
    "crash",
    "logic_error",
    "performance",
    "build_error",
    "compatibility",
    "ui_bug",
    "security",
    "non_bug",
    "unknown",
}

STRUCTURED_TICKET_DEFAULTS: dict[str, Any] = {
    "ticket_id": "",
    "title": "",
    "description": "",
    "product": "unknown",
    "severity": "unknown",
    "bug_type": "unknown",
    "component": "unknown",
    "os": "unknown",
    "version": "",
    "priority": None,
    "error_message": "",
    "steps_to_reproduce": [],
    "expected_behavior": "",
    "actual_behavior": "",
    "logs": "",
    "screenshots_text": "",
}


def repair_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    try:
        obj = json.loads(value)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", value)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def normalize_structured_ticket(raw_ticket: dict[str, Any]) -> dict[str, Any]:
    ticket = dict(STRUCTURED_TICKET_DEFAULTS)
    source = raw_ticket or {}

    predicted = source.get("predicted_json")
    if isinstance(predicted, dict):
        ticket.update(predicted)

    ticket.update({key: source[key] for key in ticket.keys() if key in source})
    ticket["ticket_id"] = _first_string(
        source,
        "ticket_id",
        "bug_id",
        "issue_id",
        "id",
        default=ticket["ticket_id"] or "RAW-UNKNOWN",
    )
    ticket["title"] = _first_string(source, "title", "summary", "subject", default=ticket["title"])
    ticket["description"] = _first_string(
        source,
        "description",
        "body",
        "content",
        "bug_report",
        "comment",
        "comments",
        default=ticket["description"],
    )

    environment = source.get("environment")
    if isinstance(environment, dict):
        if not ticket.get("os") or str(ticket.get("os")).lower() == "unknown":
            ticket["os"] = environment.get("os") or environment.get("platform") or "unknown"
        if not ticket.get("version"):
            ticket["version"] = environment.get("version") or environment.get("app_version") or ""

    bug_type = canonical_bug_type(ticket.get("bug_type"))
    ticket["bug_type"] = infer_bug_type(ticket) if bug_type == "unknown" else bug_type
    component = normalize_component(ticket.get("component"))
    ticket["component"] = infer_component(ticket) if component == "unknown" else component
    ticket["os"] = normalize_os(str(ticket.get("os") or "unknown"))
    ticket["priority"] = canonical_priority(ticket.get("priority"))
    ticket["error_message"] = str(ticket.get("error_message") or infer_error_message(ticket) or "")
    ticket["steps_to_reproduce"] = normalize_steps(ticket.get("steps_to_reproduce"))
    ticket["product"] = str(ticket.get("product") or "unknown")
    ticket["severity"] = str(ticket.get("severity") or "unknown")

    for key in ("title", "description", "version", "expected_behavior", "actual_behavior", "logs", "screenshots_text"):
        ticket[key] = str(ticket.get(key) or "")

    for key in (
        "assignee",
        "recent_similar_count",
        "reporter_past_bug_count",
        "patch",
        "proposed_patch",
        "suggested_patch",
        "unified_diff",
        "generated_tests",
        "tests",
    ):
        if key in source:
            ticket[key] = source[key]

    return validate_structured_ticket(ticket)


def validate_structured_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in STRUCTURED_TICKET_DEFAULTS if key not in ticket]
    if missing:
        raise ValueError(f"Structured ticket is missing required fields: {', '.join(missing)}")
    if ticket["bug_type"] not in BUG_TYPES:
        ticket["bug_type"] = "unknown"
    priority = ticket.get("priority")
    if priority is not None and priority not in PRIORITIES:
        ticket["priority"] = canonical_priority(priority)
    if not isinstance(ticket.get("steps_to_reproduce"), list):
        ticket["steps_to_reproduce"] = normalize_steps(ticket.get("steps_to_reproduce"))
    return ticket


def canonical_priority(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    upper = raw.upper()
    mapping = {
        "P1": "P1",
        "BLOCKER": "P1",
        "CRITICAL": "P2",
        "P2": "P2",
        "MAJOR": "P3",
        "HIGH": "P2",
        "MEDIUM": "P3",
        "NORMAL": "P3",
        "P3": "P3",
        "MINOR": "P4",
        "LOW": "P4",
        "P4": "P4",
        "TRIVIAL": "P5",
        "LOWEST": "P5",
        "P5": "P5",
    }
    return mapping.get(upper, None)


def canonical_bug_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in BUG_TYPES:
        return raw
    mapping = {
        "exception": "runtime_error",
        "runtime": "runtime_error",
        "runtimeerror": "runtime_error",
        "typeerror": "runtime_error",
        "assertionerror": "runtime_error",
        "segfault": "crash",
        "segmentation fault": "crash",
        "wrong_result": "logic_error",
        "incorrect": "logic_error",
        "compile_error": "build_error",
        "build": "build_error",
        "ui": "ui_bug",
        "feature_request": "non_bug",
        "enhancement": "non_bug",
    }
    return mapping.get(raw, "unknown")


def normalize_os(value: str) -> str:
    raw = (value or "").strip()
    lower = raw.lower()
    if not lower:
        return "unknown"
    if lower in {"mac", "macos", "osx", "mac os", "darwin"}:
        return "macOS"
    if lower in {"win", "windows", "win10", "win11"}:
        return "Windows"
    if lower in {"linux", "ubuntu", "debian", "fedora", "centos"}:
        return raw if lower != "linux" else "Linux"
    return raw


def normalize_component(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def normalize_steps(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"(?:\n+|\s*\d+\.\s+|\s*-\s+)", text)
    return [part.strip() for part in parts if part.strip()]


def infer_bug_type(ticket: dict[str, Any]) -> str:
    text = _ticket_text(ticket).lower()
    if any(word in text for word in ("segmentation fault", "segfault", "crash", "panic")):
        return "crash"
    if any(word in text for word in ("typeerror", "runtimeerror", "exception", "traceback", "assertionerror")):
        return "runtime_error"
    if any(word in text for word in ("build failed", "compile", "compiler", "linker", "cmake", "makefile")):
        return "build_error"
    if any(word in text for word in ("slow", "latency", "timeout", "performance")):
        return "performance"
    if any(word in text for word in ("ui", "dialog", "button", "screen", "display", "layout")):
        return "ui_bug"
    if any(word in text for word in ("security", "xss", "csrf", "injection", "privilege")):
        return "security"
    if any(word in text for word in ("wrong", "incorrect", "expected", "actual")):
        return "logic_error"
    return "unknown"


def infer_component(ticket: dict[str, Any]) -> str:
    text = _ticket_text(ticket).lower()
    keyword_map = {
        "authentication": ("auth", "login", "token", "password", "oauth", "session"),
        "database": ("database", "sql", "migration", "query", "postgres", "mysql"),
        "build": ("build", "compile", "compiler", "cmake", "makefile", "linker"),
        "frontend": ("frontend", "ui", "button", "screen", "dialog", "css", "layout"),
        "api": ("api", "endpoint", "request", "response", "http", "rest"),
        "security": ("security", "xss", "csrf", "injection", "permission", "privilege"),
    }
    for component, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return component
    return "unknown"


def infer_error_message(ticket: dict[str, Any]) -> str:
    logs = str(ticket.get("logs") or "")
    text = "\n".join([logs, str(ticket.get("description") or "")])
    patterns = [
        r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):[^\n]+)",
        r"(Traceback[^\n]*(?:\n.+){0,4})",
        r"(segmentation fault[^\n]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def _ticket_text(ticket: dict[str, Any]) -> str:
    keys = ("title", "description", "error_message", "logs", "component", "actual_behavior")
    return "\n".join(str(ticket.get(key) or "") for key in keys)


def _first_string(source: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default
