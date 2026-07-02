from __future__ import annotations

from typing import Any


class CommitMessageGenerator:
    """Generate a Conventional Commit style message from pipeline outputs."""

    def generate(self, ticket_json: dict[str, Any], patch: dict[str, Any], test_result: dict[str, Any]) -> dict[str, str]:
        component = _scope(ticket_json, patch)
        title = _summarize_title(ticket_json)
        prefix = "fix" if str(ticket_json.get("bug_type") or "") != "non_bug" else "chore"
        message = f"{prefix}({component}): {title}" if component else f"{prefix}: {title}"

        regression = (test_result.get("regression_result") or {}).get("regression_result")
        reproduction = (test_result.get("reproduction_tests") or {}).get("test_result")
        body_parts = [
            patch.get("explanation") or "Updates the localized code for the reported bug.",
            f"Ticket: {ticket_json.get('ticket_id', 'unknown')}.",
            f"Regression result: {regression or 'not_run'}.",
            f"Reproduction tests: {reproduction or 'not_generated'}.",
        ]
        return {"commit_message": message[:100], "commit_body": " ".join(body_parts)}


def _scope(ticket: dict[str, Any], patch: dict[str, Any]) -> str:
    component = str(ticket.get("component") or "").strip().lower()
    if component and component != "unknown":
        return _slug(component)
    files = patch.get("modified_files") or []
    if files:
        return _slug(str(files[0]).split("/")[0])
    return ""


def _summarize_title(ticket: dict[str, Any]) -> str:
    title = str(ticket.get("title") or "resolve reported bug").strip()
    title = title[0].lower() + title[1:] if title else "resolve reported bug"
    for suffix in (".", "!", "?"):
        title = title.rstrip(suffix)
    return title[:72]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")[:24]
