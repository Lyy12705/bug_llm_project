from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.json_schema import normalize_structured_ticket, repair_json_object


class TicketExtractor:
    """Convert a raw ticket into the shared structured ticket JSON."""

    def __init__(self, *, llm_client: Any | None = None, prompt_path: str | Path | None = None) -> None:
        self.llm_client = llm_client
        self.prompt_path = Path(prompt_path) if prompt_path else None

    def extract(self, raw_ticket: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw_ticket, dict):
            raise ValueError("raw_ticket must be a dictionary.")

        if self.llm_client is None:
            return normalize_structured_ticket(raw_ticket)

        prompt = self._build_prompt(raw_ticket)
        llm_output = self.llm_client.generate(prompt)
        extracted = repair_json_object(llm_output)
        merged = dict(raw_ticket)
        merged.update(extracted)
        return normalize_structured_ticket(merged)

    def _build_prompt(self, raw_ticket: dict[str, Any]) -> str:
        template = ""
        if self.prompt_path and self.prompt_path.exists():
            template = self.prompt_path.read_text(encoding="utf-8")
        if not template:
            template = (
                "Extract a structured bug ticket JSON with fields ticket_id, title, "
                "description, product, severity, bug_type, component, os, version, priority, error_message, "
                "steps_to_reproduce, expected_behavior, actual_behavior, logs, screenshots_text. "
                "Return only JSON.\n\nRaw ticket:\n{raw_ticket}"
            )
        return template.format(raw_ticket=raw_ticket)
