from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request

from utils.json_schema import repair_json_object


@dataclass(slots=True)
class OllamaClient:
    url: str = "http://localhost:11434/api/generate"
    model: str = "codellama:7b-instruct"
    timeout: int = 180

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "format": "json",
            "options": {"temperature": 0.0, "top_p": 1.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return str(body.get("response", "")).strip()

    def generate_json(self, prompt: str) -> dict[str, Any]:
        return repair_json_object(self.generate(prompt))
