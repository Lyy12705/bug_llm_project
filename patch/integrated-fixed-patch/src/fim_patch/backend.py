from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


MARKERS = ("▁<PRE>", "▁<SUF>", "▁<MID>", "▁<EOT>")


@dataclass
class OllamaFIM:
    """Code Llama PSM via raw Ollama completion; no chat/JSON template."""

    model: str = "codellama:7b-instruct"
    url: str = "http://localhost:11434"
    timeout: int = 180
    num_ctx: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.2

    def __post_init__(self):
        if self.timeout <= 0 or self.max_new_tokens <= 0 or self.num_ctx <= self.max_new_tokens + 8:
            raise ValueError("Invalid timeout or token budget")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")

    def post(self, endpoint, payload):
        req = request.Request(self.url.rstrip("/") + "/api/" + endpoint,
                              data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def describe(self):
        info = self.post("show", {"model": self.model, "verbose": True})
        metadata = info.get("model_info", {})
        tokens = metadata.get("tokenizer.ggml.tokens", [])
        if not all(t in tokens for t in MARKERS):
            raise ValueError("unsupported_fim_backend: Code Llama FIM tokens missing")
        tags = self._tags()
        matches = [m for m in tags.get("models", []) if m.get("name") == self.model]
        if not matches:
            raise ValueError("Model must be installed with an explicit matching tag")
        if self.num_ctx > metadata.get("llama.context_length", 0):
            raise ValueError("Configured context exceeds model metadata")
        return {"model": self.model, "digest": matches[0]["digest"],
                "details": info.get("details"),
                "fim_token_ids": {t: tokens.index(t) for t in MARKERS},
                "num_ctx": self.num_ctx, "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature, "backend": "ollama-raw-psm-v1"}

    def _tags(self):
        with request.urlopen(self.url.rstrip("/") + "/api/tags", timeout=self.timeout) as r:
            return json.load(r)

    def infill(self, prefix, suffix, *, seed):
        for marker in ("<PRE>", "<SUF>", "<MID>", "<EOT>", "<FILL_ME>"):
            if marker in prefix or marker in suffix:
                raise ValueError("FIM marker collision in source/evidence")
        prompt = MARKERS[0] + prefix + MARKERS[1] + suffix + MARKERS[2]
        # Byte-fallback tokenizer: byte length is a conservative upper bound.
        # Refuse oversize input rather than let Ollama silently truncate it.
        if len(prompt.encode("utf-8")) + self.max_new_tokens + 8 > self.num_ctx:
            raise ValueError("context_budget_exceeded: conservative UTF-8 byte bound")
        reply = self.post("generate", {
            "model": self.model, "prompt": prompt, "raw": True, "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": self.max_new_tokens,
                        "temperature": self.temperature, "top_p": 0.95, "seed": seed},
        })
        raw = reply.get("response", "")
        if not reply.get("done") or reply.get("done_reason") == "length":
            raise GenerationError("truncated_or_incomplete", reply)
        # Do not add a stop string: retain visible EOT evidence, distinct from EOS.
        pos = raw.find("<EOT>")
        if pos < 0:
            raise GenerationError("missing_visible_eot", reply)
        middle = raw[:pos]
        if middle.endswith("▁"):
            middle = middle[:-1]
        # Remove only EOT-adjacent indentation, not code indentation/newlines.
        middle = middle.rstrip(" \t")
        if any(t in middle for t in ("<PRE>", "<SUF>", "<MID>", "```")):
            raise GenerationError("non_code_output", reply)
        return {"middle": middle, "raw_response": raw, "seed": seed,
                "finish_reason": "visible_eot", "prompt": prompt,
                "metrics": {k: reply.get(k) for k in (
                    "done_reason", "prompt_eval_count", "eval_count", "total_duration")}}


class GenerationError(ValueError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response
