from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (str(SRC), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from config import PipelineConfig
from pipeline.orchestrator import build_default_orchestrator
from utils.llm_client import OllamaClient


class _Response:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps({"response": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b"}).encode()


class MainStage3IntegrationTests(unittest.TestCase):
    def test_default_orchestrator_has_stage3_and_opt_in_patch_model(self):
        disabled = build_default_orchestrator(PipelineConfig(save_checkpoints=False))
        self.assertTrue(disabled.bug_localizer.symbol_localization)
        self.assertIsNone(disabled.patch_generator.llm_client)
        enabled = build_default_orchestrator(PipelineConfig(
            save_checkpoints=False, patch_generation_enabled=True,
        ))
        client = enabled.patch_generator.llm_client
        self.assertIsNotNone(client)
        self.assertEqual(client.model, "codellama:7b-instruct")
        self.assertFalse(client.json_mode)
        self.assertFalse(client.raw_mode)
        self.assertEqual(client.num_ctx, 16384)

    def test_patch_client_omits_json_format_and_sets_context_options(self):
        client = OllamaClient(json_mode=False, num_ctx=8192, num_predict=1024)
        captured = {}
        def fake_open(req, timeout):
            captured.update(json.loads(req.data.decode()))
            return _Response()
        with patch("utils.llm_client.request.urlopen", side_effect=fake_open):
            response = client.generate("prompt")
        self.assertNotIn("format", captured)
        self.assertEqual(captured["options"]["num_ctx"], 8192)
        self.assertEqual(captured["options"]["num_predict"], 1024)
        self.assertTrue(captured["raw"])
        self.assertTrue(response.startswith("--- a/x"))

    def test_patch_client_uses_ollama_instruct_template(self):
        client = OllamaClient(json_mode=False, raw_mode=False)
        captured = {}
        def fake_open(req, timeout):
            captured.update(json.loads(req.data.decode()))
            return _Response()
        with patch("utils.llm_client.request.urlopen", side_effect=fake_open):
            client.generate("prompt")
        self.assertFalse(captured["raw"])

    def test_existing_json_client_contract_remains_default(self):
        captured = {}
        def fake_open(req, timeout):
            captured.update(json.loads(req.data.decode()))
            return _Response()
        with patch("utils.llm_client.request.urlopen", side_effect=fake_open):
            OllamaClient().generate("prompt")
        self.assertEqual(captured["format"], "json")


if __name__ == "__main__":
    unittest.main()
