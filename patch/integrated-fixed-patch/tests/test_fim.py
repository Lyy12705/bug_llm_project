import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fim_patch.backend import OllamaFIM, GenerationError
from fim_patch.generator import (FIMPatchGenerator, function_span, build_patch,
                                 git, local_prompt_context, sha)
from fim_patch.validation import UnittestValidator, run_suite


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
        self.prompts = []

    def describe(self):
        return {"model": "test-double"}

    def infill(self, prefix, suffix, *, seed):
        self.calls += 1
        self.prompts.append((prefix, suffix))
        return {"middle": next(self.outputs), "seed": seed}


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.raw = b'def f(x):\n    return x.strip()\n\n\ndef g():\n    return f(" hi ")\n'
        (self.repo / "app.py").write_bytes(self.raw)
        git(self.repo, "init")
        git(self.repo, "config", "core.autocrlf", "false")
        git(self.repo, "add", ".")
        git(self.repo, "-c", "user.name=Test", "-c", "user.email=t@example.invalid", "commit", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD").decode().strip()
        self.row = {"file_path": "app.py", "symbol_qualified_name": "f", "start_line": 1, "end_line": 2}
        self.loc = {"base_commit": self.base, "confidence_level": "high", "should_manual_review": False,
                    "recommend_patch_generation": True, "stage3_ranked_symbols": [self.row]}
        self.ticket = {"base_commit": self.base, "description": "Return empty string for None"}

    def generate(self, output, **kwargs):
        return FIMPatchGenerator(FakeBackend([output]), candidates=1, **kwargs).generate(self.ticket, self.loc, self.repo)

    def test_patch_applies_and_preserves_suffix_and_repository(self):
        result = self.generate('    return "" if x is None else x.strip()\n')
        self.assertEqual(result["patch_status"], "generated")
        self.assertEqual(result["validation_status"], "not_run")
        self.assertEqual((self.repo / "app.py").read_bytes(), self.raw)
        git(self.repo, "apply", data=result["patch"].encode())
        self.assertIn(b'def g():\n    return f(" hi ")', (self.repo / "app.py").read_bytes())

    def test_missing_confidence_is_advisory_and_still_calls_model(self):
        backend = FakeBackend(['    return None\n'])
        self.loc.pop("should_manual_review")
        r = FIMPatchGenerator(backend, candidates=1).generate(self.ticket, self.loc, self.repo)
        self.assertEqual(r["patch_status"], "generated")
        self.assertFalse(r["production_eligible"])
        self.assertEqual(r["confidence_policy"], "advisory_only")
        self.assertTrue(r["generation_allowed"])
        self.assertTrue(r["requires_manual_review"])
        self.assertEqual(backend.calls, 1)

    def test_malformed_input_blocks(self):
        backend = FakeBackend([])
        r = FIMPatchGenerator(backend).generate([], self.loc, self.repo)
        self.assertEqual(r["patch_status"], "blocked_invalid_context")
        self.loc["stage3_ranked_symbols"] = "not a list"
        r = FIMPatchGenerator(backend).generate(self.ticket, self.loc, self.repo)
        self.assertEqual(r["patch_status"], "blocked_invalid_context")
        self.assertEqual(backend.calls, 0)

    def test_research_mode_preserves_production_ineligibility(self):
        self.loc.update(confidence_level='medium', recommend_patch_generation=False,
                        should_manual_review=True)
        r=FIMPatchGenerator(FakeBackend(['    return None\n']), candidates=1,
                            research_mode=True,local_context=True).generate(self.ticket,self.loc,self.repo)
        self.assertEqual(r['patch_status'],'generated')
        self.assertFalse(r['production_eligible'])
        self.assertFalse(r['recommend_patch_generation'])
        self.assertTrue(r['requires_manual_review'])
        self.assertEqual((self.repo/'app.py').read_bytes(),self.raw)

    def test_local_prompt_crop_does_not_crop_reconstructed_file(self):
        before=b'# distant context\n'*400
        raw=before+self.raw+b'# after context\n'*300
        (self.repo/'app.py').write_bytes(raw)
        git(self.repo,'add','app.py')
        git(self.repo,'-c','user.name=Test','-c','user.email=t@example.invalid','commit','-m','long context')
        base=git(self.repo,'rev-parse','HEAD').decode().strip()
        self.ticket['base_commit']=self.loc['base_commit']=base
        self.row.update(start_line=401,end_line=402)
        r=FIMPatchGenerator(FakeBackend(['    return None\n']),candidates=1,
                            local_context=True).generate(self.ticket,self.loc,self.repo)
        self.assertEqual(r['patch_status'],'generated')
        self.assertTrue(r['prompt_context']['truncated'])
        self.assertLessEqual(r['prompt_context']['input_bytes_upper_bound'],
                             r['prompt_context']['input_budget'])
        git(self.repo,'apply',data=r['patch'].encode())
        self.assertEqual((self.repo/'app.py').read_bytes(),raw.replace(b'return x.strip()',b'return None'))

    def test_local_prompt_budget_matches_backend_byte_guard(self):
        span = function_span(self.raw, self.row)
        prefix, suffix, info = local_prompt_context(
            span, self.row, '# issue\n', num_ctx=512, max_new_tokens=32)
        prompt = '▁<PRE>' + prefix + '▁<SUF>' + suffix + '▁<MID>'
        self.assertLessEqual(len(prompt.encode('utf-8')) + 32 + 8, 512)
        self.assertEqual(info['input_budget'], 472)

    def test_prompt_contains_verified_body_but_not_gold_fields(self):
        forbidden = ('patch', 'reference_patch', 'ground_truth', 'test_patch',
                     'FAIL_TO_PASS', 'PASS_TO_PASS', 'hints_text', 'original_middle')
        ticket = dict(self.ticket, **{key: 'SECRET_' + key for key in forbidden})
        for local in (False, True):
            with self.subTest(local_context=local):
                backend = FakeBackend(['    return None\n'])
                result = FIMPatchGenerator(backend, candidates=1, local_context=local).generate(
                    ticket, self.loc, self.repo)
                self.assertEqual(result['patch_status'], 'generated')
                prefix, suffix = backend.prompts[0]
                self.assertIn('#     return x.strip()\n', prefix)
                self.assertTrue(prefix.endswith('def f(x):\n'))
                self.assertNotIn('SECRET_', prefix + suffix)
                self.assertEqual(result['prompt_version'], 'issue-original-body-psm-v2')
                self.assertTrue(result['prompt_context']['original_body_included'])
                self.assertFalse(result['prompt_context']['original_body_truncated'])
                self.assertEqual((self.repo / 'app.py').read_bytes(), self.raw)

    def test_original_body_budget_blocks_before_model_call(self):
        for local in (False, True):
            with self.subTest(local_context=local):
                backend = FakeBackend([])
                backend.num_ctx = 128
                backend.max_new_tokens = 32
                result = FIMPatchGenerator(backend, candidates=1, local_context=local).generate(
                    self.ticket, self.loc, self.repo)
                self.assertEqual(result['patch_status'], 'blocked_invalid_context')
                self.assertIn('context_budget_exceeded', result['explanation'])
                self.assertEqual(backend.calls, 0)

    def test_reference_unicode_crlf_and_exact_budget_boundary(self):
        raw = 'def f():\r\n    # 中文\r\n    return "值"'.encode('utf-8')
        row = dict(self.row, end_line=3)
        span = function_span(raw, row)
        prefix, suffix, info = local_prompt_context(span, row, '# issue\n')
        self.assertIn('#     return "值"\n', prefix)
        self.assertTrue(prefix.endswith('def f():\n    # 中文\n'))
        self.assertNotIn('\r', prefix)
        exact_ctx = info['input_bytes_upper_bound'] + 512 + 8
        p2, s2, _ = local_prompt_context(span, row, '# issue\n', num_ctx=exact_ctx)
        self.assertEqual((prefix, suffix), (p2, s2))
        with self.assertRaisesRegex(ValueError, 'original body'):
            local_prompt_context(span, row, '# issue\n', num_ctx=exact_ctx - 1)

    def test_modified_source_blocks(self):
        (self.repo / "app.py").write_bytes(self.raw + b"# changed\n")
        self.assertEqual(self.generate("    pass\n")["patch_status"], "blocked_invalid_context")

    def test_wrong_hash_blocks(self):
        self.loc["source_file_sha256"] = {"app.py": "wrong"}
        self.assertEqual(self.generate("    pass\n")["patch_status"], "blocked_invalid_context")

    def test_wrong_coordinates_block(self):
        self.row["end_line"] = 5
        self.assertEqual(self.generate("    pass\n")["patch_status"], "blocked_invalid_context")

    def test_research_ast_expansion_requires_matching_start(self):
        raw = b'def f(x):\n    y = x\n    return y\n'
        row = dict(self.row, end_line=2)
        with self.assertRaises(ValueError):
            function_span(raw, row)
        span = function_span(raw, row, expand_truncated_end=True)
        self.assertEqual(span['line_end'], 3)
        self.assertTrue(span['coordinates_expanded'])
        for invalid in (dict(row, start_line=2), dict(row, end_line=4)):
            with self.assertRaises(ValueError):
                function_span(raw, invalid, expand_truncated_end=True)

    def test_research_skips_test_and_class_targets_with_audit(self):
        self.loc['stage3_ranked_symbols'] = [dict(self.row, file_path='test_app.py'),
                                           dict(self.row, symbol_qualified_name='MissingClass'), self.row]
        backend = FakeBackend(['    return None\n'])
        result = FIMPatchGenerator(backend, candidates=1, research_mode=True,
                                  target_policy='first_supported_ast_top5').generate(self.ticket,self.loc,self.repo)
        self.assertEqual(result['patch_status'], 'generated')
        self.assertEqual([x['status'] for x in result['target_resolution']], ['rejected','rejected','selected'])
        self.assertEqual(backend.calls, 1)
        self.assertEqual((self.repo/'app.py').read_bytes(), self.raw)

    def test_alternative_policy_cannot_enable_production_or_search_beyond_five(self):
        with self.assertRaises(ValueError):
            FIMPatchGenerator(target_policy='first_supported_ast_top5')
        self.loc['stage3_ranked_symbols'] = [dict(self.row,file_path='../app.py')]*5+[self.row]
        backend = FakeBackend([])
        result = FIMPatchGenerator(backend,research_mode=True,target_policy='first_supported_ast_top5').generate(self.ticket,self.loc,self.repo)
        self.assertEqual(result['patch_status'],'blocked_invalid_context')
        self.assertEqual(backend.calls,0)

    def test_ambiguous_symbol_still_rejected_with_expansion(self):
        with self.assertRaises(ValueError):
            function_span(b'def f():\n    pass\ndef f():\n    pass\n',self.row,expand_truncated_end=True)

    def test_expanded_function_rebuilds_whole_body_and_preserves_following_function(self):
        raw = b'def f(x):\n    y = x.strip()\n    return y\n\ndef g():\n    return 3\n'
        (self.repo/'app.py').write_bytes(raw)
        git(self.repo,'add','app.py')
        git(self.repo,'-c','user.name=Test','-c','user.email=t@example.invalid','commit','-m','longer body')
        self.ticket['base_commit']=self.loc['base_commit']=git(self.repo,'rev-parse','HEAD').decode().strip()
        result=FIMPatchGenerator(FakeBackend(['    return "" if x is None else x.strip()\n']),
                                 candidates=1,research_mode=True,target_policy='first_supported_ast_top5').generate(self.ticket,self.loc,self.repo)
        self.assertEqual(result['patch_status'],'generated')
        self.assertTrue(result['target_resolution'][0]['coordinates_expanded'])
        self.assertEqual((self.repo/'app.py').read_bytes(),raw)
        git(self.repo,'apply',data=result['patch'].encode('utf-8'))
        self.assertEqual((self.repo/'app.py').read_bytes(),b'def f(x):\n    return "" if x is None else x.strip()\n\ndef g():\n    return 3\n')

    def test_research_target_policy_does_not_bypass_source_hash(self):
        self.loc['source_file_sha256']={'app.py':'invalid'}
        backend=FakeBackend([])
        result=FIMPatchGenerator(backend,research_mode=True,target_policy='first_supported_ast_top5').generate(self.ticket,self.loc,self.repo)
        self.assertEqual(result['patch_status'],'blocked_invalid_context')
        self.assertIn('Source hash differs',result['target_resolution'][0]['reason'])
        self.assertEqual(backend.calls,0)

    def test_path_escape_blocks(self):
        self.row["file_path"] = "../app.py"
        self.assertEqual(self.generate("    pass\n")["patch_status"], "blocked_invalid_context")

    def test_noop_rejected(self):
        self.assertEqual(self.generate("    return x.strip()\n")["patch_status"], "no_valid_candidate")

    def test_body_escape_rejected(self):
        r = self.generate("    pass\n\ndef malicious():\n    pass\n")
        self.assertEqual(r["patch_status"], "no_valid_candidate")

    def test_invalid_syntax_rejected(self):
        self.assertEqual(self.generate("return x\n")["patch_status"], "no_valid_candidate")

    def test_roundtrip_crlf_bom_unicode_decorator(self):
        raw = ('\ufeff# 中文\r\n@deco\r\ndef f(x):\r\n    return x\r\n').encode("utf-8")
        row = dict(self.row, start_line=2, end_line=4)
        span = function_span(raw, row)
        self.assertEqual((span["prefix"] + span["original_middle"] + span["suffix"]).encode("utf-8-sig"), raw)
        patch, after = build_patch(raw, span, '    return None\n', "app.py")
        self.assertTrue(after.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(after.count(b"\r\n"), 4)

    def test_missing_final_newline_diff(self):
        raw = b"def f(x):\n    return x"
        span = function_span(raw, self.row)
        patch, after = build_patch(raw, span, "    return None", "app.py")
        (self.repo / "app.py").write_bytes(raw)
        git(self.repo, "apply", data=patch.encode())
        self.assertEqual((self.repo / "app.py").read_bytes(), after)

    def test_inline_suite_rejected(self):
        with self.assertRaises(ValueError):
            function_span(b"def f(x): return x\n", dict(self.row, end_line=1))

    def test_duplicate_records_remain(self):
        backend = FakeBackend(["    return None\n"] * 2)
        r = FIMPatchGenerator(backend, candidates=2).generate(self.ticket, self.loc, self.repo)
        self.assertEqual(len(r["candidates"]), 2)
        self.assertEqual(r["candidates"][1]["reason"], "duplicate")

    def test_test_gate_rejects_syntactically_valid_wrong_patch(self):
        r = self.generate("    return None\n", validator=lambda p, r: {"status": "failed"})
        self.assertEqual(r["patch_status"], "no_valid_candidate")


class BackendTests(unittest.TestCase):
    def test_preserve_indent_and_eot(self):
        b = OllamaFIM()
        b.post = lambda *a: {"response": "    return 1\n    <EOT>", "done": True, "done_reason": "stop"}
        self.assertEqual(b.infill("def f():\n", "\n", seed=1)["middle"], "    return 1\n")

    def test_missing_eot_and_truncation(self):
        for response in ({"response": "pass", "done": True, "done_reason": "stop"},
                         {"response": "pass<EOT>", "done": True, "done_reason": "length"}):
            b = OllamaFIM()
            b.post = lambda *a: response
            with self.assertRaises(GenerationError):
                b.infill("def f():\n", "", seed=1)

    def test_marker_collision(self):
        with self.assertRaises(ValueError):
            OllamaFIM().infill("# <MID>", "", seed=1)

    def test_budget_checked_before_request(self):
        with self.assertRaises(ValueError):
            OllamaFIM().infill("x" * 4096, "", seed=1)


class ValidationTests(unittest.TestCase):
    def test_zero_tests_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run_suite(tmp, {"discover": "."}, 10)
        self.assertNotEqual(r["status"], "passed")

    def test_real_before_after_and_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text('def f(x):\n    return x.strip()\n')
            (root / "test_app.py").write_text(
                'import unittest\nfrom app import f\nclass Bug(unittest.TestCase):\n'
                ' def test_bug(self): self.assertEqual(f(None), "")\n'
                'class Old(unittest.TestCase):\n def test_old(self): self.assertEqual(f(" a "), "a")\n')
            spec = {"reproducer": {"names": ["test_app.Bug"]},
                    "regression": {"names": ["test_app.Old"]}, "expected_failure": "AttributeError"}
            raw = (root / "app.py").read_bytes()
            span = function_span(raw, {"symbol_qualified_name": "f", "start_line": 1, "end_line": 2})
            patch, _ = build_patch(raw, span, '    return "" if x is None else x.strip()\n', "app.py")
            self.assertEqual(UnittestValidator(spec)(patch, root)["status"], "plausible")
            self.assertEqual((root / "app.py").read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
