import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fim_patch.generator import FIMPatchGenerator, git
from fim_patch.pipeline import RepairPipeline, localization_for_patch, write_run
from fim_patch.validation import UnittestValidator


class Backend:
    def describe(self):
        return {'model': 'test-double'}

    def infill(self, prefix, suffix, *, seed):
        return {'middle': '    return "" if value is None else value.strip().lower()\n'}


class PipelineTests(unittest.TestCase):
    def test_real_localization_to_validated_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            raw = 'def normalize_username(value):\n    return value.strip().lower()\n'
            (repo / 'username.py').write_text(raw, encoding='utf-8')
            (repo / 'test_username.py').write_text(
                'import unittest\nfrom username import normalize_username\n'
                'class Tests(unittest.TestCase):\n'
                ' def test_none(self): self.assertEqual(normalize_username(None), "")\n'
                ' def test_string(self): self.assertEqual(normalize_username(" A "), "a")\n',
                encoding='utf-8')
            git(repo, 'init')
            git(repo, 'config', 'core.autocrlf', 'false')
            git(repo, 'add', '.')
            git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'base')
            ticket = {'ticket_id': 'smoke', 'base_commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
                      'title': 'normalize_username crashes on None',
                      'description': 'normalize_username(None) must return an empty string; strip and lowercase strings.',
                      'logs': 'File "username.py", line 2, in normalize_username: AttributeError'}
            spec = {'expected_failure': 'AttributeError',
                    'reproducer': {'names': ['test_username.Tests.test_none']},
                    'regression': {'names': ['test_username.Tests.test_string']}}
            pipeline = RepairPipeline(FIMPatchGenerator(Backend(), candidates=1, validator=UnittestValidator(spec)))
            loc, result = pipeline.run(ticket, repo)
            self.assertTrue(loc['stage3_ranked_symbols'])
            self.assertEqual(result['validation_status'], 'plausible')
            self.assertEqual(git(repo, 'status', '--porcelain'), b'')
            out = Path(tmp) / 'output'
            write_run(out, ticket, loc, result)
            self.assertEqual(json.loads((out / 'predictions.jsonl').read_text())['model_patch'], result['patch'])
            with self.assertRaises(FileExistsError):
                write_run(out, ticket, loc, result)
            (repo / 'username.py').write_text(raw + '# dirty\n')
            with self.assertRaisesRegex(ValueError, 'clean repository'):
                pipeline.run(ticket, repo)

    def test_adapter_does_not_fabricate_confidence(self):
        index = SimpleNamespace(settings={'file_fingerprints': {}, 'base_commit': 'a'*40, 'repository_name': 'r'})
        with self.assertRaisesRegex(ValueError, 'confidence_level'):
            localization_for_patch({'stage3_ranked_symbols': []}, index)
        low = {'confidence_level': 'low', 'should_manual_review': True,
               'recommend_patch_generation': False, 'stage3_ranked_symbols': []}
        adapted = localization_for_patch(low, index)
        self.assertFalse(adapted['recommend_patch_generation'])
        self.assertNotIn('schema_version', low)
