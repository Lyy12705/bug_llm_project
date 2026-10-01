"""Integrated deterministic localization and FIM repair, bound to one snapshot."""
import copy
import json
from pathlib import Path

from utils.fault_localization import build_code_index, localize_ticket
from .generator import FIMPatchGenerator, git


def localization_for_patch(location, index):
    """Adapt a freshly computed FL result without inventing confidence."""
    location = copy.deepcopy(location)
    for key in ('confidence_level', 'should_manual_review', 'recommend_patch_generation'):
        if key not in location:
            raise ValueError('Localization missing ' + key)
    rows = location.get('stage3_ranked_symbols')
    if not isinstance(rows, list):
        raise ValueError('Localization requires Stage-3 symbol list')
    fingerprints = index.settings['file_fingerprints']
    for row in rows:
        if not isinstance(row, dict) or not all(k in row for k in
                ('file_path', 'symbol_qualified_name', 'start_line', 'end_line')):
            raise ValueError('Malformed Stage-3 symbol')
        if row['file_path'] not in fingerprints:
            raise ValueError('Symbol source is absent from indexed snapshot')
    location.update(schema_version='LocalizationForPatchV1',
                    base_commit=index.settings['base_commit'],
                    repo=index.settings['repository_name'])
    location['source_file_sha256'] = {r['file_path']: fingerprints[r['file_path']] for r in rows}
    return location


class RepairPipeline:
    def __init__(self, generator=None):
        self.generator = generator or FIMPatchGenerator()

    def run(self, ticket, repo):
        if not isinstance(ticket, dict) or not ticket.get('base_commit'):
            raise ValueError('Ticket requires base_commit')
        base = str(ticket['base_commit'])
        import re
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}', base):
            raise ValueError('base_commit must be a commit SHA')
        commit = git(repo, 'rev-parse', '--verify', base + '^{commit}').decode().strip()
        if git(repo, 'rev-parse', 'HEAD').decode().strip() != commit:
            raise ValueError('HEAD does not match base_commit')
        # Index only a clean tracked snapshot; never relabel dirty source as base.
        if git(repo, 'status', '--porcelain'):
            raise ValueError('Pipeline requires a clean repository')
        index = build_code_index(repo, repository_name=ticket.get('repo') or Path(repo).name,
                                 base_commit=commit)
        location = localize_ticket(ticket, code_index=index, symbol_localization=True,
                                  symbol_llm_rerank=False, symbol_retrieval_mode='b1-structured',
                                  symbol_selection_mode='coverage-aware-v1')
        location = localization_for_patch(location, index)
        # Confidence is inherited from FL, never manufactured by the adapter.
        result = self.generator.generate(ticket, location, repo)
        return location, result


def write_run(output, ticket, location, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    chosen = next((c for c in result['candidates']
                   if c['candidate_id'] == result.get('selected_candidate_id')), {})
    evaluation = {
        'patch_status': result['patch_status'],
        'validation_status': result['validation_status'],
        'interpretation': ('Passed supplied tests; requires review' if result['validation_status'] == 'plausible'
                           else 'Generated but not functionally verified' if result['patch']
                           else result.get('explanation', 'No patch')),
        'validation': chosen.get('validation'), 'official_resolved': None,
    }
    for name, value in [('ticket', ticket), ('localization', location), ('result', result),
                        ('candidates', result['candidates']), ('evaluation', evaluation)]:
        (output / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'selected.diff').write_bytes(result['patch'].encode('utf-8'))
    prediction = {'instance_id': ticket.get('instance_id') or ticket.get('ticket_id'),
                  'model_name_or_path': result.get('model', {}).get('model', 'unknown'),
                  'model_patch': result['patch']}
    (output / 'predictions.jsonl').write_text(json.dumps(prediction) + '\n', encoding='utf-8')
