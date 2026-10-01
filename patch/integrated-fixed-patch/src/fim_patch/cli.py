"""Single supported command-line entry for FL -> FIM -> evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backend import OllamaFIM
from .evaluation import (SWEbenchEvaluationConfig, compare_ground_truth,
                         load_ground_truth, run_swebench_evaluation,
                         update_evaluation_artifact)
from .generator import TARGET_POLICY_DEPTH, FIMPatchGenerator, RESEARCH_ONLY_TARGET_POLICIES
from .pipeline import RepairPipeline, write_run
from .validation import UnittestValidator


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticket', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-spec', type=Path, help='Trusted local unittest specification')
    parser.add_argument('--ground-truth', type=Path, help='Reference patch JSON read only after generation')
    parser.add_argument('--model', default='codellama:7b-instruct')
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-ctx', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--research-mode', action='store_true')
    parser.add_argument('--local-context', action='store_true')
    parser.add_argument('--target-policy', choices=tuple(TARGET_POLICY_DEPTH),
                        default='first_supported_ast',
                        help='strict_top1 patches Stage-3 rank 1 or nothing; '
                             'first_supported_ast walks down to the first rank the AST '
                             'supports and reports target_rank (default)')
    parser.add_argument('--expand-truncated-end', dest='expand_truncated_end',
                        action='store_true', default=None,
                        help='Accept a Stage-3 end_line that falls inside the definition '
                             '(the 80-line chunk boundary case). Default: on for scanning '
                             'policies, off for strict_top1')
    parser.add_argument('--no-expand-truncated-end', dest='expand_truncated_end',
                        action='store_false')
    parser.add_argument('--max-body-lines', type=int, default=0,
                        help='Skip a target whose body exceeds this many lines, before any '
                             'model call. 0 disables the limit')
    parser.add_argument('--official-eval', action='store_true',
                        help='Run the official Docker-based SWE-bench harness after generation')
    parser.add_argument('--dataset-name', default='SWE-bench/SWE-bench_Lite')
    parser.add_argument('--split', default='test')
    parser.add_argument('--run-id')
    parser.add_argument('--max-workers', type=int, default=1)
    parser.add_argument('--eval-timeout', type=int, default=1800)
    parser.add_argument('--swebench-python', default=sys.executable)
    parser.add_argument('--modal', action='store_true')
    return parser


def result_exit_code(result, official=None):
    if official is not None:
        return 0 if official.get('official_resolved') is True else 1
    if result.get('validation_status') == 'plausible':
        return 0
    return 2 if result.get('patch') else 1


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def execute(args, parser):
    if args.target_policy in RESEARCH_ONLY_TARGET_POLICIES and not args.research_mode:
        parser.error('Alternative target selection requires --research-mode')
    if args.max_body_lines < 0:
        parser.error('--max-body-lines must be zero or positive')
    if args.output.exists() or args.output.resolve().is_relative_to(args.repo.resolve()):
        parser.error('Use a new output directory outside the input repository')
    if args.official_eval and not args.run_id:
        parser.error('--official-eval requires --run-id')
    ticket = _read(args.ticket)
    validator = UnittestValidator(_read(args.test_spec)) if args.test_spec else None
    generator = FIMPatchGenerator(
        OllamaFIM(model=args.model, url=args.url, num_ctx=args.num_ctx,
                  max_new_tokens=args.max_new_tokens),
        candidates=args.candidates, seed=args.seed, validator=validator,
        research_mode=args.research_mode, local_context=args.local_context,
        target_policy=args.target_policy,
        expand_truncated_end=args.expand_truncated_end,
        max_body_lines=args.max_body_lines,
    )
    location, result = RepairPipeline(generator).run(ticket, str(args.repo.resolve()))
    write_run(args.output, ticket, location, result)

    instance_id = ticket.get('instance_id') or ticket.get('ticket_id')
    if args.ground_truth:
        ground_truth = compare_ground_truth(
            result['patch'], load_ground_truth(args.ground_truth, instance_id),
            instance_id=instance_id,
            base_commit=result.get('base_commit'),
        )
        (args.output / 'ground_truth.json').write_text(
            json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding='utf-8')
        update_evaluation_artifact(args.output, ground_truth=ground_truth)

    official = None
    if args.official_eval:
        if not result.get('patch'):
            raise ValueError('Official evaluation requires a generated patch')
        if not instance_id:
            raise ValueError('Official evaluation requires instance_id')
        official = run_swebench_evaluation(
            args.output / 'predictions.jsonl', instance_id,
            args.output / 'official_harness',
            SWEbenchEvaluationConfig(
                dataset_name=args.dataset_name, split=args.split, run_id=args.run_id,
                max_workers=args.max_workers, timeout=args.eval_timeout,
                python_executable=args.swebench_python, modal=args.modal,
            ),
        )
        (args.output / 'official_evaluation.json').write_text(
            json.dumps(official, ensure_ascii=False, indent=2), encoding='utf-8')
        update_evaluation_artifact(args.output, official=official)

    summary = {'output': str(args.output.resolve()), 'patch_status': result['patch_status'],
               'validation_status': result['validation_status'],
               'official_resolved': None if official is None else official['official_resolved']}
    print(json.dumps(summary, ensure_ascii=False))
    return result_exit_code(result, official)


def main(argv=None):
    parser = build_parser()
    return execute(parser.parse_args(argv), parser)
