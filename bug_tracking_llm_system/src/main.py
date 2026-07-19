from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from config import PipelineConfig
from modules.assignee_deployment import load_deployment_bundle
from pipeline.orchestrator import build_default_orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the integrated bug tracking LLM pipeline.")
    parser.add_argument("--raw-ticket", required=True, help="Path to raw_ticket.json.")
    parser.add_argument("--repo-path", required=True, help="Repository path used for bug localization and patch validation.")
    parser.add_argument("--output", default="final_pipeline_result.json", help="Output JSON path.")
    parser.add_argument("--historical-tickets", default=None, help="Optional historical_tickets.jsonl path.")
    parser.add_argument("--assignee-dataset", default=None, help="Optional assignee history JSONL path.")
    parser.add_argument(
        "--assignee-deployment-bundle",
        default=None,
        help="Optional deployment_bundle.json; explicit assignee CLI arguments override bundle values.",
    )
    parser.add_argument("--assignee-active-roster", default=None, help="Optional JSON roster of active assignees.")
    parser.add_argument("--assignee-inactive", default=None, help="Optional JSON/list of inactive or departed assignees.")
    parser.add_argument("--assignee-component-ownership", default=None, help="Optional JSON component ownership map.")
    parser.add_argument("--assignee-file-ownership", default=None, help="Optional JSON file/module ownership map.")
    parser.add_argument("--assignee-feedback", default=None, help="Optional confirmed assignee feedback JSONL path.")
    parser.add_argument("--assignee-routing-policy", default=None, help="Optional CSV/JSON calibration routing policy.")
    parser.add_argument("--assignee-routing-policy-name", default="", help="Routing policy row/name to use from the policy file.")
    parser.add_argument("--assignee-calibration-artifact", default=None, help="Optional approved confidence calibration artifact JSON.")
    parser.add_argument(
        "--assignee-allow-uncalibrated-auto-assignment",
        action="store_true",
        help="Allow research-only auto-assignment without an approved calibrator; unsafe for production.",
    )
    parser.add_argument("--assignee-open-set", action="store_true", help="Enable rule-based open-set risk gate for assignee routing.")
    parser.add_argument("--assignee-open-set-risk-threshold", type=float, default=None)
    parser.add_argument("--assignee-open-set-artifact", default=None, help="Optional approved open-set detector artifact JSON.")
    parser.add_argument("--duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--run-regression-tests", action="store_true", help="Run tests in the temporary patched copy.")
    parser.add_argument("--test-command", default="python3 -m pytest", help="Command to run when regression tests are enabled.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable step-level JSON checkpoints.")
    parser.add_argument("--fault-code-index", default=None, help="Optional prebuilt code index for fault localization.")
    parser.add_argument("--fault-top-k", type=int, default=5, help="Number of fault-localization candidates.")
    parser.add_argument(
        "--fault-embedding-backend",
        choices=("auto", "tfidf", "sbert", "sentence-transformers", "tfidf-sbert-rerank"),
        default="tfidf",
        help="Fault-localization retrieval backend.",
    )
    parser.add_argument("--fault-sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--fault-sbert-cache-dir", default=None, help="Optional persistent SBERT embedding cache directory.")
    parser.add_argument(
        "--fault-allow-sbert-download",
        action="store_true",
        help="Allow sentence-transformers model download for fault localization.",
    )
    parser.add_argument("--fault-llm-rerank", action="store_true", help="Use Ollama/Code Llama to rerank localization candidates.")
    parser.add_argument("--fault-llm-candidate-k", type=int, default=10, help="Number of retrieved candidates sent to LLM rerank.")
    parser.add_argument("--fault-llm-cache-dir", default=None, help="Optional JSONL cache directory for fault-localization LLM rerank.")
    parser.add_argument("--fault-ollama-model", default="codellama:7b-instruct")
    parser.add_argument("--fault-ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--fault-ollama-timeout", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_ticket_path = Path(args.raw_ticket)
    with raw_ticket_path.open("r", encoding="utf-8") as handle:
        raw_ticket = json.load(handle)

    project_root = Path(__file__).resolve().parents[1]
    bundle_config = {}
    if args.assignee_deployment_bundle:
        bundle = load_deployment_bundle(Path(args.assignee_deployment_bundle))
        bundle_config = bundle["pipeline_config"]

    def assignee_path(cli_value: str | None, config_key: str) -> Path | None:
        value = cli_value or bundle_config.get(config_key)
        return Path(value) if value else None

    config = PipelineConfig(
        project_root=project_root,
        historical_tickets_path=Path(args.historical_tickets) if args.historical_tickets else None,
        assignee_dataset_path=assignee_path(args.assignee_dataset, "assignee_dataset_path"),
        assignee_active_roster_path=assignee_path(args.assignee_active_roster, "assignee_active_roster_path"),
        assignee_inactive_path=assignee_path(args.assignee_inactive, "assignee_inactive_path"),
        assignee_component_ownership_path=assignee_path(
            args.assignee_component_ownership, "assignee_component_ownership_path"
        ),
        assignee_file_ownership_path=assignee_path(args.assignee_file_ownership, "assignee_file_ownership_path"),
        assignee_feedback_path=assignee_path(args.assignee_feedback, "assignee_feedback_path"),
        assignee_routing_policy_path=assignee_path(args.assignee_routing_policy, "assignee_routing_policy_path"),
        assignee_routing_policy_name=(
            args.assignee_routing_policy_name or str(bundle_config.get("assignee_routing_policy_name") or "")
        ),
        assignee_calibration_artifact_path=assignee_path(
            args.assignee_calibration_artifact, "assignee_calibration_artifact_path"
        ),
        assignee_allow_uncalibrated_auto_assignment=(
            args.assignee_allow_uncalibrated_auto_assignment
            or bool(bundle_config.get("assignee_allow_uncalibrated_auto_assignment", False))
        ),
        assignee_open_set_enabled=(
            args.assignee_open_set or bool(bundle_config.get("assignee_open_set_enabled", False))
        ),
        assignee_open_set_risk_threshold=(
            args.assignee_open_set_risk_threshold
            if args.assignee_open_set_risk_threshold is not None
            else float(bundle_config.get("assignee_open_set_risk_threshold", 0.75))
        ),
        assignee_open_set_artifact_path=assignee_path(
            args.assignee_open_set_artifact, "assignee_open_set_artifact_path"
        ),
        duplicate_threshold=args.duplicate_threshold,
        run_regression_tests=args.run_regression_tests,
        test_command=shlex.split(args.test_command),
        save_checkpoints=not args.no_checkpoints,
        fault_localization_code_index_path=Path(args.fault_code_index) if args.fault_code_index else None,
        fault_localization_top_k=args.fault_top_k,
        fault_localization_embedding_backend=args.fault_embedding_backend,
        fault_localization_sbert_model=args.fault_sbert_model,
        fault_localization_sbert_local_files_only=not args.fault_allow_sbert_download,
        fault_localization_sbert_cache_dir=Path(args.fault_sbert_cache_dir) if args.fault_sbert_cache_dir else None,
        fault_localization_llm_rerank=args.fault_llm_rerank,
        fault_localization_llm_candidate_k=args.fault_llm_candidate_k,
        fault_localization_llm_cache_dir=Path(args.fault_llm_cache_dir) if args.fault_llm_cache_dir else None,
        fault_localization_ollama_model=args.fault_ollama_model,
        fault_localization_ollama_url=args.fault_ollama_url,
        fault_localization_ollama_timeout=args.fault_ollama_timeout,
    )
    orchestrator = build_default_orchestrator(config)
    result = orchestrator.run_pipeline(raw_ticket, args.repo_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result.get("status"), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
