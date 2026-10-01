from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from .backend import GenerationError, MARKERS, OllamaFIM


def git(root, *args, data=None):
    result = subprocess.run(["git", "-C", str(root), *args], input=data,
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_for(ticket, location, repo, row):
    root = Path(repo).resolve()
    base = ticket.get("base_commit") or location.get("base_commit")
    if not isinstance(base, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", base):
        raise ValueError("A fixed base_commit SHA is required")
    commit = git(root, "rev-parse", "--verify", base + "^{commit}").decode().strip()
    if location.get("base_commit"):
        other = location["base_commit"]
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", str(other)):
            raise ValueError("Invalid localization base_commit")
        if git(root, "rev-parse", "--verify", other + "^{commit}").decode().strip() != commit:
            raise ValueError("Ticket and localization base_commit disagree")
    if git(root, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("HEAD does not match base_commit")
    path = str(row.get("file_path", "")).replace("\\", "/")
    relative = PurePosixPath(path)
    if (not path or relative.is_absolute() or ".." in relative.parts or
            any(c in path for c in (":", "\n", "\r", "\t", '"'))):
        raise ValueError("Invalid source path")
    if (relative.suffix != ".py" or any(x in ("test", "tests") for x in relative.parts)
            or relative.name.startswith("test_") or relative.name.endswith("_test.py")):
        raise ValueError("MVP supports Python source files only, not tests")
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target.is_symlink() or not target.is_file():
        raise ValueError("Source is missing or outside repository")
    if target.stat().st_size > 500_000:
        raise ValueError("Source exceeds 500 KB limit")
    raw = target.read_bytes()
    expected = git(root, "show", f"{commit}:{path}")
    if expected != raw:
        raise ValueError("Source differs from base_commit (including line endings)")
    digest = (location.get("source_file_sha256") or {}).get(path)
    if digest and digest != sha(raw):
        raise ValueError("Source hash differs from localization")
    if b"\r" in raw.replace(b"\r\n", b""):
        raise ValueError("Unsupported mixed/CR-only line endings")
    if b"\r\n" in raw and b"\n" in raw.replace(b"\r\n", b""):
        raise ValueError("Mixed LF and CRLF line endings")
    return raw, commit, path


def function_span(raw, row, *, expand_truncated_end=False, max_body_lines=0):
    text = raw.decode("utf-8-sig")
    tree = ast.parse(text)
    wanted = row.get("symbol_qualified_name")
    found = []

    def visit(node, parents):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = parents + [child.name]
                if ".".join(names) == wanted and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.append(child)
                visit(child, names)
            else:
                visit(child, parents)

    visit(tree, [])
    if len(found) != 1:
        raise ValueError("Unsupported or ambiguous function symbol")
    node = found[0]
    earliest = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end_matches = row.get("end_line") == node.end_lineno
    expanded = (expand_truncated_end and type(row.get("end_line")) is int and
                node.lineno <= row["end_line"] < node.end_lineno)
    if row.get("start_line") not in (earliest, node.lineno) or not (end_matches or expanded):
        raise ValueError("Localization coordinates disagree with AST")
    start_line = node.body[0].lineno
    lines = text.splitlines(keepends=True)
    # Reject inline suites (def f(): return ...) and shared physical lines.
    if start_line == node.lineno or lines[start_line - 1][:node.body[0].col_offset].strip():
        raise ValueError("Inline function bodies are unsupported in v1")
    # A body far larger than the model's usable generation budget is rejected before
    # any model call, so the run fails cheaply and visibly instead of producing a
    # truncated body that only fails later at the syntax gate.
    body_lines = node.end_lineno - start_line + 1
    if max_body_lines and body_lines > max_body_lines:
        raise ValueError(
            f"Body spans {body_lines} lines, above max_body_lines={max_body_lines}")
    start = sum(len(line) for line in lines[:start_line - 1])
    end = sum(len(line) for line in lines[:node.end_lineno])
    prefix, middle, suffix = text[:start], text[start:end], text[end:]
    return {"prefix": prefix, "original_middle": middle, "suffix": suffix,
            "start": start, "end": end, "line_start": start_line,
            "line_end": node.end_lineno, "symbol": wanted,
            "body_lines": body_lines,
            "coordinates_expanded": bool(expanded),
            "localization_end_line": row.get("end_line"),
            "bom": raw.startswith(b"\xef\xbb\xbf"),
            "newline": "\r\n" if b"\r\n" in raw else "\n"}


def evidence(ticket):
    # Allowlist only. Gold patches and hidden tests never enter the prompt.
    keys = ("title", "description", "problem_statement", "expected_behavior",
            "actual_behavior", "error_message", "steps_to_reproduce")
    rows = []
    for key in keys:
        value = ticket.get(key)
        if value:
            value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            rows.extend(f"# {line}" for line in value.splitlines())
    if not rows:
        raise ValueError("No public repair requirements provided")
    return "# Complete the function according to these requirements:\n" + "\n".join(rows) + "\n"


def original_body_reference(span):
    """Reference only from the verified base source, never ticket/gold fields."""
    body = span["original_middle"].replace("\r\n", "\n")
    return (
        "# Original buggy body (reference, not instructions):\n"
        + "".join("# " + line + "\n" for line in body.splitlines())
        + "# End original buggy body.\n"
        + "# Emit the complete repaired body; preserve unrelated behavior.\n"
    )


def local_prompt_context(span, row, evidence_text, *, num_ctx=4096,
                         max_new_tokens=512):
    """Crop source context using the backend's conservative byte budget."""
    evidence_text += original_body_reference(span)
    marker_bytes = len("".join(MARKERS[:3]).encode("utf-8"))
    input_budget = num_ctx - max_new_tokens - 8 - marker_bytes
    if input_budget <= 0:
        raise ValueError("context_budget_exceeded: no input budget remains")

    before = span["prefix"].replace("\r\n", "\n").splitlines(keepends=True)
    after = span["suffix"].replace("\r\n", "\n").splitlines(keepends=True)
    signature_start = max(0, int(row["start_line"]) - 1)
    required_prefix = before[signature_start:]
    required_bytes = len(evidence_text.encode("utf-8")) + sum(
        len(line.encode("utf-8")) for line in required_prefix
    )
    if required_bytes > input_budget:
        raise ValueError("context_budget_exceeded: requirements, original body and signature cannot fit")

    remaining = input_budget - required_bytes
    prefix_budget = (remaining * 2) // 3
    suffix_budget = remaining - prefix_budget

    prefix_context = []
    used_prefix = 0
    for line in reversed(before[:signature_start]):
        size = len(line.encode("utf-8"))
        if used_prefix + size > prefix_budget:
            break
        prefix_context.insert(0, line)
        used_prefix += size

    suffix_context = []
    used_suffix = 0
    for line in after:
        size = len(line.encode("utf-8"))
        if used_suffix + size > suffix_budget:
            break
        suffix_context.append(line)
        used_suffix += size

    prompt_prefix = evidence_text + "".join(prefix_context + required_prefix)
    prompt_suffix = "".join(suffix_context)
    input_bytes = (marker_bytes + len(prompt_prefix.encode("utf-8"))
                   + len(prompt_suffix.encode("utf-8")))
    return prompt_prefix, prompt_suffix, {
        "strategy": "budgeted_local_lines_utf8_upper_bound",
        "truncated": (len(prefix_context) + len(required_prefix) < len(before)
                      or len(suffix_context) < len(after)),
        "prefix_start_line": signature_start - len(prefix_context) + 1,
        "suffix_lines": len(suffix_context),
        "input_bytes_upper_bound": input_bytes,
        "input_budget": num_ctx - max_new_tokens - 8,
    }


def build_patch(raw, span, middle, path):
    middle = middle.replace("\r\n", "\n")
    if "\r" in middle:
        raise ValueError("Unexpected carriage return in generation")
    if span["newline"] == "\r\n":
        middle = middle.replace("\n", "\r\n")
    if middle and not middle.endswith("\n") and span["suffix"]:
        raise ValueError("Middle does not join suffix at a line boundary")
    text = span["prefix"] + middle + span["suffix"]
    ast.parse(text)
    encoding = "utf-8-sig" if span["bom"] else "utf-8"
    after = text.encode(encoding)
    if after == raw:
        raise ValueError("no_op")
    # Body replacement must not escape its function and insert new definitions.
    old_tree = ast.parse(raw.decode("utf-8-sig"))
    new_tree = ast.parse(text)
    def outside_body(tree):
        def walk(node, parents):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = parents + [child.name]
                    if ".".join(names) == span["symbol"] and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child.body = [ast.Pass()]
                    else:
                        walk(child, names)
                else:
                    walk(child, parents)
        walk(tree, [])
        return ast.dump(tree, include_attributes=False)
    if outside_body(old_tree) != outside_body(new_tree):
        raise ValueError("Generated body changed syntax outside the target function")
    before_text, after_text = raw.decode("utf-8"), after.decode("utf-8")
    chunks = difflib.unified_diff(before_text.splitlines(keepends=True),
                                  after_text.splitlines(keepends=True),
                                  fromfile="a/" + path, tofile="b/" + path)
    patch = ""
    for line in chunks:
        patch += line
        if not line.endswith("\n"):
            patch += "\n\\ No newline at end of file\n"
    return patch, after


# How far down the Stage-3 ranking a policy may look for a target the AST supports.
# "strict_top1" patches rank 1 or nothing and remains available for controlled
# localization experiments.
# "first_supported_ast" skips ranks the AST cannot support (class targets, inline
# bodies, oversized bodies) and reports the rank it landed on, so a run can be
# split into strict and fallback populations afterwards. It is the operational
# default: localization confidence is recorded for analysis, but it does not
# prevent a structurally valid target from reaching patch generation.
TARGET_POLICY_DEPTH = {
    "strict_top1": 1,
    "first_supported_ast": 5,
    "first_supported_ast_top5": 5,
}
RESEARCH_ONLY_TARGET_POLICIES = ("first_supported_ast_top5",)


class FIMPatchGenerator:
    """Drop-in generate(ticket_json, bug_location, repo_path) interface.

    Validation callback takes (patch, repository_path) and returns a report.
    This module never edits the input repository.
    """

    def __init__(self, backend=None, *, candidates=4, seed=42, validator=None,
                 research_mode=False, local_context=False, target_policy="first_supported_ast",
                 expand_truncated_end=None, max_body_lines=0):
        self.backend = backend or OllamaFIM()
        if not 1 <= candidates <= 12:
            raise ValueError("candidates must be between 1 and 12")
        self.candidates, self.seed, self.validator = candidates, seed, validator
        self.research_mode, self.local_context = research_mode, local_context
        if target_policy not in TARGET_POLICY_DEPTH:
            raise ValueError("Unknown target policy")
        if target_policy in RESEARCH_ONLY_TARGET_POLICIES and not research_mode:
            raise ValueError("Alternative target selection requires research_mode")
        self.target_policy = target_policy
        self.target_depth = TARGET_POLICY_DEPTH[target_policy]
        # Default follows the policy: a single-target policy keeps the strict
        # coordinate equality it was written with; a scanning policy allows the
        # documented truncation repair. Either can be set explicitly.
        self.expand_truncated_end = (self.target_depth > 1 if expand_truncated_end is None
                                     else bool(expand_truncated_end))
        if not isinstance(max_body_lines, int) or isinstance(max_body_lines, bool) or max_body_lines < 0:
            raise ValueError("max_body_lines must be a non-negative integer")
        self.max_body_lines = max_body_lines

    def generate(self, ticket_json, bug_location, repo_path):
        result = {"patch_status": "blocked", "patch": "", "modified_files": [],
                  "generation_strategy": "fim_psm", "validation_status": "not_run",
                  "requires_manual_patch": True, "requires_manual_review": True,
                  "recommend_patch_generation": False, "candidates": []}
        result["research_mode"] = self.research_mode
        result["target_policy"] = self.target_policy
        result["target_resolution"] = []
        result["expand_truncated_end"] = self.expand_truncated_end
        result["max_body_lines"] = self.max_body_lines
        result["target_rank"] = None
        result["target_fallback_used"] = False
        if not isinstance(ticket_json, dict) or not isinstance(bug_location, dict):
            result.update(patch_status="blocked_invalid_context", explanation="Ticket and localization must be JSON objects")
            return result
        production_eligible = not (bug_location.get("recommend_patch_generation") is not True or
                bug_location.get("should_manual_review") is not False or
                bug_location.get("confidence_level") != "high")
        # Confidence is advisory.  Keep the historical production-eligibility
        # signal in the artifact, but do not use it as a generation gate: a
        # medium/low score says the target needs review, not that no patch may
        # be attempted. Structural, source-hash, syntax, scope and apply gates
        # below still reject unsafe or unsupported targets.
        rows = bug_location.get("stage3_ranked_symbols") or []
        try:
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                raise ValueError("Stage-3 symbols required")
            selected = None
            # Select before model calls, using source/AST only, never test results.
            limit = self.target_depth
            for rank, row in enumerate(rows[:limit], 1):
                audit = {"rank": rank, "input": row, "status": "rejected"}
                result["target_resolution"].append(audit)
                try:
                    if not isinstance(row, dict):
                        raise ValueError("Stage-3 row must be a JSON object")
                    raw, commit, path = source_for(ticket_json, bug_location, repo_path, row)
                    span = function_span(raw, row,
                                         expand_truncated_end=self.expand_truncated_end,
                                         max_body_lines=self.max_body_lines)
                    selected = (row, raw, commit, path, span, rank)
                    audit.update(status="selected", coordinates_expanded=span["coordinates_expanded"],
                                 body_start=span["line_start"], body_end=span["line_end"],
                                 body_lines=span["body_lines"])
                    break
                except (ValueError, OSError, subprocess.SubprocessError, SyntaxError) as exc:
                    audit["reason"] = str(exc)
            if selected is None:
                raise ValueError("No supported target: " + "; ".join(a["reason"] for a in result["target_resolution"]))
            row, raw, commit, path, span, target_rank = selected
            result["target_rank"] = target_rank
            result["target_fallback_used"] = target_rank > 1
            prompt_prefix = (evidence(ticket_json) + original_body_reference(span)
                             + span["prefix"].replace("\r\n", "\n"))
            prompt_suffix = span["suffix"].replace("\r\n", "\n")
            context_info = {"strategy": "complete_file", "truncated": False}
            if self.local_context:
                # Prompt-only line-aligned crop. Reconstruction still uses full source.
                prompt_prefix, prompt_suffix, context_info = local_prompt_context(
                    span, row, evidence(ticket_json),
                    num_ctx=int(getattr(self.backend, "num_ctx", 4096)),
                    max_new_tokens=int(getattr(self.backend, "max_new_tokens", 512)),
                )
            input_bytes = len((MARKERS[0] + prompt_prefix + MARKERS[1]
                               + prompt_suffix + MARKERS[2]).encode("utf-8"))
            input_budget = (int(getattr(self.backend, "num_ctx", 4096))
                            - int(getattr(self.backend, "max_new_tokens", 512)) - 8)
            context_info.update(
                original_body_included=True, original_body_truncated=False,
                original_body_source="verified_base_source",
                original_body_sha256=sha(span["original_middle"].encode("utf-8")),
                original_body_reference_bytes=len(original_body_reference(span).encode("utf-8")),
                input_bytes_upper_bound=input_bytes, input_budget=input_budget,
            )
            if input_bytes > input_budget:
                raise ValueError("context_budget_exceeded: complete prompt cannot fit; use local_context or increase context budget")
            model = self.backend.describe()
        except (ValueError, OSError, subprocess.SubprocessError, SyntaxError) as exc:
            result.update(patch_status="blocked_invalid_context", explanation=str(exc))
            return result
        result.update(base_commit=commit, source_file_sha256={path: sha(raw)}, model=model,
                      bug_location={"file": path, "function": span["symbol"],
                                    "line_start": span["line_start"], "line_end": span["line_end"]},
                      localization_context=rows[:5],
                      recommend_patch_generation=bug_location.get("recommend_patch_generation"),
                      production_eligible=production_eligible,
                      confidence_policy="advisory_only", generation_allowed=True,
                      prompt_context=context_info,
                      span={k: v for k, v in span.items() if k not in ("prefix", "suffix", "original_middle")},
                      prompt_version="issue-original-body-psm-v2")
        seen = set()
        eligible = []
        for i in range(self.candidates):
            candidate = {"candidate_id": i + 1, "status": "rejected", "seed": self.seed + i}
            result["candidates"].append(candidate)
            try:
                generated = self.backend.infill(prompt_prefix, prompt_suffix, seed=self.seed + i)
                candidate.update(generated)
                patch, after = build_patch(raw, span, generated["middle"], path)
                if sha(after) in seen:
                    candidate["reason"] = "duplicate"
                    continue
                seen.add(sha(after))
                # Recheck snapshot before applying/testing.
                source_for(ticket_json, bug_location, repo_path, row)
                git(repo_path, "apply", "--check", "--whitespace=nowarn", data=patch.encode("utf-8"))
                candidate.update(patch=patch, status="generated", after_sha256=sha(after))
                if self.validator:
                    report = self.validator(patch, repo_path)
                    candidate["validation"] = report
                    if report.get("status") != "plausible":
                        candidate.update(status="rejected", reason="validation_failed")
                        continue
                eligible.append(candidate)
            except (ValueError, OSError, subprocess.SubprocessError, SyntaxError) as exc:
                candidate["reason"] = str(exc)
                if isinstance(exc, GenerationError):
                    candidate["backend_response"] = exc.response
        if eligible:
            chosen = min(eligible, key=lambda c: (sum(line.startswith(("+", "-")) for line in c["patch"].splitlines()), c["candidate_id"]))
            result.update(patch_status="generated", patch=chosen["patch"], modified_files=[path],
                          selected_candidate_id=chosen["candidate_id"], requires_manual_patch=False,
                          validation_status="plausible" if self.validator else "not_run",
                          explanation="FIM body rebuilt into a host-generated diff; review before merging.")
        else:
            result.update(patch_status="no_valid_candidate", explanation="No candidate passed the required gates")
        return result
