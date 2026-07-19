from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
STATIC_ROOT = DEMO_ROOT / "static"
DATA_PATH = DEMO_ROOT / "data" / "scenarios.json"
SRC_ROOT = PROJECT_ROOT / "src"
MAX_REQUEST_BYTES = 1_000_000
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import PipelineConfig  # noqa: E402
from modules.assignee_triager import AssigneeTriager  # noqa: E402
from modules.duplicate_detector import DuplicateDetector  # noqa: E402
from modules.priority_classifier import PriorityClassifier  # noqa: E402
from utils.json_schema import normalize_structured_ticket  # noqa: E402


DEMO_HISTORICAL_TICKETS = [
    {
        "ticket_id": "BUG-038",
        "title": "Login crash when token missing",
        "description": "Missing auth token causes TypeError during login validation.",
        "product": "Auth Service",
        "component": "authentication",
        "priority": "P2",
        "assignee": "auth-team@example.com",
    },
    {
        "ticket_id": "BUG-214",
        "title": "Auth validator returns 500 for empty credentials",
        "description": "Empty credentials trigger an auth validation exception.",
        "product": "Auth Service",
        "component": "authentication",
        "priority": "P3",
        "assignee": "auth-team@example.com",
    },
    {
        "ticket_id": "BUG-271",
        "title": "Login form shows blank error when session token expires",
        "description": "Expired session token makes the login form render an empty validation state.",
        "product": "Auth Service",
        "component": "authentication",
        "priority": "P3",
        "assignee": "auth-team@example.com",
    },
    {
        "ticket_id": "BUG-312",
        "title": "OAuth callback fails when refresh token is missing",
        "description": "OAuth callback raises an exception if the refresh token field is absent.",
        "product": "Auth Service",
        "component": "authentication",
        "priority": "P2",
        "assignee": "auth-team@example.com",
    },
    {
        "ticket_id": "BUG-039",
        "title": "Password reset email not sent",
        "description": "SMTP timeout prevents password reset emails.",
        "product": "Auth Service",
        "component": "email",
        "priority": "P3",
        "assignee": "backend-team@example.com",
    },
    {
        "ticket_id": "BUG-319",
        "title": "Checkout total does not update after coupon removal",
        "description": "Checkout total shows stale subtotal after a coupon is removed.",
        "product": "Checkout UI",
        "component": "frontend",
        "priority": "P3",
        "assignee": "frontend-team@example.com",
    },
    {
        "ticket_id": "BUG-288",
        "title": "Cart page shows stale subtotal after refresh",
        "description": "Cart subtotal is stale after refreshing the checkout page.",
        "product": "Checkout UI",
        "component": "frontend",
        "priority": "P3",
        "assignee": "frontend-team@example.com",
    },
    {
        "ticket_id": "BUG-335",
        "title": "Apply coupon button remains disabled after editing code",
        "description": "Checkout coupon input keeps the apply button disabled after a user changes the code.",
        "product": "Checkout UI",
        "component": "frontend",
        "priority": "P4",
        "assignee": "frontend-team@example.com",
    },
    {
        "ticket_id": "BUG-347",
        "title": "Order summary panel overlaps payment form",
        "description": "Responsive checkout layout overlaps the payment form on small screens.",
        "product": "Checkout UI",
        "component": "frontend",
        "priority": "P4",
        "assignee": "ui-platform@example.com",
    },
    {
        "ticket_id": "BUG-401",
        "title": "Payment API returns timeout during card authorization",
        "description": "Backend payment gateway times out while authorizing saved cards.",
        "product": "Payments",
        "component": "backend",
        "priority": "P2",
        "assignee": "backend-team@example.com",
    },
    {
        "ticket_id": "BUG-418",
        "title": "Inventory count is stale after checkout rollback",
        "description": "Database rollback leaves inventory count stale after checkout failure.",
        "product": "Inventory",
        "component": "database",
        "priority": "P2",
        "assignee": "data-team@example.com",
    },
    {
        "ticket_id": "BUG-433",
        "title": "Search suggestions show outdated product names",
        "description": "Search suggestion cache returns outdated product names after catalog update.",
        "product": "Storefront",
        "component": "search",
        "priority": "P3",
        "assignee": "backend-team@example.com",
    },
    {
        "ticket_id": "BUG-447",
        "title": "Profile page avatar upload fails on large image",
        "description": "Large avatar image upload fails with a 413 response from the profile API.",
        "product": "User Profile",
        "component": "api",
        "priority": "P3",
        "assignee": "backend-team@example.com",
    },
]

DEMO_COMPONENT_OWNERS = {
    "authentication": "auth-team@example.com",
    "email": "backend-team@example.com",
    "frontend": "frontend-team@example.com",
    "backend": "backend-team@example.com",
    "database": "data-team@example.com",
    "search": "backend-team@example.com",
    "api": "backend-team@example.com",
}


def load_scenarios() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "BugAssistantDemo/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/scenarios":
            payload = load_scenarios()
            summaries = [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "kind": item["kind"],
                    "one_line": item["one_line"],
                }
                for item in payload["scenarios"]
            ]
            self._send_json({"scenarios": summaries})
            return

        if path.startswith("/api/scenarios/"):
            scenario_id = unquote(path.removeprefix("/api/scenarios/"))
            for item in load_scenarios()["scenarios"]:
                if item["id"] == scenario_id:
                    self._send_json(item)
                    return
            self.send_error(404, "Scenario not found")
            return

        if path == "/":
            self._send_file(STATIC_ROOT / "index.html")
            return

        requested = (STATIC_ROOT / path.lstrip("/")).resolve()
        if STATIC_ROOT.resolve() not in requested.parents and requested != STATIC_ROOT.resolve():
            self.send_error(403, "Forbidden")
            return
        if requested.exists() and requested.is_file():
            self._send_file(requested)
            return
        self.send_error(404, "File not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path != "/api/analyze":
            self.send_error(404, "Endpoint not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                self.send_error(413, f"Request body must be at most {MAX_REQUEST_BYTES} bytes")
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            text = str(payload.get("text") or "")
            if not text.strip():
                self.send_error(400, "Input text is required")
                return
            self._send_json(
                analyze_text(
                    text,
                    duplicate_decision=str(payload.get("duplicate_decision") or ""),
                    selected_duplicate_id=str(payload.get("selected_duplicate_id") or ""),
                )
            )
        except Exception as exc:  # pragma: no cover - demo endpoint guard
            self.send_error(500, f"Analysis failed: {exc}")

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def analyze_text(text: str, *, duplicate_decision: str = "", selected_duplicate_id: str = "") -> dict[str, Any]:
    config = PipelineConfig(
        project_root=PROJECT_ROOT,
        duplicate_threshold=0.50,
        duplicate_top_k=10,
        component_owner_mapping=DEMO_COMPONENT_OWNERS,
    )
    structured_ticket = normalize_structured_ticket(parse_user_input(text))
    matching_ticket = dict(structured_ticket)
    matching_ticket["title"] = _with_demo_keywords(str(structured_ticket.get("title", "")), text)
    matching_ticket["description"] = _with_demo_keywords(str(structured_ticket.get("description", "")), text)
    duplicate_result = DuplicateDetector(config=config).detect(matching_ticket, DEMO_HISTORICAL_TICKETS)
    duplicate_result["review_mode"] = "semi_automatic_top_10"
    duplicate_result["engineer_action"] = "請工程師檢查 Top-10 候選清單後，確認是否真的為 duplicate。"

    decision = duplicate_decision.strip().lower()
    duplicate_result["engineer_decision"] = "pending"
    duplicate_result["review_required"] = decision not in {"duplicate", "not_duplicate"}

    result: dict[str, Any] = {
        "status": "needs_duplicate_review",
        "structured_ticket": structured_ticket,
        "duplicate": duplicate_result,
    }

    if decision == "duplicate":
        top_candidate = (duplicate_result.get("top_k_candidates") or [{}])[0]
        confirmed_duplicate = (
            selected_duplicate_id.strip()
            or duplicate_result.get("duplicate_of")
            or top_candidate.get("ticket_id")
            or ""
        )
        duplicate_result["engineer_decision"] = "confirmed_duplicate"
        duplicate_result["confirmed_duplicate_of"] = confirmed_duplicate
        duplicate_result["review_required"] = False
        result["status"] = "duplicate_confirmed"
        result["duplicate_of"] = confirmed_duplicate
        return result

    if decision != "not_duplicate":
        return result

    duplicate_result["engineer_decision"] = "confirmed_not_duplicate"
    duplicate_result["review_required"] = False
    priority_result = PriorityClassifier().predict(structured_ticket, duplicate_result.get("top_k_candidates", []))
    assignee_result = AssigneeTriager(config=config).assign(structured_ticket, priority_result)
    result["status"] = "completed"
    result["priority"] = priority_result
    result["assignee"] = assignee_result
    return result


def parse_user_input(text: str) -> dict[str, Any]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in normalized.splitlines()]
    title = next((line for line in lines if line and not _is_label_line(line)), "RAW-DEMO")
    environment = _field(normalized, ("Environment", "環境", "執行環境"))
    product = _field(normalized, ("Product", "產品", "專案", "Project"), default="Demo Project")
    component = _normalize_component(_field(normalized, ("Component", "元件", "模組", "功能")))
    severity = _normalize_severity(_field(normalized, ("Severity", "嚴重性", "重要性")))
    expected = _field(normalized, ("Expected", "預期", "期望結果"))
    actual = _field(normalized, ("Actual", "實際", "實際結果"))
    logs = _field(normalized, ("Log", "Error Log", "錯誤訊息", "錯誤", "日誌"))
    steps = _steps(normalized)
    description = _description(normalized)

    if component == "unknown":
        component = _infer_component(normalized)
    bug_type = _infer_bug_type(normalized)
    version = _infer_version(environment)

    return {
        "ticket_id": _field(normalized, ("Ticket ID", "ID", "編號"), default="RAW-DEMO"),
        "title": title,
        "description": description or normalized,
        "product": product,
        "bug_type": bug_type,
        "component": component,
        "severity": severity,
        "os": _infer_os(environment),
        "version": version,
        "error_message": logs,
        "steps_to_reproduce": steps,
        "expected_behavior": expected,
        "actual_behavior": actual,
        "logs": logs,
    }


def _field(text: str, labels: tuple[str, ...], *, default: str = "") -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"^(?:{label_pattern})\s*[:：]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


def _steps(text: str) -> list[str]:
    match = re.search(
        r"(?:Steps to reproduce|Steps|重現步驟|操作步驟)\s*[:：]?\s*([\s\S]+?)(?=\n\s*(?:Expected|預期|Actual|實際|Log|錯誤|日誌)\s*[:：]|\Z)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    steps = []
    for line in match.group(1).splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\.\)、)]|[-*])\s*", "", line).strip()
        if cleaned:
            steps.append(cleaned)
    return steps


def _description(text: str) -> str:
    labelled = _field(text, ("Description", "描述", "問題描述"))
    if labelled:
        return labelled
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    for block in blocks[1:]:
        if not _is_label_line(block.splitlines()[0]) and not re.match(r"^(?:Steps|重現步驟|Expected|預期|Actual|實際|Log|錯誤)", block, re.I):
            return block
    return ""


def _is_label_line(line: str) -> bool:
    return bool(re.match(r"^[A-Za-z ]+[:：]|^(?:回報者|元件|模組|功能|嚴重性|重要性|環境|預期|實際|錯誤|日誌|重現步驟)[:：]", line))


def _normalize_component(value: str) -> str:
    raw = value.strip().lower()
    mapping = {
        "前端": "frontend",
        "介面": "frontend",
        "使用者介面": "frontend",
        "登入": "authentication",
        "驗證": "authentication",
        "認證": "authentication",
        "後端": "backend",
        "資料庫": "database",
    }
    return mapping.get(raw, raw or "unknown")


def _normalize_severity(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"高", "嚴重", "重大", "major"}:
        return "major"
    if raw in {"中", "普通", "normal", "medium"}:
        return "normal"
    if raw in {"低", "輕微", "minor", "low"}:
        return "minor"
    return raw


def _infer_component(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("login", "token", "auth", "password", "登入", "權杖", "密碼", "驗證", "認證")):
        return "authentication"
    if any(term in lower for term in ("checkout", "cart", "discount", "button", "ui", "frontend", "結帳", "購物車", "折扣", "按鈕", "畫面", "前端")):
        return "frontend"
    if any(term in lower for term in ("database", "sql", "資料庫")):
        return "database"
    return "unknown"


def _infer_bug_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("crash", "panic", "當機", "崩潰", "閃退", "空白")):
        return "crash"
    if any(term in lower for term in ("typeerror", "exception", "traceback", "錯誤", "例外")):
        return "runtime_error"
    if any(term in lower for term in ("slow", "timeout", "效能", "逾時")):
        return "performance"
    return "unknown"


def _infer_os(environment: str) -> str:
    lower = environment.lower()
    if "mac" in lower:
        return "macOS"
    if "win" in lower or "windows" in lower:
        return "Windows"
    if "linux" in lower:
        return "Linux"
    return "unknown"


def _infer_version(environment: str) -> str:
    match = re.search(r"(?:app|version|版本)?\s*([0-9]+(?:\.[0-9]+){1,3})", environment, flags=re.I)
    return match.group(1) if match else ""


def _with_demo_keywords(description: str, full_text: str) -> str:
    lower = full_text.lower()
    expansions = []
    keyword_map = {
        ("登入", "權杖", "token", "login"): "login token auth validation crash",
        ("結帳", "折扣", "購物車", "checkout", "discount", "cart"): "checkout discount cart total blank page",
        ("空白", "blank"): "blank page crash",
    }
    for terms, expansion in keyword_map.items():
        if any(term in lower for term in terms):
            expansions.append(expansion)
    return " ".join([description, *expansions]).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the bug assistant demo interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Bug Assistant demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
