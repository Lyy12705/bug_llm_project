from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4


DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
STATIC_ROOT = DEMO_ROOT / "static"
DATA_PATH = DEMO_ROOT / "data" / "scenarios.json"
DEMO_RUNS_ROOT = DEMO_ROOT / "runs"
ADMIN_DATA_ROOT = DEMO_ROOT / "admin_data"
ADMIN_TICKETS_PATH = ADMIN_DATA_ROOT / "imported_tickets.json"
ADMIN_MEMBERS_PATH = ADMIN_DATA_ROOT / "imported_members.json"
ADMIN_ASSIGNEE_HISTORY_PATH = ADMIN_DATA_ROOT / "assignee_history.jsonl"
ADMIN_ROSTER_PATH = ADMIN_DATA_ROOT / "assignee_roster.json"
SRC_ROOT = PROJECT_ROOT / "src"
MAX_REQUEST_BYTES = 1_000_000
MAX_IMPORT_RECORDS = 5_000
ADMIN_PREVIEW_LIMIT = 100
RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
RUN_ARTIFACTS = {
    "structured_ticket.json",
    "duplicate_detection.json",
    "priority_prediction.json",
    "assignee_recommendation.json",
    "assignee_selection.json",
    "analysis_result.json",
}
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

DEMO_TICKET_DETAILS = {
    "BUG-038": {
        "status": "處理中",
        "environment": "macOS 14 · Chrome 127 · Auth Service 1.2.0",
        "steps_to_reproduce": ["開啟登入頁", "清除工作階段中的 token", "送出登入表單"],
        "expected_behavior": "顯示重新登入或驗證失敗提示。",
        "actual_behavior": "登入驗證拋出 TypeError，頁面直接中止。",
        "error_message": "TypeError: token is missing in auth validator",
    },
    "BUG-214": {
        "status": "待處理",
        "environment": "Ubuntu 22.04 · Auth API 2.8.1",
        "steps_to_reproduce": ["送出空白帳號與密碼", "呼叫登入驗證 API"],
        "expected_behavior": "回傳 400 並提示必填欄位。",
        "actual_behavior": "驗證器發生例外並回傳 HTTP 500。",
        "error_message": "ValidationError: credentials cannot be empty",
    },
    "BUG-271": {
        "status": "已排程",
        "environment": "Windows 11 · Edge 126 · Web 4.1.6",
        "steps_to_reproduce": ["登入後等待 session token 過期", "重新送出登入表單"],
        "expected_behavior": "顯示 session 已過期並要求重新登入。",
        "actual_behavior": "錯誤訊息區塊為空白，使用者不知道如何繼續。",
        "error_message": "SessionExpiredError without message payload",
    },
    "BUG-312": {
        "status": "調查中",
        "environment": "OAuth callback · Auth Service 2.9.0",
        "steps_to_reproduce": ["使用缺少 refresh token 的 OAuth 回應", "進入 callback URL"],
        "expected_behavior": "導回登入頁並要求重新授權。",
        "actual_behavior": "callback handler 因欄位不存在而停止。",
        "error_message": "KeyError: refresh_token",
    },
    "BUG-039": {
        "status": "已修復",
        "environment": "Production · Mail Service 3.4.2",
        "steps_to_reproduce": ["在忘記密碼頁輸入有效信箱", "送出重設密碼要求"],
        "expected_behavior": "一分鐘內收到重設密碼郵件。",
        "actual_behavior": "頁面顯示已寄出，但郵件未送達。",
        "error_message": "SMTPTimeoutError after 30 seconds",
    },
    "BUG-319": {
        "status": "處理中",
        "environment": "Windows 11 · Chrome 127 · Checkout UI 4.2.0",
        "steps_to_reproduce": ["在結帳頁套用折扣碼", "移除折扣碼", "查看訂單總額"],
        "expected_behavior": "移除折扣碼後立即恢復原始總額。",
        "actual_behavior": "折扣碼已移除，但訂單總額仍保留折扣。",
        "error_message": "Checkout total state was not refreshed",
    },
    "BUG-288": {
        "status": "待驗證",
        "environment": "macOS 14 · Safari 17 · Checkout UI 4.1.8",
        "steps_to_reproduce": ["修改購物車商品數量", "重新整理結帳頁", "查看小計"],
        "expected_behavior": "小計應與最新商品數量一致。",
        "actual_behavior": "頁面重新整理後仍顯示修改前的小計。",
        "error_message": "Cached subtotal returned by cart store",
    },
    "BUG-335": {
        "status": "待處理",
        "environment": "Android 15 · Chrome Mobile · Checkout UI 4.2.0",
        "steps_to_reproduce": ["輸入無效折扣碼", "修改折扣碼內容", "再次嘗試套用"],
        "expected_behavior": "修改內容後重新啟用套用按鈕。",
        "actual_behavior": "按鈕持續停用，必須重新整理頁面。",
        "error_message": "Coupon form dirty state not updated",
    },
    "BUG-347": {
        "status": "已排程",
        "environment": "iPhone 15 · Safari · 390 × 844",
        "steps_to_reproduce": ["以手機開啟結帳頁", "捲動至付款表單"],
        "expected_behavior": "訂單摘要與付款表單依序排列。",
        "actual_behavior": "訂單摘要覆蓋信用卡輸入欄位。",
        "error_message": "Responsive grid overflow below 420px",
    },
    "BUG-401": {
        "status": "緊急處理中",
        "environment": "Production · Payment API 5.0.3",
        "steps_to_reproduce": ["選擇已儲存的信用卡", "送出付款授權", "等待交易結果"],
        "expected_behavior": "付款授權在 10 秒內完成並更新訂單。",
        "actual_behavior": "閘道逾時，訂單停留在待付款狀態。",
        "error_message": "GatewayTimeout during card authorization",
    },
    "BUG-418": {
        "status": "調查中",
        "environment": "Production · Inventory Service 3.7.0",
        "steps_to_reproduce": ["建立訂單並保留庫存", "讓付款失敗觸發 rollback", "查看庫存數量"],
        "expected_behavior": "交易回復後庫存數量同步還原。",
        "actual_behavior": "訂單已取消，但庫存仍維持扣除狀態。",
        "error_message": "Inventory rollback event was not consumed",
    },
    "BUG-433": {
        "status": "待處理",
        "environment": "Production · Search Service 2.6.4",
        "steps_to_reproduce": ["更新商品名稱", "等待索引完成", "輸入關鍵字查看建議"],
        "expected_behavior": "搜尋建議顯示更新後的商品名稱。",
        "actual_behavior": "建議清單仍顯示快取中的舊名稱。",
        "error_message": "Suggestion cache was not invalidated",
    },
    "BUG-447": {
        "status": "已排程",
        "environment": "Windows 11 · Chrome 127 · Profile API 2.4.1",
        "steps_to_reproduce": ["開啟個人資料頁", "選擇大於 8 MB 的圖片", "上傳頭像"],
        "expected_behavior": "壓縮圖片或提示檔案大小限制。",
        "actual_behavior": "上傳失敗，只顯示未說明原因的錯誤。",
        "error_message": "HTTP 413 Payload Too Large",
    },
}

DEMO_COMPONENT_OWNERS = {
    "authentication": "auth-team@example.com",
    "checkout": "checkout-team@example.com",
    "email": "backend-team@example.com",
    "frontend": "frontend-team@example.com",
    "backend": "backend-team@example.com",
    "database": "data-team@example.com",
    "search": "backend-team@example.com",
    "api": "backend-team@example.com",
}

DEMO_TEAM_MEMBERS = [
    {
        "email": "auth-team@example.com",
        "name": "身分驗證團隊",
        "role": "功能維運團隊",
        "team": "身分驗證與登入",
        "component": "authentication",
        "specialties": ["登入流程", "Token／Session", "OAuth 與權限"],
        "similar_cases": 26,
        "resolved_90_days": 41,
        "open_items": 8,
        "availability": "需由團隊排程",
        "status": "active",
    },
    {
        "email": "frontend-team@example.com",
        "name": "前端體驗團隊",
        "role": "功能維運團隊",
        "team": "前端體驗與購物流程",
        "component": "frontend",
        "specialties": ["結帳流程", "前端狀態管理", "跨瀏覽器介面"],
        "similar_cases": 21,
        "resolved_90_days": 36,
        "open_items": 6,
        "availability": "本週可排入",
        "status": "active",
    },
    {
        "email": "backend-team@example.com",
        "name": "後端平台團隊",
        "role": "服務維運團隊",
        "team": "後端平台與交易服務",
        "component": "backend",
        "specialties": ["交易 API", "訊息佇列", "服務穩定性"],
        "similar_cases": 18,
        "resolved_90_days": 32,
        "open_items": 7,
        "availability": "需確認優先度",
        "status": "active",
    },
    {
        "email": "data-team@example.com",
        "name": "資料平台團隊",
        "role": "資料平台團隊",
        "team": "資料庫與資料一致性",
        "component": "database",
        "specialties": ["資料一致性", "交易回復", "效能調校"],
        "similar_cases": 15,
        "resolved_90_days": 24,
        "open_items": 4,
        "availability": "本週可承接",
        "status": "active",
    },
    {
        "email": "ui-platform@example.com",
        "name": "介面平台團隊",
        "role": "平台支援團隊",
        "team": "介面系統與響應式設計",
        "component": "frontend",
        "specialties": ["設計系統", "響應式版面", "無障礙介面"],
        "similar_cases": 12,
        "resolved_90_days": 29,
        "open_items": 5,
        "availability": "本週可承接",
        "status": "active",
    },
    {
        "email": "checkout-team@example.com",
        "name": "結帳與付款團隊",
        "role": "產品功能團隊",
        "team": "結帳與付款體驗",
        "component": "checkout",
        "specialties": ["結帳流程", "付款狀態", "訂單同步"],
        "similar_cases": 24,
        "resolved_90_days": 38,
        "open_items": 5,
        "availability": "本週可承接",
        "status": "active",
    },
]


def load_scenarios() -> dict:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_record_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def load_imported_tickets() -> list[dict[str, Any]]:
    return _read_record_list(ADMIN_TICKETS_PATH)


def load_imported_members() -> list[dict[str, Any]]:
    return _read_record_list(ADMIN_MEMBERS_PATH)


def _merge_records(
    defaults: list[dict[str, Any]],
    imported: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*defaults, *imported]:
        identifier = str(item.get(key) or "").strip().lower()
        if identifier:
            merged[identifier] = dict(item)
    return list(merged.values())


def all_historical_tickets() -> list[dict[str, Any]]:
    return _merge_records(
        DEMO_HISTORICAL_TICKETS,
        load_imported_tickets(),
        key="ticket_id",
    )


def all_team_members() -> list[dict[str, Any]]:
    return _merge_records(
        DEMO_TEAM_MEMBERS,
        load_imported_members(),
        key="email",
    )


def current_component_owners() -> dict[str, str]:
    owners = dict(DEMO_COMPONENT_OWNERS)
    for member in load_imported_members():
        if str(member.get("status") or "active").lower() != "active":
            continue
        email = str(member.get("email") or "").strip()
        component = _normalize_component(str(member.get("component") or ""))
        if email and component != "unknown":
            owners[component] = email
    return owners


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[\n|;、]+", str(value or "")) if item.strip()]


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return default


def normalize_imported_ticket(row: dict[str, Any]) -> dict[str, Any]:
    ticket_id = str(row.get("ticket_id") or row.get("id") or row.get("編號") or "").strip()
    title = str(row.get("title") or row.get("標題") or "").strip()
    description = str(row.get("description") or row.get("描述") or row.get("問題描述") or "").strip()
    if not ticket_id or not title or not description:
        raise ValueError("Ticket 必須包含 ticket_id、title 與 description")
    component = _normalize_component(str(row.get("component") or row.get("影響功能") or ""))
    return {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "product": str(row.get("product") or row.get("產品") or "未分類產品").strip(),
        "component": component,
        "priority": str(row.get("priority") or row.get("優先級") or "P3").strip().upper(),
        "assignee": str(row.get("assignee") or row.get("owner") or row.get("負責人") or "").strip(),
        "status": str(row.get("status") or row.get("狀態") or "已匯入").strip(),
        "environment": str(row.get("environment") or row.get("環境") or "尚未提供").strip(),
        "steps_to_reproduce": _coerce_list(row.get("steps_to_reproduce") or row.get("steps") or row.get("重現步驟")),
        "expected_behavior": str(row.get("expected_behavior") or row.get("expected") or row.get("預期結果") or "尚未提供").strip(),
        "actual_behavior": str(row.get("actual_behavior") or row.get("actual") or row.get("實際結果") or description).strip(),
        "error_message": str(row.get("error_message") or row.get("log") or row.get("錯誤訊息") or "尚未提供").strip(),
        "imported_at": datetime.now(UTC).isoformat(),
    }


def normalize_imported_member(row: dict[str, Any]) -> dict[str, Any]:
    email = str(row.get("email") or row.get("assignee") or row.get("帳號") or "").strip().lower()
    name = str(row.get("name") or row.get("姓名") or row.get("團隊名稱") or "").strip()
    component = _normalize_component(str(row.get("component") or row.get("影響功能") or ""))
    if not email or not name or component == "unknown":
        raise ValueError("團隊成員必須包含 email、name 與 component")
    status = str(row.get("status") or row.get("狀態") or "active").strip().lower()
    return {
        "email": email,
        "name": name,
        "role": str(row.get("role") or row.get("職務") or "軟體工程師").strip(),
        "team": str(row.get("team") or row.get("團隊") or "產品工程團隊").strip(),
        "component": component,
        "specialties": _coerce_list(row.get("specialties") or row.get("專長")),
        "similar_cases": _coerce_int(row.get("similar_cases") or row.get("相似案件數")),
        "resolved_90_days": _coerce_int(row.get("resolved_90_days") or row.get("近90日完成數")),
        "open_items": _coerce_int(row.get("open_items") or row.get("目前待辦數")),
        "availability": str(row.get("availability") or row.get("可用狀態") or "需確認排程").strip(),
        "status": "inactive" if status in {"inactive", "停用", "離職"} else "active",
        "imported_at": datetime.now(UTC).isoformat(),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_record_list(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write_text(path, json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def _refresh_admin_analysis_sources() -> None:
    history_rows = [ticket for ticket in all_historical_tickets() if str(ticket.get("assignee") or "").strip()]
    _atomic_write_text(
        ADMIN_ASSIGNEE_HISTORY_PATH,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history_rows),
    )
    roster = [
        {
            "email": member.get("email", ""),
            "name": member.get("name", ""),
            "status": member.get("status", "active"),
            "active": member.get("status", "active") == "active",
        }
        for member in all_team_members()
    ]
    _write_record_list(ADMIN_ROSTER_PATH, roster)


def import_admin_records(dataset: str, records: Any, *, mode: str = "append") -> dict[str, Any]:
    if dataset not in {"tickets", "members"}:
        raise ValueError("不支援的資料類型")
    if not isinstance(records, list) or not records:
        raise ValueError("請提供至少一筆資料")
    if len(records) > MAX_IMPORT_RECORDS:
        raise ValueError(f"單次最多匯入 {MAX_IMPORT_RECORDS} 筆資料")

    normalizer = normalize_imported_ticket if dataset == "tickets" else normalize_imported_member
    key = "ticket_id" if dataset == "tickets" else "email"
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            errors.append(f"第 {index} 筆不是有效的資料列")
            continue
        try:
            normalized.append(normalizer(row))
        except ValueError as exc:
            errors.append(f"第 {index} 筆：{exc}")
    if errors:
        raise ValueError("；".join(errors[:5]))

    path = ADMIN_TICKETS_PATH if dataset == "tickets" else ADMIN_MEMBERS_PATH
    existing = [] if mode == "replace" else _read_record_list(path)
    merged = _merge_records(existing, normalized, key=key)
    _write_record_list(path, merged)
    _refresh_admin_analysis_sources()
    return {
        "status": "imported",
        "dataset": dataset,
        "imported_count": len(normalized),
        "stored_count": len(merged),
        "overview": admin_overview(),
    }


def _admin_latest_update() -> str:
    mtimes = [
        path.stat().st_mtime
        for path in (ADMIN_TICKETS_PATH, ADMIN_MEMBERS_PATH)
        if path.exists()
    ]
    if not mtimes:
        return "內建示範資料"
    return datetime.fromtimestamp(max(mtimes), tz=UTC).isoformat()


def admin_overview() -> dict[str, Any]:
    tickets = all_historical_tickets()
    members = all_team_members()
    components = {
        str(item.get("component") or "unknown")
        for item in [*tickets, *members]
        if str(item.get("component") or "").strip()
    }
    return {
        "summary": {
            "ticket_count": len(tickets),
            "member_count": len([item for item in members if item.get("status", "active") == "active"]),
            "component_count": len(components),
            "imported_ticket_count": len(load_imported_tickets()),
            "imported_member_count": len(load_imported_members()),
            "latest_update": _admin_latest_update(),
        },
        "tickets": tickets[:ADMIN_PREVIEW_LIMIT],
        "members": members[:ADMIN_PREVIEW_LIMIT],
        "limits": {"preview": ADMIN_PREVIEW_LIMIT, "import": MAX_IMPORT_RECORDS},
    }


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

        if path == "/api/admin/overview":
            self._send_json(admin_overview())
            return

        if path.startswith("/api/runs/"):
            parts = [unquote(part) for part in path.removeprefix("/api/runs/").split("/") if part]
            if len(parts) != 2:
                self.send_error(404, "Artifact not found")
                return
            run_id, filename = parts
            if not RUN_ID_PATTERN.fullmatch(run_id) or filename not in RUN_ARTIFACTS:
                self.send_error(403, "Forbidden")
                return
            artifact = DEMO_RUNS_ROOT / run_id / filename
            if not artifact.exists():
                self.send_error(404, "Artifact not found")
                return
            self._send_file(artifact, download=True)
            return

        if path == "/":
            self._send_file(STATIC_ROOT / "index.html")
            return

        if path == "/admin":
            self._send_file(STATIC_ROOT / "admin.html")
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
        if path not in {"/api/analyze", "/api/assign", "/api/admin/import"}:
            self.send_error(404, "Endpoint not found")
            return

        try:
            payload = self._read_json_body()
            if payload is None:
                return

            if path == "/api/admin/import":
                dataset = str(payload.get("dataset") or "").strip().lower()
                mode = str(payload.get("mode") or "append").strip().lower()
                self._send_json(import_admin_records(dataset, payload.get("records"), mode=mode))
                return

            if path == "/api/assign":
                run_id = str(payload.get("run_id") or "")
                selected_assignee = str(payload.get("selected_assignee") or "").strip()
                decision = str(payload.get("decision") or "accept").strip()
                if not RUN_ID_PATTERN.fullmatch(run_id) or not selected_assignee:
                    self.send_error(400, "A valid run_id and selected_assignee are required")
                    return
                self._send_json(save_assignee_selection(run_id, selected_assignee, decision))
                return

            text = str(payload.get("text") or "")
            if not text.strip():
                self.send_error(400, "Input text is required")
                return
            analysis = analyze_text(
                text,
                duplicate_decision=str(payload.get("duplicate_decision") or ""),
                selected_duplicate_id=str(payload.get("selected_duplicate_id") or ""),
            )
            run_id = normalize_run_id(str(payload.get("run_id") or ""))
            self._send_json(persist_analysis(analysis, run_id=run_id))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - demo endpoint guard
            self._send_json({"error": f"Analysis failed: {exc}"}, status=500)

    def _read_json_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            self.send_error(413, f"Request body must be at most {MAX_REQUEST_BYTES} bytes")
            return None
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _send_json(self, payload: dict, *, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, *, download: bool = False) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def analyze_text(text: str, *, duplicate_decision: str = "", selected_duplicate_id: str = "") -> dict[str, Any]:
    has_imported_tickets = ADMIN_TICKETS_PATH.exists()
    has_imported_members = ADMIN_MEMBERS_PATH.exists()
    config = PipelineConfig(
        project_root=PROJECT_ROOT,
        duplicate_threshold=0.60,
        duplicate_top_k=10,
        component_owner_mapping=current_component_owners(),
        assignee_dataset_path=ADMIN_ASSIGNEE_HISTORY_PATH if has_imported_tickets else None,
        assignee_active_roster_path=ADMIN_ROSTER_PATH if has_imported_tickets or has_imported_members else None,
    )
    historical_tickets = all_historical_tickets()
    structured_ticket = normalize_structured_ticket(parse_user_input(text))
    matching_ticket = dict(structured_ticket)
    matching_ticket["title"] = _with_demo_keywords(str(structured_ticket.get("title", "")), text)
    matching_ticket["description"] = _with_demo_keywords(str(structured_ticket.get("description", "")), text)
    duplicate_result = DuplicateDetector(config=config).detect(matching_ticket, historical_tickets)
    duplicate_result["top_k_candidates"] = enrich_duplicate_candidates(
        duplicate_result.get("top_k_candidates", []),
        historical_tickets=historical_tickets,
    )
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
    assignee_result = enrich_assignee_candidates(assignee_result, structured_ticket)
    result["status"] = "completed"
    result["priority"] = priority_result
    result["assignee"] = assignee_result
    return result


def enrich_duplicate_candidates(
    candidates: list[dict[str, Any]],
    *,
    historical_tickets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    source = historical_tickets if historical_tickets is not None else all_historical_tickets()
    historical_by_id = {str(item.get("ticket_id")): item for item in source}
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        ticket_id = str(candidate.get("ticket_id", ""))
        historical = historical_by_id.get(ticket_id, {})
        enriched.append(
            {
                **candidate,
                "description": historical.get("description", ""),
                "product": historical.get("product", ""),
                "status": historical.get("status", "待確認"),
                "environment": historical.get("environment", "尚未提供"),
                "steps_to_reproduce": historical.get("steps_to_reproduce", []),
                "expected_behavior": historical.get("expected_behavior", "尚未提供"),
                "actual_behavior": historical.get("actual_behavior", historical.get("description", "尚未提供")),
                "error_message": historical.get("error_message", "尚未提供"),
                **DEMO_TICKET_DETAILS.get(ticket_id, {}),
            }
        )
    return enriched


def enrich_assignee_candidates(
    assignee_result: dict[str, Any],
    structured_ticket: dict[str, Any],
) -> dict[str, Any]:
    component = str(structured_ticket.get("component") or "unknown")
    details_by_assignee = {
        str(item.get("assignee")): dict(item)
        for item in assignee_result.get("candidate_details", [])
    }
    ranked = [str(item) for item in assignee_result.get("ranked_candidates", [])]
    enriched_details = []
    for index, assignee in enumerate(ranked):
        detail = details_by_assignee.get(assignee, {"assignee": assignee, "score": 0.0, "signals": []})
        enriched_details.append({**detail, **build_demo_assignee_profile(assignee, component, detail, index)})
    return {**assignee_result, "candidate_details": enriched_details}


def build_demo_assignee_profile(
    assignee: str,
    component: str,
    detail: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    component_profiles = {
        "authentication": ("身分驗證小組", ["登入流程", "Token／Session", "OAuth 與權限"]),
        "checkout": ("結帳與付款小組", ["結帳流程", "付款狀態", "訂單同步"]),
        "frontend": ("前端體驗小組", ["購物流程", "狀態管理", "跨瀏覽器介面"]),
        "backend": ("後端服務小組", ["交易 API", "非同步事件", "服務穩定性"]),
        "database": ("資料平台小組", ["資料一致性", "交易回復", "效能調校"]),
        "email": ("訊息服務小組", ["郵件派送", "通知佇列", "失敗重試"]),
        "search": ("搜尋服務小組", ["搜尋索引", "快取更新", "結果排序"]),
        "api": ("平台 API 小組", ["API 設計", "檔案處理", "存取控制"]),
        "unknown": ("產品工程小組", ["問題分析", "跨系統除錯", "服務維運"]),
    }
    team_profiles = {
        "auth-team@example.com": {
            "role": "功能維運團隊",
            "team": "身分驗證與登入",
            "specialties": ["登入流程", "Token／Session", "OAuth 與權限"],
            "similar_cases": 26,
            "resolved_90_days": 41,
            "open_items": 8,
            "availability": "需由團隊排程",
        },
        "frontend-team@example.com": {
            "role": "功能維運團隊",
            "team": "前端體驗與購物流程",
            "specialties": ["結帳流程", "前端狀態管理", "跨瀏覽器介面"],
            "similar_cases": 21,
            "resolved_90_days": 36,
            "open_items": 6,
            "availability": "本週可排入",
        },
        "backend-team@example.com": {
            "role": "服務維運團隊",
            "team": "後端平台與交易服務",
            "specialties": ["交易 API", "訊息佇列", "服務穩定性"],
            "similar_cases": 18,
            "resolved_90_days": 32,
            "open_items": 7,
            "availability": "需確認優先度",
        },
        "data-team@example.com": {
            "role": "資料平台團隊",
            "team": "資料庫與資料一致性",
            "specialties": ["資料一致性", "交易回復", "效能調校"],
            "similar_cases": 15,
            "resolved_90_days": 24,
            "open_items": 4,
            "availability": "本週可承接",
        },
        "ui-platform@example.com": {
            "role": "平台支援團隊",
            "team": "介面系統與響應式設計",
            "specialties": ["設計系統", "響應式版面", "無障礙介面"],
            "similar_cases": 12,
            "resolved_90_days": 29,
            "open_items": 5,
            "availability": "本週可承接",
        },
        "checkout-team@example.com": {
            "role": "產品功能團隊",
            "team": "結帳與付款體驗",
            "specialties": ["結帳流程", "付款狀態", "訂單同步"],
            "similar_cases": 24,
            "resolved_90_days": 38,
            "open_items": 5,
            "availability": "本週可承接",
        },
    }
    member = next(
        (item for item in all_team_members() if str(item.get("email") or "").lower() == assignee.lower()),
        None,
    )
    if member:
        profile = {
            "name": str(member.get("name") or assignee),
            "role": str(member.get("role") or "團隊成員"),
            "team": str(member.get("team") or "產品工程團隊"),
            "specialties": _coerce_list(member.get("specialties")),
            "similar_cases": _coerce_int(member.get("similar_cases")),
            "resolved_90_days": _coerce_int(member.get("resolved_90_days")),
            "open_items": _coerce_int(member.get("open_items")),
            "availability": str(member.get("availability") or "需確認排程"),
        }
    elif assignee in team_profiles:
        profile = dict(team_profiles[assignee])
    else:
        digits = re.search(r"(\d+)$", assignee)
        numeric_id = int(digits.group(1)) if digits else sum(ord(char) for char in assignee)
        team, specialties = component_profiles.get(component, component_profiles["unknown"])
        roles = ["軟體工程師", "資深軟體工程師", "全端工程師", "平台工程師"]
        open_items = 2 + numeric_id % 6
        profile = {
            "role": roles[numeric_id % len(roles)],
            "team": team,
            "specialties": specialties,
            "similar_cases": 4 + numeric_id % 13,
            "resolved_90_days": 9 + numeric_id % 18,
            "open_items": open_items,
            "availability": (
                "可立即承接" if open_items <= 3 else "本週可承接" if open_items <= 5 else "目前工作量偏高"
            ),
        }

    signals = set(detail.get("signals", []))
    if "component_owner_mapping" in signals:
        match_reason = "目前負責此功能，對相關程式與既有處理方式最熟悉。"
    elif "component_history" in signals or "product_component_history" in signals:
        match_reason = "近期處理過同一功能的問題，具備直接相關的除錯經驗。"
    elif "text_similarity" in signals:
        match_reason = "處理紀錄中包含多筆內容相近的問題，可較快掌握問題背景。"
    else:
        match_reason = "具備團隊整體處理經驗，但與本次問題的直接紀錄較少。"

    if int(profile["open_items"]) >= 7:
        caution = "目前待辦較多，確認前建議先核對可投入時間。"
    elif not signals.intersection({"component_owner_mapping", "component_history", "product_component_history", "text_similarity"}):
        caution = "相關性主要來自整體處理經驗，建議再確認是否熟悉本次功能。"
    else:
        caution = "目前沒有明顯限制，仍建議由團隊確認實際排程。"

    return {
        **profile,
        "match_reason": match_reason,
        "caution": caution,
        "rank": rank + 1,
    }


def normalize_run_id(value: str) -> str:
    candidate = value.strip()
    if RUN_ID_PATTERN.fullmatch(candidate):
        return candidate
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"demo-{timestamp}-{uuid4().hex[:8]}"


def persist_analysis(
    analysis: dict[str, Any],
    *,
    run_id: str = "",
    output_root: Path = DEMO_RUNS_ROOT,
) -> dict[str, Any]:
    safe_run_id = normalize_run_id(run_id)
    run_dir = output_root / safe_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, Any]] = [
        ("structured_ticket.json", analysis.get("structured_ticket", {})),
        ("duplicate_detection.json", analysis.get("duplicate", {})),
    ]
    if "priority" in analysis:
        payloads.append(("priority_prediction.json", analysis["priority"]))
    if "assignee" in analysis:
        payloads.append(("assignee_recommendation.json", analysis["assignee"]))

    artifact_names = [name for name, _ in payloads]
    enriched = dict(analysis)
    enriched["run_id"] = safe_run_id
    enriched["saved_at"] = datetime.now(UTC).isoformat()
    enriched["artifacts"] = [*artifact_names, "analysis_result.json"]

    for filename, payload in payloads:
        _write_json(run_dir / filename, payload)
    _write_json(run_dir / "analysis_result.json", enriched)
    return enriched


def save_assignee_selection(
    run_id: str,
    selected_assignee: str,
    decision: str,
    *,
    output_root: Path = DEMO_RUNS_ROOT,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run id")
    run_dir = output_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError("Analysis run does not exist")
    analysis_path = run_dir / "analysis_result.json"
    try:
        complete_result = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Analysis result is unavailable") from exc
    if not isinstance(complete_result, dict):
        raise ValueError("Analysis result is unavailable")
    normalized_decision = decision if decision in {"accept", "override", "manual"} else "override"
    selection = {
        "run_id": run_id,
        "selected_assignee": selected_assignee,
        "decision": normalized_decision,
        "selected_at": datetime.now(UTC).isoformat(),
    }
    artifacts = complete_result.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
    if "assignee_selection.json" not in artifacts:
        artifacts = [*artifacts, "assignee_selection.json"]
    complete_result["assignee_selection"] = selection
    complete_result["artifacts"] = artifacts
    _write_json(run_dir / "assignee_selection.json", selection)
    _write_json(analysis_path, complete_result)
    return {"status": "saved", **selection}


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parse_user_input(text: str) -> dict[str, Any]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in normalized.splitlines()]
    title = next((line for line in lines if line and not _is_label_line(line)), "RAW-DEMO")
    environment = _field(normalized, ("Environment", "環境", "執行環境"))
    product = _field(normalized, ("Product", "產品", "專案", "Project"), default="Demo Project")
    component = _normalize_component(_field(normalized, ("Component", "元件", "模組", "影響功能", "功能")))
    severity = _normalize_severity(_field(normalized, ("Severity", "嚴重程度", "嚴重性", "重要性")))
    expected = _field(normalized, ("Expected", "預期結果", "預期", "期望結果"))
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
        r"^(?:Steps to reproduce|Steps|重現步驟|操作步驟)\s*[:：]?\s*([\s\S]+?)(?=\n\s*(?:Expected|預期|Actual|實際|Log|錯誤|日誌)\s*[:：]|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
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
    return bool(re.match(r"^[A-Za-z ]+[:：]|^(?:回報者|元件|模組|影響功能|功能|嚴重程度|嚴重性|重要性|執行環境|環境|預期結果|預期|實際結果|實際|錯誤|日誌|重現步驟)[:：]", line))


def _normalize_component(value: str) -> str:
    raw = value.strip().lower()
    if any(term in raw for term in ("結帳", "付款", "訂單", "checkout", "payment", "order")):
        return "checkout"
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
    if any(term in lower for term in ("checkout", "cart", "payment", "order", "discount", "結帳", "購物車", "付款", "訂單", "折扣")):
        return "checkout"
    if any(term in lower for term in ("login", "token", "auth", "password", "登入", "權杖", "密碼", "驗證", "認證")):
        return "authentication"
    if any(term in lower for term in ("button", "ui", "frontend", "按鈕", "畫面", "前端")):
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
