from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo"
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(DEMO_ROOT), str(SRC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import demo_app


class DemoAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = {item["id"]: item for item in demo_app.load_scenarios()["scenarios"]}

    def test_initial_analysis_requires_duplicate_review(self) -> None:
        scenario = self.scenarios["integrated-triage-flow"]

        result = demo_app.analyze_text(scenario["user_input"])

        self.assertEqual(result["status"], "needs_duplicate_review")
        self.assertEqual(result["structured_ticket"]["ticket_id"], "RAW-202")
        self.assertEqual(result["structured_ticket"]["product"], "Checkout UI")
        candidate = result["duplicate"]["top_k_candidates"][0]
        self.assertTrue(candidate["description"])
        self.assertTrue(candidate["environment"])
        self.assertTrue(candidate["steps_to_reproduce"])
        self.assertTrue(candidate["actual_behavior"])
        self.assertTrue(candidate["status"])
        self.assertNotIn("priority", result)
        self.assertNotIn("assignee", result)

    def test_confirmed_duplicate_stops_later_stages(self) -> None:
        scenario = self.scenarios["duplicate-login-token"]

        result = demo_app.analyze_text(
            scenario["user_input"],
            duplicate_decision="duplicate",
            selected_duplicate_id="BUG-038",
        )

        self.assertEqual(result["status"], "duplicate_confirmed")
        self.assertEqual(result["duplicate_of"], "BUG-038")
        self.assertNotIn("priority", result)
        self.assertNotIn("assignee", result)

    def test_confirmed_non_duplicate_runs_priority_and_assignee(self) -> None:
        scenario = self.scenarios["integrated-triage-flow"]

        result = demo_app.analyze_text(scenario["user_input"], duplicate_decision="not_duplicate")

        self.assertEqual(result["status"], "completed")
        self.assertRegex(result["priority"]["predicted_priority"], r"^P[1-5]$")
        self.assertTrue(result["assignee"]["ranked_candidates"])
        candidate = result["assignee"]["candidate_details"][0]
        self.assertTrue(candidate["role"])
        self.assertTrue(candidate["team"])
        self.assertTrue(candidate["specialties"])
        self.assertGreaterEqual(candidate["similar_cases"], 0)
        self.assertGreaterEqual(candidate["resolved_90_days"], 0)
        self.assertGreaterEqual(candidate["open_items"], 0)
        self.assertTrue(candidate["availability"])
        self.assertTrue(candidate["match_reason"])
        self.assertTrue(candidate["caution"])

    def test_chinese_ticket_parses_labels_and_steps(self) -> None:
        scenario = self.scenarios["manual-triage-zh"]

        result = demo_app.analyze_text(scenario["user_input"], duplicate_decision="not_duplicate")

        ticket = result["structured_ticket"]
        self.assertEqual(ticket["ticket_id"], "RAW-303")
        self.assertEqual(ticket["product"], "營運後台")
        self.assertEqual(ticket["component"], "報表匯出")
        self.assertEqual(ticket["steps_to_reproduce"], ["尚未穩定重現"])
        self.assertTrue(result["assignee"]["needs_manual_triage"])

    def test_chinese_impact_function_drives_checkout_assignee_profiles(self) -> None:
        text = """Ticket ID: WEB-1042
使用者完成付款後，訂單仍顯示為待付款

產品：線上購物平台
影響功能：結帳與訂單管理
嚴重程度：高
執行環境：Windows 11、Chrome 127、網站版本 4.2.0

問題描述：付款成功後，訂單狀態仍停留在待付款。

重現步驟：
1. 登入會員帳號
2. 完成信用卡付款

預期結果：訂單狀態更新為已付款。
實際結果：訂單狀態持續顯示為待付款。
"""

        result = demo_app.analyze_text(text, duplicate_decision="not_duplicate")

        self.assertEqual(result["structured_ticket"]["component"], "checkout")
        self.assertEqual(result["structured_ticket"]["severity"], "major")
        self.assertEqual(result["structured_ticket"]["expected_behavior"], "訂單狀態更新為已付款。")
        self.assertTrue(
            all(
                detail["team"] in {"結帳與付款小組", "結帳與付款體驗"}
                for detail in result["assignee"]["candidate_details"]
            )
        )

    def test_analysis_and_final_selection_are_saved_as_json_checkpoints(self) -> None:
        analysis = {
            "status": "completed",
            "structured_ticket": {"ticket_id": "RAW-TEST"},
            "duplicate": {"is_duplicate": False},
            "priority": {"predicted_priority": "P3"},
            "assignee": {"assignee": "owner@example.com"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            persisted = demo_app.persist_analysis(analysis, run_id="run-test", output_root=output_root)
            selection = demo_app.save_assignee_selection(
                "run-test",
                "owner@example.com",
                "accept",
                output_root=output_root,
            )
            saved_files = {path.name for path in (output_root / "run-test").iterdir()}
            complete_result = json.loads(
                (output_root / "run-test" / "analysis_result.json").read_text(encoding="utf-8")
            )

        self.assertEqual(persisted["run_id"], "run-test")
        self.assertEqual(selection["status"], "saved")
        self.assertEqual(complete_result["assignee_selection"]["selected_assignee"], "owner@example.com")
        self.assertIn("assignee_selection.json", complete_result["artifacts"])
        self.assertEqual(
            saved_files,
            {
                "structured_ticket.json",
                "duplicate_detection.json",
                "priority_prediction.json",
                "assignee_recommendation.json",
                "assignee_selection.json",
                "analysis_result.json",
            },
        )

    def test_admin_import_updates_duplicate_and_assignee_sources(self) -> None:
        original_paths = (
            demo_app.ADMIN_DATA_ROOT,
            demo_app.ADMIN_TICKETS_PATH,
            demo_app.ADMIN_MEMBERS_PATH,
            demo_app.ADMIN_ASSIGNEE_HISTORY_PATH,
            demo_app.ADMIN_ROSTER_PATH,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo_app.ADMIN_DATA_ROOT = root
            demo_app.ADMIN_TICKETS_PATH = root / "imported_tickets.json"
            demo_app.ADMIN_MEMBERS_PATH = root / "imported_members.json"
            demo_app.ADMIN_ASSIGNEE_HISTORY_PATH = root / "assignee_history.jsonl"
            demo_app.ADMIN_ROSTER_PATH = root / "assignee_roster.json"
            try:
                demo_app.import_admin_records(
                    "tickets",
                    [
                        {
                            "ticket_id": "REPORT-900",
                            "title": "月報表缺少最後一天資料",
                            "description": "匯出的月報表沒有包含月份最後一天的交易紀錄",
                            "product": "營運後台",
                            "component": "報表匯出",
                            "assignee": "report-owner@example.com",
                        }
                    ],
                )
                imported = demo_app.import_admin_records(
                    "members",
                    [
                        {
                            "email": "report-owner@example.com",
                            "name": "報表功能負責人",
                            "team": "營運系統團隊",
                            "component": "報表匯出",
                            "specialties": "Excel 匯出|資料查詢",
                            "open_items": 3,
                        }
                    ],
                )

                self.assertEqual(imported["overview"]["summary"]["imported_ticket_count"], 1)
                self.assertEqual(imported["overview"]["summary"]["imported_member_count"], 1)

                text = """Ticket ID: REPORT-NEW
月報表缺少最後一天資料

Product: 營運後台
Component: 報表匯出
Severity: normal
Environment: Windows 11 / Chrome 127

Description: 匯出的月報表沒有包含月份最後一天的交易紀錄
Steps to reproduce:
1. 選擇整月日期
2. 匯出報表
Expected: 包含月底資料
Actual: 缺少月底資料
"""
                duplicate_result = demo_app.analyze_text(text)
                candidate_ids = [
                    item["ticket_id"]
                    for item in duplicate_result["duplicate"]["top_k_candidates"]
                ]
                self.assertIn("REPORT-900", candidate_ids)

                completed = demo_app.analyze_text(text, duplicate_decision="not_duplicate")
                self.assertIn("report-owner@example.com", completed["assignee"]["ranked_candidates"])
                profile = next(
                    item
                    for item in completed["assignee"]["candidate_details"]
                    if item["assignee"] == "report-owner@example.com"
                )
                self.assertEqual(profile["name"], "報表功能負責人")
            finally:
                (
                    demo_app.ADMIN_DATA_ROOT,
                    demo_app.ADMIN_TICKETS_PATH,
                    demo_app.ADMIN_MEMBERS_PATH,
                    demo_app.ADMIN_ASSIGNEE_HISTORY_PATH,
                    demo_app.ADMIN_ROSTER_PATH,
                ) = original_paths


if __name__ == "__main__":
    unittest.main()
