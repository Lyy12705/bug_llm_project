# Phase 8：負責人自動分流操作準備

本目錄保存 v2 的時間安全資料 manifest、候選 roster、模型／routing 報告、2022 Q2 一次性 holdout 證據，以及保持 fail-closed 的 deployment candidate。

v2 證據已凍結；後續工程實作與新實驗登錄請使用 `../phase9_v3_protocol/`。2022 Q2 已在該處明確登記為 `diagnosis_only_not_model_or_policy_selection`。

## 已完成

- v2 時間協定：Ranker、calibration/open-set、policy selection、sealed holdout 完全分離。
- Primary auto routing：2022 Q2 正式 holdout 的 accuracy 87.79%、coverage 25.72%、unseen auto rate 點估計 4.00%；嚴格 gate 因 unseen Wilson 95% 上界 7.69% 高於 5% 而不通過。
- Top-3 安全 fallback：90% confirmation 目標未達，功能已停用；非 auto 工單一律人工分流。
- Roster 技術盤點：165 active、99 inactive；未冒充組織簽核。
- Shadow 防偽與營運 gate：歷史回放／合成資料不能通過；另要求連續兩週、unseen 信賴上界、完整 latency 與 p95 ≤ 5 秒。
- Operational guard：短效狀態綁定 bundle hash，只允許 0%→5%→10%→25%；過期、跳級或任何健康 gate 失敗立即 manual。
- Q2 post-holdout diagnostics：candidate miss 238、ranking miss 1,156；known-owner recall@30 89.32%，報告禁止用於調整 v2。

## 尚需真實外部證據

1. 維護者把 2022 roster 當作稽核起點，提供目前時點的 reviewed active/inactive 清單，並以本人識別資料及當期 `--as-of` 執行 roster builder 的 `--reviewer` 與 `--confirm-organizational-review`；過期快照不能通過部署 gate。
2. 在真實 shadow 流程中，每張工單先執行 `recommend_assignee_rolling_shadow.py`；最終負責人確定後，以 `record_assignee_feedback.py --feedback-origin live_shadow --source-system ... --source-event-id ...` 記錄事件。
3. 累積至少 500 筆或 28 天後執行 `evaluate_assignee_shadow.py`。只有 provenance、accuracy、coverage、unseen 雙側統計保護、連續週期與 latency 全部通過，shadow gate 才會通過。
4. 以新開發窗建立含 `minimum_candidate_source_count >= 2` 的下一版 frozen policy，並使用另一份更晚且未看過的 holdout；不得用 Q2 調整 v2。
5. 重新建立 deployment bundle，最後由有權限的操作人明示 `--approve`，再以 `evaluate_assignee_operational_guard.py` 從 5% 開始漸進 rollout。

## 主要證據

- `docs/reports/assignee_auto_routing_current_results_and_improvement_plan.md`：目前結果的完整判讀與後續改善步驟。
- `reports/2022q2_holdout/rolling_holdout_report.json`：一次性正式 holdout 結果。
- `reports/2022q2_holdout/routing_diagnostics.json`：只供錯誤定位、禁止調參的 post-holdout 診斷。
- `reports/rolling_primary_v2/rolling_open_set_artifact.json`：Top-3 關閉後的 frozen primary policy。
- `roster/bmo_2022q2_candidate_roster_audit.json`：roster 技術稽核。
- `shadow/shadow_readiness_report.json`：目前 0 筆 live shadow 的 fail-closed 狀態。
- `deployment_candidate_v2/deployment_bundle.json`：整合後仍為 `research_only` 的原因與逐項 gate。
- `operational/operational_guard_state.json`：目前 0%／kill-switch-on 的短效狀態快照；過期後必須重建，不能直接作 runtime 授權。
