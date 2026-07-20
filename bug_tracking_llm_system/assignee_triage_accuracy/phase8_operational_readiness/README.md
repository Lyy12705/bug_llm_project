# Phase 8：負責人自動分流操作準備

本目錄保存 v2 的時間安全資料 manifest、候選 roster、模型／routing 報告、2022 Q2 一次性 holdout 證據，以及保持 fail-closed 的 deployment candidate。

## 已完成

- v2 時間協定：Ranker、calibration/open-set、policy selection、sealed holdout 完全分離。
- Primary auto routing：2022 Q2 正式 holdout 的 accuracy 87.79%、coverage 25.72%、unseen auto rate 4.00%，所有 hard gates 通過。
- Top-3 安全 fallback：90% confirmation 目標未達，功能已停用；非 auto 工單一律人工分流。
- Roster 技術盤點：165 active、99 inactive；未冒充組織簽核。
- Shadow 防偽：歷史回放／合成資料不能通過 live deployment gate。

## 尚需真實外部證據

1. 維護者把 2022 roster 當作稽核起點，提供目前時點的 reviewed active/inactive 清單，並以本人識別資料及當期 `--as-of` 執行 roster builder 的 `--reviewer` 與 `--confirm-organizational-review`；過期快照不能通過部署 gate。
2. 在真實 shadow 流程中，每張工單先執行 `recommend_assignee_rolling_shadow.py`；最終負責人確定後，以 `record_assignee_feedback.py --feedback-origin live_shadow --source-system ... --source-event-id ...` 記錄事件。
3. 累積至少 500 筆或 28 天後執行 `evaluate_assignee_shadow.py`。只有 provenance、accuracy、coverage、unseen 與 Wilson 下界全部通過，shadow gate 才會通過。
4. 重新建立 deployment bundle，最後由有權限的操作人明示 `--approve`。在此之前執行模式只能是 `shadow_only`。

## 主要證據

- `reports/2022q2_holdout/rolling_holdout_report.json`：一次性正式 holdout 結果。
- `reports/rolling_primary_v2/rolling_open_set_artifact.json`：Top-3 關閉後的 frozen primary policy。
- `roster/bmo_2022q2_candidate_roster_audit.json`：roster 技術稽核。
- `shadow/shadow_readiness_report.json`：目前 0 筆 live shadow 的 fail-closed 狀態。
- `deployment_candidate_v2/deployment_bundle.json`：整合後仍為 `research_only` 的原因與逐項 gate。
