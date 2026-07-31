# Phase 9：v3 實驗協定與改善管線

此目錄承接 v2 的 No-Go 結論，提供 v3 的實驗登錄、資料洩漏稽核、嚴格跨窗 policy gate 與新 holdout 樣本數規劃。它不會把 2022 Q2 重新包裝成未看過的 holdout，也不會自動核准部署。

## 已落地的保護

- `protected_holdouts.json` 將 2022 Q2 永久標記為 `diagnosis_only_not_model_or_policy_selection`。
- `build_assignee_temporal_protocol.py` 強制分開 ranker fit／selection、calibrator fit、open-set fit 與至少兩個 policy-selection window。
- protocol audit 會檢查 ticket ID、完全相同文字、duplicate family、SimHash 近重複文字與 owner event 的跨切分重疊。
- 只登記 protected／sealed holdout manifest 與 SHA-256，不讀取或複製 holdout label。
- label availability 必須來自真正 assignment event timestamp；`last_change_time` 只列為 proxy，protocol gate 不會因此通過。
- 新 policy search 預設要求每個開發窗同時通過 unseen Wilson 上界、auto accuracy Wilson 下界、最小樣本、component floor 與兩個獨立候選來源 family。
- `train_assignee_ltr.py --new-holdout` 已停用，正式 holdout 只能交給一次性 evaluator。
- v3 候選新增 reporter／component-family 與具 report-time provenance 的 file ownership；修復後才取得的 file path 不得當作路由特徵。
- duplicate family 在 v3 只作嚴格跨切分排除，不作候選來源，避免使用事後 duplicate 判定洩漏未來資訊；v2 replay 維持舊語意。
- ranker 會比較 pointwise 模型與每張 ticket 最多十個 hard negatives 的 pairwise logistic，仍由最差開發窗決定是否選用。
- `candidate_source_ablation_report.json` 會分開報告來源與 evidence family 的保守移除下界，相關 component 訊號不會冒充多個獨立來源。

## 建立 v3 protocol

先準備互斥且依時間排序的開發窗，再執行：

```bash
python3 assignee_triage_accuracy/scripts/build_assignee_temporal_protocol.py \
  --experiment-id assignee-routing-v3-001 \
  --partition ranker-fit:ranker_fit=path/to/ranker_fit.jsonl \
  --partition ranker-selection:ranker_selection=path/to/ranker_selection.jsonl \
  --partition calibrator-fit:calibrator_fit=path/to/calibrator_fit.jsonl \
  --partition open-set-fit:open_set_fit=path/to/open_set_fit.jsonl \
  --partition policy-1:policy_selection=path/to/policy_window_1.jsonl \
  --partition policy-2:policy_selection=path/to/policy_window_2.jsonl \
  --protected-holdout-manifest assignee_triage_accuracy/phase8_operational_readiness/data/bmo_public_future_2022q2_sealed_manifest.json \
  --planned-holdout-start 2023-01-01T00:00:00Z \
  --output assignee_triage_accuracy/phase9_v3_protocol/experiments/assignee-routing-v3-001/protocol_manifest.json
```

輸出會記錄 git commit、dirty state、seed、每個輸入 SHA-256、時間邊界、row hash、資料稽核、protected holdout 與待產生 artifact registry。工作樹未提交、label timestamp 仍使用 proxy、資料有洩漏或新 holdout 不晚於所有開發窗時，協定仍可留下可追蹤的阻擋紀錄，但 `protocol_gate.passed=false`。

## Holdout 樣本規劃

```bash
python3 assignee_triage_accuracy/scripts/plan_assignee_holdout_power.py \
  --expected-unseen-auto-error-rate 0.02 \
  --expected-unseen-owner-fraction 0.07
```

工具使用與 release gate 相同的雙側 Wilson 95% 上界。這是依預期事件率做的規劃，不取代正式 holdout 的實際觀測與一次性驗收。

## 尚不能由離線程式完成

- 取得晚於所有 v3 開發窗、從未讀取 label 的新 sealed holdout。
- 維護者簽核當期 active／inactive／cold-start roster。
- 累積真實 live shadow 事件與連續兩週證據。
- 有權限操作人明示核准，再從 5% rollout 開始。
