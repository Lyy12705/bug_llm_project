# 負責人自動分流：目前結果判讀與改善計劃

- 報告日期：2026-07-22
- 評估階段：負責人推薦與選擇性自動分流
- 正式評估資料：2022 Q2 sealed temporal holdout
- 有效樣本數：2,803 筆
- 目前部署狀態：`research_only`
- 目前 rollout：0%，`manual_only`

## 0. 2026-07-22 工程實施更新

本輪已依本計劃落地可由程式完成的安全與模型開發補強；沒有重跑、改寫或重新解讀 2022 Q2 release evidence，部署狀態仍為 `research_only`。

- 新增 v3 experiment protocol／append-only registry builder，強制分離 ranker fit、ranker selection、calibrator fit、open-set fit 與至少兩個 policy-selection windows，並記錄 commit、dirty state、seed、輸入 SHA-256、row hash 與 artifact registry。
- 2022 Q2 已登記為 `diagnosis_only_not_model_or_policy_selection`；ranker 訓練入口不再接受 `--new-holdout`，正式封存資料只能交由一次性 holdout evaluator。
- 跨切分 audit 已從 ticket ID／完全相同文字擴充至 duplicate family、SimHash／token-Jaccard 近重複文字與 owner event ID。
- roster builder 已加入 alias chain canonicalization、reviewed roster canonicalization、assignment label timestamp provenance、owner frequency、component coverage 與資料缺失報告；`last_change_time` 僅能標記為 proxy，不能通過 v3 protocol gate。
- v3 candidate generator 新增 component-family 與 reporter-history 來源；相關 component 訊號會合併成同一 evidence family，不能重複計數冒充雙來源支持。舊 artifact 未宣告版本時仍固定走 v2 generator，保持 replay 相容。
- LTR 自動選模改為依最差時間窗，再看平均 MRR／Macro-F1；輸出 recall@10/30/50、各來源 recall、candidate/ranking miss 分群與每窗 95% recall／10% ranking-miss-reduction gate。
- rolling policy search 現在預設在每個開發窗同時強制 auto accuracy／coverage、最小 auto rows、accuracy Wilson 下界、unseen point／Wilson 上界、component point／confidence floor 與 `candidate_source_count >= 2`。
- 新增與正式 gate 相同的雙側 Wilson 95% holdout 樣本規劃工具。若預期 unseen 錯派率為 4%，最低約需 1,825 筆 unseen-owner；若為 2%，最低約需 203 筆。依目前約 7.135% unseen 比例換算，總 holdout 規模分別約為 25,579 與 2,846 筆。
- 第二輪補強加入具 report-time provenance 的 file ownership 候選與特徵；缺少 provenance 或修復後才知道的 file path 會被忽略。duplicate family 只用於 v3 split exclusion，不作路由訊號，並以版本開關保持 v2 replay 不變。
- LTR 新增每張 ticket 最多十個 hard negatives 的 pairwise logistic 比較器及 pairwise audit；自動選模仍依最差時間窗，因此 pairwise 若不穩定會被拒絕。另輸出 `candidate_source_ablation_report.json`，分開計算 source／source-family 的保守移除下界。

最新不含 SBERT、只使用既有已看過開發資料的 smoke run 顯示：pool 30 時 validation／later-development candidate recall@30 仍為 94.54%／93.51%；最差窗選模保留 HistGradientBoosting，其 Top-1 為 61.55%／53.25%，ranking miss 相對降低 14.19%／2.82%。新增 pairwise logistic 的 Top-1 僅 48.60%／38.18%，因此被穩健選模正確拒絕，沒有因新增模型就強制採用。目前資料也沒有具合格 provenance 的 reporter／file path 訊號，相關來源 ablation 為零。新 gate 繼續維持失敗：候選召回尚未在每窗達 95%，且較晚視窗的 ranking miss 降幅未達 10%。先前 pool 50 診斷可把 recall@50 提升至 95.81%／95.58%，但 recall@30 仍只有 94.80%／93.77%。這些數字只用於開發診斷，不是新 holdout 或 release 結論。

仍需外部或新資料才能完成的項目：晚於所有 v3 開發窗的新 sealed holdout、真實 assignment-event label timestamp、維護者簽核的當期 roster、live shadow 證據與有權限操作人核准。

### 0.1 Top-5 半自動候選版

已新增 `semi_auto_top5` 模式，把產品定位改為「系統提供五名候選，使用者可以指定其中一人，或明確按下自動後採用 Top-1」：

- 初始推薦固定為 `assignee=manual_triage`、`assignment_authorized=false`，不會在使用者操作前修改真實 assignee；回應會宣告 `select_top5_candidate`、`assign_top1` 與 `manual_triage` 三種操作。
- `select_top5_candidate` 允許使用者選第 1～5 名；`assign_top1` 只會採用原始第 1 名，而且兩者都必須由使用者明確確認。解析後會輸出 `assignment_authorized=true` 給 issue-tracker adapter 執行。
- `assign_top1` 會標記為 `top1_user_authorized`、`auto_top1_selected=true`、`autonomous_assignment=false`，不會被誤算成無人確認的全自動派工。
- 只有同一組凍結門檻在每個 policy-selection window 都達 Top-5 ≥ 85%，且通過最低 coverage、最低 100 筆、Wilson 95% 下界 ≥ 80%、open-set risk、至少兩個 candidate source family 與五名有效 roster 成員時，才顯示候選。
- 未達門檻、候選不足、來源不足、分布飄移、roster 含停用成員或 artifact gate 不完整時，`top5_candidates=[]` 並回到一般人工分流。
- feedback schema 已升級為 v4，除了 `selection_mode`、`selected_candidate_rank` 與 `selected_from_top5`，也記錄 `user_action`、`auto_top1_selected`、`user_authorized_assignment` 與 `autonomous_assignment`；shadow report 不會把使用者按下 Top-1 自動選項誤算成無人確認的自動派工。
- Eligible-owner feedback 另保存 `eligibility_required`、review／candidate-set 驗證結果與最終所選人資格驗證；shadow gate 若發現任何 Eligible-policy 顯示或選擇未通過資格檢查就失敗。

重要指標邊界：85% 是「正確 owner 位於五人清單內」的 Top-5 目標，不是第 1 名的準確率。三個開發窗合併後，顯示子集的 Top-1 為 71.54%、Top-5 為 86.95%。因此 `assign_top1` 是使用者選擇的便利操作，不具 85% 品質保證；若產品要把它升級為預設或無人確認的自動派工，必須另建 Top-1 policy、sealed holdout 與正式安全 gate。

#### Exact-owner 與 Eligible-owner 雙重判讀

為符合「同領域且確實有資格的人都可處理」的產品目標，系統已新增第二套集合式正確標籤，但不會改寫既有的 86.95%：

- `Exact-owner Top-5`：歷史資料中實際記錄的單一 assignee 是否位於前五名；目前 86.95% 屬於這個定義。
- `Eligible-owner Top-5`：前五名是否至少一人位於該工單當時有效的資格集合。集合定義為 active roster、人工審核的 product/component 能力、權限確認與有效期間的交集。
- `Top-5 candidate eligibility precision`：五個候選槽位中有多少比例確實合格。此指標防止「只有一人合格仍宣稱整份五人清單正確」。線上 Eligible-owner 模式採更嚴格規則：五人必須全部合格才顯示。
- 歷史 `components` 只代表曾經處理過的觀察紀錄，不能自動取得資格；缺 reviewer、權限確認、能力範圍、有效期間，或資格 snapshot 與預測時間不相符時，一律 fail closed。

目前尚不能提供新的 Eligible-owner 百分比，因為公共歷史資料沒有經維護者簽核且按時間版本化的能力／權限真值。系統已完成 schema、roster 合併、雙標籤評估、政策選擇、sealed holdout 要求、線上推薦與決策時二次檢查；待維護者建立涵蓋 Q1、Q2、Q3 與新 sealed holdout 的非重疊資格快照後，才能重算並判斷 Eligible-owner 是否達 85%。在此之前只能引用 Exact-owner 86.95%，不得把「同領域」推定為新準確率。

由於正式顯示條件要求五人全部合格，顯示子集的 Eligible-owner hit rate 理論上會是 100%；此時真正需要觀察的是 coverage、每張工單是否湊足五名合格人員、資格資料覆蓋率，以及同一顯示子集的 Exact-owner 命中率。換言之，85% Eligible hit 只能作最低檢查，不能單獨證明模型排序良好。

用本輪 v3 HistGradientBoosting 與三個後段開發窗重新執行選擇性政策後，數值 gate 可找到共同門檻：

| 開發窗 | 顯示 Top-5 筆數 | Coverage | Top-5 命中率 | Wilson 95% 下界 |
|---|---:|---:|---:|---:|
| 2021 Q1 | 1,276 / 2,671 | 47.77% | 85.11% | 83.05% |
| 2021 Q2 | 1,302 / 2,664 | 48.87% | 85.02% | 82.98% |
| 2021 Q3 | 1,291 / 2,696 | 47.89% | 90.70% | 89.00% |

這表示「約 48% 案件顯示 Top-5、其餘維持人工」在目前開發資料上可達使用者設定的 85% 目標，比對全部案件無條件顯示候選更合理。後端已能解析指定名次或使用者授權 Top-1，但實際寫入 Jira／Bugzilla 等 issue tracker 仍由外部 adapter 負責。不過這仍不是 release 結論：公共資料只有 `last_change_time` proxy，缺少真實 assignment-event timestamp，因此完整 development gate 仍為 false；而且還沒有供這個新政策使用的更晚 sealed holdout 與 live shadow 證據。現階段功能完整度可定位為「推薦、使用者決策與 feedback 契約已完成；tracker adapter、正式資料與營運驗收尚未完成」。

## 1. 執行摘要

目前模型在「通過高信心政策、準備自動派工的子集」上，準確率為 **87.79%（633/721）**，自動派工涵蓋率為 **25.72%**。這表示模型能篩出約四分之一的案件，並在離線 holdout 上達到 85% 的主要準確率目標。

但這個數字不代表所有工單都有 87.79% 準確率。若不採用拒答政策，直接對全部 2,803 筆案件取 Top-1，整體準確率只有 **43.13%**；known-owner 案件的 Top-1 也只有 **46.45%**。因此系統只能採用「高信心案件自動派工，其餘人工處理」的選擇性路由方式，不能對全部案件強制派工。

此外，unseen-owner 案件的自動錯派率點估計雖為 **4.00%（8/200）**，但 Wilson 95% 信賴區間上界為 **7.69%**，高於 5% 安全線。依目前強化後的嚴格 release gate，本版本判定為 **No-Go**，不能正式上線。

目前最準確的成果定位是：

> 可重現、具有選擇性自動派工能力，但尚未通過統計安全與營運驗收的研究候選版本。

## 2. 指標結果

### 2.1 選擇性自動派工

| 指標 | 結果 | 驗收條件 | 判讀 |
|---|---:|---:|---|
| Auto assignment rows | 721 / 2,803 | ≥ 250 | 通過 |
| Auto assignment accuracy | **87.79%** | ≥ 85% | 點估計通過 |
| Auto accuracy Wilson 95% CI | **85.20%～89.99%** | 下界 ≥ 80% | 通過，且下界略高於主要 85% 目標 |
| Auto assignment coverage | **25.72%** | ≥ 10% | 通過 |
| Manual triage rate | 74.28% | 無硬性上限 | 多數案件仍須人工處理 |
| Unseen-owner auto error | **4.00%（8/200）** | < 5% | 點估計通過 |
| Unseen-owner Wilson 95% CI | **2.04%～7.69%** | 上界 < 5% | **不通過** |

### 2.2 全部案件的推薦能力

| 指標 | 全部 2,803 筆 | Known-owner 2,603 筆 | 判讀 |
|---|---:|---:|---|
| Top-1 accuracy | **43.13%** | **46.45%** | 不可對全部案件直接派工 |
| Top-3 accuracy | 61.97% | 66.73% | 低於 90% confirmation 目標，功能維持停用 |
| Top-5 accuracy | 68.89% | 74.18% | 僅能表示候選名單品質 |
| Top-10 accuracy | 76.35% | 82.21% | 候選擴大後召回改善，但仍需人工選擇 |
| Candidate recall@30 | 82.95% | **89.32%** | known-owner 低於 95% 目標 |
| Candidate recall@50 | 84.37% | **90.86%** | 候選產生仍有改善空間 |
| MRR | 53.88% | 58.02% | 正確人選常未排在前段 |
| Macro-F1 | **12.44%** | 17.25% | 長尾 owner 表現偏弱，熱門 owner 影響明顯 |

### 2.3 Open-set 能力

| 指標 | 結果 | 判讀 |
|---|---:|---|
| AUROC | 65.13% | 有辨識訊號，但區分 known／unseen 的能力仍弱 |
| AUPRC | 11.36% | unseen-owner 為少數類別，precision-recall 表現偏弱 |
| Unknown recall @ 5% known FPR | 10.00% | 嚴格限制誤擋 known owner 時，只能找出少量 unseen owner |
| Frozen threshold unknown recall | 69.00% | 能攔截較多 unseen owner，但 known-owner false positive rate 達 47.41% |

這表示現有 open-set detector 的主要問題不是完全沒有訊號，而是無法同時取得高 unseen recall 與低 known-owner 誤擋率。

## 3. 錯誤結構

Post-holdout 診斷將 2,803 筆結果分為：

| 類型 | 筆數 | 比例 | 意義 |
|---|---:|---:|---|
| Correct Top-1 | 1,209 | 43.13% | 排名第一即為真實負責人 |
| Candidate miss | 238 | 8.49% | 真實負責人不在候選池，排序模型無法補救 |
| Ranking miss | 1,156 | 41.24% | 真實負責人在候選池，但未排第一 |
| Unseen-owner manual protected | 192 | 6.85% | 未知 owner 案件被安全擋回人工 |
| Unseen-owner auto error | 8 | 0.29% | 未知 owner 案件被錯誤自動派工 |

改善優先順序應為：

1. 先降低 candidate miss，提高 known-owner recall@30。
2. 再處理數量最大的 ranking miss。
3. 改善 open-set 的低 FPR 區域與信心校準。
4. 最後才重新選擇 auto policy；不可直接降低門檻追求 coverage。

## 4. 目前不能正式部署的原因

### 4.1 模型與統計 blocker

- `holdout_unseen_confidence_upper_bound`：unseen-owner 95% 上界 7.69%，未低於 5%。
- `routing_minimum_candidate_sources`：既有 v2 policy 沒有凍結「至少兩種候選來源支持」條件。
- `all_required_holdout_checks_passed`：原始 holdout report 使用舊版 point-estimate gate，缺少新統計檢查。

2022 Q2 已經是一次性 release evidence，只能用於診斷，不能回頭修改 v2 門檻或重新宣稱為新 holdout。下一模型版本必須使用新的開發窗，並由另一份更晚、未看過的資料驗收。

### 4.2 營運 blocker

- Active/inactive/cold-start roster 尚未由維護者正式簽核，而且目前快照已過期。
- 真實 shadow feedback 為 0 筆。
- 尚未連續兩週通過 shadow primary rates。
- 尚無真實 shadow latency、unseen label 與統計上界證據。
- 尚未取得有權限操作人的明示部署核准。

因此目前 operational guard 固定輸出 0% rollout、`manual_only` 與 `kill_switch_active=true`。

## 5. 改善目標

### 5.1 Release hard gates

下一版本必須在全新 sealed holdout 同時滿足：

| Gate | 必要條件 |
|---|---:|
| Auto accuracy | ≥ 85% |
| Auto coverage | ≥ 10% |
| Auto accuracy Wilson 95% lower bound | ≥ 80% |
| Unseen-owner auto error point estimate | < 5% |
| Unseen-owner auto error Wilson 95% upper bound | < 5% |
| Holdout rows | ≥ 2,500 |
| Auto rows | ≥ 250 |
| Candidate source support | Auto Top-1 至少兩種來源支持 |
| Component safety | 符合最小樣本數的主要 component 不得低於安全線 |
| Reproducibility | 固定資料 manifest、程式、模型與 policy 可重現相同決策 |

### 5.2 工程改善目標

- Known-owner candidate recall@30：由 89.32% 提升至至少 95%。
- Overall candidate recall@30：不得低於目前 82.95%，且至少維持原計劃的 80% hard target。
- Known-owner Top-1：必須明顯優於目前 46.45%，開發階段以 50% 以上作為第一個工程目標。
- Ranking miss：在開發窗相較 v2 至少降低 10%，且不得犧牲 unseen 安全性。
- Unknown recall @ 5% known FPR：由 10% 提升至至少 20% 的開發目標。
- Top-3 confirmation：未達 90% 前維持停用，不以降低標準換取功能開啟。

## 6. 改善步驟計劃

### 階段 0：凍結 v2 與建立 v3 實驗登錄

狀態：v2 已凍結；v3 protocol 與開發工具已建立，正式實驗仍待具備真實 assignment-event timestamp 的新資料後登錄。

工作內容：

1. 將 Q2 標記為 `diagnosis_only_not_model_or_policy_selection`。
2. 建立 v3 experiment ID、資料時間邊界、git commit、seed 與 artifact registry。
3. 明確分離 ranker fit、calibration fit、open-set fit、policy selection 與 sealed holdout。
4. 加入 ticket ID、duplicate family、近重複文字與 owner event 的跨切分洩漏檢查。

完成條件：任何訓練或政策搜尋程式都不能讀取 Q2 label；v3 每個輸出可追溯到唯一資料與程式版本。

### 階段 1：補強 owner 身分與時間資料

優先級：P0。

工作內容：

1. 建立 canonical owner identity mapping，處理別名、帳號變更與轉組。
2. 將 owner 分成 active、inactive、cold-start known 與 unresolvable unknown。
3. 使用「當時可取得」的 roster／ownership snapshot，禁止使用事件發生後資訊。
4. 補足 assignment label availability timestamp，避免以最後修改時間代替實際指派可得時間。
5. 產出 owner 頻率、component coverage、unseen 比例與資料缺失報告。

完成條件：roster schema、互斥性、效期與來源全部通過；每個 label 都能證明在 query 時點是否可用。

### 階段 2：改善候選召回

優先級：P0，先於排序模型。

工作內容：

1. 分別評估 component ownership、product/component 歷史、近期活動、BM25、semantic retrieval 與共同修改關係的 recall@10/30/50。
2. 針對 238 筆 candidate miss 建立 component、owner frequency、時間漂移與資料缺失分群。
3. 對各候選來源設定 quota，避免單一熱門來源占滿候選池。
4. 加入時間衰減、近期 component owner 與 cold-start roster 候選。
5. 只在開發窗決定 candidate pool size；正式 holdout 不再改動。

完成條件：每個開發時間窗的 known-owner recall@30 ≥ 95%，overall recall@30 不低於目前 82.95%。

### 階段 3：改善候選排序

優先級：P0。

工作內容：

1. 對 1,156 筆 ranking miss 分析真實 owner 的候選名次與主要競爭 owner。
2. 增加 owner-component 時間權重、近期 ownership、來源一致性、文字相似度與長尾 owner 特徵。
3. 比較 pointwise logistic、pairwise LTR 與受限 gradient boosting；所有實驗使用相同 rolling folds。
4. 報告 micro／macro、熱門／長尾 owner 與主要 component 指標，禁止只依整體 accuracy 選模。
5. 保留 portable、可重現且符合 latency budget 的最小模型。

完成條件：所有開發窗 Top-1 均優於 v2；ranking miss 至少降低 10%，候選召回與 unseen 指標不得退步。

### 階段 4：改善 open-set 與信心校準

優先級：P0。

工作內容：

1. 使用 leave-owner-out 與時間較晚 cold-start owner 建立 unknown 訓練案例。
2. 分開擬合 Top-1 correctness calibrator 與 unseen-owner detector。
3. 增加候選來源數、候選分數熵、Top-1 margin、owner recency、component drift 與 roster 狀態特徵。
4. 輸出 AUROC、AUPRC、unknown recall@5% known FPR、Brier、ECE 與 reliability curve。
5. 檢查各時間窗與 component 的 calibration drift。

完成條件：開發窗 unknown recall@5% known FPR 至少達 20%，校準與低 FPR 表現都優於 v2，且沒有以大量誤擋 known-owner 換取結果。

### 階段 5：重新建立受約束 routing policy

優先級：P0。

工作內容：

1. 在獨立 policy-selection windows 搜尋 threshold，不與 calibrator fit 共用資料。
2. Auto 必須同時符合低 open-set risk、高 calibrated correctness、active owner 與 `candidate_source_count >= 2`。
3. 每個開發窗都必須通過 accuracy、coverage、unseen point/CI、最小 auto rows 與 component floor。
4. Top-3 未達 90% 時維持關閉，其他案件送 manual。
5. 凍結 v3 ranker、calibrator、detector、policy 與 drift reference。

完成條件：同一組 policy 在所有開發窗通過，不允許為單一時間窗建立特例門檻。

### 階段 6：建立新的 sealed holdout

優先級：P0。

工作內容：

1. 取得時間上晚於所有 v3 開發窗的新資料。
2. 在下載前決定樣本數、時間範圍與 manifest；評估前不得讀 label。
3. 除總樣本至少 2,500 與 auto 至少 250 外，先依預期 unseen error 做統計 power analysis。
4. 只執行一次正式 release evaluation；失敗後該資料轉為 diagnosis-only，不得繼續調門檻。

樣本數注意事項：依與正式 gate 一致的雙側 Wilson 95% 精確連續期望值計算，若 unseen 錯派率仍接近 4%，Wilson 上界要低於 5% 最低約需 1,825 筆 unseen-owner 樣本；若能把錯派率降低到約 2%，最低約需 203 筆。正式封存前仍須按預期事件率與 unseen 比例重新執行規劃工具。

完成條件：新 holdout 的全部 hard gates 由程式自動判定通過，且 report、manifest 與 evaluation registry 完整。

### 階段 7：Roster 簽核與真實 shadow

優先級：P0，需維護者與實際流程配合。

工作內容：

1. 由維護者確認當期 active/inactive/cold-start roster、reviewer、as-of 與 expiry。
2. 在真實工單流程執行 shadow prediction，不改變實際負責人。
3. 保存當下預測、source event ID、最終 owner、known/unseen label 與端到端 latency。
4. 累積至少 500 筆或 28 天，並至少連續兩週通過主要 gate。
5. malformed event、重複來源 ID、缺失 latency 或 provenance 不完整都計入失敗率，不得靜默略過。

完成條件：shadow accuracy、coverage、unseen point/CI、連續週期、invalid-event rate 與 p95 latency 全部通過。

### 階段 8：漸進部署與回滾

優先級：P1，只能在前述階段全部通過後執行。

工作內容：

1. 由有權限操作人明示核准 approved deployment bundle。
2. 每日產生最長 24 小時、與 bundle SHA-256 綁定的 operational state。
3. 依 5% → 10% → 25% 開放，每階段至少觀察七天，不可跳級。
4. 任一 bundle、roster、shadow、runtime hash、latency 或資料品質 gate 失敗時立即切回 0% manual。
5. 每日監控 coverage、fallback、錯誤、latency 與 freshness；每週評估延遲標記後的真實 accuracy 與 unseen 指標。

完成條件：每個 rollout 階段均無安全 gate 失敗，且可用單一 kill switch 立即停止所有自動派工。

## 7. 建議執行順序

| 順序 | 工作 | 主要輸出 |
|---:|---|---|
| 1 | 凍結 v2、建立 v3 registry 與新時間邊界 | v3 protocol manifest |
| 2 | Owner identity／roster／label availability 清理 | canonical roster、data audit |
| 3 | Candidate source ablation 與 candidate miss 修正 | candidate recall report |
| 4 | LTR ranking 改善與 rolling-fold 比較 | ranker artifact、error taxonomy |
| 5 | Open-set／calibration 改善 | detector、calibration report |
| 6 | 受約束 policy selection | frozen v3 routing artifact |
| 7 | 全新 sealed holdout 一次性驗收 | holdout report 或明確 No-Go |
| 8 | Reviewed roster 與真實 shadow | shadow readiness report |
| 9 | 人工核准與 5/10/25% rollout | approved bundle、operational states |

## 8. 不可違反的實驗規則

- 不得使用 2022 Q2 調整 v2 或 v3 特徵、模型、open-set threshold 或 routing policy。
- 不得只報 87.79% 而省略 25.72% coverage、43.13% overall Top-1 與 7.69% unseen CI 上界。
- 不得用 historical replay、synthetic feedback 或過期 roster 冒充 production evidence。
- 不得降低 unseen、accuracy 或 Top-3 標準以換取較好 coverage。
- 不得在正式 holdout 失敗後繼續使用同一份資料調參，再宣稱為 untouched holdout。
- 未通過全部 hard gates 時，deployment status 必須維持 `research_only`，rollout 必須維持 0%。

## 9. 最終驗收清單

- [ ] 新時間 v3 sealed holdout 未參與任何訓練、選模、校準或政策搜尋。
- [ ] Auto accuracy ≥ 85%，coverage ≥ 10%。
- [ ] Auto accuracy Wilson 95% 下界 ≥ 80%。
- [ ] Unseen-owner point estimate 與 Wilson 95% 上界均 < 5%。
- [ ] Auto 至少 250 筆，且每筆至少兩種候選來源支持。
- [ ] Known-owner candidate recall@30 ≥ 95%。
- [ ] Active/inactive/cold-start roster 已簽核且未過期。
- [ ] Eligible-owner 模式的 product/component 能力、權限與有效期間已由具名維護者簽核，且 policy-selection／holdout 每筆都有當時有效 snapshot。
- [ ] Exact-owner 與 Eligible-owner 分開呈現；Eligible Top-5 hit rate 與候選資格 precision 均通過門檻。
- [ ] 真實 shadow 至少 500 筆或 28 天，並連續兩週通過。
- [ ] Shadow latency、invalid-event、provenance 與 unseen label 完整通過。
- [ ] Bundle、runtime source、model、data、policy hash 全部一致。
- [ ] 有權限操作人明示核准。
- [ ] Operational guard 從 5% 開始且可立即回滾至 0%。

## 10. 證據來源

- `assignee_triage_accuracy/phase8_operational_readiness/reports/2022q2_holdout/rolling_holdout_report.json`
- `assignee_triage_accuracy/phase8_operational_readiness/reports/2022q2_holdout/routing_diagnostics.json`
- `assignee_triage_accuracy/phase8_operational_readiness/deployment_candidate_v2/deployment_bundle.json`
- `assignee_triage_accuracy/phase8_operational_readiness/shadow/shadow_readiness_report.json`
- `assignee_triage_accuracy/phase8_operational_readiness/operational/operational_guard_state.json`
- `docs/plans/assignee_auto_routing_completion_plan.md`
