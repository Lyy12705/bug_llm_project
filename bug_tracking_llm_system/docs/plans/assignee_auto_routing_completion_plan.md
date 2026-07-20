# 負責人自動分流模型完成與部署計劃書

- 文件版本：1.1
- 制定日期：2026-07-19
- 專題階段：負責人推薦與自動分流
- 文件狀態：執行中（模型／政策 sealed holdout 已通過；roster 與 shadow 待完成）

## 0. 2026-07-19 執行進度

| 項目 | 狀態 | 已完成內容／剩餘工作 |
|---|---|---|
| M0 時間協定 | 已完成程式補強 | train、calibration fit、policy selection、sealed test 分離；輸出時間邊界、row hash 與跨切分洩漏 audit |
| M1 roster／身分 | 部分完成 | 支援 reviewed active roster、inactive hard block、alias chain canonicalization 與 cold-start owner；仍需專案維護者提供並簽核實際 roster |
| M2 LTR | 程式實作已完成 | rolling LTR 可做時間安全 shadow routing；新增 self-contained bundle builder／loader、SHA-256、效期、交叉 artifact 驗證與 production entrypoint；未 approved、過期、竄改或 runtime 例外一律輸出 manual，真正啟用仍須等 roster 與 shadow gate 通過 |
| M3 信心校準 | 已補強 | calibration fit 與 policy selection 不再共用相同 validation rows；加入 Wilson 95% 信賴區間 |
| M4 Open-set | 部分完成 | rolling open-set 已驗證；generic deployment 暫時固定使用與 gate 相同的 fallback score，禁止把不同公式的 candidate detector 誤標 approved |
| M5 路由政策 | 已完成程式補強 | 85%／10%／低於 5%、最小樣本數、信賴區間、component floor 與 fail-closed reason code 已程式化 |
| M6 Shadow | 評估器已完成 | 可讀 append-only feedback、去除同 ticket 舊事件、產出週報與 gate；仍需累積至少 500 筆或 28 天真實 feedback |
| M7 最終驗收 | 技術 gate 已通過 | 2022 Q1 sealed holdout 第一次評估通過全部 hard gates；整體上線驗收仍缺 reviewed roster、真實 shadow gate 與人工核准 |

Hardened gate 對既有 2021 Q4 holdout 的重跑結果如下。這是既有 holdout 的一致性驗證，不重新宣稱為全新 untouched holdout：

| 指標 | 結果 | Gate |
|---|---:|---|
| Holdout rows | 2,779 | ≥ 2,500，通過 |
| Auto rows | 668 | ≥ 250，通過 |
| Auto accuracy | 97.75% | ≥ 85%，通過 |
| Auto accuracy Wilson 95% 下界 | 96.33% | ≥ 80%，通過 |
| Auto coverage | 24.04% | ≥ 10%，通過 |
| Unseen-owner auto rate | 0% | < 5%，通過 |
| Component accuracy floor | 無失敗 component | 通過 |
| Top-3 confirmation accuracy | 71.11% | 低於 90% 輔助目標，需重新 finalization |

主要 component 中目前最弱的是 `DOM: Security`：47 筆 auto、正確率 76.60%、Wilson 下界 62.78%。它通過 75% point floor 與 60% confidence floor，但安全餘裕小，shadow 期間應列為優先監控與可能暫停 auto 的分群。

## 0.1 2026-07-20 sealed holdout 與 deployment candidate 結果

2022 Q1 資料先以 `--seal-output` 寫入 3,000 筆，manifest 記錄時間範圍、截斷狀態與 SHA-256 `6aaf8a291301553d3d9ab1b31040c3bcbe376909ab377473adea59300b2bb791`。正式評估前沒有使用其 label 調整模型或政策；2021 Q4 只更新候選歷史，沒有重訓 ranker、calibrator、open-set detector 或重新選門檻。正規化與跨窗 overlap 排除後，有效評估 2,777 筆。

| 指標 | 2022 Q1 結果 | Gate |
|---|---:|---|
| Holdout rows | 2,777 | ≥ 2,500，通過 |
| Auto rows | 605 | ≥ 250，通過 |
| Auto accuracy | 99.17%（600/605） | ≥ 85%，通過 |
| Auto accuracy Wilson 95% 下界 | 98.08% | ≥ 80%，通過 |
| Auto coverage | 21.79% | ≥ 10%，通過 |
| Unseen-owner auto rate | 0%（0/178） | < 5%，通過 |
| Unseen-owner rate Wilson 95% 上界 | 2.11% | 低於 5% 安全線 |
| Component point／confidence floor | 無失敗 component | 通過 |
| Maximum PSI | 0.0525 | < 0.25，未觸發 drift fallback |
| Top-3 confirmation accuracy | 67.23% | 低於 90% 輔助目標，不阻擋 auto hard gate |

因此，模型與凍結政策已達成 sealed holdout 的主要 85%／10%／5% 聯合門檻。這不等於整個自動派工階段已可上線：目前產生的 rolling deployment candidate 正確保持 `research_only`，blocker 為 `active_roster_review_confirmed`、`active_roster_nonempty`、`shadow_gate_passed`、`shadow_unseen_labels_complete` 與 `explicit_operator_approval`。候選包已驗證 23,533 筆歷史快照及七個必要 artifact 的完整性。

同一 holdout 的 registered replay 不被視為第二份 release evidence，只用來驗證 G7。2,777 筆的 ticket、Top-1／Top-3、候選順序、known/unseen、routing status 與 fallback reason 全部逐筆一致，decision SHA-256 皆為 `a30103be4ed2b6484b2f235ac688a54c852d830b3cbea6eaa697e563ddf24d2b`。底層浮點最大差異為 `0.0001268`（entropy 特徵），低於 `0.0002` 容忍值，且沒有造成任何路由決策差異。

## 1. 計劃目的

本計劃要把目前的「負責人候選推薦研究模型」提升為可受控部署的「負責人自動分流系統」。系統必須在無法安全判斷時主動退回 Top-3 人工確認或人工分流，不能為了提高自動化比例而強制指派。

本階段的最終完成標準，是在一份模型與門檻都未曾看過的新時間切分 holdout 上，同時達到：

1. 自動派工正確率至少 85%。
2. 自動派工涵蓋率至少 10%。
3. unseen-owner 樣本被錯誤自動派工的比例低於 5%。

三項門檻必須同時通過；單獨通過其中一項不能視為完成。

## 2. 專題範圍

### 2.1 本計劃包含

- 負責人資料清理、身分合併及在職／活躍狀態管理。
- 候選負責人產生與候選池召回率改善。
- 候選人排序模型（learning-to-rank, LTR）改善。
- Top-1 預測正確機率的校準。
- 未見負責人與資訊不足案件的 open-set／拒答辨識。
- auto、Top-3 confirmation、manual 三段式派工政策。
- 時間切分、shadow evaluation、部署 gate、監控與回滾。

### 2.2 本計劃不包含

- 錯誤程式碼定位。
- 自動生成補丁。
- 補丁測試、regression test 或自動修復成功率。

上述項目屬於其他階段與其他模型，不納入本階段的完成度與驗收指標。

## 3. 目前基準與判讀

### 3.1 現有結果

| 資料切分 | 指標 | 目前結果 | 判讀 |
|---|---:|---:|---|
| Development validation（788 筆） | Candidate recall | 94.42% | 候選池在開發資料表現良好 |
| Development validation | Top-1 | 62.06% | 可作為模型開發比較，不可作部署結論 |
| Development validation | Top-3 | 81.47% | 適合候選推薦，但未達直接自動派工標準 |
| Exploratory test（2,740 筆） | Candidate recall | 78.39% | 時間／資料分布改變後明顯下降 |
| Exploratory test | Top-1 | 40.40% | 與基準模型相近，泛化改善尚未成立 |
| Future open-set holdout（2,672 筆） | 整體 Top-1 | 34.28% | 不能對所有案件直接派工 |
| Future open-set holdout | 整體 candidate recall | 72.64% | 受到 unseen owner 與時間漂移影響 |
| Future open-set holdout | Known-owner Top-1 | 43.87% | 即使真實負責人在已知集合，排序仍需改善 |
| Future open-set holdout | Known-owner candidate recall | 92.96% | 候選產生不是唯一瓶頸，排序與信心估計同樣重要 |
| Future open-set holdout | Unseen-owner rate | 21.86%（584 筆） | 約五分之一案件不能沿用封閉集合假設 |
| Future open-set holdout | Open-set AUROC | 66.61% | 有辨識訊號，但不足以單獨支撐安全派工 |
| Future open-set holdout | Unknown recall @ 5% known FPR | 9.42% | 在低誤殺條件下，未見負責人辨識能力偏弱 |
| Future open-set holdout | Auto coverage | 0% | 現有政策安全但過度保守，尚無自動派工能力 |
| Future open-set holdout | Top-3 confirmation coverage | 19.24% | 已能篩出部分高品質推薦案件 |
| Future open-set holdout | Top-3 confirmation accuracy | 94.36% | 僅代表正確人選在 Top-3，不等於 Top-1 可直接自動派工 |

### 3.2 核心問題

目前不是單一門檻設定問題，而是四個問題疊加：

1. **候選池的時間泛化下降**：整體 candidate recall 從開發資料的 94.42% 降到 72.64%。
2. **Top-1 排序能力不足**：known-owner candidate recall 有 92.96%，但 known-owner Top-1 僅 43.87%，表示正確人選常已在候選池中，卻沒有排在第一名。
3. **Open-set 在低誤判區域能力不足**：目前若要求 known-owner FPR 為 5%，只能找出 9.42% unseen-owner；若提高 unknown recall，會錯擋過多 known-owner。
4. **信心值未能有效區分正確與錯誤 Top-1**：現有門檻找不到同時滿足 85% accuracy 與 10% coverage 的自動派工區段，因此結果為零自動派工。

因此不能直接降低 `t_high` 來取得 10% coverage。這會增加自動派錯案件，且可能把 unseen-owner 案件錯派給歷史上相似但不正確的人。

## 4. 指標定義與正式驗收規格

### 4.1 指標公式

設所有 holdout 案件數為 `N`：

- `auto coverage = 自動派工案件數 / N`
- `auto accuracy = 自動派工且 Top-1 等於最終真實負責人的案件數 / 自動派工案件數`
- `unseen-owner auto error rate = unseen-owner 案件中被系統自動派工的案件數 / unseen-owner 案件數`
- `candidate recall@K = 真實負責人存在於前 K 名候選的案件數 / 可評估案件數`

對 unseen-owner 案件而言，真實負責人不在訓練時的封閉 owner 集合中；若系統仍從舊集合自動派工，該次派工視為錯誤。若新負責人已出現在正式 active roster／ownership map 中，則應另標成 `cold_start_known`，不可與完全未知的 `unresolvable_owner` 混為一類。

### 4.2 Release hard gates

正式 holdout 必須同時滿足以下條件：

| Gate | 必要條件 |
|---|---:|
| G1 自動派工品質 | Auto accuracy ≥ 85% |
| G2 自動化效益 | Auto coverage ≥ 10% |
| G3 未知負責人安全性 | Unseen-owner auto error rate < 5% |
| G4 樣本數 | Holdout 至少 2,500 筆，且至少 250 筆進入 auto |
| G5 統計穩定性 | Auto accuracy 的 Wilson 95% 信賴區間下界 ≥ 80% |
| G6 分群安全性 | 樣本數 ≥ 30 的主要 product/component 不得有 auto accuracy < 75% |
| G7 可重現性 | 相同 commit、資料 manifest、seed 重跑，路由結果完全一致 |

G1、G2、G3 是專題的主要完成門檻；G4 至 G7 是避免以少量樣本、單一熱門元件或不可重現結果誤判為完成的工程保護條件。

### 4.3 輔助指標

下列指標不取代三項主要 gate，但用來定位改善來源：

- Known-owner candidate recall@30 ≥ 95%。
- 整體 candidate recall@30 ≥ 80%。
- Top-3 confirmation accuracy ≥ 90%。
- Brier score、ECE 與 coverage-risk curve 優於目前版本。
- Unknown recall @ 5% known FPR 必須較目前 9.42% 明顯改善。
- 依 owner 頻率、product、component、新舊期間分層報告 Top-1、Top-3、coverage 與 error rate。

## 5. 技術方案總覽

系統分成四層，每一層單獨評估，避免錯誤被下一層掩蓋：

1. **候選層**：從歷史指派、component ownership、active roster、文字相似度與近期活動產生候選負責人。
2. **排序層**：對候選人預測相對適合度並輸出 Top-1／Top-3。
3. **風險層**：估計「Top-1 是否正確」以及「此案件是否屬於未知／資訊不足情況」。
4. **政策層**：根據校準後的風險決定 auto、Top-3 confirmation 或 manual。

建議的路由規則如下：

```text
若候選負責人不在 active roster、模型 artifact 過期或必要特徵缺失：manual
否則若 open-set 風險低、Top-1 正確機率高、且至少兩種候選來源支持 Top-1：auto
否則若候選池可信且 Top-3 風險可接受：top3_confirmation
否則：manual
```

任何 artifact 缺失、版本不符、資料 schema 不符或計算例外都必須 fail closed 到 manual，不可預設自動派工。

## 6. 分階段實作計劃

### 階段 0：凍結實驗協定與資料邊界（第 1 週）

#### 工作內容

1. 以 bug 建立時間或首次回報時間作為唯一切分依據，禁止隨機打散後再切 train/test。
2. 建立至少三段資料：
   - rolling train／validation：模型與特徵開發。
   - policy calibration：校準信心值及選擇政策門檻。
   - sealed future holdout：完全不參與特徵、模型、門檻選擇，只在最終驗收執行一次。
3. 對 ticket ID、duplicate family、相同描述近重複、相同 owner event 進行跨切分洩漏檢查。
4. 凍結每個時間點的 active roster 與 component ownership snapshot，禁止使用事件發生後才出現的負責人資訊。
5. 建立 `temporal_protocol_manifest.json`，記錄資料雜湊、時間邊界、筆數、owner 數、unseen-owner 比例、程式 commit 與 seed。

#### 交付物

- 新增 `assignee_triage_accuracy/scripts/build_assignee_temporal_protocol.py`。
- 新增 sealed holdout manifest，但不輸出 holdout label 給訓練流程。
- 資料洩漏與時間順序測試。

#### 完成條件

- 任一 ticket／duplicate family 不跨 train 與 holdout。
- 所有 owner 特徵都只能使用 ticket 建立當時可取得的歷史。
- 訓練及門檻搜尋程式無法直接讀取 sealed holdout label。

### 階段 1：改善資料品質與候選池（第 1–2 週）

#### 工作內容

1. 建立 assignee canonical ID mapping，合併 email、帳號更名、大小寫差異與別名。
2. 為負責人建立 `active`、`inactive`、`cold_start_known`、`unresolvable` 狀態。
3. 將候選來源拆開記錄：
   - component／product 歷史 owner。
   - 最近 30／90／180 天活躍 owner。
   - 正式 ownership map／active roster。
   - 相似 ticket owner。
   - reporter、stack trace、file path 或 keyword 所對應的專長 owner。
4. 保存每個候選人的來源、支援數、最近活動時間與歷史樣本數，供排序與安全政策使用。
5. 對冷門 component 動態擴大候選池；對候選很多的熱門 component 使用來源 quota，避免單一來源壟斷前 K 名。
6. 每一種候選來源分別報 recall，再報 union recall，確認改善來自何處。

#### 交付物

- 擴充 `assignee_triage_accuracy/scripts/train_assignee_ltr.py` 的候選產生與來源追蹤。
- 新增 roster／ownership snapshot 載入器與 schema validation。
- 產出 `candidate_source_recall_report.json`。

#### 階段目標

- Known-owner candidate recall@30 從 92.96% 提升至至少 95%。
- 整體 candidate recall@30 從 72.64% 提升至至少 80%。
- inactive owner 不得進入 auto 候選；必要時仍可出現在人工查核資訊中。

### 階段 2：提升 Top-1 排序的時間泛化（第 2–4 週）

#### 工作內容

1. 將單次 validation 改為多個 rolling temporal folds，每一 fold 都只能用過去預測未來。
2. 加入下列時間安全特徵：
   - owner 在 component／product 的近期處理量與成功指派率。
   - owner 最近一次處理同類 ticket 的時間差。
   - component-owner、product-owner 的平滑後條件機率。
   - 候選來源數與各來源排名。
   - 文字相似度、標題／描述 embedding、關鍵字與 stack/file 訊號。
   - owner 歷史樣本量、活躍狀態及負載代理值。
3. 對近期資料增加合理權重，但在 rolling folds 上搜尋權重，避免只記住最後一個期間。
4. 對 long-tail owner 使用 class/sample weighting，並同時觀察 Top-1 與 Macro-F1。
5. 比較現有模型、線性／pairwise ranker、gradient boosting ranker；使用多期間平均與最差期間結果選模，不以單一 validation 最高分決定。
6. 對錯誤分成：真實 owner 不在候選池、在候選池但排序錯、inactive／identity 錯、open-set 錯，以 error taxonomy 追蹤。

#### 交付物

- 更新 `train_assignee_ltr.py` 支援 rolling folds、時間權重及分層指標。
- 產出 `candidate_ltr_artifact.json`、`rolling_fold_metrics.csv`、`ltr_error_taxonomy.csv`。

#### 階段目標

- 每個 rolling fold 的 known-owner Top-1 都優於目前版本，而非僅平均值提升。
- Macro-F1 不得因熱門 owner 準確率提高而退步。
- 候選池錯誤與排序錯誤可被獨立量化。

### 階段 3：建立 Top-1 正確性校準器（第 3–5 週）

#### 工作內容

1. 使用 out-of-fold 預測建立 `P(Top-1 正確)`，不得使用 ranker 訓練內樣本直接校準。
2. 比較 Platt scaling、isotonic regression 與 beta calibration。
3. 校準輸入至少包含：Top-1 score、Top-1/Top-2 margin、候選分數 entropy、候選來源數、owner 歷史量、component 新穎度與資料缺失旗標。
4. 檢查全域校準與下列分群校準：熱門／長尾 owner、熱門／冷門 component、近期／較舊期間、候選來源數。
5. 輸出 reliability diagram、Brier score、ECE、risk-coverage curve。

#### 交付物

- 新增或擴充信心校準模組，產出 `confidence_calibrator.json`。
- 產出 `calibration_report.json` 與 risk-coverage 資料。

#### 階段目標

- 在 policy calibration set 中能找到至少一個達到 85% Top-1 accuracy 的非零 coverage 區段。
- 校準後信心區間需符合實際正確率，不能把未校準的 rank score 當作機率。

### 階段 4：改善 open-set 與 cold-start 處理（第 4–6 週）

#### 工作內容

1. 將目前 unseen-owner 拆為：
   - `cold_start_known`：訓練資料未出現，但事件當下 roster／ownership 已知。
   - `inactive_or_transferred`：歷史 owner 已離開或 ownership 已轉移。
   - `unresolvable_owner`：事件當下無任何合法候選資訊。
2. 使用 leave-owner-out 加 rolling-time folds 模擬未見負責人，不只在單一 holdout 才觀察 open-set。
3. 擴充 open-set 特徵：
   - component／product 新穎度。
   - 候選最高分、margin、entropy 與校準後正確機率。
   - 候選來源一致性與來源數。
   - active roster／ownership 是否有合法匹配。
   - Top-1 owner 的近期活動、歷史支援量與資料缺失程度。
4. 模型選擇以低 known-owner FPR 區域為主，不以整體 AUROC 單獨決定。
5. 對 `unresolvable_owner` 預設 manual；對 `cold_start_known` 只有在正式 ownership 證據充分時才允許進入排序／自動派工。

#### 交付物

- 更新 `assignee_triage_accuracy/scripts/train_assignee_open_set.py`。
- 更新 `assignee_triage_accuracy/scripts/evaluate_assignee_open_set_detector.py`。
- 產出 `open_set_detector_artifact.json` 與各未知類型的 confusion matrix。

#### 階段目標

- Unknown recall @ 5% known FPR 明顯高於目前 9.42%。
- 最終政策在 calibration set 的 unseen-owner auto error rate 低於 5%。
- 不再用單一 unseen label 混合可由 roster 解決與完全不可解的案件。

### 階段 5：聯合搜尋安全路由政策（第 5–6 週）

#### 工作內容

1. 僅使用 policy calibration set 聯合搜尋：
   - open-set risk threshold。
   - auto confidence threshold `t_high`。
   - Top-3 confirmation threshold `t_low`。
   - 最小候選來源數、最小 owner 歷史支援量及 artifact freshness。
2. 搜尋目標不是最高 overall accuracy，而是「在滿足 accuracy 與 unseen safety 約束下最大化 auto coverage」。
3. 每組門檻都計算 Wilson confidence interval、各時間 fold 與各主要 component 的結果。
4. 若沒有任何門檻組合同時通過 85%／10%／5%，政策必須保持未通過狀態，不能以人工挑選個案或降低驗收標準補足 coverage。
5. 門檻與模型分開版本化；任何模型更新都必須重新校準門檻。

#### 建議最佳化形式

```text
maximize: auto_coverage
subject to:
  auto_accuracy >= 0.85
  unseen_owner_auto_error_rate < 0.05
  auto_rows >= 250（正式 holdout）
  active_owner == true
  candidate_source_count >= 2
```

#### 交付物

- 擴充 `assignee_triage_accuracy/scripts/assignee_open_set_common.py`：
  - Wilson interval。
  - 約束式 policy search。
  - per-window／per-component gate。
  - fail-closed 決策原因碼。
- 產出 `routing_policy.json` 與 `policy_search_frontier.csv`。

#### 完成條件

- calibration set 上存在通過三項主要 gate 的政策候選。
- 每一筆決策都能輸出 `route_status`、`reason_code`、模型版本與門檻版本。

### 階段 6：Shadow evaluation（第 7–10 週，至少四週）

#### 工作內容

1. 在線上或最新批次資料執行模型，但先不實際改派負責人。
2. 保存當下預測，等待人工流程產生最終 assignee 後再評分，禁止事後重算預測取代原紀錄。
3. 至少累積四週或 500 筆可判讀案件；若 10% coverage 導致 auto 樣本太少，延長 shadow 期間。
4. 每週報告：
   - auto accuracy、coverage、unseen-owner auto error rate。
   - Top-3 confirmation accuracy／coverage。
   - product、component、owner frequency 分群。
   - owner roster drift、候選召回、信心分布、資料缺失及 latency。
5. 人工確認者應能標記「模型候選正確但最終 assignee 因排班／負載改變」等非模型因素，另行分析但不得任意從主指標刪除。

#### 交付物

- 新增 `assignee_triage_accuracy/scripts/run_assignee_shadow_evaluation.py`。
- 產出 append-only `shadow_predictions.jsonl` 及每週報告。

#### Shadow 通過條件

- 三項主要 gate 連續兩個週期通過。
- 無單一 component 的系統性錯派。
- 模型或 roster 異常時可確實降級到 manual。

### 階段 7：新時間 holdout 最終驗收與漸進部署（第 11 週或資料足量後）

#### 工作內容

1. 凍結程式 commit、模型 artifact、校準器、open-set detector 與 routing policy。
2. 在 sealed future holdout 上只執行一次正式評估。
3. `evaluate_assignee_open_set_holdout.py` 自動產出 gate 結果，禁止手動修改通過狀態。
4. 只有所有 hard gates 通過時，`prepare_assignee_deployment.py` 才可建立 deployment bundle。
5. 部署依序開放 5%、10%、25% 流量；每一階段至少觀察一個完整監控週期。
6. 未通過則保留 Top-3 recommendation／manual 功能，回到對應錯誤類型的階段改善，不宣稱完成自動派工。

#### 交付物

- `final_holdout_report.json`。
- `deployment_bundle.json`，包含資料／程式／模型／政策雜湊、建立時間、有效期限及通過的 gate。
- 研究結果表與 deployment card。

## 7. 程式修改對照

| 現有檔案 | 預定修改 |
|---|---|
| `assignee_triage_accuracy/scripts/train_assignee_ltr.py` | rolling temporal folds、候選來源、active roster、時間權重、分層評估 |
| `assignee_triage_accuracy/scripts/train_assignee_open_set.py` | leave-owner-out 訓練、未知類型拆分、新增風險特徵 |
| `assignee_triage_accuracy/scripts/assignee_open_set_common.py` | 聯合 threshold search、Wilson interval、分群 gate、reason code |
| `assignee_triage_accuracy/scripts/evaluate_assignee_open_set_detector.py` | 低 FPR 指標、cold-start 分類與各期間評估 |
| `assignee_triage_accuracy/scripts/evaluate_assignee_open_set_holdout.py` | 一次性正式 holdout、CI、分群、drift 與 deployment gate |
| `assignee_triage_accuracy/scripts/prepare_assignee_deployment.py` | gate 未通過禁止產出 approved bundle；加入版本、hash、expiry |
| `src/modules/assignee_triager.py` | 只載入通過 gate 的 bundle；active owner／schema／artifact freshness 檢查；例外 fail closed |
| `tests/test_assignee_ltr.py` | 時間切分、候選召回、身分 mapping、重現性、threshold boundary 測試 |
| `tests/test_pipeline.py` | auto／Top-3／manual 整合、inactive owner 與 artifact failure 降級測試 |

建議新增：

- `assignee_triage_accuracy/scripts/build_assignee_temporal_protocol.py`
- `assignee_triage_accuracy/scripts/run_assignee_shadow_evaluation.py`
- `assignee_triage_accuracy/schemas/assignee_roster.schema.json`
- `assignee_triage_accuracy/schemas/deployment_bundle.schema.json`

## 8. 測試計劃

### 8.1 單元測試

- Alias mapping 不會把兩位不同負責人錯誤合併。
- inactive owner 永遠不會進入 auto。
- 缺少 roster、artifact、必要特徵時必須 manual。
- `t_high`、open-set threshold 邊界值的決策一致。
- Wilson interval 與 gate 計算正確。
- 固定 seed 與 manifest 可重現相同候選及排序。

### 8.2 資料洩漏測試

- 特徵時間戳不得晚於 ticket 建立時間。
- future assignee event 不可進入候選與特徵。
- 相同 ticket／duplicate family 不可跨 train 與 holdout。
- 校準器只能讀 out-of-fold 預測。

### 8.3 整合測試

- 完整流程可由 ticket 輸入產生候選、排名、風險、route status 與 reason code。
- 模型 artifact 與 policy version 不相容時拒絕啟動 auto。
- deployment gate 未通過時無法建立 approved bundle。
- shadow 模式不會實際變更 assignee。

### 8.4 回歸測試

- 新模型不得降低既有 Top-3 confirmation 品質。
- 主要 product/component 的 candidate recall 不得顯著退步。
- 推論時間、記憶體與 artifact 大小設定上限並納入 CI。

## 9. 監控與回滾

### 9.1 上線監控

- 每日：auto coverage、信心分布、manual fallback、artifact／roster freshness、錯誤率與 latency。
- 每週：延遲標記後的 auto accuracy、unseen-owner auto error rate、分群品質及 owner drift。
- 每月：重新建立 rolling report，判斷是否需要重訓與重新校準。

### 9.2 自動停止 auto 的條件

任一條件發生時，立即把所有案件降級成 Top-3 或 manual：

- 最近可用評估窗的 auto accuracy 低於 80%。
- unseen-owner auto error rate 達到或超過 5%。
- active roster／ownership snapshot 過期。
- 輸入 schema、模型 artifact 或 policy version 不一致。
- 主要 component 的候選召回或信心分布出現顯著漂移。
- 模型服務錯誤或必要特徵缺失率超過預設上限。

回滾只需停用 approved bundle 的 auto flag；Top-3 recommendation 與 manual 流程仍可保留。

## 10. 時程與里程碑

| 週次 | 里程碑 | 主要輸出 |
|---|---|---|
| 第 1 週 | M0 實驗協定凍結 | temporal manifest、資料洩漏測試 |
| 第 1–2 週 | M1 候選池改善 | roster、identity mapping、candidate source report |
| 第 2–4 週 | M2 LTR 時間泛化 | rolling-fold model、error taxonomy |
| 第 3–5 週 | M3 信心校準 | calibrator、risk-coverage curve |
| 第 4–6 週 | M4 Open-set 改善 | detector、cold-start／unknown 報告 |
| 第 5–6 週 | M5 路由政策 | constrained policy、三項 gate 的 calibration 結果 |
| 第 7–10 週以上 | M6 Shadow | append-only predictions、週報、連續通過紀錄 |
| 第 11 週或資料足量後 | M7 正式驗收 | sealed holdout report、deployment bundle 或明確 no-go |

若 shadow 樣本不足，應延長觀察期，不應為配合固定日期縮小樣本數。

## 11. 風險與因應

| 風險 | 影響 | 因應方式 |
|---|---|---|
| 新負責人比例持續升高 | 候選 recall 與 auto coverage 下降 | active roster／ownership snapshot、cold-start 分類、manual fallback |
| 熱門 owner 主導模型 | Macro-F1 與冷門 component 表現差 | 時間／類別權重、分群 gate、來源 quota |
| 只在單一 validation 調到好看 | 正式 holdout 失敗 | rolling folds、sealed holdout、一次性驗收 |
| 直接降低信心門檻追求 coverage | 自動錯派增加 | 約束式搜尋，accuracy／unseen gate 優先 |
| 人員轉組或離職 | 派給無效 owner | roster freshness、inactive hard block、artifact expiry |
| 最終 assignee 受排班而非專長影響 | 標籤噪音 | shadow reason code、另報營運因素，但保留主指標 |
| 低量 component 指標波動大 | 分群結論不穩定 | 最小樣本數、Wilson interval、合併多個時間窗 |

## 12. 第一個 Sprint 的具體工作清單

建議先完成以下工作，再調整模型超參數：

1. 定義 train、calibration、sealed future holdout 的時間邊界與最小樣本數。
2. 產出 `temporal_protocol_manifest.json` 並加入跨切分洩漏檢查。
3. 建立 canonical assignee mapping 與 active roster schema。
4. 將 584 筆 unseen-owner 拆成 cold-start known、inactive/transferred、unresolvable。
5. 對現有候選來源逐一計算 known／unseen／component 分層 recall。
6. 將現有 holdout 每筆錯誤標成 candidate miss、ranking miss、open-set miss 或 policy abstain。
7. 在 `assignee_open_set_common.py` 加入 Wilson interval 與約束式 policy search。
8. 產出現有模型的 risk-coverage curve，確認距離 85% accuracy、10% coverage 還差多少。
9. 建立 fail-closed 的 deployment bundle schema 與測試。
10. 完成 Sprint 報告後，再決定排序模型、校準器與 open-set detector 的優先實驗組合。

## 13. Go／No-Go 驗收清單

只有以下項目全部為「是」，才可宣稱負責人自動分流階段完成：

- [x] 資料採時間切分，sealed holdout 未用於訓練、選模、校準或選門檻。
- [x] Auto accuracy ≥ 85%。
- [x] Auto coverage ≥ 10%。
- [x] Unseen-owner auto error rate < 5%。
- [x] Auto 樣本至少 250 筆且統計保護條件通過。
- [ ] Active/inactive/cold-start owner 有明確資料來源與時間版本。
- [x] 主要 component 分群未出現不可接受的自動錯派。
- [ ] Shadow evaluation 連續兩個週期通過。
- [x] Artifact、policy、資料與程式版本可重現且可稽核。
- [x] 任一故障或漂移情況可自動退回 Top-3／manual。
- [x] 本階段報告只宣稱負責人分流成果，不混入錯誤定位或補丁生成能力。

## 14. Definition of Done

本階段的 Definition of Done 不是「模型能輸出一個負責人名稱」，而是：系統能在新時間資料中辨識一小部分足夠安全的案件自動派工，對不確定、未知、失效或資料不足的案件可靠拒答，並以 sealed holdout 與 shadow data 證明 85% accuracy、10% coverage、低於 5% unseen-owner auto error 三項條件同時成立。

目前 sealed holdout 已達聯合門檻，但 reviewed roster 與真實 shadow 證據尚未完成，因此成果應定義為「已通過離線技術驗收、等待營運安全驗收的選擇性自動分流 candidate」，而不是已核准上線的自動派工系統。
