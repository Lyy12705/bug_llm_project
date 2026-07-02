# 改良實驗紀錄

本文件記錄針對老師建議進行的三個改良方向：P2 邊界分類、REP- related-report 特徵、文字關鍵詞特徵。

## 1. 改善 P2 邊界分類

新增 `scripts/train_boundary_priority_model.py`。

做法：

- 先訓練原本的 direct DRONE/REP- classifier。
- 再用 train set 中真實標籤為 P1、P2、P3 的資料，訓練一個 P1/P2/P3 boundary classifier。
- 預測時，若 base model 預測結果落在 P1/P2/P3，則交給 boundary classifier 重新判斷。
- boundary model selection 目標加入 P2 recall，但同時懲罰 P1/P3 recall collapse，避免只為了提高 P2 而讓 P1 或 P3 崩掉。

結果：

| Model | Accuracy | Macro F1 | Off-by-one | MAE | P1 recall | P2 recall | P3 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Improved direct base | 0.6995 | 0.6990 | 0.8975 | 0.4472 | 0.6935 | 0.5859 | 0.8150 |
| Boundary-refined improved | 0.7116 | 0.7106 | 0.8975 | 0.4291 | 0.6683 | 0.6364 | 0.8500 |
| BM25/BM25F tuned boundary | 0.7116 | 0.7108 | 0.8955 | 0.4302 | 0.6734 | 0.6515 | 0.8400 |

解讀：

- P2 recall 從 `0.5859` 提升到 `0.6364`。
- Accuracy 從 `0.6995` 提升到 `0.7116`。
- Macro F1 從 `0.6990` 提升到 `0.7106`。
- P1 recall 稍微下降，但 P2/P3 與整體表現都有改善。

## 1b. Boundary Objective 限制條件搜尋

新增 `scripts/grid_search_boundary_constraints.py`。

搜尋目標：

- 提高 P2 recall 權重。
- 設定 P1 recall 下限，避免 P1 collapse。
- 設定 P3 recall 下限，避免 P3 collapse。
- 限制 validation accuracy 不下降或只允許小幅下降。

Improved feature profile 結果：

| Model | Accuracy | Macro F1 | P1 recall | P2 recall | P3 recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| base direct | 0.6995 | 0.6990 | 0.6935 | 0.5859 | 0.8150 | 0.4472 |
| constraint tuned boundary | 0.7116 | 0.7106 | 0.6683 | 0.6364 | 0.8500 | 0.4291 |

BM25 summary-high feature profile 結果：

| Model | Accuracy | Macro F1 | P1 recall | P2 recall | P3 recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| base direct | 0.7025 | 0.7016 | 0.7186 | 0.5859 | 0.8150 | 0.4432 |
| constraint tuned boundary | 0.7116 | 0.7108 | 0.6734 | 0.6515 | 0.8400 | 0.4302 |

解讀：

- 在目前候選模型池中，提高 P2 recall 權重或加入 P1/P3/accuracy 約束，沒有找到比現有 boundary 設定更好的組合。
- 最佳 boundary classifier 仍是 `sgd_log alpha=0.1`，套用在 base model 預測為 P1/P2 的資料上。
- 如果強制保留較高 P1 recall，P2 recall 會下降，形成明顯 trade-off。
- BM25 summary-high feature profile 仍是目前最佳整體結果，P2 recall 為 `0.6515`。

## 2. 調整 REP- related-report 特徵

更新 `scripts/train_rep_minus_weights.py` 與 `scripts/build_features.py`。

改良內容：

- REP- 權重訓練新增 `same_severity`。
- BM25Fext 的 summary、description 權重與 k1/k3/b 參數改成可由 CLI 調整。
- related-report similarity 從四個因素改成五個因素：
  - unigram BM25Fext
  - bigram BM25Fext
  - same product
  - same component
  - same severity
- related top-k 統計新增：
  - same product rate
  - same component rate
  - same severity rate

改良版 REP- 權重：

| Feature | Weight |
|---|---:|
| unigram BM25Fext | 0.2420 |
| bigram BM25Fext | 0.0100 |
| same product | 2.3854 |
| same component | 1.0108 |
| same severity | 0.2904 |

解讀：

- `same product` 與 `same component` 仍是 duplicate-related report 判斷中最強的訊號。
- `same severity` 也有正權重，表示 severity 相同對 related-report similarity 有幫助。

## 2b. BM25 / BM25F 參數搜尋

新增 `scripts/grid_search_bm25_boundary.py`。

搜尋方式：

- 每組參數都重新用 `dupe_of` duplicate links 訓練 REP- 權重。
- 每組參數都重新建立 train / validation / natural_holdout features。
- 每組參數都使用 P1/P2/P3 boundary-refined model 評估。
- 評估仍以 `natural_holdout` 作為最後測試集。

測試參數組合：

| Config | 說明 |
|---|---|
| default | 原本 BM25F 欄位權重與長度正規化 |
| summary_high | 提高 summary 權重，降低 description 權重 |
| description_balanced | summary 與 description 權重相同 |
| strong_norm | 加強 summary/description 長度正規化 |
| soft_norm | 降低長度正規化 |

結果：

| Config | Accuracy | Macro F1 | P1 recall | P2 recall | P3 recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| summary_high | 0.7116 | 0.7108 | 0.6734 | 0.6515 | 0.8400 | 0.4302 |
| default | 0.7116 | 0.7106 | 0.6683 | 0.6364 | 0.8500 | 0.4291 |
| description_balanced | 0.7116 | 0.7106 | 0.6683 | 0.6313 | 0.8500 | 0.4281 |
| soft_norm | 0.7065 | 0.7058 | 0.6935 | 0.5960 | 0.8250 | 0.4302 |
| strong_norm | 0.7065 | 0.7052 | 0.6533 | 0.6313 | 0.8500 | 0.4281 |

最佳組合是 `summary_high`。

解讀：

- 提高 summary 欄位權重後，P2 recall 從 `0.6364` 提升到 `0.6515`。
- Accuracy 維持 `0.7116`，macro F1 小幅提升到 `0.7108`。
- 這表示 bug report 的 summary 對 P1/P2/P3 邊界判斷有幫助。
- 強化或弱化長度正規化沒有帶來更好結果。

## 3. 加強文字與 error-driven features

更新 `scripts/build_features.py`。

新增/強化的關鍵詞特徵包含：

- crash / fatal / abort / core dump
- regression / no longer works / used to work
- data loss / corrupted / overwrite
- blocker / blocking / showstopper
- API break / incompatible API
- exception / stack trace / NullPointerException
- freeze / hang / deadlock
- performance / timeout / memory leak
- install / update / migration / startup
- security / vulnerability
- build break / compile error
- test failure

這些特徵會分別從全文與 summary 建立 binary features，例如：

- `has_crash_terms`
- `summary_has_crash_terms`
- `has_regression_terms`
- `has_stack_trace`

## 4. P2 Error Analysis

新增較細的 P2 錯誤分析輸出：

- `reports/p2_error_analysis_boundary_refined_improved_natural.csv`
- `reports/p2_error_analysis_boundary_refined_improved_summary.md`

Boundary-refined model 在 natural holdout 中：

- 真實 P2 筆數：198
- P2 被正確抓出：126
- P2 被判成 P1/P3：53

BM25/BM25F tuned boundary model 在 natural holdout 中：

- 真實 P2 筆數：198
- P2 被正確抓出：129
- P2 被判成 P1/P3：50

剩餘錯誤的主要分布：

- 預測成 P1：37
- 預測成 P3：16
- 常見 product：platform、jdt
- 常見 component：ui、debug、core
- 常見 severity：normal、major

在剩餘錯誤中較常出現的 keyword signals：

- stack trace
- exception terms
- install/update terms
- API break terms
- blocking terms

這些結果可以作為下一輪 error-driven feature 或人工分析依據。
