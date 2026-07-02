# DRONE/GRAY 文獻方法與本專題實作說明

本文件說明本專題如何參考 DRONE/GRAY 文獻進行 bug report priority prediction，以及目前版本做了哪些延伸與改良。

主要參考：

- Tian et al., `Automated Prediction of Bug Report Priority Using Multi-Factor Analysis`
- Sun et al., `Towards More Accurate Retrieval of Duplicate Bug Reports`

## 文獻方法概念

DRONE/GRAY 的核心想法是：bug report 的 priority 不是只由文字決定，而是由多種因素共同影響。因此文獻使用多因素分析，將 bug report 轉成多類特徵，再訓練模型預測 P1-P5 priority。

本專題保留 DRONE 六類因素：

| 文獻因素 | 本專題實作 |
|---|---|
| Textual | `summary`、第一則 `description` 的文字特徵 |
| Temporal | 建立時間之前的歷史 bug 數量、相同 severity 數量、時間窗統計 |
| Author | 回報者過去提交 bug 的數量與 priority 統計 |
| Related-report | 與歷史 bug report 的相似度與相似報告 priority 統計 |
| Severity | Bugzilla `severity` 類別與衍生特徵 |
| Product / Component | Bugzilla `product`、`component` 類別與歷史統計 |

## 與文獻相同的部分

本專題與 DRONE/GRAY 文獻相同的精神包括：

- 使用 Eclipse Bugzilla bug report 作為資料來源。
- 將 P1-P5 視為有順序的 priority label。
- 不只依賴文字，而是建立多因素特徵。
- related-report factor 只使用歷史 bug report，避免使用未來資訊。
- 參考 REP / REP- 的 duplicate bug retrieval 思路，建立 related-report similarity。

## 本專題的延伸

本專題在文獻基礎上做了以下延伸：

1. 使用 Eclipse Bugzilla REST / XML 抓取資料，並保留 `dupe_of` 欄位。
2. 使用 duplicate links 訓練 REP- 權重。
3. 將 REP- 相似度擴充為 BM25Fext unigram、BM25Fext bigram、same product、same component、same severity 的線性組合。
4. 加入 TF-IDF unigram / bigram、summary / description 分開 vectorize、char n-gram。
5. 加入 P2 error-driven keyword features，例如 crash、exception、internal error、update manager、breakpoint、widget disposed 等。
6. 將單一 GRAY classifier 延伸為 direct classifier 與 P1/P2/P3 boundary refiner。
7. 額外比較 XGBoost 作為 improved classifier engine。

## REP- Related-report 實作

本專題的 REP- 相似度會針對每一筆 bug report，只和建立時間更早的歷史 bug report 比較，避免資料洩漏。

簡化公式如下：

```text
REP-(d, q) =
  w1 * BM25Fext_unigram(d, q)
  + w2 * BM25Fext_bigram(d, q)
  + w3 * same_product(d, q)
  + w4 * same_component(d, q)
  + w5 * same_severity(d, q)
```

其中：

- `d`：歷史 bug report
- `q`：目前要預測的 bug report
- `BM25Fext_unigram`：以 summary / description 欄位權重計算 unigram 相似度
- `BM25Fext_bigram`：以 summary / description 欄位權重計算 bigram 相似度
- `same_product`：兩筆 bug 是否屬於相同 product
- `same_component`：兩筆 bug 是否屬於相同 component
- `same_severity`：兩筆 bug 是否屬於相同 severity

目前最佳 `p2_error_keywords` 設定的 duplicate-trained REP- 權重如下：

| REP- feature | 權重 |
|---|---:|
| unigram BM25Fext | 0.2470 |
| bigram BM25Fext | 0.0100 |
| same product | 2.2575 |
| same component | 0.8855 |
| same severity | 0.1334 |

這代表在目前 Eclipse sample 中，相同 product / component 對 duplicate-related report 判斷很重要，文字相似度也有作用，但權重較小。

## 模型設計

目前最佳模型不是純文獻原版 GRAY，而是依據文獻特徵延伸出的 recall-balanced final model。

模型流程：

1. 先建立 DRONE 六類特徵。
2. 使用 duplicate links 訓練 REP- 權重。
3. 使用 BM25/BM25F 調整 summary、description、unigram、bigram 欄位權重。
4. 使用 `SGDClassifier(loss="log_loss")` 作為 direct classifier。
5. 先根據 P2 錯誤分析加入 keyword features。
6. 再加入 cost-sensitive class sample weights 與 P1/P2 boundary classifier。
7. 最後依 validation set 的 macro recall、minimum recall、macro F1 等目標選出最佳設定，並在 `natural_holdout` 做最終測試。

## 目前最佳結果

最佳模型：`recall_balanced_priority_model`

| 指標 | 數值 |
|---|---:|
| Accuracy | 0.7266 |
| Macro F1 | 0.7265 |
| Off-by-one accuracy | 0.9045 |
| MAE | 0.4040 |
| P1 recall | 0.6583 |
| P2 recall | 0.6869 |
| P3 recall | 0.8350 |
| P4 recall | 0.7136 |
| P5 recall | 0.7387 |

與前一版最佳模型相比：

| 模型 | Accuracy | Macro F1 | P1 recall | P2 recall | P4 recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| p2 keywords SGD | 0.7246 | 0.7240 | 0.6482 | 0.6818 | 0.6834 | 0.4090 |
| Recall-balanced cost-sensitive | 0.7266 | 0.7265 | 0.6583 | 0.6869 | 0.7136 | 0.4040 |

## 與文獻的差異

| 項目 | 文獻 DRONE/GRAY | 本專題 |
|---|---|---|
| 資料來源 | Eclipse bug report historical dataset | Eclipse Bugzilla REST / XML 重新抓取 |
| Description | 原始 report 描述 | 第一則 comment，避免使用後續討論 |
| Related-report | REP / REP- | duplicate-trained REP- + BM25Fext 欄位權重 |
| Classification engine | GRAY ordinal regression + thresholds | DRONE/REP- features + direct classifier + recall-balanced local refiners |
| P2 / per-class 改善 | 文獻未特別針對 P2 邊界做 error-driven 改良 | 根據錯誤分析加入 keyword features、P1/P2 boundary 與 cost-sensitive recall-balanced objective |
| 評估方式 | 文獻資料與切分設定 | balanced train / validation + natural_holdout |

## 限制

- 目前資料是 P1-P5 各抓 2000 筆的 quota-balanced dataset，並不是文獻中的完整 chronological stream。
- GitHub repository 不追蹤大型 raw data、processed data 與 `.joblib` 模型檔，需由 pipeline 重新產生。
- 目前 transformer 尚未納入主模型，因為 `transformers`、`sentence_transformers`、`torch` 尚未安裝，且會增加訓練成本。
- Priority 標註本身可能存在 triager 習慣差異與 label noise，尤其 P1/P2/P3 邊界較模糊。

## 可重現指令

查看目前最佳結果：

```bash
cat reports/recall_balanced_best_eval.csv
cat reports/recall_balanced_best_class_report.csv
cat reports/recall_balanced_improvement_summary.md
```

重新訓練目前最佳模型：

```bash
python3 scripts/train_recall_balanced_priority_model.py
```

## 參考文獻

- Tian et al., `Automated Prediction of Bug Report Priority Using Multi-Factor Analysis`
- Sun et al., `Towards More Accurate Retrieval of Duplicate Bug Reports`
