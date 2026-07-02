# Eclipse Bug Priority DRONE/GRAY 優先級判定專題

本專案實作 Eclipse Bugzilla bug report 的 priority prediction。研究方法以 DRONE/GRAY 文獻為基礎，建立 textual、temporal、author、related-report、severity、product/component 六類特徵，並進一步加入 duplicate-trained REP-、BM25/BM25F 欄位權重調整、P2 錯誤分析導向的 keyword features、P1/P2 boundary classifier，以及 cost-sensitive recall-balanced learning。

目前 GitHub 版本保留可重現實驗所需的程式、報告與輕量權重檔。大型資料集、feature matrix 與 `.joblib` 模型檔已透過 `.gitignore` 排除，避免 repository 過大；需要時可依照本文件指令重新產生。

## 目前最佳模型

| 項目 | 路徑 |
|---|---|
| 最佳模型 | `models/recall_balanced_priority_model.joblib` |
| REP- 權重 | `models/rep_minus_weights_bm25_p2_error_keywords.json` |
| 訓練特徵 | `data/processed/features_bm25_p2_error_keywords_train` |
| 驗證特徵 | `data/processed/features_bm25_p2_error_keywords_validation` |
| 最終測試特徵 | `data/processed/features_bm25_p2_error_keywords_natural_test` |
| 主要評估結果 | `reports/recall_balanced_best_eval.csv` |
| 各類別分類報告 | `reports/recall_balanced_best_class_report.csv` |
| 改良實驗摘要 | `reports/recall_balanced_improvement_summary.md` |

## 目前最佳結果

評估資料集為 `natural_holdout`，也就是不參與訓練與調參的最終測試資料。

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

和前一版最佳模型相比：

| 模型 | Accuracy | Macro F1 | P1 recall | P2 recall | P4 recall | MAE |
|---|---:|---:|---:|---:|---:|---:|
| p2 keywords SGD | 0.7246 | 0.7240 | 0.6482 | 0.6818 | 0.6834 | 0.4090 |
| Recall-balanced cost-sensitive | 0.7266 | 0.7265 | 0.6583 | 0.6869 | 0.7136 | 0.4040 |

## 專案流程

1. 從 Eclipse Bugzilla 抓取 P1-P5 bug report，並保留 `dupe_of` 欄位。
2. 清洗 `summary`、第一則 comment 作為 `description`、類別欄位與建立時間。
3. 建立 `train_balanced`、`validation_balanced`、`natural_holdout` 三種資料切分。
4. 建立 DRONE 六類特徵：文字、時間、作者、相關報告、嚴重度、產品/元件。
5. 使用 duplicate bug links 訓練 REP- related-report 權重。
6. 調整 BM25/BM25F summary、description、unigram、bigram 欄位權重。
7. 使用 P1/P2/P3 boundary classifier 改善 P2 容易被判成 P1 或 P3 的問題。
8. 針對 P2 剩餘錯誤加入 keyword features，重新比較模型表現。
9. 加入 recall-balanced objective、class-specific sample weights 與 P1/P2 boundary classifier。
10. 在 `natural_holdout` 上報告 accuracy、macro F1、off-by-one accuracy、MAE 與 P1-P5 recall。

## 查看目前結果

```bash
cd /Users/linyaying/Documents/bug-priority-drone-v2

python3 scripts/show_model_metrics.py
```

若要查看原始 CSV 與完整分類報告：

```bash
cat reports/recall_balanced_best_eval.csv
cat reports/recall_balanced_best_class_report.csv
cat reports/recall_balanced_improvement_summary.md
```

用較好讀的表格查看重點指標：

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("reports/recall_balanced_best_eval.csv")
cols = [
    "accuracy",
    "macro_f1",
    "off_by_one_accuracy",
    "mae",
    "p1_recall",
    "p2_recall",
    "p3_recall",
    "p4_recall",
    "p5_recall",
]
print(df[cols].to_string(index=False))
PY
```

## 重新產生資料與模型

### 1. 抓取 Eclipse Bugzilla 資料

```bash
python3 scripts/fetch_eclipse_bugzilla.py \
  --target-per-priority 2000 \
  --output data/raw/eclipse_bugzilla_raw_with_dupe.csv
```

### 2. 清洗資料

```bash
python3 scripts/clean_bug_reports.py \
  --input data/raw/eclipse_bugzilla_raw_with_dupe.csv \
  --output data/processed/eclipse_bug_reports_clean_with_dupe.csv
```

詳細清洗規則請見 `reports/cleaning_rules.md`。

### 3. 建立資料切分

```bash
python3 scripts/build_experiment_splits.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output-dir data/processed/experiment_splits_with_dupe \
  --natural-test-size 0.1 \
  --balanced-train-per-class 1200 \
  --balanced-validation-per-class 250 \
  --balanced-test-per-class 250
```

### 4. 重新訓練目前最佳 Recall-balanced 模型

若特徵資料已存在，可直接重跑目前最佳設定：

```bash
python3 scripts/train_recall_balanced_priority_model.py
```

此指令預設讀取 `data/processed/features_bm25_p2_error_keywords_*`，並輸出 `models/recall_balanced_priority_model.joblib` 與 `reports/recall_balanced_best_eval.csv`。

## P2 錯誤分析

前一版改良後，真實 P2 被判成 P1 或 P3 的錯誤由 50 筆降為 45 筆；目前最佳 Recall-balanced cost-sensitive 模型則進一步改善整體 per-class recall，尤其 P2 recall 從 0.6818 提升到 0.6869，P4 recall 從 0.6834 提升到 0.7136。

```bash
cat reports/p2_error_analysis_improved_priority_p2_keywords_summary.md
cat reports/p2_error_manual_categories_improved_priority_p2_keywords.md
```

目前主要剩餘錯誤類型：

- P2 被判成 P1：stack trace / exception 訊號讓模型判太嚴重。
- P2 被判成 P3：severity 是 normal / enhancement / minor 時容易被判太輕。
- P2 被判成 P1：UI / debug / core 模組歷史訊號偏高。

## 主要文件

- `reports/cleaning_rules.md`：資料清洗與前處理規則。
- `reports/literature_implementation.md`：本專題如何依據與延伸 DRONE/GRAY 文獻。
- `reports/model_result_terminal_steps.md`：查看模型結果的終端機指令。
- `reports/recall_balanced_improvement_summary.md`：目前最佳 Recall-balanced 實驗與結果比較。
- `reports/recall_balanced_priority_model.md`：目前最佳模型訓練摘要。
- `reports/per_class_error_analysis_recall_balanced.md`：目前最佳模型各類別錯誤分析。
- `reports/p2_error_analysis_improved_priority_p2_keywords_summary.md`：改良後 P2 錯誤摘要。
- `reports/p2_error_manual_categories_improved_priority_p2_keywords.md`：改良後 P2 錯誤人工分類。

## GitHub 版本控制說明

本 repository 不直接追蹤大型資料與模型檔：

- `data/raw/`
- `data/processed/`
- `models/*.joblib`

這些檔案可由上述 pipeline 重新產生。GitHub 上保留的是程式碼、報告、實驗結果 CSV、Markdown 說明與 REP- JSON 權重。
