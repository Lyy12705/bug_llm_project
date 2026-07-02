# 優先級判定改良實驗摘要

> 注意：本文件記錄的是前一階段 `p2_keywords_sgd` 改良模型。此模型後續已被 `recall_balanced_priority_model` 取代；目前最佳結果請以 `reports/recall_balanced_best_eval.csv`、`reports/recall_balanced_improvement_summary.md` 為準。

本文件記錄前一階段改良模型的過程與結果。這次改良依照以下順序進行：

1. 針對 P2 錯誤案例加入更多 keyword features。
2. 強化 P1/P2/P3 boundary classifier objective。
3. 繼續調整 REP- / BM25F 欄位權重。
4. 加入 XGBoost 作為 improved classifier engine 候選。
5. 評估 transformer 實驗可行性。

## 1. P2 錯誤導向 keyword features

新增 feature 主要來自 `reports/p2_error_manual_categories_bm25_boundary_best.md` 中的 P2 錯誤型態。

新增重點包含：

- `has_internal_error_terms`
- `has_widget_disposed_terms`
- `has_breakpoint_terms`
- `has_preferences_terms`
- `has_update_manager_terms`
- `has_repository_terms`
- `has_save_cancel_terms`
- `has_search_refactor_terms`
- `normal_with_stack_exception`
- `normal_with_internal_error`
- `normal_with_ui_debug_core`
- `jdt_debug_breakpoint_signal`
- `platform_ui_internal_error_signal`
- `update_install_normal_signal`
- `stack_without_crash_data_loss`
- `p2_soft_boundary_signal_count`

這些 feature 的目的不是硬編規則，而是把 P2 被錯判成 P1/P3 的常見線索提供給模型學習。

## 2. Boundary classifier objective 調整

Boundary classifier 保持 P1/P2/P3 refiner 架構，但新增以下調整：

- 搜尋 `boundary_apply`：`base_1_2`、`base_1_2_3`
- 搜尋 P2 objective weight：`0.10`、`0.16`
- 搜尋 P2 sample weight：`1.0`、`1.2`
- 加入 P1 / P3 recall 權重
- 加入 off-by-one accuracy 權重
- 加入 accuracy drop penalty
- 加入 collapse penalty，避免只提升 P2 卻讓 P1/P3 表現明顯下降

## 3. REP- / BM25F 欄位權重調整

新增兩組 BM25F config：

| Config | 說明 |
|---|---|
| `p2_error_keywords` | 提高 summary / bigram 權重，降低 description 權重，搭配 P2 keyword features |
| `bigram_high` | 更強調 summary bigram，用來測試片語對 P2 邊界是否更有效 |

`p2_error_keywords` 的 duplicate-trained REP- 權重：

| Feature | 權重 |
|---|---:|
| unigram BM25Fext | 0.2470 |
| bigram BM25Fext | 0.0100 |
| same product | 2.2575 |
| same component | 0.8855 |
| same severity | 0.1334 |

`bigram_high` 的 duplicate-trained REP- 權重：

| Feature | 權重 |
|---|---:|
| unigram BM25Fext | 0.2498 |
| bigram BM25Fext | 0.0112 |
| same product | 2.2987 |
| same component | 0.9330 |
| same severity | 0.1742 |

## 4. XGBoost / LightGBM 比較

XGBoost 已加入 classifier engine，並補裝 macOS 需要的 `libomp` 後成功執行。

LightGBM 目前未安裝，因此本次使用 XGBoost 作為 improved classifier engine 代表。

實驗結果顯示，XGBoost 在這批高維文字特徵上沒有超過 SGD baseline，但 off-by-one accuracy 較高。

## 5. 結果比較

| Model | Accuracy | Macro F1 | Off-by-one | MAE | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 舊最佳 BM25 boundary | 0.7116 | 0.7108 | 0.8955 | 0.4302 | 0.6734 | 0.6515 | 0.8400 | 0.6533 | 0.7387 |
| `p2_keywords_sgd` | **0.7246** | **0.7240** | 0.8995 | **0.4090** | 0.6482 | 0.6818 | **0.8750** | **0.6834** | 0.7337 |
| `bigram_high_sgd` | 0.7196 | 0.7195 | 0.9015 | 0.4111 | 0.6482 | **0.6869** | 0.8350 | 0.6784 | **0.7487** |
| `p2_keywords_xgboost` | 0.7095 | 0.7094 | **0.9116** | 0.4111 | **0.6985** | 0.5606 | 0.8050 | 0.7538 | 0.7286 |

## 前一階段最佳改良模型

當時建議採用 `p2_keywords_sgd` 作為新的最佳模型；後續已由 Recall-balanced 模型取代。

| 項目 | 路徑 |
|---|---|
| 模型檔 | `models/improved_priority_p2_keywords_model.joblib` |
| 評估結果 | `reports/improved_priority_p2_keywords_eval.csv` |
| 各類別分類報告 | `reports/improved_priority_p2_keywords_class_report.csv` |
| Grid search 報告 | `reports/improved_priority_p2_keywords_grid_search.md` |
| P2 錯誤分析 | `reports/p2_error_analysis_improved_priority_p2_keywords_summary.md` |
| P2 錯誤人工分類 | `reports/p2_error_manual_categories_improved_priority_p2_keywords.md` |

相較於舊最佳模型：

- Accuracy：`0.7116 -> 0.7246`
- Macro F1：`0.7108 -> 0.7240`
- P2 recall：`0.6515 -> 0.6818`
- MAE：`0.4302 -> 0.4090`

## 新版 P2 錯誤分析

舊最佳模型的 P2 -> P1/P3 錯誤共有 `50` 筆；改良後剩下 `45` 筆。

改良後錯誤方向：

| 錯誤方向 | 筆數 | 比例 |
|---|---:|---:|
| P2 -> P1 判太嚴重 | 28 | 0.6222 |
| P2 -> P3 判太輕 | 17 | 0.3778 |

改良後主要錯誤型態：

| 主要錯誤型態 | 筆數 |
|---|---:|
| P2->P1：stack/exception 訊號讓模型判太嚴重 | 17 |
| P2->P3：severity normal/enhancement/minor 讓模型判太輕 | 16 |
| P2->P1：ui/debug/core 模組歷史訊號偏高 | 7 |
| P2->P1：一般判太嚴重 | 4 |
| P2->P3：related reports 偏向 P3 | 1 |

## Transformer 可行性

目前本機環境檢查結果：

- `transformers`：未安裝
- `sentence_transformers`：未安裝
- `torch`：未安裝

Transformer 實驗需要額外安裝套件並下載模型，例如 `sentence-transformers/all-MiniLM-L6-v2`、CodeBERT 或其他 software engineering 語料模型。

目前建議先不要把 transformer 放進主模型，原因是：

- 會增加大量 dependency 與模型下載成本。
- 訓練與推論時間會明顯增加。
- 此階段 error-driven DRONE/REP-/BM25F 改良已經把 accuracy 提升到 `0.7246`。

可放在未來改善：

> 未來可加入 transformer sentence embedding 作為 textual factor，與 DRONE 六類因素結合，觀察是否能進一步改善 P1/P2/P3 邊界。

## 終端機指令

查看比較結果：

```bash
cat reports/improved_priority_comparison.csv
cat reports/improved_priority_experiment_summary.md
```

查看此階段改良模型：

```bash
cat reports/improved_priority_p2_keywords_eval.csv
cat reports/improved_priority_p2_keywords_class_report.csv
cat reports/improved_priority_p2_keywords_grid_search.md
cat reports/p2_error_analysis_improved_priority_p2_keywords_summary.md
cat reports/p2_error_manual_categories_improved_priority_p2_keywords.md
```

重新訓練新的最佳改良模型：

```bash
python3 scripts/grid_search_bm25_boundary.py \
  --config-names p2_error_keywords \
  --skip-feature-build \
  --model-types sgd_log \
  --alpha-values 0.001,0.01,0.1 \
  --boundary-apply-values base_1_2,base_1_2_3 \
  --p2-weight-values 0.10,0.16 \
  --collapse-floor-values 0.55 \
  --boundary-p2-sample-weights 1.0,1.2 \
  --p1-weight 0.02 \
  --p3-weight 0.01 \
  --off-by-one-weight 0.02 \
  --accuracy-drop-weight 0.25 \
  --collapse-penalty-weight 0.35 \
  --max-accuracy-drop 0.015 \
  --output-csv reports/improved_priority_p2_keywords_grid_search.csv \
  --summary-md reports/improved_priority_p2_keywords_grid_search.md \
  --best-model-path models/improved_priority_p2_keywords_model.joblib \
  --best-eval-csv reports/improved_priority_p2_keywords_eval.csv \
  --best-class-report-csv reports/improved_priority_p2_keywords_class_report.csv
```
