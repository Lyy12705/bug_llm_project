# Recall-Balanced Cost-Sensitive 改善實驗總結

本版將原先較像局部修正的 `P2/P4 correction classifier` 從主流程移除，改成較正式的 cost-sensitive / recall-balanced learning。

## 1. 本次保留與移除

| 項目 | 狀態 | 原因 |
|---|---|---|
| DRONE/REP- 六類特徵 | 保留 | 這是文獻基礎與主特徵來源 |
| P2 error-driven keyword features | 保留 | 補強 crash、exception、regression、data loss 等 domain 訊號 |
| P1/P2 boundary classifier | 保留 | 屬於高優先級邊界 refinement，目的明確 |
| P2/P4 correction classifier | 移除 | 太像針對單一錯誤方向的 local heuristic |
| Cost-sensitive class sample weights | 新增為主方法 | 用正式的 sample weighting 讓 P1-P5 recall 更平衡 |
| Recall-balanced validation objective | 保留 | 不只看 accuracy，也看 macro recall、minimum recall、macro F1 與 MAE |

## 2. 新模型流程

```text
DRONE/REP- feature matrix
        ↓
Cost-sensitive base classifier
        ↓
P1/P2 boundary classifier
        ↓
Recall-balanced validation objective
        ↓
Natural holdout evaluation
```

## 3. Cost-Sensitive 設計

本版在 base classifier 訓練時搜尋 class-specific sample weights：

- `base_p1_sample_weight`
- `base_p2_sample_weight`
- `base_p4_sample_weight`

並在 P1/P2 boundary classifier 中搜尋：

- `boundary_p1_sample_weight`
- `boundary_p2_sample_weight`

這樣可以用更標準的機器學習方式處理不同 priority 的 recall 平衡，而不是另外加一個 P2/P4 correction layer。

## 4. Validation Objective

選模目標仍然不是只看 accuracy，而是：

```text
macro recall
+ minimum recall
+ macro F1
+ accuracy
+ off-by-one accuracy
- MAE penalty
- low-recall penalty
```

目的：

- 避免模型只偏向容易判斷的 P3。
- 避免某一個 priority recall 太低。
- 同時考慮錯誤是否只差一級。

## 5. 結果比較

| 模型 | Accuracy | Macro F1 | Macro Recall | Min Recall | Off-by-one | MAE | P1 Recall | P2 Recall | P3 Recall | P4 Recall | P5 Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2 keywords SGD | 0.7246 | 0.7240 | 0.7244 | 0.6482 | 0.8995 | 0.4090 | 0.6482 | 0.6818 | 0.8750 | 0.6834 | 0.7337 |
| Recall-balanced cost-sensitive | 0.7266 | 0.7265 | 0.7265 | 0.6583 | 0.9045 | 0.4040 | 0.6583 | 0.6869 | 0.8350 | 0.7136 | 0.7387 |

## 6. 主要改善

| 指標 | 前一版 | 本版 | 變化 |
|---|---:|---:|---:|
| Accuracy | 0.7246 | 0.7266 | +0.0020 |
| Macro F1 | 0.7240 | 0.7265 | +0.0025 |
| Macro Recall | 0.7244 | 0.7265 | +0.0021 |
| Min Recall | 0.6482 | 0.6583 | +0.0101 |
| Off-by-one accuracy | 0.8995 | 0.9045 | +0.0050 |
| MAE | 0.4090 | 0.4040 | -0.0050 |
| P1 Recall | 0.6482 | 0.6583 | +0.0101 |
| P2 Recall | 0.6818 | 0.6869 | +0.0051 |
| P4 Recall | 0.6834 | 0.7136 | +0.0302 |
| P5 Recall | 0.7337 | 0.7387 | +0.0050 |

## 7. 錯誤方向

| 錯誤方向 | 前一版 | 本版 | 變化 |
|---|---:|---:|---:|
| P1 -> P2 | 50 | 51 | +1 |
| P4 -> P2 | 23 | 16 | -7 |

解讀：

- P1 -> P2 沒有改善，代表 P1/P2 邊界仍是主要困難點。
- P4 -> P2 明顯下降，且這次不是靠 P2/P4 correction classifier，而是透過 cost-sensitive base training 與 recall-balanced objective 達成。
- 本版方法比原本的 local correction 更適合放進報告，因為它是正式的 sample weighting 與 validation objective 設計。

## 8. 結論

本版主模型改為 `recall_balanced_cost_sensitive`。雖然 accuracy 比先前含 P2/P4 correction layer 的版本低一些，但方法更正式、較容易向教授說明，也避免把模型設計描述成針對單一錯誤方向的 heuristic。

報告建議說法：

> 本次將原本較像局部修正的 P2/P4 correction 移除，改用 cost-sensitive recall-balanced learning。模型透過 class-specific sample weights 與 validation objective 同時考慮 macro recall、minimum recall、macro F1、off-by-one accuracy 與 MAE，使各 priority 的 recall 更平衡。

## 9. 輸出檔案

- `scripts/train_recall_balanced_priority_model.py`
- `reports/recall_balanced_priority_model.md`
- `reports/recall_balanced_best_eval.csv`
- `reports/recall_balanced_best_class_report.csv`
- `reports/recall_balanced_grid_search.csv`
- `reports/recall_balanced_comparison.csv`
- `reports/per_class_error_analysis_recall_balanced.md`
- `reports/targeted_error_directions_recall_balanced.md`
