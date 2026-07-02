# 下一輪模型改進分析結果

本次依序完成三件事：

1. 針對目前最佳模型剩餘的 P2 邊界錯誤做第二輪分析
2. 做 final objective grid search，目標改成 accuracy、macro F1、P2 recall 平衡
3. 做 label noise / ambiguous priority 分析

## 1. P2 剩餘錯誤第二輪分析

使用目前最佳模型 `improved_priority_p2_keywords_model.joblib`，在 `natural_holdout` 上分析真實 P2 但被預測成 P1 / P3 的資料。

結果：

| 錯誤方向 | 筆數 | 比例 |
|---|---:|---:|
| P2 被判成 P1，模型判太嚴重 | 28 | 62.22% |
| P2 被判成 P3，模型判太輕 | 17 | 37.78% |
| 合計 | 45 | 100% |

主要觀察：

- P2 -> P1 的錯誤常出現在文字看起來很嚴重的 bug，例如 stack trace、exception、internal error。
- P2 -> P3 的錯誤常出現在 severity 是 normal / minor / enhancement，或缺少明確 high-impact keyword 的 bug。
- `stack_exception_signal` 在 P2 錯誤中的比例是 42.22%，但在全部真實 P2 中只有 13.13%，代表 stack / exception 很容易把 P2 推向 P1。
- `related_pull_to_p2` 在錯誤 P2 中只有 11.11%，但在全部真實 P2 中有 28.79%，代表相關報告若沒有明顯支持 P2，模型就更容易往 P1 或 P3 偏移。

輸出檔案：

- `reports/p2_second_round_error_analysis.csv`
- `reports/p2_second_round_error_analysis.md`

## 2. Final Objective Grid Search

這次不是單純追求 P2 recall，而是用 validation set 重新設計選模目標：

```text
accuracy + macro F1 + 0.5 * P2 recall
+ P1/P3 guardrails
+ off-by-one accuracy bonus
- MAE penalty
```

目的：

- 提升 P2 recall
- 同時避免 P1 / P3 崩掉
- 保留整體 accuracy 與 macro F1

### 與目前最佳模型比較

| 模型 | Accuracy | Macro F1 | Off-by-one | MAE | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 目前最佳模型 | 0.7246 | 0.7240 | 0.8995 | 0.4090 | 0.6482 | 0.6818 | 0.8750 | 0.6834 | 0.7337 |
| final objective validation 選出模型 | 0.7236 | 0.7235 | 0.9025 | 0.4090 | 0.6030 | 0.7172 | 0.8650 | 0.6985 | 0.7337 |

解讀：

- P2 recall 從 0.6818 提升到 0.7172。
- P2 被誤判成 P1 / P3 的錯誤從 45 筆降到 37 筆。
- accuracy 從 0.7246 小降到 0.7236，差距只有 0.0010。
- P1 recall 從 0.6482 降到 0.6030，代表模型為了抓更多 P2，仍犧牲了一部分 P1。
- 如果專題重點是「整體準確率」，目前最佳模型仍可保留。
- 如果專題重點是「改善 P2 邊界」，final objective 模型比較有說服力。

補充：grid search 中有候選模型在 natural_holdout 上達到 accuracy 0.7317、macro F1 0.7314、P2 recall 0.7222，但它不是 validation objective 選出的第一名，因此不能直接把它當正式最佳模型，否則會有用測試集挑模型的資料洩漏問題。

輸出檔案：

- `reports/final_objective_grid_search.csv`
- `reports/final_objective_grid_search.md`
- `reports/final_objective_best_eval.csv`
- `reports/final_objective_best_class_report.csv`
- `reports/final_objective_comparison.csv`
- `reports/p2_error_analysis_final_objective_best_natural.csv`
- `reports/p2_error_analysis_final_objective_best_summary.md`

## 3. Label Noise / Ambiguous Priority 分析

這一步檢查資料集中是否存在「相同條件卻被標成不同 priority」的情況。

結果：

| 分析項目 | 數量 |
|---|---:|
| 清洗後資料筆數 | 9,949 |
| 高混淆群組 | 55 |
| duplicate priority 不一致群組 | 25 |

### 重要發現

| 情況 | 筆數 | 說明 |
|---|---:|---|
| normal / minor / trivial / enhancement 但 priority 是 P1 或 P2 | 2,893 | severity 看起來不嚴重，但 priority 被標高 |
| major / critical / blocker 但 priority 是 P4 或 P5 | 175 | severity 看起來嚴重，但 priority 被標低 |
| enhancement 但 priority 不是 P4 / P5 | 588 | enhancement 並不一定被標低優先級 |
| duplicate reports 中 priority 不一致 | 25 組 | 語意相關的 duplicate bug 可能有不同 priority |

解讀：

- P2 本身就是介於 P1 和 P3 的邊界類別。
- Eclipse Bugzilla 的歷史資料中，severity、product、component 與 priority 不是一對一關係。
- 即使兩筆 bug 是 duplicate，也可能被標成不同 priority，表示資料標註本身存在不一致。
- 因此模型的部分錯誤不一定完全是模型能力不足，也可能是歷史標註模糊或 triage 規則不一致造成。

輸出檔案：

- `reports/label_noise_analysis.md`
- `reports/label_noise_ambiguous_groups.csv`
- `reports/label_noise_duplicate_disagreements.csv`

## 總結

本次實驗顯示：

- P2 錯誤確實可以再降低，final objective 模型把 P2 邊界錯誤從 45 筆降到 37 筆。
- 但提高 P2 recall 會壓低 P1 recall，因此需要在報告中說明這是 priority boundary 的 trade-off。
- 資料本身存在 ambiguous priority 與 duplicate label disagreement，這可以合理解釋為什麼 P2 很難完全判準。
- 目前建議保留「目前最佳模型」作為主要 accuracy 結果，同時把 final objective 模型作為「改善 P2 recall 的延伸實驗」。

