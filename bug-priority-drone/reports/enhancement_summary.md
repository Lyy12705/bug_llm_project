# DRONE/GRAY 改良摘要

本次改善目標是提升 P2 recall，並避免單純提高全域 P2 權重時造成 P1 recall 崩落。

## 已實作的改良

- 調整資料切分：`natural-test-size=0.1`，balanced train 每類 1200 筆，validation/test 每類 250 筆。
- 加強文字特徵：保留文獻式 TF textual factor，新增 TF-IDF unigram/bigram 片語特徵。
- 建立 hierarchical DRONE/GRAY：
  - Stage 1：判斷 `P1/P2`, `P3`, `P4/P5`。
  - Stage 2-high：專門判斷 `P1` vs `P2`。
  - Stage 2-low：專門判斷 `P4` vs `P5`。
- High splitter objective 同時重視 P1 recall 與 P2 recall，並懲罰過度預測 P2。
- 輸出 P2 error analysis：真實 P2 但預測為 P1/P3 的案例。

## 主要結果

| Model | Split | Accuracy | Macro F1 | P1 Recall | P2 Recall | Off-by-1 |
|---|---|---:|---:|---:|---:|---:|
| Single GRAY | Balanced | 0.455 | 0.445 | 0.624 | 0.244 | 0.838 |
| Single GRAY + P2 weight | Balanced | 0.454 | 0.446 | 0.600 | 0.268 | 0.840 |
| Hierarchical GRAY | Balanced | 0.458 | 0.461 | 0.408 | 0.396 | 0.826 |
| Single GRAY | Natural holdout | 0.476 | 0.465 | 0.729 | 0.259 | 0.857 |
| Single GRAY + P2 weight | Natural holdout | 0.482 | 0.475 | 0.719 | 0.310 | 0.855 |
| Hierarchical GRAY | Natural holdout | 0.502 | 0.509 | 0.508 | 0.523 | 0.847 |

## 結果判讀

Hierarchical DRONE/GRAY 對 P2 recall 有最明顯的改善。在 balanced test 中，P2 recall 從 0.244 提升到 0.396。在 natural holdout 中，P2 recall 從 0.259 提升到 0.523，macro F1 也從 0.465 提升到 0.509。

代價是 P1 recall 下降，因為 high-priority splitter 會把部分原本預測為 P1 的 ticket 改判為 P2。這是可預期的 trade-off，也比單純調 threshold 導致 P1 幾乎崩掉的做法更平衡。

## 產出檔案

- `models/hierarchical_drone_gray_model.joblib`
- `reports/hierarchical_drone_gray_report.csv`
- `reports/balanced_distribution_eval_hierarchical_drone_gray.csv`
- `reports/natural_distribution_eval_hierarchical_drone_gray.csv`
- `reports/enhanced_model_comparison.csv`
- `reports/p2_error_analysis_hierarchical_balanced.csv`
- `reports/p2_error_analysis_hierarchical_natural.csv`
