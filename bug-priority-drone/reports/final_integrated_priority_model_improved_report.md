# Final Integrated Priority Model 報告

- Feature profile：`improved`
- Profile 說明：enhanced text features，加上擴充 REP- related-report matching 與 error-driven keyword features。
- 選定分類器：`sgd_log (alpha=0.01)`
- Validation selection metric：`guarded_macro`

## 評估結果

| eval_set        |   rows |   accuracy |   macro_f1 |   off_by_one_accuracy |    mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:----------------|-------:|-----------:|-----------:|----------------------:|-------:|------------:|------------:|------------:|------------:|------------:|
| natural_holdout |    995 |     0.6995 |      0.699 |                0.8975 | 0.4472 |      0.6935 |      0.5859 |       0.815 |      0.6633 |      0.7387 |

## Validation 候選模型排序

Final classifier 會先在 validation set 上選定，再進行測試。

| candidate_model_type   | candidate_params   |   selection_score |   accuracy |   macro_f1 |   off_by_one_accuracy |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:-----------------------|:-------------------|------------------:|-----------:|-----------:|----------------------:|------------:|------------:|------------:|------------:|------------:|
| sgd_log                | alpha=0.01         |            0.6756 |     0.6768 |     0.6756 |                0.8912 |       0.692 |       0.564 |       0.812 |       0.588 |       0.728 |
| sgd_log                | alpha=0.1          |            0.6688 |     0.6744 |     0.6688 |                0.8936 |       0.664 |       0.52  |       0.9   |       0.556 |       0.732 |
| sgd_log                | alpha=0.001        |            0.6592 |     0.6576 |     0.6592 |                0.8768 |       0.644 |       0.476 |       0.788 |       0.772 |       0.608 |

## 各 Priority 分類報告

| eval_set        | priority   | label       |   precision |   recall |     f1 |   support |
|:----------------|:-----------|:------------|------------:|---------:|-------:|----------:|
| natural_holdout | P1         | P1 Blocker  |      0.6509 |   0.6935 | 0.6715 |       199 |
| natural_holdout | P2         | P2 Critical |      0.6073 |   0.5859 | 0.5964 |       198 |
| natural_holdout | P3         | P3 Major    |      0.7653 |   0.815  | 0.7893 |       200 |
| natural_holdout | P4         | P4 Minor    |      0.6769 |   0.6633 | 0.6701 |       199 |
| natural_holdout | P5         | P5 Trivial  |      0.7989 |   0.7387 | 0.7676 |       199 |

## Recall 指標判讀

- `p1_recall` 到 `p5_recall` 分別代表真實 P1-P5 被模型正確抓出的比例。
- `precision` 代表模型預測成該 priority 時，有多少比例真的屬於該 priority。
- `f1` 是 precision 與 recall 的平衡分數，適合用來比較各 priority 的整體表現。
- `support` 是該 evaluation set 中每個 priority 的真實資料筆數。
- `off_by_one_accuracy` 代表預測 priority 只差一級時也視為可接受的比例，適合 priority 這種序位任務。
- `mae` 是預測 priority 與真實 priority 的平均距離，越低代表錯誤越接近真實等級。

## 混淆矩陣

### natural_holdout

列為真實 P1-P5，欄為預測 P1-P5。

```text
[[138  47   9   3   2]
 [ 50 116  13  13   6]
 [  5   4 163  19   9]
 [ 13  18  16 132  20]
 [  6   6  12  28 147]]
```
