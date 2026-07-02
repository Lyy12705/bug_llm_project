# Final Integrated Priority Model 報告

- Feature profile：`literature`
- Profile 說明：TF-only DRONE/REP- features，曾作為 final integrated model profile。
- 選定分類器：`sgd_log (alpha=0.01)`
- Validation selection metric：`guarded_macro`

## 評估結果

| eval_set        |   rows |   accuracy |   macro_f1 |   off_by_one_accuracy |    mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:----------------|-------:|-----------:|-----------:|----------------------:|-------:|------------:|------------:|------------:|------------:|------------:|
| natural_holdout |    995 |     0.7015 |     0.7008 |                0.8985 | 0.4442 |      0.6734 |      0.5909 |       0.825 |      0.6633 |      0.7538 |

## Validation 候選模型排序

Final classifier 會先在 validation set 上選定，再進行測試。

| candidate_model_type   | candidate_params   |   selection_score |   accuracy |   macro_f1 |   off_by_one_accuracy |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:-----------------------|:-------------------|------------------:|-----------:|-----------:|----------------------:|------------:|------------:|------------:|------------:|------------:|
| sgd_log                | alpha=0.01         |            0.6696 |     0.6712 |     0.6696 |                0.8936 |       0.636 |       0.552 |       0.852 |       0.588 |       0.728 |
| sgd_log                | alpha=0.1          |            0.6627 |     0.668  |     0.6627 |                0.8856 |       0.612 |       0.6   |       0.896 |       0.5   |       0.732 |
| sgd_log                | alpha=0.001        |            0.6502 |     0.6496 |     0.6502 |                0.8816 |       0.532 |       0.6   |       0.784 |       0.6   |       0.732 |
| linear_svm             | C=0.1              |            0.6342 |     0.6344 |     0.6342 |                0.8728 |       0.556 |       0.512 |       0.784 |       0.6   |       0.72  |
| sgd_log                | alpha=1.0          |            0.6268 |     0.64   |     0.6268 |                0.8584 |       0.612 |       0.568 |       0.956 |       0.36  |       0.704 |

## 各 Priority 分類報告

| eval_set        | priority   | label       |   precision |   recall |     f1 |   support |
|:----------------|:-----------|:------------|------------:|---------:|-------:|----------:|
| natural_holdout | P1         | P1 Blocker  |      0.6505 |   0.6734 | 0.6617 |       199 |
| natural_holdout | P2         | P2 Critical |      0.6094 |   0.5909 | 0.6    |       198 |
| natural_holdout | P3         | P3 Major    |      0.7604 |   0.825  | 0.7914 |       200 |
| natural_holdout | P4         | P4 Minor    |      0.6839 |   0.6633 | 0.6735 |       199 |
| natural_holdout | P5         | P5 Trivial  |      0.8021 |   0.7538 | 0.7772 |       199 |

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
[[134  49  10   3   3]
 [ 50 117  12  13   6]
 [  5   2 165  19   9]
 [ 12  17  19 132  19]
 [  5   7  11  26 150]]
```
