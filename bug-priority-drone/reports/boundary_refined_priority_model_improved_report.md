# 邊界修正優先級模型報告

本實驗在 direct DRONE/REP- classifier 上加入 P1/P2/P3 boundary classifier。

## 評估結果

| model_stage      | eval_set        |   rows |   accuracy |   macro_f1 |   off_by_one_accuracy |    mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:-----------------|:----------------|-------:|-----------:|-----------:|----------------------:|-------:|------------:|------------:|------------:|------------:|------------:|
| base_direct      | natural_holdout |    995 |     0.6995 |     0.699  |                0.8975 | 0.4472 |      0.6935 |      0.5859 |       0.815 |      0.6633 |      0.7387 |
| boundary_refined | natural_holdout |    995 |     0.7116 |     0.7106 |                0.8975 | 0.4291 |      0.6683 |      0.6364 |       0.85  |      0.6633 |      0.7387 |

## Boundary 候選模型排序

| boundary_model_type   | boundary_params   |   selection_score |   accuracy |   macro_f1 |   p1_recall |   p2_recall |   p3_recall |   off_by_one_accuracy |    mae |
|:----------------------|:------------------|------------------:|-----------:|-----------:|------------:|------------:|------------:|----------------------:|-------:|
| sgd_log               | alpha=0.1         |            0.7302 |     0.6776 |     0.6746 |       0.652 |       0.556 |       0.864 |                0.8952 | 0.4744 |
| sgd_log               | alpha=0.01        |            0.7245 |     0.6776 |     0.6741 |       0.732 |       0.504 |       0.836 |                0.892  | 0.4848 |
| sgd_log               | alpha=0.001       |            0.6723 |     0.636  |     0.6352 |       0.76  |       0.412 |       0.692 |                0.8768 | 0.5496 |

## 各 Priority 分類報告

| eval_set        | priority   | label       |   precision |   recall |     f1 |   support |
|:----------------|:-----------|:------------|------------:|---------:|-------:|----------:|
| natural_holdout | P1         | P1 Blocker  |      0.7189 |   0.6683 | 0.6927 |       199 |
| natural_holdout | P2         | P2 Critical |      0.6207 |   0.6364 | 0.6284 |       198 |
| natural_holdout | P3         | P3 Major    |      0.7456 |   0.85   | 0.7944 |       200 |
| natural_holdout | P4         | P4 Minor    |      0.6769 |   0.6633 | 0.6701 |       199 |
| natural_holdout | P5         | P5 Trivial  |      0.7989 |   0.7387 | 0.7676 |       199 |

## 混淆矩陣

### base_direct_natural_holdout

列為真實 P1-P5，欄為預測 P1-P5。

```text
[[138  47   9   3   2]
 [ 50 116  13  13   6]
 [  5   4 163  19   9]
 [ 13  18  16 132  20]
 [  6   6  12  28 147]]
```

### boundary_refined_natural_holdout

列為真實 P1-P5，欄為預測 P1-P5。

```text
[[133  48  13   3   2]
 [ 37 126  16  13   6]
 [  2   0 170  19   9]
 [  9  21  17 132  20]
 [  4   8  12  28 147]]
```
