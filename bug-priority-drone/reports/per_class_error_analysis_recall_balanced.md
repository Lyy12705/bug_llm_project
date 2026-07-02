# Per-Class Error Analysis

This report answers where each true priority class is predicted when the model is wrong.

## Overall Metrics

|   rows |   removed_training_overlap |   accuracy |   macro_f1 |   off_by_one_accuracy |   mae |
|-------:|---------------------------:|-----------:|-----------:|----------------------:|------:|
|    995 |                          0 |     0.7266 |     0.7265 |                0.9045 | 0.404 |

## P1-P5 被錯判到哪裡

| true_priority   |   support |   correct |   wrong |   recall | most_common_wrong_prediction   |   most_common_wrong_count |   predicted_P1 |   predicted_P2 |   predicted_P3 |   predicted_P4 |   predicted_P5 |
|:----------------|----------:|----------:|--------:|---------:|:-------------------------------|--------------------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| P1 Blocker      |       199 |       131 |      68 |   0.6583 | P2 Critical                    |                        51 |            131 |             51 |              9 |              5 |              3 |
| P2 Critical     |       198 |       136 |      62 |   0.6869 | P1 Blocker                     |                        28 |             28 |            136 |             14 |             14 |              6 |
| P3 Major        |       200 |       167 |      33 |   0.835  | P4 Minor                       |                        21 |              3 |              0 |            167 |             21 |              9 |
| P4 Minor        |       199 |       142 |      57 |   0.7136 | P3 Major                       |                        17 |              7 |             16 |             17 |            142 |             17 |
| P5 Trivial      |       199 |       147 |      52 |   0.7387 | P4 Minor                       |                        29 |              2 |              7 |             14 |             29 |            147 |

## Wrong-Prediction Destinations

| true_priority   | predicted_priority   |   count |   percent_of_true_class |   percent_of_wrong_errors |
|:----------------|:---------------------|--------:|------------------------:|--------------------------:|
| P1 Blocker      | P2 Critical          |      51 |                  0.2563 |                    0.75   |
| P1 Blocker      | P3 Major             |       9 |                  0.0452 |                    0.1324 |
| P1 Blocker      | P4 Minor             |       5 |                  0.0251 |                    0.0735 |
| P1 Blocker      | P5 Trivial           |       3 |                  0.0151 |                    0.0441 |
| P2 Critical     | P1 Blocker           |      28 |                  0.1414 |                    0.4516 |
| P2 Critical     | P3 Major             |      14 |                  0.0707 |                    0.2258 |
| P2 Critical     | P4 Minor             |      14 |                  0.0707 |                    0.2258 |
| P2 Critical     | P5 Trivial           |       6 |                  0.0303 |                    0.0968 |
| P3 Major        | P4 Minor             |      21 |                  0.105  |                    0.6364 |
| P3 Major        | P5 Trivial           |       9 |                  0.045  |                    0.2727 |
| P3 Major        | P1 Blocker           |       3 |                  0.015  |                    0.0909 |
| P4 Minor        | P3 Major             |      17 |                  0.0854 |                    0.2982 |
| P4 Minor        | P5 Trivial           |      17 |                  0.0854 |                    0.2982 |
| P4 Minor        | P2 Critical          |      16 |                  0.0804 |                    0.2807 |
| P4 Minor        | P1 Blocker           |       7 |                  0.0352 |                    0.1228 |
| P5 Trivial      | P4 Minor             |      29 |                  0.1457 |                    0.5577 |
| P5 Trivial      | P3 Major             |      14 |                  0.0704 |                    0.2692 |
| P5 Trivial      | P2 Critical          |       7 |                  0.0352 |                    0.1346 |
| P5 Trivial      | P1 Blocker           |       2 |                  0.0101 |                    0.0385 |

## Confusion Matrix

Rows are true priorities and columns are predicted priorities.

| true_priority   |   P1 Blocker |   P2 Critical |   P3 Major |   P4 Minor |   P5 Trivial |
|:----------------|-------------:|--------------:|-----------:|-----------:|-------------:|
| P1 Blocker      |          131 |            51 |          9 |          5 |            3 |
| P2 Critical     |           28 |           136 |         14 |         14 |            6 |
| P3 Major        |            3 |             0 |        167 |         21 |            9 |
| P4 Minor        |            7 |            16 |         17 |        142 |           17 |
| P5 Trivial      |            2 |             7 |         14 |         29 |          147 |

## Interpretation

- `recall` 表示該 priority 的真實資料中，有多少比例被模型正確抓到。
- `most_common_wrong_prediction` 表示該 priority 最常被錯判成哪一類。
- 如果錯誤集中在相鄰 priority，例如 P2 -> P1 / P3，代表主要問題是邊界模糊。
- 如果錯誤跨很多級，例如 P1 -> P5，才代表模型有較嚴重的排序判斷問題。
