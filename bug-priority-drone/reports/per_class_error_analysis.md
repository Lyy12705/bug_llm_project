# Per-Class Error Analysis

This report answers where each true priority class is predicted when the model is wrong.

## Overall Metrics

|   rows |   removed_training_overlap |   accuracy |   macro_f1 |   off_by_one_accuracy |   mae |
|-------:|---------------------------:|-----------:|-----------:|----------------------:|------:|
|    995 |                          0 |     0.7246 |      0.724 |                0.8995 | 0.409 |

## P1-P5 被錯判到哪裡

| true_priority   |   support |   correct |   wrong |   recall | most_common_wrong_prediction   |   most_common_wrong_count |   predicted_P1 |   predicted_P2 |   predicted_P3 |   predicted_P4 |   predicted_P5 |
|:----------------|----------:|----------:|--------:|---------:|:-------------------------------|--------------------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| P1 Blocker      |       199 |       129 |      70 |   0.6482 | P2 Critical                    |                        50 |            129 |             50 |             12 |              4 |              4 |
| P2 Critical     |       198 |       135 |      63 |   0.6818 | P1 Blocker                     |                        28 |             28 |            135 |             17 |             12 |              6 |
| P3 Major        |       200 |       175 |      25 |   0.875  | P4 Minor                       |                        15 |              3 |              1 |            175 |             15 |              6 |
| P4 Minor        |       199 |       136 |      63 |   0.6834 | P2 Critical                    |                        23 |              5 |             23 |             21 |            136 |             14 |
| P5 Trivial      |       199 |       146 |      53 |   0.7337 | P4 Minor                       |                        28 |              1 |              8 |             16 |             28 |            146 |

## Wrong-Prediction Destinations

| true_priority   | predicted_priority   |   count |   percent_of_true_class |   percent_of_wrong_errors |
|:----------------|:---------------------|--------:|------------------------:|--------------------------:|
| P1 Blocker      | P2 Critical          |      50 |                  0.2513 |                    0.7143 |
| P1 Blocker      | P3 Major             |      12 |                  0.0603 |                    0.1714 |
| P1 Blocker      | P4 Minor             |       4 |                  0.0201 |                    0.0571 |
| P1 Blocker      | P5 Trivial           |       4 |                  0.0201 |                    0.0571 |
| P2 Critical     | P1 Blocker           |      28 |                  0.1414 |                    0.4444 |
| P2 Critical     | P3 Major             |      17 |                  0.0859 |                    0.2698 |
| P2 Critical     | P4 Minor             |      12 |                  0.0606 |                    0.1905 |
| P2 Critical     | P5 Trivial           |       6 |                  0.0303 |                    0.0952 |
| P3 Major        | P4 Minor             |      15 |                  0.075  |                    0.6    |
| P3 Major        | P5 Trivial           |       6 |                  0.03   |                    0.24   |
| P3 Major        | P1 Blocker           |       3 |                  0.015  |                    0.12   |
| P3 Major        | P2 Critical          |       1 |                  0.005  |                    0.04   |
| P4 Minor        | P2 Critical          |      23 |                  0.1156 |                    0.3651 |
| P4 Minor        | P3 Major             |      21 |                  0.1055 |                    0.3333 |
| P4 Minor        | P5 Trivial           |      14 |                  0.0704 |                    0.2222 |
| P4 Minor        | P1 Blocker           |       5 |                  0.0251 |                    0.0794 |
| P5 Trivial      | P4 Minor             |      28 |                  0.1407 |                    0.5283 |
| P5 Trivial      | P3 Major             |      16 |                  0.0804 |                    0.3019 |
| P5 Trivial      | P2 Critical          |       8 |                  0.0402 |                    0.1509 |
| P5 Trivial      | P1 Blocker           |       1 |                  0.005  |                    0.0189 |

## Confusion Matrix

Rows are true priorities and columns are predicted priorities.

| true_priority   |   P1 Blocker |   P2 Critical |   P3 Major |   P4 Minor |   P5 Trivial |
|:----------------|-------------:|--------------:|-----------:|-----------:|-------------:|
| P1 Blocker      |          129 |            50 |         12 |          4 |            4 |
| P2 Critical     |           28 |           135 |         17 |         12 |            6 |
| P3 Major        |            3 |             1 |        175 |         15 |            6 |
| P4 Minor        |            5 |            23 |         21 |        136 |           14 |
| P5 Trivial      |            1 |             8 |         16 |         28 |          146 |

## Interpretation

- `recall` 表示該 priority 的真實資料中，有多少比例被模型正確抓到。
- `most_common_wrong_prediction` 表示該 priority 最常被錯判成哪一類。
- 如果錯誤集中在相鄰 priority，例如 P2 -> P1 / P3，代表主要問題是邊界模糊。
- 如果錯誤跨很多級，例如 P1 -> P5，才代表模型有較嚴重的排序判斷問題。
