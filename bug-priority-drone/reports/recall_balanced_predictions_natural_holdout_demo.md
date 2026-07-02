# Natural Holdout Row-Level Prediction Demo

這份檔案用來展示目前最佳模型對每一筆測試 ticket 的判斷結果。

## Metrics

| Metric                   |    Value |
|:-------------------------|---------:|
| Rows                     | 995      |
| Removed training overlap |   0      |
| Accuracy                 |   0.7266 |
| Macro F1                 |   0.7265 |
| Off-by-one accuracy      |   0.9045 |
| MAE                      |   0.404  |
| P1 recall                |   0.6583 |
| P2 recall                |   0.6869 |
| P3 recall                |   0.835  |
| P4 recall                |   0.7136 |
| P5 recall                |   0.7387 |

## Example Correct Predictions

|   id | true_priority   | predicted_priority   | is_correct   | summary                                                                                                 | severity   | product   | component   |
|-----:|:----------------|:---------------------|:-------------|:--------------------------------------------------------------------------------------------------------|:-----------|:----------|:------------|
| 1739 | P1              | P1                   | True         | save java class with errors confuses hot code replace (1gkxl6w)                                         | critical   | jdt       | debug       |
| 3416 | P1              | P1                   | True         | jck 1.4 - binc - the new method is less accessible than the old one (1gk7vxd)                           | normal     | jdt       | core        |
| 5083 | P1              | P1                   | True         | breakpoint not hit                                                                                      | normal     | jdt       | debug       |
| 5150 | P1              | P1                   | True         | compare with patch cannot read vcm's cvs patch file                                                     | normal     | jdt       | ui          |
| 5162 | P1              | P1                   | True         | 1.0 -- jsp breakpoints don't get removed                                                                | normal     | jdt       | debug       |
| 5216 | P1              | P1                   | True         | tvt1: cannot type characters a, c, f, p, v, x, y and z in task list description field on german machine | critical   | platform  | ui          |
| 5225 | P1              | P1                   | True         | 1.0 -- casting problem in runtolineaction class                                                         | critical   | jdt       | debug       |
| 5318 | P1              | P1                   | True         | jface textviewer illegal setstylerange argument (styledtext indexoutofbounds error)                     | normal     | platform  | ui          |

## Example Wrong Predictions

|   id | true_priority   | predicted_priority   | is_correct   | summary                                                          | severity    | product   | component   |
|-----:|:----------------|:---------------------|:-------------|:-----------------------------------------------------------------|:------------|:----------|:------------|
| 1615 | P1              | P3                   | False        | launching on j9 lets pop up a dos console (1geuoei)              | critical    | jdt       | debug       |
| 1623 | P1              | P3                   | False        | stackframe selected but toolbar actions disable (1gev0l7)        | minor       | jdt       | debug       |
| 1660 | P1              | P3                   | False        | do not prompt for source when no source attachment               | normal      | jdt       | debug       |
| 1665 | P1              | P2                   | False        | drop to frame hangs if after invoke (1gh3xda)                    | minor       | jdt       | debug       |
| 1725 | P1              | P3                   | False        | do not scroll current line to top of page (1gkde48)              | normal      | jdt       | ui          |
| 2041 | P1              | P3                   | False        | post-release: paf: workbench crash (1gdsb6z)                     | enhancement | platform  | ui          |
| 2394 | P1              | P3                   | False        | [ui] walkback printed to console (not log) on shutdown (1gey4c6) | normal      | platform  | ui          |
| 2481 | P1              | P3                   | False        | [jface] strange shift select behavior in debug view (1gf89mp)    | minor       | platform  | ui          |

## Output CSV

- `reports/recall_balanced_predictions_natural_holdout.csv`：全部測試資料逐筆預測結果。
