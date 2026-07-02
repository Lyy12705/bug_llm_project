# P2 邊界錯誤人工分類

- 模型：`improved_priority_p2_keywords`
- 選取錯誤筆數：`45`
- 範圍：真實 P2 但預測成 P1 或 P3

## 錯誤方向

| direction         |   count |   percent |
|:------------------|--------:|----------:|
| P2 -> P1 判太嚴重 |      28 |    0.6222 |
| P2 -> P3 判太輕   |      17 |    0.3778 |

## 主要人工分類

| primary_category                                       |   count |   percent |
|:-------------------------------------------------------|--------:|----------:|
| P2->P1：stack/exception 訊號讓模型判太嚴重             |      17 |    0.3778 |
| P2->P3：severity normal/enhancement/minor 讓模型判太輕 |      16 |    0.3556 |
| P2->P1：ui/debug/core 模組歷史訊號偏高                 |       7 |    0.1556 |
| P2->P1：一般判太嚴重                                   |       4 |    0.0889 |
| P2->P3：related reports 偏向 P3                        |       1 |    0.0222 |

## 多標籤分類 Flags

| category                      |   count |   percent |
|:------------------------------|--------:|----------:|
| product_focus_platform_or_jdt |      43 |    0.9556 |
| component_focus_ui_debug_core |      34 |    0.7556 |
| related_pull_to_p3            |      30 |    0.6667 |
| severity_is_normal            |      29 |    0.6444 |
| predicted_too_high_p1         |      28 |    0.6222 |
| product_is_jdt                |      22 |    0.4889 |
| product_is_platform           |      21 |    0.4667 |
| related_pull_to_p1_p2         |      20 |    0.4444 |
| has_stack_or_exception        |      19 |    0.4222 |
| predicted_too_low_p3          |      17 |    0.3778 |
| component_is_ui               |      16 |    0.3556 |
| component_is_core             |       9 |    0.2    |
| component_is_debug            |       9 |    0.2    |
| has_install_update_terms_any  |       9 |    0.2    |
| has_high_impact_terms         |       3 |    0.0667 |
| has_api_break_terms_any       |       1 |    0.0222 |
| has_regression_terms_any      |       0 |    0      |

## 錯誤方向 x Severity

| error_direction   |   enhancement |   major |   minor |   normal |   total |
|:------------------|--------------:|--------:|--------:|---------:|--------:|
| P2 -> P1 判太嚴重 |             1 |      11 |       0 |       16 |      28 |
| P2 -> P3 判太輕   |             2 |       1 |       1 |       13 |      17 |

## 錯誤方向 x Product

| error_direction   |   jdt |   pde |   platform |   total |
|:------------------|------:|------:|-----------:|--------:|
| P2 -> P1 判太嚴重 |    12 |     2 |         14 |      28 |
| P2 -> P3 判太輕   |    10 |     0 |          7 |      17 |

## 錯誤方向 x Component Focus

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |       9 |     19 |      28 |
| P2 -> P3 判太輕   |       2 |     15 |      17 |

## 錯誤方向 x Stack/Exception

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |      11 |     17 |      28 |
| P2 -> P3 判太輕   |      15 |      2 |      17 |

## 錯誤方向 x Related-report Pull

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |      14 |     14 |      28 |
| P2 -> P3 判太輕   |       1 |     16 |      17 |

## 判讀

- `P2 -> P1 判太嚴重` 表示模型把 critical bug 判成 blocker。
- `P2 -> P3 判太輕` 表示模型把 critical bug 判成 major。
- 多標籤 flags 不是互斥分類；同一筆 bug 可以同時是 `severity_is_normal`、`has_stack_or_exception`、`product_is_jdt`。
- `manual_primary_category` 是為了報告方便而指定的一個主要錯誤原因。
