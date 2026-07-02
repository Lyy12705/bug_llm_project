# P2 邊界錯誤人工分類

- 模型：`bm25_boundary_best`
- 選取錯誤筆數：`50`
- 範圍：真實 P2 但預測成 P1 或 P3

## 錯誤方向

| direction         |   count |   percent |
|:------------------|--------:|----------:|
| P2 -> P1 判太嚴重 |      34 |      0.68 |
| P2 -> P3 判太輕   |      16 |      0.32 |

## 主要人工分類

| primary_category                                       |   count |   percent |
|:-------------------------------------------------------|--------:|----------:|
| P2->P1：stack/exception 訊號讓模型判太嚴重             |      16 |      0.32 |
| P2->P3：severity normal/enhancement/minor 讓模型判太輕 |      16 |      0.32 |
| P2->P1：ui/debug/core 模組歷史訊號偏高                 |      13 |      0.26 |
| P2->P1：一般判太嚴重                                   |       4 |      0.08 |
| P2->P1：crash/block/data-loss 等高影響詞讓模型判太嚴重 |       1 |      0.02 |

## 多標籤分類 Flags

| category                      |   count |   percent |
|:------------------------------|--------:|----------:|
| product_focus_platform_or_jdt |      48 |      0.96 |
| component_focus_ui_debug_core |      40 |      0.8  |
| predicted_too_high_p1         |      34 |      0.68 |
| related_pull_to_p3            |      33 |      0.66 |
| severity_is_normal            |      33 |      0.66 |
| product_is_platform           |      25 |      0.5  |
| related_pull_to_p1_p2         |      24 |      0.48 |
| product_is_jdt                |      23 |      0.46 |
| component_is_ui               |      18 |      0.36 |
| has_stack_or_exception        |      18 |      0.36 |
| predicted_too_low_p3          |      16 |      0.32 |
| component_is_debug            |      13 |      0.26 |
| has_install_update_terms_any  |      12 |      0.24 |
| component_is_core             |       9 |      0.18 |
| has_high_impact_terms         |       4 |      0.08 |
| has_api_break_terms_any       |       2 |      0.04 |
| has_regression_terms_any      |       0 |      0    |

## 錯誤方向 x Severity

| error_direction   |   enhancement |   major |   minor |   normal |   total |
|:------------------|--------------:|--------:|--------:|---------:|--------:|
| P2 -> P1 判太嚴重 |             1 |      13 |       0 |       20 |      34 |
| P2 -> P3 判太輕   |             2 |       0 |       1 |       13 |      16 |

## 錯誤方向 x Product

| error_direction   |   jdt |   pde |   platform |   total |
|:------------------|------:|------:|-----------:|--------:|
| P2 -> P1 判太嚴重 |    13 |     2 |         19 |      34 |
| P2 -> P3 判太輕   |    10 |     0 |          6 |      16 |

## 錯誤方向 x Component Focus

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |       9 |     25 |      34 |
| P2 -> P3 判太輕   |       1 |     15 |      16 |

## 錯誤方向 x Stack/Exception

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |      18 |     16 |      34 |
| P2 -> P3 判太輕   |      14 |      2 |      16 |

## 錯誤方向 x Related-report Pull

| error_direction   |   False |   True |   total |
|:------------------|--------:|-------:|--------:|
| P2 -> P1 判太嚴重 |      16 |     18 |      34 |
| P2 -> P3 判太輕   |       1 |     15 |      16 |

## 判讀

- `P2 -> P1 判太嚴重` 表示模型把 critical bug 判成 blocker。
- `P2 -> P3 判太輕` 表示模型把 critical bug 判成 major。
- 多標籤 flags 不是互斥分類；同一筆 bug 可以同時是 `severity_is_normal`、`has_stack_or_exception`、`product_is_jdt`。
- `manual_primary_category` 是為了報告方便而指定的一個主要錯誤原因。
