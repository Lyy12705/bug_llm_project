# P2 錯誤人工分類比較

本文件比較 boundary-refined improved 與 BM25/BM25F tuned boundary model 的剩餘 P2 邊界錯誤。

## Summary Counts

| model                     |   total_p2_to_p1_or_p3_errors |   p2_to_p1_too_high |   p2_to_p3_too_low |   severity_normal |   stack_or_exception |   product_platform_or_jdt |   component_ui_debug_core |   related_pull_to_p3 |   high_impact_terms |
|:--------------------------|------------------------------:|--------------------:|-------------------:|------------------:|---------------------:|--------------------------:|--------------------------:|---------------------:|--------------------:|
| boundary_refined_improved |                            53 |                  37 |                 16 |                35 |                   18 |                        51 |                        42 |                   34 |                   4 |
| bm25_boundary_best        |                            50 |                  34 |                 16 |                33 |                   18 |                        48 |                        40 |                   33 |                   4 |

## Primary Manual Categories

| model                     | primary_category                                       |   count |   percent |
|:--------------------------|:-------------------------------------------------------|--------:|----------:|
| bm25_boundary_best        | P2->P1：stack/exception 訊號讓模型判太嚴重             |      16 |    0.32   |
| bm25_boundary_best        | P2->P3：severity normal/enhancement/minor 讓模型判太輕 |      16 |    0.32   |
| bm25_boundary_best        | P2->P1：ui/debug/core 模組歷史訊號偏高                 |      13 |    0.26   |
| bm25_boundary_best        | P2->P1：一般判太嚴重                                   |       4 |    0.08   |
| bm25_boundary_best        | P2->P1：crash/block/data-loss 等高影響詞讓模型判太嚴重 |       1 |    0.02   |
| boundary_refined_improved | P2->P1：stack/exception 訊號讓模型判太嚴重             |      16 |    0.3019 |
| boundary_refined_improved | P2->P3：severity normal/enhancement/minor 讓模型判太輕 |      16 |    0.3019 |
| boundary_refined_improved | P2->P1：ui/debug/core 模組歷史訊號偏高                 |      15 |    0.283  |
| boundary_refined_improved | P2->P1：一般判太嚴重                                   |       5 |    0.0943 |
| boundary_refined_improved | P2->P1：crash/block/data-loss 等高影響詞讓模型判太嚴重 |       1 |    0.0189 |

## Interpretation

- BM25/BM25F tuned 後，P2 被判成 P1/P3 的錯誤從 53 筆降到 50 筆。
- 減少的 3 筆主要來自 P2 -> P1 判太嚴重，也就是模型較少把 P2 誤判成 P1。
- 剩餘錯誤仍高度集中在 platform/jdt 產品，以及 ui/debug/core component。
- P2 -> P1 錯誤常伴隨 stack trace 或 exception 訊號，模型容易把這類 critical bug 拉高成 blocker。
- P2 -> P3 錯誤多數 severity 是 normal/enhancement/minor，模型容易因 severity 不夠高而判太輕。
