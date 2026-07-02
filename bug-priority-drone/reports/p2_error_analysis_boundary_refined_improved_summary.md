# 邊界錯誤分析摘要

- 檢查的真實標籤：`[2]`
- 選取的預測標籤：`[1, 3]`
- 真實標籤資料筆數：`198`
- 選取的邊界錯誤筆數：`53`

## 預測標籤統計

- `P1 Blocker`: 37
- `P3 Major`: 16

## 錯誤案例中最常見的 Product

- `platform`: 27
- `jdt`: 24
- `pde`: 2

## 錯誤案例中最常見的 Component

- `ui`: 19
- `debug`: 14
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `resources`: 1
- `build`: 1

## 錯誤案例中最常見的 Severity

- `normal`: 35
- `major`: 14
- `enhancement`: 3
- `minor`: 1

## 錯誤率較高的 Keyword Signals

| keyword_feature                  |   selected_error_rate |   true_label_rate |   difference |
|:---------------------------------|----------------------:|------------------:|-------------:|
| has_stack_trace                  |                0.3396 |            0.1212 |       0.2184 |
| has_exception_terms              |                0.2830 |            0.1010 |       0.1820 |
| has_install_update_terms         |                0.2264 |            0.1162 |       0.1103 |
| summary_has_exception_terms      |                0.1321 |            0.0404 |       0.0917 |
| summary_has_stack_trace          |                0.0943 |            0.0303 |       0.0640 |
| has_test_failure_terms           |                0.0755 |            0.0455 |       0.0300 |
| has_api_break_terms              |                0.0377 |            0.0101 |       0.0276 |
| has_blocking_terms               |                0.0377 |            0.0152 |       0.0226 |
| summary_has_crash_terms          |                0.0189 |            0.0051 |       0.0138 |
| has_crash_terms                  |                0.0189 |            0.0051 |       0.0138 |
| summary_has_install_update_terms |                0.0566 |            0.0455 |       0.0111 |
| has_data_loss_terms              |                0.0189 |            0.0101 |       0.0088 |
