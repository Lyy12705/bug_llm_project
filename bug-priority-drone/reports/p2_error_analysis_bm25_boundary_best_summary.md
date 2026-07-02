# 邊界錯誤分析摘要

- 檢查的真實標籤：`[2]`
- 選取的預測標籤：`[1, 3]`
- 真實標籤資料筆數：`198`
- 選取的邊界錯誤筆數：`50`

## 預測標籤統計

- `P1 Blocker`: 34
- `P3 Major`: 16

## 錯誤案例中最常見的 Product

- `platform`: 25
- `jdt`: 23
- `pde`: 2

## 錯誤案例中最常見的 Component

- `ui`: 18
- `debug`: 13
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `resources`: 1
- `build`: 1

## 錯誤案例中最常見的 Severity

- `normal`: 33
- `major`: 13
- `enhancement`: 3
- `minor`: 1

## 錯誤率較高的 Keyword Signals

| keyword_feature                  |   selected_error_rate |   true_label_rate |   difference |
|:---------------------------------|----------------------:|------------------:|-------------:|
| has_stack_trace                  |                0.3600 |            0.1212 |       0.2388 |
| has_exception_terms              |                0.3000 |            0.1010 |       0.1990 |
| has_install_update_terms         |                0.2400 |            0.1162 |       0.1238 |
| summary_has_exception_terms      |                0.1400 |            0.0404 |       0.0996 |
| summary_has_stack_trace          |                0.1000 |            0.0303 |       0.0697 |
| has_test_failure_terms           |                0.0800 |            0.0455 |       0.0345 |
| has_api_break_terms              |                0.0400 |            0.0101 |       0.0299 |
| has_blocking_terms               |                0.0400 |            0.0152 |       0.0248 |
| summary_has_crash_terms          |                0.0200 |            0.0051 |       0.0149 |
| has_crash_terms                  |                0.0200 |            0.0051 |       0.0149 |
| summary_has_install_update_terms |                0.0600 |            0.0455 |       0.0145 |
| has_data_loss_terms              |                0.0200 |            0.0101 |       0.0099 |
