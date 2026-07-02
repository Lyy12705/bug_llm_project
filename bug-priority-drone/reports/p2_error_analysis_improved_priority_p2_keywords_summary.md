# 邊界錯誤分析摘要

- 檢查的真實標籤：`[2]`
- 選取的預測標籤：`[1, 3]`
- 真實標籤資料筆數：`198`
- 選取的邊界錯誤筆數：`45`

## 預測標籤統計

- `P1 Blocker`: 28
- `P3 Major`: 17

## 錯誤案例中最常見的 Product

- `jdt`: 22
- `platform`: 21
- `pde`: 2

## 錯誤案例中最常見的 Component

- `ui`: 16
- `debug`: 9
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `user assistance`: 1
- `resources`: 1

## 錯誤案例中最常見的 Severity

- `normal`: 29
- `major`: 12
- `enhancement`: 3
- `minor`: 1

## 錯誤率較高的 Keyword Signals

| keyword_feature                   |   selected_error_rate |   true_label_rate |   difference |
|:----------------------------------|----------------------:|------------------:|-------------:|
| has_file_line_stack_terms_count   |               11.6667 |            2.8535 |       8.8131 |
| has_stack_trace_count             |                3.7111 |            1.0404 |       2.6707 |
| has_compiler_terms_count          |                1.2444 |            0.3232 |       0.9212 |
| has_exception_terms_count         |                0.6444 |            0.1869 |       0.4576 |
| has_workspace_project_terms_count |                0.8000 |            0.3737 |       0.4263 |
| has_install_update_terms_count    |                0.6222 |            0.2980 |       0.3242 |
| has_stack_trace                   |                0.4222 |            0.1212 |       0.3010 |
| has_debug_terms_count             |                0.6222 |            0.3535 |       0.2687 |
| has_exception_terms               |                0.3556 |            0.1010 |       0.2545 |
| has_file_line_stack_terms         |                0.3111 |            0.0859 |       0.2253 |
| has_internal_error_terms_count    |                0.2444 |            0.0556 |       0.1889 |
| has_widget_disposed_terms_count   |                0.2000 |            0.0455 |       0.1545 |
