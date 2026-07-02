# Boundary Error Analysis Summary

- True labels inspected: `[2]`
- Predicted labels selected: `[1, 3]`
- True-label rows: `198`
- Selected boundary errors: `37`

## Predicted Label Counts

- `P1 Blocker`: 24
- `P3 Major`: 13

## Top Products In Selected Errors

- `platform`: 21
- `jdt`: 15
- `pde`: 1

## Top Components In Selected Errors

- `ui`: 13
- `debug`: 7
- `core`: 7
- `update  (deprecated - use eclipse>equinox>p2)`: 3
- `ant`: 2
- `swt`: 2
- `compare`: 1
- `resources`: 1

## Top Severities In Selected Errors

- `normal`: 23
- `major`: 9
- `enhancement`: 3
- `minor`: 1
- `critical`: 1

## Keyword Signals With Higher Error Rates

| keyword_feature                   |   selected_error_rate |   true_label_rate |   difference |
|:----------------------------------|----------------------:|------------------:|-------------:|
| has_file_line_stack_terms_count   |                9.8649 |            2.8535 |       7.0113 |
| has_stack_trace_count             |                3.4595 |            1.0404 |       2.4191 |
| has_compiler_terms_count          |                1.5135 |            0.3232 |       1.1903 |
| has_install_update_terms_count    |                0.7568 |            0.2980 |       0.4588 |
| has_exception_terms_count         |                0.6216 |            0.1869 |       0.4348 |
| has_debug_terms_count             |                0.7297 |            0.3535 |       0.3762 |
| has_workspace_project_terms_count |                0.7297 |            0.3737 |       0.3560 |
| has_stack_trace                   |                0.3784 |            0.1212 |       0.2572 |
| has_exception_terms               |                0.3243 |            0.1010 |       0.2233 |
| has_breakpoint_terms_count        |                0.3784 |            0.1818 |       0.1966 |
| has_internal_error_terms_count    |                0.2432 |            0.0556 |       0.1877 |
| has_file_line_stack_terms         |                0.2703 |            0.0859 |       0.1844 |
