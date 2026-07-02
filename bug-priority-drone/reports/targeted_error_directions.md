# Targeted Error Direction Analysis

- Rows analyzed after overlap removal: `995`
- Removed overlapping training rows: `0`
- Directions: `P1->P2, P4->P2`

## Direction Counts

| error_direction           |   count |   percent |
|:--------------------------|--------:|----------:|
| P1 Blocker -> P2 Critical |      50 |    0.6849 |
| P4 Minor -> P2 Critical   |      23 |    0.3151 |

## P1 Blocker -> P2 Critical

- Error rows: `50`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      41 |      0.82 |
| major       |       6 |      0.12 |
| critical    |       1 |      0.02 |
| enhancement |       1 |      0.02 |
| blocker     |       1 |      0.02 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      31 |      0.62 |
| platform  |      16 |      0.32 |
| pde       |       3 |      0.06 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |      29 |      0.58 |
| debug                                         |      11 |      0.22 |
| core                                          |       7 |      0.14 |
| swt                                           |       2 |      0.04 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |      0.02 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| severity_normal_or_lower |                   0.84 |                      0.5678 |       0.2722 |
| component_ui_debug_core  |                   0.94 |                      0.8342 |       0.1058 |
| related_pull_to_low      |                   0.22 |                      0.1206 |       0.0994 |
| low_impact_signal        |                   0.18 |                      0.1156 |       0.0644 |
| high_impact_signal       |                   0.08 |                      0.0955 |      -0.0155 |
| update_install_signal    |                   0.14 |                      0.1608 |      -0.0208 |
| product_platform_or_jdt  |                   0.94 |                      0.9648 |      -0.0248 |
| debug_signal             |                   0.18 |                      0.2312 |      -0.0512 |
| related_pull_to_high     |                   0.6  |                      0.7236 |      -0.1236 |
| severity_major_or_higher |                   0.16 |                      0.4322 |      -0.2722 |
| stack_exception_signal   |                   0.14 |                      0.4221 |      -0.2821 |

### Top Phrases

| phrase        |   count |
|:--------------|--------:|
| java          |     424 |
| code          |     128 |
| compiled      |      89 |
| compiled code |      87 |
| java compiled |      87 |
| org           |      86 |
| eclipse       |      86 |
| org eclipse   |      81 |
| main          |      72 |
| core          |      59 |
| run           |      41 |
| eclipse core  |      40 |

## P4 Minor -> P2 Critical

- Error rows: `23`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      18 |    0.7826 |
| enhancement |       3 |    0.1304 |
| major       |       1 |    0.0435 |
| blocker     |       1 |    0.0435 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      12 |    0.5217 |
| platform  |      11 |    0.4783 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |       9 |    0.3913 |
| core                                          |       5 |    0.2174 |
| swt                                           |       3 |    0.1304 |
| team                                          |       2 |    0.087  |
| debug                                         |       1 |    0.0435 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |    0.0435 |
| ant                                           |       1 |    0.0435 |
| resources                                     |       1 |    0.0435 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| related_pull_to_high     |                 0.4348 |                      0.3216 |       0.1132 |
| severity_major_or_higher |                 0.087  |                      0.0352 |       0.0518 |
| product_platform_or_jdt  |                 1      |                      0.9648 |       0.0352 |
| stack_exception_signal   |                 0.087  |                      0.0804 |       0.0066 |
| update_install_signal    |                 0.1304 |                      0.1256 |       0.0048 |
| high_impact_signal       |                 0.0435 |                      0.0452 |      -0.0017 |
| low_impact_signal        |                 0.1739 |                      0.206  |      -0.0321 |
| debug_signal             |                 0      |                      0.0352 |      -0.0352 |
| severity_normal_or_lower |                 0.913  |                      0.9648 |      -0.0518 |
| component_ui_debug_core  |                 0.6522 |                      0.7186 |      -0.0664 |
| related_pull_to_low      |                 0.3478 |                      0.4975 |      -0.1497 |

### Top Phrases

| phrase    |   count |
|:----------|--------:|
| error     |      13 |
| view      |      12 |
| class     |      11 |
| file      |      11 |
| composite |       9 |
| new       |       9 |
| build     |       8 |
| swt       |       8 |
| does      |       8 |
| dialog    |       8 |
| message   |       7 |
| package   |       7 |

## Interpretation

- P1 -> P2 表示高優先級 bug 被模型判太輕，可用 P1/P2 boundary refinement 補強高優先級邊界。
- P4 -> P2 表示低優先級 bug 被模型拉太高，本版改用 cost-sensitive / recall-balanced learning 讓 P4 recall 納入正式選模目標。
