# Targeted Error Direction Analysis

- Rows analyzed after overlap removal: `995`
- Removed overlapping training rows: `0`
- Directions: `P1->P2, P4->P2`

## Direction Counts

| error_direction           |   count |   percent |
|:--------------------------|--------:|----------:|
| P1 Blocker -> P2 Critical |      51 |    0.7612 |
| P4 Minor -> P2 Critical   |      16 |    0.2388 |

## P1 Blocker -> P2 Critical

- Error rows: `51`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      41 |    0.8039 |
| major       |       6 |    0.1176 |
| minor       |       1 |    0.0196 |
| critical    |       1 |    0.0196 |
| enhancement |       1 |    0.0196 |
| blocker     |       1 |    0.0196 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      32 |    0.6275 |
| platform  |      16 |    0.3137 |
| pde       |       3 |    0.0588 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |      30 |    0.5882 |
| debug                                         |      11 |    0.2157 |
| core                                          |       7 |    0.1373 |
| swt                                           |       2 |    0.0392 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |    0.0196 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| severity_normal_or_lower |                 0.8431 |                      0.5678 |       0.2753 |
| component_ui_debug_core  |                 0.9412 |                      0.8342 |       0.107  |
| related_pull_to_low      |                 0.2157 |                      0.1206 |       0.0951 |
| low_impact_signal        |                 0.1765 |                      0.1156 |       0.0609 |
| high_impact_signal       |                 0.0784 |                      0.0955 |      -0.017  |
| update_install_signal    |                 0.1373 |                      0.1608 |      -0.0235 |
| product_platform_or_jdt  |                 0.9412 |                      0.9648 |      -0.0236 |
| debug_signal             |                 0.1765 |                      0.2312 |      -0.0547 |
| related_pull_to_high     |                 0.6275 |                      0.7236 |      -0.0962 |
| stack_exception_signal   |                 0.1765 |                      0.4221 |      -0.2456 |
| severity_major_or_higher |                 0.1569 |                      0.4322 |      -0.2753 |

### Top Phrases

| phrase        |   count |
|:--------------|--------:|
| java          |     429 |
| code          |     129 |
| eclipse       |     100 |
| org           |      98 |
| org eclipse   |      93 |
| compiled      |      88 |
| java compiled |      87 |
| compiled code |      87 |
| main          |      82 |
| core          |      64 |
| run           |      50 |
| method        |      45 |

## P4 Minor -> P2 Critical

- Error rows: `16`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      14 |    0.875  |
| major       |       1 |    0.0625 |
| enhancement |       1 |    0.0625 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      11 |    0.6875 |
| platform  |       5 |    0.3125 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |       6 |    0.375  |
| core                                          |       5 |    0.3125 |
| swt                                           |       2 |    0.125  |
| debug                                         |       1 |    0.0625 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |    0.0625 |
| resources                                     |       1 |    0.0625 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| related_pull_to_high     |                 0.4375 |                      0.3216 |       0.1159 |
| product_platform_or_jdt  |                 1      |                      0.9648 |       0.0352 |
| component_ui_debug_core  |                 0.75   |                      0.7186 |       0.0314 |
| severity_major_or_higher |                 0.0625 |                      0.0352 |       0.0273 |
| update_install_signal    |                 0.125  |                      0.1256 |      -0.0006 |
| stack_exception_signal   |                 0.0625 |                      0.0804 |      -0.0179 |
| low_impact_signal        |                 0.1875 |                      0.206  |      -0.0185 |
| severity_normal_or_lower |                 0.9375 |                      0.9648 |      -0.0273 |
| debug_signal             |                 0      |                      0.0352 |      -0.0352 |
| high_impact_signal       |                 0      |                      0.0452 |      -0.0452 |
| related_pull_to_low      |                 0.375  |                      0.4975 |      -0.1225 |

### Top Phrases

| phrase    |   count |
|:----------|--------:|
| error     |      11 |
| view      |      10 |
| composite |       9 |
| class     |       9 |
| swt       |       8 |
| file      |       7 |
| message   |       7 |
| package   |       7 |
| build     |       6 |
| label     |       6 |
| new       |       6 |
| does      |       6 |

## Interpretation

- P1 -> P2 表示高優先級 bug 被模型判太輕，可用 P1/P2 boundary refinement 補強高優先級邊界。
- P4 -> P2 表示低優先級 bug 被模型拉太高，本版改用 cost-sensitive / recall-balanced learning 讓 P4 recall 納入正式選模目標。
