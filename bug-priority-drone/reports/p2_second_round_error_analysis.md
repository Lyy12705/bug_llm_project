# P2 Second-Round Error Analysis

This report inspects the remaining true-P2 rows predicted as P1 or P3 by the current best model.

## Scope

- Remaining P2 boundary errors: `45`
- All true-P2 rows in natural_holdout: `198`

## Error Direction

| direction           |   count |   percent |
|:--------------------|--------:|----------:|
| P2_to_P1_too_severe |      28 |    0.6222 |
| P2_to_P3_too_light  |      17 |    0.3778 |

## Main Second-Round Tags

| tag_combo                                                                                                                       |   count |   percent |
|:--------------------------------------------------------------------------------------------------------------------------------|--------:|----------:|
| normal/lower severity pulled to P3; missing obvious high-impact text; related reports lean P3; platform/jdt boundary case       |       9 |    0.2    |
| stack/exception pushed to P1; ui/debug/core historical signal; related reports lean P1; normal severity but severe-looking text |       4 |    0.0889 |
| general boundary ambiguity                                                                                                      |       4 |    0.0889 |
| ui/debug/core historical signal; related reports lean P1                                                                        |       4 |    0.0889 |
| stack/exception pushed to P1; ui/debug/core historical signal; normal severity but severe-looking text                          |       3 |    0.0667 |
| ui/debug/core historical signal                                                                                                 |       3 |    0.0667 |
| stack/exception pushed to P1; ui/debug/core historical signal; related reports lean P1                                          |       3 |    0.0667 |
| normal/lower severity pulled to P3; related reports lean P3; install/update wording is ambiguous; platform/jdt boundary case    |       3 |    0.0667 |
| normal/lower severity pulled to P3; related reports lean P3; platform/jdt boundary case                                         |       3 |    0.0667 |
| stack/exception pushed to P1; related reports lean P1; normal severity but severe-looking text                                  |       2 |    0.0444 |
| stack/exception pushed to P1                                                                                                    |       1 |    0.0222 |
| stack/exception pushed to P1; normal severity but severe-looking text                                                           |       1 |    0.0222 |

## Signal Rates Compared With All True P2

| signal                     |   error_rate |   all_true_p2_rate |   difference |
|:---------------------------|-------------:|-------------------:|-------------:|
| stack_exception_signal     |       0.4222 |             0.1313 |       0.2909 |
| related_pull_to_p1         |       0.3333 |             0.197  |       0.1364 |
| severity_major_or_critical |       0.2667 |             0.1465 |       0.1202 |
| update_install_signal      |       0.2    |             0.1212 |       0.0788 |
| debug_signal               |       0.1778 |             0.1364 |       0.0414 |
| high_impact_signal         |       0.0667 |             0.0303 |       0.0364 |
| related_pull_to_p3         |       0.6667 |             0.6566 |       0.0101 |
| regression_signal          |       0      |             0      |       0      |
| ui_signal                  |       0      |             0      |       0      |
| product_platform_or_jdt    |       0.9556 |             0.9646 |      -0.0091 |
| component_ui_debug_core    |       0.7556 |             0.8182 |      -0.0626 |
| severity_normal_or_lower   |       0.7333 |             0.8535 |      -0.1202 |
| related_pull_to_p2         |       0.1111 |             0.2879 |      -0.1768 |
| weak_text_signal           |       0.4444 |             0.6869 |      -0.2424 |

## P2 -> P1 Top Phrases

| phrase      |   count |
|:------------|--------:|
| java        |      87 |
| build       |      27 |
| eclipse     |      25 |
| org         |      16 |
| org eclipse |      15 |
| code        |      13 |
| error       |      12 |
| project     |      12 |
| core        |      11 |
| following   |      10 |
| lang        |      10 |
| java lang   |      10 |
| exception   |       9 |
| internal    |       9 |
| menu        |       9 |

## P2 -> P3 Top Phrases

| phrase        |   count |
|:--------------|--------:|
| java          |      14 |
| view          |      10 |
| project       |       9 |
| notes         |       9 |
| package       |       9 |
| path          |       8 |
| build         |       8 |
| source        |       7 |
| folder        |       7 |
| files         |       7 |
| packages      |       7 |
| int           |       6 |
| code          |       6 |
| errors        |       6 |
| packages view |       6 |

## Direction x Severity

| error_direction     |   enhancement |   major |   minor |   normal |
|:--------------------|--------------:|--------:|--------:|---------:|
| P2_to_P1_too_severe |             1 |      11 |       0 |       16 |
| P2_to_P3_too_light  |             2 |       1 |       1 |       13 |

## Direction x Product

| error_direction     |   jdt |   pde |   platform |
|:--------------------|------:|------:|-----------:|
| P2_to_P1_too_severe |    12 |     2 |         14 |
| P2_to_P3_too_light  |    10 |     0 |          7 |

## Interpretation

- P2 -> P1 errors are mostly cases where text looks severe, especially stack traces, exceptions, UI/debug/core modules, or related reports leaning to P1.
- P2 -> P3 errors are mostly normal/lower severity rows or rows without clear high-impact keywords, so they look closer to ordinary major bugs.
- This suggests the next improvement should focus on a calibrated P1/P2/P3 boundary layer and better ambiguity handling, not simply adding more global P2 weight.
