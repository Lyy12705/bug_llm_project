# Label Noise / Ambiguous Priority Analysis

This report looks for dataset patterns where the same metadata context maps to mixed priorities.

## Overall Priority Distribution

|   rows | dominant_priority   |   dominant_share |   entropy |   normalized_entropy |   unique_priorities |   P1 |   P2 |   P3 |   P4 |   P5 |
|-------:|:--------------------|-----------------:|----------:|---------------------:|--------------------:|-----:|-----:|-----:|-----:|-----:|
|   9949 | P3                  |           0.2009 |    2.3219 |                    1 |                   5 | 1989 | 1977 | 1999 | 1995 | 1989 |

## Severity / Priority Mismatch Signals

| case                                    |   rows | dominant_priority   |   dominant_share |   P1 |   P2 |   P3 |   P4 |   P5 |
|:----------------------------------------|-------:|:--------------------|-----------------:|-----:|-----:|-----:|-----:|-----:|
| normal_or_lower_but_high_priority       |   2893 | P2                  |           0.5769 | 1224 | 1669 |    0 |    0 |    0 |
| major_critical_blocker_but_low_priority |    175 | P4                  |           0.6    |    0 |    0 |    0 |  105 |   70 |
| enhancement_but_not_low_priority        |    588 | P3                  |           0.6769 |   27 |  163 |  398 |    0 |    0 |

## Top Ambiguous Groups

| severity   | grouping                            |   rows | dominant_priority   |   dominant_share |   entropy |   normalized_entropy |   unique_priorities |   P1 |   P2 |   P3 |   P4 |   P5 | product   | component   | severity_group   | impact_keyword_group    |
|:-----------|:------------------------------------|-------:|:--------------------|-----------------:|----------:|---------------------:|--------------------:|-----:|-----:|-----:|-----:|-----:|:----------|:------------|:-----------------|:------------------------|
| nan        | component                           |   5086 | P3                  |           0.2401 |    2.2995 |               0.9903 |                   5 |  890 | 1000 | 1221 | 1210 |  765 | nan       | ui          | nan              | nan                     |
| normal     | severity                            |   5909 | P2                  |           0.2447 |    2.2956 |               0.9887 |                   5 | 1163 | 1446 | 1388 |  829 | 1083 | nan       | nan         | nan              | nan                     |
| nan        | product                             |   5969 | P5                  |           0.2781 |    2.281  |               0.9824 |                   5 |  974 |  840 | 1331 | 1164 | 1660 | platform  | nan         | nan              | nan                     |
| nan        | severity_group+impact_keyword_group |    471 | P2                  |           0.276  |    2.27   |               0.9776 |                   5 |   56 |  130 |  108 |   82 |   95 | nan       | nan         | normal           | no_clear_impact_keyword |
| nan        | product+component                   |   2855 | P3                  |           0.317  |    2.2363 |               0.9631 |                   5 |  404 |  344 |  905 |  545 |  657 | platform  | ui          | nan              | nan                     |
| nan        | component                           |     58 | P1                  |           0.2759 |    2.2353 |               0.9627 |                   5 |   16 |   14 |   14 |    8 |    6 | nan       | ant         | nan              | nan                     |
| nan        | product+component                   |     58 | P1                  |           0.2759 |    2.2353 |               0.9627 |                   5 |   16 |   14 |   14 |    8 |    6 | platform  | ant         | nan              | nan                     |
| normal     | severity+product                    |   3335 | P5                  |           0.2942 |    2.227  |               0.9591 |                   5 |  476 |  529 |  942 |  407 |  981 | platform  | nan         | nan              | nan                     |
| nan        | product                             |   3689 | P2                  |           0.289  |    2.2206 |               0.9564 |                   5 |  936 | 1066 |  589 |  781 |  317 | jdt       | nan         | nan              | nan                     |
| nan        | severity_group+impact_keyword_group |     52 | P2                  |           0.3462 |    2.2168 |               0.9547 |                   5 |   11 |   18 |    9 |    6 |    8 | nan       | nan         | major            | no_clear_impact_keyword |
| nan        | component                           |    692 | P2                  |           0.2977 |    2.1714 |               0.9352 |                   5 |  139 |  206 |  199 |   47 |  101 | nan       | core        | nan              | nan                     |
| nan        | product+component                   |    692 | P2                  |           0.2977 |    2.1714 |               0.9352 |                   5 |  139 |  206 |  199 |   47 |  101 | jdt       | core        | nan              | nan                     |
| nan        | severity_group+impact_keyword_group |     30 | P5                  |           0.3    |    2.1675 |               0.9335 |                   5 |    2 |    4 |    8 |    7 |    9 | nan       | nan         | minor            | no_clear_impact_keyword |
| minor      | severity+product                    |    205 | P4                  |           0.4    |    2.1512 |               0.9265 |                   5 |   21 |   36 |   37 |   82 |   29 | jdt       | nan         | nan              | nan                     |
| nan        | product+component                   |    265 | P1                  |           0.2755 |    2.1403 |               0.9218 |                   5 |   73 |   68 |   66 |   48 |   10 | pde       | ui          | nan              | nan                     |
| nan        | product                             |    283 | P1                  |           0.2792 |    2.1386 |               0.921  |                   5 |   79 |   70 |   74 |   49 |   11 | pde       | nan         | nan              | nan                     |
| nan        | component                           |    277 | P3                  |           0.3935 |    2.1202 |               0.9131 |                   5 |   67 |   35 |  109 |   42 |   24 | nan       | resources   | nan              | nan                     |
| nan        | product+component                   |    277 | P3                  |           0.3935 |    2.1202 |               0.9131 |                   5 |   67 |   35 |  109 |   42 |   24 | platform  | resources   | nan              | nan                     |
| nan        | component                           |    126 | P2                  |           0.2778 |    2.1194 |               0.9128 |                   5 |   32 |   35 |   33 |   22 |    4 | nan       | compare     | nan              | nan                     |
| nan        | product+component                   |    126 | P2                  |           0.2778 |    2.1194 |               0.9128 |                   5 |   32 |   35 |   33 |   22 |    4 | platform  | compare     | nan              | nan                     |

## Duplicate-Link Priority Disagreements

|   dupe_of | bug_ids             | products     | components   |   rows | dominant_priority   |   dominant_share |   entropy |   normalized_entropy |   unique_priorities |   P1 |   P2 |   P3 |   P4 |   P5 |
|----------:|:--------------------|:-------------|:-------------|-------:|:--------------------|-----------------:|----------:|---------------------:|--------------------:|-----:|-----:|-----:|-----:|-----:|
|      6513 | 2844,11367,14563    | platform     | ui           |      3 | P2                  |           0.3333 |     1.585 |               0.6826 |                   3 |    0 |    1 |    1 |    0 |    1 |
|      1546 | 1616,1620,1722,5573 | jdt          | debug        |      4 | P3                  |           0.5    |     1.5   |               0.646  |                   3 |    1 |    1 |    2 |    0 |    0 |
|      2051 | 2404,2610,2611,2758 | platform     | ui           |      4 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    2 |    0 |    2 |    0 |    0 |
|      1663 | 1522,1664           | jdt          | debug        |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      2135 | 2785,24027          | platform     | ui           |      2 | P3                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    0 |    1 |    1 |    0 |
|      3035 | 3094,5685           | platform     | resources    |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      3106 | 9894,10005          | platform     | resources    |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |
|      3435 | 3308,3421           | jdt          | core         |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      3840 | 4008,5771           | platform     | ui           |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    0 |    1 |    0 |
|      4133 | 4276,11359          | jdt          | ui           |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    0 |    0 |    1 |    0 |
|      4383 | 1604,1732           | jdt          | debug        |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      5163 | 1608,5510           | jdt          | debug        |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      5640 | 5565,5928           | jdt          | debug,ui     |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    0 |    0 |    0 |    1 |
|      7245 | 3780,10402          | jdt          | debug        |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |
|      7743 | 1965,2819           | platform     | ui           |      2 | P2                  |           0.5    |     1     |               0.4307 |                   2 |    0 |    1 |    1 |    0 |    0 |
|      8772 | 6949,11986          | platform     | debug        |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |
|      9211 | 9856,12003          | platform     | swt          |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    0 |    0 |    1 |    0 |
|     12665 | 12842,15158         | jdt,platform | debug        |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |
|     13131 | 13203,13354         | jdt,platform | ui           |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |
|     13554 | 13640,15567         | platform     | debug        |      2 | P1                  |           0.5    |     1     |               0.4307 |                   2 |    1 |    1 |    0 |    0 |    0 |

## Interpretation

- Ambiguous groups mean the same severity/product/component or keyword context has multiple priority labels, so some prediction errors may be label ambiguity rather than pure model failure.
- Duplicate-link disagreements are especially important because duplicate reports should be semantically related; mixed priorities inside duplicate groups suggest historical triage inconsistency.
- For reporting, this supports explaining why P2 is hard: P2 sits between P1 and P3, and many P2-like contexts are not labeled consistently.
