# BM25 / BM25F 邊界模型參數搜尋

本實驗測試多組 BM25F 欄位權重與 normalization 參數。

每一組設定都會使用 duplicate links 重新訓練 REP- 權重，重建 DRONE/REP- 特徵，並評估 boundary-refined model。

## 最佳設定

- Config: `summary_high`
- Accuracy: `0.7116`
- Macro F1: `0.7108`
- P2 recall: `0.6515`
- MAE: `0.4302`

## 所有設定結果

| config_name          |   refined_accuracy |   refined_macro_f1 |   refined_p1_recall |   refined_p2_recall |   refined_p3_recall |   refined_mae |   rep_product_weight |   rep_component_weight |   rep_severity_weight |
|:---------------------|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| summary_high         |             0.7116 |             0.7108 |              0.6734 |              0.6515 |               0.84  |        0.4302 |               2.3259 |                 0.9431 |                0.1775 |
| default              |             0.7116 |             0.7106 |              0.6683 |              0.6364 |               0.85  |        0.4291 |               2.3854 |                 1.0108 |                0.2904 |
| description_balanced |             0.7116 |             0.7106 |              0.6683 |              0.6313 |               0.85  |        0.4281 |               2.5432 |                 1.1662 |                0.4445 |
| soft_norm            |             0.7065 |             0.7058 |              0.6935 |              0.596  |               0.825 |        0.4302 |               2.4054 |                 1.0726 |                0.3512 |
| strong_norm          |             0.7065 |             0.7052 |              0.6533 |              0.6313 |               0.85  |        0.4281 |               2.3455 |                 0.9452 |                0.2388 |
