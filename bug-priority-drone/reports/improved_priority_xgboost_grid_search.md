# BM25 / BM25F 邊界模型參數搜尋

本實驗測試多組 BM25F 欄位權重與 normalization 參數。

每一組設定都會使用 duplicate links 重新訓練 REP- 權重，重建 DRONE/REP- 特徵，並評估 boundary-refined model。

## 最佳設定

- Config: `p2_error_keywords`
- Accuracy: `0.7095`
- Macro F1: `0.7094`
- P2 recall: `0.5606`
- MAE: `0.4111`

## 所有設定結果

| config_name       | base_model_type   | base_params                                     | boundary_model_type   | boundary_params                                | boundary_apply   |   boundary_p2_sample_weight |   objective_p1_weight |   objective_p2_weight |   objective_p3_weight |   objective_off_by_one_weight |   objective_collapse_floor |   refined_accuracy |   refined_macro_f1 |   refined_p1_recall |   refined_p2_recall |   refined_p3_recall |   refined_mae |   rep_product_weight |   rep_component_weight |   rep_severity_weight |
|:------------------|:------------------|:------------------------------------------------|:----------------------|:-----------------------------------------------|:-----------------|----------------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|---------------------------:|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| p2_error_keywords | xgboost           | n_estimators=120,max_depth=3,learning_rate=0.08 | xgboost               | n_estimators=80,max_depth=3,learning_rate=0.08 | base_1_2         |                         1.2 |                  0.02 |                  0.16 |                  0.01 |                          0.02 |                       0.55 |             0.7095 |             0.7094 |              0.6985 |              0.5606 |               0.805 |        0.4111 |               2.2575 |                 0.8855 |                0.1334 |
