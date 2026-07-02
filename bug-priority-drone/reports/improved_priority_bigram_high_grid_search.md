# BM25 / BM25F 邊界模型參數搜尋

本實驗測試多組 BM25F 欄位權重與 normalization 參數。

每一組設定都會使用 duplicate links 重新訓練 REP- 權重，重建 DRONE/REP- 特徵，並評估 boundary-refined model。

## 最佳設定

- Config: `bigram_high`
- Accuracy: `0.7196`
- Macro F1: `0.7195`
- P2 recall: `0.6869`
- MAE: `0.4111`

## 所有設定結果

| config_name   | base_model_type   | base_params   | boundary_model_type   | boundary_params   | boundary_apply   |   boundary_p2_sample_weight |   objective_p1_weight |   objective_p2_weight |   objective_p3_weight |   objective_off_by_one_weight |   objective_collapse_floor |   refined_accuracy |   refined_macro_f1 |   refined_p1_recall |   refined_p2_recall |   refined_p3_recall |   refined_mae |   rep_product_weight |   rep_component_weight |   rep_severity_weight |
|:--------------|:------------------|:--------------|:----------------------|:------------------|:-----------------|----------------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|---------------------------:|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| bigram_high   | sgd_log           | alpha=0.01    | sgd_log               | alpha=0.1         | base_1_2_3       |                         1.2 |                  0.02 |                  0.16 |                  0.01 |                          0.02 |                       0.55 |             0.7196 |             0.7195 |              0.6482 |              0.6869 |               0.835 |        0.4111 |               2.2987 |                  0.933 |                0.1742 |
