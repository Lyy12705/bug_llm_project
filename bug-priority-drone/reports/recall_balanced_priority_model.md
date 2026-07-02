# Recall-Balanced Priority Model

This experiment keeps the DRONE/REP- feature design and uses a more formal recall-balanced training strategy:

1. Cost-sensitive class sample weights for the base classifier.
2. P1/P2 boundary classifier for high-priority boundary refinement.

Selection uses validation-set `macro recall + minimum recall + macro F1`, with MAE and low-recall penalties.

## Best Natural-Holdout Result

| model_stage          | eval_set        |   accuracy |   macro_f1 |   macro_recall |   min_recall |   off_by_one_accuracy |   mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:---------------------|:----------------|-----------:|-----------:|---------------:|-------------:|----------------------:|------:|------------:|------------:|------------:|------------:|------------:|
| recall_balanced_best | natural_holdout |     0.7266 |     0.7265 |         0.7265 |       0.6583 |                0.9045 | 0.404 |      0.6583 |      0.6869 |       0.835 |      0.7136 |      0.7387 |

## Top Validation Candidates

| stage                        | base_model_type   | base_params   |   base_p1_sample_weight |   base_p2_sample_weight |   base_p4_sample_weight | p1p2_model_type   | p1p2_params   |   boundary_p1_sample_weight |   boundary_p2_sample_weight |   validation_objective |   validation_macro_recall |   validation_min_recall |   validation_p1_recall |   validation_p2_recall |   validation_p4_recall |   natural_accuracy |   natural_macro_recall |   natural_min_recall |
|:-----------------------------|:------------------|:--------------|------------------------:|------------------------:|------------------------:|:------------------|:--------------|----------------------------:|----------------------------:|-----------------------:|--------------------------:|------------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------------:|---------------------:|
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.03    |                       1 |                    1    |                     1.2 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.614  |                    0.6944 |                   0.624 |                  0.624 |                  0.624 |                  0.644 |             0.7266 |                 0.7265 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1.4 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6106 |                    0.6936 |                   0.62  |                  0.62  |                  0.624 |                  0.668 |             0.7246 |                 0.7245 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1.15 |                     1.4 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6102 |                    0.6936 |                   0.62  |                  0.62  |                  0.636 |                  0.66  |             0.7266 |                 0.7265 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1.2 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6094 |                    0.6936 |                   0.62  |                  0.62  |                  0.636 |                  0.632 |             0.7286 |                 0.7285 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.03    |                       1 |                    1.15 |                     1.2 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6083 |                    0.692  |                   0.624 |                  0.624 |                  0.636 |                  0.632 |             0.7246 |                 0.7245 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1.15 |                     1.4 | sgd_log           | alpha=0.05    |                           1 |                           1 |                 1.6054 |                    0.6912 |                   0.62  |                  0.624 |                  0.62  |                  0.66  |             0.7216 |                 0.7215 |               0.6533 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1   | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6053 |                    0.6944 |                   0.616 |                  0.624 |                  0.632 |                  0.616 |             0.7307 |                 0.7305 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1.2 | sgd_log           | alpha=0.05    |                           1 |                           1 |                 1.6046 |                    0.6912 |                   0.62  |                  0.624 |                  0.62  |                  0.632 |             0.7236 |                 0.7234 |               0.6533 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1.15 |                     1.4 | sgd_log           | alpha=0.03    |                           1 |                           1 |                 1.6038 |                    0.6904 |                   0.62  |                  0.62  |                  0.62  |                  0.66  |             0.7156 |                 0.7154 |               0.6382 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1.2 | sgd_log           | alpha=0.03    |                           1 |                           1 |                 1.603  |                    0.6904 |                   0.62  |                  0.62  |                  0.62  |                  0.632 |             0.7176 |                 0.7174 |               0.6382 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.03    |                       1 |                    1.15 |                     1.4 | sgd_log           | alpha=0.1     |                           1 |                           1 |                 1.6024 |                    0.6912 |                   0.616 |                  0.616 |                  0.624 |                  0.66  |             0.7276 |                 0.7275 |               0.6583 |
| cost_sensitive_p1p2_boundary | sgd_log           | alpha=0.05    |                       1 |                    1    |                     1   | sgd_log           | alpha=0.05    |                           1 |                           1 |                 1.6023 |                    0.6928 |                   0.616 |                  0.628 |                  0.62  |                  0.616 |             0.7256 |                 0.7254 |               0.6533 |

## Confusion Matrix

Rows are true P1-P5; columns are predicted P1-P5.

```text
[[131  51   9   5   3]
 [ 28 136  14  14   6]
 [  3   0 167  21   9]
 [  7  16  17 142  17]
 [  2   7  14  29 147]]
```

## Interpretation

- This model should be compared with the current best model before replacing it.
- If macro recall or minimum recall improves but accuracy drops, keep it as an improvement experiment rather than the main model.
