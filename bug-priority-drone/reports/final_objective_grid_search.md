# Final Objective Grid Search

This search selects the model by a balanced validation objective:

`accuracy + macro F1 + 0.5 * P2 recall + P1/P3 guardrails + off-by-one bonus - MAE penalty`

The objective is used only on the validation set. The reported result is evaluated on natural_holdout.

## Best Natural-Holdout Result

| eval_set        |   accuracy |   macro_f1 |   off_by_one_accuracy |   mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:----------------|-----------:|-----------:|----------------------:|------:|------------:|------------:|------------:|------------:|------------:|
| natural_holdout |     0.7236 |     0.7235 |                0.9025 | 0.409 |       0.603 |      0.7172 |       0.865 |      0.6985 |      0.7337 |

## Top Validation Candidates

| stage            | base_model_type   | base_params   | boundary_model_type   | boundary_params   | boundary_apply   |   boundary_p2_sample_weight |   validation_objective |   validation_accuracy |   validation_macro_f1 |   validation_p1_recall |   validation_p2_recall |   validation_p3_recall |   natural_accuracy |   natural_macro_f1 |   natural_p2_recall |   natural_mae |
|:-----------------|:------------------|:--------------|:----------------------|:------------------|:-----------------|----------------------------:|-----------------------:|----------------------:|----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-------------------:|--------------------:|--------------:|
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.03        | base_1_2_3       |                         1.3 |                 1.8722 |                0.696  |                0.6949 |                  0.576 |                  0.704 |                  0.88  |             0.7236 |             0.7235 |              0.7172 |        0.409  |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.03        | base_1_2         |                         1.3 |                 1.8711 |                0.6976 |                0.6956 |                  0.576 |                  0.692 |                  0.9   |             0.7256 |             0.7248 |              0.7071 |        0.407  |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.03        | base_1_2_3       |                         1.4 |                 1.8699 |                0.696  |                0.6948 |                  0.564 |                  0.72  |                  0.876 |             0.7236 |             0.7236 |              0.7323 |        0.4101 |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.1         | base_1_2         |                         1.4 |                 1.8682 |                0.6952 |                0.6934 |                  0.6   |                  0.66  |                  0.896 |             0.7307 |             0.7302 |              0.702  |        0.404  |
| boundary_refined | sgd_log           | alpha=0.1     | sgd_log               | alpha=0.03        | base_1_2_3       |                         1.3 |                 1.8671 |                0.6944 |                0.6933 |                  0.576 |                  0.704 |                  0.872 |             0.7206 |             0.7204 |              0.7172 |        0.4101 |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.1         | base_1_2         |                         1.2 |                 1.866  |                0.696  |                0.6941 |                  0.612 |                  0.648 |                  0.9   |             0.7307 |             0.7302 |              0.6869 |        0.404  |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.03        | base_1_2         |                         1.4 |                 1.865  |                0.6968 |                0.6948 |                  0.564 |                  0.704 |                  0.896 |             0.7276 |             0.7267 |              0.7222 |        0.405  |
| boundary_refined | sgd_log           | alpha=0.1     | sgd_log               | alpha=0.03        | base_1_2_3       |                         1.4 |                 1.8647 |                0.6944 |                0.6932 |                  0.564 |                  0.72  |                  0.868 |             0.7206 |             0.7206 |              0.7323 |        0.4111 |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.1         | base_1_2         |                         1.3 |                 1.8645 |                0.6944 |                0.6926 |                  0.6   |                  0.656 |                  0.896 |             0.7317 |             0.7312 |              0.697  |        0.403  |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.1         | base_1_2         |                         1.1 |                 1.8644 |                0.696  |                0.6941 |                  0.616 |                  0.644 |                  0.9   |             0.7296 |             0.7291 |              0.6869 |        0.406  |
| boundary_refined | sgd_log           | alpha=0.05    | sgd_log               | alpha=0.1         | base_1_2_3       |                         1.2 |                 1.8642 |                0.6952 |                0.6935 |                  0.612 |                  0.648 |                  0.896 |             0.7296 |             0.7294 |              0.6869 |        0.404  |
| boundary_refined | sgd_log           | alpha=0.1     | sgd_log               | alpha=0.1         | base_1_2         |                         1.4 |                 1.864  |                0.6944 |                0.6923 |                  0.6   |                  0.656 |                  0.896 |             0.7246 |             0.724  |              0.697  |        0.409  |

## Interpretation

- If this best result is not higher than the current saved best model, keep the current best model and treat this as a negative tuning result.
- A validation objective can improve robustness, but it cannot guarantee a better natural_holdout score because the holdout remains unseen during selection.
