# Candidate LTR Experiment

- Deployment status: `research_only`
- Validation candidate: `False`
- Selected model: `hist_gradient_boosting_shallow`
- Embedding backend: `sbert_local:sentence-transformers/all-MiniLM-L6-v2`
- Promotion gate passed: `False`

## Validation

| Metric | V2 base | Logistic LTR | Delta |
|---|---:|---:|---:|
| Top-1 | 0.600756 | 0.641058 | 0.040302 |
| MRR | 0.709695 | 0.740881 | 0.031186 |
| Macro-F1 | 0.356798 | 0.406468 | 0.049669 |
| Candidate recall | - | 0.960957 | - |

## Existing Test (Exploratory Only)

The repository's prior frozen test has already been inspected, so these values are not a new final estimate.

- Top-1: `0.555841`
- Hit@3: `0.774069`
- Hit@5: `0.839538`
- MRR: `0.679848`
- Macro-F1: `0.340105`
