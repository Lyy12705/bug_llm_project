# Literature Method Notes

Reference: Haruna Isotani et al., "Sentence embedding and fine-tuning to automatically identify duplicate bugs", Frontiers in Computer Science, 2023.

## What the paper does

- Input: a new inquiry/failure report with a title and content.
- Output: previous reports ranked by duplicate likelihood.
- Embedding: Sentence-BERT.
- Training: triplet network with anchor, positive duplicate, and negative non-duplicate.
- Models: one SBERT model for titles and one SBERT model for contents.
- Distance: cosine distance, equivalent to `1 - cosine_similarity`.
- Report similarity candidates:
  - `1.0 * title_similarity`
  - `1.0 * content_similarity`
  - `0.5 * title_similarity + 0.5 * content_similarity`
  - `0.75 * title_similarity + 0.25 * content_similarity`
  - `0.25 * title_similarity + 0.75 * content_similarity`
  - `max(title_similarity, content_similarity)`
  - `min(title_similarity, content_similarity)`
- Evaluation: rank previous reports and compute MAP.

## Implementation mapping in this repo

- `dataset.py`: normalize CSV/JSONL fields and derive duplicate groups.
- `triplets.py`: build anchor/positive/negative examples from duplicate groups.
- `tfidf_detector.py`: runnable baseline ranking detector.
- `sbert_detector.py`: optional SBERT triplet fine-tuning detector.
- `decision.py`: converts ranking scores into a binary duplicate decision by tuning a validation threshold.
- `metrics.py`: AP and MAP ranking metrics.
- `cli.py`: experiment commands.

## System decision layer

The paper primarily evaluates duplicate detection as an information-retrieval ranking problem: a new report is compared with previous reports, previous reports are sorted by similarity, and MAP measures whether true duplicates appear near the top.

For the student-project system, the ranking step is kept as the core literature method, then a thin decision layer is added:

1. Split labeled historical tickets into train and validation sets.
2. Fit the detector on train tickets.
3. Score validation query-candidate pairs.
4. Select the similarity threshold with the best F1 score.
5. For a new ticket, rank historical tickets and mark it as duplicate when the highest similarity is at least the tuned threshold.

This keeps the implementation faithful to the paper while producing the `Duplicate` / `New ticket` decision needed by the planned workflow.

## Dataset note

The paper used an internal NTT dataset, so it cannot be reproduced exactly unless that data is available. For a public student-project dataset, use Bugzilla-derived data such as Mozilla or Eclipse and keep the duplicate relation (`dupe_of` / `duplicate_of`) as the label source.
