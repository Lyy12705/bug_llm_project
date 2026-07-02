# Duplicate Ticket Experiments

## Small verified samples

The repository includes commands that were verified on small public Bugzilla samples:

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugzilla.mozilla.org/rest \
  --product Firefox \
  --resolution DUPLICATE \
  --limit 20 \
  --with-comments \
  --include-duplicate-masters \
  --output data/mozilla_firefox_duplicates_sample.csv
```

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugs.eclipse.org/bugs/rest \
  --product Platform \
  --resolution DUPLICATE \
  --limit 20 \
  --with-comments \
  --include-duplicate-masters \
  --output data/eclipse_platform_duplicates_sample.csv
```

Each sample produced 38 tickets: 20 duplicate reports plus 18 master reports.

## Verified TF-IDF 5-fold results

Mozilla Firefox sample:

```text
MAP=0.7761
top_1_accuracy=0.6897
top_10_hit_rate=0.9528
MRR=0.7866
```

Eclipse Platform sample:

```text
MAP=0.8063
top_1_accuracy=0.6921
top_10_hit_rate=0.9464
MRR=0.8044
```

## Verified SBERT 5-fold result

Mozilla Firefox sample, `epochs=1`, `max_triplets=200`:

```text
MAP=0.7850
top_1_accuracy=0.7433
top_10_hit_rate=0.9206
MRR=0.8067
```

These samples are intentionally small smoke tests. For a report-quality experiment, increase `--limit`, keep `--include-duplicate-masters`, and report both TF-IDF and SBERT 5-fold results.
