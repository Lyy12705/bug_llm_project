# Description Missing Root-Cause Verification

## Root Cause

- The original BMO 3k raw file has no nonempty `description` values.
- The fetch script stores first comments only when comment fetching is enabled.
- The prepare script maps `description` from `row.get("description")`, so missing raw descriptions propagate to processed rows.
- This enriched version preserves original ticket fields and adds `description_source = first_comment`.

## Enriched Dataset Audit

- Total rows: `3000`
- Nonempty descriptions: `2683` (0.894333)
- Mean / median description length: `1283.548` / `493.0`
- Mean / median token length: `186.091` / `75.5`
- HTML content ratio: `0.258667`
- Code/log presence ratio: `0.283`
- Fetch errors: `10`

First public comment is a reasonable operational proxy for bug description, but it can include boilerplate, logs, quotes, and code snippets.
