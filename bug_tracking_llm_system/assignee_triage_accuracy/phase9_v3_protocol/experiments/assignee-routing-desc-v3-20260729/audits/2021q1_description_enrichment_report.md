# Description Missing Root-Cause Verification

## Root Cause

- The original BMO 3k raw file has no nonempty `description` values.
- The fetch script stores first comments only when comment fetching is enabled.
- The prepare script maps `description` from `row.get("description")`, so missing raw descriptions propagate to processed rows.
- This enriched version preserves original ticket fields and adds `description_source = first_comment`.

## Enriched Dataset Audit

- Total rows: `3000`
- Nonempty descriptions: `2726` (0.908667)
- Mean / median description length: `1228.082` / `448.0`
- Mean / median token length: `172.806` / `69.0`
- HTML content ratio: `0.238`
- Code/log presence ratio: `0.298667`
- Fetch errors: `0`

First public comment is a reasonable operational proxy for bug description, but it can include boilerplate, logs, quotes, and code snippets.
