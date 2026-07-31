# Description Missing Root-Cause Verification

## Root Cause

- The original BMO 3k raw file has no nonempty `description` values.
- The fetch script stores first comments only when comment fetching is enabled.
- The prepare script maps `description` from `row.get("description")`, so missing raw descriptions propagate to processed rows.
- This enriched version preserves original ticket fields and adds `description_source = first_comment`.

## Enriched Dataset Audit

- Total rows: `3000`
- Nonempty descriptions: `2696` (0.898667)
- Mean / median description length: `1184.451` / `498.0`
- Mean / median token length: `168.71` / `75.0`
- HTML content ratio: `0.23`
- Code/log presence ratio: `0.283`
- Fetch errors: `11`

First public comment is a reasonable operational proxy for bug description, but it can include boilerplate, logs, quotes, and code snippets.
