# Assignee Responsibility Profiles: controlled

> These profiles are inferred from historical ticket ownership only. They are not verified job titles.

- History path: `/Users/linyaying/Documents/bug_llm_project/bug_tracking_llm_system/assignee_triage_accuracy/data/controlled_history_train.jsonl`
- Source rows: `18`
- Profile count: `7`

## Summary

| Assignee | Tickets | Confidence | Top Components | Top Keywords |
|---|---:|---|---|---|
| `backend-team@example.com` | 5 | medium | email (3); api (1); backend (1) | email, backend, queue, api, item |
| `auth-team@example.com` | 3 | low | authentication (3) | token, authentication, expired, refresh, reset |
| `payments-team@example.com` | 3 | low | payment (3) | payment, authorization, capture, payment capture, refund |
| `build-team@example.com` | 2 | low | build (1); compiler (1) | build, compiler, generated, generated header, header |
| `data-team@example.com` | 2 | low | search (2) | search, archived, records, archived migration, archived records |
| `frontend-team@example.com` | 2 | low | frontend (1); ui (1) | button, checkout, checkout button, label, modal |
| `security-team@example.com` | 1 | low | security (1) | security, attributes, attributes security, cookie, cookie settings |

## Details

### `backend-team@example.com`

Inferred from 5 historical tickets: likely focuses on email, api, backend. Common work themes: email, backend, queue, api, item.

- Ticket count: `5`
- Confidence: `medium`
- Top products: n/a
- Top components: email (3); api (1); backend (1)
- Top keywords: email, backend, queue, api, item, queue item, retries, welcome
- Representative tickets:
  - `HIST-API-002` [backend] Backend job double processes queue item
  - `HIST-EMAIL-002` [email] Email worker retries forever
  - `HIST-API-001` [api] Orders endpoint returns 500
  - `HIST-EMAIL-001` [email] Password reset email is not sent
  - `HIST-EMAIL-003` [email] Welcome email renders missing link

### `auth-team@example.com`

Inferred from 3 historical tickets: likely focuses on authentication. Common work themes: token, authentication, expired, refresh, reset.

- Ticket count: `3`
- Confidence: `low`
- Top products: n/a
- Top components: authentication (3)
- Top keywords: token, authentication, expired, refresh, reset, returns, fails, auth
- Representative tickets:
  - `HIST-AUTH-002` [authentication] Session refresh returns invalid token
  - `HIST-AUTH-001` [authentication] Login fails when token is expired
  - `HIST-AUTH-003` [authentication] Password reset token cannot be verified

### `payments-team@example.com`

Inferred from 3 historical tickets: likely focuses on payment. Common work themes: payment, authorization, capture, payment capture, refund.

- Ticket count: `3`
- Confidence: `low`
- Top products: n/a
- Top components: payment (3)
- Top keywords: payment, authorization, capture, payment capture, refund, refund webhook, webhook, gateway
- Representative tickets:
  - `HIST-PAY-003` [payment] Card authorization status is stale
  - `HIST-PAY-001` [payment] Payment capture fails after retry
  - `HIST-PAY-002` [payment] Refund webhook is ignored

### `build-team@example.com`

Inferred from 2 historical tickets: likely focuses on build, compiler. Common work themes: build, compiler, generated, generated header, header.

- Ticket count: `2`
- Confidence: `low`
- Top products: n/a
- Top components: build (1); compiler (1)
- Top keywords: build, compiler, generated, generated header, header, incremental, fails generated, fails
- Representative tickets:
  - `HIST-BUILD-001` [build] Build fails when generated header is missing
  - `HIST-BUILD-002` [compiler] Compiler error on incremental build

### `data-team@example.com`

Inferred from 2 historical tickets: likely focuses on search. Common work themes: search, archived, records, archived migration, archived records.

- Ticket count: `2`
- Confidence: `low`
- Top products: n/a
- Top components: search (2)
- Top keywords: search, archived, records, archived migration, archived records, database, database row, duplicate rows
- Representative tickets:
  - `HIST-SEARCH-002` [search] Search results duplicate rows
  - `HIST-SEARCH-001` [search] Search index misses archived records

### `frontend-team@example.com`

Inferred from 2 historical tickets: likely focuses on frontend, ui. Common work themes: button, checkout, checkout button, label, modal.

- Ticket count: `2`
- Confidence: `low`
- Top products: n/a
- Top components: frontend (1); ui (1)
- Top keywords: button, checkout, checkout button, label, modal, total, total label, action
- Representative tickets:
  - `HIST-FRONT-001` [frontend] Checkout button overlaps total label
  - `HIST-FRONT-002` [ui] Modal stays open after cancel

### `security-team@example.com`

Inferred from 1 historical tickets: likely focuses on security. Common work themes: security, attributes, attributes security, cookie, cookie settings.

- Ticket count: `1`
- Confidence: `low`
- Top products: n/a
- Top components: security (1)
- Top keywords: security, attributes, attributes security, cookie, cookie settings, cookies, cookies missing, flags
- Representative tickets:
  - `HIST-SEC-001` [security] Security scanner flags weak cookie settings

