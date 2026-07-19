# Fault Localization Demo Repository

This small repository is used for the stage-5 fault localization demo.

Demo ticket `FL-DEMO-HIGH` reports a login crash when an auth token is missing.
The intended gold file is `src/auth/validator.py`.

The JSONL demo set also includes:

- a medium-confidence password-reset case,
- a low-confidence manual-review case,
- a path-hint / identifier case for `src/auth/settings.py`,
- an invalid-input case that demonstrates the user-facing validation message.

These cases are intentionally small so the presentation can show confidence
gates, manual review, and patch handoff limits without depending on Ollama.
