# Continuous integration

The source-controlled workflow template is `bug-tracking-llm.yml`. In this
monorepo it must also exist at repository root as:

```text
.github/workflows/bug-tracking-llm.yml
```

The workflow verifies Python 3.10 and 3.13, installed-package imports, the CLI,
compile/lint checks, the full unittest suite, coverage reporting, and integrated
duplicate/priority smoke metrics. Coverage is enforced at the threshold in
`pyproject.toml` (currently 70%).
