## What changed

Describe the user-visible behavior and motivation.

## Verification

- [ ] Tests cover the new behavior and failure modes.
- [ ] `python -m ruff check .` passes.
- [ ] `python -m ruff format --check .` passes.
- [ ] `python -m mypy` passes.
- [ ] `python scripts/check_release.py` and `python scripts/check_docs.py` pass.
- [ ] `python scripts/check_namespace_cutover.py` passes.
- [ ] Built wheel and source archive pass `python scripts/check_distribution.py`.
- [ ] The README is updated when user-visible behavior changes.
- [ ] No credentials, private prompts, unlicensed data, or generated secrets are included.
