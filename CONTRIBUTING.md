# Contributing to OpenRoutiQ

Thank you for helping make model routing more open, reproducible, and useful.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,proxy]"
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pip_audit
python scripts/check_release.py
python scripts/check_docs.py
python -m build
```

The core package must remain dependency-free. Put provider, proxy, embedding, and evaluation integrations behind optional extras and guarded imports.

## Pull requests

- Open an issue first for breaking APIs, new scoring semantics, or new persisted schemas.
- Add focused tests for behavior and failure modes.
- Update the README when user-visible behavior changes.
- Preserve provider-native message/tool objects; do not normalize away opaque state.
- Never include real API keys, private prompts, proprietary datasets, or paid model outputs.
- Keep example model data clearly labeled synthetic or illustrative.

## Commit and review expectations

Small, single-purpose commits are easiest to review. Maintainers may request changes for correctness, security, privacy, compatibility, or reproducibility. By contributing, you agree that your contribution is licensed under the repository's MIT License.

## Security reports

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability
reporting for this repository and omit real provider keys, private prompts, and customer data.
