# Security policy

## Supported versions

Security fixes are provided for the latest released minor version. Pre-1.0 releases may include breaking changes in a new minor version.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue for suspected credential exposure, authentication bypass, SSRF, prompt/telemetry disclosure, dependency compromise, or remote code execution.

Include the affected version/commit, impact, reproduction steps, and any suggested mitigation. Maintainers will acknowledge a complete report as soon as practical and coordinate disclosure after a fix is available.

## Security boundaries

- The local routing core makes no network call by default.
- Provider clients, custom adapters, embedders, evaluation commands, catalogs, and proxy deployments are operator-controlled trust boundaries.
- Never commit provider keys. Provider and proxy integrations read named environment variables.
- Treat prompts, tool payloads, embeddings, agent metadata, evaluation inputs, and model outputs as potentially sensitive.
- Treat persisted selection/adaptive state as sensitive operational telemetry even though the
  built-in selection state contains only hashed feature weights and numeric aggregates; hashing
  is not a formal anonymization guarantee.
- Do not expose the optional proxy publicly without authentication, TLS, egress restrictions, request/rate limits, and reviewed logging.
- Built-in proxy request, concurrency, queue, provider, and stream-idle limits are process-local
  safeguards. Internet-facing deployments still require tenant-aware edge quotas and distributed
  abuse protection.
- A custom `base_url`, command adapter, or provider adapter can send data to its configured destination. Accept these only from trusted configuration.

## Dependency and release security

The core has no runtime dependencies. CI audits the provider/proxy release environment,
Dependabot reviews Python and GitHub Actions manifests. Heavy embedding and external integration
environments remain operator-selected and must be audited when enabled. Releases are built by GitHub Actions with PyPI trusted
publishing and attestations. Verify package hashes and provenance for sensitive deployments.
