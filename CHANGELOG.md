# Changelog

All notable changes are documented here. The project follows Semantic Versioning after `1.0`;
before `1.0`, minor releases may contain breaking changes with migration notes.

## 0.1.0 - 2026-08-29

Initial public release of OpenRoutiQ. Earlier version numbers were internal development snapshots
and were never published to PyPI.

### Added

- Opt-in risk-aware routing with joint contextual outcome scenarios, chance constraints,
  latency/cost quantiles, evaluated success and failure probabilities, CVaR scoring, and
  propensity-aware exploration telemetry.
- Capability-gated request/model selection intelligence with calibrated success, quality, cost,
  and latency components plus prompt-free persisted hashed state.
- Normalized routing/model/provider/capability/tool/protocol/timeout/rate-limit failure taxonomy
  and proxy request, concurrency, queue, provider, and stream-idle safety limits.
- README guidance covering secure topology, tenant isolation, SQLite limits, backups,
  observability, champion/challenger promotion, and rollback.
- Routing-space documentation covering exact variant cardinality, the $T\times V$
  task-variant matrix versus $V^T$ possible static policies, capability-gated eligibility, and
  request-level policy-adjusted expected-utility maximization.
- Encounter-driven `AdaptiveRouter` support for public, private, fine-tuned, on-premise, and future
  models, backed by an opt-in local SQLite registry and a pluggable storage boundary.
- Automatic proxy collection of latency, execution success, provider-reported cost, and token
  counts; evaluated quality remains explicitly supplied by the application.
- OpenAI-compatible LiteLLM proxy, JSON CLI, provider-native request preparation, custom task
  labels, multimodal capability detection, and typed-package support.
- Release automation with metadata, namespace, documentation, dependency, distribution, and
  clean-install validation across supported Python versions and operating systems.
- Public README coverage for the SDK, adaptive and selection intelligence, proxy, risk policy,
  benchmarks, release scope, and operational boundaries.
- A packaged benchmark toolkit with an installed CLI, recorded-replay example, live-adapter
  approval gates, cost/call estimates, machine-readable results, and an HTML report.

### Changed

- Renamed the unreleased product, distribution, import package, CLI, proxy contract, repository
  links, documentation, and result presentation to OpenRoutiQ. Because no public release
  exists, this is a hard cutover with no legacy package, command, header, or deprecation alias.
- The proxy now uses endpoint-specific inference-field allowlists to reject client-controlled
  provider destinations, cloud endpoints, credentials, headers, clients, retries, timeouts, and
  unknown future LiteLLM controls. Client input-token declarations are conservative routing
  floors, and every dispatched provider request receives the routed output-token cap. Routing
  runs off the event loop inside the concurrency gate, and timed-out routing or synchronous
  provider workers retain their slot until uncancellable work exits.
- Adaptive profile refreshes now stream time-decayed quality aggregates and ask SQLite for only
  the configured recent latency/cost window instead of materializing every retained observation.
- Static catalogs and per-call provider options now reject credential-like fields and credentials
  embedded in endpoint URLs. Provider `extra` parameters cannot override routed model, token,
  reasoning, tool, stream, metadata, or provider contracts.
- `OutcomeStore` lookups validate hard row bounds, filter eligible model IDs in SQL, use a matching
  recent-lookup index, and create the SQLite file with owner-only permissions on POSIX systems.
- GitHub Actions are pinned to immutable upstream commits; release CI installs through reviewed
  direct dependency constraints; and the current provider/proxy release environment is audited
  in a fresh isolated environment.
- README branding and public result plots now render OpenRoutiQ consistently from repository-local
  image assets suitable for GitHub and package-index presentation.
- Release metadata now declares the stable library/internal-proxy scope, and the development
  dependency contract is exercised by the clean CI test suite.
- Automatic adaptive-model promotion is disabled by default; operators may opt in with
  workload-validated thresholds.
- Proxy authentication fails closed for an explicitly empty token, validates streaming flags,
  runs synchronous stream iterators off the ASGI event loop, and closes streams after completion
  or disconnect.
- Private evaluation data and generated work products are excluded from release artifacts.
- Async provider invocation moves synchronous clients off the event loop.
- Public proxy binds now require bearer-token configuration in the CLI.
- Release builds rerun lint/tests, validate distributions, and smoke-test the wheel.
- Source distributions exclude private evaluation work products and Python cache files.
- Runtime, adaptive, selection, provider, proxy, quick-start, and benchmark code now live in
  explicit domain packages and use absolute `openroutiq.*` imports throughout.
