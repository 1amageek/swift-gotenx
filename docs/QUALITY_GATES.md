# Quality Gates

This document defines the repository-level checks that keep swift-Gotenx at a production-ready reliability baseline. These gates target the current maturity goal of 80-85% across implementation foundation, configuration validation, API naming, and numerical reliability.

## Gate Matrix

| Area | Guardrail | Test Coverage |
|---|---|---|
| Implementation foundation | Source code must not retain deprecated public API aliases, swallow errors with `try?`, or force-unwrap throwing calls with `try!`. | `RepositoryQualityGateTests` |
| Configuration and input validation | Transport parameters must use the final Swift API names and unknown keys must be rejected from JSON, CLI, and environment sources. | `GotenxConfigReaderTests`, `TransportParameterValidationTests`, `ConfigurationPriorityTests`, `ConfigurationLoaderTests` |
| API naming consistency | Obsolete short names must not remain in source declarations or implementation files. | `RepositoryQualityGateTests` |
| Numerical reliability | Energy diagnostics must preserve source metadata accounting, reject missing or non-finite metadata, preserve W-to-MW conversion, enforce `tau_E = W_thermal / P_loss`, and reject invalid profiles, sources, transport coefficients, and block coefficients at simulation step boundaries. | `DerivedQuantitiesComputerTests`, `SimulationRunnerEnergyDiagnosticsExperimentTests`, `NumericalValidationTests`, `Block1DCoeffsBuilderTests` |

## Required Local Verification

Use `xcodebuild` for verification. Do not use `swift test` for this repository.

```bash
perl -e 'alarm 120; exec @ARGV' xcodebuild test \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -only-testing:GotenxTests/RepositoryQualityGateTests \
  -only-testing:GotenxTests/GotenxConfigReaderTests \
  -only-testing:GotenxTests/TransportParameterValidationTests \
  -only-testing:GotenxTests/ConfigurationPriorityTests \
	  -only-testing:GotenxTests/ConfigurationLoaderTests \
	  -only-testing:GotenxTests/DerivedQuantitiesComputerTests \
	  -only-testing:GotenxTests/SimulationRunnerEnergyDiagnosticsExperimentTests \
	  -only-testing:GotenxTests/NumericalValidationTests \
	  -only-testing:GotenxTests/Block1DCoeffsBuilderTests
```

## Review Policy

Changes that affect public API, configuration keys, source metadata, derived quantities, solver convergence, or transport parameter handling must either pass an existing gate or add a targeted gate before merging.

Power diagnostics are metadata-only. Fixed-ratio or profile-estimated power accounting is not an accepted path for production diagnostics. Diagnostic source computations are throwing; computation failures must propagate instead of being represented as zero source terms.

Compatibility aliases are not part of the current product strategy. Prefer removing stale names and documenting the final name rather than retaining deprecated entry points.
