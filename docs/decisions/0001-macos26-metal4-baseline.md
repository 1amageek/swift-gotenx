# macOS 26.4 and Metal 4 Baseline

Date: 2026-05-31

Status: Accepted

## Context

swift-gotenx is currently pre-user and does not need compatibility with older Apple platforms. The solver is performance-sensitive and already depends on MLX for Apple Silicon GPU execution.

Metal 4 is available through the macOS 26.4 SDK and introduces lower-overhead command encoding, more explicit compilation workflows, and native machine-learning-oriented resources. MLX owns the low-level command submission for most tensor work, so the package should make the latest Metal stack the baseline rather than carrying older OS compatibility branches.

## Decision

The package targets macOS 26.4 or newer and requires a default Metal device that supports the Metal 4 GPU family. iOS, visionOS, Linux, and older macOS deployment targets are not supported.

QLKNN is included directly because the package no longer needs a non-macOS build path.

## Consequences

- All builds use the macOS 26.4 SDK deployment baseline.
- Startup fails early if the default Metal device is missing or does not support Metal 4.
- Compatibility code for iOS, visionOS, Linux, and older macOS releases should not be added unless this decision is explicitly revisited.
- Future hot paths can use MLXFast or direct Metal 4 APIs without availability fallbacks.
