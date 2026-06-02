# swift-Gotenx Documentation

Technical documentation for swift-Gotenx, a Swift implementation of tokamak core transport simulation.

## 📚 Documentation Overview

This directory contains the technical design and architecture documentation for swift-Gotenx. For quick-start development guidance, see [CLAUDE.md](../CLAUDE.md) in the project root.

---

## Core Architecture

### System Design

**[ARCHITECTURE.md](ARCHITECTURE.md)**
- TORAX core concepts (static vs dynamic parameters, FVM architecture)
- Swift design patterns and MLX integration
- Future extensions roadmap (sensitivity analysis, time-dependent geometry)

**[UNIT_SYSTEM.md](UNIT_SYSTEM.md)**
- SI-based unit standard: Temperature (eV), Density (m⁻³), Power (MW/m³)
- Unit conversion guidelines and display conventions
- Critical: Prevents 1000× errors in calculations

### Configuration System

**[CONFIGURATION_SYSTEM.md](CONFIGURATION_SYSTEM.md)**
- Hierarchical configuration: CLI > Environment > JSON > Defaults
- Swift Configuration integration patterns
- Usage examples and override mechanisms

**[CONFIGURATION_ARCHITECTURE_REFACTORING.md](CONFIGURATION_ARCHITECTURE_REFACTORING.md)**
- Design document for configuration system refactoring
- CFL-aware default value computation
- Separation of concerns (loading vs validation)
- Status: Design phase under review

**[CONFIGURATION_VALIDATION_SPEC.md](CONFIGURATION_VALIDATION_SPEC.md)**
- Pre-simulation validation specification
- Stability checks (CFL, ECRH, source terms)
- Physical range validation
- Error types and actionable feedback

**[QUALITY_GATES.md](QUALITY_GATES.md)**
- Repository-level reliability gates
- xcodebuild verification command for critical suites
- API naming, configuration, and numerical diagnostic guardrails

---

## Numerical Computing

### MLX Framework

**[MLX_BEST_PRACTICES.md](MLX_BEST_PRACTICES.md)**
- Lazy evaluation and `eval()` patterns
- MLXArray initialization methods (critical guide)
- JIT compilation and optimization strategies
- Common pitfalls and solutions

**[NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md)**
- Float32-only policy (Apple Silicon GPU constraint)
- Numerical stability techniques
- Variable scaling, preconditioning, conservation enforcement
- Time accumulation strategy

**[SWIFT_CONCURRENCY.md](SWIFT_CONCURRENCY.md)**
- `EvaluatedArray` wrapper for Sendable compliance
- Actor isolation patterns
- `compile()` best practices (pure functions, no self capture)
- Swift 6 strict concurrency requirements

**[NUMERICAL_ROBUSTNESS_DESIGN.md](NUMERICAL_ROBUSTNESS_DESIGN.md)**
- NaN propagation prevention (IonElectronExchange crash fix)
- `ValidatedProfiles` and numerical validation contracts
- Constrained line search for Newton-Raphson solver
- Input validation and fail-fast source diagnostics

---

## Physics Models

**[TRANSPORT_MODELS.md](TRANSPORT_MODELS.md)**
- Constant transport model (testing/debugging)
- Bohm-GyroBohm empirical model
- QLKNN neural network model (high-fidelity)
- Performance characteristics and use cases

**[FVM_NUMERICAL_IMPROVEMENTS_PLAN.md](FVM_NUMERICAL_IMPROVEMENTS_PLAN.md)**
- Power-law convection scheme (Patankar)
- Sauter bootstrap current formula
- Non-uniform grid support
- Configurable per-equation tolerances
- Status: Design complete, ready for implementation

---

## Document Usage Guide

### For Developers

**Starting a new feature?**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns

**Working with MLX?**
→ Check [MLX_BEST_PRACTICES.md](MLX_BEST_PRACTICES.md) for initialization and evaluation patterns

**Handling configuration?**
→ See [CONFIGURATION_SYSTEM.md](CONFIGURATION_SYSTEM.md) for hierarchical config patterns

**Improving FVM numerics?** 🔥
→ [FVM_NUMERICAL_IMPROVEMENTS_PLAN.md](FVM_NUMERICAL_IMPROVEMENTS_PLAN.md) has detailed implementation plan

**Debugging numerical issues?**
→ [NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md) explains Float32 constraints and stability techniques

**Encountering NaN/Inf crashes?** 🔥
→ [NUMERICAL_ROBUSTNESS_DESIGN.md](NUMERICAL_ROBUSTNESS_DESIGN.md) addresses crash prevention and validation

### For Code Review

**Checking concurrency?**
→ [SWIFT_CONCURRENCY.md](SWIFT_CONCURRENCY.md) defines Sendable patterns and actor isolation

**Validating physics?**
→ [TRANSPORT_MODELS.md](TRANSPORT_MODELS.md) documents model implementations

**Reviewing configuration changes?**
→ [CONFIGURATION_VALIDATION_SPEC.md](CONFIGURATION_VALIDATION_SPEC.md) specifies validation rules

**Checking reliability gates?**
→ [QUALITY_GATES.md](QUALITY_GATES.md) maps risk areas to required test coverage

---

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| ARCHITECTURE.md | ✅ Active reference | 2025-10-21 |
| CONFIGURATION_SYSTEM.md | ✅ Active reference | 2025-10-21 |
| CONFIGURATION_ARCHITECTURE_REFACTORING.md | 🚧 Design phase | 2025-10-25 |
| CONFIGURATION_VALIDATION_SPEC.md | 📋 Specification | 2025-10-24 |
| QUALITY_GATES.md | ✅ Active reference | 2026-06-02 |
| FVM_NUMERICAL_IMPROVEMENTS_PLAN.md | 🔥 Ready for impl | 2025-10-22 |
| NUMERICAL_ROBUSTNESS_DESIGN.md | 🔥 Implementation ready | 2025-10-25 |
| MLX_BEST_PRACTICES.md | ✅ Active reference | 2025-10-21 |
| NUMERICAL_PRECISION.md | ✅ Active reference | 2025-10-21 |
| SWIFT_CONCURRENCY.md | ✅ Active reference | 2025-10-21 |
| TRANSPORT_MODELS.md | ✅ Active reference | 2025-10-21 |
| UNIT_SYSTEM.md | ✅ Active reference | 2025-10-21 |

**Legend:**
- ✅ Active reference: Current implementation guide
- 🔥 Ready for impl: Design complete, awaiting implementation
- 🚧 Design phase: Under review or development
- 📋 Specification: Requirements document

---

## Contributing to Documentation

When adding or updating documentation:

1. **Place in appropriate category**: Architecture, Numerical Computing, or Physics Models
2. **Update this index**: Add link with clear description
3. **Cross-reference**: Link related documents
4. **Update status**: Mark as active, design, or specification
5. **Reference from CLAUDE.md**: If relevant for active development

### Documentation Standards

- **Be concise**: Extract details from CLAUDE.md to focused docs
- **Include examples**: Show concrete usage patterns
- **Explain "why"**: Not just "how" but reasoning behind decisions
- **Mark critical constraints**: Use ⚠️ for important warnings
- **Keep updated**: Remove outdated information, archive historical docs

---

## Related Resources

- **[CLAUDE.md](../CLAUDE.md)**: Quick development guide for Claude Code
- **[README.md](../README.md)**: Project overview and getting started
- **[Examples/Configurations/](../Examples/Configurations/)**: Configuration file examples

---

*Last updated: 2026-06-02*
