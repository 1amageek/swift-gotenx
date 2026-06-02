# Configuration Architecture Refactoring

**Date**: 2025-10-25
**Status**: Implemented Reference
**Priority**: High

## Revision History

**2026-06-02 (Rev 3)**: Current transport key contract
- Transport parameter keys are canonical Swift `camelCase`.
- Obsolete and unknown transport keys are rejected during decode, validation, factory creation, and parameter-based model initialization.
- The validator reports missing required constant-transport parameters instead of providing fallback defaults.

**2025-10-25 (Rev 2)**: Critical design revision based on expert review
- ⚠️ **IMPORTANT**: Original design has fundamental flaws
- Context-dependent defaults placed in context-free layer (TransportConfig)
- Silent missing value fallback (0.0) hides configuration errors
- Key naming convention inconsistencies across sources
- No compile-time validation enforcement
- Missing default change propagation mechanism

**See "Critical Design Issues" section below for detailed analysis and revised approach.**

**2025-10-25 (Rev 1)**: Initial design

## Executive Summary

This document describes a comprehensive refactoring of the configuration system to address fundamental architectural issues discovered during test suite validation. The refactoring separates concerns between configuration loading, default value management, and validation, following SOLID principles and improving testability.

## Problem Statement

### Issue: Test Failures in ToraxConfigReaderTests

All 7 tests in `ToraxConfigReaderTests.swift` are failing with CFL (Courant-Friedrichs-Lewy) violation errors:

```
Caught error: cflViolation(
    parameter: "ionHeatDiffusivity",
    cfl: 10.000001,
    limit: 0.5,
    suggestion: "Reduce ionHeatDiffusivity to 0.049999997 m²/s or decrease dt to 5e-05 s"
)
```

### Root Cause Analysis

The failures reveal **three fundamental architectural problems**:

#### Problem 1: Ambiguous Default Value Ownership

**Previous Flow**:
```
TransportConfig
  parameters: [:]  (empty dictionary allowed)
       ↓
ConfigurationValidator
  validator supplied fallback heat diffusivity values
       ↓
CFL Calculation
  CFL could be computed from unintended values
```

**Issue**: Default values were embedded in the **validation layer**
- The old validator supplied fallback heat diffusivity values
- Validator's responsibility is **validation**, not **default value provisioning**
- This violates the Single Responsibility Principle

**Evidence from Code**:
```swift
try transport.validateParameterKeys()
let ionHeatDiffusivity = try transport.requireParameter("ionHeatDiffusivity")
let electronHeatDiffusivity = try transport.requireParameter("electronHeatDiffusivity")
```

#### Problem 2: Model-Dependent Defaults Not Reflected in Types

**Domain Reality**:
- `ConstantTransportModel`: Requires explicit parameters (ionHeatDiffusivity, electronHeatDiffusivity)
- `BohmGyroBohmTransportModel`: Computes parameters from plasma state (no explicit params needed)
- `QLKNNTransportModel`: Neural network computes parameters (no explicit params needed)

**Current Type System**:
```swift
struct TransportConfig {
    let modelType: TransportModelType
    let parameters: [String: Float]  // ❌ Model dependency not expressed
}
```

**Problem**: The type system doesn't distinguish between models that require parameters vs. those that compute them.

**Evidence from Production Configs**:
```json
// Examples/Configurations/minimal.json (line 36)
"transport": {
  "modelType": "constant",
  "parameters": {}  // Empty allowed but triggers validator defaults
}

// Examples/Configurations/iter_like.json (line 33-35)
"transport": {
  "modelType": "bohmGyrobohm",
  "parameters": {}  // Empty is correct for this model
}
```

#### Problem 3: Tight Coupling Between Loading and Validation

**Current Architecture**:
```swift
// GotenxConfigReader.swift:76-91
public func fetchConfiguration() async throws -> SimulationConfiguration {
    let config = SimulationConfiguration(...)

    // ❌ Validation tightly coupled to loading
    try ConfigurationValidator.validate(config)

    return config
}

// ConfigurationLoader.swift:71
try ConfigurationValidator.validate(finalConfig)  // Same issue
```

**Consequences**:
1. **Test Inflexibility**: Cannot test configuration loading without physics validation
2. **Responsibility Mixing**: Reader does both reading AND validation
3. **Test Scope Confusion**: Integration tests (ToraxConfigReaderTests) forced to satisfy physics constraints

**What Tests Actually Want to Verify**:
- ✅ JSON parsing correctness
- ✅ CLI override priority (CLI > Env > JSON > Default)
- ✅ Environment variable handling
- ❌ NOT: CFL stability, physical plausibility, numerical constraints

## Critical Design Issues (Rev 2)

⚠️ **The original design below has fundamental flaws identified during expert review.**

### Issue 1: Context-Dependent Defaults in Context-Free Layer

**Problem**:
The original design places CFL-dependent default values (0.05 m²/s) in `TransportConfig`, which has no access to `MeshConfig` or `TimeConfiguration`.

```swift
// FLAWED: TransportConfig doesn't know about mesh/time
static func hardcodedTransportDefaults(for modelType: TransportModelType) -> [String: Float] {
    case .constant:
        return ["ionHeatDiffusivity": 0.05]  // ❌ Assumes dx=0.01m, dt=1e-3s
}
```

**Consequence**:
```
User changes: cellCount: 100 → 200
→ cellSpacing: 0.01m → 0.005m
→ CFL: 0.05 × 0.001 / 0.005² = 2.0 >> 0.5  ❌ VIOLATION!
```

**Correct Approach**:
- Move default calculation to `GotenxConfigReader` (has mesh/time context)
- Compute CFL-safe defaults: `maximumHeatDiffusivity = cflLimit * dx² / dt`
- Automatically adapts to any mesh resolution

### Issue 2: Silent Missing Value Fallback

**Problem**:
```swift
// FLAWED: Silent 0.0 fallback
func parameter(_ key: String, default: Float? = nil) -> Float {
    parameters[key] ?? defaultParams[key] ?? default ?? 0.0  // ❌
}
```

**Consequence**:
- Missing required parameter → Returns 0.0 silently
- Validator cannot distinguish "intentional zero" from "missing default"
- Error messages are unclear

**Correct Approach**:
```swift
// Return Optional - caller handles missing values explicitly
func parameter(_ key: String) -> Float?

// Or throw on required parameters
func requireParameter(_ key: String) throws -> Float
```

### Issue 3: Transport Key Contract

**Problem**:
- Multiple configuration sources can provide transport parameters.
- Any alias layer can hide obsolete names and silently change physics.
- Unknown keys must fail before model creation.

**Consequence**:
Silent fallback to model defaults can run a different simulation than the user requested.

**Correct Approach**:
Use canonical `camelCase` keys and reject every model-specific unknown key:
```swift
enum ConfigurationKeys {
    static let ionHeatDiffusivity = "ionHeatDiffusivity"
    static let electronHeatDiffusivity = "electronHeatDiffusivity"
}
```

### Issue 4: No Compile-Time Validation Enforcement

**Problem**:
```swift
// ❌ Easy to forget validation
let config = try await reader.fetchConfiguration()
try await SimulationRunner(config: config).run()  // Oops! No validation
```

**Consequence**:
- Validation is optional (runtime contract only)
- More call sites = more risk of forgetting
- Missing: SimulationPresets, test utilities, interactive notebooks

**Correct Approach**:
Type-safe wrapper:
```swift
struct ValidatedConfiguration {
    private init(_ config: SimulationConfiguration)
    static func validate(_ config: SimulationConfiguration) throws -> Self
}

// API only accepts validated configs
class SimulationRunner {
    init(config: ValidatedConfiguration)  // ✅ Compile-time guarantee
}
```

### Issue 5: No Default Change Propagation

**Problem**:
```swift
@Test func testDefaults() {
    #expect(config.parameter("ionHeatDiffusivity") == 0.05)  // ❌ Hardcoded
}
```

**Consequence**:
- Changing default 0.05 → 0.1 breaks all tests
- No migration guide for users
- No deprecation warnings

**Correct Approach**:
- Versioned defaults with changelog
- Deprecation detection system
- Migration documentation

## Design Principles (Revised)

### 1. Single Responsibility Principle (SRP)

Each component should have **one reason to change**:

| Component | Single Responsibility | Has Context? |
|-----------|----------------------|--------------|
| `TransportConfig` | Define domain model, parameter storage | ❌ No mesh/time |
| `GotenxConfigReader` | Read + merge sources, compute context-aware defaults | ✅ Has mesh/time |
| `ConfigurationValidator` | Validate physics and numerical constraints | ✅ Has full config |
| `ValidatedConfiguration` | Enforce compile-time validation guarantee | N/A (wrapper) |
| `TransportModel` | Compute transport coefficients from plasma state | ✅ Has plasma state |

### 2. Context-Aware Default Calculation

**Revised Principle**: Defaults that depend on context (mesh/time) must be calculated **in a layer that has access to that context**.

```
❌ WRONG: TransportConfig.hardcodedTransportDefaults()
   → No access to mesh/time
   → Fixed values become invalid when mesh changes

✅ CORRECT: GotenxConfigReader.computeCFLSafeDefaults(mesh, time)
   → Has mesh/time context
   → Automatically adapts to any configuration
```

### 3. Explicit Over Implicit

- Validation should be **explicitly invoked** by the caller
- Missing values should be **explicit** (Optional/throwing) not **silent** (0.0)
- Default value application should be **visible** in the code
- Dependencies should be **injected**, not hidden

### 4. Fail-Fast for Configuration Errors

**Revised Principle**: Prefer compile-time guarantees over runtime contracts.

```swift
// ❌ Runtime contract (easy to forget)
let config = try await reader.fetchConfiguration()
try ConfigurationValidator.validate(config)  // Can be forgotten

// ✅ Compile-time guarantee (enforced by type system)
let validated = try ValidatedConfiguration.validate(config)
let runner = SimulationRunner(config: validated)  // Only accepts validated
```

### 5. Separation of Concerns (Revised)

```
┌─────────────────────────────────────────────┐
│   TransportConfig (Domain Model)            │
│   - Model type, parameter storage           │
│   - NO defaults (context-free)              │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│   GotenxConfigReader (Reading + Defaults)   │
│   - Read from JSON/CLI/Env                  │
│   - Compute CFL-aware defaults (w/ context) │
│   - Merge explicit values                   │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│   ValidatedConfiguration (Type Wrapper)     │
│   - Runs validation (throws on error)      │
│   - Provides compile-time guarantee         │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│   SimulationRunner (Execution)              │
│   - Accepts ONLY ValidatedConfiguration     │
│   - No validation responsibility            │
└─────────────────────────────────────────────┘
```

## Proposed Architecture (Revised)

### Phase 1: Remove Context-Dependent Defaults from TransportConfig

**Location**: `Sources/GotenxCore/Configuration/TransportConfig.swift`

**⚠️ IMPORTANT**: TransportConfig should NOT contain CFL-dependent defaults because it has no access to mesh/time context.

```swift
extension TransportConfig {
    /// Get parameter value (returns nil if missing)
    ///
    /// Use this when you need to handle missing values explicitly.
    ///
    /// - Parameter key: Parameter key
    /// - Returns: Parameter value or nil if not found
    public func parameter(_ key: String) -> Float? {
        parameters[key]
    }

    /// Get required parameter (throws if missing)
    ///
    /// Use this for parameters that are mandatory for the model.
    ///
    /// - Parameter key: Parameter key
    /// - Returns: Parameter value
    /// - Throws: ConfigurationError.missingRequired if parameter not found
    public func requireParameter(_ key: String) throws -> Float {
        guard let value = parameters[key] else {
            throw ConfigurationError.missingRequired(
                key: "transport.parameters.\(key) for model \(modelType)"
            )
        }
        return value
    }

    /// Get parameter with explicit default
    ///
    /// Use this when you have a context-independent fallback value.
    ///
    /// - Parameters:
    ///   - key: Parameter key
    ///   - defaultValue: Fallback value
    /// - Returns: Parameter value or default
    public func parameter(_ key: String, default defaultValue: Float) -> Float {
        parameters[key] ?? defaultValue
    }
}
```

**Rationale**:
- ✅ No context-dependent defaults in context-free layer
- ✅ Missing values return `nil` (explicit)
- ✅ Caller chooses error handling strategy (Optional, throwing, or explicit default)
- ✅ Distinguishes "missing" from "zero"

### Phase 2: CFL-Aware Defaults in Reader (REVISED)

**Location**: `Sources/GotenxCLI/Configuration/GotenxConfigReader.swift`

**Before**:
```swift
private func fetchTransportConfig() async throws -> TransportConfig {
    let modelType = try await fetchEnum(...)

    var parameters: [String: Float] = [:]

    // Only loads explicit values (no defaults)
    let ionHeatDiffusivity = try await configReader.fetchDouble(forKey: "...")
    parameters["ionHeatDiffusivity"] = Float(ionHeatDiffusivity)

    return try TransportConfig(modelType: modelType, parameters: parameters)
}
```

**After**:
```swift
private func fetchTransportConfig() async throws -> TransportConfig {
    let modelType = try await fetchEnum(
        forKey: "runtime.dynamic.transport.modelType",
        default: TransportModelType.constant
    )

    // ✅ Compute CFL-safe defaults WITH context
    let mesh = try await fetchMeshConfig()  // Already being fetched
    let time = try await fetchTimeConfig()  // Already being fetched
    let safeDefaults = computeCFLSafeDefaults(
        modelType: modelType,
        mesh: mesh,
        time: time
    )

    // Start with computed defaults
    var parameters = safeDefaults

    // Override with explicit JSON values (unified key handling)
    for (key, jsonKey) in Self.transportParameterKeys {
        let value = try await configReader.fetchDouble(forKey: jsonKey)
        parameters[key] = Float(value)
    }

    return try TransportConfig(modelType: modelType, parameters: parameters)
}

/// Key mapping: internal -> JSON
private static let transportParameterKeys: [String: String] = [
    "ionHeatDiffusivity": "runtime.dynamic.transport.parameters.ionHeatDiffusivity",
    "electronHeatDiffusivity": "runtime.dynamic.transport.parameters.electronHeatDiffusivity",
    "particleDiffusivity": "runtime.dynamic.transport.parameters.particleDiffusivity"
]

/// Compute CFL-safe transport parameter defaults
///
/// Design constraint: CFL = χ * Δt / Δx² < cflLimit
///   => χ_max = cflLimit * Δx² / Δt
///
/// - Parameters:
///   - modelType: Transport model type
///   - mesh: Mesh configuration (for Δx)
///   - time: Time configuration (for Δt)
///   - cflLimit: CFL stability limit (default: 0.5)
/// - Returns: CFL-safe default parameters
private func computeCFLSafeDefaults(
    modelType: TransportModelType,
    mesh: MeshConfig,
    time: TimeConfiguration,
    cflLimit: Float = 0.5
) -> [String: Float] {
    // Calculate cell spacing
    let dx = mesh.minorRadius / Float(mesh.cellCount)
    let dt = time.initialTimeStep

    // CFL-safe maximum diffusivity
    let chiMax = cflLimit * dx * dx / dt

    switch modelType {
    case .constant:
        // Conservative defaults: Use 90% of CFL limit for safety margin
        let safetyFactor: Float = 0.9
        return [
            "ionHeatDiffusivity": chiMax * safetyFactor,
            "electronHeatDiffusivity": chiMax * safetyFactor,
            "particleDiffusivity": chiMax * safetyFactor * 0.2  // Typically lower
        ]

    case .bohmGyrobohm, .qlknn:
        // Computed by model, no explicit parameters
        return [:]

    case .densityTransition:
        // Model-specific parameters (not CFL-limited)
        return [
            "riCoefficient": 0.5,
            "transitionDensity": 2.5e19,
            "transitionWidth": 0.5e19,
            "ionMassNumber": 2.0
        ]
    }
}
```

**Rationale**:
- ✅ Defaults computed WITH mesh/time context
- ✅ Automatically adapts to different mesh resolutions
- ✅ Safety factor (0.9) provides margin
- ✅ Centralized key mapping for consistency

**Example Behavior**:
```
Mesh: cellCount=100, minorRadius=1.0m → dx=0.01m
Time: dt=1e-3s
→ chiMax = 0.5 × 0.01² / 0.001 = 0.05 m²/s
→ default ionHeatDiffusivity = 0.05 × 0.9 = 0.045 m²/s  (safe)

Mesh: cellCount=200, minorRadius=1.0m → dx=0.005m
Time: dt=1e-3s
→ chiMax = 0.5 × 0.005² / 0.001 = 0.0125 m²/s
→ default ionHeatDiffusivity = 0.0125 × 0.9 = 0.01125 m²/s  (auto-adjusted!)
```

### Phase 3: Validator Uses Optional API (REVISED)

**Location**: `Sources/GotenxCore/Configuration/ConfigurationValidator.swift:344-407`

**Current behavior**:
```swift
private static func validateCFLCondition(
    transport: TransportConfig,
    dt: Float,
    cellSpacing: Float
) throws {
    guard let ionHeatDiffusivity = transport.parameter("ionHeatDiffusivity") else {
        throw ConfigurationValidationError.missingRequiredParameter(
            parameter: "ionHeatDiffusivity",
            modelType: transport.modelType,
            suggestion: "Specify ionHeatDiffusivity in transport.parameters or use a model that computes it"
        )
    }

    guard let electronHeatDiffusivity = transport.parameter("electronHeatDiffusivity") else {
        throw ConfigurationValidationError.missingRequiredParameter(
            parameter: "electronHeatDiffusivity",
            modelType: transport.modelType,
            suggestion: "Specify electronHeatDiffusivity in transport.parameters or use a model that computes it"
        )
    }

    // particleDiffusivity is optional for some models
    let particleDiff = transport.parameter("particleDiffusivity", default: 0.0)

    // ✅ Validation only - no default provisioning
    if ionHeatDiffusivity <= 0 {
        throw ConfigurationValidationError.invalidParameter(
            parameter: "ionHeatDiffusivity",
            value: ionHeatDiffusivity,
            reason: "Must be positive"
        )
    }

    if electronHeatDiffusivity <= 0 {
        throw ConfigurationValidationError.invalidParameter(
            parameter: "electronHeatDiffusivity",
            value: electronHeatDiffusivity,
            reason: "Must be positive"
        )
    }

    if particleDiff < 0 {
        throw ConfigurationValidationError.invalidParameter(
            parameter: "particleDiffusivity",
            value: particleDiff,
            reason: "Must be non-negative"
        )
    }

    // Compute CFL numbers
    let ionCFL = ionHeatDiffusivity * dt / (cellSpacing * cellSpacing)
    let electronCFL = electronHeatDiffusivity * dt / (cellSpacing * cellSpacing)
    let CFL_particle = particleDiff * dt / (cellSpacing * cellSpacing)

    if ionCFL > 0.5 {
        throw ConfigurationValidationError.cflViolation(
            parameter: "ionHeatDiffusivity",
            cfl: ionCFL,
            limit: 0.5,
            suggestion: "Reduce ionHeatDiffusivity to \(ionHeatDiffusivity * 0.5 / ionCFL) m²/s or decrease dt to \(dt * 0.5 / ionCFL) s"
        )
    }

    // ... (similar for electronHeatDiffusivity, particleDiffusivity)
}
```

**Rationale**:
- ✅ Explicit error for missing required parameters
- ✅ Distinguishes "missing" from "zero"
- ✅ Clear error messages with suggestions
- ✅ Validator only validates, never provides defaults

    if electronHeatDiffusivity <= 0 {
        throw ConfigurationValidationError.negativeTransportCoefficient(
            parameter: "electronHeatDiffusivity",
            value: electronHeatDiffusivity
        )
    }

    if particleDiff < 0 {
        throw ConfigurationValidationError.negativeTransportCoefficient(
            parameter: "particleDiffusivity",
            value: particleDiff
        )
    }

    // Compute CFL numbers
    let ionCFL = ionHeatDiffusivity * dt / (cellSpacing * cellSpacing)
    let electronCFL = electronHeatDiffusivity * dt / (cellSpacing * cellSpacing)
    let CFL_particle = particleDiff * dt / (cellSpacing * cellSpacing)

    if ionCFL > 0.5 {
        throw ConfigurationValidationError.cflViolation(
            parameter: "ionHeatDiffusivity",
            cfl: ionCFL,
            limit: 0.5,
            suggestion: "Reduce ionHeatDiffusivity to \(ionHeatDiffusivity * 0.5 / ionCFL) m²/s or decrease dt to \(dt * 0.5 / ionCFL) s"
        )
    }

    // ... (similar for electronHeatDiffusivity, particleDiffusivity)
}
```

**Rationale**:
- Validator only **validates**, never **provides** defaults
- Uses domain model's API (`.parameter()`)
- Single Responsibility Principle maintained

### Phase 4: Decouple Loading from Validation

**Location**: `Sources/GotenxCLI/Configuration/GotenxConfigReader.swift:76-91`

**Before**:
```swift
public func fetchConfiguration() async throws -> SimulationConfiguration {
    let runtime = try await fetchRuntimeConfig()
    let time = try await fetchTimeConfig()
    let output = try await fetchOutputConfig()

    let config = SimulationConfiguration(
        runtime: runtime,
        time: time,
        output: output
    )

    // ❌ Validation coupled to loading
    try ConfigurationValidator.validate(config)

    return config
}
```

**After**:
```swift
public func fetchConfiguration() async throws -> SimulationConfiguration {
    let runtime = try await fetchRuntimeConfig()
    let time = try await fetchTimeConfig()
    let output = try await fetchOutputConfig()

    let config = SimulationConfiguration(
        runtime: runtime,
        time: time,
        output: output
    )

    // ✅ No validation - let caller decide
    return config
}
```

**Location**: `Sources/GotenxCore/Configuration/ConfigurationLoader.swift:71`

**Before**:
```swift
public func load() async throws -> SimulationConfiguration {
    // ... loading logic

    try ConfigurationValidator.validate(finalConfig)
    return finalConfig
}
```

**After**:
```swift
public func load() async throws -> SimulationConfiguration {
    // ... loading logic

    // ✅ No validation - return raw loaded config
    return finalConfig
}
```

**Rationale**:
- **Separation of Concerns**: Reading ≠ Validation
- **Testability**: Can test loading without validation
- **Flexibility**: Caller chooses when/if to validate

### Phase 5: Explicit Validation in Production Code

**Location**: `Sources/GotenxCLI/Commands/RunCommand.swift`

**After**:
```swift
let reader = try await GotenxConfigReader.create(
    jsonPath: configPath,
    cliOverrides: cliOverrides
)
let config = try await reader.fetchConfiguration()

// ✅ Explicit validation before use
try ConfigurationValidator.validate(config)

let runner = try await SimulationRunner(config: config)
try await runner.run()
```

**Location**: `Sources/GotenxCLI/Commands/InteractiveMenu.swift:302`

**After**:
```swift
currentConfig = builder.build()

// ✅ Explicit validation with clear error handling
try ConfigurationValidator.validate(currentConfig)

print("✓ Configuration updated")
```

**Rationale**:
- Validation is **explicit** and **visible** at call sites
- Error handling can be customized per use case
- Clear control flow

### Phase 6: Tests Without Validation

**Location**: `Tests/GotenxTests/Configuration/ToraxConfigReaderTests.swift`

**After**:
```swift
@Test("Load minimal configuration from JSON")
func testLoadMinimalConfig() async throws {
    let configPath = try createTestConfig(cellCount: 100)
    defer { cleanupTestConfig(atPath: configPath) }

    let reader = try await GotenxConfigReader.create(
        jsonPath: configPath,
        cliOverrides: [:]
    )

    let config = try await reader.fetchConfiguration()

    // ✅ Test configuration loading only (no validation)
    #expect(config.runtime.static.mesh.cellCount > 0)
    #expect(config.time.end > config.time.start)
    #expect(config.time.initialTimeStep > 0)

    // ✅ Test that defaults were applied
    #expect(config.runtime.dynamic.transport.parameter("ionHeatDiffusivity") == 0.05)

    // ❌ No physics validation - that's a separate concern
}

@Test("Validation catches CFL violations")
func testCFLValidation() throws {
    // Separate test for validation logic
    let config = SimulationConfiguration(...)  // Invalid CFL

    #expect(throws: ConfigurationValidationError.self) {
        try ConfigurationValidator.validate(config)
    }
}
```

**Rationale**:
- **Unit Test Clarity**: Each test has a single concern
- **Fast Tests**: No physics validation overhead
- **Separate Validation Tests**: Explicit tests for validator

## Implementation Plan

### Task Breakdown

| Phase | Task | File | Status |
|-------|------|------|--------|
| 1 | Add strict `TransportConfig` construction | `TransportConfig.swift` | Completed |
| 1 | Add `parameter(_:default:)` instance method | `TransportConfig.swift` | Completed |
| 1 | Add `TransportConfig.defaultConstant` | `TransportConfig.swift` | Completed |
| 2 | Apply CFL-aware defaults before transport validation | `GotenxConfigReader.swift` | Completed |
| 3 | Update `validateCFLCondition()` to use explicit parameters | `ConfigurationValidator.swift` | Completed |
| 3 | Remove default value fallbacks from validator | `ConfigurationValidator.swift` | Completed |
| 4 | Reject obsolete transport keys at decode and construction | `TransportConfig.swift` | Completed |
| 5 | Add dedicated transport validation tests | `TransportParameterValidationTests.swift` | Completed |
| 6 | Run focused xcodebuild validation tests | `TransportParameterValidationTests` | Completed |

### Migration Strategy

**Step 1: Add Strict Construction APIs**
- Make `TransportConfig` validate model-specific keys and required constant parameters at construction.
- Add `TransportConfig.defaultConstant` for explicit built-in defaults.
- Add `TransportConfig.parameter(_:default:)` for optional context-independent fallbacks.

**Step 2: Update Internal Implementations**
- Update `GotenxConfigReader.fetchTransportConfig()`
- Update `ConfigurationValidator.validateCFLCondition()`

**Step 3: Remove Validation Coupling**
- Remove `validate()` calls from readers
- Add explicit `validate()` calls in production code

**Step 4: Update Tests**
- Remove validation expectations from reader tests
- Add dedicated validation tests

**Step 5: Verification**
- Run full test suite
- Test all example configurations
- Verify CLI still works

### Backward Compatibility

**Existing JSON Files**: ✅ **No changes required**

All existing configuration files remain valid:

```json
// Still works - defaults applied automatically
{
  "transport": {
    "modelType": "constant",
    "parameters": {}
  }
}

// Still works - explicit values override defaults
{
  "transport": {
    "modelType": "constant",
    "parameters": {
      "ionHeatDiffusivity": 1.0,
      "electronHeatDiffusivity": 1.0
    }
  }
}
```

**Existing Code**: ⚠️ **Minimal breaking changes**

Code that directly creates `TransportConfig` is unaffected. Only code that relied on implicit validation needs updating:

```swift
// Before (implicit validation)
let config = try await reader.fetchConfiguration()  // Throws if invalid

// After (explicit validation)
let config = try await reader.fetchConfiguration()
try ConfigurationValidator.validate(config)  // Explicit call
```

## Benefits

### 1. Architectural Clarity

```
Before: Reader → Config (with implicit validation)
After:  Reader → Config → Explicit Validation
```

Each component has a single, clear responsibility.

### 2. Improved Testability

```swift
// Can now test loading independently
let config = try await reader.fetchConfiguration()
#expect(config.runtime.static.mesh.cellCount == 100)

// Can now test validation independently
#expect(throws: Error.self) {
    try ConfigurationValidator.validate(invalidConfig)
}
```

### 3. Domain Model Encapsulation

```swift
// Context-free built-in defaults are explicit.
let defaults = TransportConfig.defaultConstant
// => ion/electron heat diffusivity keys are present and validated.
```

### 4. Explicit Control Flow

```swift
// Clear intent at call sites
let config = try await reader.fetchConfiguration()
try ConfigurationValidator.validate(config)  // Explicit validation
```

### 5. Better Error Messages

Since validation is model-aware, error messages can be specific:

```
Before: "ionHeatDiffusivity not found, using default 1.0"
After:  "Missing required parameter - ionHeatDiffusivity"
```

## Risks and Mitigations

### Risk 1: Breaking Existing Code

**Mitigation**:
- Staged rollout (add APIs first, then migrate)
- Comprehensive test coverage
- Document migration guide

### Risk 2: Test Suite Disruption

**Mitigation**:
- Update tests incrementally
- Keep validation tests separate
- Maintain existing test coverage

### Risk 3: Default Value Changes

**Mitigation**:
- Document default value changes clearly
- Add migration notes for users
- Provide override mechanism

## Testing Strategy

### Unit Tests

```swift
// Test obsolete key rejection
@Test("TransportConfig rejects obsolete parameter keys")
func testTransportKeyValidation() {
    #expect(throws: ConfigurationValidationError.self) {
        _ = try TransportConfig(
            modelType: .bohmGyrobohm,
            parameters: ["obsoleteParameter": 0.5]
        )
    }
}

// Test explicit default configuration
@Test("TransportConfig.defaultConstant has required parameters")
func testDefaultConstantTransport() {
    let config = TransportConfig.defaultConstant
    #expect(config.parameter("ionHeatDiffusivity") == 1.0)
    #expect(config.parameter("electronHeatDiffusivity") == 1.0)
    #expect(config.parameter("custom", default: 0.1) == 0.1)  // Provided default
}

// Test reading without validation
@Test("GotenxConfigReader loads without validation")
func testLoadWithoutValidation() async throws {
    let reader = try await GotenxConfigReader.create(...)
    let config = try await reader.fetchConfiguration()
    // No validation error even with invalid CFL
}

// Test validation separately
@Test("ConfigurationValidator catches CFL violations")
func testCFLValidation() throws {
    let config = createInvalidConfig()  // CFL > 0.5
    #expect(throws: ConfigurationValidationError.cflViolation) {
        try ConfigurationValidator.validate(config)
    }
}
```

### Integration Tests

```swift
@Test("Full workflow with explicit validation")
func testFullWorkflow() async throws {
    let reader = try await GotenxConfigReader.create(...)
    let config = try await reader.fetchConfiguration()

    // Explicit validation before use
    try ConfigurationValidator.validate(config)

    let runner = try await SimulationRunner(config: config)
    try await runner.run()
}
```

## Success Criteria

✅ **All existing tests pass** with minimal modifications
✅ **All example configurations work** without changes
✅ **ToraxConfigReaderTests pass** without CFL violations
✅ **Validation tests** explicitly cover physics constraints
✅ **Code coverage** maintained or improved
✅ **Documentation** updated to reflect new architecture

## References

### Related Documents
- [CONFIGURATION_SYSTEM.md](../docs/CONFIGURATION_SYSTEM.md) - Current configuration documentation
- [CONFIGURATION_VALIDATION_SPEC.md](./CONFIGURATION_VALIDATION_SPEC.md) - Validation specification

### Design Patterns
- **Single Responsibility Principle** (SOLID)
- **Separation of Concerns**
- **Domain-Driven Design** (defaults as domain knowledge)
- **Explicit over Implicit** (The Zen of Python)

### Code References
- `Sources/GotenxCore/Configuration/TransportConfig.swift` - Domain model
- `Sources/GotenxCLI/Configuration/GotenxConfigReader.swift` - Configuration reader
- `Sources/GotenxCore/Configuration/ConfigurationValidator.swift` - Validation logic
- `Tests/GotenxTests/Configuration/ToraxConfigReaderTests.swift` - Failing tests

## Appendix A: Complete Example

### Previous Problem

```swift
// TransportConfig.swift
struct TransportConfig {
    let modelType: TransportModelType
    let parameters: [String: Float]  // No defaults
}

// ConfigurationValidator.swift
// The validator previously supplied fallback values for missing transport keys.

// GotenxConfigReader.swift
let config = try await reader.fetchConfiguration()  // ❌ Implicit validation

// ToraxConfigReaderTests.swift
let config = try await reader.fetchConfiguration()
// ❌ Fails: CFL = 10 >> 0.5
```

### Current Implemented State

```swift
// TransportConfig.swift
extension TransportConfig {
    func validateParameterKeys() throws {
        try transportParameters().validateParameterKeys()
    }
}

// ConfigurationValidator.swift
try transport.validateParameterKeys()
let ionHeatDiffusivity = try transport.requireParameter("ionHeatDiffusivity")
let electronHeatDiffusivity = try transport.requireParameter("electronHeatDiffusivity")

// GotenxConfigReader.swift
private func fetchTransportConfig() throws -> TransportConfig {
    // Apply context-aware defaults only in the configuration reader.
    return try TransportConfig(modelType: modelType, parameters: parameters)
}

// RunCommand.swift
let config = try await reader.fetchConfiguration()
try ConfigurationValidator.validate(config)

// ToraxConfigReaderTests.swift
let config = try await reader.fetchConfiguration()
try config.runtime.dynamic.transport.validateParameterKeys()
```

## Appendix B: CFL-Safe Default Calculation

For numerical stability, the CFL condition must be satisfied:

```
CFL = χ * Δt / Δx² < 0.5
```

For typical test scenarios:
- `Δt = 1e-3 s` (1 millisecond timestep)
- `dx = minorRadius / cellCount = 1.0 / 100 = 0.01 m`

Therefore:
```
χ_max = 0.5 * Δx² / Δt
      = 0.5 * (0.01)² / 0.001
      = 0.5 * 0.0001 / 0.001
      = 0.05 m²/s
```

Hence the default value `ionHeatDiffusivity = 0.05 m²/s` for the constant transport model.

---

**Document Version**: 1.0
**Last Updated**: 2026-06-02
**Review Status**: Completed
