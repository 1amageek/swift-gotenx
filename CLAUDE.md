# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

swift-Gotenx is a Swift implementation of Google DeepMind's TORAX (https://github.com/google-deepmind/torax), a differentiable tokamak core transport simulator. It leverages Swift 6.2 and Apple's MLX framework to achieve high-performance fusion plasma simulations optimized for Apple Silicon.

### Key Technologies

- **Swift 6.2**: Strict concurrency, value semantics, protocol-oriented design
- **MLX-Swift**: GPU-accelerated array operations, automatic differentiation, JIT compilation
- **Swift Numerics**: Special functions (gamma, erfc), complex numbers, high-precision arithmetic
- **Swift Configuration**: Type-safe hierarchical configuration management
- **NetCDF-4**: Compressed output with DEFLATE level 6

---

## 📚 Documentation Structure

All detailed technical documentation is in `docs/` for easier maintenance:

| Topic | Document | Purpose |
|-------|----------|---------|
| **Architecture** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | TORAX concepts, Swift patterns, future extensions |
| **Unit System** | [docs/UNIT_SYSTEM.md](docs/UNIT_SYSTEM.md) | SI-based units (eV, m⁻³), conversion guidelines |
| **Configuration** | [docs/CONFIGURATION_SYSTEM.md](docs/CONFIGURATION_SYSTEM.md) | Hierarchical config (CLI, env, JSON) |
| **Numerical Precision** | [docs/NUMERICAL_PRECISION.md](docs/NUMERICAL_PRECISION.md) | Float32 policy, GPU constraints, stability |
| **Numerical Robustness** | [docs/NUMERICAL_ROBUSTNESS_DESIGN.md](docs/NUMERICAL_ROBUSTNESS_DESIGN.md) | 🔥 **NaN crash prevention, validation** |
| **MLX Best Practices** | [docs/MLX_BEST_PRACTICES.md](docs/MLX_BEST_PRACTICES.md) | Lazy evaluation, eval() patterns |
| **Swift Concurrency** | [docs/SWIFT_CONCURRENCY.md](docs/SWIFT_CONCURRENCY.md) | EvaluatedArray, actor isolation |
| **Transport Models** | [docs/TRANSPORT_MODELS.md](docs/TRANSPORT_MODELS.md) | Constant, Bohm-GyroBohm, QLKNN |
| **🔥 FVM Improvements** | [docs/FVM_NUMERICAL_IMPROVEMENTS_PLAN.md](docs/FVM_NUMERICAL_IMPROVEMENTS_PLAN.md) | **PRIORITY**: Power-law scheme, Sauter bootstrap |

**Complete documentation catalog**: [docs/README.md](docs/README.md)

---

## ⚠️ Critical Constraints for Development

### Apple Silicon GPU Limitations

**Float64 is NOT supported on Apple Silicon GPUs.** All MLXArray computations MUST use Float32.

```swift
// ❌ FAILS at runtime
let array = MLXArray([1.0, 2.0], dtype: .float64)

// ✅ WORKS
let array = MLXArray([1.0, 2.0], dtype: .float32)
```

**Mitigation**: Use algorithmic stability techniques (variable scaling, preconditioning, conservation enforcement) instead of higher precision.

**Full details**: [docs/NUMERICAL_PRECISION.md](docs/NUMERICAL_PRECISION.md)

---

### MLXArray Initialization

**CRITICAL**: MLXArray initialization methods differ from standard Swift arrays. Using incorrect initializers will cause compilation errors.

#### Common Mistakes and Corrections

**❌ WRONG - `repeating:` does NOT exist**:
```swift
let Ti = MLXArray(repeating: 5000.0, [cellCount])  // ❌ Compilation error
```

**✅ CORRECT - Use `MLXArray.full()`**:
```swift
let Ti = MLXArray.full([cellCount], values: MLXArray(5000.0))
```

**❌ WRONG - `linspace` is NOT a standalone function**:
```swift
let psi = MLXArray(linspace(0.0, 1.0, count: cellCount))  // ❌ Compilation error
```

**✅ CORRECT - Use `MLXArray.linspace()`**:
```swift
let psi = MLXArray.linspace(0.0, 1.0, count: cellCount)
```

#### Standard Initialization Methods

```swift
// 1. Fill with constant value
let ones = MLXArray.full([cellCount], values: MLXArray(1.0))

// 2. Zeros and ones
let zeros = MLXArray.zeros([cellCount])
let ones = MLXArray.ones([cellCount])

// 3. Linearly spaced values
let linspace = MLXArray.linspace(0.0, 1.0, count: 100)

// 4. From array literal
let array = MLXArray([Float(1.0), Float(2.0), Float(3.0)])

// 5. Scalar value
let scalar = MLXArray(Float(42.0))
```

**Key Rules**:
1. Always use `MLXArray.` static methods for array creation
2. Use `MLXArray(value)` to wrap scalar values for `values:` parameter
3. Prefer explicit `Float()` casts to avoid type ambiguity
4. Use `.float32` dtype for GPU operations

**Full details**: [docs/MLX_BEST_PRACTICES.md](docs/MLX_BEST_PRACTICES.md)

---

### MLX Lazy Evaluation

**Operations are NOT executed immediately** - they queue until `eval()` is called.

```swift
// ❌ WRONG: Unevaluated graph returned
let result = exp(-1000.0 / temperature)
return result

// ✅ CORRECT: Force evaluation
let result = exp(-1000.0 / temperature)
eval(result)
return result
```

**What eval() actually does:**
- Executes pending operations in the computation graph
- Materializes results (stores computed values)
- Does **NOT** clear the graph (graph persists after eval())
- Prevents redundant recomputation by finalizing intermediate results

**When eval() is mandatory**:
- End of computation chains
- Before wrapping in `EvaluatedArray` (automatic)
- Before crossing actor boundaries
- End of each timestep in loops
- **In loops with independent iterations** (e.g., Jacobian computation) to prevent graph accumulation

**Critical**: Without eval() in independent iteration loops, the graph accumulates causing exponential slowdown:
- 200 iterations without eval(): 0.06s → 1.0s per iteration (17x slower, ~120s total)
- 200 iterations with eval(): consistent 0.06s per iteration (~12s total, **10x faster**)

**Full details**: [docs/MLX_BEST_PRACTICES.md](docs/MLX_BEST_PRACTICES.md)

---

### MLX Optimization Strategy - Graph Fusion

**CRITICAL**: Excessive eval() calls prevent MLX from optimizing your computation graph and can cause 2-10× performance degradation.

#### How MLX Optimizes: Operation Fusion

MLX fuses multiple operations into a single GPU kernel when it sees a large computation graph. This is the primary optimization mechanism.

```swift
// ✅ OPTIMAL: Chain operations → MLX fuses into 1 GPU kernel
let a = x + y      // lazy - added to graph
let b = a * 2      // lazy - added to graph
let c = sqrt(b)    // lazy - added to graph
eval(c)  // ← MLX fuses: (x + y) * 2 → sqrt into single kernel

// Performance: 1 GPU call, 1 memory transfer

// ❌ SUBOPTIMAL: eval() breaks fusion → 3 separate GPU kernels
let a = x + y
eval(a)  // ❌ Forces execution → GPU call #1
let b = a * 2
eval(b)  // ❌ Forces execution → GPU call #2
let c = sqrt(b)
eval(c)  // ❌ Forces execution → GPU call #3

// Performance: 3 GPU calls, 3 memory transfers (2-3× slower)
```

**Why fusion matters:**
- **Kernel launch overhead**: Each GPU call has ~10-100μs overhead
- **Memory bandwidth**: Intermediate results don't need to go to DRAM
- **Register reuse**: Fused operations keep data in GPU registers
- **Cache efficiency**: Better temporal locality

#### Golden Rule: Let Graphs Grow Across Function Boundaries

**Design pattern for swift-gotenx:**

```swift
// ✅ CORRECT: Functions return lazy computation graphs
public func computeOhmicHeating(
    Te: MLXArray,
    jParallel: MLXArray,
    geometry: Geometry
) -> MLXArray {
    let eta = computeResistivity(Te, geometry)  // lazy
    let Q_ohm = eta * jParallel * jParallel     // lazy
    return Q_ohm  // Return graph - caller decides when to eval
}

public func computeTransport(...) -> TransportCoefficients {
    let ionHeatDiffusivity = exp(-1000.0 / Ti)  // lazy
    let electronHeatDiffusivity = exp(-1000.0 / Te)  // lazy
    let D = electronHeatDiffusivity * 0.5            // lazy

    // EvaluatedArray() calls eval() - optimal timing
    // At this point, all operations are fused
    return TransportCoefficients(
        ionHeatDiffusivity: EvaluatedArray(evaluating: ionHeatDiffusivity),
        electronHeatDiffusivity: EvaluatedArray(evaluating: electronHeatDiffusivity),
        particleDiffusivity: EvaluatedArray(evaluating: D)
    )
}

// Caller can chain multiple computations
let heating = computeOhmicHeating(Te, j, geom)  // lazy
let transport = computeTransport(...)           // lazy + eval()
// heating and transport operations can be fused where possible
```

**Anti-pattern to avoid:**

```swift
// ❌ WRONG: eval() inside function breaks fusion chain
public func computeOhmicHeating(...) -> MLXArray {
    let eta = computeResistivity(Te, geometry)
    let Q_ohm = eta * jParallel * jParallel
    eval(Q_ohm)  // ❌ Breaks graph - caller can't fuse with next operation
    return Q_ohm
}

// ❌ Result: Caller cannot fuse Q_ohm with subsequent operations
let Q_ohm = computeOhmicHeating(...)  // Already evaluated
let total = Q_ohm + Q_ecrh  // Separate operation - cannot fuse
```

#### Performance Cost of Unnecessary eval()

| Impact | Optimal (late eval) | Suboptimal (early eval) | Performance Loss |
|--------|--------------------|-----------------------|------------------|
| GPU kernel calls | 1 | N (per eval) | N× launch overhead |
| Memory bandwidth | Minimal | Intermediate results written/read | 2-5× bandwidth waste |
| Operation fusion | Full | None | 2-10× slower execution |
| Memory allocation | Optimized | Extra buffers | Memory pressure |
| Cache efficiency | High | Low | More cache misses |

**Real-world example from swift-gotenx:**

```swift
// Spatial operator pipeline (Block1DCoeffs.swift)
// Bad: ~50ms, Good: ~10ms (5× faster with fusion)

// ❌ BAD: Multiple eval() calls
let face_values = interpolateToFaces(cell_values)
eval(face_values)  // ❌
let fluxes = computeFluxes(face_values)
eval(fluxes)  // ❌
let divergence = computeDivergence(fluxes)
eval(divergence)  // ❌

// ✅ GOOD: Single eval() at end
let face_values = interpolateToFaces(cell_values)  // lazy
let fluxes = computeFluxes(face_values)            // lazy
let divergence = computeDivergence(fluxes)         // lazy
eval(divergence)  // All fused into optimized kernel
```

#### When to eval() - Decision Framework

```
Got computation result
    ↓
[1] Need actual values? (.item(), .asArray(), print)
    YES → eval() REQUIRED
    NO  ↓
[2] Passing to another function?
    YES → DON'T eval() - let graphs chain
    NO  ↓
[3] Wrapping in EvaluatedArray?
    YES → Use EvaluatedArray(evaluating:) - auto eval at optimal time
    NO  ↓
[4] Inside independent iteration loop?
    (result doesn't affect next iteration)
    YES → eval() REQUIRED - prevent graph accumulation
    NO  ↓
[5] End of timestep/iteration in sequential loop?
    (result used in next iteration)
    YES → eval() REQUIRED - finalize iteration state
    NO  → DON'T eval() - keep building graph
```

#### Loop Patterns: When to eval()

```swift
// Pattern 1: Sequential loop (state carries forward)
var state = initialState
for step in 0..<1000 {
    state = updateState(state)  // lazy
    eval(state)  // ✅ Required: finalize iteration state
}

// Pattern 2: Independent iterations (parallel-safe)
for i in 0..<200 {
    let result = computeJacobianColumn(i)  // independent
    eval(result)  // ✅ Required: prevent graph accumulation
    columns.append(result)
}

// Pattern 3: Batch processing (vectorizable)
let results = inputs.map { input in
    process(input)  // lazy
}
eval(results...)  // ✅ Single eval at end allows cross-operation fusion
```

#### Testing for Fusion Effectiveness

**Add timing diagnostics:**

```swift
let t0 = Date()
for _ in 0..<100 {
    let result = yourComputation(...)
    eval(result)
}
let elapsed = Date().timeIntervalSince(t0)
print("Average time per iteration: \(elapsed / 100)s")

// Compare with:
// - Early eval(): expect ~2-10× slower
// - Late eval(): baseline performance
```

**Check for fusion using MLX profiling** (if available):
- Number of kernel launches should be minimal
- Memory transfers should be optimized
- Intermediate allocations should be eliminated

#### Summary: Optimization Checklist

✅ **DO:**
- Return lazy MLXArrays from functions (let caller eval)
- Chain operations as much as possible
- Use EvaluatedArray for type-safe eval at optimal time
- eval() at end of computation chains
- eval() in independent iteration loops (prevent accumulation)
- eval() at end of each timestep in sequential loops

❌ **DON'T:**
- eval() in middle of function if result will be used in next operation
- eval() the same array multiple times (no-op, but suggests design issue)
- eval() intermediate steps in computation chains
- Assume "more eval() = more correct" (opposite is often true)

**Full technical details**: [docs/MLX_BEST_PRACTICES.md](docs/MLX_BEST_PRACTICES.md)

---

### Swift 6 Concurrency

MLXArray is NOT Sendable. Use `EvaluatedArray` wrapper for all data crossing actor boundaries:

```swift
// ✅ Type-safe wrapper
public struct EvaluatedArray: @unchecked Sendable {
    public init(evaluating array: MLXArray) {
        eval(array)  // Guaranteed evaluation
        self.array = array
    }
    public var value: MLXArray { array }
}

// ✅ Sendable data structures
public struct CoreProfiles: Sendable {
    public let ionTemperature: EvaluatedArray
    public let electronTemperature: EvaluatedArray
}
```

**Full details**: [docs/SWIFT_CONCURRENCY.md](docs/SWIFT_CONCURRENCY.md)

---

## Unit System Standard

**Critical**: Consistent SI-based units throughout to prevent 1000× errors.

| Quantity | Unit | Symbol | Notes |
|----------|------|--------|-------|
| Temperature | electron volt | **eV** | NOT keV |
| Density | particles/m³ | **m⁻³** | NOT 10²⁰ m⁻³ |
| Power | megawatts/m³ | **MW/m³** | Source terms only |

**Conversion point**: `Block1DCoeffsBuilder` converts MW/m³ → eV/(m³·s) via `UnitConversions.megawattsToEvDensity()`.

**Display units** (output only): Temperature in keV (`/1000`), density in 10²⁰ m⁻³ (`/1e20`).

**Full details**: [docs/UNIT_SYSTEM.md](docs/UNIT_SYSTEM.md)

---

## Configuration System

Hierarchical priority (highest to lowest):
1. CLI arguments (`--mesh-cell-count 200`)
2. Environment variables (`GOTENX_MESH_CELL_COUNT=150`)
3. JSON file
4. Default values

```swift
let configReader = try await GotenxConfigReader.create(
    jsonPath: "config.json",
    cliOverrides: ["runtime.static.mesh.cellCount": "200"]
)
let config = try await configReader.fetchConfiguration()
```

**Full details**: [docs/CONFIGURATION_SYSTEM.md](docs/CONFIGURATION_SYSTEM.md)

---

## Common Development Commands

### Build and Test

```bash
# Build package
xcodebuild build \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Debug

# Run tests
perl -e 'alarm shift; exec @ARGV' 700 xcodebuild test \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Debug \
  -derivedDataPath /tmp/Gotenx-Package-Xcodebuild-DerivedData

# Run specific test
perl -e 'alarm shift; exec @ARGV' 120 xcodebuild test \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Debug \
  -derivedDataPath /tmp/Gotenx-Package-Xcodebuild-DerivedData \
  -only-testing:GotenxTests/<SuiteName>

# Release build
xcodebuild build \
  -scheme swift-gotenx-Package \
  -destination 'platform=macOS' \
  -configuration Release
```

### CLI Usage

```bash
# Run simulation
.build/release/GotenxCLI run --config Examples/Configurations/iter_like.json

# Install CLI
swift package experimental-install -c release
```

### Package Management

```bash
swift package update
swift package resolve
swift package show-dependencies
```

---

## SwiftUI Preview Best Practices

**CRITICAL**: `@Previewable` declarations MUST appear first in `#Preview` body:

```swift
#Preview("Example") {
    // ✅ CORRECT: @Previewable first
    @Previewable @State var value = 0
    @Previewable @State var isOn = false

    // ✅ Other declarations after
    let config = PlotConfiguration.default

    // ✅ View construction last
    MyView(value: $value, isOn: $isOn)
}
```

**Minimum**: macOS 26.4+ on a Metal 4 capable Apple Silicon GPU

---

## Project Status

**Current Capabilities** (October 2025):
- ✅ Core simulation infrastructure
- ✅ Transport models (Constant, Bohm-GyroBohm, QLKNN)
- ✅ Source models (Fusion, Ohmic, Ion-Electron, Bremsstrahlung)
- ✅ Solvers (Linear, Newton-Raphson)
- ✅ CLI integration (GotenxCLI)
- ✅ NetCDF output with compression
- ✅ Conservation enforcement

**Priority Improvements**:
- 🔥 FVM numerical enhancements (power-law scheme, Sauter bootstrap) - See [docs/FVM_NUMERICAL_IMPROVEMENTS_PLAN.md](docs/FVM_NUMERICAL_IMPROVEMENTS_PLAN.md)
- 🚧 Configuration system refactoring - See [docs/CONFIGURATION_ARCHITECTURE_REFACTORING.md](docs/CONFIGURATION_ARCHITECTURE_REFACTORING.md)

---

## Quick Reference for Common Tasks

### Starting a New Feature
1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design patterns
2. Check relevant technical docs in [docs/README.md](docs/README.md)
3. Follow Swift 6 concurrency patterns from [docs/SWIFT_CONCURRENCY.md](docs/SWIFT_CONCURRENCY.md)

### Working with MLX
1. Use correct initialization: `MLXArray.full()`, `MLXArray.linspace()`
2. Call `eval()` at computation chain ends
3. Wrap results in `EvaluatedArray` for Sendable compliance

### Handling Configuration
1. Use hierarchical priority: CLI > Env > JSON > Default
2. Validate with `ConfigurationValidator` - See [docs/CONFIGURATION_VALIDATION_SPEC.md](docs/CONFIGURATION_VALIDATION_SPEC.md)
3. Follow CFL-aware defaults pattern

### Debugging Numerical Issues
1. **Encountering NaN/Inf crashes?** 🔥 See [docs/NUMERICAL_ROBUSTNESS_DESIGN.md](docs/NUMERICAL_ROBUSTNESS_DESIGN.md) for validation and crash prevention
2. Check Float32 constraints: [docs/NUMERICAL_PRECISION.md](docs/NUMERICAL_PRECISION.md)
3. Verify unit consistency: [docs/UNIT_SYSTEM.md](docs/UNIT_SYSTEM.md)
4. Review MLX evaluation patterns: [docs/MLX_BEST_PRACTICES.md](docs/MLX_BEST_PRACTICES.md)

---

## References

### Original TORAX
- **GitHub**: https://github.com/google-deepmind/torax
- **Paper**: arXiv:2406.06718v2 - "TORAX: A Differentiable Tokamak Transport Simulator"
- **DeepWiki**: https://deepwiki.com/google-deepmind/torax

### MLX Framework
- **GitHub**: https://github.com/ml-explore/mlx-swift
- **DeepWiki**: https://deepwiki.com/ml-explore/mlx-swift

### Swift Packages
- **Swift Numerics**: https://github.com/apple/swift-numerics (https://deepwiki.com/apple/swift-numerics)
- **Swift Configuration**: https://github.com/apple/swift-configuration (https://deepwiki.com/apple/swift-configuration)
- **Swift Argument Parser**: https://github.com/apple/swift-argument-parser

---

**For detailed information, see**: [docs/README.md](docs/README.md)

*Last updated: 2025-10-26* (MLX eval() corrections, graph accumulation, and optimization strategy documentation)
