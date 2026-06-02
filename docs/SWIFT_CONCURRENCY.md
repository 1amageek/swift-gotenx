# Swift 6 Concurrency Patterns

## The Challenge

MLXArray is **not Sendable** by design - it's a reference type (class) that wraps a C++ `mlx_array` with reference counting. However, Swift 6's strict concurrency requires all data crossing actor boundaries or async contexts to be Sendable.

## Why We MUST Keep MLXArray Throughout Computation

**Critical requirement**: The entire computation chain must remain as MLXArray for:

### 1. Automatic Differentiation

`grad()` requires an unbroken computation graph:

```swift
// Newton-Raphson needs this:
let jacobian = grad(computeResidual)(profiles)  // Must be MLXArray chain
```

### 2. compile() Optimization

Graph must be continuous for fusion and optimization:

```swift
let step = compile { state in
    // All operations on MLXArray - optimized as single kernel
    let coeffs = calculateCoeffs(state.profiles)  // MLXArray
    let residual = computeResidual(state.profiles, coeffs)  // MLXArray
    return solveNewton(residual)  // MLXArray
}
```

### 3. Iterative Solvers

10-100 iterations per timestep:
- Converting to Swift arrays would break the computation graph
- 30 Newton iterations × conversion overhead = unacceptable
- Need continuous MLXArray chain for grad() at each iteration

---

## The Solution: Type-Safe EvaluatedArray Wrapper

**KEY DESIGN DECISION**: We use a type-safe wrapper to enforce evaluation at compile time, not runtime comments.

```swift
/// Type-safe wrapper ensuring MLXArray has been evaluated
/// This is the ONLY type marked @unchecked Sendable
public struct EvaluatedArray: @unchecked Sendable {
    private let array: MLXArray

    /// Only way to create: forces evaluation
    public init(evaluating array: MLXArray) {
        eval(array)  // Guaranteed evaluation
        self.array = array
    }

    /// Batch evaluation for efficiency
    public static func evaluatingBatch(_ arrays: [MLXArray]) -> [EvaluatedArray] {
        // Force evaluation of all arrays
        arrays.forEach { eval($0) }
        return arrays.map { EvaluatedArray(preEvaluated: $0) }
    }

    private init(preEvaluated: MLXArray) {
        self.array = preEvaluated
    }

    /// Read-only access to evaluated array
    public var value: MLXArray { array }

    // MARK: - Convenience accessors

    /// Shape of the evaluated array
    public var shape: [Int] { array.shape }

    /// Number of dimensions
    public var ndim: Int { array.ndim }

    /// Data type
    public var dtype: DType { array.dtype }
}

/// Core data structures are now truly Sendable
public struct CoreProfiles: Sendable {
    public let ionTemperature: EvaluatedArray
    public let electronTemperature: EvaluatedArray
    public let electronDensity: EvaluatedArray
    public let poloidalFlux: EvaluatedArray
}

public struct TransportCoefficients: Sendable {
    public let ionHeatDiffusivity: EvaluatedArray
    public let electronHeatDiffusivity: EvaluatedArray
    public let particleDiffusivity: EvaluatedArray
    public let convectionVelocity: EvaluatedArray
}
```

---

## Why This Design is Superior

1. **Type-Level Safety**: Cannot create unevaluated arrays that cross actor boundaries
2. **No Comment Dependencies**: Compiler enforces evaluation, not developer discipline
3. **Minimal @unchecked Sendable**: Only `EvaluatedArray` needs it, not every data structure
4. **Clear Intent**: `EvaluatedArray` vs `MLXArray` clearly communicates evaluation state
5. **Batch Optimization**: `evaluatingBatch()` enables efficient multi-array evaluation

---

## Data Flow Pattern

### 1. Computation layer: Use MLXArray for chained operations

```swift
func computeTransport(_ profiles: CoreProfiles) -> TransportCoefficients {
    // Extract lazy MLXArrays for computation
    let ti = profiles.ionTemperature.value
    let te = profiles.electronTemperature.value

    // Chain operations (lazy)
    let ionHeatDiffusivity = exp(-1000.0 / ti)
    let electronHeatDiffusivity = exp(-1000.0 / te)

    // Force evaluation before wrapping
    return TransportCoefficients(
        ionHeatDiffusivity: EvaluatedArray(evaluating: ionHeatDiffusivity),
        electronHeatDiffusivity: EvaluatedArray(evaluating: electronHeatDiffusivity),
        particleDiffusivity: EvaluatedArray(evaluating: electronHeatDiffusivity * 0.5),
        convectionVelocity: EvaluatedArray.zeros([profiles.ionTemperature.shape[0]])
    )
}
```

### 2. Batch evaluation for efficiency

```swift
func computeStep(_ profiles: CoreProfiles) -> CoreProfiles {
    let ti = profiles.ionTemperature.value
    let te = profiles.electronTemperature.value
    let ne = profiles.electronDensity.value
    let psi = profiles.poloidalFlux.value

    // All operations are lazy
    let newTi = ti + computeDeltaTi(ti, te)
    let newTe = te + computeDeltaTe(ti, te, ne)
    let newNe = ne + computeDeltaNe(ne, psi)
    let newPsi = psi + computeDeltaPsi(psi)

    // Single batch evaluation (efficient)
    let evaluated = EvaluatedArray.evaluatingBatch([newTi, newTe, newNe, newPsi])

    return CoreProfiles(
        ionTemperature: evaluated[0],
        electronTemperature: evaluated[1],
        electronDensity: evaluated[2],
        poloidalFlux: evaluated[3]
    )
}
```

### 3. I/O boundary: Convert for serialization

```swift
struct SerializableProfiles: Sendable, Codable {
    let ionTemperature: [Float]
    let electronTemperature: [Float]
    let electronDensity: [Float]
    let poloidalFlux: [Float]
}

extension CoreProfiles {
    func toSerializable() -> SerializableProfiles {
        SerializableProfiles(
            ionTemperature: ionTemperature.value.asArray(Float.self),
            electronTemperature: electronTemperature.value.asArray(Float.self),
            electronDensity: electronDensity.value.asArray(Float.self),
            poloidalFlux: poloidalFlux.value.asArray(Float.self)
        )
    }
}
```

### 4. Safe usage across actors

```swift
let profiles = computeStep(initialProfiles)
// Safe: EvaluatedArray guarantees evaluation
await actor1.process(profiles)
await actor2.analyze(profiles)
// Convert only for output
try profiles.toSerializable().saveToFile("output.json")
```

---

## Actor Isolation and compile()

**CRITICAL**: Swift 6 forbids capturing actor `self` in `compile()` closures.

### ❌ WRONG: Actor Self Capture

```swift
public actor SimulationOrchestrator {
    private let transport: any TransportModel

    init(...) {
        // ❌ Undefined behavior: captures actor self
        self.compiledStep = compile { state in
            self.transport.compute(...)  // Actor self escapes!
        }
    }
}
```

### ✅ CORRECT: Pure Function Compilation

```swift
public actor SimulationOrchestrator {
    private let staticParams: StaticRuntimeParams
    private let transport: any TransportModel
    private let sources: any SourceModel

    // Compiled pure function (no actor dependency)
    private let compiledStep: (CoreProfiles, DynamicRuntimeParams) -> CoreProfiles

    public init(
        staticParams: StaticRuntimeParams,
        transport: any TransportModel,
        sources: any SourceModel
    ) {
        self.staticParams = staticParams
        self.transport = transport
        self.sources = sources

        // Compile a pure function with all dependencies captured
        self.compiledStep = compile(
            Self.makeStepFunction(
                staticParams: staticParams,
                transport: transport,
                sources: sources
            )
        )
    }

    /// Create pure function (not actor-isolated)
    private static func makeStepFunction(
        staticParams: StaticRuntimeParams,
        transport: any TransportModel,
        sources: any SourceModel
    ) -> (CoreProfiles, DynamicRuntimeParams) -> CoreProfiles {
        return { profiles, dynamicParams in
            // Pure computation - all dependencies captured
            let geometry = Geometry(config: staticParams.mesh)
            let transportCoeffs = transport.computeCoefficients(
                profiles: profiles,
                geometry: geometry,
                params: dynamicParams.transportParams
            )
            let sourceTerms = sources.computeTerms(
                profiles: profiles,
                geometry: geometry,
                params: dynamicParams.sourceParams
            )
            // ... Newton-Raphson solver
            return updatedProfiles
        }
    }

    public func step(
        _ profiles: CoreProfiles,
        dynamicParams: DynamicRuntimeParams
    ) async -> CoreProfiles {
        // Call compiled function (no actor isolation issues)
        return compiledStep(profiles, dynamicParams)
    }
}
```

---

## CoeffsCallback Design: Synchronous API

**CRITICAL**: MLX operations are synchronous. Do NOT use async unnecessarily.

```swift
/// Synchronous callback for coefficient computation
public typealias CoeffsCallback = @Sendable (CoreProfiles, Geometry) -> Block1DCoeffs

/// Design Pattern: Closure Capture for Additional Context
///
/// The callback accepts only (CoreProfiles, Geometry) to keep the signature simple.
/// Additional context (dynamicParams, staticParams, transport models, etc.) is
/// provided via closure capture from the enclosing scope.
///
/// Example:
/// let coeffsCallback: CoeffsCallback = { profiles, geometry in
///     // Capture transport, sources, dynamicParams from outer scope
///     let transportCoeffs = transport.computeCoefficients(
///         profiles: profiles,
///         geometry: geometry,
///         params: dynamicParams.transportParams  // Captured
///     )
///     return buildBlock1DCoeffs(transport: transportCoeffs, ...)
/// }
///
/// This allows the solver to remain agnostic about parameter sources while
/// maintaining access to all necessary context through closure capture.

/// Thread-safe synchronous cache
public final class CoeffsCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cache: [CacheKey: Block1DCoeffs] = [:]
    private let maxEntries: Int

    public init(maxEntries: Int = 100) {
        self.maxEntries = maxEntries
    }

    /// Synchronous cache lookup and computation
    public func getOrCompute(
        profiles: CoreProfiles,
        geometry: Geometry,
        compute: CoeffsCallback
    ) -> Block1DCoeffs {
        let key = CacheKey(profiles: profiles, geometry: geometry)

        lock.lock()
        defer { lock.unlock() }

        if let cached = cache[key] {
            return cached
        }

        // Compute while holding lock (prevents duplicate work)
        let result = compute(profiles, geometry)

        if cache.count >= maxEntries {
            cache.removeFirst()
        }
        cache[key] = result

        return result
    }
}

// Usage in solver
let coeffs = cache.getOrCompute(
    profiles: currentProfiles,
    geometry: geometry
) { profiles, geo in
    // Synchronous computation
    combineCoefficients(
        transport: transportModel.computeCoefficients(profiles, geo, params),
        sources: sourceModel.computeTerms(profiles, geo, params)
    )
}
```

---

## When to Use Each Approach

| Type | Sendable Status | Use Case | Example |
|------|----------------|----------|---------|
| `EvaluatedArray` | `@unchecked Sendable` | Evaluated MLXArray wrapper | Core infrastructure |
| Structs with `EvaluatedArray` | Pure `Sendable` | Data structures | CoreProfiles, TransportCoefficients |
| Pure `Sendable` | Pure `Sendable` | Configuration, I/O | SimulationConfig, SerializableProfiles |
| Actor | Thread-safe reference | Mutable state management | SimulationOrchestrator |
| Synchronous cache | `@unchecked Sendable` with locks | CoeffsCache | CoeffsCache |

---

## Summary

✅ **DO**: Use `EvaluatedArray` wrapper for all MLXArray data crossing actor boundaries
✅ **DO**: Keep computation chains as MLXArray for grad() and compile()
✅ **DO**: Use pure functions for compile() to avoid actor isolation issues
✅ **DO**: Keep MLX operations synchronous (no unnecessary async)
✅ **DO**: Use batch evaluation for efficiency

❌ **DON'T**: Convert MLXArray to Swift arrays during computation
❌ **DON'T**: Capture actor self in compile() closures
❌ **DON'T**: Make MLX operations async
❌ **DON'T**: Mark individual data structures as @unchecked Sendable (use EvaluatedArray wrapper)

---

*See also:*
- [MLX_BEST_PRACTICES.md](MLX_BEST_PRACTICES.md) for lazy evaluation patterns
- [NUMERICAL_PRECISION.md](NUMERICAL_PRECISION.md) for GPU-first design
- [CLAUDE.md](../CLAUDE.md) for development guidelines
