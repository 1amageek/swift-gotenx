 # FVM Numerical Improvements Plan

**Date**: 2025-10-21
**Status**: Design & Implementation Roadmap
**Priority**: P0 - Critical for Physics Accuracy
**Target**: TORAX-equivalent numerical fidelity

---

## Executive Summary

This document consolidates two critical reviews with implementation-ready solutions:
1. **FVM Implementation Gaps**: Missing power-law scheme, simplified bootstrap current, uniform-grid assumptions
2. **Hardcoded Tolerances**: Magic number `1e-6` used inconsistently across 15+ locations

**Key Goals**:
- ✅ Implement TORAX-compliant power-law scheme for convection stability
- ✅ Upgrade bootstrap current to Sauter formula with collisionality dependence
- ✅ Support non-uniform grids with proper metric tensors
- ✅ Centralize all numerical tolerances in configuration system with proper scaling
- ✅ Establish comprehensive integration tests using Swift Testing

**Total Effort Estimate**: 45-55 hours (~1 week)

---

## Design Principles

### 1. Residual Scaling and Tolerance Comparison

**Critical**: The solver scales residuals to O(1) for uniform precision. Tolerances must be scaled identically:

```
residual_scaled = residual_physical / reference
tolerance_scaled = tolerance_physical / reference
converged = residual_scaled < tolerance_scaled
```

This is implemented via `ToleranceScaler` that applies the same reference state scaling to both residuals and tolerances.

### 2. Configuration Backward Compatibility

**Preserve existing structure**: Extend `AdaptiveTimestepConfig` rather than replacing it. Old configs with explicit `minimumTimeStep` continue working; new configs can use `minimumTimeStepFraction` for adaptive scaling.

### 3. Physical Correctness

**Bootstrap current sign**: Bootstrap current can be negative at the plasma edge (counter-current drive). Clamp **magnitude only**, preserve sign for physical accuracy.

### 4. Testing Framework

**Swift Testing standard**: All tests use `@Suite`, `@Test`, and `#expect` (not XCTest).

### 5. MLX API Compliance

**Use actual APIs**: `MLX.select()` for conditionals (not non-existent `expandedDimensions` or bare `where`).

---

## Configuration System Design

### 1. Per-Equation Tolerances with Scaling

**File**: `Sources/Gotenx/Configuration/NumericalTolerances.swift` (NEW)

```swift
/// Per-equation numerical tolerances
public struct EquationTolerances: Codable, Sendable {
    /// Absolute tolerance for residual norm [physical units]
    public let absoluteTolerance: Float

    /// Relative tolerance for residual norm [dimensionless]
    public let relativeTolerance: Float

    /// Minimum value threshold (below this, use absolute tolerance only)
    public let minValueThreshold: Float

    /// Compute combined tolerance for state value x
    /// tol = max(absoluteTolerance, relativeTolerance * |x|)
    public func combinedTolerance(for value: Float) -> Float {
        if abs(value) < minValueThreshold {
            return absoluteTolerance
        }
        return max(absoluteTolerance, relativeTolerance * abs(value))
    }
}

/// Numerical tolerance configuration for all equations
public struct NumericalTolerances: Codable, Sendable {
    public let ionTemperature: EquationTolerances
    public let electronTemperature: EquationTolerances
    public let electronDensity: EquationTolerances
    public let poloidalFlux: EquationTolerances

    /// Default ITER-scale tolerances
    public static let iterScale = NumericalTolerances(
        ionTemperature: EquationTolerances(
            absoluteTolerance: 10.0,        // 10 eV absolute
            relativeTolerance: 1e-4,        // 0.01% relative
            minValueThreshold: 100.0        // Below 100 eV, use absolute only
        ),
        electronTemperature: EquationTolerances(
            absoluteTolerance: 10.0,
            relativeTolerance: 1e-4,
            minValueThreshold: 100.0
        ),
        electronDensity: EquationTolerances(
            absoluteTolerance: 1e17,        // 1e17 m⁻³ absolute
            relativeTolerance: 1e-4,
            minValueThreshold: 1e18
        ),
        poloidalFlux: EquationTolerances(
            absoluteTolerance: 1e-3,        // 1 mWb absolute
            relativeTolerance: 1e-5,
            minValueThreshold: 0.1
        )
    )
}

/// Tolerance scaler for residual-space convergence checks
public struct ToleranceScaler {
    let referenceState: FlattenedState
    let tolerances: NumericalTolerances

    /// Compute scaled tolerance for each equation
    ///
    /// **Critical**: Residuals are scaled to O(1) in solver, so we must
    /// scale the physical tolerances by the same reference state:
    ///
    /// ```
    /// residual_scaled = residual_physical / reference
    /// tolerance_scaled = tolerance_physical / reference
    /// converged = residual_scaled < tolerance_scaled
    /// ```
    ///
    /// - Parameters:
    ///   - layout: State layout (equation ranges)
    ///   - physicalState: Current state in physical units (for relative tolerance)
    /// - Returns: Scaled tolerance vector [4*cellCount]
    public func scaledTolerances(
        layout: StateLayout,
        physicalState: FlattenedState
    ) -> MLXArray {
        let cellCount = layout.cellCount

        // Extract per-equation physical values and reference scales
        let Ti_phys = physicalState.values.value[layout.tiRange]
        let Te_phys = physicalState.values.value[layout.teRange]
        let ne_phys = physicalState.values.value[layout.neRange]
        let psi_phys = physicalState.values.value[layout.psiRange]

        let Ti_ref = referenceState.values.value[layout.tiRange]
        let Te_ref = referenceState.values.value[layout.teRange]
        let ne_ref = referenceState.values.value[layout.neRange]
        let psi_ref = referenceState.values.value[layout.psiRange]

        // Compute per-equation tolerances
        let tol_Ti = computeScaledTolerance(
            values: Ti_phys,
            reference: Ti_ref,
            eqTol: tolerances.ionTemperature
        )
        let tol_Te = computeScaledTolerance(
            values: Te_phys,
            reference: Te_ref,
            eqTol: tolerances.electronTemperature
        )
        let tol_ne = computeScaledTolerance(
            values: ne_phys,
            reference: ne_ref,
            eqTol: tolerances.electronDensity
        )
        let tol_psi = computeScaledTolerance(
            values: psi_phys,
            reference: psi_ref,
            eqTol: tolerances.poloidalFlux
        )

        return concatenated([tol_Ti, tol_Te, tol_ne, tol_psi], axis: 0)
    }

    /// Compute scaled tolerance for single equation
    private func computeScaledTolerance(
        values: MLXArray,
        reference: MLXArray,
        eqTol: EquationTolerances
    ) -> MLXArray {
        // Combined physical tolerance: max(abs, rel * |x|)
        // Vectorized over all cells
        let absTol = MLXArray(eqTol.absoluteTolerance)
        let relTol = eqTol.relativeTolerance * abs(values)
        let physicalTol = maximum(absTol, relTol)

        // Scale to residual space: tol_scaled = tol_phys / reference
        return physicalTol / (reference + 1e-30)
    }
}
```

### 2. Physical Thresholds Configuration

**File**: `Sources/Gotenx/Configuration/PhysicalThresholds.swift` (NEW)

```swift
/// Physical quantity thresholds (scaled to problem)
public struct PhysicalThresholds: Codable, Sendable {
    /// Fusion fuel fraction sum tolerance (default: 1e-4)
    /// Physical tolerance: 0.01% is reasonable for fraction sums
    public let fuelFractionTolerance: Float

    /// Minimum fusion power for Q calculation [MW] (default: 1e-3)
    /// Below 1 kW, fusion gain Q is meaningless
    public let minFusionPowerForQ: Float

    /// Minimum heating power for τE calculation [MW] (default: 1e-2)
    /// Below 10 kW, energy confinement time is unreliable
    public let minHeatingPowerForTauE: Float

    /// Poloidal flux relative variation threshold (default: 1e-5)
    /// Skip Ohmic heating if dψ/ψ < threshold
    public let fluxVariationThreshold: Float

    /// Minimum stored energy for diagnostics [MJ] (default: 1e-3)
    /// Below 1 kJ, plasma is negligible
    public let minStoredEnergy: Float

    public static let `default` = PhysicalThresholds(
        fuelFractionTolerance: 1e-4,      // 0.01% (was 1e-6, too strict)
        minFusionPowerForQ: 1e-3,         // 1 kW (was 1e-6 MW, unrealistic)
        minHeatingPowerForTauE: 1e-2,     // 10 kW (was implicit 1e-6)
        fluxVariationThreshold: 1e-5,     // 0.001% flux change (was 1e-6)
        minStoredEnergy: 1e-3             // 1 kJ (was 1e-6 MJ, too small)
    )
}
```

### 3. Time Configuration Extension

**File**: `Sources/Gotenx/Configuration/TimeConfiguration.swift` (MODIFY)

```swift
/// Adaptive timestep configuration (EXTENDED, backward compatible)
public struct AdaptiveTimestepConfig: Codable, Sendable, Equatable {
    /// Minimum timestep [s] (absolute) - optional for backward compat
    public let minimumTimeStep: Float?

    /// Minimum timestep fraction of maximumTimeStep (default: 0.001)
    /// Ignored if minimumTimeStep is explicitly set
    public let minimumTimeStepFraction: Float?

    /// Maximum timestep [s]
    public let maximumTimeStep: Float

    /// CFL safety factor (< 1.0)
    public let safetyFactor: Float

    /// Maximum timestep growth rate per step (default: 1.2)
    public let maximumTimeStepGrowth: Float

    /// Computed minimum timestep (backward compatible)
    public var effectiveMinDt: Float {
        if let minimumTimeStep = minimumTimeStep {
            return minimumTimeStep  // Explicit value takes precedence (old configs)
        } else if let fraction = minimumTimeStepFraction {
            return maximumTimeStep * fraction
        } else {
            return maximumTimeStep * 0.001  // Default fallback
        }
    }

    public static let `default` = AdaptiveTimestepConfig(
        minimumTimeStep: nil,              // Use fraction instead
        minimumTimeStepFraction: 0.001,    // maximumTimeStep / 1000
        maximumTimeStep: 1e-1,
        safetyFactor: 0.9,
        maximumTimeStepGrowth: 1.2
    )

    public init(
        minimumTimeStep: Float? = nil,
        minimumTimeStepFraction: Float? = 0.001,
        maximumTimeStep: Float,
        safetyFactor: Float,
        maximumTimeStepGrowth: Float = 1.2
    ) {
        self.minimumTimeStep = minimumTimeStep
        self.minimumTimeStepFraction = minimumTimeStepFraction
        self.maximumTimeStep = maximumTimeStep
        self.safetyFactor = safetyFactor
        self.maximumTimeStepGrowth = maximumTimeStepGrowth
    }
}
```

**Migration Example**:
```json
// Old config (still works):
{
  "adaptive": {
    "minimumTimeStep": 1e-6,
    "maximumTimeStep": 1e-1,
    "safetyFactor": 0.9
  }
}

// New config (recommended):
{
  "adaptive": {
    "minimumTimeStepFraction": 0.001,
    "maximumTimeStep": 1e-1,
    "safetyFactor": 0.9,
    "maximumTimeStepGrowth": 1.2
  }
}
```

---

## Implementation Plan

### Phase 1: Configuration System Refactor (P0, 10-12 hours)

#### 1.1 Create Configuration Files (2 hours)
- `Sources/Gotenx/Configuration/NumericalTolerances.swift` (with `ToleranceScaler`)
- `Sources/Gotenx/Configuration/PhysicalThresholds.swift`

#### 1.2 Extend AdaptiveTimestepConfig (2 hours)

Modify `Sources/Gotenx/Configuration/TimeConfiguration.swift` as shown above.

#### 1.3 Update NewtonRaphsonSolver (4 hours)

**File**: `Sources/Gotenx/Solver/NewtonRaphsonSolver.swift` (MODIFY)

```swift
public struct NewtonRaphsonSolver: PDESolver {
    /// Per-equation tolerances (replaces single tolerance: Float)
    public let tolerances: NumericalTolerances

    public func solve(...) -> SolverResult {
        // ... existing setup ...

        // Create tolerance scaler
        let toleranceScaler = ToleranceScaler(
            referenceState: referenceState,
            tolerances: tolerances
        )

        // Newton-Raphson iteration in SCALED space
        for iter in 0..<maximumIterations {
            // Compute residual in scaled space
            let residualScaled = residualFnScaled(xScaled.values.value)
            eval(residualScaled)

            // Compute scaled tolerances for current state
            let xPhysical = xScaled.unscaled(by: referenceState)
            let scaledTols = toleranceScaler.scaledTolerances(
                layout: layout,
                physicalState: xPhysical
            )
            eval(scaledTols)

            // Per-equation convergence check
            let R_Ti = residualScaled[layout.tiRange]
            let R_Te = residualScaled[layout.teRange]
            let R_ne = residualScaled[layout.neRange]
            let R_psi = residualScaled[layout.psiRange]

            let tol_Ti = scaledTols[layout.tiRange]
            let tol_Te = scaledTols[layout.teRange]
            let tol_ne = scaledTols[layout.neRange]
            let tol_psi = scaledTols[layout.psiRange]

            // RMS residual norm for each equation
            let Ti_norm = sqrt((R_Ti * R_Ti).mean())
            let Te_norm = sqrt((R_Te * R_Te).mean())
            let ne_norm = sqrt((R_ne * R_ne).mean())
            let psi_norm = sqrt((R_psi * R_psi).mean())

            // RMS tolerance for each equation
            let Ti_tol = sqrt((tol_Ti * tol_Ti).mean())
            let Te_tol = sqrt((tol_Te * tol_Te).mean())
            let ne_tol = sqrt((tol_ne * tol_ne).mean())
            let psi_tol = sqrt((tol_psi * tol_psi).mean())

            eval(Ti_norm, Te_norm, ne_norm, psi_norm)
            eval(Ti_tol, Te_tol, ne_tol, psi_tol)

            // Check convergence for each equation
            let Ti_converged = Ti_norm.item(Float.self) < Ti_tol.item(Float.self)
            let Te_converged = Te_norm.item(Float.self) < Te_tol.item(Float.self)
            let ne_converged = ne_norm.item(Float.self) < ne_tol.item(Float.self)
            let psi_converged = psi_norm.item(Float.self) < psi_tol.item(Float.self)

            let converged = Ti_converged && Te_converged && ne_converged && psi_converged

            if converged {
                break
            }

            // ... rest of Newton iteration ...
        }

        // Store per-equation convergence info
        metadata["Ti_residual"] = Ti_norm.item(Float.self)
        metadata["Te_residual"] = Te_norm.item(Float.self)
        metadata["ne_residual"] = ne_norm.item(Float.self)
        metadata["psi_residual"] = psi_norm.item(Float.self)
    }
}
```

#### 1.4 Update Physical Modules (2 hours)

Replace hardcoded `1e-6` with `PhysicalThresholds`:

**OhmicHeating.swift** (Line 307):
```swift
// OLD: if fluxVariation < 1e-6
// NEW:
if fluxVariation < (fluxRange * thresholds.fluxVariationThreshold) {
    return zero
}
```

**FusionPower.swift** (Line 99):
```swift
// OLD: guard sum > 1e-6
// NEW:
guard abs(sum - 1.0) < thresholds.fuelFractionTolerance else {
    throw FusionPowerError.invalidFuelFractionSum
}
```

**DerivedQuantitiesComputer.swift**:
```swift
// Line 337 - Energy confinement time
if totalHeatingPower < thresholds.minHeatingPowerForTauE {
    return 0.0
}

// Line 579 - Fusion gain Q
if fusionPower < thresholds.minFusionPowerForQ {
    return 0.0
}
```

---

### Phase 2: Power-Law Scheme Implementation (P0, 7-9 hours)

#### 2.1 Create Power-Law Module (4 hours)

**File**: `Sources/Gotenx/FVM/PowerLawScheme.swift` (NEW)

```swift
import MLX

/// Patankar power-law scheme for convection-diffusion face weighting
///
/// **Physics**: High Péclet number (Pe = V·Δx/D >> 1) causes numerical oscillations
/// with central differencing. Power-law scheme provides smooth transition:
///
/// - Pe < 0.1: Central differencing (2nd order accurate)
/// - 0.1 ≤ Pe ≤ 10: Power-law interpolation
/// - Pe > 10: First-order upwinding (stable but diffusive)
///
/// **References**:
/// - Patankar, S.V. (1980). "Numerical Heat Transfer and Fluid Flow"
/// - TORAX: arXiv:2406.06718v2, Section 2.2.3
public struct PowerLawScheme {

    /// Compute Péclet number at faces
    ///
    /// Pe = V·Δx / D
    ///
    /// - Parameters:
    ///   - vFace: Convection velocity at faces [m/s], shape [nFaces]
    ///   - dFace: Diffusion coefficient at faces [m²/s], shape [nFaces]
    ///   - dx: Cell spacing [m], shape [nFaces-1] or scalar
    /// - Returns: Péclet number [dimensionless], shape [nFaces]
    public static func computePecletNumber(
        vFace: MLXArray,
        dFace: MLXArray,
        dx: MLXArray
    ) -> MLXArray {
        let dFace_safe = dFace + 1e-30

        // Broadcast dx to [nFaces] if needed
        let dx_broadcast: MLXArray
        if dx.ndim == 0 {
            // Scalar: create full array
            dx_broadcast = MLXArray.full([vFace.shape[0]], values: dx)
        } else if dx.shape[0] == vFace.shape[0] - 2 {
            // dx is [nFaces-1] (interior only): pad boundaries
            let dx_left = dx[0..<1]
            let dx_right = dx[(dx.shape[0]-1)..<dx.shape[0]]
            dx_broadcast = concatenated([dx_left, dx, dx_right], axis: 0)
        } else {
            // Already correct size
            dx_broadcast = dx
        }

        return vFace * dx_broadcast / dFace_safe
    }

    /// Compute power-law weighting factor α for face interpolation
    ///
    /// Face value: x_face = α·x_upwind + (1-α)·x_downwind
    ///
    /// **Patankar formula**:
    /// ```
    /// α(Pe) = max(0, (1 - 0.1·|Pe|)^5)  for |Pe| ≤ 10
    /// α(Pe) = 0 (full upwinding)        for |Pe| > 10
    /// ```
    ///
    /// - Parameter peclet: Péclet number [dimensionless], shape [nFaces]
    /// - Returns: Weighting factor α ∈ [0,1], shape [nFaces]
    public static func computeWeightingFactor(peclet: MLXArray) -> MLXArray {
        let absPe = abs(peclet)

        // Power-law formula: (1 - 0.1*|Pe|)^5
        let clamped = maximum(0.0, 1.0 - 0.1 * absPe)
        let powerLaw = pow(clamped, 5.0)

        // For |Pe| > 10: full upwinding
        // Use MLX.select() (actual API, not non-existent where())
        let alpha = MLX.select(absPe > 10.0, MLXArray(0.0), powerLaw)

        return alpha
    }

    /// Compute face values using power-law weighting
    ///
    /// - Parameters:
    ///   - cellValues: Values at cell centers [cellCount]
    ///   - peclet: Péclet number at faces [nFaces]
    /// - Returns: Weighted face values [nFaces]
    public static func interpolateToFaces(
        cellValues: MLXArray,
        peclet: MLXArray
    ) -> MLXArray {
        let cellCount = cellValues.shape[0]

        // Interior faces: power-law weighted
        let leftCells = cellValues[0..<(cellCount-1)]
        let rightCells = cellValues[1..<cellCount]
        let pecletInterior = peclet[1..<(peclet.shape[0]-1)]

        let alpha = computeWeightingFactor(peclet: pecletInterior)

        // Upwind selection based on flow direction
        // Use MLX.select() (actual MLX API)
        let upwindValues = MLX.select(
            pecletInterior > 0,  // condition
            leftCells,           // if Pe > 0: upwind = left
            rightCells          // if Pe < 0: upwind = right
        )
        let downwindValues = MLX.select(
            pecletInterior > 0,
            rightCells,          // if Pe > 0: downwind = right
            leftCells           // if Pe < 0: downwind = left
        )

        let faceInterior = alpha * upwindValues + (1.0 - alpha) * downwindValues

        // Boundary faces: use adjacent cell value
        let faceLeft = cellValues[0..<1]
        let faceRight = cellValues[(cellCount-1)..<cellCount]

        return concatenated([faceLeft, faceInterior, faceRight], axis: 0)
    }
}
```

#### 2.2 Integrate into NewtonRaphsonSolver (2 hours)

**File**: `Sources/Gotenx/Solver/NewtonRaphsonSolver.swift` (MODIFY)

Update `interpolateToFacesVectorized`:
```swift
/// Interpolate cell values to faces using power-law scheme
private func interpolateToFacesVectorized(
    _ u: MLXArray,
    vFace: MLXArray,
    dFace: MLXArray,
    dx: MLXArray
) -> MLXArray {
    let peclet = PowerLawScheme.computePecletNumber(
        vFace: vFace,
        dFace: dFace,
        dx: dx
    )

    return PowerLawScheme.interpolateToFaces(
        cellValues: u,
        peclet: peclet
    )
}
```

Update call site in `applySpatialOperatorVectorized` (Line 368):
```swift
// OLD: let u_face = interpolateToFacesVectorized(u)
// NEW:
let u_face = interpolateToFacesVectorized(
    u,
    vFace: vFace,
    dFace: dFace,
    dx: geometry.cellDistances.value
)
```

#### 2.3 Add Tests (3 hours)

**File**: `Tests/GotenxTests/FVM/PowerLawSchemeTests.swift` (NEW)

```swift
import Testing
import MLX
@testable import GotenxCore

@Suite("Power-Law Scheme Tests")
struct PowerLawSchemeTests {

    @Test("Péclet number calculation")
    func pecletNumber() {
        let vFace = MLXArray([0.0, 1.0, 10.0, 100.0])
        let dFace = MLXArray([1.0, 1.0, 1.0, 1.0])
        let dx = MLXArray(1.0)

        let peclet = PowerLawScheme.computePecletNumber(
            vFace: vFace,
            dFace: dFace,
            dx: dx
        )
        eval(peclet)

        let result = peclet.asArray(Float.self)
        #expect(abs(result[0] - 0.0) < 1e-6)
        #expect(abs(result[1] - 1.0) < 1e-6)
        #expect(abs(result[2] - 10.0) < 1e-6)
        #expect(abs(result[3] - 100.0) < 1e-6)
    }

    @Test("Power-law weighting for different Péclet numbers")
    func weightingFactor() {
        let peclet = MLXArray([0.0, 1.0, 5.0, 10.0, 50.0, -10.0])
        let alpha = PowerLawScheme.computeWeightingFactor(peclet: peclet)
        eval(alpha)

        let result = alpha.asArray(Float.self)

        // Pe = 0: central → α ≈ 1
        #expect(abs(result[0] - 1.0) < 1e-5)

        // Pe = 1: α = (1 - 0.1)^5 = 0.59049
        #expect(abs(result[1] - 0.59049) < 1e-4)

        // Pe = 5: α = (1 - 0.5)^5 = 0.03125
        #expect(abs(result[2] - 0.03125) < 1e-4)

        // Pe = 10: α = 0
        #expect(abs(result[3]) < 1e-5)

        // |Pe| > 10: upwinding → α = 0
        #expect(abs(result[4]) < 1e-5)
        #expect(abs(result[5]) < 1e-5)
    }

    @Test("Upwind selection for positive/negative Pe")
    func upwindSelection() {
        let cellValues = MLXArray([1.0, 2.0, 3.0, 4.0])

        // Positive Pe: upwind from left
        let pecletPos = MLXArray([0.0, 15.0, 15.0, 15.0, 0.0])
        let facePos = PowerLawScheme.interpolateToFaces(
            cellValues: cellValues,
            peclet: pecletPos
        )
        eval(facePos)

        let resultPos = facePos.asArray(Float.self)
        #expect(abs(resultPos[1] - 1.0) < 1e-5)  // Left upwind
        #expect(abs(resultPos[2] - 2.0) < 1e-5)
        #expect(abs(resultPos[3] - 3.0) < 1e-5)

        // Negative Pe: upwind from right
        let pecletNeg = MLXArray([0.0, -15.0, -15.0, -15.0, 0.0])
        let faceNeg = PowerLawScheme.interpolateToFaces(
            cellValues: cellValues,
            peclet: pecletNeg
        )
        eval(faceNeg)

        let resultNeg = faceNeg.asArray(Float.self)
        #expect(abs(resultNeg[1] - 2.0) < 1e-5)  // Right upwind
        #expect(abs(resultNeg[2] - 3.0) < 1e-5)
        #expect(abs(resultNeg[3] - 4.0) < 1e-5)
    }
}
```

---

### Phase 3: Sauter Bootstrap Current (P0, 9-11 hours)

#### 3.1 Implement Collisionality Calculation (4 hours)

**File**: `Sources/Gotenx/Solver/CollisionalityHelpers.swift` (NEW)

```swift
import MLX

/// Collisionality and neoclassical transport helpers
///
/// **References**:
/// - Sauter et al., "Neoclassical conductivity and bootstrap current", PoP 6, 2834 (1999)
/// - Wesson, "Tokamak Physics" (2nd ed.), Chapter 7
public struct CollisionalityHelpers {

    /// Compute electron-ion collision time τₑ [s]
    ///
    /// Formula: τₑ ≈ 3.44e5 * Tₑ^(3/2) / (nₑ * ln(Λ))
    ///
    /// - Parameters:
    ///   - Te: Electron temperature [eV], shape [cellCount]
    ///   - ne: Electron density [m⁻³], shape [cellCount]
    ///   - coulombLog: Coulomb logarithm (default: 17.0)
    /// - Returns: Collision time [s], shape [cellCount]
    public static func computeCollisionTime(
        Te: MLXArray,
        ne: MLXArray,
        coulombLog: Float = 17.0
    ) -> MLXArray {
        return 3.44e5 * pow(Te, 1.5) / (ne * coulombLog)
    }

    /// Compute normalized collisionality ν*
    ///
    /// Formula: ν* = (R₀ q) / (ε^(3/2) vₜₕ τₑ)
    ///
    /// Where:
    /// - ε = r/R₀ (inverse aspect ratio)
    /// - q: safety factor
    /// - vₜₕ = √(2Tₑ/mₑ): thermal velocity
    ///
    /// - Parameters:
    ///   - Te: Electron temperature [eV], shape [cellCount]
    ///   - ne: Electron density [m⁻³], shape [cellCount]
    ///   - geometry: Tokamak geometry
    /// - Returns: Normalized collisionality ν* [dimensionless], shape [cellCount]
    public static func computeNormalizedCollisionality(
        Te: MLXArray,
        ne: MLXArray,
        geometry: Geometry
    ) -> MLXArray {
        let tau_e = computeCollisionTime(Te: Te, ne: ne)

        let epsilon = geometry.radii.value / geometry.majorRadius
        let q = approximateSafetyFactor(geometry: geometry)

        // Thermal velocity: vₜₕ = √(2Tₑ/mₑ)
        // With Tₑ in eV: vₜₕ = √(3.514e11 * Tₑ)  [m/s]
        let vth = sqrt(3.514e11 * Te)

        let nu_star = (geometry.majorRadius * q) / (pow(epsilon, 1.5) * vth * tau_e)

        return nu_star
    }

    /// Approximate safety factor q from geometry
    ///
    /// Parabolic approximation: q ≈ 1 + (r/a)²
    ///
    /// - Parameter geometry: Tokamak geometry
    /// - Returns: Safety factor [dimensionless], shape [cellCount]
    private static func approximateSafetyFactor(geometry: Geometry) -> MLXArray {
        let r_norm = geometry.radii.value / geometry.minorRadius
        return 1.0 + r_norm * r_norm
    }
}
```

#### 3.2 Implement Sauter Formula (4 hours)

**File**: `Sources/Gotenx/Solver/Block1DCoeffsBuilder.swift` (MODIFY)

Replace `computeBootstrapCurrent` (Line 474-507):

```swift
/// Compute bootstrap current using Sauter neoclassical formula
///
/// **Full Sauter Formula**:
/// ```
/// J_BS = -C_BS(ν*, ft, ε) · (∇P / B_φ)
/// where C_BS = L₃₁·fₜ + L₃₂·fₜ·α + L₃₄·fₜ·α²
/// ```
///
/// **Critical**: Preserves sign (can be negative at edge for counter-current drive)
///
/// **References**:
/// - Sauter et al., PoP 6, 2834 (1999), Eqs. 13-14, Table I
///
/// - Parameters:
///   - profiles: Current core profiles
///   - geometry: Tokamak geometry
/// - Returns: Bootstrap current density [A/m²], shape [cellCount]
private func computeBootstrapCurrent(
    profiles: CoreProfiles,
    geometry: Geometry
) -> MLXArray {
    let Ti = profiles.ionTemperature.value
    let Te = profiles.electronTemperature.value
    let ne = profiles.electronDensity.value

    // 1. Total pressure: P = n_e (T_i + T_e) * e
    let P = ne * (Ti + Te) * UnitConversions.eV  // [Pa]

    // 2. Pressure gradient: ∇P [Pa/m]
    let geoFactors = GeometricFactors.from(geometry: geometry)
    let gradP = computeGradient(P, cellDistances: geoFactors.cellDistances.value)

    // 3. Normalized collisionality ν*
    let nu_star = CollisionalityHelpers.computeNormalizedCollisionality(
        Te: Te,
        ne: ne,
        geometry: geometry
    )

    // 4. Trapped particle fraction
    let epsilon = geometry.radii.value / geometry.majorRadius
    let ft = 1.0 - sqrt(maximum(1e-10, 1.0 - epsilon))

    // 5. Sauter coefficients L₃₁, L₃₂, L₃₄
    let L31 = computeSauterL31(nu_star: nu_star, ft: ft)
    let L32 = computeSauterL32(nu_star: nu_star, ft: ft)
    let L34 = computeSauterL34(nu_star: nu_star, ft: ft)

    // 6. Pressure anisotropy parameter α (assume isotropic: α = 0)
    let alpha = MLXArray.zeros(like: Te)

    // 7. Bootstrap coefficient: C_BS = L₃₁·ft + L₃₂·ft·α + L₃₄·ft·α²
    let C_BS = L31 * ft + L32 * ft * alpha + L34 * ft * alpha * alpha

    // 8. Bootstrap current: J_BS = -C_BS · (∇P / B_φ)
    let J_BS = -C_BS * gradP / geometry.toroidalField

    // 9. Clamp MAGNITUDE only, preserve sign (CORRECTED)
    // Bootstrap current can be negative at edge (counter-current drive)
    let J_BS_magnitude = abs(J_BS)
    let J_BS_clamped_magnitude = minimum(J_BS_magnitude, MLXArray(1e7))  // Max 10 MA/m²
    let J_BS_final = sign(J_BS) * J_BS_clamped_magnitude

    return J_BS_final
}

/// Compute Sauter L₃₁ coefficient (bootstrap current, main term)
///
/// **Formula** (Sauter Table I, simplified):
/// ```
/// L₃₁(ν*, ft) = ((1 + 0.15/ft) - 0.22/(1 + 0.01·ν*)) / (1 + 0.5·√ν*)
/// ```
///
/// - Parameters:
///   - nu_star: Normalized collisionality [dimensionless]
///   - ft: Trapped fraction [dimensionless]
/// - Returns: L₃₁ coefficient [dimensionless]
private func computeSauterL31(nu_star: MLXArray, ft: MLXArray) -> MLXArray {
    let ft_safe = ft + 1e-10
    let nu_safe = nu_star + 1e-10

    let numerator = (1.0 + 0.15 / ft_safe) - 0.22 / (1.0 + 0.01 * nu_safe)
    let denominator = 1.0 + 0.5 * sqrt(nu_safe)

    return numerator / denominator
}

/// Compute Sauter L₃₂ coefficient (pressure anisotropy correction)
private func computeSauterL32(nu_star: MLXArray, ft: MLXArray) -> MLXArray {
    // Simplified: L₃₂ ≈ 0.05
    return MLXArray.full(nu_star.shape, values: MLXArray(0.05))
}

/// Compute Sauter L₃₄ coefficient (second-order pressure anisotropy)
private func computeSauterL34(nu_star: MLXArray, ft: MLXArray) -> MLXArray {
    // Simplified: L₃₄ ≈ 0.01
    return MLXArray.full(nu_star.shape, values: MLXArray(0.01))
}
```

#### 3.3 Add Tests (3 hours)

**File**: `Tests/GotenxTests/Solver/BootstrapCurrentTests.swift` (NEW)

```swift
import Testing
import MLX
@testable import GotenxCore

@Suite("Bootstrap Current Tests")
struct BootstrapCurrentTests {

    @Test("Collision time calculation against reference")
    func collisionTime() {
        // ITER typical core: Te = 10 keV, ne = 1e20 m⁻³
        let Te = MLXArray([10000.0])
        let ne = MLXArray([1e20])

        let tau_e = CollisionalityHelpers.computeCollisionTime(
            Te: Te,
            ne: ne,
            coulombLog: 17.0
        )
        eval(tau_e)

        // Expected: τₑ ≈ 2.02e-6 s
        let result = tau_e.item(Float.self)
        let expected: Float = 2.02e-6
        #expect(abs(result - expected) < expected * 0.1)
    }

    @Test("Sauter L₃₁ coefficient - banana regime")
    func sauterL31BananaRegime() {
        // Low collisionality: ν* = 0.01, ft = 0.3
        let nu_star = MLXArray([0.01])
        let ft = MLXArray([0.3])

        let L31 = computeSauterL31(nu_star: nu_star, ft: ft)
        eval(L31)

        let result = L31.item(Float.self)
        // Banana regime: L₃₁ ≈ 1.5 (from Sauter Table I)
        #expect(abs(result - 1.5) < 0.2)
    }

    @Test("Sauter L₃₁ coefficient - plateau regime")
    func sauterL31PlateauRegime() {
        // Moderate collisionality: ν* = 1.0, ft = 0.3
        let nu_star = MLXArray([1.0])
        let ft = MLXArray([0.3])

        let L31 = computeSauterL31(nu_star: nu_star, ft: ft)
        eval(L31)

        let result = L31.item(Float.self)
        // Plateau regime: L₃₁ ≈ 0.8
        #expect(abs(result - 0.8) < 0.2)
    }

    @Test("Bootstrap current preserves sign")
    func bootstrapSignPreservation() {
        // Create scenario with negative gradient (edge)
        // Bootstrap current should be negative (counter-current)

        // TODO: Set up profiles with outward pressure gradient
        // Verify J_BS < 0 at edge (no artificial clamp to zero)
    }
}
```

---

### Phase 4: Non-Uniform Grid Support (P1, 6-8 hours)

#### 4.1 Enhance GeometricFactors (4 hours)

**File**: `Sources/Gotenx/Core/GeometricFactors.swift` (MODIFY)

Add metric tensor support:
```swift
public struct GeometricFactors: Sendable {
    // ... existing fields ...

    /// Metric tensor component g₀ = √g (Jacobian of flux coordinates)
    /// Shape: [cellCount]
    public let jacobian: EvaluatedArray

    /// Metric tensor component g₁
    /// Shape: [cellCount]
    public let g1: EvaluatedArray

    /// Metric tensor component g₂
    /// Shape: [cellCount]
    public let g2: EvaluatedArray

    /// Create from full geometry (use g0, g1, g2)
    public static func from(geometry: Geometry) -> GeometricFactors {
        let cellCount = geometry.cellCount

        // Use actual metric tensors from Geometry
        let jacobian = geometry.g0  // √g = F/B_p
        let g1 = geometry.g1
        let g2 = geometry.g2

        // Cell volumes: V = ∫ √g dr dθ dζ = 2π ∫ g₀ dr
        let dr = geometry.radii.value[1] - geometry.radii.value[0]
        let cellVolumes = jacobian.value * dr * Float(2 * .pi)

        // Face areas: A = 2π g₀(r_face)
        let jacobianFaces = interpolateToFaces(jacobian.value, mode: .arithmetic)
        let faceAreas = jacobianFaces * Float(2 * .pi)

        // ... rest of implementation ...

        return GeometricFactors(
            cellVolumes: EvaluatedArray(evaluating: cellVolumes),
            faceAreas: EvaluatedArray(evaluating: faceAreas),
            cellDistances: ...,
            jacobian: jacobian,
            g1: g1,
            g2: g2
        )
    }
}
```

#### 4.2 Update Spatial Operators (2 hours)

Modify `applySpatialOperatorVectorized` to use metric tensors for flux divergence:
```swift
// Flux divergence with metric tensors:
// ∇·F = (1/√g) ∂(√g·F)/∂ψ

let jacobianCells = geometry.jacobian.value
let jacobianFaces = interpolateToFaces(jacobianCells, mode: .arithmetic)
let jacobian_right = jacobianFaces[1..<(cellCount + 1)]
let jacobian_left = jacobianFaces[0..<cellCount]

let weightedFlux_right = jacobian_right * flux_right
let weightedFlux_left = jacobian_left * flux_left

let fluxDivergence = (weightedFlux_right - weightedFlux_left) /
                     (jacobianCells * geometry.cellDistances.value + 1e-10)
```

---

### Phase 5: Integration Testing (P2, 7-9 hours)

#### 5.1 Analytical Solution Tests (3 hours)

**File**: `Tests/GotenxTests/Integration/FVMAnalyticalTests.swift` (NEW)

```swift
import Testing
import MLX
@testable import GotenxCore

@Suite("FVM Analytical Solution Tests")
struct FVMAnalyticalTests {

    @Test("1D diffusion against analytical solution")
    func diffusionAnalytical() async throws {
        let chi: Float = 1.0    // [m²/s]
        let T0: Float = 1000.0  // [eV]
        let r0: Float = 0.5     // [m]
        let tFinal: Float = 1.0 // [s]

        // TODO: Set up simulation with D=χ, V=0, source=0
        // Run to t=tFinal, compare with analytical solution
        // T(r,t) = T₀ exp(-r²/(4χt)) / √(1 + 4χt/r₀²)
        // Expected error < 5% for cellCount=100
    }

    @Test("Steady-state convection-diffusion", arguments: [0.1, 1.0, 5.0, 10.0, 50.0])
    func convectionDiffusionSteadyState(peclet: Float) async throws {
        // PDE: V ∂T/∂r = χ ∂²T/∂r²
        // Analytical: T(r) = (exp(Pe·r/L) - 1) / (exp(Pe) - 1)

        // TODO: Set up V and χ to achieve target Pe
        // Run to steady state, compare with analytical
        // Verify power-law scheme prevents oscillations for Pe > 10
    }
}
```

#### 5.2 Conservation Tests (2 hours)

**File**: `Tests/GotenxTests/Integration/ConservationTests.swift` (NEW)

```swift
import Testing
import MLX
@testable import GotenxCore

@Suite("Conservation Tests")
struct ConservationIntegrationTests {

    @Test("Particle conservation over 100 timesteps")
    func particleConservation() async throws {
        // Initial particles: N₀ = ∫ n_e dV
        // After 100 steps with no source: |N - N₀| / N₀ < 1%

        // TODO: Run simulation with particle source = 0
        // Compute integrated particle number at each step
        // Verify drift < 1% over 100 steps
    }

    @Test("Energy conservation (no heating/loss)")
    func energyConservation() async throws {
        // Total energy: E = ∫ (3/2)nT dV
        // With Q_heat = 0, Q_loss = 0: dE/dt ≈ 0

        // TODO: Run simulation with all sources = 0
        // Verify |E(t=100Δt) - E(t=0)| / E(t=0) < 1%
    }

    @Test("Current conservation (bootstrap + Ohmic)")
    func currentConservation() async throws {
        // Total current: I_p = ∫ J dA
        // Verify I_p matches specified boundary condition

        // TODO: Set fixed current drive, run to steady state
        // Compare integrated current with specified value
    }
}
```

#### 5.3 TORAX Benchmark (2 hours)

**File**: `Tests/GotenxTests/Integration/TORAXBenchmarkTests.swift` (NEW)

```swift
import Testing
@testable import GotenxCore

@Suite("TORAX Benchmark Tests")
struct TORAXBenchmarkTests {

    @Test("ITER-like scenario comparison with Python TORAX")
    func iterScenario() async throws {
        // Load ITER_LIKE configuration
        // Run both Gotenx and TORAX (via subprocess or pre-computed reference)
        // Compare Ti, Te, ne, psi profiles at t = 1.0 s

        // Acceptance criteria:
        // - RMS error < 5% for all profiles
        // - Peak values within 3%
        // - Gradient scales within 10%
    }
}
```

---

## Validation Criteria

### Phase 1 (Configuration): ✅
- [ ] All 15 `1e-6` occurrences replaced with config values
- [ ] `ToleranceScaler` correctly computes scaled tolerances
- [ ] `AdaptiveTimestepConfig` extended (backward compatible)
- [ ] Old configs with explicit `minimumTimeStep` still work
- [ ] New configs with `minimumTimeStepFraction` work
- [ ] JSON schema validates

### Phase 2 (Power-Law): ✅
- [ ] `MLX.select()` used (not non-existent helpers)
- [ ] Pe ∈ [0.01, 100] tested
- [ ] No oscillations for Pe = 50
- [ ] All tests use Swift Testing (@Test, #expect)

### Phase 3 (Sauter Bootstrap): ✅
- [ ] Bootstrap current preserves sign
- [ ] Magnitude clamp only (|J_BS| < 10 MA/m²)
- [ ] Collisionality ν* matches references (±10%)
- [ ] L₃₁ coefficients match Sauter Table I (±20%)
- [ ] Tests use Swift Testing

### Phase 4 (Metrics): ✅
- [ ] Metric tensors g₀, g₁, g₂ propagated
- [ ] Shaped plasma (δ=0.4, κ=1.8): error < 5%
- [ ] Exponential grid: O(Δr²) convergence maintained
- [ ] Uniform grid results unchanged

### Phase 5 (Integration): ✅
- [ ] All tests use Swift Testing
- [ ] 1D diffusion: error < 5% vs. analytical
- [ ] Particle conservation: drift < 1% over 100 steps
- [ ] Energy conservation: drift < 1%
- [ ] TORAX benchmark: RMS error < 5%

---

## Configuration Migration Guide

### JSON Schema Update

**File**: `Examples/Configurations/iter_like_improved.json` (NEW)

```json
{
  "runtime": {
    "static": {
      "mesh": {
        "cellCount": 200
      },
      "solver": {
        "tolerances": {
          "ionTemperature": {
            "absoluteTolerance": 10.0,
            "relativeTolerance": 1e-4,
            "minValueThreshold": 100.0
          },
          "electronTemperature": {
            "absoluteTolerance": 10.0,
            "relativeTolerance": 1e-4,
            "minValueThreshold": 100.0
          },
          "electronDensity": {
            "absoluteTolerance": 1e17,
            "relativeTolerance": 1e-4,
            "minValueThreshold": 1e18
          },
          "poloidalFlux": {
            "absoluteTolerance": 1e-3,
            "relativeTolerance": 1e-5,
            "minValueThreshold": 0.1
          }
        },
        "physicalThresholds": {
          "fuelFractionTolerance": 1e-4,
          "minFusionPowerForQ": 1e-3,
          "minHeatingPowerForTauE": 1e-2,
          "fluxVariationThreshold": 1e-5,
          "minStoredEnergy": 1e-3
        },
        "maximumIterations": 30,
        "lineSearchEnabled": true
      },
      "time": {
        "start": 0.0,
        "end": 10.0,
        "initialTimeStep": 1e-3,
        "adaptive": {
          "minimumTimeStepFraction": 0.001,
          "maximumTimeStep": 0.1,
          "safetyFactor": 0.9,
          "maximumTimeStepGrowth": 1.2
        }
      }
    }
  }
}
```

### CLI Migration Path

**Backward Compatibility**: Old configs with `minimumTimeStep: 1e-6` will continue working without changes.

---

## Dependencies

### Package.swift (No Changes Needed)

Current dependencies already include:
- `swift-numerics` ✅
- `mlx-swift` ✅
- `swift-testing` ✅

**Verification**:
```bash
swift package describe
```

---

## Timeline Summary

| Phase | Priority | Effort | Deliverables |
|-------|----------|--------|--------------|
| 1. Configuration Refactor | P0 | 10-12h | `NumericalTolerances`, `PhysicalThresholds`, solver updates |
| 2. Power-Law Scheme | P0 | 7-9h | `PowerLawScheme.swift`, tests, integration |
| 3. Sauter Bootstrap | P0 | 9-11h | `CollisionalityHelpers`, Sauter formula, tests |
| 4. Metric Tensors | P1 | 6-8h | Enhanced `GeometricFactors`, spatial operators |
| 5. Integration Tests | P2 | 7-9h | Analytical, conservation, TORAX benchmark |
| **Total** | | **45-55h** | **~1 week full-time** |

### Suggested Schedule

**Week 1**:
- Day 1-2: Phase 1 (Configuration) - Foundation
- Day 3: Phase 2 (Power-Law) - Stability
- Day 4-5: Phase 3 (Sauter Bootstrap) - Physics accuracy

**Optional Week 2**:
- Day 1-2: Phase 4 (Metric Tensors) - Advanced geometry
- Day 3-5: Phase 5 (Integration Tests) - Validation

**Minimum Viable Product**: Phases 1-3 only (26-32 hours, ~4 days)

---

## File Impact Summary

### Files to Create (7 new files)
1. `Sources/Gotenx/Configuration/NumericalTolerances.swift`
2. `Sources/Gotenx/Configuration/PhysicalThresholds.swift`
3. `Sources/Gotenx/FVM/PowerLawScheme.swift`
4. `Sources/Gotenx/Solver/CollisionalityHelpers.swift`
5. `Tests/GotenxTests/FVM/PowerLawSchemeTests.swift`
6. `Tests/GotenxTests/Solver/BootstrapCurrentTests.swift`
7. `Tests/GotenxTests/Integration/FVMIntegrationTests.swift`

### Files to Modify (12 existing files)
1. `Sources/Gotenx/Configuration/TimeConfiguration.swift` (+20 lines)
2. `Sources/Gotenx/Configuration/SolverConfig.swift` (+15 lines)
3. `Sources/Gotenx/Configuration/RuntimeParams.swift` (+10 lines)
4. `Sources/GotenxCLI/Configuration/GotenxConfigReader.swift` (+50 lines)
5. `Sources/Gotenx/Solver/NewtonRaphsonSolver.swift` (+100 lines)
6. `Sources/Gotenx/Solver/Block1DCoeffsBuilder.swift` (+150 lines)
7. `Sources/Gotenx/Core/GeometricFactors.swift` (+30 lines)
8. `Sources/Gotenx/Orchestration/TimeStepCalculator.swift` (+15 lines)
9. `Sources/GotenxPhysics/Heating/OhmicHeating.swift` (+5 lines)
10. `Sources/GotenxPhysics/Heating/FusionPower.swift` (+5 lines)
11. `Sources/Gotenx/Configuration/ConfigurationValidator.swift` (+5 lines)
12. `Sources/Gotenx/Diagnostics/DerivedQuantitiesComputer.swift` (+20 lines)

**Total New Code**: ~1600 lines
**Total Modified Code**: ~425 lines
**Total Impact**: ~2025 lines across 19 files

---

## References

### TORAX Core
1. **TORAX Paper**: arXiv:2406.06718v2
2. **TORAX GitHub**: https://github.com/google-deepmind/torax
3. **DeepWiki**: https://deepwiki.com/google-deepmind/torax

### Numerical Methods
4. **Patankar (1980)**: "Numerical Heat Transfer and Fluid Flow"
5. **Hairer & Wanner (1996)**: "Solving ODEs II"
6. **Higham (2002)**: "Accuracy and Stability of Numerical Algorithms"

### Neoclassical Physics
7. **Sauter et al. (1999)**: PoP 6, 2834
8. **Wesson (2011)**: "Tokamak Physics" (2nd ed.), Chapter 7
9. **Hirshman & Sigmar (1981)**: NF 21, 1079

### Tokamak Geometry
10. **Miller et al. (1998)**: PoP 5, 973
11. **Lao et al. (1985)**: NF 25, 1611

---

**Document Status**: Ready for Implementation
**Next Action**: Begin Phase 1 (Configuration Refactor)

---

*Last updated: 2025-10-21*
