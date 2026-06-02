// NumericalTolerances.swift
// Per-equation numerical tolerance configuration

import Foundation
import MLX

/// Per-equation numerical tolerances
public struct EquationTolerances: Codable, Sendable, Equatable, Hashable {
    /// Absolute tolerance for residual norm [physical units]
    public let absoluteTolerance: Float

    /// Relative tolerance for residual norm [dimensionless]
    public let relativeTolerance: Float

    /// Minimum value threshold (below this, use absolute tolerance only)
    public let minimumValueThreshold: Float

    public init(
        absoluteTolerance: Float,
        relativeTolerance: Float,
        minimumValueThreshold: Float
    ) {
        self.absoluteTolerance = absoluteTolerance
        self.relativeTolerance = relativeTolerance
        self.minimumValueThreshold = minimumValueThreshold
    }

    /// Compute combined tolerance for state value x
    /// tol = max(absoluteTolerance, relativeTolerance * |x|)
    public func combinedTolerance(for value: Float) -> Float {
        if abs(value) < minimumValueThreshold {
            return absoluteTolerance
        }
        return max(absoluteTolerance, relativeTolerance * abs(value))
    }
}

/// Numerical tolerance configuration for all equations
public struct NumericalTolerances: Codable, Sendable, Equatable, Hashable {
    public let ionTemperature: EquationTolerances
    public let electronTemperature: EquationTolerances
    public let electronDensity: EquationTolerances
    public let poloidalFlux: EquationTolerances

    public init(
        ionTemperature: EquationTolerances,
        electronTemperature: EquationTolerances,
        electronDensity: EquationTolerances,
        poloidalFlux: EquationTolerances
    ) {
        self.ionTemperature = ionTemperature
        self.electronTemperature = electronTemperature
        self.electronDensity = electronDensity
        self.poloidalFlux = poloidalFlux
    }

    /// Default ITER-scale tolerances
    public static let iterScale = NumericalTolerances(
        ionTemperature: EquationTolerances(
            absoluteTolerance: 10.0,        // 10 eV absolute
            relativeTolerance: 1e-4,        // 0.01% relative
            minimumValueThreshold: 100.0        // Below 100 eV, use absolute only
        ),
        electronTemperature: EquationTolerances(
            absoluteTolerance: 10.0,
            relativeTolerance: 1e-4,
            minimumValueThreshold: 100.0
        ),
        electronDensity: EquationTolerances(
            absoluteTolerance: 1e17,        // 1e17 m⁻³ absolute
            relativeTolerance: 1e-4,
            minimumValueThreshold: 1e18
        ),
        poloidalFlux: EquationTolerances(
            absoluteTolerance: 1e-3,        // 1 mWb absolute
            relativeTolerance: 1e-5,
            minimumValueThreshold: 0.1
        )
    )

    /// Legacy tolerance conversion from single value
    /// Used for backward compatibility with old configs
    public static func fromLegacy(tolerance: Float) -> NumericalTolerances {
        return NumericalTolerances(
            ionTemperature: EquationTolerances(
                absoluteTolerance: tolerance * 1e4,  // Scale to eV
                relativeTolerance: tolerance,
                minimumValueThreshold: 100.0
            ),
            electronTemperature: EquationTolerances(
                absoluteTolerance: tolerance * 1e4,
                relativeTolerance: tolerance,
                minimumValueThreshold: 100.0
            ),
            electronDensity: EquationTolerances(
                absoluteTolerance: tolerance * 1e20,  // Scale to m⁻³
                relativeTolerance: tolerance,
                minimumValueThreshold: 1e18
            ),
            poloidalFlux: EquationTolerances(
                absoluteTolerance: tolerance * 10,    // Scale to Wb
                relativeTolerance: tolerance,
                minimumValueThreshold: 0.1
            )
        )
    }
}

/// Tolerance scaler for residual-space convergence checks
public struct ToleranceScaler {
    let referenceState: FlattenedState
    let tolerances: NumericalTolerances

    public init(referenceState: FlattenedState, tolerances: NumericalTolerances) {
        self.referenceState = referenceState
        self.tolerances = tolerances
    }

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

// StateLayout is now defined in FlattenedState.swift to avoid duplication
// Import it via typealias for backward compatibility
public typealias StateLayout = FlattenedState.StateLayout
