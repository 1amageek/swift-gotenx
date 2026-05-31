import MLX
import Foundation

// MARK: - CoreProfiles Extension

extension CoreProfiles {
    /// Convert to tuple of CellVariables for solver interface
    ///
    /// - Parameters:
    ///   - dr: Cell spacing
    ///   - boundaryConditions: Boundary conditions to apply
    /// - Returns: Tuple of (Ti, Te, ne, psi) as CellVariables
    public func asTuple(
        dr: Float,
        boundaryConditions: BoundaryConditions
    ) -> (CellVariable, CellVariable, CellVariable, CellVariable) {
        // Extract boundary conditions
        let (tiLeft, tiRight) = extractBoundaryValues(boundaryConditions.ionTemperature)
        let (teLeft, teRight) = extractBoundaryValues(boundaryConditions.electronTemperature)
        let (neLeft, neRight) = extractBoundaryValues(boundaryConditions.electronDensity)
        let (psiLeft, psiRight) = extractBoundaryValues(boundaryConditions.poloidalFlux)

        return (
            CellVariable(
                value: ionTemperature.value,
                dr: dr,
                leftFaceConstraint: tiLeft.value,
                leftFaceGradConstraint: tiLeft.gradient,
                rightFaceConstraint: tiRight.value,
                rightFaceGradConstraint: tiRight.gradient
            ),
            CellVariable(
                value: electronTemperature.value,
                dr: dr,
                leftFaceConstraint: teLeft.value,
                leftFaceGradConstraint: teLeft.gradient,
                rightFaceConstraint: teRight.value,
                rightFaceGradConstraint: teRight.gradient
            ),
            CellVariable(
                value: electronDensity.value,
                dr: dr,
                leftFaceConstraint: neLeft.value,
                leftFaceGradConstraint: neLeft.gradient,
                rightFaceConstraint: neRight.value,
                rightFaceGradConstraint: neRight.gradient
            ),
            CellVariable(
                value: poloidalFlux.value,
                dr: dr,
                leftFaceConstraint: psiLeft.value,
                leftFaceGradConstraint: psiLeft.gradient,
                rightFaceConstraint: psiRight.value,
                rightFaceGradConstraint: psiRight.gradient
            )
        )
    }

    /// Create from tuple of CellVariables
    public static func fromTuple(_ tuple: (CellVariable, CellVariable, CellVariable, CellVariable)) -> CoreProfiles {
        CoreProfiles(
            ionTemperature: tuple.0.value,
            electronTemperature: tuple.1.value,
            electronDensity: tuple.2.value,
            poloidalFlux: tuple.3.value
        )
    }

    /// Clamp electron density to a minimum value (useful for solver projections)
    ///
    /// - Parameter minimum: Minimum allowed density [m^-3]
    /// - Returns: New CoreProfiles instance with clamped density
    public func withElectronDensityClamped(minimum: Float = 1e18) -> CoreProfiles {
        let clampedDensity = maximum(electronDensity.value, MLXArray(minimum))

        return CoreProfiles(
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: EvaluatedArray(evaluating: clampedDensity),
            poloidalFlux: poloidalFlux
        )
    }

    /// Floor temperatures and density to physical minima for use in source/transport
    /// coefficient evaluation.
    ///
    /// Many source terms (ohmic ∝ T_e^-1.5, Bremsstrahlung ∝ √T_e, ion–electron
    /// exchange ∝ T_e^-1.5) and their derivatives blow up — or become NaN — as a
    /// temperature transiently overshoots toward or below zero during a Newton
    /// iteration. Flooring the temperatures that feed the coefficient evaluation
    /// keeps those derivatives bounded (so the Jacobian stays well-conditioned)
    /// without affecting the time-derivative term of the residual, which continues to
    /// use the un-floored state and so still drives the solution to the true value.
    /// The floors sit far below any physical plasma temperature/density, so they are
    /// inactive at convergence.
    ///
    /// - Parameters:
    ///   - temperatureMin: Minimum allowed temperature [eV]
    ///   - densityMin: Minimum allowed density [m^-3]
    public func withPhysicalFloors(
        temperatureMin: Float = 1.0,
        densityMin: Float = 1e18
    ) -> CoreProfiles {
        let clampedTi = maximum(ionTemperature.value, MLXArray(temperatureMin))
        let clampedTe = maximum(electronTemperature.value, MLXArray(temperatureMin))
        let clampedNe = maximum(electronDensity.value, MLXArray(densityMin))

        return CoreProfiles(
            ionTemperature: EvaluatedArray(evaluating: clampedTi),
            electronTemperature: EvaluatedArray(evaluating: clampedTe),
            electronDensity: EvaluatedArray(evaluating: clampedNe),
            poloidalFlux: poloidalFlux
        )
    }
}

// MARK: - Helper Functions

/// Boundary value pair
private struct BoundaryValuePair {
    let value: Float?
    let gradient: Float?
}

/// Extract boundary values from BoundaryCondition
private func extractBoundaryValues(_ bc: BoundaryCondition) -> (left: BoundaryValuePair, right: BoundaryValuePair) {
    let left: BoundaryValuePair
    let right: BoundaryValuePair

    // Extract left boundary
    switch bc.left {
    case .value(let v):
        left = BoundaryValuePair(value: v, gradient: nil)
    case .gradient(let g):
        left = BoundaryValuePair(value: nil, gradient: g)
    }

    // Extract right boundary
    switch bc.right {
    case .value(let v):
        right = BoundaryValuePair(value: v, gradient: nil)
    case .gradient(let g):
        right = BoundaryValuePair(value: nil, gradient: g)
    }

    return (left, right)
}
