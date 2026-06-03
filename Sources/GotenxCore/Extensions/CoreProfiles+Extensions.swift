import MLX
import Foundation

// MARK: - CoreProfiles Extension

extension CoreProfiles {
    /// Convert to tuple of CellVariables for solver interface
    ///
    /// - Parameters:
    ///   - radialSpacing: Cell spacing
    ///   - boundaryConditions: Boundary conditions to apply
    /// - Returns: Tuple of (Ti, Te, ne, psi) as CellVariables
    public func asTuple(
        radialSpacing: Float,
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
                radialSpacing: radialSpacing,
                leftFaceConstraint: tiLeft.value,
                leftFaceGradientConstraint: tiLeft.gradient,
                rightFaceConstraint: tiRight.value,
                rightFaceGradientConstraint: tiRight.gradient
            ),
            CellVariable(
                value: electronTemperature.value,
                radialSpacing: radialSpacing,
                leftFaceConstraint: teLeft.value,
                leftFaceGradientConstraint: teLeft.gradient,
                rightFaceConstraint: teRight.value,
                rightFaceGradientConstraint: teRight.gradient
            ),
            CellVariable(
                value: electronDensity.value,
                radialSpacing: radialSpacing,
                leftFaceConstraint: neLeft.value,
                leftFaceGradientConstraint: neLeft.gradient,
                rightFaceConstraint: neRight.value,
                rightFaceGradientConstraint: neRight.gradient
            ),
            CellVariable(
                value: poloidalFlux.value,
                radialSpacing: radialSpacing,
                leftFaceConstraint: psiLeft.value,
                leftFaceGradientConstraint: psiLeft.gradient,
                rightFaceConstraint: psiRight.value,
                rightFaceGradientConstraint: psiRight.gradient
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
        withElectronDensityClamped(minimum: minimum, evaluationMode: .eager)
    }

    package func withElectronDensityClamped(
        minimum: Float = 1e18,
        evaluationMode: MLXEvaluationMode
    ) -> CoreProfiles {
        let clampedDensity = maximum(electronDensity.value, MLXArray(minimum))

        return CoreProfiles(
            ionTemperature: ionTemperature,
            electronTemperature: electronTemperature,
            electronDensity: evaluationMode.wrap(clampedDensity),
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
        withPhysicalFloors(
            temperatureMin: temperatureMin,
            densityMin: densityMin,
            evaluationMode: .eager
        )
    }

    package func withPhysicalFloors(
        temperatureMin: Float = 1.0,
        densityMin: Float = 1e18,
        evaluationMode: MLXEvaluationMode
    ) -> CoreProfiles {
        let clampedTi = maximum(ionTemperature.value, MLXArray(temperatureMin))
        let clampedTe = maximum(electronTemperature.value, MLXArray(temperatureMin))
        let clampedNe = maximum(electronDensity.value, MLXArray(densityMin))

        return CoreProfiles(
            ionTemperature: evaluationMode.wrap(clampedTi),
            electronTemperature: evaluationMode.wrap(clampedTe),
            electronDensity: evaluationMode.wrap(clampedNe),
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
