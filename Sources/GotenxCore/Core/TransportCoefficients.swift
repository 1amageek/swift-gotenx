import Foundation
import MLX

// MARK: - Transport Coefficients

/// Transport coefficients for heat and particle transport
public struct TransportCoefficients: Sendable, Equatable {
    /// Ion heat diffusivity [m^2/s]
    public let ionHeatDiffusivity: EvaluatedArray

    /// Electron heat diffusivity [m^2/s]
    public let electronHeatDiffusivity: EvaluatedArray

    /// Particle diffusivity [m^2/s]
    public let particleDiffusivity: EvaluatedArray

    /// Convection velocity [m/s]
    public let convectionVelocity: EvaluatedArray

    public init(
        ionHeatDiffusivity: EvaluatedArray,
        electronHeatDiffusivity: EvaluatedArray,
        particleDiffusivity: EvaluatedArray,
        convectionVelocity: EvaluatedArray
    ) {
        self.ionHeatDiffusivity = ionHeatDiffusivity
        self.electronHeatDiffusivity = electronHeatDiffusivity
        self.particleDiffusivity = particleDiffusivity
        self.convectionVelocity = convectionVelocity
    }

    public init(
        ionHeatDiffusivity: MLXArray,
        electronHeatDiffusivity: MLXArray,
        particleDiffusivity: MLXArray,
        convectionVelocity: MLXArray
    ) {
        let evaluated = EvaluatedArray.evaluatingBatch([
            ionHeatDiffusivity,
            electronHeatDiffusivity,
            particleDiffusivity,
            convectionVelocity
        ])
        self.init(
            ionHeatDiffusivity: evaluated[0],
            electronHeatDiffusivity: evaluated[1],
            particleDiffusivity: evaluated[2],
            convectionVelocity: evaluated[3]
        )
    }
}

extension TransportCoefficients {
    public func validateNumerics(expectedCellCount: Int) throws {
        try NumericalValidation.validateShape(
            ionHeatDiffusivity.value,
            field: "ionHeatDiffusivity",
            expected: [expectedCellCount]
        )
        try NumericalValidation.validateShape(
            electronHeatDiffusivity.value,
            field: "electronHeatDiffusivity",
            expected: [expectedCellCount]
        )
        try NumericalValidation.validateShape(
            particleDiffusivity.value,
            field: "particleDiffusivity",
            expected: [expectedCellCount]
        )
        try NumericalValidation.validateShape(
            convectionVelocity.value,
            field: "convectionVelocity",
            expected: [expectedCellCount]
        )

        try NumericalValidation.validate([
            .nonNegative(ionHeatDiffusivity.value, field: "ionHeatDiffusivity"),
            .nonNegative(electronHeatDiffusivity.value, field: "electronHeatDiffusivity"),
            .nonNegative(particleDiffusivity.value, field: "particleDiffusivity"),
            .finite(convectionVelocity.value, field: "convectionVelocity")
        ])
    }
}
