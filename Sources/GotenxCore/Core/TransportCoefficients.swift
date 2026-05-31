import Foundation
import MLX

// MARK: - Transport Coefficients

/// Transport coefficients for heat and particle transport
public struct TransportCoefficients: Sendable, Equatable {
    /// Ion heat diffusivity [m^2/s]
    public let chiIon: EvaluatedArray

    /// Electron heat diffusivity [m^2/s]
    public let chiElectron: EvaluatedArray

    /// Particle diffusivity [m^2/s]
    public let particleDiffusivity: EvaluatedArray

    /// Convection velocity [m/s]
    public let convectionVelocity: EvaluatedArray

    public init(
        chiIon: EvaluatedArray,
        chiElectron: EvaluatedArray,
        particleDiffusivity: EvaluatedArray,
        convectionVelocity: EvaluatedArray
    ) {
        self.chiIon = chiIon
        self.chiElectron = chiElectron
        self.particleDiffusivity = particleDiffusivity
        self.convectionVelocity = convectionVelocity
    }

    public init(
        evaluatingChiIon chiIon: MLXArray,
        chiElectron: MLXArray,
        particleDiffusivity: MLXArray,
        convectionVelocity: MLXArray
    ) {
        let evaluated = EvaluatedArray.evaluatingBatch([
            chiIon,
            chiElectron,
            particleDiffusivity,
            convectionVelocity
        ])
        self.init(
            chiIon: evaluated[0],
            chiElectron: evaluated[1],
            particleDiffusivity: evaluated[2],
            convectionVelocity: evaluated[3]
        )
    }
}
