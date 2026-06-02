import MLX
import Foundation

// MARK: - Constant Transport Model

/// Constant transport model with fixed diffusivity values
///
/// This is the simplest transport model, using spatially uniform, constant
/// heat and particle diffusivities. Useful for testing and benchmarking.
public struct ConstantTransportModel: TransportModel {
    // MARK: - Properties

    public let name = "constant"

    /// Ion heat diffusivity [m^2/s]
    public let ionHeatDiffusivityValue: Float

    /// Electron heat diffusivity [m^2/s]
    public let electronHeatDiffusivityValue: Float

    /// Particle diffusivity [m^2/s]
    public let particleDiffusivityValue: Float

    /// Convection velocity [m/s]
    public let convectionVelocityValue: Float

    // MARK: - Initialization

    /// Initialize constant transport model
    ///
    /// - Parameters:
    ///   - ionHeatDiffusivity: Ion heat diffusivity [m^2/s]
    ///   - electronHeatDiffusivity: Electron heat diffusivity [m^2/s]
    ///   - particleDiffusivity: Particle diffusivity [m^2/s]
    ///   - convectionVelocity: Convection velocity [m/s]
    public init(
        ionHeatDiffusivity: Float,
        electronHeatDiffusivity: Float,
        particleDiffusivity: Float = 0.0,
        convectionVelocity: Float = 0.0
    ) {
        self.ionHeatDiffusivityValue = ionHeatDiffusivity
        self.electronHeatDiffusivityValue = electronHeatDiffusivity
        self.particleDiffusivityValue = particleDiffusivity
        self.convectionVelocityValue = convectionVelocity
    }

    /// Initialize from parameters dictionary
    ///
    /// - Parameter parameters: Transport parameters
    public init(parameters: TransportParameters) throws {
        self.ionHeatDiffusivityValue = try parameters.requireParameter(
            "ionHeatDiffusivity",
            modelType: .constant,
            suggestion: "Specify ionHeatDiffusivity in constant transport parameters"
        )
        self.electronHeatDiffusivityValue = try parameters.requireParameter(
            "electronHeatDiffusivity",
            modelType: .constant,
            suggestion: "Specify electronHeatDiffusivity in constant transport parameters"
        )
        self.particleDiffusivityValue = parameters.parameters["particleDiffusivity"] ?? 0.0
        self.convectionVelocityValue = parameters.parameters["convectionVelocity"] ?? 0.0
    }

    // MARK: - TransportModel Protocol

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: TransportParameters
    ) -> TransportCoefficients {
        let cellCount = profiles.ionTemperature.shape[0]

        // Create constant arrays
        return TransportCoefficients(
            ionHeatDiffusivity: MLXArray.full([cellCount], values: MLXArray(ionHeatDiffusivityValue)),
            electronHeatDiffusivity: MLXArray.full([cellCount], values: MLXArray(electronHeatDiffusivityValue)),
            particleDiffusivity: MLXArray.full([cellCount], values: MLXArray(particleDiffusivityValue)),
            convectionVelocity: MLXArray.full([cellCount], values: MLXArray(convectionVelocityValue))
        )
    }
}
