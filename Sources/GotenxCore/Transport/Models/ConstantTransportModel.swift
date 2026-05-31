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
    public let chiIonValue: Float

    /// Electron heat diffusivity [m^2/s]
    public let chiElectronValue: Float

    /// Particle diffusivity [m^2/s]
    public let particleDiffusivityValue: Float

    /// Convection velocity [m/s]
    public let convectionVelocityValue: Float

    // MARK: - Initialization

    /// Initialize constant transport model
    ///
    /// - Parameters:
    ///   - chiIon: Ion heat diffusivity [m^2/s]
    ///   - chiElectron: Electron heat diffusivity [m^2/s]
    ///   - particleDiffusivity: Particle diffusivity [m^2/s]
    ///   - convectionVelocity: Convection velocity [m/s]
    public init(
        chiIon: Float,
        chiElectron: Float,
        particleDiffusivity: Float = 0.0,
        convectionVelocity: Float = 0.0
    ) {
        self.chiIonValue = chiIon
        self.chiElectronValue = chiElectron
        self.particleDiffusivityValue = particleDiffusivity
        self.convectionVelocityValue = convectionVelocity
    }

    /// Initialize from parameters dictionary
    ///
    /// - Parameter params: Transport parameters
    public init(params: TransportParameters) {
        self.chiIonValue = params.params["chi_ion"] ?? 1.0
        self.chiElectronValue = params.params["chi_electron"] ?? 1.0
        self.particleDiffusivityValue = params.params["particle_diffusivity"] ?? 0.0
        self.convectionVelocityValue = params.params["convection_velocity"] ?? 0.0
    }

    // MARK: - TransportModel Protocol

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        params: TransportParameters
    ) -> TransportCoefficients {
        let nCells = profiles.ionTemperature.shape[0]

        // Create constant arrays
        return TransportCoefficients(
            evaluatingChiIon: MLXArray.full([nCells], values: MLXArray(chiIonValue)),
            chiElectron: MLXArray.full([nCells], values: MLXArray(chiElectronValue)),
            particleDiffusivity: MLXArray.full([nCells], values: MLXArray(particleDiffusivityValue)),
            convectionVelocity: MLXArray.full([nCells], values: MLXArray(convectionVelocityValue))
        )
    }
}
