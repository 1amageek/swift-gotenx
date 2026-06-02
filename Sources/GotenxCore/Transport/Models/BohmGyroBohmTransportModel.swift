import MLX
import Foundation

// MARK: - Bohm-GyroBohm Transport Model

/// Bohm-GyroBohm transport model
///
/// Empirical transport model using Bohm and GyroBohm scaling:
/// χ = C_Bohm * χ_Bohm + C_GB * χ_GB
public struct BohmGyroBohmTransportModel: TransportModel {
    // MARK: - Properties

    public let name = "bohm-gyrobohm"

    /// Bohm coefficient
    public let bohmCoefficient: Float

    /// GyroBohm coefficient
    public let gyroBohmCoefficient: Float

    /// Ion mass number (1=H, 2=D, 3=T)
    ///
    /// **Default**: 2.0 (deuterium)
    public let ionMassNumber: Float

    // MARK: - Initialization

    public init(bohmCoefficient: Float = 1.0, gyroBohmCoefficient: Float = 1.0, ionMassNumber: Float = 2.0) {
        self.bohmCoefficient = bohmCoefficient
        self.gyroBohmCoefficient = gyroBohmCoefficient
        self.ionMassNumber = ionMassNumber
    }

    public init(parameters: TransportParameters) throws {
        try parameters.validateParameterKeys(for: .bohmGyrobohm)
        self.bohmCoefficient = parameters.parameters["bohmCoefficient"] ?? 1.0
        self.gyroBohmCoefficient = parameters.parameters["gyroBohmCoefficient"] ?? 1.0
        self.ionMassNumber = parameters.parameters["ionMassNumber"] ?? 2.0
    }

    // MARK: - TransportModel Protocol

    public func computeCoefficients(
        profiles: CoreProfiles,
        geometry: Geometry,
        parameters: TransportParameters
    ) -> TransportCoefficients {
        let cellCount = profiles.ionTemperature.shape[0]

        // Extract temperature profiles
        let te = profiles.electronTemperature.value

        // Bohm diffusivity: χ_Bohm = (1/16) * (c * k_B * T_e) / (e * B)
        // Reformulate to avoid underflow:
        // χ_Bohm = (1/16) * (c * k_B * T) / (e * B)
        //        = (1/16) * (c * T_e[eV] * e) / (e * B)    [k_B * T = T_e[eV] * e]
        //        = (1/16) * (c * T_e[eV]) / B              [simplify]
        let electronCharge: Float = 1.602e-19  // C
        let speedOfLight: Float = 3.0e8  // m/s
        let B = geometry.toroidalField

        let chiBohmElectron = (1.0 / 16.0) * (speedOfLight * te) / B

        // Clip to prevent Float32 overflow
        let chiBohmElectron_safe = clip(chiBohmElectron, min: MLXArray(Float(1e-6)), max: MLXArray(Float(100.0)))

        // GyroBohm diffusivity: χ_GB = (ρ_s / a)^2 * χ_Bohm
        // where ρ_s = sqrt(m_i * k_B * T_e) / (e * B) is ion sound radius
        //
        // CRITICAL: Avoid Float32 underflow by reformulating:
        // ρ_s = sqrt(m_i × k_B × T) / (e × B)
        //     = sqrt(m_i × T_e[eV] × e) / (e × B)    [k_B × T = T_e[eV] × e]
        //     = sqrt(m_i × T_e[eV] / e) / B          [simplify]
        //
        // This avoids underflow: m_i × T_e[eV] / e ≈ 1e-27 × 2500 / 1e-19 ≈ 2.6e-5 ✓
        let ionMass = PlasmaPhysics.ionMass(massNumber: ionMassNumber)
        let ionMass_array = MLXArray(Float(ionMass))

        let rhoS = sqrt(ionMass_array * te / electronCharge) / B

        // Clip rhoS to prevent overflow in pow()
        // CRITICAL: min must be low enough to preserve isotope effect
        // Physical ρ_s ~ 0.5-2 mm for typical tokamak conditions
        let rhoS_safe = clip(rhoS, min: MLXArray(Float(1e-5)), max: MLXArray(Float(0.1)))

        let minorRadius = geometry.minorRadius
        let chiGyroBohm = pow(rhoS_safe / MLXArray(Float(minorRadius)), MLXArray(Float(2.0))) * chiBohmElectron_safe

        // Clip GyroBohm to prevent overflow
        let chiGyroBohm_safe = clip(chiGyroBohm, min: MLXArray(Float(1e-6)), max: MLXArray(Float(100.0)))

        // Combined diffusivity
        let electronHeatDiffusivity = MLXArray(Float(bohmCoefficient)) * chiBohmElectron_safe + MLXArray(Float(gyroBohmCoefficient)) * chiGyroBohm_safe

        // Final safety clip
        let clampedElectronHeatDiffusivity = clip(
            electronHeatDiffusivity,
            min: MLXArray(Float(1e-6)),
            max: MLXArray(Float(100.0))
        )

        let ionHeatDiffusivity = clampedElectronHeatDiffusivity  // Assume same for ions

        return TransportCoefficients(
            ionHeatDiffusivity: ionHeatDiffusivity,
            electronHeatDiffusivity: clampedElectronHeatDiffusivity,
            particleDiffusivity: clampedElectronHeatDiffusivity * MLXArray(Float(0.5)),  // D = 0.5 * χ
            convectionVelocity: MLXArray.zeros([cellCount])
        )
    }
}
